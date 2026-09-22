"""Toy flow-matching action head, using the same formulation as pi0 / pi0.5 (openpi).

Task: a 2D "robot" at the origin must reach a goal at (goal_x, 0) with an obstacle
in the middle. It can go around either ABOVE or BELOW -- two equally valid action
chunks for the same observation. This is the multimodality that motivates using a
generative action head instead of plain regression.

openpi conventions reproduced here:
  * t ~ Beta(1.5, 1) * 0.999 + 0.001   (biased toward t=1, i.e. the noisy end)
  * x_t = t * noise + (1 - t) * actions    (t=1 -> pure noise, t=0 -> clean actions)
  * target velocity u_t = noise - actions, loss = ||v_theta(x_t, t, obs) - u_t||^2
  * inference: start at x_1 ~ N(0, I), 10 Euler steps with dt = -0.1 down to t=0
  * pi0.5-style timestep conditioning: t goes through adaptive norm (scale/shift)
    in the action-expert layers, not concatenated to the action tokens (that was pi0)
"""
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
H = 16          # action horizon (pi0.5 uses 50; kept small here)
A = 2           # action dim (x, y waypoint)
STEPS = 10      # denoising steps, same default as openpi


def make_batch(n):
    """Observation = goal x-position. Action chunk = H waypoints around the obstacle."""
    goal = torch.rand(n, 1) * 1.5 + 1.5                   # goal x in [1.5, 3]
    side = torch.randint(0, 2, (n, 1)).float() * 2 - 1    # +1 above, -1 below (the two modes)
    s = torch.linspace(0, 1, H)[None, :]                  # progress along the path
    x = s * goal
    y = side * 0.8 * torch.sin(math.pi * s) + 0.03 * torch.randn(n, H)
    return goal, torch.stack([x, y], -1)                  # (n,1), (n,H,2)


def sinusoidal(t, dim=64):
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), dim // 2))
    ang = t[:, None] * freqs[None] * 2 * math.pi
    return torch.cat([ang.sin(), ang.cos()], -1)


class AdaNormBlock(nn.Module):
    """Residual MLP block whose RMSNorm scale/shift/gate come from the timestep (adaRMSNorm)."""

    def __init__(self, d, cond_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Linear(cond_dim, 3 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, h, cond):
        scale, shift, gate = self.ada(cond)[:, None].chunk(3, -1)
        hn = F.rms_norm(h, (h.shape[-1],)) * (1 + scale) + shift
        return h + (1 + gate) * self.mlp(hn)


class ActionExpert(nn.Module):
    """Stand-in for pi0.5's action expert. The real one is a ~300M transformer attending
    to the PaliGemma VLM's KV cache; here the "VLM prefix" is just an MLP embedding of obs."""

    def __init__(self, d=128, depth=4):
        super().__init__()
        self.obs_embed = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.time_embed = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        self.act_in = nn.Linear(A, d)
        self.pos = nn.Parameter(torch.randn(1, H, d) * 0.02)
        self.mix = nn.Linear(H * d, H * d)  # crude token mixing instead of attention
        self.blocks = nn.ModuleList([AdaNormBlock(d, d) for _ in range(depth)])
        self.act_out = nn.Linear(d, A)

    def forward(self, x_t, t, obs):
        n = x_t.shape[0]
        h = self.act_in(x_t) + self.pos + self.obs_embed(obs)[:, None]
        h = h + self.mix(h.reshape(n, -1)).reshape(n, H, -1)
        cond = self.time_embed(sinusoidal(t))
        for blk in self.blocks:
            h = blk(h, cond)
        return self.act_out(h)  # predicted velocity v = d x_t / d t


def flow_matching_loss(model, obs, actions):
    n = actions.shape[0]
    t = torch.distributions.Beta(1.5, 1.0).sample((n,)) * 0.999 + 0.001
    noise = torch.randn_like(actions)
    x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
    u_t = noise - actions
    return F.mse_loss(model(x_t, t, obs), u_t)


@torch.no_grad()
def sample(model, obs, steps=STEPS, return_path=False):
    x = torch.randn(obs.shape[0], H, A)
    t, dt = 1.0, -1.0 / steps
    path = [x.clone()]
    for _ in range(steps):
        v = model(x, torch.full((obs.shape[0],), t), obs)
        x = x + dt * v
        t += dt
        path.append(x.clone())
    return (x, path) if return_path else x


class Regressor(nn.Module):
    """Baseline: deterministic MSE regression obs -> action chunk (what you'd get without a generative head)."""

    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, H * A))

    def forward(self, obs):
        return self.net(obs).view(-1, H, A)


def train(model, loss_fn, iters=4000, bs=256):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    model.train()
    for i in range(iters):
        obs, act = make_batch(bs)
        loss = loss_fn(model, obs, act)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if i % 1000 == 0 or i == iters - 1:
            print(f"  iter {i:5d}  loss {loss.item():.4f}")
    model.eval()
    return model


if __name__ == "__main__":
    print("Training flow-matching action expert...")
    fm = train(ActionExpert(), flow_matching_loss)
    print("Training MSE regression baseline...")
    reg = train(Regressor(), lambda m, o, a: F.mse_loss(m(o), a))

    obs = torch.full((40, 1), 2.5)
    fm_actions, path = sample(fm, obs, return_path=True)
    reg_actions = reg(obs).detach()
    above = (fm_actions[:, H // 2, 1] > 0).float().mean().item()
    print(f"Flow matching: {above:.0%} of samples go above, {1 - above:.0%} below")
    print(f"Regression midpoint y = {reg_actions[0, H // 2, 1]:.3f}  (averages the modes -> hits the obstacle)")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a in (ax[0], ax[1]):
        a.add_patch(plt.Circle((1.25, 0), 0.35, color="gray", alpha=0.4))
        a.set_xlim(-0.3, 2.8); a.set_ylim(-1.2, 1.2); a.set_aspect("equal")
    for tr in fm_actions:
        ax[0].plot(tr[:, 0], tr[:, 1], "-", alpha=0.5, color="tab:blue")
    ax[0].set_title("Flow matching: 40 samples, same obs")
    ax[1].plot(reg_actions[0, :, 0], reg_actions[0, :, 1], "-o", color="tab:red", ms=3)
    ax[1].set_title("MSE regression: one averaged answer")
    # denoising trajectory of a few samples: y of the midpoint waypoint over the steps
    ts = [1 - k / STEPS for k in range(STEPS + 1)]
    for j in range(12):
        ax[2].plot(ts, [p[j, H // 2, 1].item() for p in path], "-o", ms=3, alpha=0.7)
    ax[2].invert_xaxis(); ax[2].set_xlabel("flow time t  (1 = noise -> 0 = action)")
    ax[2].set_ylabel("y of midpoint waypoint"); ax[2].set_title("10 Euler steps: noise flows to a mode")
    plt.tight_layout()
    plt.savefig("toy_flow_matching.png", dpi=120)
    print("Saved toy_flow_matching.png")
