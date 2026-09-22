"""Autoregressive (RT-2 / OpenVLA style) vs flow matching (pi0 / pi0.5 style) on the same toy task.

Autoregressive policy:
  * each action value is rounded into one of BINS bins -> a discrete token (a "word")
  * the 16x2 chunk becomes a sequence of 32 tokens: x0, y0, x1, y1, ...
  * a small causal transformer predicts the next token given the observation + previous tokens
  * trained with cross-entropy (next-token prediction, exactly like an LLM)
  * generated one token at a time: 32 forward passes per chunk

Flow-matching policy: the ActionExpert from toy_flow_matching.py (10 forward passes per chunk).
"""
import math
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_flow_matching import H, A, STEPS, ActionExpert, flow_matching_loss, sample, train

torch.manual_seed(0)
BINS = 256
L = H * A                                  # 32 tokens per chunk
LO = torch.tensor([-0.1, -1.2])            # value range per action dim (x, y)
HI = torch.tensor([3.1, 1.2])


def tokenize(actions):                     # (n, H, 2) floats -> (n, 32) ints
    b = ((actions - LO) / (HI - LO) * (BINS - 1)).round().clamp(0, BINS - 1).long()
    return b.reshape(-1, L)


def detokenize(tokens):                    # (n, 32) ints -> (n, H, 2) floats
    b = tokens.reshape(-1, H, A).float()
    return LO + b / (BINS - 1) * (HI - LO)


class ARPolicy(nn.Module):
    """Tiny decoder-only transformer. Position 0 holds the observation (the 'prompt');
    the output at position k predicts token k."""

    def __init__(self, d=128, depth=4, heads=4):
        super().__init__()
        self.obs_embed = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.tok_embed = nn.Embedding(BINS, d)
        self.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(d, BINS)

    def forward(self, obs, prev_tokens):   # prev_tokens: (n, k) with k in [0, L-1]
        h = torch.cat([self.obs_embed(obs)[:, None], self.tok_embed(prev_tokens)], 1)
        h = h + self.pos[:, : h.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(h.shape[1])
        return self.head(self.tf(h, mask=mask, is_causal=True))  # (n, k+1, BINS)


def ar_loss(model, obs, actions):
    tokens = tokenize(actions)
    logits = model(obs, tokens[:, :-1])    # teacher forcing: feed the true previous tokens
    return F.cross_entropy(logits.reshape(-1, BINS), tokens.reshape(-1))


@torch.no_grad()
def ar_sample(model, obs, return_partial=False):
    tokens = torch.zeros(obs.shape[0], 0, dtype=torch.long)
    partial = []
    for _ in range(L):                     # one forward pass per token (no KV cache here)
        logits = model(obs, tokens)[:, -1]
        nxt = torch.multinomial(logits.softmax(-1), 1)[:, 0]
        tokens = torch.cat([tokens, nxt[:, None]], 1)
        partial.append(tokens.clone())
    return (detokenize(tokens), partial) if return_partial else detokenize(tokens)


# ---------- metrics ----------
def ideal_path(goal, side):
    s = torch.linspace(0, 1, H)
    return torch.stack([s * goal, side * 0.8 * torch.sin(math.pi * s)], -1)


def metrics(actions, goal):
    up, down = ideal_path(goal, 1), ideal_path(goal, -1)
    err = torch.minimum((actions - up).norm(dim=-1).mean(-1), (actions - down).norm(dim=-1).mean(-1))
    hit = ((actions - torch.tensor([goal / 2, 0.0])).norm(dim=-1) < 0.35).any(-1)
    y = actions[..., 1]
    jerk = (y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2]).abs().mean(-1)
    return {
        "above %": 100 * (actions[:, H // 2, 1] > 0).float().mean().item(),
        "collisions %": 100 * hit.float().mean().item(),
        "error vs ideal path": err.mean().item(),
        "roughness (2nd diff)": jerk.mean().item(),
    }


def timed(fn, reps=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return 1000 * (time.perf_counter() - t0) / reps


if __name__ == "__main__":
    print("Training flow-matching policy...")
    fm = train(ActionExpert(), flow_matching_loss)
    print("Training autoregressive policy...")
    ar = train(ARPolicy(), ar_loss)

    goal, N = 2.5, 200
    obs = torch.full((N, 1), goal)
    torch.manual_seed(1)
    results = {
        "Flow matching (10 steps)": (sample(fm, obs), STEPS, timed(lambda: sample(fm, obs))),
        "Autoregressive": (ar_sample(ar, obs), L, timed(lambda: ar_sample(ar, obs), reps=2)),
    }
    # reference: the demonstration data itself
    sides = torch.randint(0, 2, (N,)) * 2 - 1
    demo = torch.stack([ideal_path(goal, s.item()) for s in sides])
    demo[..., 1] += 0.03 * torch.randn(N, H)       # same noise as make_batch

    n_fm = sum(p.numel() for p in fm.parameters())
    n_ar = sum(p.numel() for p in ar.parameters())
    print(f"\nParams: flow matching {n_fm/1e6:.2f}M, autoregressive {n_ar/1e6:.2f}M")
    print(f"AR bin width: x {((HI - LO) / (BINS - 1))[0]:.4f}, y {((HI - LO) / (BINS - 1))[1]:.4f}\n")
    hdr = f"{'':28s}{'passes':>8s}{'ms/200':>9s}{'above %':>9s}{'collide %':>11s}{'err':>8s}{'rough':>8s}"
    print(hdr)
    for name, (acts, passes, ms) in results.items():
        m = metrics(acts, goal)
        print(f"{name:28s}{passes:8d}{ms:9.1f}{m['above %']:9.0f}{m['collisions %']:11.1f}"
              f"{m['error vs ideal path']:8.3f}{m['roughness (2nd diff)']:8.3f}")
    m = metrics(demo, goal)
    print(f"{'(demonstration data)':28s}{'':8s}{'':9s}{m['above %']:9.0f}{m['collisions %']:11.1f}"
          f"{m['error vs ideal path']:8.3f}{m['roughness (2nd diff)']:8.3f}")

    # ---------- figure 1: final samples ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, (name, (acts, _, _)) in zip(ax, results.items()):
        a.add_patch(plt.Circle((goal / 2, 0), 0.35, color="gray", alpha=0.4))
        for tr in acts[:40]:
            a.plot(tr[:, 0], tr[:, 1], "-", alpha=0.4)
        a.set_xlim(-0.3, 2.8); a.set_ylim(-1.2, 1.2); a.set_aspect("equal"); a.set_title(name + ": 40 samples")
    plt.tight_layout(); plt.savefig("compare_samples.png", dpi=120)

    # ---------- figure 2: HOW each one builds a chunk ----------
    torch.manual_seed(3)
    o = torch.full((3, 1), goal)
    _, fm_path = sample(fm, o, return_path=True)
    _, ar_partial = ar_sample(ar, o, return_partial=True)
    fig, ax = plt.subplots(2, 4, figsize=(16, 6.5))
    for col, k in enumerate([0, 3, 6, 10]):
        a = ax[0, col]
        for j in range(3):
            a.plot(fm_path[k][j, :, 0], fm_path[k][j, :, 1], "-o", ms=3)
        a.set_title(f"Flow matching: after {k}/{STEPS} passes")
    for col, k in enumerate([8, 16, 24, 32]):
        a = ax[1, col]
        for j in range(3):
            toks = ar_partial[k - 1][j]
            n_full = k // 2                      # complete (x, y) waypoints so far
            pts = detokenize(torch.cat([toks[: 2 * n_full], torch.zeros(L - 2 * n_full, dtype=torch.long)])[None])[0]
            a.plot(pts[:n_full, 0], pts[:n_full, 1], "-o", ms=3)
        a.set_title(f"Autoregressive: after {k}/{L} tokens")
    for a in ax.flat:
        a.add_patch(plt.Circle((goal / 2, 0), 0.35, color="gray", alpha=0.3))
        a.set_xlim(-0.5, 3.0); a.set_ylim(-2.2, 2.2); a.set_aspect("equal")
    fig.suptitle("Flow matching refines the WHOLE chunk at once; autoregressive WRITES it left to right")
    plt.tight_layout(); plt.savefig("compare_generation.png", dpi=120)
    print("\nSaved compare_samples.png and compare_generation.png")
