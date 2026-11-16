"""LSTM handwriting network with an attention window and MDN output [Graves, ref 1].

Three LSTM layers read the previous pen offset, the attention window over the
text and a style vector. The head predicts a mixture of bivariate Gaussians for
the next offset and the probability that the pen lifts. The LSTM gates are plain
nn.Linear layers so that LoRA adapters (see lora.py) can be attached to them.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = " !\"#&'(),-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(VOCAB)}  # 0 = unknown / padding
VOCAB_SIZE = len(VOCAB) + 1


def encode_text(text: str) -> list[int]:
    return [CHAR_TO_ID.get(c, 0) for c in text]


class LinearLSTMCell(nn.Module):
    def __init__(self, inp: int, hid: int):
        super().__init__()
        self.hid = hid
        self.ih = nn.Linear(inp, 4 * hid)
        self.hh = nn.Linear(hid, 4 * hid, bias=False)
        with torch.no_grad():
            self.ih.bias.zero_()
            self.ih.bias[hid : 2 * hid].fill_(1.0)  # forget-gate bias

    def forward(self, x, state):
        h, c = state
        i, f, g, o = (self.ih(x) + self.hh(h)).chunk(4, -1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class StyleEncoder(nn.Module):
    """Turns a writer's reference strokes into a fixed-size style vector."""

    def __init__(self, style_dim: int = 64, hid: int = 128):
        super().__init__()
        self.rnn = nn.GRU(3, hid, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hid, style_dim)

    def forward(self, ref: torch.Tensor, ref_mask: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(ref)
        m = ref_mask.unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return torch.tanh(self.out(pooled))


class GravesNet(nn.Module):
    def __init__(self, hidden: int = 400, n_window: int = 10, n_mix: int = 20, style_dim: int = 64):
        super().__init__()
        self.config = dict(hidden=hidden, n_window=n_window, n_mix=n_mix, style_dim=style_dim)
        H, K, V, S = hidden, n_window, VOCAB_SIZE, style_dim
        self.K, self.M, self.H = K, n_mix, H
        self.style_enc = StyleEncoder(S)
        self.cell1 = LinearLSTMCell(3 + V + S, H)
        self.window = nn.Linear(H, 3 * K)
        self.cell2 = LinearLSTMCell(3 + V + H + S, H)
        self.cell3 = LinearLSTMCell(3 + V + H + S, H)
        self.head = nn.Linear(3 * H, 1 + 6 * n_mix)

    def init_state(self, B: int, device) -> dict:
        z = lambda n: torch.zeros(B, n, device=device)  # noqa: E731
        return dict(h1=z(self.H), c1=z(self.H), h2=z(self.H), c2=z(self.H), h3=z(self.H), c3=z(self.H),
                    w=z(VOCAB_SIZE), kappa=z(self.K))

    def step(self, x, chars, char_mask, style, st):
        """One time step. chars is one-hot (B, U, V); returns head output, new state, phi."""
        h1, c1 = self.cell1(torch.cat([x, st["w"], style], -1), (st["h1"], st["c1"]))
        a, b, k = torch.exp(self.window(h1)).chunk(3, -1)
        kappa = st["kappa"] + 0.1 * k
        u = torch.arange(chars.shape[1], device=x.device, dtype=x.dtype)
        phi = (a.unsqueeze(-1) * torch.exp(-b.unsqueeze(-1) * (kappa.unsqueeze(-1) - u) ** 2)).sum(1)
        phi = phi * char_mask
        w = torch.bmm(phi.unsqueeze(1), chars).squeeze(1)
        h2, c2 = self.cell2(torch.cat([x, h1, w, style], -1), (st["h2"], st["c2"]))
        h3, c3 = self.cell3(torch.cat([x, h2, w, style], -1), (st["h3"], st["c3"]))
        y = self.head(torch.cat([h1, h2, h3], -1))
        new = dict(h1=h1, c1=c1, h2=h2, c2=c2, h3=h3, c3=c3, w=w, kappa=kappa, a=a, b=b)
        return y, new, phi

    def forward(self, x, chars, char_mask, style):
        """Teacher-forced pass. x: (B, T, 3) inputs (targets shifted right by one)."""
        st = self.init_state(x.shape[0], x.device)
        ys = []
        for t in range(x.shape[1]):
            y, st, _ = self.step(x[:, t], chars, char_mask, style, st)
            ys.append(y)
        return torch.stack(ys, 1)

    def split(self, y, bias: float = 0.0):
        M = self.M
        e_hat = y[..., 0]
        pi_hat, mu1, mu2, s1, s2, rho = y[..., 1:].split(M, -1)
        log_pi = F.log_softmax(pi_hat * (1.0 + bias), -1)
        return e_hat, log_pi, mu1, mu2, torch.exp(s1 - bias), torch.exp(s2 - bias), torch.tanh(rho) * 0.999

    @torch.no_grad()
    def sample(self, text: str, style: torch.Tensor, bias: float = 1.0, max_steps: int | None = None,
               generator: torch.Generator | None = None) -> np.ndarray:
        """Write one line. Returns standardised stroke-3 (N, 3)."""
        dev = style.device
        ids = torch.tensor([encode_text(text)], device=dev)
        chars = F.one_hot(ids, VOCAB_SIZE).float()
        mask = torch.ones(ids.shape, device=dev)
        U = ids.shape[1]
        st = self.init_state(1, dev)
        x = torch.zeros(1, 3, device=dev)
        out = []
        for _ in range(max_steps or 30 * U + 100):
            y, st, phi = self.step(x, chars, mask, style, st)
            e_hat, log_pi, mu1, mu2, s1, s2, rho = self.split(y, bias)
            j = torch.multinomial(log_pi.exp(), 1, generator=generator).item()
            n1, n2 = torch.randn(2, generator=generator, device=dev)
            d1 = mu1[0, j] + s1[0, j] * n1
            d2 = mu2[0, j] + s2[0, j] * (rho[0, j] * n1 + torch.sqrt(1 - rho[0, j] ** 2) * n2)
            lift = float(torch.rand(1, generator=generator, device=dev) < torch.sigmoid(e_hat[0]))
            out.append((d1.item(), d2.item(), lift))
            x = torch.tensor([[d1, d2, lift]], device=dev)
            # Stop once the window has moved past the last character.
            phi_end = (st["a"] * torch.exp(-st["b"] * (st["kappa"] - U) ** 2)).sum()
            if phi_end > phi.max() and lift:
                break
        return np.asarray(out, dtype=np.float32)


def mdn_loss(net: GravesNet, y, target, mask):
    """Negative log-likelihood of the next point: mixture term plus pen-lift term."""
    e_hat, log_pi, mu1, mu2, s1, s2, rho = net.split(y)
    x1, x2, p = target[..., :1], target[..., 1:2], target[..., 2]
    z1, z2 = (x1 - mu1) / s1, (x2 - mu2) / s2
    one_m = 1 - rho**2
    log_n = -(z1**2 + z2**2 - 2 * rho * z1 * z2) / (2 * one_m) - torch.log(2 * math.pi * s1 * s2 * torch.sqrt(one_m))
    nll_xy = -torch.logsumexp(log_pi + log_n, -1)
    nll_p = F.binary_cross_entropy_with_logits(e_hat, p, reduction="none")
    return ((nll_xy + nll_p) * mask).sum() / mask.sum().clamp(min=1.0)


def save_checkpoint(path, net: GravesNet, norm: dict, extra: dict | None = None):
    torch.save({"config": net.config, "state_dict": net.state_dict(), "norm": norm, "vocab": VOCAB,
                **(extra or {})}, path)


def load_checkpoint(path, device="cpu") -> tuple[GravesNet, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = GravesNet(**ck["config"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck["norm"]
