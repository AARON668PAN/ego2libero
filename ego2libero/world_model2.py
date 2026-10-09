"""World model v2: both 256 px cameras, predicted in the latent space of a pretrained image VAE.

Each camera frame is encoded by a Stable Diffusion VAE to a 4x32x32 latent, so one time step is an
8x32x32 tensor (front and wrist camera stacked). The denoiser predicts the next step's latents from
the noisy target, 12 past steps reaching 2.4 s back (dense near the present, sparse further away),
the last 8 actions and the robot state. Next to it, a small MLP predicts the next robot state (eef
position, quaternion, gripper fingers) and a CNN on the latents says whether the task is done, so a
policy can run closed-loop inside the model. Trained with EDM preconditioning, DIAMOND-style.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Fourier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("w", torch.randn(dim // 2) * 4.0)

    def forward(self, x):                       # x: (B,)
        f = x[:, None] * self.w[None] * 2 * math.pi
        return torch.cat([f.sin(), f.cos()], 1)


class Res(nn.Module):
    def __init__(self, cin, cout, cond):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(32, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(cond, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(c)[:, :, None, None].chunk(2, 1)
        h = self.c2(F.silu(self.n2(h) * (1 + scale) + shift))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n, self.qkv, self.o = nn.GroupNorm(32, ch), nn.Conv2d(ch, 3 * ch, 1), nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, c, h * w).unbind(1)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return x + self.o(a.transpose(1, 2).reshape(b, c, h, w))


OFFSETS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)   # context steps before the current one (0 = now)
N_ACT = 8            # past actions, the last one is the action taken now
ACT_DIM, ZC = 7, 8   # action size; latent channels per step (2 cameras x 4)
P_MEAN, P_STD = -0.4, 1.2


# ---- robot state: the policy sees [eef pos, axis-angle, gripper qpos]; the model keeps the quaternion ----

def aa2quat(aa):
    """Invert LeRobot's quat2axisangle exactly (angle in [0, 2pi]), so the quaternion keeps the sign the
    simulator gave it. aa: (..., 3) tensor -> (..., 4) xyzw."""
    th = aa.norm(dim=-1, keepdim=True)
    axis = torch.where(th > 1e-8, aa / th.clamp_min(1e-8), torch.zeros_like(aa))
    return torch.cat([axis * torch.sin(th / 2), torch.cos(th / 2)], -1)


def quat2aa(q):
    """LeRobot's conversion (processor/env_processor.py)."""
    w = q[..., 3:].clamp(-1, 1)
    den = (1 - w * w).clamp_min(0).sqrt()
    return torch.where(den > 1e-10, q[..., :3] / den.clamp_min(1e-10) * 2 * torch.acos(w), torch.zeros_like(q[..., :3]))


def state_to_robot(s):
    """(..., 8) policy state -> (..., 9) [pos 3, quat 4, gripper 2]."""
    return torch.cat([s[..., :3], aa2quat(s[..., 3:6]), s[..., 6:8]], -1)


def robot_to_state(r):
    return torch.cat([r[..., :3], quat2aa(r[..., 3:7]), r[..., 7:9]], -1)


def quat_mul(a, b):
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a.unbind(-1); bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def rotvec_to_quat(w):
    th = w.norm(dim=-1, keepdim=True)
    s = torch.where(th > 1e-8, torch.sin(th / 2) / th.clamp_min(1e-8), 0.5 * torch.ones_like(th))
    return torch.cat([w * s, torch.cos(th / 2)], -1)


class StateModel(nn.Module):
    """Next robot state [pos 3, quat 4, fingers 2] from the current one and the last 4 actions.

    The core is linear in the actions only (position and rotation are moved by a fixed filter of the
    last 4 commands; the fingers relax towards an open or closed target), so feeding the model its own
    predictions cannot make it blow up. A small MLP adds a bounded per-step correction. An earlier
    version also took the last predicted velocity as input and diverged within 20 steps of open loop.
    """

    def __init__(self, hidden=256):
        super().__init__()
        self.Bp = nn.Parameter(torch.zeros(4, 3, 3))       # last 4 position commands -> displacement
        self.Br = nn.Parameter(torch.zeros(4, 3, 3))       # last 4 rotation commands -> rotation vector
        self.g_rate = nn.Parameter(torch.tensor(0.0))
        self.g_open = nn.Parameter(torch.tensor([0.04, -0.04]))
        self.g_close = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.res = nn.Sequential(nn.Linear(9 + 4 * ACT_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 8))
        nn.init.zeros_(self.res[-1].weight); nn.init.zeros_(self.res[-1].bias)
        self.register_buffer("res_max", torch.tensor([0.004] * 3 + [0.02] * 3 + [0.002] * 2))

    def forward(self, r_now, r_prev, acts):                # r_prev is unused (kept for the call sites)
        pos, q, g = r_now[:, :3], r_now[:, 3:7], r_now[:, 7:9]
        res = torch.tanh(self.res(torch.cat([r_now, acts.flatten(1)], 1))) * self.res_max
        dpos = torch.einsum("bki,kji->bj", acts[..., :3], self.Bp) + res[:, :3]
        w = torch.einsum("bki,kji->bj", acts[..., 3:6], self.Br) + res[:, 3:6]
        q_n = F.normalize(quat_mul(rotvec_to_quat(w), q), dim=1)
        target = torch.where(acts[:, -1, 6:7] > 0, self.g_close, self.g_open)
        g_n = g + torch.sigmoid(self.g_rate) * (target - g) + res[:, 6:8]
        return torch.cat([pos + dpos, q_n, g_n], 1)

    @torch.no_grad()
    def fit_linear(self, R, A, max_samples=500_000):
        """Least squares for the linear core. R (N,H+1,9) states, A (N,H+3,7) actions with 3 before R[:,0]."""
        from scipy.spatial.transform import Rotation
        H = R.shape[1] - 1
        X = torch.stack([A[:, t:t + 4] for t in range(H)], 1).flatten(0, 1)        # (N*H,4,7)
        r0, r1 = R[:, :-1].flatten(0, 1), R[:, 1:].flatten(0, 1)
        sel = torch.randperm(len(X))[:max_samples]
        X, r0, r1 = X[sel], r0[sel], r1[sel]
        sol = torch.linalg.lstsq(X[..., :3].flatten(1), r1[:, :3] - r0[:, :3]).solution    # (12,3)
        self.Bp.copy_(sol.reshape(4, 3, 3).transpose(1, 2))
        dq = Rotation.from_quat(r1[:, 3:7].numpy()) * Rotation.from_quat(r0[:, 3:7].numpy()).inv()
        sol = torch.linalg.lstsq(X[..., 3:6].flatten(1), torch.from_numpy(dq.as_rotvec()).float()).solution
        self.Br.copy_(sol.reshape(4, 3, 3).transpose(1, 2))
        closing = X[:, -1, 6] > 0
        self.g_close.copy_(r1[closing, 7:9].median(0).values); self.g_open.copy_(r1[~closing, 7:9].median(0).values)


# ---- denoiser over latents ----

class Denoiser2(nn.Module):
    def __init__(self, base=128, mults=(1, 2, 2, 2), cond=512):
        super().__init__()
        self.noise_emb = nn.Sequential(Fourier(cond), nn.Linear(cond, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.act_emb = nn.Sequential(nn.Linear(N_ACT * ACT_DIM + 9 + 1, cond), nn.SiLU(), nn.Linear(cond, cond))
        chs = [base * m for m in mults]
        self.inp = nn.Conv2d(ZC * (len(OFFSETS) + 1), chs[0], 3, padding=1)
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        c = chs[0]
        for i, ch in enumerate(chs):
            self.down.append(nn.ModuleList([Res(c, ch, cond), Res(ch, ch, cond)]))
            c = ch
            self.pool.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1) if i < len(chs) - 1 else nn.Identity())
        self.mid = nn.ModuleList([Res(c, c, cond), Res(c, c, cond)])
        self.mid_attn = Attn(c)
        self.up, self.unpool = nn.ModuleList(), nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            self.up.append(nn.ModuleList([Res(c + ch, ch, cond), Res(ch + ch, ch, cond)]))
            c = ch
            self.unpool.append(nn.Upsample(scale_factor=2) if i > 0 else nn.Identity())
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, ZC, 3, padding=1))

    def forward(self, x_noisy, c_noise, ctx, acts, robot, ctx_sigma):
        # x_noisy (B,ZC,32,32) scaled by c_in; ctx (B,len(OFFSETS),ZC,32,32); acts (B,N_ACT,7); robot (B,9)
        cond = self.noise_emb(c_noise) + self.act_emb(torch.cat([acts.flatten(1), robot, ctx_sigma[:, None]], 1))
        h = self.inp(torch.cat([x_noisy, ctx.flatten(1, 2)], 1))
        skips = []
        for (r1, r2), pool in zip(self.down, self.pool):
            h = r1(h, cond); skips.append(h)
            h = r2(h, cond); skips.append(h)
            h = pool(h)
        for r in self.mid:
            h = r(h, cond)
        h = self.mid_attn(h)
        for (r1, r2), unpool in zip(self.up, self.unpool):
            h = r1(torch.cat([h, skips.pop()], 1), cond)
            h = r2(torch.cat([h, skips.pop()], 1), cond)
            h = unpool(h)
        return self.out(h)


def precond(sigma, sd):
    s2 = sigma ** 2 + sd ** 2
    return sd ** 2 / s2, sigma * sd / s2.sqrt(), 1 / s2.sqrt(), sigma.log() / 4


def edm_loss(model, x, ctx, acts, robot, ctx_sigma, sd):
    sigma = (torch.randn(x.shape[0], device=x.device) * P_STD + P_MEAN).exp()
    c_skip, c_out, c_in, c_noise = precond(sigma, sd)
    v = lambda t: t[:, None, None, None]
    x_noisy = x + v(sigma) * torch.randn_like(x)
    target = (x - v(c_skip) * x_noisy) / v(c_out)
    return F.mse_loss(model(v(c_in) * x_noisy, c_noise, ctx, acts, robot, ctx_sigma), target)


@torch.no_grad()
def sample(model, ctx, acts, robot, sd, steps=3, sigma_min=2e-3, sigma_max=5.0, rho=7.0):
    b = ctx.shape[0]
    i = torch.arange(steps, device=ctx.device)
    sig = (sigma_max ** (1 / rho) + i / max(steps - 1, 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sig = torch.cat([sig, torch.zeros(1, device=sig.device)])
    x = torch.randn(b, ZC, *ctx.shape[-2:], device=ctx.device) * sig[0]
    zero = torch.zeros(b, device=x.device)
    v = lambda t: t[:, None, None, None]
    for k in range(steps):
        s = sig[k].expand(b)
        c_skip, c_out, c_in, c_noise = precond(s, sd)
        d = v(c_skip) * x + v(c_out) * model(v(c_in) * x, c_noise, ctx, acts, robot, zero)
        x = x + (x - d) / sig[k] * (sig[k + 1] - sig[k])
    return x


class SuccessHead(nn.Module):
    """Is the can in the basket? From one step's latents (both cameras)."""

    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ZC, ch, 3, padding=1), nn.SiLU(), nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(2 * ch, 2 * ch, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2 * ch, 1))

    def forward(self, z):
        return self.net(z)[:, 0]


class ValueHead(nn.Module):
    """How good is this moment? Predicts the discounted chance that the episode ends in success,
    gamma ** (steps left) on successful episodes and 0 on failed ones, from one step's latents and the
    robot state. Used to score imagined futures of candidate action chunks."""

    def __init__(self, ch=64, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ZC, ch, 3, padding=1), nn.SiLU(), nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(2 * ch, 2 * ch, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(2 * ch + 9, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, z, robot):
        return self.mlp(torch.cat([self.net(z), robot], 1))[:, 0]


class ActionValueHead(nn.Module):
    """The value after running the next H actions, predicted from now and the actions alone, without
    imagining anything (the no-world-model control for picking among candidate chunks)."""

    def __init__(self, horizon=20, ch=64, hidden=256):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(nn.Conv2d(ZC, ch, 3, padding=1), nn.SiLU(), nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(2 * ch, 2 * ch, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(2 * ch + 9 + horizon * ACT_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, z, robot, acts):                     # acts (B, horizon, 7)
        return self.mlp(torch.cat([self.net(z), robot, acts.flatten(1)], 1))[:, 0]


# ---- the history buffer used both in training and in closed-loop rollouts ----

class History:
    """Keeps the last max(OFFSETS)+1 latents, robot states and N_ACT actions of a batch of episodes.
    At the start the first step is repeated and the actions before it are zero, as in training."""

    def __init__(self, z0, r0):
        self.z = z0[:, None].repeat(1, max(OFFSETS) + 1, 1, 1, 1)       # oldest ... newest
        self.r = r0[:, None].repeat(1, 2, 1)
        self.a = torch.zeros(z0.shape[0], N_ACT, ACT_DIM, device=z0.device)

    def context(self):
        idx = torch.tensor([self.z.shape[1] - 1 - o for o in OFFSETS], device=self.z.device)
        return self.z[:, idx]

    def push_action(self, a):
        self.a = torch.cat([self.a[:, 1:], a[:, None]], 1)

    def push_step(self, z, r):
        self.z = torch.cat([self.z[:, 1:], z[:, None]], 1)
        self.r = torch.cat([self.r[:, 1:], r[:, None]], 1)


def load_vae(path, device="cuda"):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(path, torch_dtype=torch.float16).to(device).eval()
    vae.requires_grad_(False)
    return vae


@torch.no_grad()
def encode(vae, img):
    """img (B,3,256,256) float in [0,1] -> (B,4,32,32) float32 latents (scaled)."""
    x = img.to(vae.device, torch.float16) * 2 - 1
    return vae.encode(x).latent_dist.mode().float() * vae.config.scaling_factor


@torch.no_grad()
def decode(vae, z, chunk=64):
    """(B,4,32,32) scaled latents -> (B,3,256,256) float in [0,1]."""
    out = []
    for i in range(0, z.shape[0], chunk):
        x = vae.decode((z[i:i + chunk] / vae.config.scaling_factor).to(torch.float16)).sample
        out.append(((x.float() + 1) / 2).clamp(0, 1))
    return torch.cat(out)
