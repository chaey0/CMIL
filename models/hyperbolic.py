# models/hyperbolic.py
import torch


def clamp_tangent_norm(u, max_norm=1.0, eps=1e-6):
    if max_norm is None or max_norm <= 0:
        return u
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    factor = torch.clamp(max_norm / norm, max=1.0)
    return u * factor


def expmap0_poincare(u, c=1.0, max_tangent_norm=1.0, eps=1e-6):
    c = float(c)
    sqrt_c = c ** 0.5
    u = clamp_tangent_norm(u, max_tangent_norm, eps=eps)

    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    x = torch.tanh(sqrt_c * norm) * u / (sqrt_c * norm)

    # keep inside ball
    max_ball_norm = (1.0 - 1e-5) / sqrt_c
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    factor = torch.clamp(max_ball_norm / x_norm, max=1.0)
    return x * factor


def poincare_distance(x, y, c=1.0, eps=1e-6):
    c = float(c)
    x2 = (x ** 2).sum(dim=-1, keepdim=True)
    y2 = (y ** 2).sum(dim=-1, keepdim=True).transpose(0, 1)

    diff2 = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(dim=-1)
    denom = (1.0 - c * x2).clamp_min(eps) * (1.0 - c * y2).clamp_min(eps)
    z = 1.0 + 2.0 * c * diff2 / denom
    return torch.acosh(z.clamp_min(1.0 + eps)) / (c ** 0.5)


def expmap0_lorentz(u, c=1.0, max_tangent_norm=1.0, eps=1e-6):
    c = float(c)
    sqrt_c = c ** 0.5
    u = clamp_tangent_norm(u, max_tangent_norm, eps=eps)

    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)

    time = torch.cosh(sqrt_c * norm) / sqrt_c
    space = torch.sinh(sqrt_c * norm) * u / (sqrt_c * norm)

    return torch.cat([time, space], dim=-1)


def lorentz_inner(x, y):
    return -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)


def lorentz_distance(x, y, c=1.0, eps=1e-6):
    # x: [B, D+1], y: [C, D+1]
    ip = -x[:, :1] * y[:, :1].T + x[:, 1:] @ y[:, 1:].T
    z = (-float(c) * ip).clamp_min(1.0 + eps)
    return torch.acosh(z) / (float(c) ** 0.5)


def hyperbolic_distance_from_tangent(u_img, u_bank, hyp_cfg):
    geometry = str(hyp_cfg.get("geometry", "poincare")).lower()
    c = float(hyp_cfg.get("curvature", 1.0))
    max_tangent_norm = float(hyp_cfg.get("max_tangent_norm", 1.0))

    if geometry == "poincare":
        x = expmap0_poincare(u_img, c=c, max_tangent_norm=max_tangent_norm)
        t = expmap0_poincare(u_bank, c=c, max_tangent_norm=max_tangent_norm)
        return poincare_distance(x, t, c=c)

    if geometry == "lorentz":
        x = expmap0_lorentz(u_img, c=c, max_tangent_norm=max_tangent_norm)
        t = expmap0_lorentz(u_bank, c=c, max_tangent_norm=max_tangent_norm)
        return lorentz_distance(x, t, c=c)

    raise ValueError(f"Unsupported hyperbolic geometry: {geometry}")


def hyperbolic_logits_from_tangent(u_img, u_bank, hyp_cfg):
    dist = hyperbolic_distance_from_tangent(u_img, u_bank, hyp_cfg)

    tau = float(hyp_cfg.get("tau", 0.07))
    squared = bool(hyp_cfg.get("squared_distance", True))

    if squared:
        return -dist.pow(2) / tau
    return -dist / tau