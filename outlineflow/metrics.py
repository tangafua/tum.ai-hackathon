"""FID / Density / Coverage + a numpy-only proxy so the eval plumbing runs today.

Two feature backends:
  * phi()            -- permutation-invariant layout descriptor, numpy only,
                        zero extra deps; proves the whole pipeline end-to-end.
  * inception_features() -- the real 2048-d InceptionV3 features used by the
                        canonical metrics (needs `pip install pytorch-fid` or
                        torchvision).  Swap in with a one-line change.

Both feed the SAME fid() and the SAME vendored compute_prdc(), so Density and
Coverage always share the FID feature space (the standard convention).
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import sqrtm

from prdc import compute_prdc


# ----------------------------------------------------------------- phi proxy
def phi(rooms, outline, cfg) -> np.ndarray:
    """Layout -> [K+17] permutation-invariant feature (numpy only)."""
    K = cfg.k
    A = max(outline.area, 1e-9)
    n = len(rooms)
    type_hist = np.zeros(K)
    areas, ws, hs, aspects = [], [], [], []
    for poly, t in rooms:
        type_hist[int(t) % K] += 1
        a = poly.area / A
        areas.append(a)
        x0, y0, x1, y1 = poly.bounds
        w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        ws.append(w); hs.append(h)
        aspects.append(max(w, h) / min(w, h))
    type_hist = type_hist / max(type_hist.sum(), 1.0)
    area_hist, _ = np.histogram(areas, bins=8, range=(0.0, 1.0))
    area_hist = area_hist / max(area_hist.sum(), 1.0)

    def ms(v):
        v = np.asarray(v) if v else np.zeros(1)
        return [float(v.mean()), float(v.std())]

    polys = [p for p, _ in rooms]
    contacts = 0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].distance(polys[j]) < 1e-6:
                contacts += 1
    covered = 0.0
    if polys:
        from shapely.ops import unary_union
        covered = unary_union(polys).intersection(outline).area / A
    bg_frac = max(0.0, 1.0 - covered)

    return np.array(
        [n] + type_hist.tolist() + area_hist.tolist()
        + ms(ws) + ms(hs) + ms(aspects) + [bg_frac, contacts],
        dtype=np.float64,
    )


def phi_features(plans, cfg) -> np.ndarray:
    """plans: list of (rooms, outline) -> [N, K+17]."""
    return np.stack([phi(r, o, cfg) for (r, o) in plans])


def standardize(real, fake):
    """z-score both sets by the REAL set's per-dim mean/std (proxy only)."""
    mu = real.mean(0)
    sd = real.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (real - mu) / sd, (fake - mu) / sd


# ----------------------------------------------------------------- FID
def fid(fr: np.ndarray, fg: np.ndarray, eps: float = 1e-6) -> float:
    mu1, mu2 = fr.mean(0), fg.mean(0)
    s1, s2 = np.cov(fr, rowvar=False), np.cov(fg, rowvar=False)
    s1 = np.atleast_2d(s1); s2 = np.atleast_2d(s2)
    diff = mu1 - mu2
    cov = sqrtm(s1 @ s2)
    if np.iscomplexobj(cov):
        cov = cov.real
    if not np.isfinite(cov).all():
        off = np.eye(s1.shape[0]) * eps
        cov = sqrtm((s1 + off) @ (s2 + off))
        cov = cov.real if np.iscomplexobj(cov) else cov
    return float(diff @ diff + np.trace(s1 + s2 - 2 * cov))


def score(real_feats, fake_feats, cfg, proxy=True):
    """Return dict(fid, density, coverage). If proxy, z-score features first."""
    r, f = (standardize(real_feats, fake_feats) if proxy
            else (real_feats, fake_feats))
    out = {"fid": fid(r, f)}
    out.update(compute_prdc(r, f, nearest_k=cfg.nearest_k))
    return {"fid": out["fid"], "density": out["density"], "coverage": out["coverage"]}


# ----------------------------------------------------------------- sanity harness
def sanity_check(real_feats, cfg, proxy=True):
    """Calibration: real-vs-real ~ (FID~0, D~1, C~1); noise ~ (FID large, D~0, C~0)."""
    rng = np.random.default_rng(0)
    same = score(real_feats, real_feats.copy(), cfg, proxy=proxy)
    noise = real_feats + 50.0 * rng.standard_normal(real_feats.shape)
    shifted = score(real_feats, noise, cfg, proxy=proxy)
    print(f"  [sanity] real-vs-real : FID={same['fid']:.3f}  "
          f"D={same['density']:.3f}  C={same['coverage']:.3f}  (expect ~0, ~1, ~1)")
    print(f"  [sanity] real-vs-noise: FID={shifted['fid']:.1f}  "
          f"D={shifted['density']:.3f}  C={shifted['coverage']:.3f}  (expect large, ~0, ~0)")
    ok = same["fid"] < 1.0 and same["coverage"] > 0.8 and shifted["coverage"] < 0.2
    print(f"  [sanity] {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------- OFFICIAL eval
def official_eval(real_imgs, gen_imgs, cfg, device="cpu"):
    """The exact challenge metrics.

      FID      -> torchmetrics.image.fid.FrechetInceptionDistance (feature=2048)
      Density  -> clovaai/generative-evaluation-prdc compute_prdc (vendored, k=5)
      Coverage -> same

    Density/Coverage reuse the SAME InceptionV3 features torchmetrics uses for FID
    (fid.inception), so all three share one feature space, as is standard.

    Requires (in the SUBMISSION env): pip install torchmetrics torch-fidelity torchvision.
    Render real_imgs and gen_imgs with render.render_msd so they match the MSD plot.py.
    """
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    def to_u8(a):
        return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).to(torch.uint8)

    r, g = to_u8(real_imgs).to(device), to_u8(gen_imgs).to(device)
    fid.update(r, real=True)
    fid.update(g, real=False)
    fid_val = float(fid.compute())
    with torch.no_grad():                       # share Inception features with prdc
        rf = fid.inception(r).cpu().numpy().astype(np.float64)
        gf = fid.inception(g).cpu().numpy().astype(np.float64)
    dc = compute_prdc(rf, gf, nearest_k=cfg.nearest_k)
    return {"fid": fid_val, "density": dc["density"], "coverage": dc["coverage"]}


# ----------------------------------------------------------------- Inception (alt.)
def inception_features(images_uint8, device="cpu", batch=32):
    """images_uint8: [N,H,W,3] -> [N,2048] InceptionV3 pool3 features.

    Requires `pip install pytorch-fid` (preferred) or torchvision.
    """
    import torch
    try:
        from pytorch_fid.inception import InceptionV3
        block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        net = InceptionV3([block]).to(device).eval()
        use_pf = True
    except Exception:
        from torchvision.models import inception_v3, Inception_V3_Weights
        net = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        net.fc = torch.nn.Identity()
        net = net.to(device).eval()
        use_pf = False

    feats = []
    with torch.no_grad():
        for i in range(0, len(images_uint8), batch):
            x = torch.from_numpy(images_uint8[i:i + batch]).float().permute(0, 3, 1, 2) / 255.0
            x = torch.nn.functional.interpolate(x, size=(299, 299),
                                                mode="bilinear", align_corners=False)
            x = x.to(device)
            if use_pf:
                f = net(x)[0].squeeze(-1).squeeze(-1)
            else:
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                f = net((x - mean) / std)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, 0).astype(np.float64)
