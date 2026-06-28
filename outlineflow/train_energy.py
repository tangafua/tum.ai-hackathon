"""Train the EnergyCritic to separate REAL vs flow-GENERATED layout tokens
(time-conditioned, so it gives useful guidance gradients at every ODE step).

    python train_energy.py --flow_ckpt outputs_full_plan_id/ckpt.pt \
        --data_csv ../mds_V2_5.372k.csv --steps 4000

Saves <out_dir>/energy_critic.pt {model, cfg, flow_ckpt}.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F

import params, synth_data, msd_data
from cfg import CFG, seed_everything
from model import OutlineFlow
from flow import EMA, sample
from energy import EnergyCritic


def load_flow(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("n_max", "k", "n_gen_classes", "p_outline", "d_model", "n_layers",
              "n_heads", "mlp_ratio", "canvas", "nearest_k", "min_area_frac", "use_scale"):
        if k in ck["cfg"]:
            setattr(CFG, k, ck["cfg"][k])
    m = OutlineFlow(CFG); m.load_state_dict(ck["model"], strict=False)
    e = EMA(m, CFG.ema_decay); e.load_state_dict(ck["ema"])
    return e.make_model(m).to(dev), ck["stats"], ck.get("scale_stats")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_ckpt", default="outputs_full_plan_id/ckpt.pt")
    ap.add_argument("--data_csv", default="/root/jiahua_code/tum.ai-hackathon/mds_V2_5.372k.csv")
    ap.add_argument("--group", default="plan_id")
    ap.add_argument("--msd_limit", type=int, default=2500)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n_critic_layers", type=int, default=3)
    ap.add_argument("--churn", type=float, default=0.0,
                    help="generate fakes with this SDE churn (match the eval sampler)")
    ap.add_argument("--out_dir", default="outputs_full_plan_id")
    ap.add_argument("--ckpt_name", default="energy_critic.pt")
    ap.add_argument("--device", default=CFG.device)
    args = ap.parse_args()

    seed_everything(42)
    dev = args.device
    model, stats, scale_stats = load_flow(args.flow_ckpt, dev)

    # real layouts (token space) + their outlines, using the FLOW ckpt's stats/n_max
    print("[data] loading real plans...")
    samples, _ = msd_data.load_msd_samples(args.data_csv, CFG, limit=args.msd_limit,
                                           group=args.group, set_cfg=False)
    rng = np.random.default_rng(42)
    X, OUT = synth_data.build_tensors(samples, stats, CFG, rng)
    Xr = torch.from_numpy(X).to(dev)              # real tokens [N,n_max,D]
    Ot = torch.from_numpy(OUT).to(dev)            # outline points [N,P,4]
    N = Xr.shape[0]
    print(f"[data] real {Xr.shape}, outlines {Ot.shape}")

    # scale condition for the flow (if it uses it) -- generated fakes match the flow
    St = None
    if getattr(CFG, "use_scale", False) and scale_stats is not None:
        S = (np.stack([params.outline_scale(s["outline"]) for s in samples]) - scale_stats[0]) / scale_stats[1]
        St = torch.from_numpy(S.astype(np.float32)).to(dev)

    # generate fakes from the flow model on the SAME outlines
    print("[data] generating fake layouts from the flow model...")
    Xf = torch.empty_like(Xr)
    with torch.no_grad():
        for i in range(0, N, args.batch):
            sb = St[i:i + args.batch] if St is not None else None
            Xf[i:i + args.batch] = sample(model, Ot[i:i + args.batch], CFG, scale=sb,
                                          churn=args.churn)

    critic = EnergyCritic(CFG, n_layers=args.n_critic_layers).to(dev)
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[critic] params={sum(p.numel() for p in critic.parameters())/1e6:.2f}M")

    def noise(x1, t):
        x0 = torch.randn_like(x1)
        return (1 - t)[:, None, None] * x0 + t[:, None, None] * x1

    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, N, (args.batch,), device=dev)
        t = torch.rand(args.batch, device=dev)
        o = Ot[idx]
        xr = noise(Xr[idx], t); xf = noise(Xf[idx], t)
        lr_real = critic(xr, t, o); lr_fake = critic(xf, t, o)
        loss = (F.binary_cross_entropy_with_logits(lr_real, torch.ones_like(lr_real))
                + F.binary_cross_entropy_with_logits(lr_fake, torch.zeros_like(lr_fake)))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            acc = ((lr_real > 0).float().mean() + (lr_fake < 0).float().mean()).item() / 2
            print(f"  step {step:5d}/{args.steps}  loss {loss.item():.4f}  acc {acc:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = f"{args.out_dir}/{args.ckpt_name}"
    torch.save({"model": critic.state_dict(), "cfg": vars(CFG),
                "n_critic_layers": args.n_critic_layers, "flow_ckpt": args.flow_ckpt,
                "train_churn": args.churn}, out)
    print(f"[done] saved {out}")


if __name__ == "__main__":
    main()
