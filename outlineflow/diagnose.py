"""Root-cause diagnostics for low Density/Coverage: mode-collapse / diversity."""
import numpy as np, pickle, torch
import params, metrics
from cfg import CFG, ROOM_NAMES, seed_everything
from model import OutlineFlow
from flow import EMA, sample
from postprocess import layout_from_tokens, align_layout

seed_everything(42)
N = 300
ck = torch.load("outputs_full_plan_id/ckpt.pt", map_location="cpu", weights_only=False)
for k in ("n_max","k","n_gen_classes","p_outline","d_model","n_layers","n_heads",
          "mlp_ratio","canvas","nearest_k","min_area_frac"):
    if k in ck["cfg"]: setattr(CFG, k, ck["cfg"][k])
m = OutlineFlow(CFG); m.load_state_dict(ck["model"])
ema = EMA(m, CFG.ema_decay); ema.load_state_dict(ck["ema"]); model = ema.make_model(m)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev); stats = ck["stats"]

held = pickle.load(open("outputs_full_plan_id/held.pkl","rb"))[:N]
outlines = [s["outline"] for s in held]
OUT = np.stack([params.sample_outline_points(o, CFG.p_outline) for o in outlines])
Ot = torch.from_numpy(OUT).to(dev)
raw = []
for i in range(0, len(outlines), 256):
    raw.extend(list(sample(model, Ot[i:i+256], CFG).cpu().numpy()))

real_plans = [(s["room_polys"], s["outline"]) for s in held]
target = float(np.mean([len(r) for r,_ in real_plans]))
# calibrate threshold to real mean (same as eval)
sub = list(zip(raw, outlines))[:40]
def mc(t): return float(np.mean([len(layout_from_tokens(x,o,stats,CFG,presence_thresh=t)) for x,o in sub]))
lo=float(min(x[:,0].min() for x in raw)); hi=float(max(x[:,0].max() for x in raw))
for _ in range(18):
    mid=.5*(lo+hi);
    if mc(mid)>target: lo=mid
    else: hi=mid
th=.5*(lo+hi)
gen_plans = [(layout_from_tokens(x,o,stats,CFG,presence_thresh=th),o) for x,o in zip(raw,outlines)]
gen_al    = [(align_layout(r,o,CFG,grid=16),o) for r,o in gen_plans]

def report(tag, gp):
    rf = metrics.phi_features(real_plans, CFG)
    gf = metrics.phi_features(gp, CFG)
    rz, gz = metrics.standardize(rf, gf)
    print(f"\n================== {tag} ==================")
    # 1 room count
    rc=np.array([len(r) for r,_ in real_plans]); gc=np.array([len(r) for r,_ in gp])
    print(f"[房间数] real {rc.mean():.1f}±{rc.std():.1f} (med {np.median(rc):.0f}) | "
          f"gen {gc.mean():.1f}±{gc.std():.1f} (med {np.median(gc):.0f})")
    # 2 room-type distribution (phi dims 1..K are type hist)
    K=CFG.k
    rt=rf[:,1:1+K].mean(0); gt=gf[:,1:1+K].mean(0)
    print("[房间类型分布] (real% vs gen%)")
    for i in range(K):
        if rt[i]>0.005 or gt[i]>0.005:
            print(f"    {ROOM_NAMES[i]:14s} real {100*rt[i]:5.1f}%  gen {100*gt[i]:5.1f}%")
    # 3 per-dim variance ratio gen/real (collapse if <<1)
    rv=rf.std(0); gv=gf.std(0); ratio=gv/np.where(rv<1e-9,1,rv)
    print(f"[方差塌缩] gen/real 标准差比值: 中位 {np.median(ratio):.2f}  均值 {ratio.mean():.2f}  "
          f"(<1 表示生成更单一; 接近1才健康)")
    # 4 within-set diversity: mean pairwise dist (standardized)
    def mpd(z):
        idx=np.random.default_rng(0).choice(len(z), min(150,len(z)), replace=False)
        z=z[idx]; d=np.sqrt(((z[:,None,:]-z[None,:,:])**2).sum(-1))
        return d[np.triu_indices(len(z),1)].mean()
    print(f"[生成多样性] 集合内平均两两距离: real {mpd(rz):.2f} | gen {mpd(gz):.2f}  "
          f"-> gen/real={mpd(gz)/max(mpd(rz),1e-9):.2f} (远小于1=塌缩)")
    # 5 nearest gen-gen "duplicate" vs nearest real-real
    def nn_self(z):
        d=np.sqrt(((z[:,None,:]-z[None,:,:])**2).sum(-1)); np.fill_diagonal(d,np.inf)
        return d.min(1).mean()
    print(f"[最近邻自距] real {nn_self(rz):.3f} | gen {nn_self(gz):.3f} (gen很小=大量近似重复)")
    # 6 proxy density/coverage (confirms inception trend)
    sc=metrics.score(rf,gf,CFG,proxy=True)
    print(f"[proxy D/C] Density {sc['density']:.3f}  Coverage {sc['coverage']:.3f}  FID {sc['fid']:.2f}")

report("decode only (target~38)", gen_plans)
report("decode + align g16", gen_al)
print("\n[真实房间数分布] 分位:", np.percentile([len(r) for r,_ in real_plans],[10,25,50,75,90]).round(1))
