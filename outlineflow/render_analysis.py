"""Render two analysis figures from our data:
  Fig 1 - the conditioning 'decisive test' (richer condition -> r up, metric flat).
  Fig 2 - sampling-side diversity hacks fall off-manifold (worse on both axes).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY="#1f3a5f"; ACC="#e6550d"; GREEN="#2ca02c"; GREY="#777777"; BLUE="#3a7bd5"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})

# ---------------- Figure 1: conditioning decisive test ----------------
# all metric rows use align g16 + churn0.3 (baseline = 3-seed; cross rows = single seed)
cond_labels = ["Baseline\n(pooled vector)", "+ cross-attn\n(shape)", "+ cross-attn + scale\n(shape+size)"]
r_vals   = [0.06, 0.09, 0.67]      # count <-> area correlation (model's outline reading)
cov_vals = [0.111, 0.098, 0.098]   # Coverage (higher better)
fid_vals = [135.3, 134.8, 142.6]   # FID (lower better)

fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
fig.suptitle("Decisive test: un-compressing the outline conditioning does NOT move the metrics",
             fontsize=15, fontweight="bold", color=NAVY, y=0.99)

ax=axes[0]; x=np.arange(3)
ax.bar(x, r_vals, color=[GREY,BLUE,NAVY], width=0.6)
ax.axhline(0.94, ls="--", color=ACC, lw=1.5); ax.text(2.0,0.95,"real = 0.94",color=ACC,fontsize=10)
for i,v in enumerate(r_vals): ax.text(i,v+0.02,f"{v:.2f}",ha="center",fontweight="bold")
ax.annotate("", xy=(2,0.70), xytext=(0,0.10),
            arrowprops=dict(arrowstyle="->",color=GREEN,lw=2))
ax.text(0.5,0.55,"outline-reading\nrises 10×",color=GREEN,fontsize=11,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cond_labels,fontsize=9)
ax.set_ylabel("count ↔ area correlation  r"); ax.set_ylim(0,1.05)
ax.set_title("Model reads the outline far better →",fontsize=12,color=NAVY)

ax=axes[1]
ax.bar(x, cov_vals, color=[GREEN,GREY,GREY], width=0.6)
for i,v in enumerate(cov_vals): ax.text(i,v+0.002,f"{v:.3f}",ha="center",fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cond_labels,fontsize=9)
ax.set_ylabel("Coverage  (higher better)"); ax.set_ylim(0,0.14)
ax.set_title("… but Coverage does not improve (even dips)",fontsize=12,color=ACC)
ax.text(1.0,0.125,"flat / worse  →  bottleneck = L2 loss,\nnot the conditioning",
        ha="center",color=ACC,fontsize=10,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("outputs_full_plan_id/fig_conditioning_decisive.png",dpi=150); plt.close(fig)

# ---------------- Figure 2: sampling-side hacks fall off-manifold ----------------
# all align g16; single seed (churn0.3 ref shown as 3-seed mean)
pts = [
 ("churn 0.3  (best)", 135.3, 0.111, GREEN, "*", 360),
 ("count_cond+churn",  153.9, 0.103, "#888", "o", 90),
 ("count_cond",        151.2, 0.080, "#888", "o", 90),
 ("repel 1+churn",     158.1, 0.075, ACC,   "o", 90),
 ("repel 3+churn",     189.1, 0.030, ACC,   "o", 90),
 ("temp 1.15+churn",   144.1, 0.107, BLUE,  "o", 90),
 ("temp 1.3+churn",    155.8, 0.080, BLUE,  "o", 90),
]
# explicit label offsets (FID,Cov) in data units to avoid overlaps
off = {
 "churn 0.3  (best)": (-1.5, 0.003),
 "count_cond+churn":  (3.0, 0.004),
 "count_cond":        (-3.0, 0.004),
 "repel 1+churn":     (5.5, -0.003),
 "repel 3+churn":     (-1.0, 0.006),
 "temp 1.15+churn":   (3.0, 0.004),
 "temp 1.3+churn":    (4.0, -0.006),
}
fig, ax = plt.subplots(figsize=(11.5, 6.2))
for name,fid,cov,col,mk,sz in pts:
    ax.scatter(fid,cov,c=col,marker=mk,s=sz,zorder=3,edgecolor="white",linewidth=0.6)
    dx,dy=off[name]
    ax.annotate(name,(fid,cov),xytext=(fid+dx,cov+dy),fontsize=9.5,
                color=NAVY if "best" in name else "#333",fontweight="bold" if "best" in name else "normal")
ax.scatter([],[],c=ACC,label="(2) particle repulsion"); ax.scatter([],[],c=BLUE,label="(3) noise temperature")
ax.scatter([],[],c="#888",label="(1) per-outline count")
ax.scatter([],[],c=GREEN,marker="*",s=180,label="churn 0.3 (reference)")
ax.annotate("more aggressive sampling\n→ off-manifold → worse on BOTH axes",
            xy=(184,0.040),xytext=(168,0.055),fontsize=11,color=ACC,fontweight="bold",
            arrowprops=dict(arrowstyle="->",color=ACC,lw=1.8))
ax.set_xlabel("FID  (lower better →)  — note: left is better"); ax.invert_xaxis()
ax.set_ylim(0.02,0.122)
ax.set_ylabel("Coverage  (higher better)")
ax.set_title("Sampling-side diversity hacks all fall below churn — they leave the model's manifold",
             fontsize=13,fontweight="bold",color=NAVY)
ax.legend(loc="lower right",fontsize=9,framealpha=0.95); ax.grid(alpha=0.25,zorder=0)
fig.tight_layout()
fig.savefig("outputs_full_plan_id/fig_sampling_offmanifold.png",dpi=150); plt.close(fig)
print("saved fig_conditioning_decisive.png and fig_sampling_offmanifold.png")
