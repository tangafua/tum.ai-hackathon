"""Render the pitch deck to PITCH.pdf using matplotlib (no browser/network needed)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import os

NAVY="#1f3a5f"; ACC="#e6550d"; GREY="#555555"; GREEN="#2ca02c"
plt.rcParams.update({"font.family":"DejaVu Sans"})

def base(ax):
    ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off")

def title(ax, t, sub=None):
    ax.add_patch(plt.Rectangle((0,8.05),16,0.06,color=ACC,lw=0))
    ax.text(0.6,8.35,t,fontsize=30,fontweight="bold",color=NAVY,va="bottom")
    if sub: ax.text(0.62,7.7,sub,fontsize=15,color=GREY,va="bottom")

def bullets(ax,items,y0=7.0,dy=0.95,size=18,x=0.9):
    for i,it in enumerate(items):
        lvl,txt=(it if isinstance(it,tuple) else (0,it))
        ax.text(x+lvl*0.7, y0-i*dy, ("•  " if lvl==0 else "–  ")+txt,
                fontsize=size-lvl*2, color="#222222", va="top")

def table(ax, rows, x=0.9, y=6.6, col_w=(6.0,2.6,2.9,3.0), rh=0.85, hl_row=None):
    for r,row in enumerate(rows):
        yy=y-r*rh; xx=x
        head=(r==0)
        if hl_row is not None and r==hl_row:
            ax.add_patch(plt.Rectangle((x-0.2,yy-rh+0.18),sum(col_w)+0.2,rh,color="#fdebd9",lw=0))
        for c,cell in enumerate(row):
            ax.text(xx, yy, cell, fontsize=15.5,
                    fontweight="bold" if head or c==0 else "normal",
                    color=NAVY if head else "#222222", va="top")
            xx+=col_w[c]
        if head:
            ax.plot([x-0.2,x+sum(col_w)],[yy-rh+0.55]*2,color=NAVY,lw=1.2)

pdf=PdfPages("PITCH.pdf")
def page():
    fig=plt.figure(figsize=(12.8,7.2),dpi=150); ax=fig.add_axes([0,0,1,1]); base(ax); return fig,ax

# 1 Title
fig,ax=page()
ax.add_patch(plt.Rectangle((0,0),16,9,color="#f7f9fc",lw=0))
ax.add_patch(plt.Rectangle((0,5.0),16,0.08,color=ACC,lw=0))
ax.text(0.8,5.7,"OutlineFlow",fontsize=52,fontweight="bold",color=NAVY)
ax.text(0.85,4.3,"From a bare outline  →  a full apartment, as vector polygons",fontsize=21,color="#333333")
ax.text(0.85,3.1,"Mirror Mirror on the Wall — DAVIS × Paris 2026",fontsize=16,color=GREY)
ax.text(0.85,2.55,"Conditional layout generation on Modified Swiss Dwellings",fontsize=16,color=GREY)
pdf.savefig(fig); plt.close(fig)

# 2 Task
fig,ax=page(); title(ax,"The task")
bullets(ax,[
 "Input: one apartment / floor OUTLINE  (the only condition)",
 "Output: a set of TYPED ROOM POLYGONS  (vector, never pixels)",
 "Model: diffusion or flow matching, from scratch, seed 42",
 "Scored on:  FID (realism ↓)  +  Density & Coverage (diversity ↑)",
])
ax.text(0.9,2.0,"Data: Modified Swiss Dwellings — full set: 5,372 plans / 203k rooms",
        fontsize=16,color=GREEN,fontweight="bold")
pdf.savefig(fig); plt.close(fig)

# 3 Model
fig,ax=page(); title(ax,"Our model — pure flow matching")
bullets(ax,[
 "From-scratch rectified-flow set-Transformer  v(x_t, t, outline)",
 "Rooms are a SET → no positional encoding; full self-attention",
 (1,"learns a joint, non-overlapping, outline-filling layout"),
 "Outline: permutation-invariant PointNet;  time+outline via DiT AdaLN-Zero",
 "Decode clips + resolves overlaps + gap-fills  →  union(rooms) == outline",
],dy=0.92)
ax.text(0.9,1.7,"1.33M params · trained on all 5,372 plans",fontsize=16,color=NAVY,fontweight="bold")
pdf.savefig(fig); plt.close(fig)

# 4 First results & wall
fig,ax=page(); title(ax,"First results — and a wall")
table(ax,[["","FID ↓","Density ↑","Coverage ↑"],
          ["Baseline",  "167","0.097","0.067"],
          ["ideal",     "0","1.0","1.0"]],y=6.6)
ax.text(0.9,3.4,"Geometry clean, room counts right …  but Density & Coverage ≈ 0.07–0.09.",fontsize=17,color="#222")
ax.text(0.9,2.6,"The model is NOT diverse enough — it keeps drawing the same",fontsize=17,color=ACC,fontweight="bold")
ax.text(0.9,2.0,"'average' apartment.",fontsize=17,color=ACC,fontweight="bold")
pdf.savefig(fig); plt.close(fig)

# 5 Diagnosis
fig,ax=page(); title(ax,"Diagnosis — it's the objective, not the model")
ax.add_patch(plt.Rectangle((0.7,6.2),14.6,1.1,color="#fdebd9",lw=0))
ax.text(0.95,6.75,"L2 training regresses to the CONDITIONAL MEAN of all valid layouts.",
        fontsize=18,fontweight="bold",color=NAVY,va="center")
bullets(ax,[
 "Many valid plans per outline → their mean is one blurred plan → variance collapse",
 "Generated room-count std  ±6   vs real  ±24  (can't do tiny or huge plans)",
 "Room-count ↔ area correlates at r = 0.88, but the encoder normalizes size away",
],y0=5.3,dy=0.95,size=17)
ax.text(0.9,1.7,"→ This predicts what will and won't work.",fontsize=17,color=GREEN,fontweight="bold")
pdf.savefig(fig); plt.close(fig)

# 6 Three levers
fig,ax=page(); title(ax,"Three complementary levers")
table(ax,[["Lever","Targets","Idea"],
          ["1 · Grid-align (post-proc)","FID","snap rooms to building axes → clean tiling"],
          ["2 · SDE / churn sampling","Coverage","inject noise → escape the mean trajectory"],
          ["3 · Adversarial / EBM","Density","critic pulls samples to the real manifold"]],
      col_w=(5.3,2.6,7.0),y=6.6)
ax.text(0.9,2.4,"Each attacks a DIFFERENT metric.  Lever 2 (churn) directly undoes the",fontsize=16,color="#222")
ax.text(0.9,1.8,"variance collapse — no retraining needed.",fontsize=16,color=ACC,fontweight="bold")
pdf.savefig(fig); plt.close(fig)

# 7 Results (numbers)
fig,ax=page(); title(ax,"Results")
table(ax,[["Config (3-seed mean)","FID ↓","Density ↑","Coverage ↑"],
          ["Baseline","167","0.097","0.067"],
          ["+ grid-align","131.5","0.091","0.098"],
          ["+ churn  (final)","135.3","0.088","0.111"]],y=6.4,hl_row=3)
ax.text(0.9,2.9,"FID −19%,   Coverage +66%   over the raw baseline.",
        fontsize=20,color=GREEN,fontweight="bold")
ax.text(0.9,2.0,"Shipped as  generate(outline)  →  vector polygons, deterministic at seed 42.",
        fontsize=16,color="#222")
pdf.savefig(fig); plt.close(fig)

# 7b Qualitative
fig,ax=page(); title(ax,"Qualitative samples","left: real plan      right: generate(outline)")
img_path="outputs_full_plan_id/SUBMISSION_generate_demo.png"
if os.path.exists(img_path):
    im=mpimg.imread(img_path)
    ih,iw=im.shape[:2]; aspect=iw/ih               # width/height
    bh=0.74; bw=bh*(7.2/12.8)*aspect               # keep aspect, fit by height
    iax=fig.add_axes([0.5-bw/2,0.04,bw,bh]); iax.imshow(im); iax.axis("off")
pdf.savefig(fig); plt.close(fig)

# 8 Paradigms
fig,ax=page(); title(ax,"We tested all three paradigms — honestly")
table(ax,[["Paradigm (align g16)","FID","Density","Coverage"],
          ["Flow matching + churn","135","0.088","0.111"],
          ["DDPM (pure diffusion)","137","0.068","0.090"],
          ["Flow + EBM guidance","136","0.091","0.103"]],y=6.6,hl_row=1)
bullets(ax,[
 "Diffusion hits the SAME L2 ceiling — no better.",
 "EBM gains wash out under multi-seed (a churn run of 0.122 → 0.097 averaged!).",
],y0=2.7,dy=0.8,size=16)
pdf.savefig(fig); plt.close(fig)

# 9 Takeaways
fig,ax=page(); title(ax,"Key takeaways")
bullets(ax,[
 "1.  The bottleneck is the LOSS, not the architecture",
 (1,"capacity, depth, paradigm swaps all fail"),
 "2.  Sampling-time stochasticity is the real diversity lever",
 (1,"simple, robust, free"),
 "3.  Multi-seed or it didn't happen",
 (1,"stochastic pipelines make single runs lie"),
],y0=7.0,dy=0.92,size=19)
ax.add_patch(plt.Rectangle((0.7,1.0),14.6,1.2,color="#eef3f9",lw=0))
ax.text(0.95,1.6,"Final: flow matching + grid-align + churn  →  FID 135 / Den 0.088 / Cov 0.111",
        fontsize=16,fontweight="bold",color=NAVY,va="center")
pdf.savefig(fig); plt.close(fig)

pdf.close()
print("saved PITCH.pdf  (%d slides)"% 10)
