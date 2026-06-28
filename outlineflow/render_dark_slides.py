"""Two full-slide PNGs in the deck's dark Google-Slides theme (bg #15212A, teal accent)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG="#15212A"; TEAL="#009F8C"; WHITE="#FFFFFF"; LGREY="#AEB8BD"; AMBER="#E8A33D"; GREY="#5b6b75"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":WHITE,
                     "axes.edgecolor":"#3a4a55","xtick.color":LGREY,"ytick.color":LGREY,
                     "axes.labelcolor":LGREY})

def newslide():
    fig=plt.figure(figsize=(13.333,7.5),dpi=150); fig.patch.set_facecolor(BG)
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off"); ax.set_facecolor(BG)
    return fig,ax

def header(ax,tag,section):
    ax.text(0.55,8.6,tag,color=TEAL,fontsize=12,fontweight="bold")
    ax.text(2.6,8.6,"OutlineFlow",color=LGREY,fontsize=12)
    ax.text(15.45,8.6,section,color=TEAL,fontsize=12,fontweight="bold",ha="right")
    ax.add_patch(plt.Rectangle((0.55,8.42),14.9,0.03,color=TEAL,lw=0))

def darkax(rect,fig):
    a=fig.add_axes(rect); a.set_facecolor("#1b2d38")
    for s in a.spines.values(): s.set_color("#33454f")
    return a

def takeaway(ax,line1,line2=None):
    ax.add_patch(plt.Rectangle((0.55,0.35),14.9,0.95,color="#22333d",lw=0))
    ax.text(0.85,0.83 if line2 else 0.78,line1,color=TEAL,fontsize=13,fontweight="bold",va="center")
    if line2: ax.text(0.85,0.5,line2,color=TEAL,fontsize=13,fontweight="bold",va="center")

# ================= SLIDE 1: conditioning decisive test =================
fig,ax=newslide(); header(ax,"MAIN / 05·","DECISIVE TEST")
ax.text(0.55,7.78,"Conditioning is not the bottleneck — the loss is",fontsize=27,fontweight="bold",color=WHITE)
ax.text(0.57,7.12,"We un-compressed the outline (cross-attention over all 128 boundary points + scale).",fontsize=13.5,color=LGREY)
ax.text(0.57,6.72,"The model reads the outline far better — yet the metric does not move.",fontsize=13.5,color=LGREY)

labels=["Baseline\n(pooled vec)","+ cross-attn\n(shape)","+ cross-attn\n+ scale"]
r_vals=[0.06,0.09,0.67]; cov=[0.111,0.098,0.098]; x=np.arange(3)
a=darkax([0.07,0.235,0.40,0.40],fig)
a.bar(x,r_vals,color=[GREY,"#3a7fb0",TEAL],width=0.62)
a.axhline(0.94,ls="--",color=AMBER,lw=1.4); a.text(1.35,0.965,"real = 0.94",color=AMBER,fontsize=10)
for i,v in enumerate(r_vals): a.text(i,v+0.03,f"{v:.2f}",ha="center",color=WHITE,fontweight="bold",fontsize=12)
a.annotate("",xy=(2,0.66),xytext=(0,0.10),arrowprops=dict(arrowstyle="->",color=TEAL,lw=2.2))
a.text(0.32,0.46,"outline-reading\nrises 10×",color=TEAL,fontsize=11.5,fontweight="bold")
a.set_xticks(x); a.set_xticklabels(labels,fontsize=9.5,color=LGREY); a.set_ylim(0,1.08)
a.set_ylabel("count ↔ area  r",color=LGREY); a.set_title("Model reads the outline far better →",color=WHITE,fontsize=13)

a=darkax([0.56,0.235,0.40,0.40],fig)
a.bar(x,cov,color=[TEAL,GREY,GREY],width=0.62)
for i,v in enumerate(cov): a.text(i,v+0.003,f"{v:.3f}",ha="center",color=WHITE,fontweight="bold",fontsize=12)
a.set_xticks(x); a.set_xticklabels(labels,fontsize=9.5,color=LGREY); a.set_ylim(0,0.14)
a.set_ylabel("Coverage (↑)",color=LGREY); a.set_title("… but Coverage does not improve",color=AMBER,fontsize=13)

takeaway(ax,"r jumps 10× while the metric stays flat → the limit is the L2 conditional-mean",
         "objective, not how well the model can read the outline.")
fig.savefig("slide_cond_decisive.png",facecolor=BG); plt.close(fig)

# ================= SLIDE 2: sampling off-manifold =================
fig,ax=newslide(); header(ax,"APPENDIX","SAMPLING ABLATION")
ax.text(0.55,7.78,"Sampling-side diversity hacks fall off-manifold",fontsize=27,fontweight="bold",color=WHITE)
ax.text(0.57,7.12,"Beyond churn, three ways to force diversity were tried — all worse on every metric.",fontsize=13.5,color=LGREY)
ax.text(0.57,6.72,"They push samples off the model's learned manifold.",fontsize=13.5,color=LGREY)

pts=[("churn 0.3 (best)",135.3,0.111,TEAL,"*",420),
     ("count_cond+churn",153.9,0.103,GREY,"o",110),
     ("count_cond",151.2,0.080,GREY,"o",110),
     ("repel 1+churn",158.1,0.075,"#d9663a","o",110),
     ("repel 3+churn",189.1,0.030,"#d9663a","o",110),
     ("temp 1.15+churn",144.1,0.107,"#3a7fb0","o",110),
     ("temp 1.3+churn",155.8,0.080,"#3a7fb0","o",110)]
off={"churn 0.3 (best)":(-3.0,-0.006),"count_cond+churn":(2.5,0.004),"count_cond":(-3.5,0.004),
     "repel 1+churn":(5.5,-0.004),"repel 3+churn":(-0.3,0.006),"temp 1.15+churn":(2.5,0.004),"temp 1.3+churn":(4.3,-0.006)}
a=darkax([0.06,0.235,0.50,0.40],fig)
for nm,f,c,col,mk,sz in pts:
    a.scatter(f,c,c=col,marker=mk,s=sz,zorder=3,edgecolor=BG,linewidth=0.8)
    dx,dy=off[nm]; a.annotate(nm,(f,c),xytext=(f+dx,c+dy),fontsize=9,color=WHITE if "best" in nm else LGREY,
                              fontweight="bold" if "best" in nm else "normal")
a.annotate("more aggressive sampling\n→ off-manifold → worse on BOTH",xy=(184,0.040),xytext=(171,0.052),
           fontsize=10,color=AMBER,fontweight="bold",arrowprops=dict(arrowstyle="->",color=AMBER,lw=1.6))
a.set_xlabel("FID (lower better →, left is better)",color=LGREY); a.invert_xaxis()
a.set_ylabel("Coverage (↑)",color=LGREY); a.set_ylim(0.02,0.122); a.grid(alpha=0.15)

tx=10.1; ty=5.2
rows=[("config","FID","Cov"),("churn0.3 (best)","135.3","0.111"),("count_cond","151–154","0.08–0.10"),
      ("repel","158–189","0.03–0.08"),("temp","144–156","0.08–0.11")]
for r,(c1,c2,c3) in enumerate(rows):
    yy=ty-r*0.6; hd=(r==0); col=TEAL if r==1 else (WHITE if hd else LGREY)
    ax.text(tx,yy,c1,fontsize=12,color=col,fontweight="bold" if hd or r==1 else "normal")
    ax.text(tx+2.7,yy,c2,fontsize=12,color=col); ax.text(tx+4.3,yy,c3,fontsize=12,color=col)
    if hd: ax.plot([tx-0.1,tx+5.4],[yy-0.18]*2,color=TEAL,lw=1)
takeaway(ax,"Sampling-time diversity is already saturated by churn; cruder perturbations only leave the manifold.")
fig.savefig("slide_sampling_offmanifold.png",facecolor=BG); plt.close(fig)
print("saved both dark slides")
