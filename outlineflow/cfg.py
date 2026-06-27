"""Single config for OutlineFlow.

Everything tunable lives here so train / sample / render / eval stay in sync.
The PALETTE + CANVAS + NEAREST_K are *load-bearing* for the metrics: to compare
against the organizers' numbers they MUST match the organizers' eval/render
snippet.  Until that snippet is in hand we use a self-consistent convention
(real and generated are always rendered with the SAME function, so our internal
scores are valid even if absolute FID differs from theirs).
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import os
import random
import numpy as np
import torch


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_everything(seed: int = 42) -> None:
    """Fix every RNG the pipeline touches.

    The challenge brief mandates a FIXED seed of 42 throughout data, training,
    sampling and evaluation so results are reproducible / comparable across teams.
    Call this once at the top of every entry point (train / sample_eval / generate).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)              # also seeds the default (CPU) generator
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# Room taxonomy + colors are LOCKED to the official MSD repo
# (caspervanengelenburg/msd  constants.py: ROOM_NAMES / COLORS_ROOMTYPE), because
# the organizers render with that repo's plot.py before computing FID/Density/Coverage.
# type id == index into ROOM_NAMES == index into PALETTE == the MSD room_type integer.
ROOM_NAMES: List[str] = [
    "Bedroom", "Livingroom", "Kitchen", "Dining", "Corridor", "Stairs",
    "Storeroom", "Bathroom", "Balcony",          # 0-8: dwelling rooms (generated)
    "Structure", "Door", "Entrance Door", "Window",  # 9-12: structural/openings
]

# COLORS_ROOMTYPE from MSD constants.py, hex -> RGB.
PALETTE: List[Tuple[int, int, int]] = [
    (31, 119, 180),   # 0  Bedroom        #1f77b4
    (230, 85, 13),    # 1  Livingroom     #e6550d
    (253, 141, 60),   # 2  Kitchen        #fd8d3c
    (253, 174, 107),  # 3  Dining         #fdae6b
    (253, 208, 162),  # 4  Corridor       #fdd0a2
    (114, 36, 108),   # 5  Stairs         #72246c
    (82, 84, 163),    # 6  Storeroom      #5254a3
    (107, 110, 207),  # 7  Bathroom       #6b6ecf
    (44, 160, 44),    # 8  Balcony        #2ca02c
    (0, 0, 0),        # 9  Structure      #000000
    (255, 192, 0),    # 10 Door           #ffc000
    (152, 223, 138),  # 11 Entrance Door  #98df8a
    (214, 39, 40),    # 12 Window         #d62728
]

BG_COLOR = (0, 0, 0)         # MSD renders plans on a BLACK background
UNASSIGNED_COLOR = (0, 0, 0)  # uncovered interior reads as wall/black, as in MSD


@dataclass
class Config:
    # --- parameterization (params.py is the single source of truth for ORDER) ---
    n_max: int = 16          # max rooms per plan (set from MSD room-count 99th pct later)
    k: int = 13              # room-class count = len(ROOM_NAMES) (MSD); locked palette width
    n_synth_classes: int = 9  # synthetic rooms use the 9 dwelling classes (0-8)
    # Only the first n_gen_classes type channels are generatable: synthetic uses 0-8 (9);
    # real MSD 'area' rooms use 0-9 (10, incl. Structure). Decoding argmaxes over these
    # so the model can NEVER emit Door/Window/Entrance-Door (10-12), which are openings,
    # not 'area' rooms. Set from data in msd_data; persisted in the checkpoint.
    n_gen_classes: int = 9
    p_outline: int = 128     # boundary points sampled for the outline encoder

    # channel layout (DO NOT reorder without updating params.py):
    #   [0]   presence  (+1 real / -1 padding)
    #   [1:5] cx, cy, w, h   (standardized)
    #   [5:7] sin2theta, cos2theta
    #   [7:7+k] type one-hot in {-1,+1}
    @property
    def d(self) -> int:
        return 7 + self.k

    # --- model ---
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4

    # --- training ---
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    steps: int = 6000
    warmup_steps: int = 200

    # loss weights
    w_presence: float = 2.0
    w_geometry: float = 1.0
    w_type: float = 0.5
    w_pad_slot: float = 0.3   # down-weight padding (absent) slots

    # --- sampling ---
    sample_steps: int = 100

    # --- data ---
    n_train: int = 4000
    n_held: int = 1000
    min_rooms: int = 3
    max_rooms: int = 8
    seed: int = 42                # brief: fixed seed 42 throughout

    # --- real MSD ---
    msd_group: str = "plan_id"    # brief appendix groups by plan_id; "unit_id" = per-apartment
    msd_residential_only: bool = False  # brief does not filter usage; keep all 'area' rooms

    # --- render / eval ---
    canvas: int = 256
    nearest_k: int = 5
    min_area_frac: float = 0.005   # drop slivers below this fraction of outline area

    device: str = field(default_factory=pick_device)
    out_dir: str = "outputs"


CFG = Config()
