"""
HouseDiffusion Inference Script
Generates vectorized floorplans using the HouseDiffusion model.

Setup:
    git clone https://github.com/aminshabani/house_diffusion.git
    cd house_diffusion
    pip install -r requirements.txt
    pip install -e .

Download the pretrained checkpoint from:
    https://drive.google.com/file/d/16zKmtxwY5lF6JE-CJGkRf3-OFoD1TrdR/view

Download the RPLAN dataset (or MagicPlan alternative):
    https://github.com/sepidsh/PuzzleFussion?tab=readme-ov-file#magicplan
"""

import os
import io
import argparse
import numpy as np
import torch as th
import PIL.Image as Image
import drawSvg as drawsvg
import cairosvg
import webcolors
from tqdm import tqdm

from house_diffusion.rplanhg_datasets import load_rplanhg_data
from house_diffusion import dist_util, logger
from house_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    update_arg_parser,
)

# ── Color palette for RPLAN room types ──────────────────────────────────────
ID_COLOR = {
    1:  '#EE4D4D',  # living room
    2:  '#C67C7B',  # master room
    3:  '#FFD274',  # kitchen
    4:  '#BEBEBE',  # bathroom
    5:  '#BFE3E8',  # dining room
    6:  '#7BA779',  # child room
    7:  '#E87A90',  # study room
    8:  '#FF8C69',  # second room
    10: '#1F849B',  # guest room
    11: '#727171',  # balcony
    12: '#D3A2C7',  # entrance
    13: '#785A67',  # storage
}
NUM_ROOM_TYPES = 14
DOOR_INDICES   = [11, 12, 13]


# ── Helpers ──────────────────────────────────────────────────────────────────

def bin_to_int(x):
    return int("".join([str(int(i.cpu().data)) for i in x]), 2)


def bin_to_int_sample(sample, resolution=256):
    """Convert binary-encoded coordinates back to continuous values."""
    sample_new = th.zeros([*sample.shape[:3], 2])
    sample = sample.clone()
    sample[sample < 0] = 0
    sample[sample > 0] = 1
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            for k in range(sample.shape[2]):
                sample_new[i, j, k, 0] = bin_to_int(sample[i, j, k, :8])
                sample_new[i, j, k, 1] = bin_to_int(sample[i, j, k, 8:])
    sample_new = sample_new / (resolution / 2) - 1
    return sample_new


def render_floorplan(sample, model_kwargs, out_dir, start_idx, resolution=256):
    """Render the last diffusion step as colored floorplan PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    # sample shape: [steps, batch, seq_len, 2]  — we only use the last step
    last = sample[-1]

    for i in tqdm(range(last.shape[0]), desc="Rendering"):
        draw = drawsvg.Drawing(resolution, resolution, displayInline=False)
        draw.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='white'))

        poly, polys, types = [], [], []
        for j, point in enumerate(last[i]):
            if model_kwargs['src_key_padding_mask'][i][j] == 1:
                continue
            point = point.cpu().data.numpy()
            if j > 0 and (
                model_kwargs['room_indices'][i, j] !=
                model_kwargs['room_indices'][i, j - 1]
            ).any():
                polys.append(poly)
                c = int(np.argmax(
                    model_kwargs['room_types'][i][j - 1].cpu().numpy()
                ))
                types.append(c)
                poly = []
            coord = (point / 2 + 0.5) * resolution
            poly.append((coord[0], coord[1]))
        if poly:
            polys.append(poly)
            c = int(np.argmax(
                model_kwargs['room_types'][i][-1].cpu().numpy()
            ))
            types.append(c)

        for poly, c in zip(polys, types):
            color = ID_COLOR.get(c, '#CCCCCC')
            if len(poly) >= 2:
                draw.append(drawsvg.Lines(
                    *np.array(poly).flatten().tolist(),
                    close=True,
                    fill=color, fill_opacity=1.0,
                    stroke='black', stroke_width=1,
                ))

        out_path = os.path.join(out_dir, f'{start_idx + i}.png')
        Image.open(io.BytesIO(cairosvg.svg2png(draw.asSvg()))).save(out_path)

    print(f"Saved {last.shape[0]} floorplans to '{out_dir}/'")


# ── Main inference loop ──────────────────────────────────────────────────────

def run_inference(args):
    dist_util.setup_dist()
    logger.configure()

    logger.log("Loading model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("Loading dataset...")
    data = load_rplanhg_data(
        batch_size=args.batch_size,
        analog_bit=args.analog_bit,
        set_name=args.set_name,
        target_set=args.target_set,
    )

    sample_fn = (
        diffusion.ddim_sample_loop if args.use_ddim
        else diffusion.p_sample_loop
    )

    os.makedirs('outputs/pred', exist_ok=True)
    tmp_count = 0

    logger.log("Sampling...")
    while tmp_count < args.num_samples:
        data_sample, model_kwargs = next(data)
        for key in model_kwargs:
            model_kwargs[key] = model_kwargs[key].to(dist_util.dev())

        with th.no_grad():
            sample = sample_fn(
                model,
                data_sample.shape,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                analog_bit=args.analog_bit,
            )

        # [steps, batch, seq_len, coords] → permute to [steps, batch, coords, seq_len]
        sample = sample.permute([0, 1, 3, 2])

        if args.analog_bit:
            sample = bin_to_int_sample(sample)

        render_floorplan(sample, model_kwargs, 'outputs/pred', tmp_count)
        tmp_count += data_sample.shape[0]
        logger.log(f"Generated {tmp_count}/{args.num_samples} samples")

    logger.log("Done.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        dataset='rplan',
        clip_denoised=True,
        num_samples=16,        # how many floorplans to generate
        batch_size=8,
        use_ddim=False,
        model_path="ckpts/model250000.pt",  # path to downloaded checkpoint
        set_name="eval",
        target_set=8,
        draw_graph=False,
        save_svg=False,
    ))
    parser = argparse.ArgumentParser(description="HouseDiffusion Inference")
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    update_arg_parser(args)
    run_inference(args)
