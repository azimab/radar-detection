#!/usr/bin/env python3
"""Average the weights of several epoch checkpoints into one checkpoint.

This is a post-hoc stand-in for weight EMA, which RadarPillar's train_utils does not
implement. Epoch-to-epoch val AP swings by ~4.5 AP at high LR, and averaging the
weights of neighbouring epochs is a cheap way to get a flatter, more stable model.

Input checkpoints use the layout of ``train_utils.checkpoint_state``:
    {'epoch', 'it', 'model_state', 'optimizer_state', 'version'}
The output keeps that layout, so ``test.py --ckpt <out>`` loads it through
``load_params_from_file``, which only reads ``model_state`` (and ``version`` for
logging). ``optimizer_state`` is set to None because an averaged model has no
matching optimizer state, so do not resume training from the output.

Floating-point tensors are averaged in float64 and cast back to their dtype.
Integer tensors (BatchNorm ``num_batches_tracked``) are copied from the newest
checkpoint. BatchNorm running statistics are averaged like any other float buffer.
That is the usual approximation. Recomputing them on training data would be exact,
but it needs a forward pass over the train set.

Examples:
    # last 5 epoch checkpoints in a run
    average_checkpoints.py --run_dir output/rear_models/rear_radarpillar_car_v3/v3_seed1 --last 5
    # an explicit epoch window, e.g. the smoothed pick from summarize_eval.py
    average_checkpoints.py --run_dir ... --epochs 26-28
    average_checkpoints.py --ckpt_dir .../ckpt --epochs 24,27,30 --out /tmp/avg.pth

The default output is ``<run_dir>/ckpt_avg/avg_epoch_<first>_to_<last>.pth``. The
name deliberately avoids ``checkpoint_epoch_`` so ``test.py --eval_all`` never picks
it up. ``test.py`` takes the last number in the file name as the epoch id, so the
result is written under ``eval/epoch_<last>/<split>/<eval_tag>``. Pass a distinct
``--eval_tag`` so it cannot be confused with the single epoch.
"""

import argparse
import re
from collections import OrderedDict
from pathlib import Path

import torch

CKPT_PATTERN = re.compile(r"checkpoint_epoch_(\d+)\.pth$")


def find_epoch_checkpoints(ckpt_dir):
    """{epoch: path} for every checkpoint_epoch_<n>.pth in ckpt_dir."""
    found = {}
    for path in Path(ckpt_dir).iterdir():
        match = CKPT_PATTERN.search(path.name)
        if match:
            found[int(match.group(1))] = path
    return found


def parse_epochs(spec):
    """'26-28' -> [26, 27, 28]; '24,27,30' -> [24, 27, 30]; mixed forms allowed."""
    epochs = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"empty epoch range {part!r}")
            epochs.extend(range(lo, hi + 1))
        else:
            epochs.append(int(part))
    if not epochs:
        raise ValueError(f"no epochs in {spec!r}")
    return sorted(set(epochs))


def select_epochs(available, epochs=None, last=None):
    available = sorted(available)
    if (epochs is None) == (last is None):
        raise ValueError("give exactly one of epochs / last")
    if last is not None:
        if last < 1 or last > len(available):
            raise ValueError(f"--last {last} but {len(available)} checkpoints available")
        return available[-last:]
    missing = sorted(set(epochs) - set(available))
    if missing:
        raise ValueError(f"no checkpoint for epochs {missing}")
    return sorted(epochs)


def average_model_states(states):
    """Uniform average of a list of model state dicts (same keys and shapes)."""
    if not states:
        raise ValueError("nothing to average")
    reference = states[-1]
    keys = list(reference.keys())
    for index, state in enumerate(states):
        if set(state.keys()) != set(keys):
            diff = sorted(set(state.keys()) ^ set(keys))
            raise ValueError(f"state {index} has different keys: {diff[:5]}")
        for key in keys:
            if state[key].shape != reference[key].shape:
                raise ValueError(f"state {index} key {key}: shape {tuple(state[key].shape)} "
                                 f"!= {tuple(reference[key].shape)}")

    averaged = OrderedDict()
    for key in keys:
        ref = reference[key]
        if ref.is_floating_point():
            acc = torch.zeros(ref.shape, dtype=torch.float64)
            for state in states:
                acc += state[key].detach().to("cpu", torch.float64)
            averaged[key] = (acc / len(states)).to(ref.dtype)
        else:
            # num_batches_tracked and other integer buffers: keep the newest.
            averaged[key] = ref.detach().cpu().clone()
    return averaged


def average_checkpoint_files(paths):
    """Load checkpoints (oldest first) and return one averaged checkpoint dict."""
    checkpoints = [torch.load(str(p), map_location="cpu", weights_only=False) for p in paths]
    for path, ckpt in zip(paths, checkpoints):
        if "model_state" not in ckpt or ckpt["model_state"] is None:
            raise ValueError(f"{path} has no model_state")
    newest = checkpoints[-1]
    return {
        "epoch": newest.get("epoch"),
        "it": newest.get("it"),
        "model_state": average_model_states([c["model_state"] for c in checkpoints]),
        "optimizer_state": None,
        "version": newest.get("version", "none"),
        "averaged_from_epochs": [c.get("epoch") for c in checkpoints],
        "averaged_from_files": [str(p) for p in paths],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--run_dir", help="output/<exp>/<cfg>/<run_tag> (uses its ckpt/)")
    where.add_argument("--ckpt_dir", help="directory holding checkpoint_epoch_*.pth")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--last", type=int, help="average the last K epoch checkpoints")
    which.add_argument("--epochs", help="epoch list/range, e.g. 26-28 or 24,27,30")
    parser.add_argument("--out", default=None, help="output .pth path")
    args = parser.parse_args(argv)

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(args.run_dir) / "ckpt"
    available = find_epoch_checkpoints(ckpt_dir)
    if not available:
        raise SystemExit(f"no checkpoint_epoch_*.pth in {ckpt_dir}")
    epochs = select_epochs(
        available, epochs=parse_epochs(args.epochs) if args.epochs else None, last=args.last
    )
    paths = [available[e] for e in epochs]

    if args.out:
        out = Path(args.out)
    else:
        base = Path(args.run_dir) if args.run_dir else ckpt_dir.parent
        out = base / "ckpt_avg" / f"avg_epoch_{epochs[0]}_to_{epochs[-1]}.pth"
    if CKPT_PATTERN.search(out.name) or "checkpoint_epoch_" in out.name:
        raise SystemExit(f"refusing output name {out.name}: test.py --eval_all would pick it up")
    out.parent.mkdir(parents=True, exist_ok=True)

    averaged = average_checkpoint_files(paths)
    torch.save(averaged, str(out))
    print(f"averaged epochs {epochs} ({len(epochs)} checkpoints) -> {out}")
    return out


if __name__ == "__main__":
    main()
