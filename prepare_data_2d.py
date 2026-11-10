#!/usr/bin/env python
"""Build a train2D.py-compatible 2D dataset from ./data (PAS testis histology).

What train2D.py expects (derived from train2D.py + snmi/_custom/seg_reg_dataset.py +
snmi/core/base_dataset.py + snmi/utils/utils.load_file):

  Layout            <data_path>/{train,valid,test,source_test}/
  Triplet per       <stem>org.tif    grayscale image, loaded by PIL -> [H, W] float32,
  sample            (cfg.img_suffix) then min_max + ExpandDim(0) -> [1, H, W]
                    <stem>lab_b.npy  label, np.load -> see --label-format below
  (cfg.lab_suffix)
                    <stem>weight.tif confidence map, PIL -> thresholded (>0.5) -> [1, H, W]
  (cfg.weight_suffix)
  Pairing           file names are built by str.replace(img_suffix -> lab_suffix /
                    weight_suffix), so the three files of a sample share one stem and
                    live in the same folder.
  Shape             H, W divisible by 2**(n_layers-1) = 16 and identical for every
                    sample (batches mix images, and the registration branch
                    concatenates a source and a target image).
  Splits            train/   source (labeled) + target (unlabeled) images. SegRegDataset
                             always loads *every* suffix for *both* source and target,
                             so unlabeled images need placeholder label/weight files.
                    valid/   labeled: drives loss/dice/hausdorff + best checkpoint.
                    test/    labeled: final segmentation/registration metrics.
                    source_test/ labeled atlases warped onto test/ images by the
                             registration test (paired by sorted order).
                    unlabeled_test/ patches from donors held out of training entirely.
                             No labels, so no dice; this is the only set that can show
                             whether the model behaves on tissue it has never seen.
                    annotated/ the whole annotated section, its mask and the split
                             geometry (regions.json), for pooled region-wise scoring
                             instead of patch-averaged scoring.

Only tile.jpg has annotations (tile_annotations.geojson), so this script
  * rasterizes every annotated polygon part (holes respected) into a binary mask,
  * cuts tile.jpg into patches and splits them into disjoint regions ->
    train source / valid / test / source_test (see --split-mode),
  * cuts the remaining .jpg images into patches -> train targets (unlabeled pool
    for the iterative pseudo-labeling), each with a zero placeholder label and a
    zero weight map.

WHAT THE SPLITS ACTUALLY ARE (one annotated image only):
  train/valid/test labeled patches all come from the SAME slide, the same tissue
  section and the same stain batch. Two ways to divide that one image:
    --split-mode blocks (default): a grid of blocks of about --block-size pixels,
      dealt to the splits in order of foreground density so that each split gets a
      comparable mix of dense and sparse tissue. Use this: with contiguous bands the
      test region of the testis tile had half the foreground density of the training
      region (0.33 against 0.56 per patch), which by itself explained most of the
      0.18 gap between validation dice (0.77) and test dice (0.59).
    --split-mode bands: the three contiguous vertical bands used for the first
      dataset, kept for reproducibility.
  Either way:
    + no patch is shared between splits and no patch crosses a region border
      (splits are pixel-disjoint),
    - the splits are NOT statistically independent: same donor, same section, and
      patches sitting on either side of a border show neighbouring tissue unless
      --block-margin is set.
  Therefore valid/ and test/ measure *within-slide* generalization (how well the
  model segments unseen regions of the annotated section), not generalization to
  new cases. The 181 unannotated images are used only as unlabeled train targets;
  nothing in valid/ or test/ comes from them.
  Patches inside one split overlap each other when --labeled-stride < --patch-size
  (default 128 < 256, i.e. 50% overlap), which is fine within a split but is the
  reason region borders are enforced between splits.
  dataset_report.json records the per-split foreground density, which is the number
  to check before trusting any valid/test comparison.
  source_test/ is a copy of a few train-band patches, by design: it supplies the
  labeled atlases that the registration test warps onto test/ images. They are
  model inputs, never evaluation ground truth, but they are train data.

Run `python prepare_data_2d.py --help` for the knobs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a snmi dependency, but stay usable
    cv2 = None

IMG_SUFFIX = "org.tif"
LAB_SUFFIX = "lab_b.npy"
WEIGHT_SUFFIX = "weight.tif"
SPLITS = ("train", "valid", "test", "source_test", "unlabeled_test")


# --------------------------------------------------------------------------- IO

def find_inputs(data_dir: Path):
    """Return (labeled_image, geojson, [unlabeled images])."""
    geojsons = sorted(data_dir.glob("*.geojson"))
    if len(geojsons) != 1:
        raise SystemExit(f"Expected exactly one .geojson in {data_dir}, found {len(geojsons)}")
    gj = geojsons[0]

    images = sorted(p for p in data_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    if not images:
        raise SystemExit(f"No images found in {data_dir}")

    # the annotated image is the one whose stem prefixes the geojson stem
    # (tile.jpg <- tile_annotations.geojson)
    candidates = [p for p in images if gj.stem.startswith(p.stem)]
    if not candidates:
        raise SystemExit(f"Could not match {gj.name} to an image in {data_dir}")
    labeled = max(candidates, key=lambda p: len(p.stem))
    unlabeled = [p for p in images if p != labeled]
    return labeled, gj, unlabeled


def load_gray(path: Path) -> np.ndarray:
    """Load an image as single-channel uint8 (the nets take in_channels=1)."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def donor_of(path: Path) -> str:
    """Donor/case id: everything before the first underscore ('1PAS_01' -> '1PAS')."""
    return Path(path).stem.split("_")[0]


# -------------------------------------------------------------- overlap check

def _box_sums(img: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """Exact local sums over every kh x kw window, via a float64 integral image."""
    ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1), np.float64)
    np.cumsum(np.cumsum(img, 0), 1, out=ii[1:, 1:])
    return ii[kh:, kw:] - ii[:-kh, kw:] - ii[kh:, :-kw] + ii[:-kh, :-kw]


def ncc_max(img: np.ndarray, tpl: np.ndarray, min_std: float = 3.0) -> float:
    """Max normalized cross correlation of `tpl` anywhere in `img`.

    Positions whose local standard deviation is below `min_std` (blank slide) are
    skipped: there the correlation is numerically meaningless.
    """
    from scipy.signal import fftconvolve

    kh, kw = tpl.shape
    if img.shape[0] < kh or img.shape[1] < kw:
        return 0.0
    t = tpl - tpl.mean()
    tn = float(np.sqrt((t * t).sum()))
    if tn == 0:
        return 0.0
    num = fftconvolve(img, t[::-1, ::-1], mode="valid")
    n = float(t.size)
    s1 = _box_sums(img, kh, kw)
    var = _box_sums(img * img, kh, kw) - s1 * s1 / n
    ok = np.sqrt(np.maximum(var, 0.0)) / np.sqrt(n) > min_std
    if not ok.any():
        return 0.0
    return float((num[ok] / (np.sqrt(np.maximum(var[ok], 1e-9)) * tn)).max())


def find_overlapping_images(labeled: Path, images, scale: int = 4, templ: int = 96,
                            threshold: float = 0.9):
    """Images that contain the pixels of the annotated image.

    The annotated tile of this dataset turned out to be a re-export of one of the case
    images (tile.jpg == 3PAS_02.jpg, offset x+120 y-8, NCC 0.98), which silently put
    validation and test tissue into the unlabeled training pool. This check catches that
    class of mistake instead of trusting file names.
    """
    def small(path):
        im = Image.open(path).convert("L")
        return np.asarray(im.resize((im.width // scale, im.height // scale),
                                    Image.BILINEAR), np.float64)

    tile = small(labeled)
    tpls = []
    for (r, c) in ((0.35, 0.35), (0.55, 0.50), (0.40, 0.60)):
        y, x = int(tile.shape[0] * r), int(tile.shape[1] * c)
        patch = tile[y:y + templ, x:x + templ]
        if patch.shape == (templ, templ):
            tpls.append(patch)
    scores = {}
    for i, path in enumerate(images, 1):
        scores[path.name] = max(ncc_max(small(path), t) for t in tpls)
        if i % 50 == 0:
            print(f"    overlap check {i}/{len(images)}", flush=True)
    matches = sorted([n for n, s in scores.items() if s >= threshold])
    return scores, matches


def ring_area(ring) -> float:
    """Shoelace area of a ring in pixel units."""
    r = np.asarray(ring, dtype=np.float64)
    if r.ndim != 2 or len(r) < 3:
        return 0.0
    return 0.5 * abs(np.dot(r[:, 0], np.roll(r[:, 1], 1)) - np.dot(r[:, 1], np.roll(r[:, 0], 1)))


def load_polygons(geojson_path: Path):
    """Annotated polygon parts as [exterior_ring, hole_ring...] in pixel coords.

    A GeoJSON *feature* is one annotation object, but a MultiPolygon feature carries
    several polygon parts, so the number of parts is larger than the number of
    features (QuPath also emits near-zero-area slivers as extra parts).
    """
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    features = gj["features"] if isinstance(gj, dict) else gj

    polygons, stats = [], {"features": len(features), "multipolygon_features": 0,
                           "unsupported_geometries": 0}
    for feat in features:
        geom = (feat or {}).get("geometry") or {}
        gtype, coords = geom.get("type"), geom.get("coordinates")
        if gtype == "Polygon":
            polygons.append(coords)
        elif gtype == "MultiPolygon":
            stats["multipolygon_features"] += 1
            polygons.extend(coords)
        else:
            stats["unsupported_geometries"] += 1
    stats["polygon_parts"] = len(polygons)
    return polygons, stats


def rasterize(polygons, height: int, width: int, min_area: float = 10.0):
    """Binary (0/255) mask: every polygon part is foreground, inner rings are holes.

    Parts whose exterior is smaller than `min_area` px^2 are dropped: the export
    contains degenerate slivers that would otherwise add single-pixel labels.
    Exteriors are drawn first and holes afterwards, so a hole never hides a
    neighbouring annotation drawn later.
    """
    kept, dropped = [], 0
    for rings in polygons:
        if not rings or ring_area(rings[0]) < min_area:
            dropped += 1
            continue
        kept.append(rings)

    exteriors = [np.asarray(p[0], dtype=np.float64) for p in kept]
    holes = [np.asarray(r, dtype=np.float64) for p in kept for r in p[1:]]
    stats = {"parts_used": len(kept), "parts_dropped_degenerate": dropped,
             "holes_punched": len(holes)}

    if cv2 is not None:
        mask = np.zeros((height, width), np.uint8)
        for ring in exteriors:
            cv2.fillPoly(mask, [np.round(ring).astype(np.int32)], 255)
        for ring in holes:
            cv2.fillPoly(mask, [np.round(ring).astype(np.int32)], 0)
        stats["rasterizer"] = "cv2.fillPoly"
        return mask, stats

    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    for ring in exteriors:
        draw.polygon([tuple(pt) for pt in ring], fill=255)
    for ring in holes:
        draw.polygon([tuple(pt) for pt in ring], fill=0)
    stats["rasterizer"] = "PIL.ImageDraw.polygon"
    return np.asarray(img, dtype=np.uint8), stats


def downscale(img: np.ndarray, factor: int, is_mask: bool) -> np.ndarray:
    if factor == 1:
        return img
    h, w = img.shape[:2]
    size = (max(1, w // factor), max(1, h // factor))  # PIL takes (w, h)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    out = np.asarray(Image.fromarray(img).resize(size, resample), dtype=np.uint8)
    return out


# ---------------------------------------------------------------------- patches

def positions(total: int, patch: int, stride: int, start: int = 0, stop: int | None = None):
    """Patch start offsets inside [start, stop), always including the last full patch."""
    stop = total if stop is None else min(stop, total)
    if stop - start < patch:
        return []
    pos = list(range(start, stop - patch + 1, stride))
    if pos[-1] != stop - patch:
        pos.append(stop - patch)
    return pos


def tissue_fraction(patch: np.ndarray, bright_cut: int) -> float:
    """Fraction of non-background (stained) pixels; slide background is bright."""
    return float(np.count_nonzero(patch < bright_cut)) / patch.size


def write_sample(out_dir: Path, stem: str, img: np.ndarray,
                 mask: np.ndarray | None, weight: np.ndarray | None,
                 label_format: str) -> str:
    """Write the org.tif / lab_b.npy / weight.tif triplet. Returns the image path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{stem}{IMG_SUFFIX}"
    Image.fromarray(img, mode="L").save(img_path)

    fg = np.zeros(img.shape, np.uint8) if mask is None else (mask > 127).astype(np.uint8)
    if label_format == "onehot":
        lab = np.zeros((2,) + img.shape, np.float32)
        lab[0] = 1.0 - fg
        lab[1] = fg
    else:  # binary255: 2D map of {0, 255}, matches cfg.pre's OneHot([0, 255])
        lab = (fg * 255).astype(np.uint8)
    np.save(out_dir / f"{stem}{LAB_SUFFIX}", lab)

    w = np.zeros(img.shape, np.uint8) if weight is None else (weight > 0).astype(np.uint8) * 255
    Image.fromarray(w, mode="L").save(out_dir / f"{stem}{WEIGHT_SUFFIX}")
    return str(img_path)


# ------------------------------------------------------------- split assignment

def assign_bands(mask, h, w, p, args):
    """Three contiguous vertical bands, borders placed on cumulative annotated area.

    Simple and obviously leak-free, but the bands end up with different tissue
    composition: in the first testis dataset the train/valid bands averaged 0.56
    foreground per patch against 0.33 for the (wider, sparser) test band, which alone
    accounted for a 0.18 dice gap between validation and test.
    """
    col_fg = (mask > 127).sum(axis=0).astype(np.float64)
    cum = np.cumsum(col_fg) / max(col_fg.sum(), 1.0)
    x_valid = int(np.searchsorted(cum, 1.0 - args.valid_frac - args.test_frac))
    x_test = int(np.searchsorted(cum, 1.0 - args.test_frac))
    x_valid = min(max(x_valid, p), w - 2 * p)
    x_test = min(max(x_test, x_valid + p), w - p)
    bounds = {"train": (0, x_valid), "valid": (x_valid, x_test), "test": (x_test, w)}
    for name, (x0, x1) in bounds.items():
        if x1 - x0 < p:
            raise SystemExit(
                f"Band '{name}' is {x1 - x0}px wide, smaller than patch {p}. "
                f"Lower --patch-size/--downscale or change --valid-frac/--test-frac.")
    regions = {name: [(0, h, x0, x1)] for name, (x0, x1) in bounds.items()}
    meta = {"split_mode": "bands",
            "bands_x": {k: list(v) for k, v in bounds.items()},
            "band_borders_from": "cumulative annotated area along x"}
    return regions, meta


def grid_edges(total, target):
    """Edges of an even grid over `total` with cells of about `target` pixels."""
    n = max(1, int(round(total / float(target))))
    return [int(round(i * total / n)) for i in range(n + 1)]


def assign_blocks(mask, h, w, p, args):
    """Disjoint blocks dealt to train/valid/test, stratified by foreground density.

    The image is cut into a grid of blocks of about --block-size pixels. Blocks are
    sorted by their foreground density and dealt out so that each split receives its
    target share of blocks while walking the density ordering, which gives every split a
    comparable mix of dense and sparse tissue. Patches are taken strictly inside a
    block, so splits never share a pixel; with --block-margin > 0 they are separated by
    a gap as well, at the cost of fewer patches.
    """
    block = args.block_size if args.block_size > 0 else 2 * p
    margin = args.block_margin
    need = p + 2 * margin
    if block < need:
        raise SystemExit(f"--block-size {block} must be at least patch + 2*margin = {need}.")

    y_edges, x_edges = grid_edges(h, block), grid_edges(w, block)
    blocks = []
    for yi in range(len(y_edges) - 1):
        for xi in range(len(x_edges) - 1):
            y0, y1 = y_edges[yi], y_edges[yi + 1]
            x0, x1 = x_edges[xi], x_edges[xi + 1]
            if (y1 - y0) < need or (x1 - x0) < need:
                continue                       # too small to hold a patch
            sub = mask[y0:y1, x0:x1]
            blocks.append({"box": (y0 + margin, y1 - margin, x0 + margin, x1 - margin),
                           "grid": (yi, xi),
                           "density": float(np.count_nonzero(sub > 127)) / sub.size})
    if len(blocks) < 3:
        raise SystemExit(f"Only {len(blocks)} usable blocks of {block}px. "
                         f"Lower --block-size or --patch-size.")

    shares = {"train": 1.0 - args.valid_frac - args.test_frac,
              "valid": args.valid_frac, "test": args.test_frac}
    regions = {k: [] for k in shares}
    counts = {k: 0 for k in shares}
    detail = []
    # Walk blocks from dense to sparse, each time giving the block to whichever split is
    # furthest below its target share. Equal shares of the density ordering follow.
    for i, blk in enumerate(sorted(blocks, key=lambda b: -b["density"]), start=1):
        pick = max(shares, key=lambda k: shares[k] * i - counts[k])
        regions[pick].append(blk["box"])
        counts[pick] += 1
        detail.append({"grid": list(blk["grid"]), "split": pick,
                       "density": round(blk["density"], 4)})

    meta = {"split_mode": "blocks",
            "block_size": block,
            "block_margin": margin,
            "block_grid": [len(y_edges) - 1, len(x_edges) - 1],
            "blocks_usable": len(blocks),
            "blocks_per_split": counts,
            "blocks": detail,
            "block_allocation": "density-sorted, proportional to the split shares"}
    return regions, meta


# ------------------------------------------------------------------------ build

def resolve_inputs(args, report: dict):
    """Find the annotated image and the usable unlabeled images, with exclusions.

    Excluded from the unlabeled pool: any image that overlaps the annotated image
    (verified by cross correlation, not by name) and, unless --keep-donor-images,
    every other image of that donor. Those pixels are already in train/valid/test.
    """
    labeled_img, geojson, unlabeled = find_inputs(Path(args.data_dir))
    info = {"labeled_image": labeled_img.name, "annotations": geojson.name,
            "candidates": len(unlabeled)}

    manual = [c.strip() for c in (args.exclude_cases or "").split(",") if c.strip()]
    excluded, reasons = set(), {}
    for name in manual:
        for p in unlabeled:
            if p.name == name or donor_of(p) == name or p.stem == name:
                excluded.add(p.name)
                reasons[p.name] = f"--exclude-cases {name}"

    if args.check_overlap:
        print(f"  checking {len(unlabeled)} images for pixel overlap with "
              f"{labeled_img.name} (--no-check-overlap to skip)", flush=True)
        scores, matches = find_overlapping_images(labeled_img, unlabeled,
                                                  threshold=args.overlap_threshold)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
        info["overlap_ncc_top5"] = {k: round(v, 4) for k, v in top}
        info["overlap_ncc_median"] = round(float(np.median(list(scores.values()))), 4)
        info["overlap_matches"] = matches
        print("    top correlations: "
              + ", ".join(f"{k} {v:.3f}" for k, v in top))
        for name in matches:
            excluded.add(name)
            reasons[name] = f"overlaps {labeled_img.name} (NCC {scores[name]:.3f})"
        if matches and not args.keep_donor_images:
            donors = {donor_of(Path(m)) for m in matches}
            for p in unlabeled:
                if donor_of(p) in donors and p.name not in excluded:
                    excluded.add(p.name)
                    reasons[p.name] = f"same donor as {sorted(donors)}"
        if matches:
            print(f"    EXCLUDED {len(excluded)} images: {sorted(donors) if not args.keep_donor_images else matches}")
        else:
            print("    no overlap found")

    kept = [p for p in unlabeled if p.name not in excluded]
    info["excluded"] = {k: reasons[k] for k in sorted(excluded)}
    info["excluded_count"] = len(excluded)
    info["usable"] = len(kept)
    report["inputs"] = info
    return labeled_img, geojson, kept


def build_labeled(args, out_root: Path, report: dict, labeled_img: Path, geojson: Path):
    """Patch the annotated image into disjoint train / valid / test regions."""
    gray_full = load_gray(labeled_img)
    polygons, geo_stats = load_polygons(geojson)
    mask_full, raster_stats = rasterize(polygons, *gray_full.shape, min_area=args.min_polygon_area)

    gray = downscale(gray_full, args.downscale, is_mask=False)
    mask = downscale(mask_full, args.downscale, is_mask=True)
    h, w = gray.shape
    p, stride = args.patch_size, args.labeled_stride

    report["labeled_image"] = {
        "file": labeled_img.name,
        "annotations": geojson.name,
        **geo_stats,
        **raster_stats,
        "size_full": [int(gray_full.shape[1]), int(gray_full.shape[0])],
        "size_used": [int(w), int(h)],
        "foreground_fraction": round(float(np.count_nonzero(mask > 127)) / mask.size, 4),
    }

    if args.split_mode == "bands":
        regions, split_meta = assign_bands(mask, h, w, p, args)
    else:
        regions, split_meta = assign_blocks(mask, h, w, p, args)

    # Training patches need enough foreground to be worth a gradient step; evaluation
    # patches do not. Sparse, stroma-heavy regions are exactly where this model fails,
    # so valid/test keep everything above a much lower floor (0 foreground is still
    # excluded because per-patch dice and hausdorff are undefined there).
    min_fg = {"train": args.min_fg, "valid": args.min_fg_eval, "test": args.min_fg_eval}

    written = {k: [] for k in SPLITS}
    dropped = 0
    for name, boxes in regions.items():
        for (y0, y1, x0, x1) in boxes:
            for y in positions(h, p, stride, start=y0, stop=y1):
                for x in positions(w, p, stride, start=x0, stop=x1):
                    m = mask[y:y + p, x:x + p]
                    fg_frac = float(np.count_nonzero(m > 127)) / m.size
                    if fg_frac < min_fg[name]:
                        dropped += 1
                        continue
                    g = gray[y:y + p, x:x + p]
                    stem = f"src_{labeled_img.stem}_x{x:05d}_y{y:05d}_"
                    written[name].append(write_sample(
                        out_root / name, stem, g, m,
                        np.full(g.shape, 255, np.uint8),  # GT -> fully trusted
                        args.label_format))

    for name in ("train", "valid", "test"):
        if not written[name]:
            raise SystemExit(
                f"Split '{name}' got no patch. Lower --min-fg, lower --block-size/--patch-size, "
                f"or raise --{name if name != 'train' else 'valid'}-frac.")

    # patch-level foreground density per split: this is the statistic that exposed the
    # train/test mismatch of the band split (0.56 vs 0.33 in the first dataset).
    density = {}
    for name in ("train", "valid", "test"):
        fgs = [float(np.load(f.replace(IMG_SUFFIX, LAB_SUFFIX))[1].mean())
               if args.label_format == "onehot" else
               float((np.load(f.replace(IMG_SUFFIX, LAB_SUFFIX)) > 127).mean())
               for f in written[name]]
        density[name] = {"patches": len(fgs),
                         "fg_mean": round(float(np.mean(fgs)), 4),
                         "fg_median": round(float(np.median(fgs)), 4)}

    # atlases for the registration test: a few patches drawn from the train band
    pool = sorted(written["train"])
    rng = np.random.RandomState(args.seed)
    take = min(args.n_source_test, len(pool))
    picks = rng.choice(len(pool), size=take, replace=False) if take else []
    copied_from = []
    for i in sorted(picks):
        img_path = Path(pool[int(i)])
        for suffix in (IMG_SUFFIX, LAB_SUFFIX, WEIGHT_SUFFIX):
            src = Path(str(img_path).replace(IMG_SUFFIX, suffix))
            shutil.copy2(src, out_root / "source_test" / src.name)
        written["source_test"].append(str(out_root / "source_test" / img_path.name))
        copied_from.append(img_path.name)

    # Whole-section artifacts. The labeled test area cannot grow beyond this one
    # section, so the way to get a less noisy test number is to evaluate over the
    # complete test region instead of 25 sampled patches. These three files carry
    # everything needed for that: the section, its mask, and the split geometry.
    ann = out_root / "annotated"
    ann.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray, mode="L").save(ann / "section.tif")
    Image.fromarray((mask > 127).astype(np.uint8) * 255, mode="L").save(ann / "section_mask.png")
    (ann / "regions.json").write_text(json.dumps({
        "image": "section.tif",
        "mask": "section_mask.png",
        "source": labeled_img.name,
        "size": [int(w), int(h)],
        "downscale": args.downscale,
        "patch_size": p,
        "regions": {k: [[int(v) for v in box] for box in boxes]
                    for k, boxes in regions.items()},
        **split_meta,
    }, indent=1), encoding="utf-8")

    report["labeled_patches"] = {
        "source_image": labeled_img.name,
        **split_meta,
        "patch_size": args.patch_size,
        "stride": stride,
        "patches_overlap_within_a_split": stride < p,
        "patches_cross_a_split_border": False,
        "splits_pixel_disjoint": True,
        "splits_independent": False,  # one slide: same donor, same section
        "foreground_density_per_split": density,
        "source_test_copied_from_train": copied_from,
        "min_fg_train": args.min_fg,
        "min_fg_eval": args.min_fg_eval,
        "dropped_low_foreground": dropped,
        **{k: len(v) for k, v in written.items()},
    }
    return written


def build_unlabeled(args, out_root: Path, report: dict, unlabeled):
    """Patch the unannotated images into the train pool and a held-out donor set.

    `--unlabeled-test-frac` of the *donors* go to unlabeled_test/ and are touched by
    nothing in training: not as pseudo-label targets, not as registration pairs. They
    have no labels, so they cannot give a dice, but they are the only data that can
    answer "does this model behave sanely on tissue it has never seen", through
    label-free checks (foreground fraction, flip consistency, confidence).
    """
    if args.limit_images:
        unlabeled = unlabeled[:args.limit_images]

    donors = sorted({donor_of(p) for p in unlabeled})
    n_hold = int(round(len(donors) * args.unlabeled_test_frac))
    held = []
    if n_hold > 0 and n_hold < len(donors):
        rng_d = np.random.RandomState(args.seed + 2)
        held = sorted(rng_d.choice(donors, size=n_hold, replace=False).tolist())

    p, stride = args.patch_size, args.unlabeled_stride
    rng = np.random.RandomState(args.seed + 1)
    written = {"train": [], "unlabeled_test": []}
    per_image, skipped_images = {}, []

    for idx, path in enumerate(unlabeled):
        gray = downscale(load_gray(path), args.downscale, is_mask=False)
        h, w = gray.shape
        cands = []
        for y in positions(h, p, stride):
            for x in positions(w, p, stride):
                g = gray[y:y + p, x:x + p]
                if tissue_fraction(g, args.bright_cut) < args.min_tissue:
                    continue
                if g.std() < args.min_std:  # flat / blank patch
                    continue
                cands.append((y, x))
        if not cands:
            skipped_images.append(path.name)
            continue
        if args.max_per_image and len(cands) > args.max_per_image:
            sel = rng.choice(len(cands), size=args.max_per_image, replace=False)
            cands = [cands[int(i)] for i in sorted(sel)]

        split = "unlabeled_test" if donor_of(path) in held else "train"
        safe_stem = path.stem.replace(".", "-")
        for (y, x) in cands:
            stem = f"unl_{safe_stem}_x{x:05d}_y{y:05d}_"
            written[split].append(write_sample(
                out_root / split, stem, gray[y:y + p, x:x + p],
                None,   # placeholder label (all background)
                None,   # zero weight -> contributes nothing if ever used in a loss
                args.label_format))
        per_image[path.name] = len(cands)

    report["unlabeled"] = {
        "images": len(unlabeled),
        "donors": len(donors),
        "donors_heldout": held,
        "images_heldout": sum(1 for p in unlabeled if donor_of(p) in held),
        "images_without_usable_patch": skipped_images,
        "patches_train": len(written["train"]),
        "patches_unlabeled_test": len(written["unlabeled_test"]),
        "patches_per_image": per_image,
    }
    return written


# ------------------------------------------------------------------- bookkeping

def write_manifests(out_root: Path, labeled, unlabeled, args, report):
    source = sorted(labeled["train"])
    target_train = sorted(unlabeled["train"])
    unlabeled_test = sorted(unlabeled["unlabeled_test"])
    splits = {
        "data_path": str(out_root),
        "img_suffix": IMG_SUFFIX,
        "lab_suffix": LAB_SUFFIX,
        "weight_suffix": WEIGHT_SUFFIX,
        "label_format": args.label_format,
        "patch_size": [args.patch_size, args.patch_size],
        "source_prefix": "src_",
        "target_prefix": "unl_",
        "source": [os.path.relpath(f, out_root) for f in source],
        "target_train": [os.path.relpath(f, out_root) for f in target_train],
        "valid": [os.path.relpath(f, out_root) for f in sorted(labeled["valid"])],
        "test": [os.path.relpath(f, out_root) for f in sorted(labeled["test"])],
        "source_test": [os.path.relpath(f, out_root) for f in sorted(labeled["source_test"])],
        "unlabeled_test": [os.path.relpath(f, out_root) for f in unlabeled_test],
        "unlabeled_test_donors": report["unlabeled"]["donors_heldout"],
        "excluded_images": report["inputs"]["excluded"],
    }
    (out_root / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    (out_root / "dataset_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    if args.label_format == "onehot":
        lab_pre = ("# lab_b.npy is already one-hot [2,H,W] float32 -> no OneHot here\n"
                   "        cfg.lab_suffix: [],")
    else:
        lab_pre = "cfg.lab_suffix: [P.OneHot([0, 255])],"
    lp = report["labeled_patches"]
    dens = lp["foreground_density_per_split"]
    if lp["split_mode"] == "bands":
        geometry = ("three contiguous vertical bands, x = "
                    + ", ".join(f"{k} [{v[0]}, {v[1]})" for k, v in lp["bands_x"].items()))
    else:
        geometry = (f"a {lp['block_grid'][0]}x{lp['block_grid'][1]} grid of "
                    f"{lp['block_size']}px blocks ({lp['blocks_usable']} usable, "
                    + ", ".join(f"{k}: {v}" for k, v in lp["blocks_per_split"].items())
                    + f"), dealt out density-first, margin {lp['block_margin']}px")
    density_rows = "\n".join(
        f"| {k} | {dens[k]['patches']} | {dens[k]['fg_mean']:.3f} | {dens[k]['fg_median']:.3f} |"
        for k in ("train", "valid", "test"))
    readme = f"""# Prepared 2D dataset ({out_root.name})

Generated by `prepare_data_2d.py` from `{args.data_dir}`.

## Layout
```
{out_root.name}/
  train/           {len(source)} labeled source patches (src_*) + {len(target_train)} unlabeled target patches (unl_*)
  valid/           {len(splits['valid'])} labeled patches
  test/            {len(splits['test'])} labeled patches
  source_test/     {len(splits['source_test'])} labeled atlas patches for the registration test
  unlabeled_test/  {len(unlabeled_test)} patches from {len(report['unlabeled']['donors_heldout'])} held-out donors, no labels, never trained on
  annotated/       section.tif + section_mask.png + regions.json (whole-section evaluation)
  splits.json      explicit file lists, held-out donors, excluded images
```
Every sample is a triplet sharing one stem:
`<stem>org.tif` (uint8 grayscale {args.patch_size}x{args.patch_size}),
`<stem>lab_b.npy`, `<stem>weight.tif`.

## Annotations

* `{report['labeled_image']['annotations']}`: {report['labeled_image']['features']} features, {report['labeled_image']['multipolygon_features']} of them MultiPolygon.
* Expanding the MultiPolygons yields {report['labeled_image']['polygon_parts']} polygon parts.
* Parts with an exterior below {args.min_polygon_area:g}px^2 are degenerate export slivers: {report['labeled_image']['parts_dropped_degenerate']} dropped, {report['labeled_image']['parts_used']} rasterized.
* {report['labeled_image']['holes_punched']} inner rings are punched back out as holes.
* Every part is the same single foreground class; foreground covers {report['labeled_image']['foreground_fraction'] * 100:.1f}% of the image.

Labels: `--label-format {args.label_format}`.
* `onehot`   -> float32 `[2, H, W]`, channel 0 background / 1 foreground. Same layout as
  the soft pseudo-labels written by `QualityAssessment` and the only layout the
  assessment pipeline (`ass_pre`, which does not one-hot) can consume.
* `binary255` -> uint8 `[H, W]` with values {{0, 255}}, matching the published
  `cfg.pre` entry `P.OneHot([0, 255])`.

Unlabeled patches carry an all-background label and an all-zero weight map purely
because `SegRegDataset` loads every suffix for source *and* target images. They must
never be used as segmentation supervision - keep them out of `source`.

## What these splits really are

`train2D.py` needs a labeled `valid/` (it drives the loss, dice, hausdorff and the
best-checkpoint choice) and a labeled `test/`. With a single annotated image, the only
way to fill them is to divide that image, so all labeled patches come from the same
slide, section and stain batch: `{report['labeled_image']['file']}`, divided into
{geometry} (details in `dataset_report.json`).

Foreground density per split, the statistic that tells you whether the three splits are
comparable (a large gap here shows up directly as a validation/test dice gap):

| split | patches | mean fg | median fg |
|---|---|---|---|
{density_rows}

* Pixel-disjoint: no patch is shared between splits and no patch crosses a region border, so no patch is partly trained on and partly evaluated on.
* Not independent: same donor, same section, and patches on opposite sides of a border show neighbouring tissue unless `--block-margin` is set. `valid/` and `test/` therefore measure **within-slide** generalization (unseen regions of the annotated section), not generalization to new cases. Report them as such.
* Patches overlap by {args.patch_size - args.labeled_stride}px *inside* a split (`--labeled-stride {args.labeled_stride}` < `--patch-size {args.patch_size}`); a redundancy, not a leak.
* `source_test/` is a copy of {len(splits['source_test'])} train-band patches, on purpose: those are the labeled atlases the registration test warps onto `test/` images. They are model inputs, never evaluation ground truth, but they are train data - do not call them held out.
* The {len(target_train)} patches from the {report['unlabeled']['images']} unannotated images are train targets only; nothing in `valid/` or `test/` comes from them.

## What each evaluation set can answer

| set | labels | question it answers |
|---|---|---|
| `valid/` | yes | when to stop, which checkpoint to keep |
| `test/` | yes | accuracy on unseen regions of the one annotated section |
| `annotated/` | yes | the same, pooled over the whole test region instead of sampled patches, and broken down per block so failing regions are identifiable |
| `unlabeled_test/` | no | does the model behave sanely on {len(report['unlabeled']['donors_heldout'])} donors it has never seen, measured label-free (foreground fraction, flip consistency, confidence) |

`test/` is small by construction: one section, {len(splits['test'])} patches, so its
patch-averaged dice has a standard error of roughly 0.04. Treat differences below ~0.08
as noise and prefer the pooled number over the patch average.

`unlabeled_test/` donors are excluded from `train/` entirely, so they are the only
honest answer to "will this generalize to a new case". It cannot produce a dice; use it
to detect collapse, drift of the predicted foreground fraction, and instability.

## Exclusions

`{report['inputs']['excluded_count']}` image(s) were dropped from the unlabeled pool:
{chr(10).join(f"* `{k}`: {v}" for k, v in list(report['inputs']['excluded'].items())[:8]) or '* none'}

This matters: the annotated tile of this dataset is a re-export of one of the case
images, so leaving that donor in the unlabeled pool puts validation and test tissue
into the pseudo-label training set. The check is correlation-based, not name-based.

If a case-level labeled test set is needed later, annotate one or more of the other
images and move their patches into `test/`; supporting several image+geojson pairs
would mean extending `find_inputs`, which currently expects exactly one geojson.

## train2D.py settings for this dataset
```python
cfg.data_path = r'{out_root}'
cfg.img_suffix = '{IMG_SUFFIX}'
cfg.lab_suffix = '{LAB_SUFFIX}'
cfg.weight_suffix = '{WEIGHT_SUFFIX}'
cfg.pre = {{cfg.img_suffix: [P.min_max, P.ExpandDim(0)],
        {lab_pre}
        cfg.weight_suffix: [lambda img: np.array(img > 0.5, np.float32), P.ExpandDim(0)]}}
cfg.num_source = {len(source)}          # all labeled patches
cfg.min_good = {max(1, len(target_train) // 10)}            # ~10% of the {len(target_train)} unlabeled targets per iteration
```
`train2D.py` picks the labeled set with `source = train[:cfg.num_source]` after
shuffling `train`, which would mix unlabeled patches into the supervised set.
Select by prefix instead (or read `splits.json`):
```python
train = glob.glob(cfg.data_path + '/train/*' + cfg.img_suffix)
source = sorted(f for f in train if os.path.basename(f).startswith('src_'))
target_train = sorted(f for f in train if os.path.basename(f).startswith('unl_'))
```
"""
    (out_root / "README_dataset.md").write_text(readme, encoding="utf-8")
    return splits


def verify_numpy(out_root: Path, args):
    """Replicate cfg.pre / ass_pre with numpy only (no torch/cv2) and print shapes."""
    import glob as _glob

    def load(path):
        p = str(path)
        if p.endswith(".npy"):
            return np.load(p).astype(np.float32)
        return np.asarray(Image.open(p), dtype=np.float32)

    def one_hot(x, classes=(0, 255)):
        out = np.zeros((len(classes),) + x.shape, np.float32)
        for i, c in enumerate(classes):
            out[i][x == c] = 1
        return out

    train = sorted(_glob.glob(str(out_root / "train" / f"*{IMG_SUFFIX}")))
    valid = sorted(_glob.glob(str(out_root / "valid" / f"*{IMG_SUFFIX}")))
    sizes = set()
    print("[verify] numpy replica of cfg.pre:")
    for tag, path in (("source(train,src_)", next(f for f in train if "src_" in os.path.basename(f))),
                      ("target(train,unl_)", next(f for f in train if "unl_" in os.path.basename(f))),
                      ("valid", valid[0])):
        img = load(path)
        img = np.expand_dims((img - img.min()) / (img.max() - img.min()), 0)
        lab = load(str(path).replace(IMG_SUFFIX, LAB_SUFFIX))
        if args.label_format == "binary255":
            lab = one_hot(lab)
        wgt = np.expand_dims(np.array(load(str(path).replace(IMG_SUFFIX, WEIGHT_SUFFIX)) > 0.5,
                                      np.float32), 0)
        sizes.add(img.shape)
        print(f"           {tag:20s} img={img.shape} lab={lab.shape} "
              f"weight={wgt.shape} fg={lab[1].mean():.3f} w={wgt.mean():.3f}")
    assert len(sizes) == 1, f"inconsistent image shapes: {sizes}"
    for d in list(sizes)[0][1:]:
        assert d % 16 == 0, f"spatial size {d} not divisible by 16"
    missing = [f for split in SPLITS
               for f in _glob.glob(str(out_root / split / f"*{IMG_SUFFIX}"))
               for s in (LAB_SUFFIX, WEIGHT_SUFFIX)
               if not os.path.isfile(str(f).replace(IMG_SUFFIX, s))]
    assert not missing, f"{len(missing)} images without a label/weight file"
    print(f"           all triplets complete, one shape {list(sizes)[0]}, 16-divisible: OK")


def verify(out_root: Path, args):
    """Load one sample through the real snmi pipeline and print the tensor shapes."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    verify_numpy(out_root, args)
    try:
        import glob as _glob
        from snmi._custom import SegRegDataset
        from snmi.utils import process_methods as P
    except Exception as exc:  # torch / cv2 / monai not installed
        print(f"[verify] snmi check skipped (cannot import snmi: {exc})")
        return

    pre = {IMG_SUFFIX: [P.min_max, P.ExpandDim(0)],
           LAB_SUFFIX: [] if args.label_format == "onehot" else [P.OneHot([0, 255])],
           WEIGHT_SUFFIX: [lambda img: np.array(img > 0.5, np.float32), P.ExpandDim(0)]}

    train = sorted(_glob.glob(str(out_root / "train" / f"*{IMG_SUFFIX}")))
    source = [f for f in train if os.path.basename(f).startswith("src_")]
    valid = sorted(_glob.glob(str(out_root / "valid" / f"*{IMG_SUFFIX}")))

    ds = SegRegDataset(source, valid, "source", "target",
                       [IMG_SUFFIX, LAB_SUFFIX, WEIGHT_SUFFIX], pre)
    sample = ds[0]
    print("[verify] SegRegDataset sample:")
    for k in sorted(sample):
        v = np.asarray(sample[k])
        print(f"           {k:24s} {str(v.shape):18s} {v.dtype} "
              f"min={v.min():.3f} max={v.max():.3f}")

    ass_pre = {IMG_SUFFIX: [P.min_max, P.ExpandDim(0), P.ToTensor(0)],
               LAB_SUFFIX: [P.ToTensor(0)]}
    targets = [f for f in train if os.path.basename(f).startswith("unl_")]
    ass = SegRegDataset(source, targets, "source", "target",
                        [IMG_SUFFIX, LAB_SUFFIX], ass_pre,
                        is_assessment=True, num_assessment=2)
    item = ass[0]
    shapes = {k: tuple(v.shape) for k, v in item["data_dict"].items()}
    shapes.update({f"assessment[0].{k}": tuple(v.shape)
                   for k, v in item["assessment_dicts"][0].items()})
    print("[verify] assessment sample:")
    for k in sorted(shapes):
        print(f"           {k:24s} {shapes[k]}")


# ------------------------------------------------------------------------- main

def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(here / "data"), help="raw .jpg + .geojson folder")
    ap.add_argument("--out-dir", default=str(here / "data_2d"), help="dataset to create")
    ap.add_argument("--patch-size", type=int, default=256,
                    help="patch side; must be a multiple of 16 for 5-layer nets")
    ap.add_argument("--downscale", type=int, default=1, help="integer downsampling before patching")
    ap.add_argument("--labeled-stride", type=int, default=128)
    ap.add_argument("--unlabeled-stride", type=int, default=256)
    ap.add_argument("--min-polygon-area", type=float, default=10.0,
                    help="drop annotation parts smaller than this (px^2, full resolution)")
    ap.add_argument("--min-fg", type=float, default=0.02,
                    help="min foreground fraction to keep a labeled TRAIN patch")
    ap.add_argument("--min-fg-eval", type=float, default=0.005,
                    help="min foreground fraction for valid/test patches; lower than "
                         "--min-fg on purpose, so sparse regions stay in the evaluation")
    ap.add_argument("--min-tissue", type=float, default=0.4,
                    help="min stained-pixel fraction to keep an unlabeled patch")
    ap.add_argument("--bright-cut", type=int, default=220, help="gray level above which a pixel is slide background")
    ap.add_argument("--min-std", type=float, default=5.0, help="reject near-flat unlabeled patches")
    ap.add_argument("--max-per-image", type=int, default=12,
                    help="cap on unlabeled patches per image (0 = keep all)")
    ap.add_argument("--limit-images", type=int, default=0, help="debug: use only the first N unlabeled images")
    ap.add_argument("--split-mode", choices=("blocks", "bands"), default="blocks",
                    help="blocks: density-stratified grid blocks (representative test set); "
                         "bands: three contiguous vertical bands (the original behaviour)")
    ap.add_argument("--block-size", type=int, default=0,
                    help="block side for --split-mode blocks (0 = 2 * patch size)")
    ap.add_argument("--block-margin", type=int, default=0,
                    help="pixels discarded inside each block border, to put a gap between splits")
    ap.add_argument("--valid-frac", type=float, default=0.2,
                    help="share of the annotated image (area for bands, blocks for blocks) for valid/")
    ap.add_argument("--test-frac", type=float, default=0.2, help="same, for test/")
    ap.add_argument("--n-source-test", type=int, default=5, help="labeled atlases copied to source_test/")
    ap.add_argument("--unlabeled-test-frac", type=float, default=0.2,
                    help="share of DONORS held out into unlabeled_test/ (no labels, never trained on)")
    ap.add_argument("--check-overlap", dest="check_overlap", action="store_true", default=True,
                    help="correlate every case image against the annotated one and exclude "
                         "any that contains the same pixels (default: on, ~1-2 min)")
    ap.add_argument("--no-check-overlap", dest="check_overlap", action="store_false")
    ap.add_argument("--overlap-threshold", type=float, default=0.9,
                    help="NCC above which an image counts as containing the annotated pixels")
    ap.add_argument("--keep-donor-images", action="store_true",
                    help="exclude only the overlapping image, not the rest of its donor")
    ap.add_argument("--exclude-cases", default="",
                    help="comma-separated donors or file names to drop from the unlabeled pool")
    ap.add_argument("--label-format", choices=("onehot", "binary255"), default="onehot")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overwrite", action="store_true", help="delete out-dir first")
    ap.add_argument("--verify", action="store_true", help="load a sample through snmi and print shapes")
    args = ap.parse_args(argv)

    if args.patch_size % 16:
        raise SystemExit(f"--patch-size {args.patch_size} is not a multiple of 16 "
                         f"(Unet2D/MrReg downsample 4x by 2).")

    out_root = Path(args.out_dir)
    if out_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_root} already exists; pass --overwrite to rebuild.")
        shutil.rmtree(out_root)
    for name in SPLITS:
        (out_root / name).mkdir(parents=True, exist_ok=True)

    report = {"args": vars(args)}
    print(f"Reading   {args.data_dir}")
    labeled_img, geojson, usable = resolve_inputs(args, report)
    labeled = build_labeled(args, out_root, report, labeled_img, geojson)
    li = report["labeled_image"]
    print(f"  labeled {li['file']} + {li['annotations']}: {li['features']} annotation features "
          f"({li['multipolygon_features']} MultiPolygon) -> {li['polygon_parts']} polygon parts "
          f"-> {li['parts_used']} rasterized "
          f"({li['parts_dropped_degenerate']} degenerate parts dropped, "
          f"{li['holes_punched']} holes punched), "
          f"{li['foreground_fraction'] * 100:.1f}% foreground")
    lp = report["labeled_patches"]
    print(f"  split   mode={lp['split_mode']}"
          + (f" grid={lp['block_grid'][0]}x{lp['block_grid'][1]} block={lp['block_size']}px "
             f"margin={lp['block_margin']}px blocks={lp['blocks_per_split']}"
             if lp["split_mode"] == "blocks" else f" bands_x={lp['bands_x']}"))
    print(f"  patches train={len(labeled['train'])} valid={len(labeled['valid'])} "
          f"test={len(labeled['test'])} source_test={len(labeled['source_test'])} "
          f"(dropped {lp['dropped_low_foreground']}: min_fg train {args.min_fg:.1%} / "
          f"eval {args.min_fg_eval:.1%})")
    for name, d in lp["foreground_density_per_split"].items():
        print(f"    {name:6s} n={d['patches']:4d}  foreground mean={d['fg_mean']:.3f} "
              f"median={d['fg_median']:.3f}")

    unlabeled = build_unlabeled(args, out_root, report, usable)
    ul = report["unlabeled"]
    print(f"  unlabeled {ul['images']} images / {ul['donors']} donors -> "
          f"{ul['patches_train']} train patches")
    print(f"    held out {len(ul['donors_heldout'])} donors "
          f"({ul['images_heldout']} images, {ul['patches_unlabeled_test']} patches) "
          f"-> unlabeled_test/: {', '.join(ul['donors_heldout']) or 'none'}")

    splits = write_manifests(out_root, labeled, unlabeled, args, report)
    print(f"Wrote     {out_root}")
    print(f"  train/ {len(splits['source'])} src_ + {len(splits['target_train'])} unl_ images")
    print(f"  valid/ {len(splits['valid'])}   test/ {len(splits['test'])}   "
          f"source_test/ {len(splits['source_test'])}   "
          f"unlabeled_test/ {len(splits['unlabeled_test'])}")
    print(f"  annotated/section.tif + section_mask.png + regions.json")
    print(f"  splits.json, dataset_report.json, README_dataset.md")

    if args.verify:
        verify(out_root, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
