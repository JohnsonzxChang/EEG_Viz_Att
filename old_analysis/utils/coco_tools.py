#utf-8

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pycocotools.coco import COCO


def _best_caption(captions: List[str], mode: str = "words") -> str:

    if not captions:
        return ""
    if mode == "chars":
        return max(captions, key=len)
    # default: words
    return max(captions, key=lambda s: len(s.split()))


def coco_pair_from_img_id(
    img_id: int,
    coco_root: str | Path,
    split: str = "val2017",
    captions_ann: str | Path = "annotations/captions_val2017.json",
    instances_ann: str | Path | None = None,  # or annotations/instances_val2017.json
    best_mode: str = "words",
) -> Dict[str, Any]:
    
    coco_root = Path(coco_root)
    cap_path = (coco_root / captions_ann) if not Path(captions_ann).is_absolute() else Path(captions_ann)
    coco_caps = COCO(str(cap_path))

    imgs = coco_caps.loadImgs([img_id])
    if not imgs:
        raise ValueError(f"img_id={img_id} not in {cap_path}")

    img = imgs[0]
    file_name = img.get("file_name", "")
    img_path = coco_root / split / file_name if file_name else None

    # captions
    ann_ids = coco_caps.getAnnIds(imgIds=[img_id])
    anns = coco_caps.loadAnns(ann_ids)
    captions_all = [a["caption"] for a in anns if "caption" in a]
    caption_best = _best_caption(captions_all, mode=best_mode)

    out: Dict[str, Any] = {
        "img_id": img_id,
        "image": {
            "file_name": file_name,
            "local_path": str(img_path) if img_path else None,
            "coco_url": img.get("coco_url"),
            "flickr_url": img.get("flickr_url"),
            "width": img.get("width"),
            "height": img.get("height"),
        },
        "captions_all": captions_all,
        "caption_best": caption_best,
    }

    # Optional: add more detailed object-level information (instances)
    if instances_ann is not None:
        inst_path = (coco_root / instances_ann) if not Path(instances_ann).is_absolute() else Path(instances_ann)
        coco_inst = COCO(str(inst_path))

        inst_ann_ids = coco_inst.getAnnIds(imgIds=[img_id])
        inst_anns = coco_inst.loadAnns(inst_ann_ids)

        # Count category occurrences
        cat_count: Dict[str, int] = {}
        for a in inst_anns:
            cat_id = a.get("category_id")
            if cat_id is None:
                continue
            cat_name = coco_inst.loadCats([cat_id])[0]["name"]
            cat_count[cat_name] = cat_count.get(cat_name, 0) + 1

        out["objects"] = {
            "num_instances": len(inst_anns),
            "category_counts": dict(sorted(cat_count.items(), key=lambda kv: (-kv[1], kv[0]))),
        }

    return out


def coco_pair_from_cap_ann_id(
    cap_ann_id: int,
    coco_root: str | Path,
    captions_ann: str | Path = "annotations/captions_val2017.json",
    split: str = "val2017",
) -> Dict[str, Any]:
    
    coco_root = Path(coco_root)
    cap_path = (coco_root / captions_ann) if not Path(captions_ann).is_absolute() else Path(captions_ann)
    coco_caps = COCO(str(cap_path))

    ann = coco_caps.loadAnns([cap_ann_id])
    if not ann:
        raise ValueError(f"caption ann_id={cap_ann_id} not in {cap_path}")
    ann = ann[0]

    img_id = ann["image_id"]
    img = coco_caps.loadImgs([img_id])[0]
    file_name = img.get("file_name", "")
    img_path = coco_root / split / file_name if file_name else None

    return {
        "cap_ann_id": cap_ann_id,
        "img_id": img_id,
        "caption": ann.get("caption", ""),
        "image": {
            "file_name": file_name,
            "local_path": str(img_path) if img_path else None,
            "coco_url": img.get("coco_url"),
            "flickr_url": img.get("flickr_url"),
            "width": img.get("width"),
            "height": img.get("height"),
        },
    }


def coco_instances_from_img_id(
    img_id: int,
    coco_root: str | Path,
    split: str = "val2017",
    instances_ann: str | Path = "annotations/instances_val2017.json",
    min_area: float = 0.0,
    return_masks: bool = False,
) -> Dict[str, Any]:
    """Load COCO instance annotations for one image.

    Returns a dict containing basic image info and a list of instances.
    Each instance includes bbox in xywh, category id/name, area, iscrowd.

    Notes:
    - bbox format is COCO standard: [x, y, w, h]
    - if return_masks=True, includes the raw COCO 'segmentation' field
    """

    coco_root = Path(coco_root)
    inst_path = (coco_root / instances_ann) if not Path(instances_ann).is_absolute() else Path(instances_ann)
    coco_inst = COCO(str(inst_path))

    imgs = coco_inst.loadImgs([img_id])
    if not imgs:
        raise ValueError(f"img_id={img_id} not in {inst_path}")
    img = imgs[0]

    file_name = img.get("file_name", "")
    img_path = coco_root / split / file_name if file_name else None

    ann_ids = coco_inst.getAnnIds(imgIds=[img_id])
    anns = coco_inst.loadAnns(ann_ids)

    instances: List[Dict[str, Any]] = []
    for a in anns:
        bbox = a.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue

        area = float(a.get("area", bbox[2] * bbox[3]))
        if area < float(min_area):
            continue

        cat_id = a.get("category_id")
        cat_name = None
        if cat_id is not None:
            cats = coco_inst.loadCats([cat_id])
            if cats:
                cat_name = cats[0].get("name")

        inst: Dict[str, Any] = {
            "ann_id": a.get("id"),
            "category_id": cat_id,
            "category_name": cat_name,
            "bbox_xywh": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "area": area,
            "iscrowd": int(a.get("iscrowd", 0)),
        }
        if return_masks:
            inst["segmentation"] = a.get("segmentation")
        instances.append(inst)

    instances.sort(key=lambda d: (-float(d.get("area", 0.0)), str(d.get("category_name", ""))))

    return {
        "img_id": img_id,
        "image": {
            "file_name": file_name,
            "local_path": str(img_path) if img_path else None,
            "coco_url": img.get("coco_url"),
            "flickr_url": img.get("flickr_url"),
            "width": img.get("width"),
            "height": img.get("height"),
        },
        "instances": instances,
    }


def draw_coco_bboxes(
    image_rgb,
    instances: List[Dict[str, Any]],
    max_boxes: Optional[int] = None,
    linewidth: int = 2,
    alpha: float = 0.15,
    fontsize: int = 10,
):
    """Draw COCO bbox annotations on an RGB image.

    Parameters
    - image_rgb: numpy array (H,W,3) RGB
    - instances: list from coco_instances_from_img_id()['instances']

    Returns
    - fig, ax: matplotlib figure/axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image_rgb)
    ax.axis("off")

    if max_boxes is None:
        draw_list = instances
    else:
        draw_list = instances[: int(max_boxes)]

    for inst in draw_list:
        x, y, w, h = inst.get("bbox_xywh", [0, 0, 0, 0])
        name = inst.get("category_name") or str(inst.get("category_id"))

        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=linewidth,
            edgecolor="lime",
            facecolor=(0, 1, 0, alpha),
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            f"{name}",
            color="black",
            fontsize=fontsize,
            bbox=dict(facecolor="lime", alpha=0.8, pad=1, edgecolor="none"),
            verticalalignment="top",
        )

    return fig, ax
