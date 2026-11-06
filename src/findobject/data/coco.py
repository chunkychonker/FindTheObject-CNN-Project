"""List COCO val2017 images usable as backgrounds (those with no real stop sign)."""

import json
from pathlib import Path

STOP_SIGN_CATEGORY_ID = 13


def list_backgrounds(images_dir: Path, instances_json: Path) -> list[Path]:
    """Background image paths sorted by COCO image id, excluding any image annotated
    with a stop sign. Raises FileNotFoundError / json.JSONDecodeError on bad inputs."""
    with open(instances_json) as f:
        coco = json.load(f)
    has_stop = {a["image_id"] for a in coco["annotations"] if a["category_id"] == STOP_SIGN_CATEGORY_ID}
    images = sorted(coco["images"], key=lambda im: im["id"])
    return [images_dir / im["file_name"] for im in images if im["id"] not in has_stop]
