import os
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from utils.pose_utils import DiscretizedAgentPose


def xyxy_to_normalized_xywh(box: tuple[int, int, int, int], size: tuple[int,int], center=True) -> tuple[float, float, float, float]:
    img_width, img_height = size
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    x = x1
    y = y1
    w = box_width
    h = box_height
    if center:
        x = ((x1 + x2) / 2)
        y = ((y1 + y2) / 2)
    x /= img_width
    y /= img_height
    w /= img_width
    h /= img_width
    return x, y, w, h


def normalized_xywh_to_xyxy(xywh: tuple[float, float, float, float], size: tuple[int,int], center=True) -> tuple[int, int, int, int]:
    x, y, w, h = xywh
    img_width, img_height = size
    x *= img_width
    y *= img_height
    w *= img_width
    h *= img_height
    if center:
        x1 = int(round(x - w / 2))
        x2 = int(round(x + w / 2))
        y1 = int(round(y - h / 2))
        y2 = int(round(y + h / 2))
    else:
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
    return x1, y1, x2, y2


def rgb_to_hex(rgb):
    r,g,b = rgb
    return '#%02x%02x%02x' % (int(r), int(g), int(b))


def hex_to_rgb(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    if len(hx) != 7:
        raise ValueError("Hex must be #------")
    hx = hx[1:]  # omit the '#'
    r = int('0x'+hx[:2], 16)
    g = int('0x'+hx[2:4], 16)
    b = int('0x'+hx[4:6], 16)
    return (r,g,b)


def make_colors(num, seed=1, ctype=1) -> list:
    """Return `num` number of unique colors in a list,
    where colors are [r,g,b] lists."""
    rng_gen = np.random.default_rng(seed)
    colors = []

    def random_unique_color(colors, ctype, rng_gen):
        """
        ctype=1: completely random
        ctype=2: red random
        ctype=3: blue random
        ctype=4: green random
        ctype=5: yellow random
        """
        if ctype == 1:
            color = "#%06x" % rng_gen.randint(0x444444, 0x999999)
            while color in colors:
                color = "#%06x" % rng_gen.randint(0x444444, 0x999999)
        elif ctype == 2:
            color = "#%02x0000" % rng_gen.randint(0xAA, 0xFF)
            while color in colors:
                color = "#%02x0000" % rng_gen.randint(0xAA, 0xFF)
        elif ctype == 4:  # green
            color = "#00%02x00" % rng_gen.randint(0xAA, 0xFF)
            while color in colors:
                color = "#00%02x00" % rng_gen.randint(0xAA, 0xFF)
        elif ctype == 3:  # blue
            color = "#0000%02x" % rng_gen.randint(0xAA, 0xFF)
            while color in colors:
                color = "#0000%02x" % rng_gen.randint(0xAA, 0xFF)
        elif ctype == 5:  # yellow
            h = rng_gen.randint(0xAA, 0xFF)
            color = "#%02x%02x00" % (h, h)
            while color in colors:
                h = rng_gen.randint(0xAA, 0xFF)
                color = "#%02x%02x00" % (h, h)
        else:
            raise ValueError("Unrecognized color type %s" % (str(ctype)))
        return color

    while len(colors) < num:
        colors.append(list(hex_to_rgb(random_unique_color(colors,ctype=ctype,rng_gen=rng_gen))))
    return colors


def saveimg(img, path):
    im = Image.fromarray(img)
    im.save(path, format="PNG")


def dataset_load_info(
    data_dir: Path,
) -> tuple[list[Path], list[str], list[tuple[int, int, int]]]:
    dataset_yaml_path = None

    for file in data_dir.parent.iterdir():
        if file.suffix == ".yaml":
            dataset_yaml_path = file
            break

    if dataset_yaml_path is None:
        raise ValueError("No YAML configuration file found in the dataset directory.")

    with open(dataset_yaml_path) as f:
        config = yaml.safe_load(f)

    class_names = list(config["names"])
    colors = list(config["colors"])

    return (enumerate_fnames(data_dir), class_names, colors)  # type: ignore


def load_gt(
    path: Path, fname: Path, class_names: list[str], img_size: tuple[int,int]
) -> list[dict]:
    w, h = img_size

    label_boxes: list[dict] = []

    with open(path) as fa:
        for line in fa.readlines():
            annot = line.strip().split()
            assert len(annot) == 5

            class_id = int(annot[0])
            x,y,w,h = list(map(float, annot[1:]))
            xyxy = normalized_xywh_to_xyxy(
                (x,y,w,h), img_size, center=True
            )

            label_boxes.append(
                dict(
                    xmin=xyxy[0],
                    ymin=xyxy[1],
                    xmax=xyxy[2],
                    ymax=xyxy[3],
                    confidence=1.0,  # GT dummy confidence
                    bbx_confidence=1.0,
                    class_wise_confidence=np.array(
                        [i == class_id for i in range(len(class_names))], dtype=float
                    ),
                    class_id=class_id,
                    class_name=class_names[class_id],
                    bbx_features=np.array([], dtype=float),
                )
            )
    return label_boxes


def load_img(fname: Path) -> np.ndarray:
    # Check image extension and try to load accordingly
    img_path_jpg = (fname.with_suffix(".jpg"))
    img_path_png = (fname.with_suffix(".png"))

    if img_path_jpg.exists():
        img_path = img_path_jpg
    elif img_path_png.exists():
        img_path = img_path_png
    else:
        raise FileNotFoundError(f"No image file found for {fname}  with .jpg or .png extension.")

    with open(img_path, "rb") as f:
        im = Image.open(f)
        im = np.array(im)

    assert isinstance(im, np.ndarray), f"Image loading failed for {fname}"

    return im


def create_dataset_yaml(
    data_dir, class_names, name
):

    content = dict(
        path=data_dir,
        train="train",  # training images relative to 'path'
        val="val",  # validation images relative to path
        nc=len(class_names),  # number of class_names
        names=class_names,
        colors=make_colors(len(class_names), seed=1),
    )
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "{}-dataset.yaml".format(name)), "w") as f:
        yaml.dump(content, f)


def pose2fname(scene_name: str, pose: DiscretizedAgentPose) -> Path:
    """Get the filename corresponding to the given pose."""
    fname = Path(f"{scene_name}-x{pose.idx_x}-z{pose.idx_z}-y{pose.idx_yaw}-p{pose.idx_pitch}-by{pose.yaw_bins}-bp{pose.pitch_bins}")
    return fname


def fname2pose(fname: Path) -> DiscretizedAgentPose:
    fname_str = str(fname.stem)

    def extract_numerical_value(string: str, prefix: str):
        where_start = fname_str.find(prefix) + len(prefix)
        where_end = where_start + (
            fname_str[where_start:].find("-")
            if fname_str[where_start:].find("-") != -1
            else len(fname_str[where_start:])
        )
        return float(fname_str[where_start:where_end])

    idx_x = int(extract_numerical_value(fname_str, "-x"))
    idx_z = int(extract_numerical_value(fname_str, "-z"))
    idx_yaw = int(extract_numerical_value(fname_str, "-y"))
    idx_pitch = int(extract_numerical_value(fname_str, "-p"))
    yaw_bins = int(extract_numerical_value(fname_str, "-by"))
    pitch_bins = int(extract_numerical_value(fname_str, "-bp"))

    return DiscretizedAgentPose(idx_x, idx_z, idx_yaw, idx_pitch, yaw_bins, pitch_bins)


def store_label(
    gt_bounding_boxes: list[dict],
    fname: Path,
    data_dir: Path,
    img_shape: tuple[int, int, int],
) -> None:
    label_dir = data_dir / "labels"
    os.makedirs(label_dir, exist_ok=True)

    final_path = label_dir / (fname.stem + ".txt")

    if final_path.exists():
        return

    annotations = []

    for bbx in sorted(gt_bounding_boxes, key=lambda x: x["class_id"]):
        x_center, y_center, w, h = xyxy_to_normalized_xywh(
            (bbx["xmin"], bbx["ymin"], bbx["xmax"], bbx["ymax"]), img_shape[:2], center=True
        )
        annotations.append(f"{bbx["class_id"]} {x_center} {y_center} {w} {h}")

    tmp_path = label_dir / f"{fname.stem}.{uuid.uuid4()}.tmp"

    try:
        with open(tmp_path, "w") as f:
            f.write("\n".join(annotations) + "\n")
        os.replace(tmp_path, final_path)
        os.sync()  # Ensure data is written to disk

    except Exception as e:
        if tmp_path.exists():
            os.remove(tmp_path)
        raise e


def store_image(
    img: np.ndarray,
    data_dir: Path,
    fname: Path,
    max_wait: float = 0.1,
) -> Path:
    img_dir = data_dir / "images"
    os.makedirs(img_dir, exist_ok=True)

    final_path = img_dir / f"{str(fname)}.jpg"

    if final_path.exists():
        return final_path

    tmp_path = img_dir / f"{str(fname)}.{uuid.uuid4()}.tmp"
    saveimg(img, tmp_path)

    if not final_path.exists():
        raise RuntimeError(f"Failed to store image: {final_path}")

    return final_path


def enumerate_fnames(source_data_dir: Path) -> list[Path]:
    l = []
    for image_name in os.listdir(source_data_dir / "images"):
        l.append(Path(image_name))
    return sorted(l)  # type: ignore