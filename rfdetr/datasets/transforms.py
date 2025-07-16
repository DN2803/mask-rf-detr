# transforms.py
# ------------------------------------------------------------------------
# Conditional DETR – clean rewrite
# Copyright (c) 2025
# ------------------------------------------------------------------------

from __future__ import annotations

import random
from typing import Iterable, List, Tuple, Union

import torch
from PIL import Image
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F

__all__ = [
    # helpers
    "interpolate", "box_xyxy_to_cxcywh", "box_cxcywh_to_xyxy",
    "crop", "hflip", "resize", "pad",
    # transform classes
    "RandomCrop", "RandomSizeCrop", "CenterCrop", "RandomHorizontalFlip",
    "RandomResize", "RandomPad", "RandomSelect", "ToTensor",
    "RandomErasing", "Normalize", "Compose", "SquareResize",
]


# ---------- 1. Rotation 90 ----------
class RandomRotation90:
    def __call__(self, img, tgt):
        k = random.randint(0, 3)
        if k == 0:
            return img, tgt
        angle = k * 90
        # PIL rotate
        img = F.rotate(img, angle, expand=True, fill=255)
        h, w = img.height, img.width

        out = tgt.copy()
        if "boxes" in out:
            x0,y0,x1,y1 = out["boxes"].unbind(-1)
            if k == 1:  # 90
                out["boxes"] = torch.stack([y0, w-x1, y1, w-x0], -1)
            elif k == 2:  # 180
                out["boxes"] = torch.stack([w-x1, h-y1, w-x0, h-y0], -1)
            else:        # 270
                out["boxes"] = torch.stack([h-y1, x0, h-y0, x1], -1)
        if "masks" in out:
            out["masks"] = torch.rot90(out["masks"], k, (-2,-1))
        out["size"] = torch.tensor([h, w])
        return img, out
# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def interpolate(
    x: Tensor,
    size: Tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
) -> Tensor:
    """
    Thin wrapper around ``torch.nn.functional.interpolate`` that
    transparently handles empty batches (B == 0).

    Args mirror those of ``F.interpolate``.
    """
    if x.shape[0] == 0:
        return x
    return torch.nn.functional.interpolate(x, size=size, scale_factor=scale_factor,
                         mode=mode, align_corners=align_corners)


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert [cx, cy, w, h] → [x0, y0, x1, y1]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1
    )


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert [x0, y0, x1, y1] → [cx, cy, w, h]."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(
        ((x0 + x1) * 0.5, (y0 + y1) * 0.5, x1 - x0, y1 - y0), dim=-1
    )

# ------------------------------------------------------------
# Core geometric ops (functional)
# ------------------------------------------------------------

def crop(image: Image.Image, target: dict, region: Tuple[int, int, int, int]):
    """Crop PIL image + target (region = (top, left, height, width))."""
    cropped = F.crop(image, *region)
    i, j, h, w = region
    tgt = target.copy()

    tgt["size"] = torch.tensor([h, w])

    fields: List[str] = ["labels", "area", "iscrowd"]

    # --- boxes
    if "boxes" in tgt:
        boxes = tgt["boxes"] - torch.tensor([j, i, j, i], dtype=torch.float32)
        boxes = boxes.view(-1, 2, 2).clamp(min=0)
        max_wh = torch.tensor([w, h])
        boxes = boxes.clamp(max=max_wh).view(-1, 4)

        area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * \
               (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

        tgt.update(boxes=boxes, area=area)
        fields.append("boxes")

    # --- masks
    if "masks" in tgt:
        tgt["masks"] = tgt["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # --- filter zero‑area
    if {"boxes", "masks"} & set(tgt):
        keep = tgt["area"] > 1 if "boxes" in tgt else tgt["masks"].flatten(1).any(1)
        for f in fields:
            tgt[f] = tgt[f][keep]

    return cropped, tgt


def hflip(image: Image.Image, target: dict):
    """Horizontal flip PIL image + target."""
    flipped = F.hflip(image)
    w, _ = image.size
    tgt = target.copy()

    if "boxes" in tgt:
        boxes = tgt["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) \
              + torch.tensor([w, 0, w, 0])
        tgt["boxes"] = boxes

    if "masks" in tgt:
        tgt["masks"] = tgt["masks"].flip(-1)

    return flipped, tgt


def _get_resize_shape(
    img_size: Tuple[int, int], size: Union[int, Tuple[int, int]], max_size: int | None
) -> Tuple[int, int]:
    """Helper that returns (height, width) after resize while keeping aspect."""
    w, h = img_size
    if isinstance(size, Iterable):
        return size[::-1]  # (w, h) → (h, w)

    min_orig, max_orig = float(min(w, h)), float(max(w, h))
    if max_size and max_orig / min_orig * size > max_size:
        size = int(round(max_size * min_orig / max_orig))

    if (w <= h and w == size) or (h <= w and h == size):
        return h, w

    if w < h:
        return int(size * h / w), size
    return size, int(size * w / h)


def resize(
    image: Image.Image,
    target: dict | None,
    size: Union[int, Tuple[int, int]],
    max_size: int | None = None,
):
    """Resize PIL image + target keeping aspect ratio unless size is tuple."""
    new_h, new_w = _get_resize_shape(image.size, size, max_size)
    rescaled = F.resize(image, (new_h, new_w))

    if target is None:
        return rescaled, None

    tgt = target.copy()
    ratio_w, ratio_h = new_w / image.width, new_h / image.height

    if "boxes" in tgt:
        tgt["boxes"] = tgt["boxes"] * torch.tensor(
            [ratio_w, ratio_h, ratio_w, ratio_h]
        )

    if "area" in tgt:
        tgt["area"] = tgt["area"] * (ratio_w * ratio_h)

    tgt["size"] = torch.tensor([new_h, new_w])

    if "masks" in tgt:
        tgt["masks"] = (
            interpolate(tgt["masks"][:, None].float(), size=(new_h, new_w), mode="nearest")[:, 0] > 0.5
        )

    return rescaled, tgt


def pad(
    image: Image.Image | Tensor,
    target: dict | None,
    padding: Tuple[int, int],
):
    """Pad bottom‑right by (pad_x, pad_y)."""
    padded = F.pad(image, (0, 0, padding[0], padding[1]))

    if target is None:
        return padded, None

    tgt = target.copy()
    if isinstance(padded, Image.Image):
        w, h = padded.size
    else:  # Tensor C×H×W
        h, w = padded.shape[-2:]

    tgt["size"] = torch.tensor([h, w])

    if "masks" in tgt:
        tgt["masks"] = torch.nn.functional.pad(
            tgt["masks"], (0, padding[0], 0, padding[1])
        )
    return padded, tgt

# ------------------------------------------------------------
# Transform objects (callable)
# ------------------------------------------------------------

class RandomCrop:
    """Fixed‑size random crop."""
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, img: Image.Image, tgt: dict):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, tgt, region)


class RandomSizeCrop:
    """Random crop w,h uniform between min_size & max_size."""
    def __init__(self, min_size: int, max_size: int) -> None:
        self.min = min_size
        self.max = max_size

    def __call__(self, img: Image.Image, tgt: dict):
        w = random.randint(self.min, min(img.width, self.max))
        h = random.randint(self.min, min(img.height, self.max))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, tgt, region)


class CenterCrop:
    """Center crop to (h, w)."""
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, img: Image.Image, tgt: dict):
        ih, iw = img.height, img.width
        ch, cw = self.size
        top = int(round((ih - ch) * 0.5))
        left = int(round((iw - cw) * 0.5))
        return crop(img, tgt, (top, left, ch, cw))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img, tgt):
        return hflip(img, tgt) if random.random() < self.p else (img, tgt)


class RandomResize:
    def __init__(self, sizes: Iterable[int | Tuple[int, int]], max_size: int | None = None):
        self.sizes = list(sizes)
        self.max_size = max_size

    def __call__(self, img, tgt):
        return resize(img, tgt, random.choice(self.sizes), self.max_size)


class RandomPad:
    def __init__(self, max_pad: int) -> None:
        self.max_pad = max_pad

    def __call__(self, img, tgt):
        pad_x, pad_y = random.randint(0, self.max_pad), random.randint(0, self.max_pad)
        return pad(img, tgt, (pad_x, pad_y))


class RandomSelect:
    """Apply either ``t1`` *p* or ``t2`` *(1 − p)*."""
    def __init__(self, t1, t2, p: float = 0.5) -> None:
        self.t1, self.t2, self.p = t1, t2, p

    def __call__(self, img, tgt):
        return self.t1(img, tgt) if random.random() < self.p else self.t2(img, tgt)


class ToTensor:
    def __call__(self, img, tgt):
        return F.to_tensor(img), tgt


class RandomErasing:
    def __init__(self, *args, **kwargs) -> None:
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, tgt):
        return self.eraser(img), tgt


class Normalize:
    """Normalize + (optionally) scale boxes to [0,1]."""
    def __init__(self, mean: List[float], std: List[float], reparam: bool = False) -> None:
        self.mean, self.std, self.reparam = mean, std, reparam

    def __call__(self, img: Tensor, tgt: dict | None = None):
        img = F.normalize(img, self.mean, self.std)

        if tgt is None:
            return img, None

        out = tgt.copy()
        if "boxes" in out:
            boxes = box_xyxy_to_cxcywh(out["boxes"].to(torch.float32))
            if not self.reparam:  # scale to 0‑1
                h, w = img.shape[-2:]
                boxes /= torch.tensor([w, h, w, h], dtype=torch.float32)
            out["boxes"] = boxes
        return img, out


class Compose:
    """Compose list of transforms that accept (img, target)."""
    def __init__(self, transforms: Iterable) -> None:
        self.tfms = list(transforms)

    def __call__(self, img, tgt):
        for t in self.tfms:
            img, tgt = t(img, tgt)
        return img, tgt

    def __repr__(self):
        body = ",\n    ".join(repr(t) for t in self.tfms)
        return f"{self.__class__.__name__}(\n    {body}\n)"


class SquareResize:
    """Resize to a random square size picked from *sizes*."""
    def __init__(self, sizes: Iterable[int]):
        self.sizes = list(sizes)

    def __call__(self, img: Image.Image, tgt: dict | None = None):
        s = random.choice(self.sizes)
        orig_w, orig_h = img.size
        img = img.resize((s, s), Image.BILINEAR)

        if tgt is None:
            return img, None

        out = tgt.copy()
        scale = torch.tensor([s / orig_w, s / orig_h, s / orig_w, s / orig_h])

        if "boxes" in out:
            out["boxes"] = out["boxes"].clone() * scale
        if "masks" in out and out["masks"].numel():
            out["masks"] = interpolate(out["masks"].unsqueeze(0).float(),
                                       size=(s, s), mode="nearest")[0]
        out["size"] = torch.tensor([s, s])
        return img, out
