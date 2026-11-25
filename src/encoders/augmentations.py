"""Augmentation conditions — the invariance "treatments".

``build_augmentation(condition, strength)`` returns a single-view transform
(PIL -> CHW float tensor in [0,1]). ``TwoViewTransform`` wraps it into the SimCLR
positive-pair generator.

Phase 1 trains the ``standard`` SimCLR stack (a strong, canonical mix) for the
reference encoder, plus a ``control`` (minimal) stack. The per-factor conditions
(color / position / orientation / scale) are the Phase-3 invariance treatments:
each varies ONLY its targeted factor across the two views and leaves the image
otherwise intact, so the induced invariance is clean and interpretable (no
confounding geometry/colour). Each is offered at ``weak`` and ``strong`` strength.
"""

from __future__ import annotations

from typing import Callable

import torchvision.transforms as T

IMAGE_SIZE = 64

CONDITIONS = ("standard", "control", "color", "position", "orientation", "scale",
              "scale_recipe")
STRENGTHS = ("weak", "strong")


class TwoViewTransform:
    """Generate two independently-augmented views of one image (SimCLR pair)."""

    def __init__(self, base: Callable):
        self.base = base

    def __call__(self, x):
        return self.base(x), self.base(x)


def eval_transform() -> Callable:
    """Deterministic transform for probing/extraction: just PIL -> tensor."""
    return T.Compose([T.ToTensor()])


def build_augmentation(
    condition: str = "standard",
    strength: str = "strong",
    image_size: int = IMAGE_SIZE,
) -> Callable:
    """Return a single-view augmentation transform for a (condition, strength)."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")
    if strength not in STRENGTHS:
        raise ValueError(f"unknown strength {strength!r}; choose from {STRENGTHS}")

    if condition == "standard":

        jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.3, 1.0), antialias=True),
                T.RandomHorizontalFlip(),
                T.RandomApply([jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
            ]
        )

    if condition == "control":


        crop_scale = (0.8, 1.0) if strength == "strong" else (0.9, 1.0)
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=crop_scale, antialias=True),
                T.ToTensor(),
            ]
        )

    weak = strength == "weak"

    if condition == "color":



        if weak:
            jitter = T.ColorJitter(0.2, 0.2, 0.2, 0.1)
            return T.Compose([T.RandomApply([jitter], p=0.8), T.ToTensor()])
        jitter = T.ColorJitter(0.8, 0.8, 0.8, 0.5)
        return T.Compose(
            [
                T.RandomApply([jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
            ]
        )

    if condition == "position":



        frac = 0.10 if weak else 0.25
        return T.Compose(
            [
                T.RandomAffine(degrees=0, translate=(frac, frac), fill=0),
                T.ToTensor(),
            ]
        )

    if condition == "orientation":


        degrees = 15 if weak else 30
        return T.Compose([T.RandomRotation(degrees, fill=0), T.ToTensor()])

    if condition == "scale":



        scale_range = (0.9, 1.1) if weak else (0.75, 1.25)
        return T.Compose(
            [
                T.RandomAffine(degrees=0, scale=scale_range, fill=0),
                T.ToTensor(),
            ]
        )

    if condition == "scale_recipe":






        scale_range = (0.9, 1.1) if weak else (0.75, 1.25)
        jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        return T.Compose(
            [
                T.RandomAffine(degrees=0, scale=scale_range, fill=0),
                T.RandomApply([jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
            ]
        )

    raise ValueError(f"unhandled condition {condition!r}")
