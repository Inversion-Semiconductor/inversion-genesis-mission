"""Frozen image preparation; frame acquisition and shot matching are external."""

import numpy as np


def _image(raw):
    image = np.asarray(raw, dtype=np.float64)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Expected one finite two-dimensional image.")
    return image


def _edge_subtract(raw):
    image = _image(raw)
    if min(image.shape) <= 40:
        raise ValueError("Image must exceed the 20-pixel background border.")
    border = np.ones(image.shape, dtype=bool)
    border[20:-20, 20:-20] = False
    return np.maximum(image - image[border].mean(), 0).astype(np.float32)


def gaia_image(raw, dust_mask):
    """Background subtraction, fixed 304x304 crop, and calibrated dust repair."""
    from skimage.restoration import inpaint_biharmonic

    image = _edge_subtract(raw)[140:444, 77:381].copy()
    mask = np.asarray(dust_mask, dtype=bool)
    if image.shape != (304, 304) or mask.shape != image.shape:
        raise ValueError(
            "Gaia requires the full camera frame and its 304x304 dust mask."
        )
    return np.maximum(
        inpaint_biharmonic(image, mask, split_into_regions=True), 0
    ).astype(np.float32)


def topview_image(raw):
    """Preserve counts in the fixed TopView ROI, without background subtraction."""
    image = _image(raw)[336:592, 128:640]
    if image.shape != (256, 512) or (image < 0).any():
        raise ValueError(
            "TopView requires a nonnegative full frame covering the calibrated ROI."
        )
    return image.astype(np.float32)


def camera1_image(codes, peak):
    """Restore exported 0..4094 camera codes to physical image intensity."""
    image = np.asarray(codes, dtype=np.float32)
    if (
        image.shape != (189, 285)
        or not np.isfinite(image).all()
        or (image < 0).any()
        or (image > 4094).any()
        or not np.isfinite(peak)
        or peak <= 0
    ):
        raise ValueError(
            "Expected camera-1 codes (189, 285) in [0, 4094] and a positive peak."
        )
    return image * (np.float32(peak) / np.float32(4094))


def frog_image(raw, crop):
    """Subtract the full-frame border background before applying the PCA ROI."""
    x0, x1, y0, y1 = crop
    return _edge_subtract(raw)[y0:y1, x0:x1]


def oap_image(raw):
    """Remove the first 270 columns, then subtract the four-corner background."""
    image = _image(raw)[:, 270:]
    if image.shape != (963, 1010):
        raise ValueError("OAP requires a full frame of shape (963, 1280).")
    p = 64
    c = [
        np.median(a)
        for a in (image[:p, :p], image[:p, -p:], image[-p:, :p], image[-p:, -p:])
    ]
    h, w = image.shape
    x = (np.arange(w) - (p - 1) / 2) / (w - p)
    y = (np.arange(h) - (p - 1) / 2) / (h - p)
    background = (1 - y[:, None]) * ((1 - x) * c[0] + x * c[1]) + y[:, None] * (
        (1 - x) * c[2] + x * c[3]
    )
    return np.maximum(image - background, 0).astype(np.float32)
