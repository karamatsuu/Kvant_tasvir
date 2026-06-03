from __future__ import annotations

from collections import deque
from typing import Iterable, Tuple

import cv2
import numpy as np


ArrayLike = np.ndarray


def _as_array(img: ArrayLike) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim not in (2, 3):
        raise ValueError("img must be a grayscale or color ndarray")
    if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
        raise ValueError("color images must have 1, 3, or 4 channels")
    return arr


def _to_uint8(img: ArrayLike) -> np.ndarray:
    arr = _as_array(img)
    if arr.dtype == np.uint8:
        return arr.copy()

    arr = arr.astype(np.float32, copy=False)
    if arr.size == 0:
        return arr.astype(np.uint8)

    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.where(finite, arr, 0.0)

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if 0.0 <= min_val and max_val <= 1.0:
        arr = arr * 255.0

    return np.clip(arr, 0, 255).astype(np.uint8)


def _to_gray_uint8(img: ArrayLike) -> np.ndarray:
    arr = _to_uint8(img)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 1:
        return arr[:, :, 0]
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)


def _odd_ksize(ksize: int | Tuple[int, int]) -> int | Tuple[int, int]:
    if isinstance(ksize, Iterable) and not isinstance(ksize, (str, bytes)):
        values = tuple(max(1, int(v)) for v in ksize)
        if len(values) != 2:
            raise ValueError("ksize tuple must contain exactly two values")
        return tuple(v if v % 2 else v + 1 for v in values)

    value = max(1, int(ksize))
    return value if value % 2 else value + 1


def _kernel(ksize: int | Tuple[int, int], kernel_shape: str | int) -> np.ndarray:
    if isinstance(ksize, Iterable) and not isinstance(ksize, (str, bytes)):
        size = tuple(max(1, int(v)) for v in ksize)
    else:
        value = max(1, int(ksize))
        size = (value, value)

    shape_name = str(kernel_shape).lower()
    if shape_name in {"rect", "rectangle", "0", str(cv2.MORPH_RECT)}:
        shape = cv2.MORPH_RECT
    elif shape_name in {"ellipse", "elliptical", "1", str(cv2.MORPH_ELLIPSE)}:
        shape = cv2.MORPH_ELLIPSE
    elif shape_name in {"cross", "2", str(cv2.MORPH_CROSS)}:
        shape = cv2.MORPH_CROSS
    else:
        raise ValueError("kernel_shape must be 'rect', 'ellipse', or 'cross'")
    return cv2.getStructuringElement(shape, size)


def rgb_to_grayscale(img: ArrayLike) -> np.ndarray:
    return _to_gray_uint8(img)


def rgb_to_hsv(img: ArrayLike) -> np.ndarray:
    arr = _to_uint8(img)
    if arr.ndim == 2 or arr.shape[2] == 1:
        arr = cv2.cvtColor(_to_gray_uint8(arr), cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)


def rgb_to_lab(img: ArrayLike) -> np.ndarray:
    arr = _to_uint8(img)
    if arr.ndim == 2 or arr.shape[2] == 1:
        arr = cv2.cvtColor(_to_gray_uint8(arr), cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)


def resize_img(
    img: ArrayLike,
    mode: str = "nearest",
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    arr = _as_array(img)
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("scale_x and scale_y must be positive")

    modes = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    key = mode.lower()
    if key not in modes:
        raise ValueError("mode must be 'nearest', 'bilinear', 'bicubic', or 'lanczos'")
    return cv2.resize(arr, None, fx=float(scale_x), fy=float(scale_y), interpolation=modes[key])


def filter_mean(img: ArrayLike, ksize: int | Tuple[int, int] = 3) -> np.ndarray:
    return cv2.blur(_to_uint8(img), _ksize_pair(ksize))


def filter_median(img: ArrayLike, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(_to_uint8(img), int(_odd_ksize(ksize)))


def filter_gaussian(
    img: ArrayLike,
    ksize: int | Tuple[int, int] = 3,
    sigma: float = 0.0,
) -> np.ndarray:
    return cv2.GaussianBlur(_to_uint8(img), _ksize_pair(_odd_ksize(ksize)), float(sigma))


def filter_bilateral(
    img: ArrayLike,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    return cv2.bilateralFilter(_to_uint8(img), int(d), float(sigma_color), float(sigma_space))


def _ksize_pair(ksize: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(ksize, Iterable) and not isinstance(ksize, (str, bytes)):
        values = tuple(max(1, int(v)) for v in ksize)
        if len(values) != 2:
            raise ValueError("ksize tuple must contain exactly two values")
        return values
    value = max(1, int(ksize))
    return value, value


def _linear_stretch_array(arr: ArrayLike) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val <= min_val:
        return np.zeros_like(arr, dtype=np.uint8)
    stretched = (arr - min_val) * (255.0 / (max_val - min_val))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def contrast_linear_stretch(img: ArrayLike) -> np.ndarray:
    return _linear_stretch_array(_as_array(img))


def transform_log(img: ArrayLike) -> np.ndarray:
    arr = _to_uint8(img).astype(np.float32)
    logged = np.log1p(arr)
    max_val = np.max(logged)
    if max_val == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((logged / max_val) * 255.0, 0, 255).astype(np.uint8)


def transform_gamma(img: ArrayLike, gamma: float = 1.0) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    arr = _to_uint8(img).astype(np.float32) / 255.0
    corrected = np.power(arr, float(gamma)) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def hist_equalization(img: ArrayLike) -> np.ndarray:
    arr = _to_uint8(img)
    if arr.ndim == 2 or arr.shape[2] == 1:
        return cv2.equalizeHist(_to_gray_uint8(arr))

    if arr.shape[2] == 4:
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    else:
        bgr = arr
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe_adjustment(
    img: ArrayLike,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    arr = _to_uint8(img)
    if arr.ndim == 2 or arr.shape[2] == 1:
        return clahe.apply(_to_gray_uint8(arr))

    if arr.shape[2] == 4:
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    else:
        bgr = arr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalize_min_max(img: ArrayLike) -> np.ndarray:
    return contrast_linear_stretch(img)


def normalize_z_score(img: ArrayLike) -> np.ndarray:
    arr = _as_array(img).astype(np.float32)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    z = (arr - mean) / std
    return _linear_stretch_array(z)


def edge_sobel(img: ArrayLike) -> np.ndarray:
    gray = _to_gray_uint8(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return contrast_linear_stretch(mag)


def edge_prewitt(img: ArrayLike) -> np.ndarray:
    gray = _to_gray_uint8(img)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    return contrast_linear_stretch(cv2.magnitude(gx, gy))


def edge_roberts(img: ArrayLike) -> np.ndarray:
    gray = _to_gray_uint8(img)
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    return contrast_linear_stretch(cv2.magnitude(gx, gy))


def edge_canny(img: ArrayLike, low_thresh: float = 50, high_thresh: float = 150) -> np.ndarray:
    return cv2.Canny(_to_gray_uint8(img), float(low_thresh), float(high_thresh))


def log_edge(img: ArrayLike) -> np.ndarray:
    gray = _to_gray_uint8(img)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    return contrast_linear_stretch(np.abs(lap))


def thresh_manual(img: ArrayLike, value: float = 127) -> np.ndarray:
    _, binary = cv2.threshold(_to_gray_uint8(img), float(value), 255, cv2.THRESH_BINARY)
    return binary


def thresh_otsu(img: ArrayLike) -> np.ndarray:
    _, binary = cv2.threshold(_to_gray_uint8(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def thresh_adaptive(img: ArrayLike) -> np.ndarray:
    gray = _to_gray_uint8(img)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


def morph_operation(
    img: ArrayLike,
    op_type: str = "erosion",
    kernel_shape: str = "rect",
    ksize: int | Tuple[int, int] = 3,
) -> np.ndarray:
    arr = _to_uint8(img)
    kern = _kernel(ksize, kernel_shape)
    operations = {
        "erosion": lambda image: cv2.erode(image, kern, iterations=1),
        "dilation": lambda image: cv2.dilate(image, kern, iterations=1),
        "opening": lambda image: cv2.morphologyEx(image, cv2.MORPH_OPEN, kern),
        "closing": lambda image: cv2.morphologyEx(image, cv2.MORPH_CLOSE, kern),
    }
    key = op_type.lower()
    if key not in operations:
        raise ValueError("op_type must be 'erosion', 'dilation', 'opening', or 'closing'")
    return operations[key](arr)


def segment_kmeans(img: ArrayLike, k: int = 3) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    arr = _to_uint8(img)
    original_shape = arr.shape
    samples = arr.reshape((-1, 1 if arr.ndim == 2 else arr.shape[2])).astype(np.float32)
    k = min(int(k), len(samples))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    segmented = centers[labels.flatten()].reshape(original_shape)
    return np.clip(segmented, 0, 255).astype(np.uint8)


def segment_region_growing(
    img: ArrayLike,
    seed: Tuple[int, int] | Tuple[int, int, float],
) -> np.ndarray:
    gray = _to_gray_uint8(img)
    if len(seed) < 2:
        raise ValueError("seed must contain row and column coordinates")

    row, col = int(seed[0]), int(seed[1])
    if not (0 <= row < gray.shape[0] and 0 <= col < gray.shape[1]):
        raise ValueError("seed coordinates are outside the image")

    threshold = float(seed[2]) if len(seed) >= 3 else 12.0
    seed_value = float(gray[row, col])
    visited = np.zeros(gray.shape, dtype=bool)
    region = np.zeros(gray.shape, dtype=np.uint8)
    queue: deque[Tuple[int, int]] = deque([(row, col)])
    visited[row, col] = True

    while queue:
        y, x = queue.popleft()
        if abs(float(gray[y, x]) - seed_value) > threshold:
            continue
        region[y, x] = 255
        for ny in (y - 1, y, y + 1):
            for nx in (x - 1, x, x + 1):
                if ny == y and nx == x:
                    continue
                if 0 <= ny < gray.shape[0] and 0 <= nx < gray.shape[1] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    return region


def segment_contours(img: ArrayLike) -> np.ndarray:
    gray = _to_gray_uint8(img)
    binary = thresh_otsu(gray)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(gray)
    cv2.drawContours(result, contours, -1, 255, 1)
    return result
