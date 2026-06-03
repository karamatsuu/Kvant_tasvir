from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


ArrayLike = np.ndarray
EPS = 1e-8


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
    arr = np.where(np.isfinite(arr), arr, 0.0)
    min_val = float(np.min(arr)) if arr.size else 0.0
    max_val = float(np.max(arr)) if arr.size else 0.0
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


def _to_bgr_uint8(img: ArrayLike) -> np.ndarray:
    arr = _to_uint8(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return arr


def _stretch(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)


def _coordinate_grid(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return yy, xx


def simulate_frqi(img: ArrayLike) -> np.ndarray:
    """Return an H x W x 3 map of normalized row, column, and FRQI intensity angle."""
    gray = _to_gray_uint8(img).astype(np.float32) / 255.0
    height, width = gray.shape
    yy, xx = _coordinate_grid(height, width)
    theta = gray * (np.pi / 2.0)
    return np.stack((yy, xx, theta.astype(np.float32)), axis=-1).astype(np.float32)


def simulate_neqr(img: ArrayLike) -> np.ndarray:
    """Return H x W x 8 uint8 bit planes representing NEQR grayscale intensity qubits."""
    gray = _to_gray_uint8(img)
    bits = ((gray[:, :, None] >> np.arange(7, -1, -1, dtype=np.uint8)) & 1).astype(np.uint8)
    return bits


def quantum_edge_assisted_segmentation(img: ArrayLike) -> np.ndarray:
    gray = _to_gray_uint8(img).astype(np.float32) / 255.0
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    phase = np.angle(spectrum)
    magnitude = np.log1p(np.abs(spectrum))

    phase_variation = cv2.Sobel(phase.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    phase_variation += cv2.Sobel(phase.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

    height, width = gray.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius /= float(np.max(radius) + EPS)

    qft_boundary_score = np.abs(phase_variation) * radius + magnitude * radius
    spatial_edges = cv2.Canny(_stretch(gray), 50, 150).astype(np.float32) / 255.0
    score = _stretch(qft_boundary_score) / 255.0
    combined = _stretch((0.65 * score) + (0.35 * spatial_edges))
    _, segmented = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return segmented


def quantum_denoising_circuit(img: ArrayLike) -> np.ndarray:
    arr = _to_uint8(img).astype(np.float32) / 255.0
    probabilities = arr * arr
    averaged_probabilities = cv2.GaussianBlur(probabilities, (3, 3), 0.0)
    measured_amplitudes = np.sqrt(np.clip(averaged_probabilities, 0.0, 1.0))
    return np.clip(measured_amplitudes * 255.0, 0, 255).astype(np.uint8)


def _pixel_state_features(img: ArrayLike) -> Tuple[np.ndarray, Tuple[int, ...]]:
    bgr = _to_bgr_uint8(img).astype(np.float32) / 255.0
    height, width = bgr.shape[:2]
    gray = cv2.cvtColor((bgr * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    yy, xx = _coordinate_grid(height, width)
    features = np.dstack((bgr, gray, yy, xx, np.ones_like(gray))).reshape(-1, 7)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, EPS), bgr.shape


def quantum_kmeans_simulation(img: ArrayLike, clusters: int = 3) -> np.ndarray:
    if clusters <= 0:
        raise ValueError("clusters must be positive")

    states, image_shape = _pixel_state_features(img)
    pixel_count = states.shape[0]
    clusters = min(int(clusters), pixel_count)
    init_idx = np.linspace(0, pixel_count - 1, clusters, dtype=np.int64)
    centers = states[init_idx].copy()

    labels = np.zeros(pixel_count, dtype=np.int32)
    for _ in range(12):
        overlaps = np.square(np.abs(states @ centers.T))
        new_labels = np.argmax(overlaps, axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(clusters):
            members = states[labels == cluster_id]
            if len(members) == 0:
                continue
            center = np.mean(members, axis=0)
            centers[cluster_id] = center / max(float(np.linalg.norm(center)), EPS)

    bgr = _to_bgr_uint8(img).reshape(-1, 3).astype(np.float32)
    color_centers = np.zeros((clusters, 3), dtype=np.float32)
    for cluster_id in range(clusters):
        members = bgr[labels == cluster_id]
        if len(members) == 0:
            color_centers[cluster_id] = bgr[init_idx[cluster_id]]
        else:
            color_centers[cluster_id] = np.mean(members, axis=0)

    segmented = color_centers[labels].reshape(image_shape)
    return np.clip(segmented, 0, 255).astype(np.uint8)


def qsvc_clustering_simulation(img: ArrayLike) -> np.ndarray:
    gray = _to_gray_uint8(img).astype(np.float32) / 255.0
    height, width = gray.shape
    yy, xx = _coordinate_grid(height, width)

    feature_map = np.dstack(
        (
            np.cos(np.pi * gray / 2.0),
            np.sin(np.pi * gray / 2.0),
            np.cos(np.pi * yy),
            np.sin(np.pi * xx),
            np.ones_like(gray),
        )
    ).reshape(-1, 5)
    feature_map /= np.maximum(np.linalg.norm(feature_map, axis=1, keepdims=True), EPS)

    quantiles = np.quantile(gray, [0.15, 0.5, 0.85])
    support_vectors = []
    flat_gray = gray.reshape(-1)
    for value in quantiles:
        support_vectors.append(feature_map[int(np.argmin(np.abs(flat_gray - value)))])
    support_vectors = np.asarray(support_vectors, dtype=np.float32)

    kernel_values = np.square(np.abs(feature_map @ support_vectors.T))
    labels = np.argmax(kernel_values, axis=1).reshape(height, width).astype(np.uint8)

    label_values = np.array([0, 127, 255], dtype=np.uint8)
    clustered = label_values[labels]
    return cv2.medianBlur(clustered, 3)
