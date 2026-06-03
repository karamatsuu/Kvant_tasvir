from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Mapping, Optional

import cv2
import numpy as np

try:
    import image_processor_classical as classical
    import image_processor_quantum as quantum
except ImportError:
    from . import image_processor_classical as classical
    from . import image_processor_quantum as quantum


Processor = Callable[[np.ndarray, Mapping[str, Any]], np.ndarray]


class ProcessingPipelineController:
    """Dispatch GUI action identifiers to pure image processing backends."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._router = self._build_router()

    def can_execute(self, action_id: str) -> bool:
        return action_id in self._router

    def execute_pipeline(
        self,
        action_id: str,
        input_image: np.ndarray,
        parameters: dict,
    ) -> np.ndarray:
        if input_image is None or not isinstance(input_image, np.ndarray):
            self.logger.warning("Pipeline skipped: missing or invalid input image for action %r", action_id)
            return input_image

        params: Mapping[str, Any] = parameters if isinstance(parameters, Mapping) else {}
        processor = self._router.get(action_id)
        if processor is None:
            self.logger.warning("Pipeline skipped: unmapped action_id %r", action_id)
            return input_image

        try:
            result = processor(input_image, params)
            if not isinstance(result, np.ndarray):
                self.logger.warning("Pipeline action %r returned non-array result; preserving input image", action_id)
                return input_image
            return result
        except Exception:
            self.logger.exception("Pipeline action %r failed; preserving input image", action_id)
            return input_image

    def _build_router(self) -> Dict[str, Processor]:
        return {
            # Color space transformations
            "actionRGB_Grayscale": lambda img, p: classical.rgb_to_grayscale(img),
            "actionRGB_HSV_Lab": self._rgb_hsv_lab,
            "actionRangni_kodlash": self._rgb_hsv_lab,
            "actionRGB_kanallarni_saqlash": self._rgb_channels_preview,
            "actionTasvirni_o_qish": lambda img, p: img,
            "actionQuantum_encoding": lambda img, p: quantum.simulate_frqi(img),
            "actionimage_processing": lambda img, p: quantum.quantum_edge_assisted_segmentation(img),
            "actionMeasurement": lambda img, p: quantum.simulate_neqr(img),
            # Interpolation and scaling
            "actionNearest_neihbor": lambda img, p: self._resize(img, p, "nearest"),
            "actionBilinear_interpolaton": lambda img, p: self._resize(img, p, "bilinear"),
            "actionBicubic_interpolation": lambda img, p: self._resize(img, p, "bicubic"),
            "actionLanczos_interpolation": lambda img, p: self._resize(img, p, "lanczos"),
            "actionImage_downsampling_upsampling": self._resize_from_params,
            "actionResize_normalization": self._resize_from_params,
            "actionResolution_normalization": self._resize_from_params,
            "actionCropping_resize": self._center_crop_resize,
            # Linear and nonlinear filters
            "actionMean": lambda img, p: classical.filter_mean(img, self._param(p, "ksize", 3)),
            "actionMean_2": lambda img, p: classical.filter_mean(img, self._param(p, "ksize", 3)),
            "actionMedian": lambda img, p: classical.filter_median(img, self._param(p, "ksize", 3)),
            "actionMedian_2": lambda img, p: classical.filter_median(img, self._param(p, "ksize", 3)),
            "actionGaussian": lambda img, p: classical.filter_gaussian(
                img,
                self._param(p, "ksize", 3),
                self._param(p, "sigma", 0.0),
            ),
            "actionGaussian_2": lambda img, p: quantum.quantum_denoising_circuit(
                classical.filter_gaussian(img, self._param(p, "ksize", 3), self._param(p, "sigma", 0.0))
            ),
            "actionBilateral": lambda img, p: classical.filter_bilateral(
                img,
                self._param(p, "d", 9),
                self._param(p, "sigma_color", 75.0),
                self._param(p, "sigma_space", 75.0),
            ),
            "actionFourier_transform": self._fourier_magnitude,
            "actionFourier_Transform": self._fourier_magnitude,
            "actionWavelet_denoising": lambda img, p: classical.filter_gaussian(img, 3, 0.0),
            "actionWavelet_Transform": lambda img, p: quantum.quantum_denoising_circuit(classical.filter_gaussian(img, 3, 0.0)),
            "actionWiener": lambda img, p: classical.filter_mean(img, 3),
            # Contrast adjustments
            "actionMin_Max": lambda img, p: classical.normalize_min_max(img),
            "actionLinear_contrast_stretching": lambda img, p: classical.contrast_linear_stretch(img),
            "actionLogarithmic_transform": lambda img, p: classical.transform_log(img),
            "actionPower_law_Gamma_correction": lambda img, p: classical.transform_gamma(
                img,
                self._param(p, "gamma", 1.0),
            ),
            "actionHistogram_equalization_HE": lambda img, p: classical.hist_equalization(img),
            "actionHistogram_equalization_QHE": lambda img, p: classical.hist_equalization(quantum.quantum_denoising_circuit(img)),
            "actionAdaptive_histogram_equalization_AHE": lambda img, p: classical.clahe_adjustment(
                img,
                self._param(p, "clip_limit", 2.0),
                self._param(p, "tile_grid_size", (8, 8)),
            ),
            "actionContrst_Limited_AHE_CLAHE": lambda img, p: classical.clahe_adjustment(
                img,
                self._param(p, "clip_limit", 2.0),
                self._param(p, "tile_grid_size", (8, 8)),
            ),
            "actionHigh_pass": self._high_pass,
            "actionUnsharp_masking": self._unsharp_mask,
            "actionamplitude_redistribution": lambda img, p: classical.transform_gamma(img, 0.75),
            "actionFourier_Transform_QGT": self._fourier_magnitude,
            "actioncontrast_enhancement_circuits": lambda img, p: classical.clahe_adjustment(img, 2.0, (8, 8)),
            "actionHybrid_qoantum_classical": lambda img, p: classical.hist_equalization(quantum.quantum_denoising_circuit(img)),
            # Intensity normalization
            "actionMin_Max_2": lambda img, p: classical.normalize_min_max(img),
            "actionZ_score": lambda img, p: classical.normalize_z_score(img),
            "actionL2": self._l2_normalize,
            "actionHistogram": lambda img, p: classical.hist_equalization(img),
            "actionDynamic_range_scaling_0_255_0_1": lambda img, p: np.asarray(img, dtype=np.float32) / 255.0,
            "actionContrast_stretching": lambda img, p: classical.contrast_linear_stretch(img),
            "actionAmplitude_normalization": self._l2_normalize,
            "actionProbability_normalization": self._probability_normalize,
            # Spatial edge extractors
            "actionSobel": lambda img, p: classical.edge_sobel(img),
            "actionPrewitt": lambda img, p: classical.edge_prewitt(img),
            "actionRoberts_cross": lambda img, p: classical.edge_roberts(img),
            "actionLaplacian_of_Gaussian_LoG": lambda img, p: classical.log_edge(img),
            "actionCanny_edge_detection": lambda img, p: classical.edge_canny(
                img,
                self._param(p, "low_thresh", 50),
                self._param(p, "high_thresh", 150),
            ),
            "actionKirsch_Schar": self._kirsch_edge,
            "actionGradient": lambda img, p: classical.edge_sobel(img),
            "actionGradient_operator": lambda img, p: quantum.quantum_edge_assisted_segmentation(img),
            "actionControlled_Not_CNOT": lambda img, p: classical.edge_roberts(img),
            "actionToffoli_gate_piksel_farqlari": lambda img, p: classical.edge_prewitt(img),
            "actionHybrid_quantum_classical": lambda img, p: quantum.quantum_edge_assisted_segmentation(classical.edge_sobel(img)),
            # Binarization
            "actionThresholding": lambda img, p: classical.thresh_manual(img, self._param(p, "value", 127)),
            "actionAniq_threshold": lambda img, p: classical.thresh_manual(img, self._param(p, "value", 127)),
            "actionDeteministik": lambda img, p: classical.thresh_manual(img, self._param(p, "value", 127)),
            "actionKodlashdan_oldin": lambda img, p: classical.thresh_otsu(img),
            "actionKodlashdan_oldin_2": lambda img, p: classical.thresh_otsu(img),
            "actionOtsu_thresholding": lambda img, p: classical.thresh_otsu(img),
            "actionAdaptive_thresholding": lambda img, p: classical.thresh_adaptive(img),
            "actionKvant_o_lcham": lambda img, p: classical.thresh_otsu(img),
            "actionEhtimoliy": self._probability_threshold,
            # Morphology
            "actionErosion": lambda img, p: self._morph(img, p, "erosion"),
            "actionDilation": lambda img, p: self._morph(img, p, "dilation"),
            "actionopening": lambda img, p: self._morph(img, p, "opening"),
            "actionCloseing": lambda img, p: self._morph(img, p, "closing"),
            "actionStructuring_element": lambda img, p: self._morph(img, p, "opening"),
            # Segmentation and token extraction
            "actionK_means": lambda img, p: classical.segment_kmeans(img, self._param(p, "k", 3)),
            "actionRegion_growing": lambda img, p: classical.segment_region_growing(
                img,
                self._region_seed(img, p),
            ),
            "actionRegion_splotting_merging": lambda img, p: classical.segment_kmeans(img, 4),
            "actionContour_detection_segmentation": lambda img, p: classical.segment_contours(img),
            "actionKontur": lambda img, p: classical.segment_contours(img),
            "actionFuzzy_C_means": lambda img, p: classical.segment_kmeans(img, self._param(p, "k", 3)),
            "actionGradient_va_topologiya_asosida_obektlarni_ajratish": lambda img, p: classical.segment_contours(
                classical.edge_sobel(img)
            ),
            "actionXaff_usullari": lambda img, p: classical.segment_contours(img),
            # Quantum representations
            "actionFRQI": lambda img, p: quantum.simulate_frqi(img),
            "actionFRQI_2": lambda img, p: quantum.simulate_frqi(img),
            "actionNEQR": lambda img, p: quantum.simulate_neqr(img),
            "actionNEQR_2": lambda img, p: quantum.simulate_neqr(img),
            "actionImage_representation_NEQR_FRQI": self._quantum_representation,
            # Quantum filtering and edge operations
            "actionQuantum_measurement_based_denoising": lambda img, p: quantum.quantum_denoising_circuit(img),
            "actionState_averaging": lambda img, p: quantum.quantum_denoising_circuit(img),
            "actionImage_denoising_circuits": lambda img, p: quantum.quantum_denoising_circuit(img),
            "actionQuantum_edge_assisted_segmentation": lambda img, p: quantum.quantum_edge_assisted_segmentation(img),
            "actionKonturni_kvantda_aniqlab_regionlarni_ajratish": lambda img, p: (
                quantum.quantum_edge_assisted_segmentation(img)
            ),
            "actionQuantum_Sobel": lambda img, p: quantum.quantum_edge_assisted_segmentation(img),
            "actionQFT_Quantum_Fourier_Transform": lambda img, p: quantum.quantum_edge_assisted_segmentation(img),
            # Hybrid quantum-classical clustering
            "actionQuantum_K_means": lambda img, p: quantum.quantum_kmeans_simulation(
                img,
                self._param(p, "clusters", self._param(p, "k", 3)),
            ),
            "actionQSVC_Quantum_Support_Vector_Clustering": lambda img, p: quantum.qsvc_clustering_simulation(img),
            "actionSuperpozitsiya_va_kvant_yadrolari": lambda img, p: quantum.qsvc_clustering_simulation(img),
            "actionQubit_amlitudalarini_kvant_holatida_taqsimlash": lambda img, p: quantum.qsvc_clustering_simulation(img),
            "actionOtsu": lambda img, p: classical.thresh_otsu(img),
        }

    @staticmethod
    def _param(parameters: Mapping[str, Any], key: str, default: Any) -> Any:
        value = parameters.get(key, default)
        return default if value is None else value

    def _resize(self, img: np.ndarray, parameters: Mapping[str, Any], mode: str) -> np.ndarray:
        return classical.resize_img(
            img,
            mode=mode,
            scale_x=float(self._param(parameters, "scale_x", 1.0)),
            scale_y=float(self._param(parameters, "scale_y", 1.0)),
        )

    def _resize_from_params(self, img: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
        return self._resize(
            img,
            parameters,
            str(self._param(parameters, "mode", "bilinear")),
        )

    def _rgb_hsv_lab(self, img: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
        color_space = str(self._param(parameters, "color_space", "hsv")).lower()
        if color_space == "lab":
            return classical.rgb_to_lab(img)
        return classical.rgb_to_hsv(img)

    def _morph(self, img: np.ndarray, parameters: Mapping[str, Any], op_type: str) -> np.ndarray:
        return classical.morph_operation(
            img,
            op_type=op_type,
            kernel_shape=str(self._param(parameters, "kernel_shape", "rect")),
            ksize=self._param(parameters, "ksize", 3),
        )

    def _region_seed(self, img: np.ndarray, parameters: Mapping[str, Any]) -> tuple:
        seed = parameters.get("seed")
        if seed is not None:
            return tuple(seed)

        height, width = img.shape[:2]
        threshold = self._param(parameters, "threshold", 12.0)
        return height // 2, width // 2, threshold

    def _quantum_representation(self, img: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
        representation = str(self._param(parameters, "representation", "frqi")).lower()
        if representation == "neqr":
            return quantum.simulate_neqr(img)
        return quantum.simulate_frqi(img)

    def _to_gray_uint8(self, img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 2:
            return np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[2] == 4:
            return cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    def _fourier_magnitude(self, img: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
        gray = self._to_gray_uint8(img).astype(np.float32)
        spectrum = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.log1p(np.abs(spectrum))
        return classical.contrast_linear_stretch(magnitude)

    def _high_pass(self, img: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
        arr = np.asarray(img)
        blurred = cv2.GaussianBlur(np.clip(arr, 0, 255).astype(np.uint8), (5, 5), 0.0)
        return cv2.subtract(np.clip(arr, 0, 255).astype(np.uint8), blurred)

    def _unsharp_mask(self, img: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
        arr = np.clip(np.asarray(img), 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(arr, (5, 5), 0.0)
        return cv2.addWeighted(arr, 1.6, blurred, -0.6, 0)

    def _l2_normalize(self, img: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm == 0:
            return np.zeros_like(arr, dtype=np.float32)
        return arr / norm

    def _probability_normalize(self, img: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32)
        total = float(np.sum(np.abs(arr)))
        if total == 0:
            return np.zeros_like(arr, dtype=np.float32)
        return np.abs(arr) / total

    def _probability_threshold(self, img: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
        gray = self._to_gray_uint8(img)
        threshold = float(np.mean(gray))
        return classical.thresh_manual(gray, threshold)

    def _kirsch_edge(self, img: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
        gray = self._to_gray_uint8(img)
        kernels = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32),
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32),
        ]
        responses = [cv2.filter2D(gray, cv2.CV_32F, kernel) for kernel in kernels]
        return classical.contrast_linear_stretch(np.max(responses, axis=0))

    def _rgb_channels_preview(self, img: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 2:
            return arr
        bgr = np.clip(arr[:, :, :3], 0, 255).astype(np.uint8)
        blue = np.dstack((bgr[:, :, 0], np.zeros_like(bgr[:, :, 0]), np.zeros_like(bgr[:, :, 0])))
        green = np.dstack((np.zeros_like(bgr[:, :, 1]), bgr[:, :, 1], np.zeros_like(bgr[:, :, 1])))
        red = np.dstack((np.zeros_like(bgr[:, :, 2]), np.zeros_like(bgr[:, :, 2]), bgr[:, :, 2]))
        return np.concatenate((blue, green, red), axis=1)

    def _center_crop_resize(self, img: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
        arr = np.asarray(img)
        height, width = arr.shape[:2]
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        cropped = arr[top : top + side, left : left + side]
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
