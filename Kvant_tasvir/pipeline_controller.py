from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Mapping, Optional

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
            # Interpolation and scaling
            "actionNearest_neihbor": lambda img, p: self._resize(img, p, "nearest"),
            "actionBilinear_interpolaton": lambda img, p: self._resize(img, p, "bilinear"),
            "actionBicubic_interpolation": lambda img, p: self._resize(img, p, "bicubic"),
            "actionLanczos_interpolation": lambda img, p: self._resize(img, p, "lanczos"),
            "actionImage_downsampling_upsampling": self._resize_from_params,
            "actionResize_normalization": self._resize_from_params,
            "actionResolution_normalization": self._resize_from_params,
            # Linear and nonlinear filters
            "actionMean": lambda img, p: classical.filter_mean(img, self._param(p, "ksize", 3)),
            "actionMedian": lambda img, p: classical.filter_median(img, self._param(p, "ksize", 3)),
            "actionGaussian": lambda img, p: classical.filter_gaussian(
                img,
                self._param(p, "ksize", 3),
                self._param(p, "sigma", 0.0),
            ),
            "actionBilateral": lambda img, p: classical.filter_bilateral(
                img,
                self._param(p, "d", 9),
                self._param(p, "sigma_color", 75.0),
                self._param(p, "sigma_space", 75.0),
            ),
            # Contrast adjustments
            "actionMin_Max": lambda img, p: classical.normalize_min_max(img),
            "actionLinear_contrast_stretching": lambda img, p: classical.contrast_linear_stretch(img),
            "actionLogarithmic_transform": lambda img, p: classical.transform_log(img),
            "actionPower_law_Gamma_correction": lambda img, p: classical.transform_gamma(
                img,
                self._param(p, "gamma", 1.0),
            ),
            "actionHistogram_equalization_HE": lambda img, p: classical.hist_equalization(img),
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
            # Intensity normalization
            "actionMin_Max_2": lambda img, p: classical.normalize_min_max(img),
            "actionZ_score": lambda img, p: classical.normalize_z_score(img),
            "actionContrast_stretching": lambda img, p: classical.contrast_linear_stretch(img),
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
            # Binarization
            "actionThresholding": lambda img, p: classical.thresh_manual(img, self._param(p, "value", 127)),
            "actionAniq_threshold": lambda img, p: classical.thresh_manual(img, self._param(p, "value", 127)),
            "actionDeteministik": lambda img, p: classical.thresh_manual(img, self._param(p, "value", 127)),
            "actionOtsu_thresholding": lambda img, p: classical.thresh_otsu(img),
            "actionAdaptive_thresholding": lambda img, p: classical.thresh_adaptive(img),
            # Morphology
            "actionErosion": lambda img, p: self._morph(img, p, "erosion"),
            "actionDilation": lambda img, p: self._morph(img, p, "dilation"),
            "actionopening": lambda img, p: self._morph(img, p, "opening"),
            "actionCloseing": lambda img, p: self._morph(img, p, "closing"),
            # Segmentation and token extraction
            "actionK_means": lambda img, p: classical.segment_kmeans(img, self._param(p, "k", 3)),
            "actionRegion_growing": lambda img, p: classical.segment_region_growing(
                img,
                self._region_seed(img, p),
            ),
            "actionContour_detection_segmentation": lambda img, p: classical.segment_contours(img),
            "actionKontur": lambda img, p: classical.segment_contours(img),
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
