from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


Callback = Callable[[Dict[str, Any]], None]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    control: str
    default: Any
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    step: int = 1
    choices: Optional[List[Any]] = None


class OddKernelSpinBox(QSpinBox):
    """Spin box constrained to common odd kernel sizes."""

    def stepBy(self, steps: int) -> None:
        values = [3, 5, 7, 9]
        current = self.value()
        try:
            index = values.index(current)
        except ValueError:
            index = min(range(len(values)), key=lambda i: abs(values[i] - current))
        self.setValue(values[max(0, min(len(values) - 1, index + steps))])


class ParameterPanelFactory:
    def __init__(self) -> None:
        self._active_widgets: Dict[str, QWidget] = {}

    def configure_panel(
        self,
        action_id: str,
        container_widget: QWidget,
        callback_on_change,
    ) -> dict:
        self._clear_container(container_widget)
        self._active_widgets = {}

        panel_layout = container_widget.layout()
        if panel_layout is None:
            panel_layout = QVBoxLayout()
            container_widget.setLayout(panel_layout)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(8)

        title = QLabel(self._title_for_action(action_id))
        title.setObjectName("paramsPanelTitle")
        title.setWordWrap(True)
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        panel_layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        panel_layout.addWidget(line)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop)
        panel_layout.addLayout(form)

        specs = self._specs_for_action(action_id)
        if not specs:
            empty = QLabel("Parametr talab qilinmaydi.")
            empty.setWordWrap(True)
            panel_layout.addWidget(empty)
            panel_layout.addStretch(1)
            return {}

        state: Dict[str, Any] = {}

        def emit_state() -> None:
            current = self._collect_state()
            if callable(callback_on_change):
                callback_on_change(current)

        for spec in specs:
            widget = self._create_control(spec, emit_state)
            self._active_widgets[spec.name] = widget
            state[spec.name] = self._value_from_widget(widget)
            form.addRow(QLabel(spec.label), widget)

        panel_layout.addStretch(1)
        if callable(callback_on_change):
            callback_on_change(dict(state))
        return state

    def _create_control(self, spec: ParameterSpec, emit_state: Callable[[], None]) -> QWidget:
        if spec.control == "odd_spin":
            widget = OddKernelSpinBox()
            widget.setRange(spec.minimum or 3, spec.maximum or 9)
            widget.setSingleStep(2)
            widget.setWrapping(False)
            widget.setValue(int(spec.default))
            widget.valueChanged.connect(lambda value: self._normalize_odd_spin(widget, emit_state))
            return widget

        if spec.control == "spin":
            widget = QSpinBox()
            widget.setRange(int(spec.minimum if spec.minimum is not None else 0), int(spec.maximum or 9999))
            widget.setSingleStep(int(spec.step))
            widget.setValue(int(spec.default))
            widget.valueChanged.connect(lambda _value: emit_state())
            return widget

        if spec.control == "double_spin":
            widget = QDoubleSpinBox()
            widget.setRange(float(spec.minimum if spec.minimum is not None else 0), float(spec.maximum or 9999))
            widget.setSingleStep(float(spec.step))
            widget.setDecimals(2)
            widget.setValue(float(spec.default))
            widget.valueChanged.connect(lambda _value: emit_state())
            return widget

        if spec.control == "slider":
            return self._slider_control(spec, emit_state)

        if spec.control == "choice":
            widget = QComboBox()
            for choice in spec.choices or []:
                widget.addItem(str(choice), choice)
            default_index = max(0, widget.findData(spec.default))
            widget.setCurrentIndex(default_index)
            widget.currentIndexChanged.connect(lambda _index: emit_state())
            return widget

        raise ValueError(f"Unsupported parameter control: {spec.control}")

    def _slider_control(self, spec: ParameterSpec, emit_state: Callable[[], None]) -> QWidget:
        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(8)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(spec.minimum if spec.minimum is not None else 0), int(spec.maximum or 255))
        slider.setSingleStep(int(spec.step))
        slider.setPageStep(max(1, int(spec.step) * 10))
        slider.setValue(int(spec.default))
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        value_label = QLabel(str(slider.value()))
        value_label.setMinimumWidth(32)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        wrapper_layout.addWidget(slider)
        wrapper_layout.addWidget(value_label)
        wrapper._parameter_slider = slider  # type: ignore[attr-defined]

        def on_change(value: int) -> None:
            value_label.setText(str(value))
            emit_state()

        slider.valueChanged.connect(on_change)
        return wrapper

    def _normalize_odd_spin(self, widget: QSpinBox, emit_state: Callable[[], None]) -> None:
        allowed = [3, 5, 7, 9]
        value = widget.value()
        if value not in allowed:
            normalized = min(allowed, key=lambda item: abs(item - value))
            widget.blockSignals(True)
            widget.setValue(normalized)
            widget.blockSignals(False)
        emit_state()

    def _collect_state(self) -> Dict[str, Any]:
        return {name: self._value_from_widget(widget) for name, widget in self._active_widgets.items()}

    def _value_from_widget(self, widget: QWidget) -> Any:
        if isinstance(widget, QComboBox):
            return widget.currentData()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QSpinBox):
            return widget.value()
        slider = getattr(widget, "_parameter_slider", None)
        if isinstance(slider, QSlider):
            return slider.value()
        return None

    def _clear_container(self, container_widget: QWidget) -> None:
        old_layout = container_widget.layout()
        if old_layout is not None:
            self._clear_layout(old_layout)

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            child_layout = item.layout()
            child_widget = item.widget()
            if child_layout is not None:
                self._clear_layout(child_layout)
                child_layout.deleteLater()
            if child_widget is not None:
                child_widget.setParent(None)
                child_widget.deleteLater()

    def _title_for_action(self, action_id: str) -> str:
        titles = {
            "actionMean": "Mean filtri sozlamalari",
            "actionMedian": "Median filtri sozlamalari",
            "actionGaussian": "Gaussian filtri sozlamalari",
            "actionBilateral": "Bilateral filtri sozlamalari",
            "actionCanny_edge_detection": "Canny kontur aniqlash",
            "actionThresholding": "Manual binarlash",
            "actionAniq_threshold": "Aniq threshold",
            "actionNearest_neihbor": "Nearest resize",
            "actionBilinear_interpolaton": "Bilinear resize",
            "actionBicubic_interpolation": "Bicubic resize",
            "actionLanczos_interpolation": "Lanczos resize",
            "actionImage_downsampling_upsampling": "Tasvir o'lchamini o'zgartirish",
            "actionRGB_HSV_Lab": "Rang modeli",
            "actionK_means": "K-means segmentatsiya",
            "actionQuantum_K_means": "Quantum K-means",
            "actionRegion_growing": "Region growing",
        }
        return titles.get(action_id, "Algoritm parametrlari")

    def _specs_for_action(self, action_id: str) -> List[ParameterSpec]:
        specs: Dict[str, List[ParameterSpec]] = {
            "actionMean": [self._kernel_spec()],
            "actionMedian": [self._kernel_spec()],
            "actionMedian_2": [self._kernel_spec()],
            "actionGaussian": [
                self._kernel_spec(),
                ParameterSpec("sigma", "Sigma", "spin", 0, 0, 50, 1),
            ],
            "actionGaussian_2": [
                self._kernel_spec(),
                ParameterSpec("sigma", "Sigma", "spin", 0, 0, 50, 1),
            ],
            "actionBilateral": [
                ParameterSpec("d", "Diametr", "odd_spin", 9, 3, 9, 2),
                ParameterSpec("sigma_color", "Rang sigma", "slider", 75, 0, 255, 1),
                ParameterSpec("sigma_space", "Fazo sigma", "slider", 75, 0, 255, 1),
            ],
            "actionCanny_edge_detection": [
                ParameterSpec("low_thresh", "Past threshold", "slider", 50, 0, 255, 1),
                ParameterSpec("high_thresh", "Yuqori threshold", "slider", 150, 0, 255, 1),
            ],
            "actionThresholding": [self._threshold_spec()],
            "actionAniq_threshold": [self._threshold_spec()],
            "actionDeteministik": [self._threshold_spec()],
            "actionPower_law_Gamma_correction": [
                ParameterSpec("gamma", "Gamma", "double_spin", 1.0, 0.1, 5.0, 0.1),
            ],
            "actionRGB_HSV_Lab": [
                ParameterSpec("color_space", "Rang fazosi", "choice", "hsv", choices=["hsv", "lab"]),
            ],
            "actionRangni_kodlash": [
                ParameterSpec("color_space", "Rang fazosi", "choice", "hsv", choices=["hsv", "lab"]),
            ],
            "actionNearest_neihbor": self._resize_specs(),
            "actionBilinear_interpolaton": self._resize_specs(),
            "actionBicubic_interpolation": self._resize_specs(),
            "actionLanczos_interpolation": self._resize_specs(),
            "actionImage_downsampling_upsampling": [
                ParameterSpec("mode", "Interpolatsiya", "choice", "bilinear", choices=["nearest", "bilinear", "bicubic", "lanczos"]),
                *self._resize_specs(),
            ],
            "actionResize_normalization": [
                ParameterSpec("mode", "Interpolatsiya", "choice", "bilinear", choices=["nearest", "bilinear", "bicubic", "lanczos"]),
                *self._resize_specs(),
            ],
            "actionResolution_normalization": [
                ParameterSpec("mode", "Interpolatsiya", "choice", "bilinear", choices=["nearest", "bilinear", "bicubic", "lanczos"]),
                *self._resize_specs(),
            ],
            "actionErosion": [self._kernel_spec(), self._kernel_shape_spec()],
            "actionDilation": [self._kernel_spec(), self._kernel_shape_spec()],
            "actionopening": [self._kernel_spec(), self._kernel_shape_spec()],
            "actionCloseing": [self._kernel_spec(), self._kernel_shape_spec()],
            "actionK_means": [ParameterSpec("k", "Klasterlar", "spin", 3, 2, 12, 1)],
            "actionQuantum_K_means": [ParameterSpec("clusters", "Klasterlar", "spin", 3, 2, 12, 1)],
            "actionRegion_growing": [
                ParameterSpec("threshold", "O'xshashlik threshold", "slider", 12, 0, 255, 1),
            ],
            "actionFRQI": [
                ParameterSpec("representation", "Kvant model", "choice", "frqi", choices=["frqi"]),
            ],
            "actionNEQR": [
                ParameterSpec("representation", "Kvant model", "choice", "neqr", choices=["neqr"]),
            ],
            "actionImage_representation_NEQR_FRQI": [
                ParameterSpec("representation", "Kvant model", "choice", "frqi", choices=["frqi", "neqr"]),
            ],
        }
        return specs.get(action_id, [])

    @staticmethod
    def _kernel_spec() -> ParameterSpec:
        return ParameterSpec("ksize", "Kernel o'lchami", "odd_spin", 3, 3, 9, 2)

    @staticmethod
    def _threshold_spec() -> ParameterSpec:
        return ParameterSpec("value", "Threshold qiymati", "slider", 127, 0, 255, 1)

    @staticmethod
    def _kernel_shape_spec() -> ParameterSpec:
        return ParameterSpec("kernel_shape", "Kernel shakli", "choice", "rect", choices=["rect", "ellipse", "cross"])

    @staticmethod
    def _resize_specs() -> List[ParameterSpec]:
        return [
            ParameterSpec("scale_x", "X masshtab", "double_spin", 1.0, 0.1, 4.0, 0.1),
            ParameterSpec("scale_y", "Y masshtab", "double_spin", 1.0, 0.1, 4.0, 0.1),
        ]
