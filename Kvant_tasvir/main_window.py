from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QAction, QApplication, QMainWindow, QMessageBox, QToolBar

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from main_window_ui import Ui_MainWindow
    from menu_actions import FileMenuController
    from parameter_gui_factory import ParameterPanelFactory
    from pipeline_controller import ProcessingPipelineController
except ImportError:
    from .main_window_ui import Ui_MainWindow
    from .menu_actions import FileMenuController
    from .parameter_gui_factory import ParameterPanelFactory
    from .pipeline_controller import ProcessingPipelineController


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.menuBar().setNativeMenuBar(False)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.progressBar = self.ui.progressBar

        self.current_action: Optional[str] = None
        self.current_parameters: Dict[str, Any] = {}
        self._configuring_panel = False
        self._last_raw_pixmap: Optional[QPixmap] = None
        self._last_processed_pixmap: Optional[QPixmap] = None
        self._last_classical_pixmap: Optional[QPixmap] = None
        self._last_quantum_pixmap: Optional[QPixmap] = None

        self.pipeline_controller = ProcessingPipelineController()
        self.parameter_factory = ParameterPanelFactory()

        self._build_toolbar()
        self._style_menu_bar()
        self.statusBar().showMessage("Tayyor", 2000)

        self.file_menu = FileMenuController(main_window=self, target_label=self.ui.label)
        self.file_menu.bind_actions(
            action_open=self.ui.actionOchish,
            action_save=self.ui.actionSaqlash,
            action_save_as=self.ui.actionQayta_saqlash,
            action_exit=self.ui.actionChiqish,
        )

        self.file_menu.image_loaded.connect(self._on_image_loaded)
        self.file_menu.image_saved.connect(self._on_image_saved)
        self.file_menu.error.connect(self._on_error)
        self.ui.algorithm_tree.algorithmSelected.connect(self._on_algorithm_selected)
        self.ui.mainSplitter.splitterMoved.connect(lambda *_: self._refresh_pixmap_sizes())
        self._bind_application_actions()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap_sizes()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Amallar", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QSize(16, 16))
        toolbar.addAction(self.ui.actionOchish)
        toolbar.addAction(self.ui.actionSaqlash)
        toolbar.addAction(self.ui.actionQayta_saqlash)
        toolbar.addSeparator()
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        toolbar.setStyleSheet(
            "QToolBar { border: none; padding: 2px 4px; spacing: 4px; }"
            "QToolButton { border: 1px solid transparent; border-radius: 5px; "
            "padding: 3px 8px; font-size: 12px; }"
            "QToolButton:hover { border-color: palette(mid); background: palette(button); }"
        )

    def _style_menu_bar(self) -> None:
        self.menuBar().setStyleSheet(
            "QMenuBar { background: palette(window); border-bottom: 1px solid palette(mid); "
            "font-size: 13px; padding: 0 4px; }"
            "QMenuBar::item { padding: 4px 10px; border-radius: 5px; }"
            "QMenuBar::item:selected { background: palette(highlight); "
            "color: palette(highlighted-text); }"
        )

    def _on_image_loaded(self, path: str) -> None:
        self.statusBar().showMessage(f"Ochildi: {path}", 3000)
        self._last_raw_pixmap = self.file_menu.state.pixmap
        self._last_processed_pixmap = None
        self._last_classical_pixmap = None
        self._last_quantum_pixmap = None
        self._update_raw_comparison_canvas()
        if self.current_action:
            self._apply_current_algorithm()

    def _on_image_saved(self, path: str) -> None:
        self.statusBar().showMessage(f"Saqlandi: {path}", 3000)

    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 5000)

    def pixmap_for_save(self) -> Optional[QPixmap]:
        current_label = self.ui.tabWidget.currentWidget().findChild(type(self.ui.label))
        if current_label is not None and current_label.pixmap() is not None:
            pixmap = current_label.pixmap()
            if pixmap is not None and not pixmap.isNull():
                if current_label is self.ui.label and self.file_menu.state.pixmap is not None:
                    return self.file_menu.state.pixmap
                return pixmap

        for pixmap in (
            self._last_processed_pixmap,
            self._last_quantum_pixmap,
            self._last_classical_pixmap,
            self.file_menu.state.pixmap,
        ):
            if pixmap is not None and not pixmap.isNull():
                return pixmap
        return None

    def _bind_application_actions(self) -> None:
        file_actions = {
            "actionOchish",
            "actionSaqlash",
            "actionQayta_saqlash",
            "actionChiqish",
            "actionToggleLeftPanel",
        }
        info_actions = self._info_action_messages()

        for action in self.findChildren(QAction):
            action_id = action.objectName()
            if not action_id or action_id in file_actions:
                continue
            try:
                action.triggered.disconnect()
            except TypeError:
                pass

            if self.pipeline_controller.can_execute(action_id):
                action.triggered.connect(lambda _checked=False, name=action_id: self._on_algorithm_selected(name))
            else:
                message = info_actions.get(action_id, "Bu bo'lim uchun alohida algoritm tanlang.")
                action.triggered.connect(
                    lambda _checked=False, title=action.text(), text=message: self._show_info(title, text)
                )

    def _info_action_messages(self) -> Dict[str, str]:
        return {
            "actionIshlab_chiqilgan_joyi": "Kvant tasvirlar bilan ishlash dasturi.",
            "actionMualliflar": "Mualliflar ma'lumoti loyiha hujjatlariga kiritiladi.",
            "actionKlassik_usullar_haqida": "Klassik usullar OpenCV asosidagi tasvir ishlov berish algoritmlarini ishga tushiradi.",
            "actionKvant_usullar_haqida": "Kvant usullar kvant tasvirlash va segmentlash jarayonlarining simulyatsiyalarini bajaradi.",
            "actionFoydalanish_qo_llanmasi": "Rasmni oching, chap panel yoki menyudan algoritm tanlang, parametrlarni sozlang va natijani saqlang.",
            "actionTasniflash_va_tanib_olish": "Tasniflash bo'limi uchun klassik yoki kvant usulni tanlang.",
            "actionTasniflash_va_tanib_olish_2": "Tasniflash bo'limi uchun klassik yoki kvant usulni tanlang.",
            "actionBelgilarni_ajratish_2": "Belgilarni ajratish bo'limidan aniq algoritm tanlang.",
            "actionMorfologik_ishlov_berish": "Morfologik ishlov berish bo'limidan erosion, dilation, opening yoki closing tanlang.",
            "actionKlassik_usul": "Klassik tasniflash moduli hozircha ko'rgazmali rejimda.",
            "actionKvant_usul": "Kvant tasniflash moduli hozircha ko'rgazmali rejimda.",
            "actionTurlarini_yozish": "Etalonli baholash moduli hozircha ko'rgazmali rejimda.",
            "actionBrisque": "BRISQUE baholash moduli hozircha ko'rgazmali rejimda.",
        }

    def _show_info(self, title: str, message: str) -> None:
        self.statusBar().showMessage(message, 5000)
        QMessageBox.information(self, title or "Ma'lumot", message)

    def _on_algorithm_selected(self, action_id: str) -> None:
        self.current_action = action_id
        self.statusBar().showMessage(action_id, 3000)

        self._configuring_panel = True
        try:
            self.current_parameters = self.parameter_factory.configure_panel(
                action_id,
                self.ui.params_panel,
                self._apply_current_algorithm,
            )
        finally:
            self._configuring_panel = False

        self._apply_current_algorithm()

    def _apply_current_algorithm(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        if parameters is not None:
            self.current_parameters = dict(parameters)
        if self._configuring_panel:
            return
        if not self.current_action:
            return

        source_pixmap = self.file_menu.state.pixmap
        if source_pixmap is None or source_pixmap.isNull():
            return

        input_image = self._pixmap_to_bgr_array(source_pixmap)
        if input_image is None:
            self.statusBar().showMessage("Rasm matritsaga aylantirilmadi.", 4000)
            return

        try:
            parameters = self._read_parameter_panel_state()

            self._set_progress(0, True)
            self._set_progress(50, True)
            result = self.pipeline_controller.execute_pipeline(
                self.current_action,
                input_image,
                parameters,
            )
            self._set_progress(100, True)

            self._last_processed_pixmap = self._array_to_pixmap(result)
            self._set_label_pixmap(self.ui.label_2, self._last_processed_pixmap)
            self._update_comparison_canvases(input_image, result, parameters)
            self.statusBar().showMessage(f"Bajarildi: {self.current_action}", 2500)
        finally:
            self._set_progress(0, False)

    def _read_parameter_panel_state(self) -> Dict[str, Any]:
        collector = getattr(self.parameter_factory, "_collect_state", None)
        if callable(collector):
            state = collector()
            if state:
                self.current_parameters = dict(state)
                return dict(state)
        return dict(self.current_parameters)

    def _update_raw_comparison_canvas(self) -> None:
        source_pixmap = self.file_menu.state.pixmap
        if source_pixmap is not None and not source_pixmap.isNull():
            self._last_raw_pixmap = source_pixmap
            self._set_label_pixmap(self.ui.label_11, self._last_raw_pixmap)

    def _update_comparison_canvases(
        self,
        input_image: np.ndarray,
        primary_result: np.ndarray,
        parameters: Dict[str, Any],
    ) -> None:
        raw_pixmap = self.file_menu.state.pixmap
        if raw_pixmap is not None and not raw_pixmap.isNull():
            self._set_label_pixmap(self.ui.label_11, raw_pixmap)

        classical_action, quantum_action = self._comparison_actions(self.current_action)
        if classical_action == self.current_action:
            classical_result = primary_result
        else:
            classical_result = self.pipeline_controller.execute_pipeline(
                classical_action,
                input_image,
                parameters,
            )

        if quantum_action == self.current_action:
            quantum_result = primary_result
        else:
            quantum_result = self.pipeline_controller.execute_pipeline(
                quantum_action,
                input_image,
                parameters,
            )

        self._last_classical_pixmap = self._array_to_pixmap(classical_result)
        self._last_quantum_pixmap = self._array_to_pixmap(quantum_result)
        self._set_label_pixmap(self.ui.label_10, self._last_classical_pixmap)
        self._set_label_pixmap(self.ui.label_12, self._last_quantum_pixmap)

    def _comparison_actions(self, action_id: Optional[str]) -> Tuple[str, str]:
        if not action_id:
            return "actionSobel", "actionQuantum_edge_assisted_segmentation"

        quantum_to_classical = {
            "actionQuantum_K_means": "actionK_means",
            "actionQSVC_Quantum_Support_Vector_Clustering": "actionK_means",
            "actionSuperpozitsiya_va_kvant_yadrolari": "actionK_means",
            "actionQuantum_edge_assisted_segmentation": "actionSobel",
            "actionKonturni_kvantda_aniqlab_regionlarni_ajratish": "actionSobel",
            "actionQuantum_Sobel": "actionSobel",
            "actionQFT_Quantum_Fourier_Transform": "actionSobel",
            "actionQuantum_measurement_based_denoising": "actionMedian",
            "actionState_averaging": "actionMedian",
            "actionImage_denoising_circuits": "actionMedian",
            "actionFRQI": "actionRGB_Grayscale",
            "actionFRQI_2": "actionRGB_Grayscale",
            "actionNEQR": "actionRGB_Grayscale",
            "actionNEQR_2": "actionRGB_Grayscale",
            "actionImage_representation_NEQR_FRQI": "actionRGB_Grayscale",
            "actionOtsu": "actionOtsu_thresholding",
        }
        classical_to_quantum = {
            "actionK_means": "actionQuantum_K_means",
            "actionContour_detection_segmentation": "actionQuantum_edge_assisted_segmentation",
            "actionKontur": "actionQuantum_edge_assisted_segmentation",
            "actionSobel": "actionQuantum_edge_assisted_segmentation",
            "actionPrewitt": "actionQuantum_edge_assisted_segmentation",
            "actionRoberts_cross": "actionQuantum_edge_assisted_segmentation",
            "actionLaplacian_of_Gaussian_LoG": "actionQuantum_edge_assisted_segmentation",
            "actionCanny_edge_detection": "actionQuantum_edge_assisted_segmentation",
            "actionMedian": "actionQuantum_measurement_based_denoising",
            "actionMean": "actionQuantum_measurement_based_denoising",
            "actionGaussian": "actionQuantum_measurement_based_denoising",
            "actionBilateral": "actionQuantum_measurement_based_denoising",
            "actionOtsu_thresholding": "actionOtsu",
        }

        if action_id in quantum_to_classical:
            return quantum_to_classical[action_id], action_id
        return action_id, classical_to_quantum.get(action_id, "actionQuantum_edge_assisted_segmentation")

    def _refresh_pixmap_sizes(self) -> None:
        if self._last_processed_pixmap is None:
            self.file_menu.on_target_resized()
            return

        if self.file_menu.state.pixmap is not None and not self.file_menu.state.pixmap.isNull():
            self._set_label_pixmap(self.ui.label, self.file_menu.state.pixmap)
        self._set_label_pixmap(self.ui.label_2, self._last_processed_pixmap)
        self._set_label_pixmap(self.ui.label_11, self._last_raw_pixmap)
        self._set_label_pixmap(self.ui.label_10, self._last_classical_pixmap)
        self._set_label_pixmap(self.ui.label_12, self._last_quantum_pixmap)

    def _set_progress(self, value: int, visible: bool) -> None:
        self.progressBar.setVisible(visible)
        self.progressBar.setValue(value)
        QApplication.processEvents()

    def _set_label_pixmap(self, label, pixmap: Optional[QPixmap]) -> None:
        if pixmap is None or pixmap.isNull():
            label.clear()
            return

        width = max(1, label.width())
        height = max(1, label.height())
        scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        label.setAlignment(Qt.AlignCenter)

    def _pixmap_to_bgr_array(self, pixmap: QPixmap) -> Optional[np.ndarray]:
        if pixmap is None or pixmap.isNull():
            return None

        image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        rgba = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4)).copy()
        return rgba[:, :, [2, 1, 0]]

    def _array_to_pixmap(self, array: np.ndarray) -> Optional[QPixmap]:
        if array is None or not isinstance(array, np.ndarray) or array.size == 0:
            return None

        display = self._array_to_display_uint8(array)
        if display.ndim == 2:
            height, width = display.shape
            qimage = QImage(
                np.ascontiguousarray(display).data,
                width,
                height,
                width,
                QImage.Format_Grayscale8,
            ).copy()
            return QPixmap.fromImage(qimage)

        height, width, channels = display.shape
        if channels == 3:
            rgb = np.ascontiguousarray(display[:, :, ::-1])
            qimage = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888).copy()
            return QPixmap.fromImage(qimage)

        if channels == 4:
            rgba = np.ascontiguousarray(display[:, :, [2, 1, 0, 3]])
            qimage = QImage(rgba.data, width, height, width * 4, QImage.Format_RGBA8888).copy()
            return QPixmap.fromImage(qimage)

        collapsed = self._normalize_to_uint8(np.mean(display, axis=2))
        return self._array_to_pixmap(collapsed)

    def _array_to_display_uint8(self, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array)
        if arr.ndim == 3 and arr.shape[2] == 8:
            packed = np.packbits(arr.astype(np.uint8), axis=2)
            return packed[:, :, 0]
        if arr.dtype == np.uint8 and arr.ndim in (2, 3):
            return arr
        return self._normalize_to_uint8(arr)

    def _normalize_to_uint8(self, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float32)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        min_value = float(np.min(arr))
        max_value = float(np.max(arr))
        if max_value <= min_value:
            return np.zeros(arr.shape, dtype=np.uint8)
        return np.clip((arr - min_value) * 255.0 / (max_value - min_value), 0, 255).astype(np.uint8)
