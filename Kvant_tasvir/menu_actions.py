# menu_actions.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel, QMainWindow


@dataclass
class FileState:
    current_path: Optional[str] = None
    pixmap: Optional[QPixmap] = None


class FileMenuController(QObject):
    """
    UI ichiga aralashmaydi.
    Faqat UI dagi ACTION larni slotlarga ulaydi va natijani signal bilan beradi.
    """
    image_loaded = pyqtSignal(str)   # path
    image_saved = pyqtSignal(str)    # path
    error = pyqtSignal(str)

    def __init__(self, main_window: QMainWindow, target_label: QLabel):
        super().__init__(main_window)
        self._mw = main_window
        self._label = target_label
        self.state = FileState()

    # -------- Wiring (main_window.py dan chaqiriladi) --------
    def bind_actions(self, action_open, action_save, action_save_as, action_exit):
        action_open.triggered.connect(self.open_image_dialog)
        action_save.triggered.connect(self.save_image)
        action_save_as.triggered.connect(self.save_image_as_dialog)
        action_exit.triggered.connect(self.exit_app)

    # -------- Core: Open / Save / Save As / Exit --------
    def open_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self._mw,
            "Rasmni tanlang",
            "",
            "Rasm fayllari (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff);;Barcha fayllar (*.*)"
        )
        if not path:
            return

        pix = QPixmap(path)
        if pix.isNull():
            msg = "Rasm ochilmadi (format yoki fayl buzilgan bo‘lishi mumkin)."
            self._show_error(msg)
            return

        self.state.current_path = path
        self.state.pixmap = pix
        self._render_pixmap()
        self.image_loaded.emit(path)
        self._mw.statusBar().showMessage(f"Ochildi: {os.path.basename(path)}", 4000)

    def save_image(self):
        """
        'Saqlash' — agar avval ochilgan rasm bo‘lsa, o‘sha faylning ustiga yozadi.
        Siz hozircha rasmni o‘zgartirmayotganingiz uchun bu 'copy' sifatida ishlaydi.
        """
        if not self._ensure_pixmap():
            return

        if not self.state.current_path:
            # Hech qachon ochilmagan bo‘lsa — Save As kabi ishlaydi
            self.save_image_as_dialog()
            return

        ok = self.state.pixmap.save(self.state.current_path)
        if not ok:
            self._show_error("Saqlab bo‘lmadi. Papkaga ruxsat yoki yo‘l muammosi bo‘lishi mumkin.")
            return

        self.image_saved.emit(self.state.current_path)
        self._mw.statusBar().showMessage(f"Saqlandi: {os.path.basename(self.state.current_path)}", 4000)

    def save_image_as_dialog(self):
        if not self._ensure_pixmap():
            return

        path, _ = QFileDialog.getSaveFileName(
            self._mw,
            "Qayerga saqlaymiz?",
            self.state.current_path or "",
            "PNG (*.png);;JPG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff)"
        )
        if not path:
            return

        ok = self.state.pixmap.save(path)
        if not ok:
            self._show_error("Qayta saqlab bo‘lmadi. Fayl nomi/format yoki ruxsat muammosi bo‘lishi mumkin.")
            return

        self.state.current_path = path
        self.image_saved.emit(path)
        self._mw.statusBar().showMessage(f"Qayta saqlandi: {os.path.basename(path)}", 4000)

    def exit_app(self):
        self._mw.close()

    # -------- Helpers --------
    def _ensure_pixmap(self) -> bool:
        if self.state.pixmap is None or self.state.pixmap.isNull():
            self._show_error("Hali rasm ochilmagan. Avval 'Fayl → Ochish' qiling.")
            return False
        return True

    def _render_pixmap(self):
        """Label o‘lchamiga moslab ko‘rsatadi (aspect ratio saqlanadi)."""
        if not self.state.pixmap or self.state.pixmap.isNull():
            return

        w = max(1, self._label.width())
        h = max(1, self._label.height())
        scaled = self.state.pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._label.setPixmap(scaled)
        self._label.setAlignment(Qt.AlignCenter)

    def on_target_resized(self):
        """main_window resize bo‘lganda chaqiriladi."""
        if self.state.pixmap and not self.state.pixmap.isNull():
            self._render_pixmap()

    def _show_error(self, msg: str):
        self.error.emit(msg)
        QMessageBox.critical(self._mw, "Xatolik", msg)
        self._mw.statusBar().showMessage(msg, 5000)
