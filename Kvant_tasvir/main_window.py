from __future__ import annotations
from PyQt5.QtWidgets import QMainWindow
from main_window_ui import Ui_MainWindow
from menu_actions import FileMenuController

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


        # Statusbar — default holat
        self.statusBar().showMessage("Tayyor", 2000)

        # 1) Fayl menyusi controllerini ulash:
        # Hozircha rasmni 1-tabdagi label ga chiqaramiz: self.ui.label
        self.file_menu = FileMenuController(main_window=self, target_label=self.ui.label)
        self.file_menu.bind_actions(
            action_open=self.ui.actionOchish,
            action_save=self.ui.actionSaqlash,
            action_save_as=self.ui.actionQayta_saqlash,
            action_exit=self.ui.actionChiqish,
        )

        # ixtiyoriy: signallarni kuzatish (debug/kelajak)
        self.file_menu.image_loaded.connect(self._on_image_loaded)
        self.file_menu.image_saved.connect(self._on_image_saved)
        self.file_menu.error.connect(self._on_error)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Oyna o‘lchami o‘zgarsa, rasm qayta moslansin
        self.file_menu.on_target_resized()

    def _on_image_loaded(self, path: str):
        # keyin shu joyda QTextEdit ga ma'lumot chiqarishni ulaysiz
        pass

    def _on_image_saved(self, path: str):
        pass

    def _on_error(self, msg: str):
        pass
