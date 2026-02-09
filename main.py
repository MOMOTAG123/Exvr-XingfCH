import sys
import os
import warnings
import time
warnings.filterwarnings("ignore")
try:
    import pyuac
    if not pyuac.isUserAdmin():
        pyuac.runAsAdmin()
        sys.exit(0)
except ImportError:
    import ctypes
    if not ctypes.windll.shell32.IsUserAnAdmin():
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        sys.exit(0)
import numpy as np
import cv2
from copy import deepcopy
from ctypes import windll
import winreg
import shutil
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget,
    QLineEdit, QComboBox, QHBoxLayout, QFrame, QCheckBox, QSlider, QMessageBox,
    QDialog, QScrollArea, QGridLayout, QSizePolicy, QSplashScreen
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap, QDoubleValidator, QFont, QPainter, QColor, QMovie
try:
    import utils.tracking
    from utils.actions import reset_head, reset_eye, reset_hand
    import utils.globals as g
    from utils.data import setup_data, save_data
    from utils.hotkeys import stop_hotkeys, apply_hotkeys
    from tracker.face.face import draw_face_landmarks
    from tracker.face.tongue import draw_tongue_position
    from tracker.hand.hand import draw_hand_landmarks
    from tracker.pose.pose import draw_pose_landmarks
    from tracker.controller.controller import ControllerApp
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError:
        def enumerate_cameras(backend):
            class CameraDevice:
                def __init__(self, index, name):
                    self.index = index
                    self.name = name
            devices = []
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    devices.append(CameraDevice(i, f"Camera {i}"))
                    cap.release()
            return devices
except ImportError as e:
    QMessageBox.critical(None, "错误", f"缺少必要文件：{str(e)}\n请确保utils和tracker文件夹存在于当前目录")
    sys.exit(1)
def init_globals():
    default_config = {
        "Version": "1.0.0",
        "Setting": {
            "flip_x": False,
            "flip_y": False,
            "camera_ip": "",
            "camera_width": 640,
            "camera_height": 480,
            "camera_fps": 60,
            "priority": "NORMAL_PRIORITY_CLASS",
            "only_ingame": False,
            "only_ingame_game": "VRChat"
        },
        "Tracking": {
            "Head": {"enable": True},
            "Face": {"enable": True},
            "Tongue": {"enable": False},
            "Hand": {
                "enable": True,
                "enable_hand_down": False,
                "enable_finger_action": False,
                "x_scalar": 1.0,
                "y_scalar": 1.0,
                "z_scalar": 1.0
            },
            "Pose": {"enable": False},
            "LeftController": {
                "enable": False,
                "base_x": 0.0,
                "base_y": 0.0,
                "base_z": 0.0,
                "length": 1.0
            },
            "RightController": {
                "enable": False,
                "base_x": 0.0,
                "base_y": 0.0,
                "base_z": 0.0,
                "length": 1.0
            }
        },
        "Mouse": {
            "enable": False,
            "scalar_x": 1.0,
            "scalar_y": 1.0,
            "dx": 0.5
        }
    }
    default_face_data = {
        "BlendShapes": [
            {"k": "None", "v": 0.0, "s": 0.0, "w": 0.0, "max": 1.0, "e": True},
            {"k": "EyeBlinkLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "EyeBlinkRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "EyeLookDownLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "EyeLookDownRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "EyeLookInLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "EyeLookInRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "EyeLookOutLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "EyeLookOutRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "EyeLookUpLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "EyeLookUpRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "EyeSquintLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "EyeSquintRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "EyeWideLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "EyeWideRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "JawForward", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "JawLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "JawRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "JawOpen", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthClose", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthFunnel", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthPucker", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthSmileLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthSmileRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthFrownLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthFrownRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthDimpleLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthDimpleRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthStretchLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthStretchRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthRollLower", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthRollUpper", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthShrugLower", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthShrugUpper", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthPressLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthPressRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthLowerDownLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthLowerDownRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthUpperUpLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "MouthUpperUpRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "BrowDownLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "BrowDownRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "BrowInnerUp", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "BrowOuterUpLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "BrowOuterUpRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "CheekPuff", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "CheekSquintLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "CheekSquintRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "NoseSneerLeft", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "NoseSneerRight", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True},
            {"k": "TongueOut", "v": 0.0, "s": 0.0, "w": 1.0, "max": 1.0, "e": True}
        ]
    }
    if not hasattr(g, 'config'):
        g.config = default_config
    if not hasattr(g, 'default_data'):
        g.default_data = default_face_data
    if not hasattr(g, 'controller'):
        class DummyController:
            def __init__(self):
                self.left_hand = type('obj', (object,), {'force_enable': False})
                self.right_hand = type('obj', (object,), {'force_enable': False})
        g.controller = DummyController()
    def save_configs():
        config_path = os.path.join(os.getenv("APPDATA"), "ExVR", "config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(g.config, f, ensure_ascii=False, indent=4)
    def update_configs():
        config_path = os.path.join(os.getenv("APPDATA"), "ExVR", "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                g.config = json.load(f)
    if not hasattr(g, 'save_configs'):
        g.save_configs = save_configs
    if not hasattr(g, 'update_configs'):
        g.update_configs = update_configs
init_globals()
MODERN_STYLE_SHEET = """
QWidget {
    background-color: #F5F7FA;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 12px;
    color: #333333;
}

QMainWindow {
    background-color: #FFFFFF;
    border-radius: 8px;
}

QWidget#LeftMenu {
    background-color: #2C3E50;
    border-radius: 8px 0 0 8px;
}

QWidget#ContentArea {
    background-color: #FFFFFF;
    border-radius: 0 8px 8px 0;
}

QPushButton#MenuItem {
    background-color: transparent;
    border: none;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 14px;
    font-weight: 500;
    color: #BDC3C7;
    text-align: left;
    margin: 4px 8px;
}

QPushButton#MenuItem:hover {
    color: #FFFFFF;
}

QPushButton#MenuItem:pressed {
    color: #FFFFFF;
}

QPushButton#MenuItem:checked {
    color: #FFFFFF;
}

QPushButton {
    background-color: #ECF0F1;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    outline: none;
}

QPushButton:hover {
    background-color: #E0E6E8;
}

QPushButton:pressed {
    background-color: #D5DDE2;
}

QPushButton#PrimaryButton {
    background-color: #3498DB;
    color: white;
}

QPushButton#PrimaryButton:hover {
    background-color: #2980B9;
}

QCheckBox {
    spacing: 8px;
    outline: none;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    background-color: #ECF0F1;
    border: 1px solid #BDC3C7;
}

QCheckBox::indicator:checked {
    background-color: #3498DB;
    border: 1px solid #3498DB;
    image: url(:/qt-project.org/styles/commonstyle/images/checkbox_check.png);
}

QSlider::groove:horizontal {
    height: 8px;
    background-color: #E0E6E8;
    border-radius: 4px;
    margin: 4px 0;
}

QSlider::handle:horizontal {
    width: 24px;
    height: 24px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4A90E2, stop:1 #357ABD);
    border-radius: 12px;
    margin: -8px 0;
    outline: none;
    border: 2px solid #FFFFFF;
}

QSlider::handle:horizontal:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5A9FF2, stop:1 #458AEB);
}

QSlider::handle:horizontal:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2D68C4, stop:1 #1A57A8);
}

QComboBox {
    background-color: #ECF0F1;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    outline: none;
}

QComboBox:hover {
    background-color: #E0E6E8;
}

QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #BDC3C7;
    border-radius: 6px;
    padding: 8px 12px;
    outline: none;
}

QLineEdit:focus {
    border-color: #3498DB;
}

QLabel#StatusLabel {
    font-weight: 600;
    font-size: 13px;
}

QFrame#Separator {
    background-color: #E0E6E8;
    height: 1px;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QWidget#ContentArea {
    background-color: #FFFFFF;
    border-radius: 0 8px 8px 0;
}

QScrollArea {
    background-color: #FFFFFF;
}
QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #DEE2E6;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
    outline: none;
}
QPushButton:hover {
    background-color: #F8F9FA;
    border-color: #CED4DA;
}
QPushButton:pressed {
    background-color: #E9ECEF;
}
QPushButton#PrimaryButton {
    background-color: #0078D7;
    background: qlineargradient(135deg, #0078D7 0%, #005A9E 100%);
    color: white;
    border: none;
}
QPushButton#PrimaryButton:hover {
    background: qlineargradient(135deg, #008EFB 0%, #006BC5 100%);
}
QPushButton#PrimaryButton:pressed {
    background: qlineargradient(135deg, #006BC5 0%, #005A9E 100%);
}
QCheckBox {
    spacing: 10px;
    outline: none;
}
QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    background-color: #FFFFFF;
    border: 2px solid #DEE2E6;
}
QCheckBox::indicator:hover {
    border-color: #0078D7;
}
QCheckBox::indicator:checked {
    background-color: #0078D7;
    border-color: #0078D7;
    image: url(:/qt-project.org/styles/commonstyle/images/checkbox_check.png);
}
QSlider::groove:horizontal {
    height: 6px;
    background-color: #E9ECEF;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 20px;
    height: 20px;
    background-color: #FFFFFF;
    border: 2px solid #0078D7;
    border-radius: 50%;
    margin: -7px 0;
    outline: none;
}
QSlider::handle:horizontal:hover {
    border-color: #008EFB;
}
QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #DEE2E6;
    border-radius: 8px;
    padding: 10px 16px;
    outline: none;
}
QComboBox:hover {
    border-color: #CED4DA;
}
QComboBox:focus {
    border-color: #0078D7;
}
QComboBox::drop-down {
    border: none;
    border-left: 1px solid #DEE2E6;
    border-radius: 0 8px 8px 0;
    width: 30px;
}
QComboBox::down-arrow {
    image: url(:/qt-project.org/styles/commonstyle/images/down_arrow.png);
    width: 16px;
    height: 16px;
}
QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #DEE2E6;
    border-radius: 8px;
    padding: 10px 16px;
    outline: none;
}
QLineEdit:focus {
    border-color: #0078D7;
}
QLabel#StatusLabel {
    font-weight: 600;
    font-size: 14px;
    color: #0078D7;
}
QFrame#Separator {
    background-color: #E9ECEF;
    height: 2px;
    border-radius: 1px;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    width: 8px;
    background: #F8F9FA;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #CED4DA;
    border-radius: 4px;
    transition: background 0.2s ease-in-out;
}
QScrollBar::handle:vertical:hover {
    background: #ADB5BD;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    height: 8px;
    background: #F8F9FA;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #CED4DA;
    border-radius: 4px;
    transition: background 0.2s ease-in-out;
}
QScrollBar::handle:horizontal:hover {
    background: #ADB5BD;
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
}
"""
class LoadingScreen(QSplashScreen):
    def __init__(self):
        pixmap = QPixmap(400, 300)
        pixmap.fill(QColor(30, 30, 30))
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
        painter.drawText(pixmap.rect().adjusted(0, -50, 0, 0), Qt.AlignCenter, "ExVR 虚拟现实体验")
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(pixmap.rect().adjusted(0, 50, 0, 0), Qt.AlignCenter, "正在初始化...(小提示：按住窗口可移动)")
        painter.end()
        super().__init__(pixmap, Qt.WindowStaysOnTopHint)
        self.status_label = QLabel("", self)
        self.status_label.setStyleSheet("color: white; font-size: 12px;")
        self.status_label.move(150, 220)
        self.loading_animation = QLabel(self)
        self.loading_animation.setGeometry(190, 240, 20, 20)
        self.loading_animation.setStyleSheet("color: #0078D7; font-size: 16px;")
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(100)
        self.animation_states = ['|', '/', '-', '|']
        self.current_state = 0
        self.dragging = False
        self.drag_start_position = QPoint()
    def update_animation(self):
        self.loading_animation.setText(self.animation_states[self.current_state])
        self.current_state = (self.current_state + 1) % len(self.animation_states)
    def update_status(self, text):
        self.status_label.setText(text)
        QApplication.processEvents()
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    def mouseMoveEvent(self, event):
        if Qt.LeftButton & event.buttons() and self.dragging:
            new_position = event.globalPos() - self.drag_start_position
            self.move(new_position)
            event.accept()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            event.accept()
class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(QImage)
    def __init__(self, source,width=640, height=480, fps=60):
        super().__init__()
        self.source = source
        self.video_capture = None
        self.is_running = True
        self.show_image = False
        self.is_using_web_controller = False
        self.controller_thread = None
        try:
            self.tracker = utils.tracking.Tracker()
        except:
            self.tracker = type('obj', (object,), {'process_frames': lambda x: None, 'stop': lambda: None})
        if width < 640 or height < 480:
            aspect_ratio = width / height
            if aspect_ratio == 1280 / 720:
                self.width, self.height = 1280, 720
            elif aspect_ratio == 640 / 480:
                self.width, self.height = 640, 480
            else:
                self.width, self.height = width, height
        else:
            self.width, self.height = width, height
        self.fps = fps
    def run(self):
        try:
            if self.is_using_web_controller:
                rgb_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                font_path = "C:/Windows/Fonts/simhei.ttf"
                font_scale = 1.5
                font_thickness = 3
                text_color = (255, 255, 255)
                while self.is_running:
                    rgb_image.fill(0)
                    if self.controller_thread:
                        try:
                            ip_addresses = self.controller_thread.get_server_ip()
                            if ip_addresses:
                                from PIL import Image, ImageDraw, ImageFont
                                pil_image = Image.fromarray(rgb_image)
                                draw = ImageDraw.Draw(pil_image)
                                for i, (_, ip) in enumerate(ip_addresses):
                                    try:
                                        font = ImageFont.truetype(font_path, 20)
                                        draw.text(
                                            (10, 10 + i*25),
                                            f"连接IP: {ip}",
                                            font=font,
                                            fill=(0, 255, 0)
                                        )
                                    except Exception as e:
                                        print(f"绘制IP出错: {e}")
                                        cv2.putText(
                                            rgb_image,
                                            f"IP: {ip}",
                                            (10, 30 + i*25),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1.0,
                                            (0, 255, 0),
                                            3,
                                            cv2.LINE_AA
                                        )
                                rgb_image = np.array(pil_image)
                        except Exception as e:
                            print(f"绘制IP信息出错: {e}")
                            pass
                    try:
                        control_params = self.controller_thread.current_control_params if hasattr(self.controller_thread, 'current_control_params') else {}
                        left_controller = self.controller_thread.controllers.get("Left")
                        right_controller = self.controller_thread.controllers.get("Right")
                        active_controller = right_controller if right_controller else left_controller
                        from PIL import Image, ImageDraw, ImageFont
                        pil_image = Image.fromarray(rgb_image)
                        draw = ImageDraw.Draw(pil_image)
                        try:
                            font = ImageFont.truetype(font_path, 30)
                            slider_value = control_params.get("slider", active_controller.slider if active_controller else 0.0)
                            draw.text(
                                (20, 60),
                                f"滑块值: {slider_value:.2f}",
                                font=font,
                                fill=text_color
                            )
                            joystick = control_params.get("joystick", active_controller.joystick if active_controller else (0.0, 0.0))
                            draw.text(
                                (20, 110),
                                f"摇杆: X={joystick[0]:.2f}, Y={joystick[1]:.2f}",
                                font=font,
                                fill=text_color
                            )
                            dial = control_params.get("dial", active_controller.dial if active_controller else (0.0, 0.0))
                            draw.text(
                                (20, 160),
                                f"旋钮: X={dial[0]:.2f}, Y={dial[1]:.2f}",
                                font=font,
                                fill=text_color
                            )
                            if active_controller and hasattr(active_controller, 'buttons'):
                                buttons = active_controller.buttons
                                if buttons:
                                    y_pos = 210
                                    for btn_name, is_pressed in buttons.items():
                                        status = "按下" if is_pressed else "松开"
                                        draw.text(
                                            (20, y_pos),
                                            f"{btn_name}: {status}",
                                            font=font,
                                            fill=text_color
                                        )
                                        y_pos += 60
                            if active_controller:
                                y_pos = self.height - 200
                                draw.text(
                                    (20, y_pos),
                                    "陀螺仪参数:",
                                    font=font,
                                    fill=text_color
                                )
                                draw.text(
                                    (20, y_pos + 60),
                                    f"W: {active_controller.w:.2f}",
                                    font=font,
                                    fill=text_color
                                )
                                draw.text(
                                    (20, y_pos + 120),
                                    f"X: {active_controller.x:.2f}",
                                    font=font,
                                    fill=text_color
                                )
                                draw.text(
                                    (150, y_pos + 60),
                                    f"Y: {active_controller.y:.2f}",
                                    font=font,
                                    fill=text_color
                                )
                                draw.text(
                                    (150, y_pos + 120),
                                    f"Z: {active_controller.z:.2f}",
                                    font=font,
                                    fill=text_color
                                )
                        except Exception as e:
                            print(f"使用PIL绘制中文出错: {e}")
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            slider_value = control_params.get("slider", active_controller.slider if active_controller else 0.0)
                            cv2.putText(
                                rgb_image,
                                f"Slider: {slider_value:.2f}",
                                (20, 60),
                                font,
                                font_scale,
                                text_color,
                                font_thickness,
                                cv2.LINE_AA
                            )
                            joystick = control_params.get("joystick", active_controller.joystick if active_controller else (0.0, 0.0))
                            cv2.putText(
                                rgb_image,
                                f"Joystick: X={joystick[0]:.2f}, Y={joystick[1]:.2f}",
                                (20, 120),
                                font,
                                font_scale,
                                text_color,
                                font_thickness,
                                cv2.LINE_AA
                            )
                    except Exception as e:
                        print(f"绘制控制参数出错: {e}")
                        pass
                    self.tracker.process_frames(rgb_image)
                    if self.show_image:
                        h, w, ch = rgb_image.shape
                        bytes_per_line = ch * w
                        convert_to_Qt_format = QImage(
                            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
                        )
                        self.frame_ready.emit(convert_to_Qt_format)
            else:
                self.video_capture = cv2.VideoCapture(self.source, cv2.CAP_ANY)
                self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.video_capture.set(cv2.CAP_PROP_FPS, self.fps)
                print(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),self.video_capture.get(cv2.CAP_PROP_FPS))
                self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                while self.is_running:
                    ret, frame = self.video_capture.read()
                    if ret:
                        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if g.config["Setting"]["camera_width"]<640 or g.config["Setting"]["camera_height"]<480:
                            rgb_image = cv2.resize(rgb_image, (g.config["Setting"]["camera_width"], g.config["Setting"]["camera_height"]))
                        if g.config["Setting"]["flip_x"]:
                            rgb_image = cv2.flip(rgb_image, 1)
                        if g.config["Setting"]["flip_y"]:
                            rgb_image = cv2.flip(rgb_image, 0)
                        self.tracker.process_frames(rgb_image)
                        if self.show_image:
                            if g.config["Tracking"]["Head"]["enable"] or g.config["Tracking"]["Face"]["enable"]:
                                rgb_image = draw_face_landmarks(rgb_image)
                            if g.config["Tracking"]["Tongue"]["enable"]:
                                rgb_image = draw_tongue_position(rgb_image)
                            if g.config["Tracking"]["Pose"]["enable"]:
                                rgb_image = draw_pose_landmarks(rgb_image)
                            if g.config["Tracking"]["Hand"]["enable"]:
                                rgb_image = draw_hand_landmarks(rgb_image)
                            h, w, ch = rgb_image.shape
                            bytes_per_line = ch * w
                            convert_to_Qt_format = QImage(
                                rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
                            )
                            self.frame_ready.emit(convert_to_Qt_format)
        except Exception as e:
            print(f"视频捕获线程出错: {e}")
        self.cleanup()
    def stop(self):
        self.is_running = False
        self.tracker.stop()
    def cleanup(self):
        if self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()
class VideoWindow(QMainWindow):
    def __init__(self, splash_screen=None):
        super().__init__()
        if splash_screen:
            splash_screen.update_status("初始化界面组件...")
        screen = QApplication.screens()[0]
        screen_size = screen.size()
        self.width = int(screen_size.width() * 0.8)
        self.height = int(screen_size.height() * 0.7)
        version=g.config["Version"]
        self.setWindowTitle(
            f"ExVR {version} - 体验虚拟现实(星芙芙ovo二改)"
        )
        self.resize(self.width, self.height)
        self.setMinimumSize(800, 600)
        self.setStyleSheet(MODERN_STYLE_SHEET)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        left_menu = QWidget()
        left_menu.setObjectName("LeftMenu")
        left_menu.setFixedWidth(200)
        left_menu_layout = QVBoxLayout(left_menu)
        left_menu_layout.setContentsMargins(0, 16, 0, 16)
        left_menu_layout.setSpacing(0)
        exvr_label = QLabel("EXVR 0.7.2.5")
        exvr_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFFFFF; margin: 0 16px 24px 16px; background-color: transparent;")
        exvr_label.setAlignment(Qt.AlignCenter)
        left_menu_layout.addWidget(exvr_label)
        self.main_menu_button = QPushButton("主界面")
        self.main_menu_button.setObjectName("MenuItem")
        self.main_menu_button.setCheckable(True)
        self.main_menu_button.setChecked(True)
        self.main_menu_button.clicked.connect(lambda: self.switch_page("main"))
        self.main_menu_button.setStyleSheet("font-size: 16px; padding: 16px 20px; background-color: rgba(255, 255, 255, 0.1);")
        left_menu_layout.addWidget(self.main_menu_button)
        self.camera_menu_button = QPushButton("摄像头")
        self.camera_menu_button.setObjectName("MenuItem")
        self.camera_menu_button.setCheckable(True)
        self.camera_menu_button.clicked.connect(lambda: self.switch_page("camera"))
        left_menu_layout.addWidget(self.camera_menu_button)
        left_menu_layout.addStretch()
        content_area = QWidget()
        content_area.setObjectName("ContentArea")
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(10, 10, 10, 10)
        self.main_page = QWidget()
        main_page_scroll = QScrollArea()
        main_page_scroll.setWidgetResizable(True)
        main_page_scroll_content = QWidget()
        self.main_page_layout = QVBoxLayout(main_page_scroll_content)
        self.main_page_layout.setContentsMargins(0, 0, 0, 0)
        self.main_page_layout.setSpacing(16)
        main_page_scroll.setWidget(main_page_scroll_content)
        main_page_layout = QVBoxLayout(self.main_page)
        main_page_layout.setContentsMargins(0, 0, 0, 0)
        main_page_layout.addWidget(main_page_scroll)
        if splash_screen:
            splash_screen.update_status("创建界面布局...")
        self.camera_page = QWidget()
        self.camera_page_layout = QVBoxLayout(self.camera_page)
        self.camera_page_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_page_layout.setSpacing(0)
        self.camera_image_label = QLabel("请先追踪再来照镜子哦~")
        self.camera_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_image_label.setAlignment(Qt.AlignCenter)
        self.camera_image_label.setStyleSheet("background-color: #F8F9FA; border-radius: 8px; font-size: 16px; color: #666666;")
        self.camera_page_layout.addWidget(self.camera_image_label)
        content_layout.addWidget(self.main_page)
        content_layout.addWidget(self.camera_page)
        content_layout.setStretch(0, 1)
        content_layout.setStretch(1, 1)
        self.camera_page.hide()
        main_layout.addWidget(left_menu)
        main_layout.addWidget(content_area, 1)
        self.initialize_components()
        if splash_screen:
            splash_screen.update_status("初始化驱动按钮...")
        self.update_checkboxes()
        self.update_sliders()
        self.video_thread = None
        self.controller_thread = None
        if splash_screen:
            splash_screen.update_status("检测摄像头设备...")
    def initialize_components(self):
        """初始化所有组件"""
        self.steamvr_status_label = QLabel()
        self.main_page_layout.insertWidget(0, self.steamvr_status_label)
        self.ip_camera_tip_label = QLabel("📌 摄像头方向")
        self.ip_camera_tip_label.setStyleSheet("font-size: 11px; color: #666666;")
        self.main_page_layout.addWidget(self.ip_camera_tip_label)
        flip_layout = QHBoxLayout()
        self.flip_x_checkbox = QCheckBox("水平翻转", self)
        self.flip_x_checkbox.clicked.connect(self.flip_x)
        self.flip_x_checkbox.setChecked(g.config["Setting"]["flip_x"])
        flip_layout.addWidget(self.flip_x_checkbox)
        self.flip_y_checkbox = QCheckBox("垂直翻转", self)
        self.flip_y_checkbox.clicked.connect(self.flip_y)
        self.flip_y_checkbox.setChecked(g.config["Setting"]["flip_y"])
        flip_layout.addWidget(self.flip_y_checkbox)
        self.main_page_layout.addLayout(flip_layout)
        self.ip_camera_tip_label = QLabel("📌 IP摄像头示例：rtsp://admin:123456@192.168.1.100:554/stream1 或 `http://192.168.1.101:8080/video` ")
        self.ip_camera_tip_label.setStyleSheet("font-size: 11px; color: #666666;")
        self.main_page_layout.addWidget(self.ip_camera_tip_label)
        self.ip_camera_url_input = QLineEdit(self)
        self.ip_camera_url_input.setPlaceholderText("输入IP摄像头URL")
        self.ip_camera_url_input.textChanged.connect(self.update_camera_ip)
        self.ip_camera_url_input.setText(g.config["Setting"].get("camera_ip", ""))
        self.main_page_layout.addWidget(self.ip_camera_url_input)
        self.camera_id_tip_label = QLabel("📌 小提示：摄像头ID超过1000为MSMF格式/1000往下为DSHOW格式")
        self.camera_id_tip_label.setStyleSheet("font-size: 11px; color: #666666;")
        self.main_page_layout.addWidget(self.camera_id_tip_label)
        camera_layout = QHBoxLayout()
        self.camera_selection = QComboBox(self)
        self.populate_camera_list()
        camera_layout.addWidget(self.camera_selection)
        self.camera_resolution_selection = QComboBox(self)
        self.populate_resolution_list()
        self.camera_resolution_selection.currentIndexChanged.connect(self.update_camera_resolution)
        camera_layout.addWidget(self.camera_resolution_selection)
        self.camera_fps_selection = QComboBox(self)
        self.populate_fps_list()
        self.camera_fps_selection.currentIndexChanged.connect(self.update_camera_fps)
        camera_layout.addWidget(self.camera_fps_selection)
        self.main_page_layout.addLayout(camera_layout)
        self.priority_tip_label = QLabel("📌 小提示：这是设置本程序优先级的")
        self.priority_tip_label.setStyleSheet("font-size: 11px; color: #666666;")
        self.main_page_layout.addWidget(self.priority_tip_label)
        self.priority_selection = QComboBox(self)
        self.priority_mapping = {
            "最低优先级": "IDLE_PRIORITY_CLASS",
            "低于正常": "BELOW_NORMAL_PRIORITY_CLASS",
            "正常优先级": "NORMAL_PRIORITY_CLASS",
            "高于正常": "ABOVE_NORMAL_PRIORITY_CLASS",
            "高优先级": "HIGH_PRIORITY_CLASS",
            "实时优先级": "REALTIME_PRIORITY_CLASS"
        }
        self.priority_selection.addItems(list(self.priority_mapping.keys()))
        self.priority_selection.currentIndexChanged.connect(self.set_process_priority)
        self.main_page_layout.addWidget(self.priority_selection)
        current_priority = g.config["Setting"]["priority"]
        for cn_name, en_name in self.priority_mapping.items():
            if en_name == current_priority:
                self.priority_selection.setCurrentText(cn_name)
                break
        self.install_state, steamvr_driver_path, vrcfacetracking_path, check_steamvr_path = self.install_checking()
        if check_steamvr_path is not None:
            self.steamvr_status_label.setText("SteamVR已安装")
            self.steamvr_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.steamvr_status_label.setText("SteamVR未安装")
            self.steamvr_status_label.setStyleSheet("color: red; font-weight: bold;")
        if self.install_state:
            self.install_button = QPushButton("卸载驱动", self)
        else:
            self.install_button = QPushButton("安装驱动", self)
            self.install_button.setStyleSheet("QPushButton { background-color: #3498DB; color: white; }")
        self.install_button.clicked.connect(self.install_function)
        self.main_page_layout.addWidget(self.install_button)
        self.toggle_button = QPushButton("开始追踪", self)
        self.toggle_button.setStyleSheet("QPushButton { background-color: #27AE60; color: white; }")
        self.toggle_button.clicked.connect(self.toggle_camera)
        self.main_page_layout.addWidget(self.toggle_button)
        only_ingame_layout = QHBoxLayout()
        self.only_ingame_checkbox = QCheckBox("仅在游戏中", self)
        self.only_ingame_checkbox.clicked.connect(lambda: self.toggle_only_in_game(self.only_ingame_checkbox.isChecked()))
        self.only_ingame_checkbox.setChecked(g.config["Setting"]["only_ingame"])
        self.only_ingame_checkbox.setToolTip("目前仅适用于热键和鼠标输入，不适用于头部移动")
        self.only_ingame_game_input = QLineEdit(self)
        self.only_ingame_game_input.setPlaceholderText("窗口标题 / 进程名称 / VRChat, VRChat.exe, javaw.exe")
        self.only_ingame_game_input.textChanged.connect(self.update_mouse_only_in_game_name)
        self.only_ingame_game_input.setText(g.config["Setting"]["only_ingame_game"])
        only_ingame_layout.addWidget(self.only_ingame_checkbox)
        only_ingame_layout.addWidget(self.only_ingame_game_input)
        self.main_page_layout.addLayout(only_ingame_layout)
        separator_0 = QFrame()
        separator_0.setFrameShape(QFrame.HLine)
        separator_0.setFrameShadow(QFrame.Sunken)
        self.main_page_layout.addWidget(separator_0)
        reset_layout = QHBoxLayout()
        self.reset_head = QPushButton("重置头部", self)
        self.reset_head.clicked.connect(reset_head)
        reset_layout.addWidget(self.reset_head)
        self.reset_eyes = QPushButton("重置眼睛", self)
        self.reset_eyes.clicked.connect(reset_eye)
        reset_layout.addWidget(self.reset_eyes)
        self.reset_l_hand = QPushButton("重置左手", self)
        self.reset_l_hand.clicked.connect(lambda: reset_hand(True))
        reset_layout.addWidget(self.reset_l_hand)
        self.reset_r_hand = QPushButton("重置右手", self)
        self.reset_r_hand.clicked.connect(lambda: reset_hand(False))
        reset_layout.addWidget(self.reset_r_hand)
        self.main_page_layout.addLayout(reset_layout)
        checkbox_layout = QHBoxLayout()
        self.checkbox1 = QCheckBox("头部", self)
        self.checkbox1.clicked.connect(
            lambda: self.set_tracking_config("Head", self.checkbox1.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox1)
        self.checkbox2 = QCheckBox("面部", self)
        self.checkbox2.clicked.connect(
            lambda: self.set_tracking_config("Face", self.checkbox2.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox2)
        self.checkbox3 = QCheckBox("舌头", self)
        self.checkbox3.clicked.connect(
            lambda: self.set_tracking_config("Tongue", self.checkbox3.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox3)
        self.checkbox4 = QCheckBox("手部", self)
        self.checkbox4.clicked.connect(
            lambda: self.set_tracking_config("Hand", self.checkbox4.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox4)
        self.checkbox5 = QCheckBox("姿态", self)
        self.checkbox5.clicked.connect(
            lambda: self.set_tracking_config("Pose", self.checkbox5.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox5)
        self.main_page_layout.addLayout(checkbox_layout)
        checkbox_layout_1 = QHBoxLayout()
        self.checkbox6 = QCheckBox("手部放下", self)
        self.checkbox6.clicked.connect(
            lambda: self.toggle_hand_down(self.checkbox6.isChecked())
        )
        checkbox_layout_1.addWidget(self.checkbox6)
        self.checkbox7 = QCheckBox("手指动作", self)
        self.checkbox7.clicked.connect(
            lambda: self.toggle_finger_action(self.checkbox7.isChecked())
        )
        checkbox_layout_1.addWidget(self.checkbox7)
        self.main_page_layout.addLayout(checkbox_layout_1)
        slider_layout = QHBoxLayout()
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider1.setRange(1, 200)
        self.slider2.setRange(1, 200)
        self.slider3.setRange(1, 100)
        self.slider1.setSingleStep(1)
        self.slider2.setSingleStep(1)
        self.slider3.setSingleStep(1)
        self.label1 = QLabel(f"x {g.config['Tracking']['Hand']['x_scalar']:.2f}")
        self.label2 = QLabel(f"y {g.config['Tracking']['Hand']['y_scalar']:.2f}")
        self.label3 = QLabel(f"z {g.config['Tracking']['Hand']['z_scalar']:.2f}")
        self.slider1.valueChanged.connect(lambda value: self.set_scalar(value, "x"))
        self.slider2.valueChanged.connect(lambda value: self.set_scalar(value, "y"))
        self.slider3.valueChanged.connect(lambda value: self.set_scalar(value, "z"))
        slider_layout.addWidget(self.label1)
        slider_layout.addWidget(self.slider1)
        slider_layout.addWidget(self.label2)
        slider_layout.addWidget(self.slider2)
        slider_layout.addWidget(self.label3)
        slider_layout.addWidget(self.slider3)
        self.main_page_layout.addLayout(slider_layout)
        separator_1 = QFrame()
        separator_1.setFrameShape(QFrame.HLine)
        separator_1.setFrameShadow(QFrame.Sunken)
        self.main_page_layout.addWidget(separator_1)
        controller_checkbox_layout = QHBoxLayout()
        self.controller_checkbox1 = QCheckBox("左手控制器", self)
        self.controller_checkbox1.clicked.connect(
            lambda: self.set_tracking_config("LeftController", self.controller_checkbox1.isChecked())
        )
        self.controller_checkbox2 = QCheckBox("右手控制器", self)
        self.controller_checkbox2.clicked.connect(
            lambda: self.set_tracking_config("RightController", self.controller_checkbox2.isChecked())
        )
        controller_checkbox_layout.addWidget(self.controller_checkbox1)
        controller_checkbox_layout.addWidget(self.controller_checkbox2)
        self.main_page_layout.addLayout(controller_checkbox_layout)
        controller_slider_layout = QHBoxLayout()
        self.controller_slider_x = QSlider(Qt.Horizontal)
        self.controller_slider_y = QSlider(Qt.Horizontal)
        self.controller_slider_z = QSlider(Qt.Horizontal)
        self.controller_slider_l = QSlider(Qt.Horizontal)
        self.controller_slider_x.setRange(-50, 50)
        self.controller_slider_y.setRange(-50, 50)
        self.controller_slider_z.setRange(-50, 50)
        self.controller_slider_l.setRange(0, 100)
        self.controller_slider_x.setSingleStep(1)
        self.controller_slider_y.setSingleStep(1)
        self.controller_slider_z.setSingleStep(1)
        self.controller_slider_l.setSingleStep(1)
        self.controller_label_x = QLabel(f"x {g.config['Tracking']['LeftController']['base_x']:.2f}")
        self.controller_label_y = QLabel(f"y {g.config['Tracking']['LeftController']['base_y']:.2f}")
        self.controller_label_z = QLabel(f"z {g.config['Tracking']['LeftController']['base_z']:.2f}")
        self.controller_label_l = QLabel(f"l {g.config['Tracking']['LeftController']['length']:.2f}")
        self.controller_slider_x.valueChanged.connect(lambda value: self.set_scalar(value, "controller_x"))
        self.controller_slider_y.valueChanged.connect(lambda value: self.set_scalar(value, "controller_y"))
        self.controller_slider_z.valueChanged.connect(lambda value: self.set_scalar(value, "controller_z"))
        self.controller_slider_l.valueChanged.connect(lambda value: self.set_scalar(value, "controller_l"))
        controller_slider_layout.addWidget(self.controller_label_x)
        controller_slider_layout.addWidget(self.controller_slider_x)
        controller_slider_layout.addWidget(self.controller_label_y)
        controller_slider_layout.addWidget(self.controller_slider_y)
        controller_slider_layout.addWidget(self.controller_label_z)
        controller_slider_layout.addWidget(self.controller_slider_z)
        controller_slider_layout.addWidget(self.controller_label_l)
        controller_slider_layout.addWidget(self.controller_slider_l)
        self.main_page_layout.addLayout(controller_slider_layout)
        separator_2 = QFrame()
        separator_2.setFrameShape(QFrame.HLine)
        separator_2.setFrameShadow(QFrame.Sunken)
        self.main_page_layout.addWidget(separator_2)
        mouse_layout = QHBoxLayout()
        self.mouse_checkbox = QCheckBox("鼠标", self)
        self.mouse_checkbox.clicked.connect(lambda: self.toggle_mouse(self.mouse_checkbox.isChecked()))
        self.mouse_checkbox.setChecked(g.config["Mouse"]["enable"])
        self.mouse_slider_x = QSlider(Qt.Horizontal)
        self.mouse_slider_y = QSlider(Qt.Horizontal)
        self.mouse_slider_dx = QSlider(Qt.Horizontal)
        self.mouse_slider_x.setRange(0, 360)
        self.mouse_slider_y.setRange(0, 360)
        self.mouse_slider_dx.setRange(0, 20)
        self.mouse_slider_x.setSingleStep(1)
        self.mouse_slider_y.setSingleStep(1)
        self.mouse_slider_dx.setSingleStep(1)
        self.mouse_label_x = QLabel(f"x {int(g.config['Mouse']['scalar_x']*100)}")
        self.mouse_label_y = QLabel(f"y {int(g.config['Mouse']['scalar_y']*100)}")
        self.mouse_label_dx = QLabel(f"dx {g.config['Mouse']['dx']:.2f}")
        self.mouse_slider_x.valueChanged.connect(lambda value: self.set_scalar(value, "mouse_x"))
        self.mouse_slider_y.valueChanged.connect(lambda value: self.set_scalar(value, "mouse_y"))
        self.mouse_slider_dx.valueChanged.connect(lambda value: self.set_scalar(value, "mouse_dx"))
        mouse_layout.addWidget(self.mouse_checkbox)
        mouse_layout.addWidget(self.mouse_label_x)
        mouse_layout.addWidget(self.mouse_slider_x)
        mouse_layout.addWidget(self.mouse_label_y)
        mouse_layout.addWidget(self.mouse_slider_y)
        mouse_layout.addWidget(self.mouse_label_dx)
        mouse_layout.addWidget(self.mouse_slider_dx)
        self.main_page_layout.addLayout(mouse_layout)
        separator_3 = QFrame()
        separator_3.setFrameShape(QFrame.HLine)
        separator_3.setFrameShadow(QFrame.Sunken)
        self.main_page_layout.addWidget(separator_3)
        config_layout = QHBoxLayout()
        self.reset_hotkey_button = QPushButton("重置热键", self)
        self.reset_hotkey_button.clicked.connect(self.reset_hotkeys)
        config_layout.addWidget(self.reset_hotkey_button)
        self.stop_hotkey_button = QPushButton("停止热键", self)
        self.stop_hotkey_button.clicked.connect(stop_hotkeys)
        config_layout.addWidget(self.stop_hotkey_button)
        self.set_face_button = QPushButton("设置面部", self)
        self.set_face_button.clicked.connect(self.face_dialog)
        config_layout.addWidget(self.set_face_button)
        self.update_config_button = QPushButton("更新配置", self)
        self.update_config_button.clicked.connect(lambda:(g.update_configs(),self.update_checkboxes(), self.update_sliders()))
        config_layout.addWidget(self.update_config_button)
        self.save_config_button = QPushButton("保存配置", self)
        self.save_config_button.clicked.connect(g.save_configs)
        config_layout.addWidget(self.save_config_button)
        self.main_page_layout.addLayout(config_layout)
    def switch_page(self, page_name):
        """切换页面"""
        if page_name == "main":
            self.main_menu_button.setChecked(True)
            self.camera_menu_button.setChecked(False)
            self.main_page.show()
            self.camera_page.hide()
            self.main_menu_button.setStyleSheet("font-size: 16px; padding: 16px 20px; background-color: rgba(255, 255, 255, 0.1);")
            self.camera_menu_button.setStyleSheet("")
            if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.show_image:
                self.video_thread.show_image = False
                print("已隐藏画面显示")
                self.camera_image_label.clear()
                self.camera_image_label.setText("请先追踪再来照镜子哦~")
                self.camera_image_label.setStyleSheet("background-color: #F8F9FA; border-radius: 8px; font-size: 16px; color: #666666;")
        elif page_name == "camera":
            self.main_menu_button.setChecked(False)
            self.camera_menu_button.setChecked(True)
            self.main_page.hide()
            self.camera_page.show()
            self.camera_menu_button.setStyleSheet("font-size: 16px; padding: 16px 20px; background-color: rgba(255, 255, 255, 0.1);")
            self.main_menu_button.setStyleSheet("")
            if hasattr(self, 'video_thread') and self.video_thread and not self.video_thread.show_image:
                self.video_thread.show_image = True
                print("已启用画面显示")
                self.camera_image_label.setText("追踪中...")
                self.camera_image_label.setStyleSheet("background-color: #F8F9FA; border-radius: 8px; font-size: 16px; color: #666666;")
    def update_config(self, path, key, value):
        parts = path.split('/')
        config = g.config
        for part in parts[:-1]:
            config = config[part]
        config[key] = value
        g.save_configs()
    def flip_x(self, value):
        g.config["Setting"]["flip_x"] = value
        g.save_configs()
    def flip_y(self, value):
        g.config["Setting"]["flip_y"] = value
        g.save_configs()
    def update_camera_ip(self, value):
        g.config["Setting"]["camera_ip"] = value
    def toggle_only_in_game(self, value):
        g.config["Setting"]["only_ingame"] = value
    def update_mouse_only_in_game_name(self, value):
        g.config["Setting"]["only_ingame_game"] = value
    def set_tracking_config(self, key, value):
        if key in g.config["Tracking"]:
            g.config["Tracking"][key]["enable"] = value
        if key == "LeftController":
            g.controller.left_hand.force_enable = value
        if key == "RightController":
            g.controller.right_hand.force_enable = value
        g.save_configs()
    def toggle_hand_down(self, value):
        g.config["Tracking"]["Hand"]["enable_hand_down"] = value
        g.save_configs()
    def toggle_finger_action(self, value):
        g.config["Tracking"]["Hand"]["enable_finger_action"] = value
        g.save_configs()
    def toggle_mouse(self, value):
        g.config["Mouse"]["enable"] = value
        g.save_configs()
    def update_checkboxes(self):
        self.flip_x_checkbox.setChecked(g.config["Setting"]["flip_x"])
        self.flip_y_checkbox.setChecked(g.config["Setting"]["flip_y"])
        self.checkbox1.setChecked(g.config["Tracking"]["Head"]["enable"])
        self.checkbox2.setChecked(g.config["Tracking"]["Face"]["enable"])
        self.checkbox3.setChecked(g.config["Tracking"]["Tongue"]["enable"])
        self.checkbox4.setChecked(g.config["Tracking"]["Hand"]["enable"])
        self.checkbox5.setChecked(g.config["Tracking"]["Pose"]["enable"])
        self.checkbox6.setChecked(g.config["Tracking"]["Hand"]["enable_hand_down"])
        self.checkbox7.setChecked(g.config["Tracking"]["Hand"]["enable_finger_action"])
        self.controller_checkbox1.setChecked(g.config["Tracking"]["LeftController"]["enable"])
        self.controller_checkbox2.setChecked(g.config["Tracking"]["RightController"]["enable"])
        self.mouse_checkbox.setChecked(g.config["Mouse"]["enable"])
    def update_sliders(self):
        x_scalar = g.config["Tracking"]["Hand"]["x_scalar"]
        y_scalar = g.config["Tracking"]["Hand"]["y_scalar"]
        z_scalar = g.config["Tracking"]["Hand"]["z_scalar"]
        self.slider1.setValue(int(x_scalar * 100))
        self.slider2.setValue(int(y_scalar * 100))
        self.slider3.setValue(int(z_scalar * 100))
        self.label1.setText(f"x {x_scalar:.2f}")
        self.label2.setText(f"y {y_scalar:.2f}")
        self.label3.setText(f"z {z_scalar:.2f}")
        controller_x = g.config["Tracking"]["LeftController"]["base_x"]
        controller_y = g.config["Tracking"]["LeftController"]["base_y"]
        controller_z = g.config["Tracking"]["LeftController"]["base_z"]
        controller_l = g.config["Tracking"]["LeftController"]["length"]
        self.controller_slider_x.setValue(int(controller_x * 100))
        self.controller_slider_y.setValue(int(controller_y * 100))
        self.controller_slider_z.setValue(int(controller_z * 100))
        self.controller_slider_l.setValue(int(controller_l * 100))
        self.controller_label_x.setText(f"x {controller_x:.2f}")
        self.controller_label_y.setText(f"y {controller_y:.2f}")
        self.controller_label_z.setText(f"z {controller_z:.2f}")
        self.controller_label_l.setText(f"l {controller_l:.2f}")
        mouse_x = g.config["Mouse"]["scalar_x"] / 100
        mouse_y = g.config["Mouse"]["scalar_y"] / 100
        mouse_dx = g.config["Mouse"]["dx"]
        self.mouse_slider_x.setValue(int(mouse_x))
        self.mouse_slider_y.setValue(int(mouse_y))
        self.mouse_slider_dx.setValue(int(mouse_dx))
        self.mouse_label_x.setText(f"x {int(mouse_x * 100)}")
        self.mouse_label_y.setText(f"y {int(mouse_y * 100)}")
        self.mouse_label_dx.setText(f"dx {mouse_dx:.2f}")
    def set_scalar(self, value, axis):
        if axis == "x":
            val = value / 100.0
            g.config["Tracking"]["Hand"]["x_scalar"] = val
            self.label1.setText(f"x {val:.2f}")
        elif axis == "y":
            val = value / 100.0
            g.config["Tracking"]["Hand"]["y_scalar"] = val
            self.label2.setText(f"y {val:.2f}")
        elif axis == "z":
            val = value / 100.0
            g.config["Tracking"]["Hand"]["z_scalar"] = val
            self.label3.setText(f"z {val:.2f}")
        elif axis == "controller_x":
            val = value / 100.0
            g.config["Tracking"]["LeftController"]["base_x"] = val
            g.config["Tracking"]["RightController"]["base_x"] = -val
            self.controller_label_x.setText(f"x {val:.2f}")
        elif axis == "controller_y":
            val = value / 100.0
            g.config["Tracking"]["LeftController"]["base_y"] = val
            g.config["Tracking"]["RightController"]["base_y"] = val
            self.controller_label_y.setText(f"y {val:.2f}")
        elif axis == "controller_z":
            val = value / 100.0
            g.config["Tracking"]["LeftController"]["base_z"] = val
            g.config["Tracking"]["RightController"]["base_z"] = val
            self.controller_label_z.setText(f"z {val:.2f}")
        elif axis == "controller_l":
            val = value / 100.0
            g.config["Tracking"]["LeftController"]["length"] = val
            g.config["Tracking"]["RightController"]["length"] = val
            self.controller_label_l.setText(f"l {val:.2f}")
        elif axis == "mouse_x":
            val = value / 100.0
            g.config["Mouse"]["scalar_x"] = val
            self.mouse_label_x.setText(f"x {int(value)}")
        elif axis == "mouse_y":
            val = value / 100.0
            g.config["Mouse"]["scalar_y"] = val
            self.mouse_label_y.setText(f"y {int(value)}")
        elif axis == "mouse_dx":
            val = value / 100.0
            g.config["Mouse"]["dx"] = val
            self.mouse_label_dx.setText(f"dx {val:.2f}")
        g.save_configs()
    def update_sliders(self):
        self.slider1.setValue(int(g.config["Tracking"]["Hand"]["x_scalar"] * 100))
        self.slider2.setValue(int(g.config["Tracking"]["Hand"]["y_scalar"] * 100))
        self.slider3.setValue(int(g.config["Tracking"]["Hand"]["z_scalar"] * 100))
        self.controller_slider_x.setValue(int(g.config["Tracking"]["LeftController"]["base_x"] * 100))
        self.controller_slider_y.setValue(int(g.config["Tracking"]["LeftController"]["base_y"] * 100))
        self.controller_slider_z.setValue(int(g.config["Tracking"]["LeftController"]["base_z"] * 100))
        self.controller_slider_l.setValue(int(g.config["Tracking"]["LeftController"]["length"] * 100))
        self.mouse_slider_x.setValue(int(g.config["Mouse"]["scalar_x"] * 100))
        self.mouse_slider_y.setValue(int(g.config["Mouse"]["scalar_y"] * 100))
        self.mouse_slider_dx.setValue(int(g.config["Mouse"]["dx"] * 100))
    def check_steamvr_status(self):
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\WOW6432Node\Valve\Steam",
                0,
                winreg.KEY_READ,
            ) as reg_key:
                steam_path, _ = winreg.QueryValueEx(reg_key, "InstallPath")
            check_steamvr_path = os.path.join(
                steam_path, "steamapps", "common", "SteamVR", "bin"
            )
            if os.path.exists(check_steamvr_path):
                self.steamvr_status_label.setText("SteamVR 已安装")
                self.steamvr_status_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.steamvr_status_label.setText("SteamVR 未安装")
                self.steamvr_status_label.setStyleSheet("color: red; font-weight: bold;")
        except Exception as e:
            self.steamvr_status_label.setText("SteamVR 未安装")
            self.steamvr_status_label.setStyleSheet("color: red; font-weight: bold;")
    def install_checking(self):
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\WOW6432Node\Valve\Steam",
                0,
                winreg.KEY_READ,
            ) as reg_key:
                steam_path, _ = winreg.QueryValueEx(reg_key, "InstallPath")
            steamvr_driver_path = os.path.join(
                steam_path, "steamapps", "common", "SteamVR", "drivers"
            )
            check_steamvr_path = os.path.join(
                steam_path, "steamapps", "common", "SteamVR", "bin"
            )
            if not os.path.exists(check_steamvr_path):
                check_steamvr_path = None
            vrcfacetracking_path = os.path.join(
                os.getenv("APPDATA"), "VRCFaceTracking", "CustomLibs"
            )
            vrcfacetracking_module_path = os.path.join(
                vrcfacetracking_path, "VRCFT-MediapipePro.dll"
            )
            required_paths = [vrcfacetracking_module_path] + [
                os.path.join(steamvr_driver_path, driver)
                for driver in ["vmt", "vrto3d"]
            ]
            install_state = all(os.path.exists(path) for path in required_paths)
            return install_state, steamvr_driver_path, vrcfacetracking_path, check_steamvr_path
        except Exception as e:
            print(f"Error accessing registry or file system: {e}")
            return False, None, None, None
    def check_driver_status_on_startup(self):
        install_state, _, _, _ = self.install_checking()
        if install_state:
            self.install_button.setStyleSheet("""
                QPushButton {
                    background-color: #0078D7; 
                    color: white; 
                    border-radius: 4px; 
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #1080E0;
                }
            """)
            self.install_button.setText("已安装（点我卸载驱动）")
        else:
            self.install_button.setStyleSheet("""
                QPushButton {
                    background-color: #D13438; 
                    color: white; 
                    border-radius: 4px; 
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #E5484D;
                }
            """)
            self.install_button.setText("安装驱动")
    def install_function(self):
        self.install_state, steamvr_driver_path, vrcfacetracking_path, check_steamvr_path = self.install_checking()
        if check_steamvr_path is not None:
            self.steamvr_status_label.setText("SteamVR 已安装")
            self.steamvr_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.steamvr_status_label.setText("SteamVR 未安装")
            self.steamvr_status_label.setStyleSheet("color: red; font-weight: bold;")
        if self.install_state:
            dll_path = os.path.join(vrcfacetracking_path, "VRCFT-MediapipePro.dll")
            error_occurred = False
            drivers_to_remove = ["vmt", "vrto3d"]
            for driver in drivers_to_remove:
                dir_path = os.path.join(steamvr_driver_path, driver)
                try:
                    shutil.rmtree(dir_path)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    error_occurred = True
                if os.path.exists(dir_path):
                    error_occurred = True
            if error_occurred:
                QMessageBox.critical(self, "错误", "SteamVR 正在运行, 请关闭 SteamVR 再尝试.")
                return
            try:
                os.remove(dll_path)
            except PermissionError:
                QMessageBox.critical(self, "错误", "VRCFT 正在运行, 请关闭 VRCFT 再尝试")
                return
            self.install_button.setStyleSheet("""
                QPushButton {
                    background-color: #D13438; 
                    color: white; 
                    border-radius: 4px; 
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #E5484D;
                }
            """)
            self.install_button.setText("安装驱动")
        else:
            for driver in ["vmt", "vrto3d"]:
                source = os.path.join("./drivers", driver)
                destination = os.path.join(steamvr_driver_path, driver)
                if not os.path.exists(destination):
                    shutil.copytree(source, destination)
            dll_source = os.path.join("./drivers", "VRCFT-MediapipePro.dll")
            dll_destination = os.path.join(vrcfacetracking_path, "VRCFT-MediapipePro.dll")
            if not os.path.exists(dll_destination):
                os.makedirs(os.path.dirname(dll_destination), exist_ok=True)
                shutil.copy(dll_source, dll_destination)
            self.install_button.setStyleSheet("""
                QPushButton {
                    background-color: #0078D7; 
                    color: white; 
                    border-radius: 4px; 
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #1080E0;
                }
            """)
            self.install_button.setText("已安装（点我卸载驱动）")
    def toggle_camera(self):
        if self.video_thread and self.video_thread.isRunning():
            stop_hotkeys()
            self.toggle_button.setText("开始追踪")
            self.thread_stopped()
            QMessageBox.information(self, "提示", "追踪已停止")
        else:
            try:
                apply_hotkeys()
                ip_url = g.config["Setting"]["camera_ip"]
                selected_camera_name = self.camera_selection.currentText()
                source = ip_url if ip_url else self.get_camera_source(selected_camera_name)
                self.is_using_web_controller = source == -1
                if not self.is_using_web_controller:
                    self.controller_thread = None
                else:
                    try:
                        self.controller_thread = ControllerApp()
                        self.controller_thread.start()
                    except Exception as e:
                        QMessageBox.warning(self, "警告", f"启动网页虚拟控制器失败：{str(e)}")
                        self.controller_thread = None
                self.video_thread = VideoCaptureThread(
                    source,
                    g.config["Setting"]["camera_width"],
                    g.config["Setting"]["camera_height"],
                    g.config["Setting"]["camera_fps"]
                )
                self.video_thread.is_using_web_controller = self.is_using_web_controller
                if self.is_using_web_controller:
                    self.video_thread.controller_thread = self.controller_thread
                self.video_thread.frame_ready.connect(self.update_frame)
                self.video_thread.show_image = False
                self.video_thread.start()
                self.toggle_button.setText("停止追踪")
                self.camera_image_label.clear()
                self.camera_image_label.setText("追踪中...")
                self.camera_image_label.setStyleSheet("background-color: #F8F9FA; border-radius: 8px; font-size: 16px; color: #666666;")
                if self.is_using_web_controller and self.controller_thread:
                    QMessageBox.information(self, "提示", f"追踪已开始\n网页虚拟控制器已启动\n请使用浏览器访问：{', '.join([f'https://{ip[1]}' for ip in self.controller_thread.get_server_ip()])}")
                else:
                    QMessageBox.information(self, "提示", "追踪已开始")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"启动失败：{str(e)}")
                self.toggle_button.setText("开始追踪")
    def get_camera_source(self, selected_camera_name):
        if "网页虚拟控制器" in selected_camera_name or "WebController" in selected_camera_name:
            return -1
        try:
            devices = enumerate_cameras(cv2.CAP_ANY)
            for device in devices:
                if device.index > 1000:
                    device.name += " (MSMF)"
                else:
                    device.name += " (DSHOW)"
            for device in devices:
                if device.name == selected_camera_name:
                    return device.index
        except:
            pass
        return 0
    def toggle_video_display(self):
        if not self.video_thread:
            QMessageBox.warning(self, "警告", "请先启动追踪")
            return
        self.video_thread.show_image = not self.video_thread.show_image
        if self.video_thread.show_image:
            print("已启用画面显示")
        else:
            print("已隐藏画面显示")
        self.update_frame(QImage())
    def update_frame(self, image):
        if self.video_thread and self.video_thread.show_image:
            if not image.isNull():
                scaled = image.scaled(self.camera_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.camera_image_label.setPixmap(QPixmap.fromImage(scaled))
            else:
                pass
    def populate_camera_list(self):
        try:
            devices = enumerate_cameras(cv2.CAP_ANY)
            dshow_devices = []
            msmf_devices = []
            for device in devices:
                if device.index > 1000:
                    device.name += " (MSMF)"
                    msmf_devices.append(device)
                else:
                    device.name += " (DSHOW)"
                    dshow_devices.append(device)
            for device in msmf_devices + dshow_devices:
                self.camera_selection.addItem(device.name, device.index)
        except:
            for i in range(4):
                self.camera_selection.addItem(f"摄像头 {i}", i)
        self.camera_selection.addItem("网页虚拟控制器 (WebController)", -1)
    def populate_resolution_list(self):
        resolutions = [
            (160, 90), (160, 120), (320, 180), (320, 240),
            (640, 360), (640, 480), (800, 450), (800, 600),
            (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)
        ]
        for w, h in resolutions:
            gcd = np.gcd(w, h)
            aspect = f"{w//gcd}:{h//gcd}"
            self.camera_resolution_selection.addItem(f"{w}x{h} ({aspect})", (w, h))
        current_res = (g.config["Setting"]["camera_width"], g.config["Setting"]["camera_height"])
        if current_res in resolutions:
            self.camera_resolution_selection.setCurrentIndex(resolutions.index(current_res))
        else:
            self.camera_resolution_selection.setCurrentIndex(5)
    def populate_fps_list(self):
        self.camera_fps_selection.addItem("30 FPS", 30)
        self.camera_fps_selection.addItem("60 FPS", 60)
        current_fps = g.config["Setting"]["camera_fps"]
        if current_fps == 30:
            self.camera_fps_selection.setCurrentIndex(0)
        else:
            self.camera_fps_selection.setCurrentIndex(1)
    def update_camera_resolution(self):
        res = self.camera_resolution_selection.currentData()
        if res:
            g.config["Setting"]["camera_width"], g.config["Setting"]["camera_height"] = res
            g.save_configs()
    def update_camera_fps(self):
        fps = self.camera_fps_selection.currentData()
        if fps:
            g.config["Setting"]["camera_fps"] = fps
            g.save_configs()
    def update_camera(self):
        show_image = getattr(self.video_thread, 'show_image', False) if hasattr(self, 'video_thread') and self.video_thread else False
        show_button_text = self.show_frame_button.text() if hasattr(self, 'show_frame_button') else "显示画面"
        selected_camera_name = self.camera_selection.currentText()
        ip_url = g.config["Setting"]["camera_ip"]
        source = ip_url if ip_url else self.get_camera_source(selected_camera_name)
        self.is_using_web_controller = source == -1
        is_tracking = self.video_thread and self.video_thread.isRunning()
        if is_tracking:
            self.thread_stopped()
            if not self.is_using_web_controller:
                self.controller_thread = None
            else:
                try:
                    self.controller_thread = ControllerApp()
                    self.controller_thread.start()
                except Exception as e:
                    QMessageBox.warning(self, "警告", f"启动网页虚拟控制器失败：{str(e)}")
                    self.controller_thread = None
            self.video_thread = VideoCaptureThread(
                source,
                g.config["Setting"]["camera_width"],
                g.config["Setting"]["camera_height"],
                g.config["Setting"]["camera_fps"]
            )
            self.video_thread.is_using_web_controller = self.is_using_web_controller
            if self.is_using_web_controller:
                self.video_thread.controller_thread = self.controller_thread
            self.video_thread.frame_ready.connect(self.update_frame)
            self.video_thread.show_image = show_image
            self.video_thread.start()
        if show_button_text == "隐藏画面":
            if hasattr(self, 'video_thread') and self.video_thread:
                self.video_thread.show_image = True
            self.show_frame_button.setText("隐藏画面")
        else:
            if hasattr(self, 'video_thread') and self.video_thread:
                self.video_thread.show_image = False
            self.show_frame_button.setText("显示画面")
    def reset_hotkeys(self):
        stop_hotkeys()
        apply_hotkeys()
        QMessageBox.information(self, "提示", "热键已重置")
    def face_dialog(self):
        self.dialog = QDialog(self)
        self.dialog.setWindowTitle("面部参数设置")
        self.dialog.setStyleSheet(MODERN_STYLE_SHEET)
        self.dialog.resize(800, 600)
        self.dialog.setWindowFlags(self.dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        layout = QVBoxLayout(self.dialog)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        form_widget = QWidget()
        form_layout = QGridLayout(form_widget)
        headers = ["面部形状", "基准值", "偏移量", "权重", "最大值", "启用"]
        for col, header in enumerate(headers):
            form_layout.addWidget(QLabel(header), 0, col)
        self.lineEdits = {}
        self.checkBoxes = {}
        double_validator = QDoubleValidator()
        blendshape_cn_mapping = {
            "None": "无",
            "EyeBlinkLeft": "左眼眨眼",
            "EyeBlinkRight": "右眼眨眼",
            "EyeLookDownLeft": "左眼向下看",
            "EyeLookDownRight": "右眼向下看",
            "EyeLookInLeft": "左眼内看",
            "EyeLookInRight": "右眼内看",
            "EyeLookOutLeft": "左眼外看",
            "EyeLookOutRight": "右眼外看",
            "EyeLookUpLeft": "左眼向上看",
            "EyeLookUpRight": "右眼向上看",
            "EyeSquintLeft": "左眼眯眼",
            "EyeSquintRight": "右眼眯眼",
            "EyeWideLeft": "左眼睁大",
            "EyeWideRight": "右眼睁大",
            "JawForward": "下巴前伸",
            "JawLeft": "下巴左移",
            "JawRight": "下巴右移",
            "JawOpen": "下巴张开",
            "MouthClose": "嘴巴闭合",
            "MouthFunnel": "嘴巴漏斗形",
            "MouthPucker": "嘴巴噘起",
            "MouthLeft": "嘴巴左移",
            "MouthRight": "嘴巴右移",
            "MouthSmileLeft": "左嘴角微笑",
            "MouthSmileRight": "右嘴角微笑",
            "MouthFrownLeft": "左嘴角皱眉",
            "MouthFrownRight": "右嘴角皱眉",
            "MouthDimpleLeft": "左嘴角酒窝",
            "MouthDimpleRight": "右嘴角酒窝",
            "MouthStretchLeft": "左嘴角拉伸",
            "MouthStretchRight": "右嘴角拉伸",
            "MouthRollLower": "下嘴唇滚动",
            "MouthRollUpper": "上嘴唇滚动",
            "MouthShrugLower": "下嘴唇收缩",
            "MouthShrugUpper": "上嘴唇收缩",
            "MouthPressLeft": "左嘴角按压",
            "MouthPressRight": "右嘴角按压",
            "MouthLowerDownLeft": "左下嘴唇下拉",
            "MouthLowerDownRight": "右下嘴唇下拉",
            "MouthUpperUpLeft": "左上嘴唇上提",
            "MouthUpperUpRight": "右上嘴唇上提",
            "BrowDownLeft": "左眉毛下压",
            "BrowDownRight": "右眉毛下压",
            "BrowInnerUp": "眉毛内侧上提",
            "BrowOuterUpLeft": "左眉毛外侧上提",
            "BrowOuterUpRight": "右眉毛外侧上提",
            "CheekPuff": "脸颊鼓起",
            "CheekSquintLeft": "左脸颊眯起",
            "CheekSquintRight": "右脸颊眯起",
            "NoseSneerLeft": "左鼻孔皱起",
            "NoseSneerRight": "右鼻孔皱起",
            "TongueOut": "舌头伸出",
            "HeadYaw": "头部偏航",
            "HeadPitch": "头部俯仰",
            "HeadRoll": "头部翻滚",
            "EyeYawLeft": "左眼偏航",
            "EyePitchLeft": "左眼俯仰",
            "EyeRollLeft": "左眼翻滚",
            "EyeYawRight": "右眼偏航",
            "EyePitchRight": "右眼俯仰",
            "EyeRollRight": "右眼翻滚",
            "TongueX": "舌头X轴",
            "TongueY": "舌头Y轴"
        }
        position_cn_mapping = {
            "x": "X位置",
            "y": "Y位置",
            "z": "Z位置"
        }
        rotation_cn_mapping = {
            "x": "偏航",
            "y": "俯仰",
            "z": "翻滚"
        }
        try:
            blendshape_data, _ = setup_data()
        except:
            blendshape_data = g.default_data
        for row, blendshape in enumerate(blendshape_data["BlendShapes"][1:], start=1):
            key = blendshape["k"]
            cn_name = blendshape_cn_mapping.get(key, key)
            v_edit = QLineEdit(str(round(blendshape["v"], 2)))
            v_edit.setValidator(double_validator)
            s_edit = QLineEdit(str(round(blendshape["s"], 2)))
            s_edit.setValidator(double_validator)
            w_edit = QLineEdit(str(round(blendshape["w"], 2)))
            w_edit.setValidator(double_validator)
            max_edit = QLineEdit(str(round(blendshape["max"], 2)))
            max_edit.setValidator(double_validator)
            e_check = QCheckBox()
            e_check.setChecked(blendshape["e"])
            self.lineEdits[key] = (v_edit, s_edit, w_edit, max_edit)
            self.checkBoxes[key] = e_check
            form_layout.addWidget(QLabel(cn_name), row, 0)
            form_layout.addWidget(v_edit, row, 1)
            form_layout.addWidget(s_edit, row, 2)
            form_layout.addWidget(w_edit, row, 3)
            form_layout.addWidget(max_edit, row, 4)
            form_layout.addWidget(e_check, row, 5)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setObjectName("Separator")
        row += 1
        form_layout.addWidget(separator, row, 0, 1, 6)
        row += 1
        form_layout.addWidget(QLabel("位置参数"), row, 0, 1, 6)
        row += 1
        for pos in blendshape_data["Position"]:
            key = "pos_" + pos["k"]
            cn_name = position_cn_mapping.get(pos["k"], pos["k"])
            v_edit = QLineEdit(str(round(pos["v"], 2)))
            v_edit.setValidator(double_validator)
            s_edit = QLineEdit(str(round(pos["s"], 2)))
            s_edit.setValidator(double_validator)
            w_edit = QLineEdit("")
            w_edit.setEnabled(False)
            max_edit = QLineEdit("")
            max_edit.setEnabled(False)
            e_check = QCheckBox()
            e_check.setChecked(pos["e"])
            self.lineEdits[key] = (v_edit, s_edit, w_edit, max_edit)
            self.checkBoxes[key] = e_check
            form_layout.addWidget(QLabel(cn_name), row, 0)
            form_layout.addWidget(v_edit, row, 1)
            form_layout.addWidget(s_edit, row, 2)
            form_layout.addWidget(w_edit, row, 3)
            form_layout.addWidget(max_edit, row, 4)
            form_layout.addWidget(e_check, row, 5)
            row += 1
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setObjectName("Separator")
        form_layout.addWidget(separator, row, 0, 1, 6)
        row += 1
        form_layout.addWidget(QLabel("旋转参数"), row, 0, 1, 6)
        row += 1
        for rot in blendshape_data["Rotation"]:
            key = "rot_" + rot["k"]
            cn_name = rotation_cn_mapping.get(rot["k"], rot["k"])
            v_edit = QLineEdit(str(round(rot["v"], 2)))
            v_edit.setValidator(double_validator)
            s_edit = QLineEdit(str(round(rot["s"], 2)))
            s_edit.setValidator(double_validator)
            w_edit = QLineEdit("")
            w_edit.setEnabled(False)
            max_edit = QLineEdit("")
            max_edit.setEnabled(False)
            e_check = QCheckBox()
            e_check.setChecked(rot["e"])
            self.lineEdits[key] = (v_edit, s_edit, w_edit, max_edit)
            self.checkBoxes[key] = e_check
            form_layout.addWidget(QLabel(cn_name), row, 0)
            form_layout.addWidget(v_edit, row, 1)
            form_layout.addWidget(s_edit, row, 2)
            form_layout.addWidget(w_edit, row, 3)
            form_layout.addWidget(max_edit, row, 4)
            form_layout.addWidget(e_check, row, 5)
            row += 1
        scroll_area.setWidget(form_widget)
        layout.addWidget(scroll_area)
        save_btn = QPushButton("保存配置")
        save_btn.setObjectName("PrimaryButton")
        save_btn.clicked.connect(self.save_face_data)
        layout.addWidget(save_btn)
        self.dialog.exec_()
    def save_face_data(self):
        try:
            data = deepcopy(g.default_data)
            blendshape_idx = 1
            for key, edits in self.lineEdits.items():
                if not key.startswith("pos_") and not key.startswith("rot_"):
                    v = float(edits[0].text())
                    s = float(edits[1].text())
                    w = float(edits[2].text())
                    max_val = float(edits[3].text())
                    e = self.checkBoxes[key].isChecked()
                    data["BlendShapes"][blendshape_idx]["v"] = v
                    data["BlendShapes"][blendshape_idx]["s"] = s
                    data["BlendShapes"][blendshape_idx]["w"] = w
                    data["BlendShapes"][blendshape_idx]["max"] = max_val
                    data["BlendShapes"][blendshape_idx]["e"] = e
                    blendshape_idx += 1
            for key, edits in self.lineEdits.items():
                if key.startswith("pos_"):
                    pos_key = key[4:]
                    for i, pos in enumerate(data["Position"]):
                        if pos["k"] == pos_key:
                            v = float(edits[0].text())
                            s = float(edits[1].text())
                            e = self.checkBoxes[key].isChecked()
                            data["Position"][i]["v"] = v
                            data["Position"][i]["s"] = s
                            data["Position"][i]["e"] = e
                            break
            for key, edits in self.lineEdits.items():
                if key.startswith("rot_"):
                    rot_key = key[4:]
                    for i, rot in enumerate(data["Rotation"]):
                        if rot["k"] == rot_key:
                            v = float(edits[0].text())
                            s = float(edits[1].text())
                            e = self.checkBoxes[key].isChecked()
                            data["Rotation"][i]["v"] = v
                            data["Rotation"][i]["s"] = s
                            data["Rotation"][i]["e"] = e
                            break
            save_data(data)
            self.dialog.close()
            QMessageBox.information(self, "提示", "面部配置已保存")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存面部配置失败：{str(e)}")
    def on_update_config(self):
        g.update_configs()
        self.update_sliders()
        self.check_steamvr_status()
        self.check_driver_status_on_startup()
        QMessageBox.information(self, "提示", "配置已更新")
    def thread_stopped(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
        if self.controller_thread:
            try:
                self.controller_thread.stop()
                self.controller_thread.wait()
            except:
                pass
            self.controller_thread = None
        self.camera_image_label.clear()
        self.camera_image_label.setText("请先追踪再来照镜子哦~")
        self.camera_image_label.setStyleSheet("background-color: #F8F9FA; border-radius: 8px; font-size: 16px; color: #666666;")
    def set_process_priority(self):
        priority_mapping = {
            0: ("IDLE_PRIORITY_CLASS", 0x00000040),
            1: ("BELOW_NORMAL_PRIORITY_CLASS", 0x00004000),
            2: ("NORMAL_PRIORITY_CLASS", 0x00000020),
            3: ("ABOVE_NORMAL_PRIORITY_CLASS", 0x00008000),
            4: ("HIGH_PRIORITY_CLASS", 0x00000080),
            5: ("REALTIME_PRIORITY_CLASS", 0x00000100)
        }
        idx = self.priority_selection.currentIndex()
        key, value = priority_mapping[idx]
        try:
            current_pid = os.getpid()
            handle = windll.kernel32.OpenProcess(0x0200 | 0x0400, False, current_pid)
            windll.kernel32.SetPriorityClass(handle, value)
            windll.kernel32.CloseHandle(handle)
            g.config["Setting"]["priority"] = key
            g.save_configs()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置优先级失败：{str(e)}")
    def set_render_device(self, index):
        if index == 0:
            g.config["Setting"]["render_device"] = "auto"
        elif index == 1:
            g.config["Setting"]["render_device"] = "cpu"
        elif index == 2:
            g.config["Setting"]["render_device"] = "gpu"
        else:
            gpu_index = index - 3
            g.config["Setting"]["render_device"] = f"gpu_{gpu_index}"
        g.save_configs()
        if hasattr(g, 'face_detector'):
            g.face_detector = None
        if hasattr(g, 'hand_detector'):
            g.hand_detector = None
        if hasattr(g, 'tongue_model'):
            g.tongue_model = None
        if hasattr(g, 'pose_detector'):
            g.pose_detector = None
        if hasattr(self, 'video_thread') and self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
            self.start_video_capture()
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "确认", "确定退出？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.thread_stopped()
            stop_hotkeys()
            g.save_configs()
            event.accept()
        else:
            event.ignore()
if __name__ == "__main__":
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    app.setStyle("Fusion")
    splash = LoadingScreen()
    splash.show()
    splash.update_status("初始化应用程序...")
    try:
        window = VideoWindow(splash_screen=splash)
        splash.finish(window)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        splash.close()
        QMessageBox.critical(None, "致命错误", f"程序启动失败：{str(e)}")
        sys.exit(1)
