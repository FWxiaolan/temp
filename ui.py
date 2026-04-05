import sys
import cv2
import time
import platform
from PySide6.QtWidgets import (QMainWindow, QLabel, QPushButton,
                               QVBoxLayout, QWidget, QHBoxLayout, QSpinBox,
                               QComboBox, QMessageBox, QDialog, QScrollArea, QStatusBar)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt, Slot

# 尝试导入 psutil 获取 CPU 占用率
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("提示：安装 psutil 可显示 CPU 占用率 (pip install psutil)")

# ==============================================================================
# Windows 下枚举摄像头名称（使用 WMI）
# ==============================================================================
def enumerate_cameras_windows():
    """返回 [(index, name), ...] 列表，按索引顺序"""
    try:
        import wmi
        c = wmi.WMI()
        # 获取所有视频输入设备
        devices = c.Win32_PnPEntity(PNPClass="Image", ConfigManagerErrorCode=0)
        # 提取友好名称，并尝试排序（按 VID/PID 或名称）
        camera_names = {}
        for dev in devices:
            if dev.Name and 'camera' in dev.Name.lower() or 'webcam' in dev.Name.lower() or 'video' in dev.Name.lower():
                # 尝试提取设备实例 ID 以获取索引线索（但不可靠）
                # 更可靠的方式：逐个尝试 open
                pass

        # 实际上，WMI 无法直接给出 OpenCV 的索引映射
        # 因此我们采用“探测法”：尝试打开每个索引，成功则记录
        # 同时用 WMI 名称按顺序匹配（近似）
        wmi_names = [dev.Name for dev in devices if dev.Name]
        
        # 探测可用索引
        available = []
        max_test = 5  # 最多测试 0~4
        cap = None
        for i in range(max_test):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # 使用 DirectShow 更快
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        # 尝试从 WMI 名称中分配一个名字
                        name = wmi_names[i] if i < len(wmi_names) else f"Camera {i}"
                        available.append((i, name))
                    cap.release()
                else:
                    break
            except Exception:
                if cap:
                    cap.release()
                break
        return available
    except Exception as e:
        print(f"摄像头枚举失败（回退到数字ID）: {e}")
        # 回退：探测前几个索引
        available = []
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append((i, f"Camera {i}"))
                cap.release()
            else:
                break
        return available

# 非 Windows 系统：使用数字 ID
def enumerate_cameras_fallback():
    available = []
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append((i, f"Camera {i}"))
            cap.release()
        else:
            break
    return available

# 统一接口
def get_available_cameras():
    if platform.system() == "Windows":
        return enumerate_cameras_windows()
    else:
        return enumerate_cameras_fallback()

# ==============================================================================
# 带标记图像显示弹窗
# ==============================================================================
class MarkedImageDialog(QDialog):
    def __init__(self, marked_image_np, selected_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"点名结果 - 已标记 {selected_count} 位同学")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.resize(800, 600)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        h, w, ch = marked_image_np.shape
        bytes_per_line = ch * w
        q_img = QImage(marked_image_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        label = QLabel()
        label.setPixmap(scaled_pixmap)
        label.setAlignment(Qt.AlignCenter)
        
        scroll.setWidget(label)
        
        layout = QVBoxLayout()
        layout.addWidget(scroll)
        self.setLayout(layout)

# ==============================================================================
# 主窗口类
# ==============================================================================
class FaceRollCallApp(QMainWindow):
    def __init__(self, frame_queue, raw_queue, shared_dict, stop_event, worker_process, selected_cam_index, available_cams):
        # available_cams now is list of (index, name)
        super().__init__()
        self.setWindowTitle("学生人脸点名器 FaceRollCall (Github: hexo141)")
        self.showMaximized()
        
        # 接收外部传入的资源
        self.frame_queue = frame_queue
        self.raw_queue = raw_queue
        self.shared_dict = shared_dict
        self.stop_event = stop_event
        self.worker_process = worker_process
        self.is_running = True
        self.selected_cam_index = selected_cam_index

        # 初始化共享字典中的摄像头索引
        if 'cam_index' not in self.shared_dict:
            self.shared_dict['cam_index'] = selected_cam_index

        # FPS 统计
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)

        # 初始化 UI
        self.init_ui(available_cams)

    def init_ui(self, available_cams):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.video_label = QLabel("等待视频流...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: #000; color: #fff; border: 1px solid #555; }")
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label, 1)

        control_layout = QHBoxLayout()
        
        # 摄像头选择下拉框
        self.cam_combo = QComboBox()
        self.cam_combo.setEditable(False)
        
        # available_cams 是 [(index, name), ...]
        for idx, name in available_cams:
            self.cam_combo.addItem(name, idx)  # userData = index
        
        # 设置初始选中项
        initial_index = -1
        for i, (idx, _) in enumerate(available_cams):
            if idx == self.selected_cam_index:
                initial_index = i
                break
        if initial_index != -1:
            self.cam_combo.setCurrentIndex(initial_index)
        
        self.cam_combo.setEnabled(True)
        self.cam_combo.currentIndexChanged.connect(self.on_camera_changed)
        
        control_layout.addWidget(QLabel("摄像头: "))
        control_layout.addWidget(self.cam_combo)

        # 选取人数
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 50)
        self.count_spin.setValue(5)
        control_layout.addWidget(QLabel("选取人数: "))
        control_layout.addWidget(self.count_spin)

        # 点名按钮
        self.roll_call_btn = QPushButton("开始点名")
        self.roll_call_btn.clicked.connect(self.start_roll_call)
        self.roll_call_btn.setEnabled(True)
        self.roll_call_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 5px; ")
        control_layout.addWidget(self.roll_call_btn)

        main_layout.addLayout(control_layout)

        # 状态栏
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self.status_label = QLabel("初始化...")
        status_bar.addWidget(self.status_label)

    @Slot(int)
    def on_camera_changed(self, index):
        if index < 0:
            return
        cam_id = self.cam_combo.itemData(index)  # 这是整数索引
        print(f"UI: 请求切换到摄像头 {cam_id} ({self.cam_combo.currentText()})")
        self.shared_dict['cam_index'] = cam_id
        self.video_label.clear()
        self.video_label.setText("切换摄像头中...")

    def update_status(self):
        detections = self.shared_dict.get('latest_detections', [])
        face_count = len(detections)
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_str = f"CPU: {cpu_percent:.1f}%"
        else:
            cpu_str = "CPU: N/A"
        status_text = f"FPS: {self.current_fps:.1f}  |  可抽取人数: {face_count}  |  {cpu_str}"
        self.status_label.setText(status_text)

    @Slot()
    def update_frame(self):
        if self.is_running and not self.frame_queue.empty():
            try:
                frame_rgb = self.frame_queue.get_nowait()
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled_pixmap)
                
                self.frame_count += 1
                now = time.time()
                elapsed = now - self.last_fps_update
                if elapsed >= 1.0:
                    self.current_fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_fps_update = now
            except Exception:
                pass

    def start_roll_call(self):
        if not self.is_running:
            return

        detections = self.shared_dict.get('latest_detections', [])
        if len(detections) == 0:
            QMessageBox.information(self, "提示", "当前画面未检测到人脸")
            return

        raw_frame = None
        if not self.raw_queue.empty():
            try:
                raw_frame = self.raw_queue.get_nowait()
            except Exception:
                pass
        
        if raw_frame is None:
            QMessageBox.warning(self, "提示", "尚未获取到画面，请稍后")
            return

        num_to_select = self.count_spin.value()
        actual_count = min(len(detections), num_to_select)
        sorted_detections = sorted(detections, key=lambda k: k['conf'], reverse=True)
        selected_detections = sorted_detections[:actual_count]

        marked_frame = raw_frame.copy()
        for det in detections:
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for det in selected_detections:
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        marked_frame_rgb = cv2.cvtColor(marked_frame, cv2.COLOR_BGR2RGB)
        dialog = MarkedImageDialog(marked_frame_rgb, actual_count, self)
        dialog.exec()
        QMessageBox.information(self, "点名完成", f"已从当前画面标记 {actual_count} 位同学")

    def closeEvent(self, event):
        if self.is_running:
            self.stop_event.set()
            if self.worker_process:
                self.worker_process.join(timeout=2)
                if self.worker_process.is_alive():
                    self.worker_process.terminate()
            self.is_running = False
        event.accept()
