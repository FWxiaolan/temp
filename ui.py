import sys
import cv2
import time
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

        # 初始化共享字典中的摄像头索引，确保子进程知道初始状态
        if 'cam_index' not in self.shared_dict:
            self.shared_dict['cam_index'] = selected_cam_index

        # FPS 统计相关
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0

        # 定时器用于更新画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)   # 约 33ms 一帧

        # 状态栏信息定时器（每秒更新一次）
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)   # 每秒更新一次

        # 初始化 UI（必须在定时器启动之后，但无妨）
        self.init_ui(available_cams)

    def init_ui(self, available_cams):
        # ---------- 底部状态栏（必须最先创建，避免后续异常导致缺失） ----------
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self.status_label = QLabel("初始化...")
        status_bar.addWidget(self.status_label)

        # ---------- 处理 available_cams 的多种可能格式 ----------
        # 期望格式: [(display_name, camera_index), ...]
        normalized_cams = []
        if isinstance(available_cams, int):
            # 如果传入了单个整数
            normalized_cams = [(f"Camera {available_cams}", available_cams)]
        elif isinstance(available_cams, list):
            if len(available_cams) == 0:
                # 空列表，至少提供一个默认摄像头
                normalized_cams = [("Camera 0", 0)]
            else:
                first = available_cams[0]
                if isinstance(first, int):
                    # 列表元素为整数：如 [0,1,2]
                    for idx in available_cams:
                        normalized_cams.append((f"Camera {idx}", idx))
                elif isinstance(first, tuple) and len(first) == 2:
                    # 已经是正确格式
                    normalized_cams = available_cams
                else:
                    # 未知格式，回退到索引0
                    normalized_cams = [("Camera 0", 0)]
        else:
            # 其他类型，回退
            normalized_cams = [("Camera 0", 0)]

        # 确保选中的摄像头索引存在于列表中，否则使用第一个
        found = any(idx == self.selected_cam_index for _, idx in normalized_cams)
        if not found and len(normalized_cams) > 0:
            self.selected_cam_index = normalized_cams[0][1]
            # 同时更新共享字典
            self.shared_dict['cam_index'] = self.selected_cam_index

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 视频显示区域
        self.video_label = QLabel("等待视频流...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: #000; color: #fff; border: 1px solid #555; }")
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label, 1)

        # 控制区域
        control_layout = QHBoxLayout()

        # 摄像头选择（使用真实名称）
        self.cam_combo = QComboBox()
        self.cam_combo.setEditable(False)
        for name, idx in normalized_cams:
            self.cam_combo.addItem(name, idx)

        # 设置初始选择
        for i, (_, idx) in enumerate(normalized_cams):
            if idx == self.selected_cam_index:
                self.cam_combo.setCurrentIndex(i)
                break

        # 启用摄像头切换，并连接信号
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

    @Slot(int)
    def on_camera_changed(self, index):
        """当用户在下拉框中选择不同摄像头时触发"""
        if index >= 0:                     # 修复原错误：原代码写成了 if index = 1.0:
            new_cam_index = self.cam_combo.currentData()
            # 更新共享字典，子进程会读取该值切换摄像头
            self.shared_dict['cam_index'] = new_cam_index

    def update_frame(self):
        """从队列获取检测结果并显示画面"""
        if not self.is_running:
            return

        try:
            # 获取带检测框的画面
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                if frame is not None:
                    # 更新 FPS 计数
                    self.frame_count += 1
                    # 转换格式并显示
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    # 缩放以适应 label 大小，保持宽高比
                    scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.video_label.setPixmap(scaled_pixmap)
        except Exception:
            # 忽略可能的队列空异常
            pass

    def update_status(self):
        """更新底部状态栏信息：FPS、可抽取人数、CPU 占用"""
        # 确保 self.status_label 存在（防御）
        if not hasattr(self, 'status_label'):
            return

        detections = self.shared_dict.get('latest_detections', [])
        face_count = len(detections)

        # 计算 FPS
        now = time.time()
        elapsed = now - self.last_fps_update
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = now

        # CPU 占用
        cpu_str = ""
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_str = f" | CPU: {cpu_percent:.1f}%"

        self.status_label.setText(
            f"FPS: {self.current_fps:.1f}  |  可抽取人数: {face_count}{cpu_str}"
        )

    def start_roll_call(self):
        """点名逻辑 - 截图整个画面并标记被点名的同学"""
        if not self.is_running:
            return

        detections = self.shared_dict.get('latest_detections', [])
        if len(detections) == 0:
            QMessageBox.information(self, "提示", "当前画面未检测到人脸")
            return

        # 获取原始画面（无检测框）
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
        # 按置信度排序，取前 actual_count 个
        sorted_detections = sorted(detections, key=lambda k: k['conf'], reverse=True)
        selected_detections = sorted_detections[:actual_count]

        # 复制原始画面，准备绘制标记
        marked_frame = raw_frame.copy()
        # 在画面上绘制所有检测框（绿色细框）
        for det in detections:
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 对被选中的同学绘制红色粗框
        for det in selected_detections:
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        marked_frame_rgb = cv2.cvtColor(marked_frame, cv2.COLOR_BGR2RGB)

        dialog = MarkedImageDialog(marked_frame_rgb, actual_count, self)
        dialog.exec()

        QMessageBox.information(self, "点名完成", f"已从当前画面标记 {actual_count} 位同学")

    def closeEvent(self, event):
        """关闭窗口时停止子进程"""
        if self.is_running:
            self.stop_event.set()
            if self.worker_process:
                self.worker_process.join(timeout=2)
                if self.worker_process.is_alive():
                    self.worker_process.terminate()
            self.is_running = False
        event.accept()
