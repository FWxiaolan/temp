import sys
import cv2
import multiprocessing
from multiprocessing import Process, Queue, Event, Manager
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                               QVBoxLayout, QWidget, QHBoxLayout, QSpinBox, 
                               QComboBox, QMessageBox, QDialog, QProgressDialog)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import QTimer, Qt, Slot
import os

# 尝试导入 ultralytics，如果未安装则提示
try:
    from ultralytics import YOLO
except ImportError:
    print("错误：请安装 ultralytics 库 (pip install ultralytics)")
    sys.exit(1)

# ==============================================================================
# 配置部分
# ==============================================================================
MODEL_PATH = "yolo26n-face.pt" 
DEFAULT_IMG_SZ = 320
CONF_THRESHOLD = 0.35
MAX_DETECTIONS = 100

# ==============================================================================
# 子进程工作函数 v2 (带初始化完成通知)
# ==============================================================================
def worker_process_v2(camera_index, model_path, frame_queue, raw_queue, 
                      stop_event, manager_dict, init_queue):
    """
    在独立进程中运行摄像头捕获和 YOLO 推理
    :param camera_index: 摄像头索引
    :param model_path: 模型路径
    :param frame_queue: 存放带框图像帧的队列
    :param raw_queue:   存放原始图像帧的队列
    :param stop_event:  停止信号
    :param manager_dict: 共享字典，存储最新检测结果
    :param init_queue:   初始化完成队列，发送 True/False 表示成功/失败
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_index}")
        init_queue.put(False)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 加载模型
    try:
        if not os.path.exists(model_path):
            print(f"错误：模型文件 {model_path} 不存在")
            init_queue.put(False)
            cap.release()
            return
        model = YOLO(model_path)
    except Exception as e:
        print(f"模型加载失败：{e}")
        init_queue.put(False)
        cap.release()
        return

    # 初始化成功，通知主进程
    init_queue.put(True)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = model(frame, imgsz=DEFAULT_IMG_SZ, conf=CONF_THRESHOLD,
                            max_det=MAX_DETECTIONS, verbose=False, device='cpu')
            result = results[0]
            boxes = result.boxes
            
            frame_display = frame.copy()
            current_detections = []

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Face {conf:.2f}"
                    cv2.putText(frame_display, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    current_detections.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'conf': conf
                    })
            
            manager_dict['latest_detections'] = current_detections
            
            if raw_queue.full():
                try: raw_queue.get_nowait()
                except: pass
            raw_queue.put(frame)
            
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except: pass
            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            frame_queue.put(frame_rgb)

        except Exception as e:
            print(f"推理错误：{e}")
            if raw_queue.full():
                try: raw_queue.get_nowait()
                except: pass
            raw_queue.put(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except: pass
            frame_queue.put(frame_rgb)

    cap.release()
    print("推理进程已停止")

# ==============================================================================
# 点名弹窗类
# ==============================================================================
class FacePopupDialog(QDialog):
    def __init__(self, face_image_np, parent=None):
        super().__init__(parent)
        self.setWindowTitle("被点名同学")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.resize(300, 400)
        
        layout = QVBoxLayout()
        h, w, ch = face_image_np.shape
        bytes_per_line = ch * w
        q_img = QImage(face_image_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        self.label = QLabel()
        self.label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.label)
        self.setLayout(layout)

# ==============================================================================
# 主窗口类
# ==============================================================================
class FaceRollCallApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("学生人脸点名器 (CPU 优化版)")
        self.showMaximized()  # 改为最大化，而不是全屏

        # 多进程相关
        self.worker_process = None
        self.stop_event = Event()
        self.frame_queue = Queue(maxsize=1)
        self.raw_queue = Queue(maxsize=1)
        self.init_queue = Queue(maxsize=1)      # 新增：初始化完成通知队列
        self.manager = Manager()
        self.shared_dict = self.manager.dict()
        self.shared_dict['latest_detections'] = []
        
        # 定时器用于更新画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.is_running = False
        self.selected_cam_index = 0

        # 初始化 UI
        self.init_ui()

        # 启动初始化过程（显示进度条，等待子进程就绪）
        self.start_initialization()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 视频显示区域
        self.video_label = QLabel("正在初始化...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: #000; color: #fff; border: 1px solid #555; }")
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label, 1)

        # 控制区域
        control_layout = QHBoxLayout()
        
        # 摄像头选择（初始化完成后禁用）
        self.cam_combo = QComboBox()
        self.cam_combo.setEditable(False)
        self.scan_cameras()
        control_layout.addWidget(QLabel("摄像头:"))
        control_layout.addWidget(self.cam_combo)

        # 选取人数
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 50)
        self.count_spin.setValue(5)
        control_layout.addWidget(QLabel("选取人数:"))
        control_layout.addWidget(self.count_spin)

        # 点名按钮（初始化完成后启用）
        self.roll_call_btn = QPushButton("开始点名")
        self.roll_call_btn.clicked.connect(self.start_roll_call)
        self.roll_call_btn.setEnabled(False)
        self.roll_call_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 5px;")
        control_layout.addWidget(self.roll_call_btn)

        main_layout.addLayout(control_layout)

    def scan_cameras(self):
        """扫描可用的摄像头索引"""
        self.cam_combo.clear()
        found = False
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cam_combo.addItem(f"Camera {i}", i)
                    found = True
                cap.release()
        if not found:
            self.cam_combo.addItem("未检测到摄像头", -1)

    def start_initialization(self):
        """显示进度条，启动子进程，等待初始化完成"""
        # 创建进度条对话框（模态）
        self.progress = QProgressDialog("正在初始化，请稍候...", None, 0, 0, self)
        self.progress.setWindowTitle("启动中")
        self.progress.setModal(True)
        self.progress.setCancelButton(None)          # 禁用取消按钮
        self.progress.show()

        # 选择第一个可用摄像头
        index = -1
        for i in range(self.cam_combo.count()):
            data = self.cam_combo.itemData(i)
            if data != -1:
                index = i
                break
        if index == -1:
            self.progress.close()
            QMessageBox.critical(self, "错误", "未检测到可用摄像头，程序将退出。")
            sys.exit(1)
        
        self.cam_combo.setCurrentIndex(index)
        self.selected_cam_index = self.cam_combo.currentData()

        # 启动子进程
        self.stop_event.clear()
        self.worker_process = Process(
            target=worker_process_v2,
            args=(self.selected_cam_index, MODEL_PATH, self.frame_queue,
                  self.raw_queue, self.stop_event, self.shared_dict, self.init_queue)
        )
        self.worker_process.start()

        # 设置定时器检查初始化队列
        self.init_check_timer = QTimer()
        self.init_check_timer.timeout.connect(self.check_initialization)
        self.init_check_timer.start(100)  # 每100ms检查一次

    def check_initialization(self):
        """检查子进程是否完成初始化"""
        if not self.init_queue.empty():
            success = self.init_queue.get_nowait()
            self.init_check_timer.stop()
            self.progress.close()
            
            if success:
                # 初始化成功，启用点名按钮，禁用摄像头选择
                self.is_running = True
                self.roll_call_btn.setEnabled(True)
                self.cam_combo.setEnabled(False)
                self.video_label.setText("")  # 清空文字，等待画面
            else:
                # 初始化失败，停止子进程并退出
                self.stop_monitor()
                QMessageBox.critical(self, "错误", "初始化失败（摄像头或模型加载错误），程序将退出。")
                sys.exit(1)

    def stop_monitor(self):
        """停止子进程（供关闭时调用）"""
        if self.is_running:
            self.stop_event.set()
            if self.worker_process:
                self.worker_process.join(timeout=2)
                if self.worker_process.is_alive():
                    self.worker_process.terminate()
            self.is_running = False
            self.roll_call_btn.setEnabled(False)
            self.cam_combo.setEnabled(True)

    @Slot()
    def update_frame(self):
        """从队列获取最新帧并显示"""
        if self.is_running and not self.frame_queue.empty():
            try:
                frame_rgb = self.frame_queue.get_nowait()
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled_pixmap)
            except Exception:
                pass

    def start_roll_call(self):
        """点名逻辑"""
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

        for i, det in enumerate(selected_detections):
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            margin = 10
            h, w, _ = raw_frame.shape
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            face_crop_bgr = raw_frame[y1:y2, x1:x2]
            if face_crop_bgr.size > 0:
                face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                popup = FacePopupDialog(face_crop_rgb, self)
                offset = i * 50
                popup.move(100 + offset, 100 + offset)
                popup.show()
        
        QMessageBox.information(self, "点名完成", f"已从当前画面选取 {actual_count} 位同学")

    def closeEvent(self, event):
        self.stop_monitor()
        event.accept()

# ==============================================================================
# 程序入口
# ==============================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    window = FaceRollCallApp()
    window.show()
    
    sys.exit(app.exec())