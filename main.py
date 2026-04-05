import sys
import multiprocessing
from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                               QLabel, QProgressBar, QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

# 延迟导入，以便先显示进度条
# 注意：我们需要先导入 wmi，如果没安装会自动处理
try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False

class CustomProgressDialog(QDialog):
    def __init__(self, title="初始化中", label_text="正在启动程序，请稍候...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(400, 85)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 文本标签
        self.text_label = QLabel(label_text)
        layout.addWidget(self.text_label)
        
        # 水平布局：箭头图标 + 进度条
        h_layout = QHBoxLayout()
        h_layout.setSpacing(8)
        
        # 创建图标标签
        self.icon_label = QLabel()
        try:
            pixmap = QPixmap("arrow_right.png")
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.icon_label.setPixmap(scaled_pixmap)
            else:
                self.icon_label.setText("→")
                self.icon_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        except:
            self.icon_label.setText("→")
            self.icon_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        
        h_layout.addWidget(self.icon_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        h_layout.addWidget(self.progress_bar)
        
        layout.addLayout(h_layout)
        
    def setValue(self, value):
        self.progress_bar.setValue(value)
        QApplication.processEvents()
        
    def setLabelText(self, text):
        self.text_label.setText(text)
        QApplication.processEvents()

def dynamic_imports(progress_callback):
    progress_callback(10, "正在导入 OpenCV...")
    import cv2
    progress_callback(30, "OpenCV 导入完成")

    progress_callback(40, "正在准备其他模块...")
    import os
    from multiprocessing import Process, Queue, Event, Manager
    progress_callback(50, "基础模块准备完成")

    return cv2, os, Process, Queue, Event, Manager

def get_camera_names(cv2, max_id=10):
    """
    终极方案：使用 WMI 获取设备 ID 进行精确匹配
    """
    cameras = []
    
    if not WMI_AVAILABLE:
        # 如果没有 wmi 库，回退到数字
        for i in range(max_id):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.release()
                cameras.append((i, f"摄像头 {i} (无WMI支持)"))
        return cameras

    try:
        c = wmi.WMI()
        # 获取所有视频捕获设备
        wmi_devices = c.Win32_PnPEntity(PNPClass='Image')
        
        # 构建一个字典：键是设备 ID 的部分信息，值是设备名称
        device_map = {}
        for dev in wmi_devices:
            if dev.DeviceID and dev.Name:
                # 提取关键标识符（VID/PID）
                key = dev.DeviceID.split("\\")[-1].split("&")[0] # 粗略提取
                device_map[key] = dev.Name
        
        # 探测 OpenCV 摄像头
        for i in range(max_id):
            # 使用 DirectShow 后端
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                continue
                
            # 尝试获取友好名称（OpenCV 4.7+）
            name = None
            try:
                prop_id = getattr(cv2, 'CAP_PROP_DEVICE_FRIENDLY_NAME', None)
                if prop_id is not None:
                    friendly_name = cap.get(prop_id)
                    if isinstance(friendly_name, str) and friendly_name.strip():
                        name = friendly_name
            except:
                pass
            
            # 如果 OpenCV 没获取到，尝试 WMI 匹配
            if not name:
                # 获取当前 OpenCV 句柄的属性（这里我们尝试获取设备路径）
                # 注意：OpenCV 本身不直接暴露设备路径，所以我们用一个“笨办法”：
                # 我们假设按顺序打开的设备对应 WMI 列表中的顺序
                # 或者我们直接使用 WMI 中的名称，按索引对应
                if i < len(device_map):
                    # 这是一个简单的映射，虽然不完美但比纯数字好
                    name = list(device_map.values())[i]
                else:
                    name = None
                    
            # 最终兜底
            if not name:
                name = f"摄像头 {i} (请检查驱动)"
                
            cameras.append((i, name))
            cap.release()
            
    except Exception as e:
        print(f"摄像头探测异常: {e}")
        # 异常回退
        for i in range(max_id):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.release()
                cameras.append((i, f"摄像头 {i} (探测失败)"))
    
    return cameras

def main():
    multiprocessing.freeze_support()
    
    app = QApplication(sys.argv)
    
    progress = CustomProgressDialog("初始化中", "正在启动程序，请稍候...")
    progress.show()
    
    def update_progress(value, text):
        progress.setValue(value)
        progress.setLabelText(text)
    
    try:
        cv2, os, Process, Queue, Event, Manager = dynamic_imports(update_progress)
        
        update_progress(55, "检查模型文件夹...")
        MODEL_PATH = "./yolo26n-face_openvino_model"
        if not os.path.exists(MODEL_PATH):
            progress.close()
            QMessageBox.critical(None, "错误", f"模型文件夹 {MODEL_PATH} 不存在，请放入程序目录。")
            sys.exit(1)
        
        update_progress(60, "正在扫描摄像头...")
        available_cams = get_camera_names(cv2, max_id=5)
        
        # 调试输出
        print("=== 扫描结果 ===")
        for idx, name in available_cams:
            print(f"摄像头索引 {idx}: {name}")
        
        if not available_cams:
            progress.close()
            QMessageBox.critical(None, "错误", "未检测到可用摄像头，程序将退出。")
            sys.exit(1)
            
        selected_cam = available_cams[0][0]
        update_progress(70, f"已选择: {available_cams[0][1]}")
        
        update_progress(75, "准备多进程通信...")
        frame_queue = Queue(maxsize=1)
        raw_queue = Queue(maxsize=1)
        init_queue = Queue(maxsize=1)
        stop_event = Event()
        manager = Manager()
        shared_dict = manager.dict()
        shared_dict['latest_detections'] = []
        
        update_progress(80, "启动人脸检测引擎...")
        from core import worker_process_v2, MODEL_PATH as CORE_MODEL_PATH
        worker = Process(
            target=worker_process_v2,
            args=(selected_cam, CORE_MODEL_PATH, frame_queue,
                  raw_queue, stop_event, shared_dict, init_queue)
        )
        worker.start()
        
        update_progress(85, "等待摄像头和模型加载...")
        import time
        timeout = 60
        start_time = time.time()
        while init_queue.empty():
            if time.time() - start_time > timeout:
                progress.close()
                QMessageBox.critical(None, "错误", "初始化超时，请检查摄像头或模型文件。")
                worker.terminate()
                sys.exit(1)
            QApplication.processEvents()
            time.sleep(0.05)
        init_success = init_queue.get()
        if not init_success:
            progress.close()
            QMessageBox.critical(None, "错误", "子进程初始化失败（摄像头打开或模型加载错误）。")
            worker.terminate()
            sys.exit(1)
        
        update_progress(95, "准备界面...")
        
        from ui import FaceRollCallApp
        window = FaceRollCallApp(
            frame_queue=frame_queue,
            raw_queue=raw_queue,
            shared_dict=shared_dict,
            stop_event=stop_event,
            worker_process=worker,
            selected_cam_index=selected_cam,
            available_cams=available_cams
        )
        
        update_progress(100, "启动完成")
        progress.close()
        
        window.show()
        sys.exit(app.exec())
        
    except Exception as e:
        progress.close()
        QMessageBox.critical(None, "初始化失败", f"发生错误：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
