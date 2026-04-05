import sys
import cv2
import os
import time
import threading
import queue
import numpy as np
from multiprocessing import Queue, Event, Manager

# OpenVINO 用于直接推理
try:
    import openvino as ov
    OV_AVAILABLE = True
except ImportError:
    OV_AVAILABLE = False
    print("错误：请安装 openvino (pip install openvino)")

# ============================================================================== 
# 配置部分
# ==============================================================================
DEFAULT_IMG_SZ = 640                     # 推理图像尺寸
CONF_THRESHOLD = 0.30                    # 置信度阈值
MAX_DETECTIONS = 100                     # 最大检测数量


# ==============================================================================
# 子进程工作函数 v3 (支持热切换摄像头版)
# ==============================================================================
def worker_process_v2(camera_index, model_path, frame_queue, raw_queue,
                      stop_event, manager_dict, init_queue):
    """
    在独立进程中运行摄像头捕获和 YOLO 推理。
    支持通过 manager_dict['cam_index'] 动态切换摄像头。
    """
    
    # ---------- 加载模型（一次性加载，全程复用） ----------



    try:
        core = ov.Core()
        ov_model = core.read_model(os.path.join(MODEL_PATH, "yolo26n-face.xml"))
        device = "GPU" if "GPU" in core.available_devices else "CPU"
        print("Use device " + device)
        compiled_model = core.compile_model(ov_model, device)
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        print("使用 OpenVINO 直接推理")
    except Exception as e:
        print(f"OpenVINO 模型加载失败：{e}")
        init_queue.put(False)
        return

    # 初始化成功
    init_queue.put(True)

    # ---------- 预处理函数 ----------
    def preprocess_frame(frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (DEFAULT_IMG_SZ, DEFAULT_IMG_SZ))
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0).astype(np.float32)
        return img

    # ---------- 后处理函数 ----------
    def postprocess_output(output, frame_shape, conf_threshold=CONF_THRESHOLD):
        output = np.squeeze(output)
        h, w = frame_shape[:2]
        scale_x = w / DEFAULT_IMG_SZ
        scale_y = h / DEFAULT_IMG_SZ
        detections = []
        
        if len(output.shape) == 1:
            output = output.reshape(1, -1)
            
        for det in output:
            if len(det) < 5: continue
            val1, val2, val3, val4, conf = det[0], det[1], det[2], det[3], det[4]
            if conf < conf_threshold: continue
            
            x1, y1, x2, y2 = 0, 0, 0, 0
            if val3 < 2.0: 
                cx, cy, bw, bh = val1, val2, val3, val4
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
            else:
                if val3 < val1: 
                    cx, cy, bw, bh = val1, val2, val3, val4
                    x1 = int((cx - bw / 2) * scale_x)
                    y1 = int((cy - bh / 2) * scale_y)
                    x2 = int((cx + bw / 2) * scale_x)
                    y2 = int((cy + bh / 2) * scale_y)
                else:
                    x1 = int(val1 * scale_x)
                    y1 = int(val2 * scale_y)
                    x2 = int(val3 * scale_x) 
                    y2 = int(val4 * scale_y)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                detections.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': float(conf)})
        return detections

    # ---------- OpenVINO 异步推理队列 ----------
    infer_queue = ov.AsyncInferQueue(compiled_model, 4)

    def inference_callback(infer_request, user_data):
        frame = user_data
        try:
            output = infer_request.get_output_tensor(output_layer.index).data
            detections = postprocess_output(output, frame.shape, CONF_THRESHOLD)

            frame_display = frame.copy()
            for det in detections:
                x1, y1, x2, y2, conf = det['x1'], det['y1'], det['x2'], det['y2'], det['conf']
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Face {conf:.2f}"
                cv2.putText(frame_display, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            manager_dict['latest_detections'] = detections

            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            try:
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put(frame_rgb)
            except:
                pass
        except Exception as e:
            print(f"推理错误：{e}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put(frame_rgb)
            except:
                pass

    infer_queue.set_callback(inference_callback)

    # ---------- 内部线程控制 ----------
    capture_stop = threading.Event()
    inference_stop = threading.Event()
    inference_queue = queue.Queue(maxsize=1)
    
    # 捕获线程
    def capture_loop(cap):
        while not capture_stop.is_set() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                if raw_queue.full():
                    raw_queue.get_nowait()
                raw_queue.put(frame)
            except:
                pass
            try:
                if inference_queue.full():
                    inference_queue.get_nowait()
                inference_queue.put(frame)
            except:
                pass
        print("捕获线程退出")

    # 推理线程
    def inference_loop():
        while not inference_stop.is_set() and not stop_event.is_set():
            try:
                frame = inference_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                input_data = preprocess_frame(frame)
                infer_queue.start_async({input_layer.any_name: input_data}, frame)
            except Exception as e:
                print(f"推理启动错误：{e}")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    if frame_queue.full():
                        frame_queue.get_nowait()
                    frame_queue.put(frame_rgb)
                except:
                    pass
        print("推理线程退出")

    # ==========================================================================
    # 主循环：监听摄像头切换
    # ==========================================================================
    current_cam_idx = camera_index
    
    while not stop_event.is_set():
        # 1. 检查是否需要切换摄像头
        target_cam_idx = manager_dict.get('cam_index', current_cam_idx)
        
        if target_cam_idx != current_cam_idx:
            print(f"检测到摄像头切换请求: {current_cam_idx} -> {target_cam_idx}")
            # 停止当前线程
            capture_stop.set()
            inference_stop.set()
            # 重置事件
            capture_stop.clear()
            inference_stop.clear()
            # 更新当前索引
            current_cam_idx = target_cam_idx
            print(f"正在初始化摄像头 {current_cam_idx} ...")

        # 2. 初始化摄像头
        cap = cv2.VideoCapture(current_cam_idx)
        if not cap.isOpened():
            print(f"无法打开摄像头 {current_cam_idx}，重试中...")
            manager_dict['latest_detections'] = []
            continue
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        print(f"摄像头 {current_cam_idx} 已打开")

        # 3. 启动线程
        c_thread = threading.Thread(target=capture_loop, args=(cap,), daemon=True)
        i_thread = threading.Thread(target=inference_loop, daemon=True)
        c_thread.start()
        i_thread.start()

        # 4. 等待直到需要切换或停止
        while not stop_event.is_set():
            # 快速轮询检查是否有切换请求
            new_target = manager_dict.get('cam_index', current_cam_idx)
            if new_target != current_cam_idx:
                break

        # 5. 清理当前摄像头资源，准备下一轮循环
        capture_stop.set()
        inference_stop.set()
        c_thread.join(timeout=1)
        i_thread.join(timeout=1)
        cap.release()
        print(f"摄像头 {current_cam_idx} 已释放")

    print("推理进程已完全停止")
    sys.exit(0)
