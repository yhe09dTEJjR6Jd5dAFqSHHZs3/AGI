import sys
import os
import subprocess
import time
import threading
import queue
import datetime
import shutil
import ctypes
import random
import glob
import bisect

def install_requirements():
    required = [
        "torch", "torchvision", "numpy", "opencv-python", "mss", 
        "pynput", "psutil", "nvidia-ml-py3", "pillow"
    ]
    for package in required:
        try:
            __import__(package.split("==")[0].replace("-", "_").replace("nvidia_ml_py3", "pynvml").replace("pillow", "PIL"))
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except Exception as e:
                print(f"Error installing {package}: {e}")

install_requirements()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import cv2
import mss
import psutil
from pynput import mouse, keyboard
import pynvml

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
base_dir = os.path.join(desktop_path, "AAA")
data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "model")
temp_dir = os.path.join(base_dir, "temp")

for d in [base_dir, data_dir, model_dir, temp_dir]:
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except Exception as e:
            print(f"Error creating directory {d}: {e}")

MODE_LEARNING = "LEARNING"
MODE_SLEEP = "SLEEP"
MODE_TRAINING = "TRAINING"
current_mode = MODE_LEARNING
stop_training_flag = False
flush_event = threading.Event()
flush_done_event = threading.Event()

capture_freq = 10
seq_len = 5
screen_w, screen_h = 2560, 1600
target_w, target_h = 256, 160
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mouse_state = {
    "x": 0, "y": 0,
    "l_down": False, "l_down_ts": 0.0, "l_up_pos": (0,0), "l_up_ts": 0.0,
    "r_down": False, "r_down_ts": 0.0, "r_up_pos": (0,0), "r_up_ts": 0.0,
    "scroll": 0
}
mouse_lock = threading.Lock()
temp_trajectory = []
frame_lock = threading.Lock()
latest_frame = {"img": None, "ts": 0.0}

def update_latest_frame(img, ts):
    with frame_lock:
        latest_frame["img"] = img
        latest_frame["ts"] = ts

def get_latest_frame():
    with frame_lock:
        return latest_frame["img"], latest_frame["ts"]

data_queue = queue.Queue()

class UniversalAI(nn.Module):
    def __init__(self):
        super(UniversalAI, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, target_h, target_w)
            conv_out = self.conv(dummy)
            self.fc_input_dim = conv_out.view(1, -1).size(1)
        self.mouse_dim = 10

        self.lstm = nn.LSTM(input_size=self.fc_input_dim + self.mouse_dim, hidden_size=512, batch_first=True)

        self.fc_action = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, img, mouse_input, hidden=None):
        batch_size, seq, c, h, w = img.size()
        img_reshaped = img.view(batch_size * seq, c, h, w)
        feat = self.conv(img_reshaped)
        feat = feat.view(batch_size, seq, -1)
        
        combined = torch.cat((feat, mouse_input), dim=2)
        
        out, hidden = self.lstm(combined, hidden)

        action = torch.sigmoid(self.fc_action(out))
        return action, hidden

def ensure_initial_model():
    try:
        model_path = os.path.join(model_dir, "ai_model.pth")
        if not os.path.exists(model_path):
            temp_model = UniversalAI()
            torch.save(temp_model.state_dict(), model_path)
            print("Generated initial AI model.")
    except Exception as e:
        print(f"Init model error: {e}")

def get_sys_usage():
    try:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        gpu = 0
        vram = 0
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu = util.gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram = (mem_info.used / mem_info.total) * 100
        except:
            pass
        return max(cpu, mem, gpu, vram)
    except Exception as e:
        print(f"Monitor Error: {e}")
        return 50

def resource_monitor():
    global capture_freq, seq_len
    last_cooldown = 0
    while True:
        try:
            m = get_sys_usage()
            now = time.time()
            if m > 90 and now - last_cooldown >= 5:
                capture_freq = max(1, int(capture_freq * 0.5))
                seq_len = max(1, int(seq_len * 0.5))
                last_cooldown = now
                time.sleep(7)
                continue
            elif m < 10:
                capture_freq = min(100, int(capture_freq * 1.1) + 1)
                seq_len = min(100, int(seq_len * 1.1) + 1)
            elif m >= 40 and m <= 60:
                pass
            elif m < 40 and capture_freq < 100: 
                 capture_freq = min(100, int(capture_freq * 1.05) + 1)
                 seq_len = min(100, int(seq_len * 1.05) + 1)

            time.sleep(1)
        except Exception as e:
            print(f"Resource Monitor Error: {e}")
            time.sleep(1)

def frame_generator_loop():
    try:
        with mss.mss() as sct:
            while True:
                start = time.time()
                img = np.array(sct.grab({"top": 0, "left": 0, "width": screen_w, "height": screen_h}))
                update_latest_frame(img, start)
                desired = max(30, capture_freq * 2)
                wait = (1.0 / desired) - (time.time() - start)
                if wait > 0:
                    time.sleep(wait)
    except Exception as e:
        print(f"Frame Generator Error: {e}")

def on_move(x, y):
    with mouse_lock:
        mouse_state["x"] = x
        mouse_state["y"] = y
        temp_trajectory.append((x, y, time.time()))

def on_click(x, y, button, pressed):
    with mouse_lock:
        t = time.time()
        if button == mouse.Button.left:
            mouse_state["l_down"] = pressed
            if pressed:
                mouse_state["l_down_ts"] = t
            else:
                mouse_state["l_up_pos"] = (x, y)
                mouse_state["l_up_ts"] = t
        elif button == mouse.Button.right:
            mouse_state["r_down"] = pressed
            if pressed:
                mouse_state["r_down_ts"] = t
            else:
                mouse_state["r_up_pos"] = (x, y)
                mouse_state["r_up_ts"] = t

def on_scroll(x, y, dx, dy):
    with mouse_lock:
        mouse_state["scroll"] = dy

def on_press_key(key):
    global current_mode, stop_training_flag
    if current_mode == MODE_TRAINING:
        stop_training_flag = True

def check_disk_space():
    try:
        total_size = 0
        files = []
        for f in glob.glob(os.path.join(data_dir, "*.npy")):
            size = os.path.getsize(f)
            total_size += size
            files.append((f, os.path.getmtime(f)))
        
        limit = 20 * 1024 * 1024 * 1024
        if total_size > limit:
            files.sort(key=lambda x: x[1])
            now = time.time()
            while total_size > limit and files:
                f, mtime = files.pop(0)
                if now - mtime < 2:
                    continue
                try:
                    s = os.path.getsize(f)
                    os.remove(f)
                    total_size -= s
                except:
                    pass
    except Exception as e:
        print(f"Disk clean error: {e}")

class GameDataset(Dataset):
    def __init__(self, file_list, cache_size=3):
        self.file_list = []
        self.file_lengths = []
        self.valid_prefix = []
        self.seq_len = seq_len
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []
        total = 0
        for f in file_list:
            try:
                arr = np.load(f, allow_pickle=True, mmap_mode='r')
                length = len(arr)
                total_valid = max(1, length - self.seq_len + 1)
                if total_valid > 0:
                    self.file_list.append(f)
                    self.file_lengths.append(length)
                    total += total_valid
                    self.valid_prefix.append(total)
            except Exception as e:
                print(f"Load error: {e}")

    def __len__(self):
        if not self.valid_prefix:
            return 0
        return self.valid_prefix[-1]

    def _get_file(self, file_idx):
        path = self.file_list[file_idx]
        if path in self.cache:
            self.cache_order.remove(path)
            self.cache_order.append(path)
            return self.cache[path]
        if len(self.cache_order) >= self.cache_size:
            old_path = self.cache_order.pop(0)
            del self.cache[old_path]
        arr = np.load(path, allow_pickle=True, mmap_mode='r')
        self.cache[path] = arr
        self.cache_order.append(path)
        return arr

    def __getitem__(self, idx):
        try:
            if not self.valid_prefix:
                return torch.zeros((self.seq_len, 3, target_h, target_w)), torch.zeros((self.seq_len, 10)), torch.zeros(4)
            file_pos = bisect.bisect_left(self.valid_prefix, idx)
            start_idx = idx if file_pos == 0 else idx - self.valid_prefix[file_pos - 1]
            data = self._get_file(file_pos)
            slice_data = []
            for i in range(self.seq_len):
                idx_in_file = start_idx + i
                if idx_in_file >= self.file_lengths[file_pos]:
                    idx_in_file = self.file_lengths[file_pos] - 1
                slice_data.append(data[idx_in_file])
            imgs = []
            m_ins = []

            for item in slice_data:
                img_data = item["screen"]
                if isinstance(img_data, bytes):
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                else:
                    img = img_data
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                img = img.transpose(2, 0, 1) / 255.0
                imgs.append(img)

                mx = item["mouse_x"] / screen_w
                my = item["mouse_y"] / screen_h
                m_vec = [
                    mx, my,
                    1.0 if item["l_down"] else 0.0,
                    1.0 if item["r_down"] else 0.0,
                    item["scroll"],
                    0, 0, 0, 0, 0
                ]
                m_ins.append(m_vec)

            last_item = slice_data[-1]
            if start_idx + self.seq_len < self.file_lengths[file_pos]:
                next_item = data[start_idx + self.seq_len]
                target_x = next_item["mouse_x"] / screen_w
                target_y = next_item["mouse_y"] / screen_h

                target_l = 1.0 if next_item["l_down"] else 0.0
                target_r = 1.0 if next_item["r_down"] else 0.0
                labels = [target_x, target_y, target_l, target_r]
            else:
                labels = [last_item["mouse_x"]/screen_w, last_item["mouse_y"]/screen_h, 0.0, 0.0]

            return torch.FloatTensor(np.array(imgs)), torch.FloatTensor(np.array(m_ins)), torch.FloatTensor(np.array(labels))
        except Exception as e:
            print(f"Data load error: {e}")
            return torch.zeros((self.seq_len, 3, target_h, target_w)), torch.zeros((self.seq_len, 10)), torch.zeros(4)

def optimize_ai():
    print("Starting Optimization...")
    model_path = os.path.join(model_dir, "ai_model.pth")
    model = UniversalAI().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except:
            pass

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=device.type == "cuda")
    
    files = []
    now = time.time()
    for f in glob.glob(os.path.join(data_dir, "*.npy")):
        try:
            if now - os.path.getmtime(f) > 1:
                files.append(f)
        except:
            continue
    if not files:
        print("No data to train.")
        return

    random.shuffle(files)
    batch_files = [files[i:i+5] for i in range(0, len(files), 5)]

    datasets = []
    total_samples = 0
    for bf in batch_files:
        dataset = GameDataset(bf)
        if len(dataset) == 0:
            continue
        datasets.append(dataset)
        total_samples += len(dataset)

    if not datasets:
        print("No data to train.")
        return

    epochs = 1
    if total_samples and total_samples < 50:
        epochs = min(10, max(3, int(50 / max(1, total_samples))))

    for dataset in datasets:
        loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)

        for _ in range(epochs):
            for imgs, mins, labels in loader:
                imgs = imgs.to(device)
                mins = mins.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with autocast(enabled=device.type == "cuda"):
                    out, _ = model(imgs, mins)
                    loss = criterion(out[:, -1, :], labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
    torch.save(model.state_dict(), model_path)
    print("Optimization Complete.")

def start_training_mode():
    global stop_training_flag, current_mode
    stop_training_flag = False
    
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    time.sleep(0.2)
    ctypes.windll.user32.ShowWindow(hwnd, 6)
    
    model_path = os.path.join(model_dir, "ai_model.pth")
    if not os.path.exists(model_path):
        print("No model found. Aborting training.")
        ctypes.windll.user32.ShowWindow(hwnd, 9) 
        return

    model = UniversalAI().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    mouse_ctrl = mouse.Controller()
    
    hidden = None
    input_buffer_img = []
    input_buffer_mouse = []
    
    with torch.no_grad():
        while not stop_training_flag:
            start_time = time.time()

            try:
                frame_img, frame_ts = get_latest_frame()
                if frame_img is None:
                    time.sleep(0.01)
                    continue
                img_small = cv2.resize(frame_img, (target_w, target_h))
                
                with mouse_lock:
                    curr_x, curr_y = mouse_state["x"], mouse_state["y"]
                    l_down = mouse_state["l_down"]
                    r_down = mouse_state["r_down"]
                    scroll = mouse_state["scroll"]

                mouse_state["scroll"] = 0 
                
                img_tensor = cv2.cvtColor(img_small, cv2.COLOR_BGRA2RGB)
                img_tensor = img_tensor.transpose(2, 0, 1) / 255.0
                
                m_vec = [
                    curr_x / screen_w, curr_y / screen_h,
                    1.0 if l_down else 0.0,
                    1.0 if r_down else 0.0,
                    scroll, 0, 0, 0, 0, 0
                ]
                
                input_buffer_img.append(img_tensor)
                input_buffer_mouse.append(m_vec)
                
                if len(input_buffer_img) > seq_len:
                    input_buffer_img.pop(0)
                    input_buffer_mouse.pop(0)
                
                if len(input_buffer_img) == seq_len:
                    t_imgs = torch.FloatTensor(np.array([input_buffer_img])).to(device)
                    t_mins = torch.FloatTensor(np.array([input_buffer_mouse])).to(device)
                    
                    out, hidden = model(t_imgs, t_mins, hidden)
                    
                    action = out[0, -1, :].cpu().numpy()
                    
                    pred_x = int(action[0] * screen_w)
                    pred_y = int(action[1] * screen_h)
                    pred_l = action[2]
                    pred_r = action[3]
                    
                    mouse_ctrl.position = (pred_x, pred_y)
                    
                    if pred_l > 0.5 and not l_down:
                        mouse_ctrl.press(mouse.Button.left)
                    elif pred_l <= 0.5 and l_down:
                        mouse_ctrl.release(mouse.Button.left)
                        
                    if pred_r > 0.5 and not r_down:
                        mouse_ctrl.press(mouse.Button.right)
                    elif pred_r <= 0.5 and r_down:
                        mouse_ctrl.release(mouse.Button.right)
                
                elapsed = time.time() - start_time
                wait = (1.0 / capture_freq) - elapsed
                if wait > 0:
                    time.sleep(wait)
                    
            except Exception as e:
                print(f"Training Runtime Error: {e}")
                break

    ctypes.windll.user32.ShowWindow(hwnd, 9)
    current_mode = MODE_LEARNING
    print("Exited Training Mode. Back to Learning.")

def record_data_loop():
    buffer = []
    last_pos = (0, 0)

    while True:
        if current_mode == MODE_LEARNING or current_mode == MODE_TRAINING:
            try:
                start_time = time.time()
                frame_img, frame_ts = get_latest_frame()
                if frame_img is None:
                    time.sleep(0.01)
                    continue
                img_small = cv2.resize(frame_img, (target_w, target_h))

                with mouse_lock:
                    c_state = mouse_state.copy()
                    mouse_state["scroll"] = 0
                    traj = temp_trajectory[:]
                    temp_trajectory.clear()
                
                entry = {
                    "ts": frame_ts if frame_ts else start_time,
                    "screen": cv2.imencode(".png", img_small)[1].tobytes(),
                    "mouse_x": c_state["x"],
                    "mouse_y": c_state["y"],
                    "l_down": c_state["l_down"],
                    "l_down_ts": c_state["l_down_ts"],
                    "l_up_pos": c_state["l_up_pos"],
                    "l_up_ts": c_state["l_up_ts"],
                    "r_down": c_state["r_down"],
                    "r_down_ts": c_state["r_down_ts"],
                    "r_up_pos": c_state["r_up_pos"],
                    "r_up_ts": c_state["r_up_ts"],
                    "scroll": c_state["scroll"],
                    "delta_x": c_state["x"] - last_pos[0],
                    "delta_y": c_state["y"] - last_pos[1],
                    "trajectory": traj,
                    "source": "AI" if current_mode == MODE_TRAINING else "Human"
                }

                buffer.append(entry)
                last_pos = (c_state["x"], c_state["y"])
                
                if len(buffer) >= 100:
                    fname = os.path.join(data_dir, f"{int(time.time()*1000)}.npy")
                    np.save(fname, buffer)
                    buffer = []
                    check_disk_space()

                elapsed = time.time() - start_time
                wait = (1.0 / capture_freq) - elapsed
                if wait > 0:
                    time.sleep(wait)
            except Exception as e:
                print(f"Recording Error: {e}")
            if flush_event.is_set():
                if buffer:
                    fname = os.path.join(data_dir, f"{int(time.time()*1000)}.npy")
                    np.save(fname, buffer)
                    buffer = []
                    check_disk_space()
                flush_event.clear()
                flush_done_event.set()
        else:
            if flush_event.is_set():
                if buffer:
                    fname = os.path.join(data_dir, f"{int(time.time()*1000)}.npy")
                    np.save(fname, buffer)
                    buffer = []
                    check_disk_space()
                flush_event.clear()
                flush_done_event.set()
            time.sleep(1)

def input_loop():
    global current_mode
    while True:
        cmd = input().strip()
        if cmd == "睡眠":
            if current_mode == MODE_LEARNING:
                flush_done_event.clear()
                flush_event.set()
                flush_done_event.wait(timeout=3)
                current_mode = MODE_SLEEP
                optimize_ai()
                current_mode = MODE_LEARNING
                print("Back to Learning.")
        elif cmd == "训练":
            if current_mode == MODE_LEARNING:
                current_mode = MODE_TRAINING
                print("Entering Training Mode...")
                t_thread = threading.Thread(target=start_training_mode)
                t_thread.daemon = True
                t_thread.start()

if __name__ == "__main__":
    ensure_initial_model()
    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
    mouse_listener.start()
    
    key_listener = keyboard.Listener(on_press=on_press_key)
    key_listener.start()
    
    t_res = threading.Thread(target=resource_monitor)
    t_res.daemon = True
    t_res.start()

    t_frame = threading.Thread(target=frame_generator_loop)
    t_frame.daemon = True
    t_frame.start()

    t_rec = threading.Thread(target=record_data_loop)
    t_rec.daemon = True
    t_rec.start()
    
    t_input = threading.Thread(target=input_loop)
    t_input.daemon = True
    t_input.start()
    
    print("System initialized. Mode: LEARNING")
    
    while True:
        time.sleep(10)
