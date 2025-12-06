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
import gc
import warnings
from collections import deque

warnings.filterwarnings("ignore", category=FutureWarning)

def progress_bar(prefix, current, total, suffix=""):
    total = max(total, 1)
    ratio = current / total
    bar_len = 30
    filled = int(bar_len * ratio)
    bar = "█" * filled + "-" * (bar_len - filled)
    percent = int(ratio * 100)
    sys.stdout.write(f"\r{prefix} [{bar}] {percent}% {suffix}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")

def install_requirements():
    package_map = {
        "torch": "torch",
        "torchvision": "torchvision",
        "numpy": "numpy",
        "opencv-python": "cv2",
        "mss": "mss",
        "pynput": "pynput",
        "psutil": "psutil",
        "nvidia-ml-py": "pynvml",
        "pillow": "PIL"
    }
    for package, import_name in package_map.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", package])
            except Exception as e:
                print(f"Error installing {package}: {e}")

install_requirements()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import cv2
import mss
import psutil
import torchvision.models as models
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
recording_pause_event = threading.Event()

def flush_buffers(timeout=3):
    flush_done_event.clear()
    flush_event.set()
    flush_done_event.wait(timeout=timeout)

capture_freq = 10
seq_len = 5
screen_w, screen_h = 2560, 1600
target_w, target_h = 256, 160
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mouse_feature_dim = 40
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

mouse_state = {
    "x": 0, "y": 0,
    "l_down": False, "l_down_ts": 0.0, "l_down_pos": (0,0), "l_up_pos": (0,0), "l_up_ts": 0.0,
    "r_down": False, "r_down_ts": 0.0, "r_down_pos": (0,0), "r_up_pos": (0,0), "r_up_ts": 0.0,
    "scroll": 0
}
mouse_lock = threading.Lock()
temp_trajectory = deque(maxlen=200)
frame_lock = threading.Lock()
latest_frame = {"img": None, "ts": 0.0}
file_lock = threading.Lock()

def update_latest_frame(img, ts):
    with frame_lock:
        latest_frame["img"] = img
        latest_frame["ts"] = ts

def get_latest_frame():
    with frame_lock:
        return latest_frame["img"], latest_frame["ts"]

data_queue = queue.Queue()
save_queue = queue.Queue()

class UniversalAI(nn.Module):
    def __init__(self):
        super(UniversalAI, self).__init__()
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        except Exception:
            weights = None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.feature_extractor = backbone.features
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.feature_extractor.eval()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, target_h, target_w)
            conv_out = self.pool(self.feature_extractor(dummy))
            self.fc_input_dim = conv_out.view(1, -1).size(1)
        self.mouse_dim = mouse_feature_dim

        self.lstm = nn.LSTM(input_size=self.fc_input_dim + self.mouse_dim, hidden_size=512, batch_first=True)

        self.fc_action = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

        self.feature_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.log_var_action = nn.Parameter(torch.tensor(0.0))
        self.log_var_prediction = nn.Parameter(torch.tensor(0.0))
        self.log_var_energy = nn.Parameter(torch.tensor(0.0))

    def forward(self, img, mouse_input, hidden=None):
        if hidden is not None:
            hidden = tuple(h.detach() for h in hidden)
        batch_size, seq, c, h, w = img.size()
        img_reshaped = img.view(batch_size * seq, c, h, w)
        feat = self.pool(self.feature_extractor(img_reshaped))
        feat = feat.view(batch_size, seq, -1)

        combined = torch.cat((feat, mouse_input), dim=2)

        out, hidden = self.lstm(combined, hidden)

        raw_action = self.fc_action(out)
        delta = torch.tanh(raw_action[:, :, :2])
        buttons_logits = raw_action[:, :, 2:]
        buttons = torch.sigmoid(buttons_logits)
        action = torch.cat((delta, buttons), dim=2)
        pred_features = self.feature_decoder(out[:, -1, :])
        return action, pred_features, buttons_logits, hidden

    def encode_features(self, img):
        with torch.no_grad():
            feat = self.pool(self.feature_extractor(img))
            feat = feat.view(feat.size(0), -1)
            if feat.size(1) > 128:
                feat = feat[:, :128]
            elif feat.size(1) < 128:
                pad = torch.zeros(feat.size(0), 128 - feat.size(1), device=feat.device)
                feat = torch.cat([feat, pad], dim=1)
        return feat

def ensure_initial_model():
    try:
        model_path = os.path.join(model_dir, "ai_model.pth")
        if not os.path.exists(model_path):
            temp_model = UniversalAI()
            alpha = float(temp_model.log_var_action.detach().item())
            beta = float(temp_model.log_var_prediction.detach().item())
            gamma = float(temp_model.log_var_energy.detach().item())
            torch.save({"model_state": temp_model.state_dict(), "alpha": alpha, "beta": beta, "gamma": gamma}, model_path)
            print("Generated initial AI model.")
    except Exception as e:
        print(f"Init model error: {e}")

def load_model_checkpoint(path, map_location=None):
    try:
        state = torch.load(path, map_location=map_location)
        if isinstance(state, dict) and "model_state" in state:
            return state
        if isinstance(state, dict):
            return {"model_state": state}
    except Exception as e:
        print(f"Load checkpoint error: {e}")
    return None

def get_sys_usage():
    try:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        gpu = 0
        vram = 0
        try:
            pynvml.nvmlInit()
            nvml_ready = True
        except Exception:
            nvml_ready = False
        if nvml_ready:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu = util.gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram = (mem_info.used / mem_info.total) * 100
            except Exception:
                gpu = 0
                vram = 0
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
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
        for pattern in ["*.npy", "*.npz"]:
            for f in glob.glob(os.path.join(data_dir, pattern)):
                try:
                    size = os.path.getsize(f)
                    total_size += size
                    files.append((f, os.path.getmtime(f)))
                except Exception:
                    continue

        limit = 20 * 1024 * 1024 * 1024
        if total_size > limit:
            files.sort(key=lambda x: x[1])
            now = time.time()
            while total_size > limit and files:
                f, mtime = files.pop(0)
                if now - mtime < 30:
                    continue
                try:
                    with file_lock:
                        s = os.path.getsize(f)
                        os.remove(f)
                    total_size -= s
                except:
                    pass
    except Exception as e:
        print(f"Disk clean error: {e}")

def disk_writer_loop():
    while True:
        try:
            fname, img_arr, act_arr = save_queue.get()
            with file_lock:
                np.savez_compressed(fname, image=img_arr, action=act_arr)
            check_disk_space()
        except Exception as e:
            print(f"Save Error: {e}")
        finally:
            save_queue.task_done()

class StreamingGameDataset(IterableDataset):
    def __init__(self, file_list):
        self.file_list = list(file_list)
        self.seq_len = seq_len

    def _prepare_item(self, slice_data, next_entry, structured, structured_npz):
        imgs = []
        m_ins = []

        def norm_coord(v, dim):
            return (2.0 * (v / dim)) - 1.0

        def load_image(item):
            if isinstance(item, np.ndarray) and item.ndim == 3:
                return item
            if structured:
                if isinstance(item, tuple):
                    return item[0]
                if hasattr(item, 'dtype'):
                    return item['image']
                return item[0]
            else:
                img_data = item.get("screen")
                if isinstance(img_data, bytes):
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                else:
                    img = img_data
            return img

        def normalize_image(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) if img.shape[2] == 4 else img
            return img.transpose(2, 0, 1) / 255.0

        for item in slice_data:
            if structured:
                action = item['action'] if hasattr(item, 'dtype') else item[1]
                action_len = len(action)

                def aval(idx, default=0.0):
                    return float(action[idx]) if idx < action_len else default

                ts_now = aval(0)
                mx = norm_coord(aval(1), screen_w)
                my = norm_coord(aval(2), screen_h)
                l_down = aval(3)
                r_down = aval(4)
                scroll = aval(5)
                delta_x = aval(16) / screen_w
                delta_y = aval(17) / screen_h
                l_down_x = norm_coord(aval(7), screen_w)
                l_down_y = norm_coord(aval(8), screen_h)
                l_up_x = norm_coord(aval(10), screen_w)
                l_up_y = norm_coord(aval(11), screen_h)
                r_down_x = norm_coord(aval(13), screen_w)
                r_down_y = norm_coord(aval(14), screen_h)
                r_up_x = norm_coord(aval(18), screen_w)
                r_up_y = norm_coord(aval(19), screen_h)
                traj_vals = []
                for ti in range(20, 40, 2):
                    traj_vals.append(norm_coord(aval(ti), screen_w))
                    traj_vals.append(norm_coord(aval(ti + 1), screen_h))
                time_since_l_down = max(0.0, ts_now - aval(6)) if aval(6) > 0 else 0.0
                time_since_l_up = max(0.0, ts_now - aval(9)) if aval(9) > 0 else 0.0
                time_since_r_down = max(0.0, ts_now - aval(12)) if aval(12) > 0 else 0.0
                time_since_r_up = max(0.0, ts_now - aval(15)) if aval(15) > 0 else 0.0
                is_ai = aval(40)
            else:
                ts_now = item.get("ts", 0.0)
                mx = norm_coord(item.get("mouse_x", 0.0), screen_w)
                my = norm_coord(item.get("mouse_y", 0.0), screen_h)
                l_down = 1.0 if item.get("l_down", False) else 0.0
                r_down = 1.0 if item.get("r_down", False) else 0.0
                scroll = item.get("scroll", 0.0)
                delta_x = item.get("delta_x", 0.0) / screen_w
                delta_y = item.get("delta_y", 0.0) / screen_h
                l_down_pos = item.get("l_down_pos", (0.0, 0.0))
                l_up_pos = item.get("l_up_pos", (0.0, 0.0))
                r_down_pos = item.get("r_down_pos", (0.0, 0.0))
                r_up_pos = item.get("r_up_pos", (0.0, 0.0))
                l_down_x = norm_coord(l_down_pos[0], screen_w)
                l_down_y = norm_coord(l_down_pos[1], screen_h)
                l_up_x = norm_coord(l_up_pos[0], screen_w)
                l_up_y = norm_coord(l_up_pos[1], screen_h)
                r_down_x = norm_coord(r_down_pos[0], screen_w)
                r_down_y = norm_coord(r_down_pos[1], screen_h)
                r_up_x = norm_coord(r_up_pos[0], screen_w)
                r_up_y = norm_coord(r_up_pos[1], screen_h)
                raw_traj = item.get("trajectory", [])
                if raw_traj:
                    traj_values = raw_traj
                else:
                    traj_values = [item.get("mouse_x", 0.0), item.get("mouse_y", 0.0)] * 10
                traj_vals = []
                for i in range(0, min(len(traj_values), 20), 2):
                    traj_vals.append(norm_coord(traj_values[i], screen_w))
                    if i + 1 < len(traj_values):
                        traj_vals.append(norm_coord(traj_values[i + 1], screen_h))
                while len(traj_vals) < 20:
                    traj_vals.append(mx)
                    traj_vals.append(my)
                time_since_l_down = max(0.0, ts_now - item.get("l_down_ts", 0.0)) if item.get("l_down_ts", 0.0) > 0 else 0.0
                time_since_l_up = max(0.0, ts_now - item.get("l_up_ts", 0.0)) if item.get("l_up_ts", 0.0) > 0 else 0.0
                time_since_r_down = max(0.0, ts_now - item.get("r_down_ts", 0.0)) if item.get("r_down_ts", 0.0) > 0 else 0.0
                time_since_r_up = max(0.0, ts_now - item.get("r_up_ts", 0.0)) if item.get("r_up_ts", 0.0) > 0 else 0.0
                is_ai = item.get("is_ai", 0.0)
            img_raw = load_image(item)
            img = normalize_image(img_raw)
            imgs.append(img)

            m_vec = [
                mx, my,
                l_down,
                r_down,
                scroll,
                delta_x,
                delta_y,
                time_since_l_down,
                time_since_r_down,
                time_since_l_up,
                time_since_r_up,
                l_down_x,
                l_down_y,
                l_up_x,
                l_up_y,
                r_down_x,
                r_down_y,
                r_up_x,
                r_up_y
            ]
            m_vec.extend(traj_vals)
            m_vec.append(is_ai)
            m_ins.append(m_vec)

        if next_entry is not None:
            if structured:
                next_action_source = next_entry[1] if isinstance(next_entry, tuple) else next_entry
                next_action = next_action_source if hasattr(next_action_source, 'dtype') else next_action_source
                action_len = len(next_action)

                def nval(idx, default=0.0):
                    return float(next_action[idx]) if idx < action_len else default

                labels = [
                    nval(16) / screen_w,
                    nval(17) / screen_h,
                    nval(3),
                    nval(4)
                ]
            else:
                labels = [
                    next_entry.get("delta_x", 0.0) / screen_w,
                    next_entry.get("delta_y", 0.0) / screen_h,
                    1.0 if next_entry.get("l_down", False) else 0.0,
                    1.0 if next_entry.get("r_down", False) else 0.0
                ]
            next_img_raw = load_image(next_entry)
            next_img = normalize_image(next_img_raw)
        else:
            labels = [0.0, 0.0, 0.0, 0.0]
            next_img = np.zeros((3, target_h, target_w), dtype=np.float32)

        return torch.FloatTensor(np.array(imgs)), torch.FloatTensor(np.array(m_ins)), torch.FloatTensor(np.array(labels)), torch.FloatTensor(np.array(next_img))

    def __iter__(self):
        output_queue = queue.Queue(maxsize=8)
        stop_event = threading.Event()

        def producer():
            for path in self.file_list:
                try:
                    with file_lock:
                        data = np.load(path, allow_pickle=True, mmap_mode="r")
                    if isinstance(data, np.ndarray) and getattr(data, "shape", None) == ():
                        try:
                            data = data.item()
                        except Exception:
                            data = [data]
                    structured_npz = isinstance(data, np.lib.npyio.NpzFile)
                    structured_array = hasattr(data, 'dtype') and data.dtype.names and 'action' in data.dtype.names
                    structured = structured_npz or structured_array
                    length = len(data['action']) if structured else len(data)
                    if length <= 0:
                        if structured_npz:
                            data.close()
                        continue
                    max_start = max(1, length - self.seq_len)
                    for start_idx in range(max_start):
                        slice_data = []
                        for i in range(self.seq_len):
                            idx_in_file = min(start_idx + i, length - 1)
                            if structured_npz:
                                slice_data.append((data['image'][idx_in_file], data['action'][idx_in_file]))
                            else:
                                slice_data.append(data[idx_in_file])
                        next_entry = None
                        next_idx = start_idx + self.seq_len
                        if next_idx < length:
                            if structured_npz:
                                next_entry = (data['image'][next_idx], data['action'][next_idx])
                            elif structured:
                                next_entry = data[next_idx]
                            else:
                                next_entry = data[next_idx]
                        item = self._prepare_item(slice_data, next_entry, structured, structured_npz)
                        output_queue.put(item)
                    if structured_npz:
                        data.close()
                except Exception as e:
                    print(f"Data load error: {e}")
            stop_event.set()

        threading.Thread(target=producer, daemon=True).start()
        while not (stop_event.is_set() and output_queue.empty()):
            try:
                yield output_queue.get(timeout=0.5)
            except queue.Empty:
                continue

def sample_trajectory(traj, ts_now, max_points=10, window=0.1, fallback_pos=(0, 0)):
    recent = [p for p in traj if ts_now - p[2] <= window]
    if not recent:
        recent = [(fallback_pos[0], fallback_pos[1], ts_now)]
    if len(recent) > max_points:
        idx = np.linspace(0, len(recent) - 1, max_points, dtype=int)
        recent = [recent[i] for i in idx]
    while len(recent) < max_points:
        recent.append(recent[-1])
    flat = []
    for px, py, _ in recent:
        flat.extend([px, py])
    return flat

def optimize_ai():
    try:
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
            except Exception:
                pass
        print("Starting Optimization...")
        model_path = os.path.join(model_dir, "ai_model.pth")
        backup_path = os.path.join(model_dir, "ai_model_prev.pth")
        model = UniversalAI().to(device)
        if os.path.exists(model_path):
            try:
                state = load_model_checkpoint(model_path, map_location=device)
                if state is not None:
                    model.load_state_dict(state.get("model_state", {}), strict=False)
            except Exception as e:
                print(f"Model load warning: {e}")

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()
        scaler = GradScaler("cuda", enabled=device.type == "cuda")
        accumulation_steps = 4

        files = []
        candidates = []
        for pattern in ["*.npy", "*.npz"]:
            candidates.extend(glob.glob(os.path.join(data_dir, pattern)))
        total_scan = len(candidates)
        scanned = 0
        print("Scanning data files...")
        torch.cuda.empty_cache()
        for f in candidates:
            scanned += 1
            progress_bar("数据扫描阶段", scanned, total_scan)
            try:
                if os.path.getsize(f) > 0:
                    files.append(f)
            except:
                continue
        if not files:
            print("No data to train.")
            torch.cuda.empty_cache()
            return

        random.shuffle(files)
        batch_files = [files[i:i+5] for i in range(0, len(files), 5)]

        def estimate_steps(file_subset):
            total = 0
            for path in file_subset:
                try:
                    with file_lock:
                        data = np.load(path, allow_pickle=True, mmap_mode="r")
                    structured_npz = isinstance(data, np.lib.npyio.NpzFile)
                    structured_array = hasattr(data, 'dtype') and data.dtype.names and 'action' in data.dtype.names
                    structured = structured_npz or structured_array
                    length = len(data['action']) if structured else len(data)
                    steps = max(1, length - seq_len)
                    total += steps
                    if structured_npz:
                        data.close()
                except Exception:
                    continue
            return total

        chunk_steps = []
        init_total = len(batch_files)
        init_done = 0
        print("Dataset Init")
        torch.cuda.empty_cache()
        for bf in batch_files:
            steps = estimate_steps(bf)
            chunk_steps.append(steps)
            init_done += 1
            progress_bar("Dataset Init", init_done, init_total, f"Chunks: {steps} steps")
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        total_steps = sum(chunk_steps)
        if total_steps <= 0:
            print("No data to train.")
            torch.cuda.empty_cache()
            return

        epochs = 1
        current_step = 0
        total_chunks = len(batch_files)
        last_loss_value = 0.0
        torch.cuda.empty_cache()
        for idx, bf in enumerate(batch_files):
            dataset = StreamingGameDataset(bf)
            loader = DataLoader(dataset, batch_size=4, drop_last=False, num_workers=0, pin_memory=True)
            optimizer.zero_grad()
            for _ in range(epochs):
                pending_steps = 0
                for batch_idx, (imgs, mins, labels, next_frames) in enumerate(loader):
                    imgs = imgs.to(device)
                    mins = mins.to(device)
                    labels = labels.to(device)
                    next_frames = next_frames.to(device)

                    with autocast(device_type="cuda", enabled=device.type == "cuda"):
                        log_var_action = torch.clamp(model.log_var_action, -6.0, 6.0)
                        log_var_prediction = torch.clamp(model.log_var_prediction, -6.0, 6.0)
                        log_var_energy = torch.clamp(model.log_var_energy, -6.0, 6.0)
                        action, pred_feat, button_logits, _ = model(imgs, mins, None)
                        delta_pred = action[:, -1, :2]
                        target_delta = labels[:, :2]
                        target_buttons = labels[:, 2:]
                        imitation_loss = mse_loss(delta_pred, target_delta) + bce_loss(button_logits[:, -1, :], target_buttons)
                        target_feat = model.encode_features(next_frames)
                        pred_loss = mse_loss(pred_feat, target_feat)
                        energy_loss = torch.mean(torch.sum(delta_pred ** 2, dim=1))
                        total_loss = 0.5 * torch.exp(-log_var_action) * imitation_loss + 0.5 * log_var_action
                        total_loss = total_loss + 0.5 * torch.exp(-log_var_prediction) * pred_loss + 0.5 * log_var_prediction
                        total_loss = total_loss + 0.5 * torch.exp(-log_var_energy) * energy_loss + 0.5 * log_var_energy
                        total_loss = total_loss / accumulation_steps
                    scaler.scale(total_loss).backward()
                    pending_steps += 1
                    if pending_steps % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        pending_steps = 0
                    current_step += 1
                    last_loss_value = total_loss.item()
                    progress_bar("模型训练阶段", current_step, total_steps, f"Loss: {last_loss_value:.4f} | Chunk {idx+1}/{total_chunks}")
                if pending_steps > 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                gc.collect()
                torch.cuda.empty_cache()
            del loader
            del dataset
            gc.collect()
            torch.cuda.empty_cache()

        temp_path = os.path.join(model_dir, "ai_model_temp.pth")
        print(f"Final Loss Snapshot: {last_loss_value:.6f} | Steps: {current_step}/{total_steps}")
        alpha = float(model.log_var_action.detach().item())
        beta = float(model.log_var_prediction.detach().item())
        gamma = float(model.log_var_energy.detach().item())
        torch.save({"model_state": model.state_dict(), "alpha": alpha, "beta": beta, "gamma": gamma}, temp_path)
        try:
            if os.path.exists(model_path):
                shutil.copy2(model_path, backup_path)
            if os.path.exists(model_path):
                os.remove(model_path)
            os.rename(temp_path, model_path)
            print("Optimization Complete (Safe Save).")
        except Exception as e:
            print(f"Save error: {e}")
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, model_path)
        focus_weight = float(torch.exp(-model.log_var_action.detach()).item())
        curiosity_weight = float(torch.exp(-model.log_var_prediction.detach()).item())
        laziness_penalty = float(torch.exp(-model.log_var_energy.detach()).item())
        print("[Optimization Done]")
        print(f"> Imitation Weight (Focus): {focus_weight:.3f}  <-- exp(-alpha)")
        print(f"> Prediction Weight (Curiosity): {curiosity_weight:.3f}  <-- exp(-beta)")
        print(f"> Energy Penalty (Laziness): {laziness_penalty:.3f}  <-- exp(-gamma)")
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        gc.collect()
    except Exception as e:
        print(f"Critical Optimization Error: {e}")
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
    finally:
        torch.cuda.empty_cache()
        gc.collect()

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
    try:
        state = load_model_checkpoint(model_path, map_location=device)
        if state is not None:
            model.load_state_dict(state.get("model_state", {}))
    except Exception as e:
        print(f"Model load error: {e}")
    model.eval()
    
    mouse_ctrl = mouse.Controller()
    
    hidden = None
    input_buffer_img = []
    input_buffer_mouse = []
    with mouse_lock:
        prev_pos = (mouse_state["x"], mouse_state["y"])
    delta_history = deque(maxlen=5)

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
                    l_up_pos = mouse_state["l_up_pos"]
                    r_up_pos = mouse_state["r_up_pos"]
                    traj = list(temp_trajectory)

                mouse_state["scroll"] = 0

                img_tensor = cv2.cvtColor(img_small, cv2.COLOR_BGRA2RGB)
                img_tensor = img_tensor.transpose(2, 0, 1) / 255.0

                ts_value = frame_ts if frame_ts else start_time
                def norm_coord(v, dim):
                    return (2.0 * (v / dim)) - 1.0
                traj_flat = sample_trajectory(traj, ts_value, fallback_pos=(curr_x, curr_y))
                traj_norm = []
                for i, v in enumerate(traj_flat):
                    if i % 2 == 0:
                        traj_norm.append(norm_coord(v, screen_w))
                    else:
                        traj_norm.append(norm_coord(v, screen_h))
                m_vec = [
                    norm_coord(curr_x, screen_w), norm_coord(curr_y, screen_h),
                    1.0 if l_down else 0.0,
                    1.0 if r_down else 0.0,
                    scroll,
                    (curr_x - prev_pos[0]) / screen_w,
                    (curr_y - prev_pos[1]) / screen_h,
                    max(0.0, ts_value - mouse_state["l_down_ts"]) if mouse_state["l_down_ts"] > 0 else 0.0,
                    max(0.0, ts_value - mouse_state["r_down_ts"]) if mouse_state["r_down_ts"] > 0 else 0.0,
                    max(0.0, ts_value - mouse_state["l_up_ts"]) if mouse_state["l_up_ts"] > 0 else 0.0,
                    max(0.0, ts_value - mouse_state["r_up_ts"]) if mouse_state["r_up_ts"] > 0 else 0.0,
                    norm_coord(mouse_state["l_down_pos"][0], screen_w),
                    norm_coord(mouse_state["l_down_pos"][1], screen_h),
                    norm_coord(l_up_pos[0], screen_w),
                    norm_coord(l_up_pos[1], screen_h),
                    norm_coord(mouse_state["r_down_pos"][0], screen_w),
                    norm_coord(mouse_state["r_down_pos"][1], screen_h),
                    norm_coord(r_up_pos[0], screen_w),
                    norm_coord(r_up_pos[1], screen_h)
                ]
                m_vec.extend(traj_norm)
                m_vec.append(1.0)

                input_buffer_img.append(img_tensor)
                input_buffer_mouse.append(m_vec)
                prev_pos = (target_x, target_y)
                
                if len(input_buffer_img) > seq_len:
                    input_buffer_img.pop(0)
                    input_buffer_mouse.pop(0)
                
                if len(input_buffer_img) == seq_len:
                    t_imgs = torch.FloatTensor(np.array([input_buffer_img])).to(device)
                    t_mins = torch.FloatTensor(np.array([input_buffer_mouse])).to(device)

                    action_out, _, _, hidden = model(t_imgs, t_mins, hidden)

                    action = action_out[0, -1, :].cpu().numpy()

                    pred_dx = action[0] * screen_w
                    pred_dy = action[1] * screen_h
                    delta_history.append((pred_dx, pred_dy))
                    avg_dx = sum(d[0] for d in delta_history) / len(delta_history)
                    avg_dy = sum(d[1] for d in delta_history) / len(delta_history)
                    target_x = int(min(max(0, curr_x + avg_dx), screen_w - 1))
                    target_y = int(min(max(0, curr_y + avg_dy), screen_h - 1))
                    pred_l = action[2]
                    pred_r = action[3]

                    mouse_ctrl.position = (target_x, target_y)
                    
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

    flush_buffers()
    ctypes.windll.user32.ShowWindow(hwnd, 9)
    current_mode = MODE_LEARNING
    print("Exited Training Mode. Back to Learning.")

def record_data_loop():
    buffer_images = []
    buffer_actions = []
    last_pos = (0, 0)

    while True:
        if recording_pause_event.is_set():
            if flush_event.is_set():
                if buffer_images:
                    fname = os.path.join(data_dir, f"{int(time.time()*1000)}.npz")
                    img_arr = np.array(buffer_images, dtype=np.uint8)
                    act_arr = np.array(buffer_actions, dtype=np.float32)
                    save_queue.put((fname, img_arr, act_arr))
                    buffer_images = []
                    buffer_actions = []
                    save_queue.join()
                flush_event.clear()
                flush_done_event.set()
            time.sleep(0.05)
            continue
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
                    traj = list(temp_trajectory)

                ts_value = frame_ts if frame_ts else start_time
                traj_flat = sample_trajectory(traj, ts_value, fallback_pos=(c_state["x"], c_state["y"]))
                img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGRA2RGB) if img_small.shape[2] == 4 else img_small
                buffer_images.append(img_rgb.astype(np.uint8))
                action_entry = [
                    ts_value,
                    c_state["x"],
                    c_state["y"],
                    1.0 if c_state["l_down"] else 0.0,
                    1.0 if c_state["r_down"] else 0.0,
                    c_state["scroll"],
                    c_state["l_down_ts"],
                    c_state["l_down_pos"][0],
                    c_state["l_down_pos"][1],
                    c_state["l_up_ts"],
                    c_state["l_up_pos"][0],
                    c_state["l_up_pos"][1],
                    c_state["r_down_ts"],
                    c_state["r_down_pos"][0],
                    c_state["r_down_pos"][1],
                    c_state["r_up_ts"],
                    c_state["x"] - last_pos[0],
                    c_state["y"] - last_pos[1],
                    c_state["r_up_pos"][0],
                    c_state["r_up_pos"][1]
                ]
                action_entry.extend(traj_flat)
                action_entry.append(1.0 if current_mode == MODE_TRAINING else 0.0)
                buffer_actions.append(action_entry)
                last_pos = (c_state["x"], c_state["y"])

                if len(buffer_images) >= 3000:
                    fname = os.path.join(data_dir, f"{int(time.time()*1000)}.npz")
                    img_arr = np.array(buffer_images, dtype=np.uint8)
                    act_arr = np.array(buffer_actions, dtype=np.float32)
                    save_queue.put((fname, img_arr, act_arr))
                    buffer_images = []
                    buffer_actions = []

                elapsed = time.time() - start_time
                wait = (1.0 / capture_freq) - elapsed
                if wait > 0:
                    time.sleep(wait)
            except Exception as e:
                print(f"Recording Error: {e}")
            if flush_event.is_set():
                if buffer_images:
                    fname = os.path.join(data_dir, f"{int(time.time()*1000)}.npz")
                    img_arr = np.array(buffer_images, dtype=np.uint8)
                    act_arr = np.array(buffer_actions, dtype=np.float32)
                    save_queue.put((fname, img_arr, act_arr))
                    buffer_images = []
                    buffer_actions = []
                    save_queue.join()
                flush_event.clear()
                flush_done_event.set()
        else:
            if flush_event.is_set():
                if buffer_images:
                    fname = os.path.join(data_dir, f"{int(time.time()*1000)}.npz")
                    img_arr = np.array(buffer_images, dtype=np.uint8)
                    act_arr = np.array(buffer_actions, dtype=np.float32)
                    save_queue.put((fname, img_arr, act_arr))
                    buffer_images = []
                    buffer_actions = []
                    save_queue.join()
                flush_event.clear()
                flush_done_event.set()
            time.sleep(1)

def input_loop():
    global current_mode
    while True:
        cmd = input().strip()
        if cmd == "睡眠":
            if current_mode == MODE_LEARNING:
                recording_pause_event.set()
                flush_buffers()
                current_mode = MODE_SLEEP
                optimize_ai()
                current_mode = MODE_LEARNING
                recording_pause_event.clear()
                print("Back to Learning.")
        elif cmd == "训练":
            if current_mode == MODE_LEARNING:
                flush_buffers()
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

    t_save = threading.Thread(target=disk_writer_loop)
    t_save.daemon = True
    t_save.start()

    t_rec = threading.Thread(target=record_data_loop)
    t_rec.daemon = True
    t_rec.start()
    
    t_input = threading.Thread(target=input_loop)
    t_input.daemon = True
    t_input.start()

    print("System initialized. Mode: LEARNING")

    while True:
        try:
            if not mouse_listener.is_alive():
                mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
                mouse_listener.start()
            if not key_listener.is_alive():
                key_listener = keyboard.Listener(on_press=on_press_key)
                key_listener.start()
        except Exception as e:
            print(f"Listener restart error: {e}")
        time.sleep(2)
