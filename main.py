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
import io
import struct
from collections import deque
if os.name == "nt":
    import winreg

warnings.filterwarnings("ignore", category=FutureWarning)

def resolve_desktop_path():
    if os.name == "nt":
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders')
            path = winreg.QueryValueEx(key, "Desktop")[0]
            winreg.CloseKey(key)
            if path:
                return path
        except Exception as e:
            print(f"Desktop path resolve error: {e}")
    return os.path.join(os.path.expanduser("~"), "Desktop")

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
from torch.utils.checkpoint import checkpoint
import numpy as np
import cv2
import mss
import psutil
import torchvision.models as models
from pynput import mouse, keyboard
import pynvml

desktop_path = resolve_desktop_path()
base_dir = os.path.join(desktop_path, "AAA")
data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "model")
temp_dir = os.path.join(base_dir, "temp")
log_path = os.path.join(data_dir, "experience.log")
index_path = os.path.join(data_dir, "experience.idx")

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
stop_optimization_flag = threading.Event()

def set_process_priority():
    try:
        proc = psutil.Process(os.getpid())
        if os.name == "nt":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            proc.nice(psutil.NORMAL_PRIORITY_CLASS if hasattr(psutil, "NORMAL_PRIORITY_CLASS") else 5)
    except Exception as e:
        print(f"Priority set error: {e}")

def flush_buffers(timeout=3):
    flush_done_event.clear()
    flush_event.set()
    flush_done_event.wait(timeout=timeout)

capture_freq = 10
seq_len = 12
screen_w, screen_h = 2560, 1600
target_w, target_h = 256, 160
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mouse_feature_dim = 40
grid_w, grid_h = 32, 20
grid_size = grid_w * grid_h
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

def cleanup_before_sleep():
    try:
        with frame_lock:
            latest_frame["img"] = None
            latest_frame["ts"] = 0.0
        temp_trajectory.clear()
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

data_queue = queue.Queue()
save_queue = queue.Queue()

def append_binary_log(img_arr, act_arr):
    try:
        buf = io.BytesIO()
        np.savez_compressed(buf, image=img_arr, action=act_arr)
        payload = buf.getvalue()
        header = struct.pack("<Q", len(payload))
        with file_lock:
            with open(log_path, "ab") as f:
                f.write(header)
                f.write(payload)
            with open(index_path, "ab") as idx:
                idx.write(header)
    except Exception as e:
        print(f"Binary log append error: {e}")

def iterate_binary_log(path):
    try:
        with file_lock:
            if not os.path.exists(path):
                return []
            records = []
            with open(path, "rb") as f:
                while True:
                    header = f.read(8)
                    if not header or len(header) < 8:
                        break
                    length = struct.unpack("<Q", header)[0]
                    payload = f.read(length)
                    if len(payload) < length:
                        break
                    records.append(payload)
        return records
    except Exception as e:
        print(f"Binary log read error: {e}")
        return []

def trim_binary_log(limit_bytes):
    try:
        if not os.path.exists(log_path):
            return
        if not os.path.exists(index_path):
            return
        with file_lock:
            with open(index_path, "rb") as idx_f:
                length_bytes = idx_f.read()
            lengths = [struct.unpack("<Q", length_bytes[i:i+8])[0] for i in range(0, len(length_bytes), 8)]
            if not lengths:
                return
            total_bytes = sum(lengths) + len(lengths) * 8
            if total_bytes <= limit_bytes:
                return
            keep_limit = int(limit_bytes * 0.8)
            keep_lengths = []
            size_acc = 0
            for l in reversed(lengths):
                if size_acc + l + 8 > keep_limit and keep_lengths:
                    break
                keep_lengths.append(l)
                size_acc += l + 8
            keep_lengths = list(reversed(keep_lengths))
            start_idx = len(lengths) - len(keep_lengths)
            offset = sum(lengths[:start_idx]) + start_idx * 8
            with open(log_path, "rb") as f:
                f.seek(offset)
                remaining = f.read()
            with open(log_path, "wb") as f:
                f.write(remaining)
            with open(index_path, "wb") as idx_f:
                for l in keep_lengths:
                    idx_f.write(struct.pack("<Q", l))
    except Exception as e:
        print(f"Binary log trim error: {e}")

class CausalBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout):
        super(CausalBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim), nn.Dropout(dropout))

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        attn_out = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        x = x + attn_out
        h2 = self.ln2(x)
        x = x + self.ff(h2)
        return x

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
        self.model_dim = 512
        self.blocks = nn.ModuleList([CausalBlock(self.model_dim, 8, 1024, 0.1) for _ in range(4)])
        self.input_proj = nn.Linear(self.fc_input_dim + self.mouse_dim, self.model_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 512, self.model_dim))
        self.action_head = nn.Linear(self.model_dim, grid_size)
        self.button_head = nn.Linear(self.model_dim, 2)
        self.feature_decoder = nn.Sequential(nn.Linear(self.model_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        self.log_var_action = nn.Parameter(torch.tensor(0.0))
        self.log_var_prediction = nn.Parameter(torch.tensor(0.0))
        self.log_var_energy = nn.Parameter(torch.tensor(0.0))
        self.use_checkpoint = True

    def forward(self, img, mouse_input, hidden=None):
        batch_size, seq, c, h, w = img.size()
        img_reshaped = img.view(batch_size * seq, c, h, w)
        feat = self.pool(self.feature_extractor(img_reshaped))
        feat = feat.view(batch_size, seq, -1)
        combined = torch.cat((feat, mouse_input), dim=2)
        x = self.input_proj(combined)
        pos = self.pos_embedding[:, :seq, :]
        x = x + pos
        mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(lambda t: block(t, mask), x)
            else:
                x = block(x, mask)
        action_token = x[:, -1, :]
        grid_logits = self.action_head(action_token)
        button_logits = self.button_head(action_token)
        pred_features = self.feature_decoder(action_token)
        return grid_logits, pred_features, button_logits, None

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

    def enable_backbone_finetune(self, portion=0.3):
        total = len(list(self.feature_extractor.children()))
        threshold = int(total * (1 - portion))
        for idx, layer in enumerate(self.feature_extractor.children()):
            requires = idx >= threshold
            for p in layer.parameters():
                p.requires_grad = requires

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
                step_x = max(1, screen_w // target_w)
                step_y = max(1, screen_h // target_h)
                img_small = img[::step_y, ::step_x]
                if img_small.shape[1] > target_w:
                    img_small = img_small[:, :target_w]
                if img_small.shape[0] > target_h:
                    img_small = img_small[:target_h, :]
                update_latest_frame(img_small, start)
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
    elif current_mode == MODE_SLEEP:
        stop_optimization_flag.set()

def file_in_use(path):
    try:
        if os.name == "nt":
            fd = os.open(path, os.O_RDONLY)
            os.close(fd)
        else:
            with open(path, "rb"):
                pass
        return False
    except PermissionError:
        return True
    except FileNotFoundError:
        return True
    except Exception:
        return False

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
        for extra in [log_path, index_path]:
            if os.path.exists(extra):
                try:
                    total_size += os.path.getsize(extra)
                except Exception:
                    pass

        limit = 20 * 1024 * 1024 * 1024
        if total_size > limit:
            files.sort(key=lambda x: x[1])
            now = time.time()
            skip_guard = 0
            while total_size > limit and files:
                f, mtime = files.pop(0)
                if now - mtime < 30:
                    files.append((f, mtime))
                    skip_guard += 1
                    if skip_guard >= len(files):
                        break
                    continue
                if file_in_use(f):
                    files.append((f, mtime))
                    skip_guard += 1
                    if skip_guard >= len(files):
                        break
                    continue
                try:
                    with file_lock:
                        s = os.path.getsize(f)
                        os.remove(f)
                    total_size -= s
                    skip_guard = 0
                except PermissionError:
                    files.append((f, mtime))
                    skip_guard += 1
                    if skip_guard >= len(files):
                        break
                except Exception:
                    continue
            if total_size > limit:
                trim_binary_log(limit)
    except Exception as e:
        print(f"Disk clean error: {e}")

def consolidate_data_files(max_frames=10000, min_commit=4000):
    try:
        candidates = sorted(glob.glob(os.path.join(data_dir, "*.npz")), key=os.path.getmtime)
        buffer_images = []
        buffer_actions = []
        collected = []
        for path in candidates:
            if stop_optimization_flag.is_set():
                break
            try:
                with file_lock:
                    data = np.load(path, allow_pickle=True)
                structured_npz = isinstance(data, np.lib.npyio.NpzFile)
                if not structured_npz or 'image' not in data or 'action' not in data:
                    if structured_npz:
                        data.close()
                    continue
                imgs = np.array(data['image'])
                acts = np.array(data['action'])
                if structured_npz:
                    data.close()
                if len(acts) >= max_frames:
                    continue
                buffer_images.append(imgs)
                buffer_actions.append(acts)
                collected.append(path)
            except Exception as e:
                print(f"Merge read error: {e}")
                continue
            total_frames = sum(len(a) for a in buffer_actions)
            if total_frames >= max_frames:
                try:
                    merged_images = np.concatenate(buffer_images, axis=0)
                    merged_actions = np.concatenate(buffer_actions, axis=0)
                    fname = os.path.join(data_dir, f"merged_{int(time.time()*1000)}.npz")
                    with file_lock:
                        np.savez(fname, image=merged_images, action=merged_actions)
                        for old in collected:
                            if os.path.exists(old):
                                os.remove(old)
                except Exception as e:
                    print(f"Merge write error: {e}")
                buffer_images = []
                buffer_actions = []
                collected = []
        remaining = sum(len(a) for a in buffer_actions)
        if remaining >= min_commit and buffer_images:
            try:
                merged_images = np.concatenate(buffer_images, axis=0)
                merged_actions = np.concatenate(buffer_actions, axis=0)
                fname = os.path.join(data_dir, f"merged_{int(time.time()*1000)}.npz")
                with file_lock:
                    np.savez(fname, image=merged_images, action=merged_actions)
                    for old in collected:
                        if os.path.exists(old):
                            os.remove(old)
            except Exception as e:
                print(f"Merge final write error: {e}")
    except Exception as e:
        print(f"Merge error: {e}")

def disk_writer_loop():
    check_counter = 0
    last_check = time.time()
    check_interval = 10
    check_time_window = 60.0
    while True:
        try:
            fname, img_arr, act_arr = save_queue.get()
            append_binary_log(img_arr, act_arr)
            check_counter += 1
            now = time.time()
            if check_counter >= check_interval or now - last_check >= check_time_window:
                check_disk_space()
                last_check = now
                check_counter = 0
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

                nx = nval(1)
                ny = nval(2)
                gx = int(max(0, min(grid_w - 1, (nx / screen_w) * grid_w)))
                gy = int(max(0, min(grid_h - 1, (ny / screen_h) * grid_h)))
                grid_idx = gy * grid_w + gx
                labels = [grid_idx, nval(3), nval(4)]
            else:
                nx = next_entry.get("mouse_x", 0.0)
                ny = next_entry.get("mouse_y", 0.0)
                gx = int(max(0, min(grid_w - 1, (nx / screen_w) * grid_w)))
                gy = int(max(0, min(grid_h - 1, (ny / screen_h) * grid_h)))
                grid_idx = gy * grid_w + gx
                labels = [grid_idx, 1.0 if next_entry.get("l_down", False) else 0.0, 1.0 if next_entry.get("r_down", False) else 0.0]
            next_img_raw = load_image(next_entry)
            next_img = normalize_image(next_img_raw)
        else:
            labels = [0.0, 0.0, 0.0]
            next_img = np.zeros((3, target_h, target_w), dtype=np.float32)

        return torch.FloatTensor(np.array(imgs)), torch.FloatTensor(np.array(m_ins)), torch.FloatTensor(np.array(labels)), torch.FloatTensor(np.array(next_img))

    def __iter__(self):
        output_queue = queue.Queue(maxsize=8)
        stop_event = threading.Event()

        def producer():
            for path in self.file_list:
                try:
                    if path == log_path:
                        for payload in iterate_binary_log(path):
                            try:
                                buf = io.BytesIO(payload)
                                data = np.load(buf, allow_pickle=True)
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
                                print(f"Binary data load error: {e}")
                        continue
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
        time.sleep(1.0)
        stop_optimization_flag.clear()
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
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        if os.path.exists(model_path):
            try:
                state = load_model_checkpoint(model_path, map_location=device)
                if state is not None:
                    model.load_state_dict(state.get("model_state", {}), strict=False)
                    if "optimizer_state" in state:
                        optimizer.load_state_dict(state["optimizer_state"])
            except Exception as e:
                print(f"Model load warning: {e}")

        model.train()
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()
        scaler = GradScaler("cuda", enabled=device.type == "cuda")
        accumulation_steps = 4
        if torch.cuda.is_available():
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_mem < 4:
                    accumulation_steps = 2
                elif total_mem > 8:
                    accumulation_steps = 6
            except Exception:
                pass
        interrupted = False

        consolidate_data_files()

        files = []
        candidates = []
        for pattern in ["*.npy", "*.npz"]:
            candidates.extend(glob.glob(os.path.join(data_dir, pattern)))
        if os.path.exists(log_path):
            candidates.append(log_path)
        total_scan = len(candidates)
        scanned = 0
        print("Scanning data files...")
        torch.cuda.empty_cache()
        for f in candidates:
            if stop_optimization_flag.is_set():
                break
            scanned += 1
            progress_bar("数据扫描阶段", scanned, total_scan)
            try:
                if os.path.getsize(f) > 0:
                    files.append(f)
            except:
                continue
        if stop_optimization_flag.is_set():
            interrupted = True
            print("Optimization interrupted before dataset selection.")
        if not files and not interrupted:
            print("No data to train.")
            torch.cuda.empty_cache()
            return

        random.shuffle(files)
        batch_files = [files[i:i+5] for i in range(0, len(files), 5)] if files else []

        def estimate_steps(file_subset):
            total = 0
            for path in file_subset:
                try:
                    if path == log_path:
                        for payload in iterate_binary_log(path):
                            buf = io.BytesIO(payload)
                            data = np.load(buf, allow_pickle=True)
                            structured_npz = isinstance(data, np.lib.npyio.NpzFile)
                            structured_array = hasattr(data, 'dtype') and data.dtype.names and 'action' in data.dtype.names
                            structured = structured_npz or structured_array
                            length = len(data['action']) if structured else len(data)
                            steps = max(1, length - seq_len)
                            total += steps
                            if structured_npz:
                                data.close()
                        continue
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
        if total_steps <= 0 and not interrupted:
            print("No data to train.")
            torch.cuda.empty_cache()
            return

        epochs = 1
        current_step = 0
        total_chunks = len(batch_files)
        last_loss_value = 0.0
        train_batch_size = 4
        plateau_counter = 0
        prev_loss = float('inf')
        finetune_enabled = False
        torch.cuda.empty_cache()
        for idx, bf in enumerate(batch_files):
            if stop_optimization_flag.is_set():
                interrupted = True
                break
            chunk_trained = False
            while not chunk_trained and not stop_optimization_flag.is_set():
                dataset = StreamingGameDataset(bf)
                loader_workers = 2 if os.name != "nt" else 0
                loader = DataLoader(dataset, batch_size=train_batch_size, drop_last=False, num_workers=loader_workers, pin_memory=True, persistent_workers=loader_workers > 0)
                optimizer.zero_grad()
                attempt_steps = 0
                try:
                    for _ in range(epochs):
                        pending_steps = 0
                        for batch_idx, (imgs, mins, labels, next_frames) in enumerate(loader):
                            if stop_optimization_flag.is_set():
                                interrupted = True
                                break
                            imgs = imgs.to(device)
                            mins = mins.to(device)
                            labels = labels.to(device)
                            next_frames = next_frames.to(device)

                            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                                log_var_action = torch.clamp(model.log_var_action, -6.0, 6.0)
                                log_var_prediction = torch.clamp(model.log_var_prediction, -6.0, 6.0)
                                log_var_energy = torch.clamp(model.log_var_energy, -6.0, 6.0)
                                grid_logits, pred_feat, button_logits, _ = model(imgs, mins, None)
                                target_grid = labels[:, 0].long()
                                target_buttons = labels[:, 1:]
                                grid_loss = nn.functional.cross_entropy(grid_logits, target_grid)
                                imitation_loss = grid_loss + bce_loss(button_logits, target_buttons)
                                target_feat = model.encode_features(next_frames)
                                pred_loss = mse_loss(pred_feat, target_feat)
                                energy_loss = torch.mean(torch.norm(button_logits, dim=1) + 1e-4 * torch.norm(grid_logits, dim=1))
                                total_loss = 0.5 * torch.exp(-log_var_action) * imitation_loss + 0.5 * log_var_action
                                total_loss = total_loss + 0.5 * torch.exp(-log_var_prediction) * pred_loss + 0.5 * log_var_prediction
                                total_loss = total_loss + 0.5 * torch.exp(-log_var_energy) * energy_loss + 0.5 * log_var_energy
                                total_loss = total_loss / accumulation_steps
                            scaler.scale(total_loss).backward()
                            pending_steps += 1
                            attempt_steps += 1
                            if pending_steps % accumulation_steps == 0:
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                                pending_steps = 0
                        current_step += 1
                        last_loss_value = total_loss.item()
                        if abs(last_loss_value - prev_loss) < 1e-4:
                            plateau_counter += 1
                        else:
                            plateau_counter = 0
                        prev_loss = last_loss_value
                        if plateau_counter >= 3 and not finetune_enabled:
                            model.enable_backbone_finetune(0.25)
                            finetune_enabled = True
                        progress_bar("模型训练阶段", current_step, total_steps, f"Loss: {last_loss_value:.4f} | Chunk {idx+1}/{total_chunks}")
                    if pending_steps > 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if stop_optimization_flag.is_set():
                            interrupted = True
                            break
                    chunk_trained = True
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        current_step = max(0, current_step - attempt_steps)
                        torch.cuda.empty_cache()
                        gc.collect()
                        if train_batch_size > 1:
                            train_batch_size = max(1, train_batch_size // 2)
                            print(f"OOM detected, reducing batch size to {train_batch_size}")
                            continue
                        interrupted = True
                        stop_optimization_flag.set()
                    else:
                        current_step = max(0, current_step - attempt_steps)
                        interrupted = True
                        print(f"Training error: {e}")
                        stop_optimization_flag.set()
                finally:
                    del loader
                    del dataset
                    gc.collect()
                    torch.cuda.empty_cache()
            if stop_optimization_flag.is_set():
                interrupted = True
                break

        temp_path = os.path.join(model_dir, "ai_model_temp.pth")
        print(f"Final Loss Snapshot: {last_loss_value:.6f} | Steps: {current_step}/{total_steps}")
        alpha = float(model.log_var_action.detach().item())
        beta = float(model.log_var_prediction.detach().item())
        gamma = float(model.log_var_energy.detach().item())
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "alpha": alpha, "beta": beta, "gamma": gamma, "last_loss": last_loss_value, "steps": current_step, "total_steps": total_steps, "interrupted": interrupted, "timestamp": time.time()}, temp_path)
        try:
            if os.path.exists(model_path):
                shutil.copy2(model_path, backup_path)
            if os.path.exists(model_path):
                os.remove(model_path)
            os.rename(temp_path, model_path)
            if interrupted:
                print("Optimization Interrupted (Safe Save).")
            else:
                print("Optimization Complete (Safe Save).")
        except Exception as e:
            print(f"Save error: {e}")
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, model_path)
        focus_weight = float(torch.exp(-model.log_var_action.detach()).item())
        curiosity_weight = float(torch.exp(-model.log_var_prediction.detach()).item())
        laziness_penalty = float(torch.exp(-model.log_var_energy.detach()).item())
        if interrupted:
            print("[Optimization Interrupted]")
            print(f"> Reason: Keyboard interrupt after {current_step}/{total_steps} steps")
        else:
            print("[Optimization Done]")
        print(f"> Imitation Weight (Focus): {focus_weight:.3f}  <-- exp(-alpha)")
        print(f"> Prediction Weight (Curiosity): {curiosity_weight:.3f}  <-- exp(-beta)")
        print(f"> Energy Penalty (Laziness): {laziness_penalty:.3f}  <-- exp(-gamma)")
        print(f"> Last Loss: {last_loss_value:.6f}")
        print(f"> Model Saved To: {model_path}")
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
        stop_optimization_flag.clear()
        torch.cuda.empty_cache()
        gc.collect()

def start_training_mode():
    global stop_training_flag, current_mode
    stop_training_flag = False

    gc.collect()
    torch.cuda.empty_cache()

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
    model.train()
    surprise_optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    surprise_scaler = GradScaler("cuda", enabled=device.type == "cuda")
    mse_surprise = nn.MSELoss()
    mouse_ctrl = mouse.Controller()
    input_buffer_img = []
    input_buffer_mouse = []
    with mouse_lock:
        prev_pos = (mouse_state["x"], mouse_state["y"])
    target_history = deque(maxlen=8)
    try:
        model.share_memory()
    except Exception:
        pass
    time.sleep(1.0)
    stop_training_flag = False

    def bezier_interp(p0, p1, p2, t):
        one_minus = 1.0 - t
        return (one_minus * one_minus * p0) + (2.0 * one_minus * t * p1) + (t * t * p2)

    while not stop_training_flag:
        time.sleep(0)
        start_time = time.time()
        if os.name == "nt":
            try:
                if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                    stop_training_flag = True
                    break
            except Exception as e:
                print(f"Priority key error: {e}")
        try:
            frame_img, frame_ts = get_latest_frame()
            if frame_img is None:
                time.sleep(0.01)
                continue
            with mouse_lock:
                curr_x, curr_y = mouse_state["x"], mouse_state["y"]
                l_down = mouse_state["l_down"]
                r_down = mouse_state["r_down"]
                scroll = mouse_state["scroll"]
                l_up_pos = mouse_state["l_up_pos"]
                r_up_pos = mouse_state["r_up_pos"]
                traj = list(temp_trajectory)
            mouse_state["scroll"] = 0
            img_tensor = cv2.cvtColor(frame_img, cv2.COLOR_BGRA2RGB)
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
            prev_pos = (curr_x, curr_y)
            if len(input_buffer_img) > seq_len:
                input_buffer_img.pop(0)
                input_buffer_mouse.pop(0)
            if len(input_buffer_img) == seq_len:
                t_imgs = torch.FloatTensor(np.array([input_buffer_img])).to(device)
                t_mins = torch.FloatTensor(np.array([input_buffer_mouse])).to(device)
                surprise_optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=device.type == "cuda"):
                    grid_logits, pred_feat, button_logits, _ = model(t_imgs, t_mins, None)
                grid_probs = torch.softmax(grid_logits[0], dim=-1)
                pred_cell = torch.argmax(grid_probs).item()
                cell_x = pred_cell % grid_w
                cell_y = pred_cell // grid_w
                target_x = int(((cell_x + 0.5) / grid_w) * screen_w)
                target_y = int(((cell_y + 0.5) / grid_h) * screen_h)
                target_history.append((target_x, target_y))
                avg_x = sum(p[0] for p in target_history) / len(target_history)
                avg_y = sum(p[1] for p in target_history) / len(target_history)
                current_mouse = mouse_ctrl.position
                control_x = (current_mouse[0] + avg_x) / 2.0
                control_y = (current_mouse[1] + avg_y) / 2.0
                eased_x = bezier_interp(current_mouse[0], control_x, avg_x, 0.35)
                eased_y = bezier_interp(current_mouse[1], control_y, avg_y, 0.35)
                filtered_x = (eased_x * 0.7) + (current_mouse[0] * 0.3)
                filtered_y = (eased_y * 0.7) + (current_mouse[1] * 0.3)
                jitter_x = random.uniform(-2.0, 2.0)
                jitter_y = random.uniform(-2.0, 2.0)
                final_x = int(min(max(0, filtered_x + jitter_x), screen_w - 1))
                final_y = int(min(max(0, filtered_y + jitter_y), screen_h - 1))
                pred_buttons = torch.sigmoid(button_logits[0]).detach().cpu().numpy()
                mouse_ctrl.position = (final_x, final_y)
                if pred_buttons[0] > 0.5 and not l_down:
                    mouse_ctrl.press(mouse.Button.left)
                elif pred_buttons[0] <= 0.5 and l_down:
                    mouse_ctrl.release(mouse.Button.left)
                if pred_buttons[1] > 0.5 and not r_down:
                    mouse_ctrl.press(mouse.Button.right)
                elif pred_buttons[1] <= 0.5 and r_down:
                    mouse_ctrl.release(mouse.Button.right)
                actual_frame, _ = get_latest_frame()
                if actual_frame is not None:
                    actual_rgb = cv2.cvtColor(actual_frame, cv2.COLOR_BGRA2RGB) if actual_frame.shape[2] == 4 else actual_frame
                    actual_tensor = torch.FloatTensor(actual_rgb.transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
                    with autocast(device_type="cuda", enabled=device.type == "cuda"):
                        actual_feat = model.encode_features(actual_tensor)
                        surprise_loss = mse_surprise(pred_feat, actual_feat)
                    if surprise_loss.item() > 0.001:
                        surprise_scaler.scale(surprise_loss).backward()
                        surprise_scaler.step(surprise_optimizer)
                        surprise_scaler.update()
                surprise_optimizer.zero_grad(set_to_none=True)
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

                with mouse_lock:
                    c_state = mouse_state.copy()
                    mouse_state["scroll"] = 0
                    traj = list(temp_trajectory)

                ts_value = frame_ts if frame_ts else start_time
                traj_flat = sample_trajectory(traj, ts_value, fallback_pos=(c_state["x"], c_state["y"]))
                img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGRA2RGB) if frame_img.shape[2] == 4 else frame_img
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

                if len(buffer_images) >= 10000:
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

def interpret_command(cmd):
    variants = set()
    for item in [cmd, cmd.lower()]:
        variants.add(item)
    encs = [sys.stdin.encoding, sys.getdefaultencoding(), "utf-8", "gbk"]
    for enc in encs:
        if not enc:
            continue
        try:
            recon = cmd.encode(enc, errors="ignore").decode("utf-8", errors="ignore")
            variants.update([recon, recon.lower()])
        except Exception:
            pass
        try:
            recon = cmd.encode(enc, errors="ignore").decode("gbk", errors="ignore")
            variants.update([recon, recon.lower()])
        except Exception:
            pass
    if any(v in ("睡眠", "sleep") for v in variants):
        return "sleep"
    if any(v in ("训练", "train") for v in variants):
        return "train"
    return None

def input_loop():
    global current_mode
    while True:
        cmd = input().strip()
        interpreted = interpret_command(cmd)
        if interpreted == "sleep":
            if current_mode == MODE_LEARNING:
                recording_pause_event.set()
                flush_buffers()
                cleanup_before_sleep()
                current_mode = MODE_SLEEP
                optimize_ai()
                current_mode = MODE_LEARNING
                recording_pause_event.clear()
                print("Back to Learning.")
        elif interpreted == "train":
            if current_mode == MODE_LEARNING:
                flush_buffers()
                current_mode = MODE_TRAINING
                print("Entering Training Mode...")
                t_thread = threading.Thread(target=start_training_mode)
                t_thread.daemon = True
                t_thread.start()

if __name__ == "__main__":
    ensure_initial_model()
    set_process_priority()
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
