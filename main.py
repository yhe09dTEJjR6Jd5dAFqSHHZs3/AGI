import sys
import os
import time
import threading
import pickle
import shutil
import glob
import math
import random
import warnings
import subprocess
import importlib
import importlib.util
import json
from collections import deque
import heapq

warnings.filterwarnings("ignore")

def import_or_install(module_name, package_name=None):
    pkg = package_name or module_name
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        result = subprocess.run([sys.executable, "-m", "pip", "install", pkg])
        if result.returncode != 0:
            sys.exit(1)
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        sys.exit(1)
    return importlib.import_module(module_name)

psutil = import_or_install("psutil")
np = import_or_install("numpy")
cv2 = import_or_install("cv2", "opencv-python")
mss = import_or_install("mss")
torch = import_or_install("torch")
torch.backends.cudnn.benchmark = True
nn = torch.nn
optim = torch.optim
F = torch.nn.functional
autocast = import_or_install("torch.cuda.amp").autocast
GradScaler = import_or_install("torch.cuda.amp").GradScaler
QtWidgets = import_or_install("PyQt5.QtWidgets", "PyQt5")
QtCore = import_or_install("PyQt5.QtCore", "PyQt5")
QtGui = import_or_install("PyQt5.QtGui", "PyQt5")
keyboard = import_or_install("pynput.keyboard")
pyautogui = import_or_install("pyautogui")
ctypes = import_or_install("ctypes")
platform = import_or_install("platform")
pynvml = import_or_install("pynvml")

pyautogui.FAILSAFE = False

DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
BASE_DIR = os.path.join(DESKTOP_PATH, "AAA")
BUFFER_DIR = os.path.join(BASE_DIR, "experience_pool_chunks")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
if not os.path.exists(BUFFER_DIR):
    os.makedirs(BUFFER_DIR)

def set_dpi_awareness():
    if platform.system().lower().startswith("win") and hasattr(ctypes, "windll") and hasattr(ctypes.windll, "user32"):
        if hasattr(ctypes.windll, "shcore") and hasattr(ctypes.windll.shcore, "SetProcessDpiAwareness"):
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        if hasattr(ctypes.windll.user32, "SetProcessDPIAware"):
            ctypes.windll.user32.SetProcessDPIAware()

set_dpi_awareness()

def clamp_value(val, lower, upper):
    return max(lower, min(upper, val))

VISION_WIDTH = 256
VISION_HEIGHT = 160

def default_config():
    fps_bounds = [1, 100]
    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
    mem_info = psutil.virtual_memory()
    mem_ratio = 1.0 - mem_info.percent / 100.0
    context_default = clamp_value(int(2 + (cpu_cores or 1) * 0.5), 1, 100)
    latent_default = clamp_value(int(4 + context_default * 0.25), 2, 12)
    fps_dynamic = clamp_value(int(fps_bounds[1] * mem_ratio), fps_bounds[0], fps_bounds[1])
    chunk_limit = int(clamp_value(mem_info.available / (1024**2 * 256), 128, 1024))
    chunk_bytes = int(clamp_value(mem_info.available * 0.05, 64 * 1024 * 1024, 512 * 1024 * 1024))
    lr_dynamic = clamp_value(1e-3 * mem_ratio * (cpu_cores or 1) / 4.0, 5e-4, 5e-3)
    mini_batch_dynamic = int(clamp_value((cpu_cores or 1) * mem_ratio, 4, 32))
    train_steps_dynamic = int(clamp_value(50 + mem_ratio * 150, 50, 300))
    config = {
        "capture": {
            "fps": fps_dynamic,
            "fps_bounds": fps_bounds,
            "scale": 1.0,
            "resolution_factor": 1.0,
            "context_window": context_default,
            "standard_resolution": [VISION_HEIGHT, VISION_WIDTH],
            "latent_pool": latent_default
        },
        "limits": {
            "max_vram_gb": 4.0,
            "max_ram_gb": 16.0,
            "max_buffer_gb": 20.0
        },
        "buffer": {
            "chunk_entry_limit": chunk_limit,
            "chunk_byte_limit": chunk_bytes
        },
        "resource": {
            "cooldown_seconds": clamp_value(0.5 + mem_ratio * 0.5, 0.3, 1.5),
            "min_scale": 1.0,
            "max_scale": 1.0,
            "min_resolution_factor": 1.0,
            "max_resolution_factor": 1.0,
            "min_context": 1,
            "max_context": 100
        },
        "learning": {
            "survival_penalty": 0.01,
            "sample_batch": int(clamp_value(32 * mem_ratio + cpu_cores, 32, 128)),
            "train_steps": train_steps_dynamic,
            "mini_batch": mini_batch_dynamic,
            "learning_rate": lr_dynamic,
            "grad_clip": clamp_value(0.5 + mem_ratio * 1.0, 0.5, 1.5)
        }
    }
    return config

def load_or_create_config():
    cfg = default_config()
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    defaults = default_config()
    def merge(defaults_dict, cfg_dict):
        for k, v in defaults_dict.items():
            if isinstance(v, dict):
                cfg_dict[k] = merge(v, cfg_dict.get(k, {}))
            else:
                cfg_dict[k] = cfg_dict.get(k, v)
        return cfg_dict
    cfg = merge(defaults, cfg)
    capture_cfg = cfg["capture"]
    fps_low = clamp_value(int(capture_cfg.get("fps_bounds", [1, 100])[0]), 1, 100)
    fps_high = clamp_value(int(capture_cfg.get("fps_bounds", [1, 100])[1]), fps_low, 100)
    capture_cfg["fps_bounds"] = [fps_low, fps_high]
    capture_cfg["scale"] = clamp_value(capture_cfg.get("scale", defaults["capture"]["scale"]), 0.01, 1.0)
    capture_cfg["resolution_factor"] = clamp_value(capture_cfg.get("resolution_factor", defaults["capture"]["resolution_factor"]), 0.01, 1.0)
    capture_cfg["context_window"] = int(clamp_value(int(capture_cfg.get("context_window", defaults["capture"]["context_window"])), 1, 100))
    resource_cfg = cfg["resource"]
    resource_cfg["max_scale"] = clamp_value(resource_cfg.get("max_scale", 1.0), 0.5, 1.0)
    resource_cfg["min_scale"] = clamp_value(resource_cfg.get("min_scale", defaults["resource"]["min_scale"]), 0.01, resource_cfg["max_scale"])
    resource_cfg["min_resolution_factor"] = clamp_value(resource_cfg.get("min_resolution_factor", defaults["resource"]["min_resolution_factor"]), 0.01, resource_cfg["max_resolution_factor"])
    resource_cfg["min_context"] = int(clamp_value(int(resource_cfg.get("min_context", defaults["resource"]["min_context"])), 1, 100))
    resource_cfg["max_context"] = int(clamp_value(int(resource_cfg.get("max_context", defaults["resource"]["max_context"])), resource_cfg["min_context"], 100))
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg

config_data = load_or_create_config()

global_running = True
global_optimizing = False
global_pause_recording = False

screen_width, screen_height = pyautogui.size()
center_x, center_y = screen_width // 2, screen_height // 2

def apply_config():
    global current_fps, current_scale, resolution_factor, temporal_context_window
    global max_vram_gb, max_ram_gb, max_buffer_gb, standard_res, latent_pool
    global chunk_entry_limit, chunk_byte_limit
    capture_cfg = config_data["capture"]
    limits_cfg = config_data["limits"]
    buffer_cfg = config_data["buffer"]
    resource_cfg = config_data["resource"]
    fps_lower, fps_upper = capture_cfg["fps_bounds"]
    current_fps = clamp_value(capture_cfg["fps"], fps_lower, fps_upper)
    current_scale = clamp_value(capture_cfg["scale"], resource_cfg["min_scale"], resource_cfg["max_scale"])
    resolution_factor = clamp_value(capture_cfg["resolution_factor"], resource_cfg["min_resolution_factor"], resource_cfg["max_resolution_factor"])
    temporal_context_window = clamp_value(int(capture_cfg["context_window"]), resource_cfg["min_context"], resource_cfg["max_context"])
    standard_res = tuple(capture_cfg["standard_resolution"])
    latent_pool = max(1, int(capture_cfg["latent_pool"]))
    max_vram_gb = limits_cfg["max_vram_gb"]
    max_ram_gb = limits_cfg["max_ram_gb"]
    max_buffer_gb = limits_cfg["max_buffer_gb"]
    chunk_entry_limit = int(buffer_cfg["chunk_entry_limit"])
    chunk_byte_limit = int(buffer_cfg["chunk_byte_limit"])

apply_config()

gpu_handle = None

def init_gpu_handle():
    global gpu_handle
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

init_gpu_handle()

def gpu_stats():
    if gpu_handle is None:
        return 0.0, 0.0
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    vram_percent = min(100.0, mem_info.used / (1024**3) / max(max_vram_gb, 1e-6) * 100)
    return float(util.gpu), float(vram_percent)

def persist_config():
    config_data["capture"]["fps"] = current_fps
    config_data["capture"]["scale"] = current_scale
    config_data["capture"]["resolution_factor"] = resolution_factor
    config_data["capture"]["context_window"] = temporal_context_window
    config_data["capture"]["standard_resolution"] = list(standard_res)
    config_data["capture"]["latent_pool"] = latent_pool
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

lock = threading.Lock()
mouse_left_down = False
mouse_right_down = False

def scaled_standard_res():
    return standard_res

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((latent_pool, latent_pool))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return x

    def decode(self, latent, output_size):
        x = self.decoder(latent)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent, x.shape[2:])

class FuturePredictor(nn.Module):
    def __init__(self):
        super(FuturePredictor, self).__init__()
        embed_dim = 64 * latent_pool * latent_pool
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, aggregated):
        return self.fc(aggregated)

class ActionPredictor(nn.Module):
    def __init__(self):
        super(ActionPredictor, self).__init__()
        embed_dim = 64 * latent_pool * latent_pool
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU()
        )
        self.pos_head = nn.Linear(256, 2)
        self.state_head = nn.Linear(256, 4)

    def forward(self, aggregated):
        features = self.fc(aggregated)
        pos = self.pos_head(features)
        state_logits = self.state_head(features)
        return pos, state_logits

class ValuePredictor(nn.Module):
    def __init__(self):
        super(ValuePredictor, self).__init__()
        embed_dim = 64 * latent_pool * latent_pool
        self.value_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, aggregated):
        return self.value_net(aggregated).squeeze(1)

class ExperienceBuffer:
    def __init__(self):
        self.buffer = []
        self.heap = []
        self.total_bytes = 0
        self.counter = 0
        self.chunk_files = []
        temp_dir = os.path.join(BUFFER_DIR, ".tmp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def add(self, state_img, mouse_state, reward, td_error, screen_novelty, latent_tensor):
        img = state_img.float()
        if img.dim() == 4:
            img = img[0]
        res_h, res_w = scaled_standard_res()
        img = F.interpolate(img.unsqueeze(0), size=(res_h, res_w), mode="bilinear", align_corners=False).squeeze(0)
        img = torch.clamp(img * 255.0, 0, 255).to(torch.uint8)
        mouse = mouse_state.float()
        if mouse.dim() > 1:
            mouse = mouse[0]
        latent_store = latent_tensor.detach().cpu().half()
        data = {
            "img": img,
            "mouse": mouse,
            "reward": reward,
            "error": td_error,
            "novelty": screen_novelty,
            "latent": latent_store
        }
        self.buffer.append(data)
        entry_bytes = self._sample_bytes(data)
        self.total_bytes += entry_bytes
        idx = len(self.buffer) - 1
        self.counter += 1
        heapq.heappush(self.heap, (reward, idx))
        self.enforce_limit()

    def _sample_bytes(self, entry):
        img_size = entry["img"].element_size() * entry["img"].nelement()
        mouse_size = entry["mouse"].element_size() * entry["mouse"].nelement()
        latent_size = entry["latent"].element_size() * entry["latent"].nelement()
        meta_size = 24
        return img_size + mouse_size + latent_size + meta_size

    def _buffer_size_gb(self):
        return self.total_bytes / (1024**3)

    def rebuild_metadata(self):
        self.total_bytes = sum(self._sample_bytes(item) for item in self.buffer)
        self.heap = [(item["reward"], idx) for idx, item in enumerate(self.buffer)]
        heapq.heapify(self.heap)
        self.counter = len(self.buffer)

    def load_chunks(self):
        if not os.path.isdir(BUFFER_DIR):
            return False
        chunk_files = sorted(glob.glob(os.path.join(BUFFER_DIR, "chunk_*.pt")))
        if not chunk_files:
            return False
        self.chunk_files = chunk_files
        retained = []
        total_bytes = 0
        limit_bytes = max_buffer_gb * (1024**3)
        for path in reversed(chunk_files):
            entries = torch.load(path, map_location="cpu")
            for entry in reversed(entries):
                entry_bytes = self._sample_bytes(entry)
                if total_bytes + entry_bytes > limit_bytes:
                    break
                retained.append(entry)
                total_bytes += entry_bytes
            if total_bytes >= limit_bytes:
                break
        retained.reverse()
        self.buffer = retained
        self.rebuild_metadata()
        return True

    def enforce_limit(self):
        removed_any = False
        while self._buffer_size_gb() > max_buffer_gb and len(self.buffer) > 0:
            removed = self.buffer.pop(0)
            self.total_bytes -= self._sample_bytes(removed)
            removed_any = True
        if removed_any:
            self.rebuild_metadata()

    def sample_sequences(self, batch_size, window):
        if len(self.buffer) <= window:
            return []
        sequences = []
        max_start = len(self.buffer) - window - 1
        for _ in range(batch_size):
            idx = random.randint(0, max_start)
            context_entries = self.buffer[idx:idx + window]
            target_entry = self.buffer[idx + window]
            sequences.append((context_entries, target_entry))
        return sequences

    def save(self):
        temp_dir = os.path.join(BUFFER_DIR, ".tmp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        chunk = []
        chunk_bytes = 0
        existing = sorted(glob.glob(os.path.join(BUFFER_DIR, "chunk_*.pt")))
        start_idx = 0
        if existing:
            last = os.path.splitext(os.path.basename(existing[-1]))[0].split("_")
            if len(last) > 1 and last[-1].isdigit():
                start_idx = int(last[-1]) + 1
        chunk_idx = start_idx
        for entry in self.buffer:
            entry_bytes = self._sample_bytes(entry)
            if chunk and (chunk_bytes + entry_bytes > chunk_byte_limit or len(chunk) >= chunk_entry_limit):
                torch.save(chunk, os.path.join(temp_dir, f"chunk_{chunk_idx}.pt"))
                chunk_idx += 1
                chunk = []
                chunk_bytes = 0
            chunk.append(entry)
            chunk_bytes += entry_bytes
        torch.save(chunk, os.path.join(temp_dir, f"chunk_{chunk_idx}.pt"))
        for existing in glob.glob(os.path.join(BUFFER_DIR, "chunk_*.pt")):
            os.remove(existing)
        for fname in os.listdir(temp_dir):
            os.replace(os.path.join(temp_dir, fname), os.path.join(BUFFER_DIR, fname))
        shutil.rmtree(temp_dir)
        self._enforce_disk_limit()

    def _disk_usage_gb(self):
        size = 0
        for path in glob.glob(os.path.join(BUFFER_DIR, "chunk_*.pt")):
            size += os.path.getsize(path)
        return size / (1024**3)

    def _enforce_disk_limit(self):
        files = sorted(glob.glob(os.path.join(BUFFER_DIR, "chunk_*.pt")))
        while self._disk_usage_gb() > 20.0 and files:
            oldest = files.pop(0)
            os.remove(oldest)

exp_buffer = ExperienceBuffer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ai_model = Autoencoder().to(device)
future_model = FuturePredictor().to(device)
predict_action_model = ActionPredictor().to(device)
policy_action_model = ActionPredictor().to(device)
value_model = ValuePredictor().to(device)
scaler = GradScaler(enabled=device.type == "cuda")

def ensure_files():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    if not os.path.exists(BUFFER_DIR):
        os.makedirs(BUFFER_DIR)
    buffer_path = os.path.join(BASE_DIR, "experience_pool.pkl")
    model_files = {
        "screen_future": os.path.join(BASE_DIR, "ai_model_screen_future.pth"),
        "mouse_predict": os.path.join(BASE_DIR, "ai_model_mouse_predict.pth"),
        "mouse_output": os.path.join(BASE_DIR, "ai_model_mouse_output.pth"),
        "autoencoder": os.path.join(BASE_DIR, "ai_model_autoencoder.pth"),
        "value": os.path.join(BASE_DIR, "ai_model_value.pth")
    }
    if not os.path.exists(buffer_path):
        with open(buffer_path, "wb") as f:
            pickle.dump([], f)
    chunk_files = glob.glob(os.path.join(BUFFER_DIR, "chunk_*.pt"))
    if not chunk_files:
        torch.save([], os.path.join(BUFFER_DIR, "chunk_0.pt"))
    if not os.path.exists(model_files["autoencoder"]):
        torch.save(ai_model.state_dict(), model_files["autoencoder"])
    if not os.path.exists(model_files["mouse_output"]):
        torch.save(policy_action_model.state_dict(), model_files["mouse_output"])
    if not os.path.exists(model_files["screen_future"]):
        torch.save(future_model.state_dict(), model_files["screen_future"])
    if not os.path.exists(model_files["mouse_predict"]):
        torch.save(predict_action_model.state_dict(), model_files["mouse_predict"])
    if not os.path.exists(model_files["value"]):
        torch.save(value_model.state_dict(), model_files["value"])

def load_state():
    global exp_buffer
    buffer_path = os.path.join(BASE_DIR, "experience_pool.pkl")
    model_files = {
        "screen_future": os.path.join(BASE_DIR, "ai_model_screen_future.pth"),
        "mouse_predict": os.path.join(BASE_DIR, "ai_model_mouse_predict.pth"),
        "mouse_output": os.path.join(BASE_DIR, "ai_model_mouse_output.pth"),
        "autoencoder": os.path.join(BASE_DIR, "ai_model_autoencoder.pth"),
        "value": os.path.join(BASE_DIR, "ai_model_value.pth")
    }
    loaded = exp_buffer.load_chunks()
    if not loaded and os.path.exists(buffer_path):
        with open(buffer_path, "rb") as f:
            exp_buffer.buffer = pickle.load(f)
        exp_buffer.save()
        exp_buffer.rebuild_metadata()
        loaded = True
    if not loaded and not glob.glob(os.path.join(BUFFER_DIR, "chunk_*.pt")):
        torch.save([], os.path.join(BUFFER_DIR, "chunk_0.pt"))
    if os.path.exists(model_files["autoencoder"]):
        ai_model.load_state_dict(torch.load(model_files["autoencoder"], map_location=device))
    if os.path.exists(model_files["mouse_output"]):
        policy_action_model.load_state_dict(torch.load(model_files["mouse_output"], map_location=device))
    if os.path.exists(model_files["screen_future"]):
        future_model.load_state_dict(torch.load(model_files["screen_future"], map_location=device))
    if os.path.exists(model_files["mouse_predict"]):
        predict_action_model.load_state_dict(torch.load(model_files["mouse_predict"], map_location=device))
    elif os.path.exists(model_files["mouse_output"]):
        predict_action_model.load_state_dict(torch.load(model_files["mouse_output"], map_location=device))
    if not os.path.exists(model_files["mouse_output"]) and os.path.exists(model_files["mouse_predict"]):
        policy_action_model.load_state_dict(torch.load(model_files["mouse_predict"], map_location=device))
    if os.path.exists(model_files["value"]):
        value_model.load_state_dict(torch.load(model_files["value"], map_location=device))

def save_state():
    torch.save(ai_model.state_dict(), os.path.join(BASE_DIR, "ai_model_autoencoder.pth"))
    torch.save(policy_action_model.state_dict(), os.path.join(BASE_DIR, "ai_model_mouse_output.pth"))
    torch.save(future_model.state_dict(), os.path.join(BASE_DIR, "ai_model_screen_future.pth"))
    torch.save(predict_action_model.state_dict(), os.path.join(BASE_DIR, "ai_model_mouse_predict.pth"))
    torch.save(value_model.state_dict(), os.path.join(BASE_DIR, "ai_model_value.pth"))
    exp_buffer.save()

def position_scale():
    return torch.tensor([float(center_x), float(center_y)], device=device)

def get_mouse_state():
    x_phys, y_phys = pyautogui.position()
    x = x_phys - center_x
    y = center_y - y_phys
    left_btn = 0
    right_btn = 0
    if platform.system().lower().startswith("win") and hasattr(ctypes, "windll") and hasattr(ctypes.windll, "user32"):
        state_func = ctypes.windll.user32.GetAsyncKeyState
        if state_func(0x01) & 0x8000: left_btn = 1
        if state_func(0x02) & 0x8000: right_btn = 1
    status = 0
    if left_btn and right_btn: status = 3
    elif left_btn: status = 1
    elif right_btn: status = 2
    return [float(x), float(y), float(status)]

def apply_mouse_buttons(status_idx):
    global mouse_left_down, mouse_right_down
    target_left = status_idx in (1, 3)
    target_right = status_idx in (2, 3)
    if target_left != mouse_left_down:
        if target_left:
            if platform.system().lower().startswith("win") and hasattr(ctypes, "windll") and hasattr(ctypes.windll, "user32"):
                ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
            pyautogui.mouseDown(button="left")
        else:
            if platform.system().lower().startswith("win") and hasattr(ctypes, "windll") and hasattr(ctypes.windll, "user32"):
                ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
            pyautogui.mouseUp(button="left")
        mouse_left_down = target_left
    if target_right != mouse_right_down:
        if target_right:
            if platform.system().lower().startswith("win") and hasattr(ctypes, "windll") and hasattr(ctypes.windll, "user32"):
                ctypes.windll.user32.mouse_event(0x0008, 0, 0, 0, 0)
            pyautogui.mouseDown(button="right")
        else:
            if platform.system().lower().startswith("win") and hasattr(ctypes, "windll") and hasattr(ctypes.windll, "user32"):
                ctypes.windll.user32.mouse_event(0x0010, 0, 0, 0, 0)
            pyautogui.mouseUp(button="right")
        mouse_right_down = target_right

def move_mouse(pred_abs_x, pred_abs_y, pred_status):
    target_x = clamp_value(pred_abs_x + center_x, 0, screen_width - 1)
    target_y = clamp_value(center_y - pred_abs_y, 0, screen_height - 1)
    if platform.system().lower().startswith("win"):
        ctypes.windll.user32.SetCursorPos(int(target_x), int(target_y))
    else:
        pyautogui.moveTo(int(target_x), int(target_y), _pause=False)
    apply_mouse_buttons(pred_status)

def aggregate_latents(latents):
    embed_dim = 64 * latent_pool * latent_pool
    if len(latents) == 0:
        return torch.zeros(1, embed_dim * 2, device=device)
    stacked = torch.stack([l.to(device) for l in latents]).view(len(latents), -1)
    mean = stacked.mean(dim=0, keepdim=True)
    std = stacked.std(dim=0, keepdim=True) + 1e-6
    return torch.cat([mean, std], dim=1)

class ResourceMonitor(threading.Thread):
    def run(self):
        global current_fps, temporal_context_window
        r_cfg = config_data["resource"]
        cooldown = r_cfg["cooldown_seconds"]
        target_high = 61.8
        target_low = 38.2
        last_adjust = time.time()
        while global_running:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            gpu_util, vram_util = gpu_stats()
            vram_torch = 0.0
            if torch.cuda.is_available():
                vram_use = torch.cuda.memory_allocated() / (1024**3)
                vram_torch = min(100.0, vram_use / max_vram_gb * 100)
            m_val = max(cpu, mem, gpu_util, vram_util, vram_torch)
            now = time.time()
            if now - last_adjust >= cooldown:
                with lock:
                    fps_lower, fps_upper = config_data["capture"]["fps_bounds"]
                    context_min = r_cfg["min_context"]
                    context_max = r_cfg["max_context"]
                    if m_val > target_high:
                        delta = max(1, int((m_val - target_high) / 5) + 1)
                        current_fps = max(fps_lower, current_fps - delta)
                        temporal_context_window = max(context_min, temporal_context_window - delta)
                    elif m_val < target_low:
                        delta = max(1, int((target_low - m_val) / 5) + 1)
                        current_fps = min(fps_upper, current_fps + delta)
                        temporal_context_window = min(context_max, temporal_context_window + delta)
                    last_adjust = now
                    persist_config()
            time.sleep(1)

class AgentThread(threading.Thread):
    def run(self):
        global global_pause_recording
        import random
        sct = mss.mss()
        criterion = nn.MSELoss()
        context_latents = deque(maxlen=config_data["resource"]["max_context"])
        last_cache_clear = time.time()
        while global_running:
            if global_optimizing or global_pause_recording:
                time.sleep(0.1)
                continue
            start_time = time.time()
            with lock:
                fps = current_fps
                window_len = temporal_context_window
            monitor = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
            img_np = np.array(sct.grab(monitor))
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)
            target_w = VISION_WIDTH
            target_h = VISION_HEIGHT
            img_resized = cv2.resize(img_np, (target_w, target_h))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                aggregated = aggregate_latents(list(context_latents)[-window_len:])
                predicted_latent_vec = future_model(aggregated)
                predicted_latent = predicted_latent_vec.view(1, 64, latent_pool, latent_pool)
                predicted_img = ai_model.decode(predicted_latent, img_tensor.shape[2:])
            mouse_val = get_mouse_state()
            mouse_tensor = torch.tensor([mouse_val], dtype=torch.float32).to(device)
            pos_scale_tensor = position_scale()
            with torch.no_grad():
                latent = ai_model.encode(img_tensor)
            context_latents.append(latent.detach().cpu())
            with torch.no_grad():
                pred_pos_raw, pred_status_logits = predict_action_model(aggregated)
                policy_pos_raw, policy_status_logits = policy_action_model(aggregated)
                policy_status_probs = torch.softmax(policy_status_logits, dim=1)
            pred_mouse = torch.zeros((1, 3), device=device)
            pred_mouse_norm = torch.tanh(pred_pos_raw[:, :2])
            pred_mouse[:, 0] = pred_mouse_norm[:, 0] * pos_scale_tensor[0]
            pred_mouse[:, 1] = pred_mouse_norm[:, 1] * pos_scale_tensor[1]
            pred_status_idx = int(torch.argmax(policy_status_probs, dim=1).item())
            mouse_norm = torch.clamp(mouse_tensor[:, :2] / pos_scale_tensor, -1.0, 1.0)
            pos_loss = torch.mean((pred_mouse_norm - mouse_norm) ** 2)
            status_target = torch.tensor([int(mouse_val[2])], device=device, dtype=torch.long)
            status_loss = F.cross_entropy(pred_status_logits, status_target)
            action_novelty_raw = (pos_loss + status_loss).item()
            action_novelty = math.log1p(action_novelty_raw) * 10000
            screen_mse = criterion(predicted_img, img_tensor).item()
            screen_novelty = math.log1p(screen_mse) * 10000
            policy_mouse = torch.zeros((1, 3), device=device)
            policy_mouse_norm = torch.tanh(policy_pos_raw[:, :2])
            noise_scale = clamp_value((screen_novelty + action_novelty) / 20000.0, 0.01, 0.5)
            action_noise = torch.randn_like(policy_mouse_norm) * noise_scale
            noisy_mouse_norm = torch.clamp(policy_mouse_norm + action_noise, -1.0, 1.0)
            policy_mouse[:, 0] = noisy_mouse_norm[:, 0] * pos_scale_tensor[0]
            policy_mouse[:, 1] = noisy_mouse_norm[:, 1] * pos_scale_tensor[1]
            survival_penalty = config_data["learning"]["survival_penalty"]
            reward = (screen_novelty * action_novelty) / 10000 - survival_penalty
            td_error = screen_novelty + action_novelty
            exp_buffer.add(img_tensor.cpu(), mouse_tensor.cpu(), reward, td_error, screen_novelty, latent.cpu())
            policy_np = policy_mouse.cpu().numpy()[0]
            move_mouse(float(policy_np[0]), float(policy_np[1]), pred_status_idx)
            if device.type == "cuda" and time.time() - last_cache_clear > 30:
                torch.cuda.empty_cache()
                last_cache_clear = time.time()
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / fps) - elapsed)
            time.sleep(sleep_time)

class SciFiWindow(QtWidgets.QWidget):
    optimizer_signal = QtCore.pyqtSignal()
    finished_signal = QtCore.pyqtSignal()
    status_signal = QtCore.pyqtSignal(str)
    info_signal = QtCore.pyqtSignal(str)
    progress_signal = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.resize(600, 400)
        self.center()
        self.initUI()
        self.optimizer_signal.connect(self.run_optimization)
        self.finished_signal.connect(self.on_optimization_finished)
        self.status_signal.connect(self.label.setText)
        self.info_signal.connect(self.info_label.setText)
        self.progress_signal.connect(self.progress.setValue)

    def center(self):
        frame = self.frameGeometry()
        center_point = QtWidgets.QDesktopWidget().availableGeometry().center()
        frame.moveCenter(center_point)
        self.move(frame.topLeft())

    def initUI(self):
        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(20, 20, 20, 20)
        card = QtWidgets.QFrame()
        card.setObjectName("card")
        card_layout = QtWidgets.QVBoxLayout()
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(20)
        self.label = QtWidgets.QLabel("SYSTEM: ONLINE")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.info_label = QtWidgets.QLabel("AI AGENT RUNNING...")
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.addWidget(self.label)
        card_layout.addStretch()
        card_layout.addWidget(self.info_label)
        card_layout.addWidget(self.progress)
        card.setLayout(card_layout)
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 0)
        shadow.setColor(QtGui.QColor(0, 240, 255, 150))
        card.setGraphicsEffect(shadow)
        outer.addWidget(card)
        self.setLayout(outer)
        style = """
            #card {
                background-color: qradialgradient(cx:0.5, cy:0.5, fx:0.5, fy:0.5, radius:1.0, stop:0 #0a1224, stop:1 #050a14);
                border: 2px solid #00f0ff;
                border-radius: 16px;
            }
            QLabel {
                color: #00f0ff;
                font-family: Consolas;
                font-size: 18px;
                font-weight: bold;
                border: none;
            }
            QProgressBar {
                border: 1px solid #00f0ff;
                background-color: #0a0f1f;
                height: 26px;
                text-align: center;
                color: #00f0ff;
            }
            QProgressBar::chunk {
                background-color: #00f0ff;
            }
        """
        self.setStyleSheet(style)

    def trigger_optimization(self):
        self.status_signal.emit("SYSTEM: OPTIMIZING")
        self.info_signal.emit("TRAINING NEURAL NETWORK...")
        self.progress_signal.emit(0)
        self.optimizer_signal.emit()

    def run_optimization(self):
        t = threading.Thread(target=self._optimize_task)
        t.start()

    def _optimize_task(self):
        global global_optimizing, global_pause_recording
        samples = exp_buffer.sample_sequences(config_data["learning"]["sample_batch"], temporal_context_window)
        if len(samples) == 0:
            global_optimizing = False
            global_pause_recording = False
            self.finished_signal.emit()
            return
        total_steps = config_data["learning"]["train_steps"]
        weight_decay = config_data["learning"]["learning_rate"] * 0.1
        optimizer = optim.Adam(list(ai_model.parameters()) + list(predict_action_model.parameters()) + list(policy_action_model.parameters()) + list(future_model.parameters()) + list(value_model.parameters()), lr=config_data["learning"]["learning_rate"], weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        for i in range(total_steps):
            if not global_running:
                break
            batch_indices = list(range(0, len(samples), config_data["learning"]["mini_batch"]))
            for start in batch_indices:
                batch = samples[start:start + config_data["learning"]["mini_batch"]]
                context_latent_list = []
                target_imgs = []
                target_mice = []
                target_latents = []
                rewards = []
                for context_entries, target_entry in batch:
                    latents = [item["latent"].float() for item in context_entries]
                    context_latent_list.append(latents)
                    img_dtype = torch.float16 if device.type == "cuda" else torch.float32
                    target_imgs.append(target_entry["img"].to(dtype=img_dtype) / 255.0)
                    target_mice.append(target_entry["mouse"].float())
                    target_latents.append(target_entry["latent"].float())
                    rewards.append(float(target_entry["reward"]))
                optimizer.zero_grad()
                agg_batch = []
                for latents in context_latent_list:
                    agg = aggregate_latents(latents)
                    agg_batch.append(agg)
                aggregated_tensor = torch.cat(agg_batch, dim=0)
                target_imgs_tensor = torch.stack(target_imgs).to(device)
                target_mice_tensor = torch.stack(target_mice).to(device)
                target_latents_tensor = torch.stack(target_latents).to(device)
                reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
                reward_norm = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std(unbiased=False) + 1e-6)
                with autocast(enabled=device.type == "cuda"):
                    pred_latent_vec = future_model(aggregated_tensor)
                    pred_latent = pred_latent_vec.view(-1, 64, latent_pool, latent_pool)
                    pred_img = ai_model.decode(pred_latent, target_imgs_tensor.shape[2:])
                    pred_pos_raw, pred_status_logits = predict_action_model(aggregated_tensor)
                    policy_pos_raw, policy_status_logits = policy_action_model(aggregated_tensor)
                    pos_scale_tensor = position_scale()
                    pred_pos_norm = torch.tanh(pred_pos_raw[:, :2])
                    policy_pos_norm = torch.tanh(policy_pos_raw[:, :2])
                    recon = ai_model(target_imgs_tensor)
                    loss_screen = loss_fn(pred_img, target_imgs_tensor)
                    target_pos = target_mice_tensor[:, :2]
                    target_pos_norm = torch.clamp(target_pos / pos_scale_tensor, -1.0, 1.0)
                    target_status = target_mice_tensor[:, 2].long()
                    min_std = torch.clamp(target_pos_norm.abs().mean(dim=0).mean(), 0.01, 0.5)
                    max_std = torch.clamp(target_pos_norm.std(dim=0, unbiased=False).mean(), 0.05, 0.8)
                    pos_std = torch.clamp((min_std + max_std) * 0.5, 0.01, 0.8) + 1e-6
                    dist = torch.distributions.Normal(policy_pos_norm, pos_std)
                    log_prob_pos = dist.log_prob(target_pos_norm).sum(dim=1)
                    log_prob_status = F.log_softmax(policy_status_logits, dim=1).gather(1, target_status.unsqueeze(1)).squeeze(1)
                    log_prob = log_prob_pos + log_prob_status
                    values = value_model(aggregated_tensor)
                    advantage = reward_norm.detach()
                    policy_loss = -(advantage * log_prob).mean()
                    value_loss = F.mse_loss(values, reward_tensor)
                    predict_loss = loss_fn(pred_pos_norm, target_pos_norm) + F.cross_entropy(pred_status_logits, target_status)
                    policy_supervised = loss_fn(policy_pos_norm, target_pos_norm) + F.cross_entropy(policy_status_logits, target_status)
                    loss_ae = loss_fn(recon, target_imgs_tensor) + loss_fn(target_latents_tensor, ai_model.encode(target_imgs_tensor))
                    loss = loss_screen + predict_loss + policy_supervised + loss_ae + policy_loss + value_loss
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(list(ai_model.parameters()) + list(predict_action_model.parameters()) + list(policy_action_model.parameters()) + list(future_model.parameters()) + list(value_model.parameters()), config_data["learning"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            progress_val = int((i + 1) / total_steps * 100)
            self.progress_signal.emit(progress_val)
            time.sleep(0.01)
        save_state()
        self.finished_signal.emit()

    def on_optimization_finished(self):
        QtWidgets.QMessageBox.information(self, "SYSTEM", "OPTIMIZATION COMPLETE.\nDATA SAVED.")
        self.label.setText("SYSTEM: ONLINE")
        self.info_label.setText("AI AGENT RUNNING...")
        self.progress.setValue(0)
        global global_optimizing, global_pause_recording
        global_optimizing = False
        global_pause_recording = False

class InputHandler:
    def __init__(self, gui_window):
        self.gui = gui_window
        self.listener_k = keyboard.Listener(on_press=self.on_press)
        self.listener_k.start()

    def on_press(self, key):
        global global_running, global_optimizing, global_pause_recording
        if key == keyboard.Key.esc:
            global_running = False
            QtWidgets.QApplication.quit()
            return False
        if key == keyboard.Key.enter:
            if not global_optimizing:
                global_pause_recording = True
                global_optimizing = True
                self.gui.trigger_optimization()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ensure_files()
    load_state()
    init_state = get_mouse_state()
    init_status = int(init_state[2])
    mouse_left_down = init_status in (1, 3)
    mouse_right_down = init_status in (2, 3)
    window = SciFiWindow()
    window.show()
    monitor_thread = ResourceMonitor()
    monitor_thread.start()
    agent_thread = AgentThread()
    agent_thread.start()
    input_handler = InputHandler(window)
    sys.exit(app.exec_())
