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
import tempfile
import json
import math
import traceback
import builtins
from collections import deque
import tkinter as tk
from tkinter import scrolledtext, ttk
if os.name == "nt":
    import winreg
    import msvcrt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pynvml")

progress_bar_lock = threading.Lock()

def log_exception(context, err=None, extra=None):
    parts = [context]
    if err is not None:
        parts.append(str(err))
    if extra:
        parts.append(str(extra))
    message = " | ".join(parts)
    report_to_window(message, "error")
    update_window_status(message, "error")
    tb = traceback.format_exc()
    if tb:
        detail = tb.strip()
        report_to_window(detail, "error")
        update_window_status(detail, "error")

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
    percent = format(ratio * 100, ".2f")
    update_window_progress(ratio * 100, f"{prefix} {percent}% {suffix}".strip())

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
        "pillow": "PIL",
        "lmdb": "lmdb"
    }
    if os.name == "nt":
        package_map["dxcam"] = "dxcam"
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
import lmdb
if os.name == "nt":
    import dxcam
else:
    dxcam = None

desktop_path = resolve_desktop_path()
base_dir = os.path.join(desktop_path, "AAA")
data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "model")
temp_dir = os.path.join(base_dir, "temp")
log_path = os.path.join(data_dir, "experience.log")
index_path = os.path.join(data_dir, "experience.idx")
lmdb_path = os.path.join(data_dir, "experience.lmdb")
meta_path = os.path.join(data_dir, "experience_meta.json")

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
capture_pause_event = threading.Event()
low_vram_mode = False
input_allowed_event = threading.Event()
input_allowed_event.set()
window_ui = None
user_stop_request_reason = None

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

def report_to_window(msg, level="info"):
    try:
        if window_ui is not None:
            window_ui.log(msg, level)
    except Exception:
        pass

def window_log(msg, level="info", to_status=False):
    report_to_window(msg, level)
    if to_status:
        update_window_status(msg, level)

def window_print(*args, level="info", status=False, **kwargs):
    if not args:
        return
    sep = kwargs.get("sep", " ")
    message = sep.join(str(a) for a in args)
    window_log(message, level, status)

builtins.print = window_print

class SciFiWindow:
    def __init__(self):
        self.queue = queue.Queue()
        self.root = tk.Tk()
        self.root.title("Nebula Core")
        self.root.geometry("520x420")
        self.root.configure(bg="#0a0f1a")
        self.root.resizable(False, False)
        self.mode_var = tk.StringVar(value=current_mode)
        self.status_var = tk.StringVar(value="Initializing...")
        self.progress_var = tk.StringVar(value="0.00%")
        title = tk.Label(self.root, text="AGI Control", fg="#7fffd4", bg="#0a0f1a", font=("Consolas", 16, "bold"))
        title.pack(pady=6)
        mode_frame = tk.Frame(self.root, bg="#0a0f1a")
        mode_frame.pack(fill="x", padx=10)
        tk.Label(mode_frame, text="Mode", fg="#70e1ff", bg="#0a0f1a", font=("Consolas", 12)).pack(side="left")
        tk.Label(mode_frame, textvariable=self.mode_var, fg="#e3f2fd", bg="#0a0f1a", font=("Consolas", 12, "bold")).pack(side="left", padx=8)
        status_frame = tk.Frame(self.root, bg="#0a0f1a")
        status_frame.pack(fill="x", padx=10, pady=4)
        tk.Label(status_frame, text="Status", fg="#70e1ff", bg="#0a0f1a", font=("Consolas", 11)).pack(side="left")
        tk.Label(status_frame, textvariable=self.status_var, fg="#d4fc79", bg="#0a0f1a", font=("Consolas", 10), wraplength=340, justify="left").pack(side="left", padx=6)
        btn_frame = tk.Frame(self.root, bg="#0a0f1a")
        btn_frame.pack(fill="x", padx=10, pady=6)
        tk.Button(btn_frame, text="睡眠", command=self.on_sleep, fg="#0a0f1a", bg="#7fffd4", activebackground="#10b981", width=10).pack(side="left", padx=5)
        tk.Button(btn_frame, text="早停", command=self.on_early_stop, fg="#0a0f1a", bg="#fbbf24", activebackground="#f59e0b", width=10).pack(side="left", padx=5)
        tk.Button(btn_frame, text="训练", command=self.on_train, fg="#0a0f1a", bg="#60a5fa", activebackground="#3b82f6", width=10).pack(side="left", padx=5)
        progress_frame = tk.Frame(self.root, bg="#0a0f1a")
        progress_frame.pack(fill="x", padx=10, pady=4)
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=380, mode="determinate", maximum=100)
        self.progress.pack(side="left", padx=4)
        tk.Label(progress_frame, textvariable=self.progress_var, fg="#e3f2fd", bg="#0a0f1a", font=("Consolas", 10)).pack(side="left", padx=4)
        log_frame = tk.Frame(self.root, bg="#0a0f1a")
        log_frame.pack(fill="both", expand=True, padx=10, pady=6)
        self.log_area = scrolledtext.ScrolledText(log_frame, state="disabled", fg="#9ae6ff", bg="#0f172a", insertbackground="#7fffd4", font=("Consolas", 9))
        self.log_area.pack(fill="both", expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self.minimize)
        self.root.after(200, self.process_queue)

    def process_queue(self):
        try:
            while not self.queue.empty():
                entry = self.queue.get()
                if isinstance(entry, tuple) and len(entry) >= 2:
                    kind, payload = entry
                    if kind == "log":
                        self.log_area.configure(state="normal")
                        self.log_area.insert("end", payload + "\n")
                        self.log_area.see("end")
                        self.log_area.configure(state="disabled")
                    elif kind == "progress":
                        pct, text = payload
                        try:
                            val = float(max(0.0, min(100.0, pct)))
                            self.progress["value"] = val
                            self.progress_var.set(text if text else f"{val:.2f}%")
                        except Exception:
                            pass
                else:
                    self.log_area.configure(state="normal")
                    self.log_area.insert("end", str(entry) + "\n")
                    self.log_area.see("end")
                    self.log_area.configure(state="disabled")
        except Exception:
            pass
        finally:
            self.root.after(200, self.process_queue)

    def log(self, msg, level="info"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.queue.put(("log", f"[{level.upper()} {ts}] {msg}"))
        self.status_var.set(msg)

    def set_mode(self, mode):
        self.mode_var.set(mode)

    def set_progress(self, pct, text=None):
        self.queue.put(("progress", (pct, text or f"{pct:.2f}%")))

    def minimize(self):
        try:
            self.root.iconify()
        except Exception:
            pass

    def on_sleep(self):
        request_sleep_mode()

    def on_early_stop(self):
        request_early_stop()

    def on_train(self):
        request_training_mode()

    def run(self):
        self.root.mainloop()

def init_window():
    global window_ui
    try:
        window_ui = SciFiWindow()
        return True
    except Exception as e:
        print(f"UI init error: {e}")
        return False

def update_window_mode(mode):
    try:
        if window_ui is not None:
            window_ui.set_mode(mode)
    except Exception:
        pass

def update_window_status(msg, level="info"):
    report_to_window(msg, level)
    try:
        if window_ui is not None:
            window_ui.status_var.set(msg)
    except Exception:
        pass

def update_window_progress(pct, text=None):
    try:
        if window_ui is not None:
            window_ui.set_progress(pct, text)
    except Exception:
        pass

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
file_read_lock = threading.Lock()
file_write_lock = threading.Lock()
log_lock = threading.Lock()
lmdb_lock = threading.Lock()
meta_lock = threading.Lock()
lmdb_env = None
lmdb_counter = 0
lmdb_start = 0
current_map_size_gb = 10
dxcam_camera = None

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
    except Exception as e:
        print(f"Cleanup warning during sleep prep: {e}")
    for _ in range(3):
        gc.collect()
        try:
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
        except Exception as ce:
            print(f"Sleep cleanup CUDA warning: {ce}")
        time.sleep(0.05)

def force_memory_cleanup(iterations=2, delay=0.05):
    for _ in range(iterations):
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"CUDA cleanup step skipped: {e}")
        time.sleep(delay)

data_queue = queue.Queue()
save_queue = queue.Queue()

def append_binary_log(img_arr, act_arr):
    try:
        buf = io.BytesIO()
        np.savez_compressed(buf, image=img_arr, action=act_arr)
        payload = buf.getvalue()
        header = struct.pack("<Q", len(payload))
        sample_count = 0
        try:
            sample_count = min(len(img_arr), len(act_arr))
        except Exception:
            sample_count = 0
        load_index_entries(upgrade=True)
        with log_lock:
            with open(log_path, "ab") as f:
                offset = f.tell()
                f.write(header)
                f.write(payload)
            with open(index_path, "ab") as idx:
                idx.write(struct.pack("<QQQ", offset, len(payload), int(sample_count)))
    except Exception as e:
        print(f"Binary log append error: {e}")

def iterate_binary_log(path):
    try:
        if not os.path.exists(path):
            return []
        entries = load_index_entries()
        if not entries:
            return []
        with log_lock:
            with open(path, "rb") as f:
                for offset, length, _ in entries:
                    f.seek(offset)
                    header = f.read(8)
                    if not header or len(header) < 8:
                        break
                    payload_len = struct.unpack("<Q", header)[0]
                    if payload_len <= 0:
                        continue
                    payload = f.read(payload_len)
                    if len(payload) < payload_len:
                        break
                    yield payload
    except Exception as e:
        print(f"Binary log read error: {e}")
        return []

def load_index_entries(upgrade=False):
    try:
        if not os.path.exists(index_path):
            return []
        with log_lock:
            with open(index_path, "rb") as idx:
                data = idx.read()
        if not data:
            return []
        entry_size = 24 if len(data) % 24 == 0 else (16 if len(data) % 16 == 0 else 8)
        entries = []
        if entry_size == 24:
            for i in range(0, len(data), 24):
                offset, length, count = struct.unpack("<QQQ", data[i:i+24])
                entries.append((offset, length, count))
        elif entry_size == 16:
            for i in range(0, len(data), 16):
                offset, length = struct.unpack("<QQ", data[i:i+16])
                entries.append((offset, length, 0))
        else:
            offset_val = 0
            for i in range(0, len(data), 8):
                length = struct.unpack("<Q", data[i:i+8])[0]
                entries.append((offset_val, length, 0))
                offset_val += length + 8
        if upgrade and entry_size != 24:
            try:
                with log_lock:
                    with open(index_path, "wb") as idx:
                        for off, ln, cnt in entries:
                            idx.write(struct.pack("<QQQ", off, ln, cnt))
            except Exception as e:
                log_exception("Index upgrade failure", e)
        return entries
    except Exception as e:
        log_exception("Index load failure", e)
        return []

def read_log_payload(offset, length):
    try:
        with log_lock:
            with open(log_path, "rb") as f:
                f.seek(offset)
                header = f.read(8)
                if not header or len(header) < 8:
                    return None
                payload_len = struct.unpack("<Q", header)[0]
                if payload_len <= 0:
                    return None
                data_len = payload_len if length <= 0 else min(payload_len, length)
                payload = f.read(data_len)
                if len(payload) < data_len:
                    return None
                return payload
    except Exception as e:
        log_exception("Log payload read error", e)
        return None

def get_lmdb_env(map_size_gb=None):
    global lmdb_env, lmdb_counter, lmdb_start, current_map_size_gb
    if map_size_gb is not None:
        current_map_size_gb = map_size_gb
    if lmdb_env is None:
        size_bytes = int(current_map_size_gb * (1024 ** 3))
        lmdb_env = lmdb.open(lmdb_path, map_size=size_bytes, subdir=False, max_dbs=1, lock=True)
        with lmdb_env.begin(write=True) as txn:
            counter_bytes = txn.get(b"__counter__")
            start_bytes = txn.get(b"__start__")
            lmdb_counter = int.from_bytes(counter_bytes, "little") if counter_bytes else 0
            lmdb_start = int.from_bytes(start_bytes, "little") if start_bytes else 0
            txn.put(b"__counter__", lmdb_counter.to_bytes(8, "little"))
            txn.put(b"__start__", lmdb_start.to_bytes(8, "little"))
    return lmdb_env

def recreate_lmdb_env(new_size_gb):
    global lmdb_env
    try:
        if lmdb_env is not None:
            lmdb_env.close()
    except Exception as e:
        print(f"LMDB close warning before resize: {e}")
    lmdb_env = None
    return get_lmdb_env(new_size_gb)

def reset_lmdb_env():
    global lmdb_env
    with lmdb_lock:
        try:
            if lmdb_env is not None:
                lmdb_env.close()
        except Exception as e:
            print(f"LMDB reset warning: {e}")
        lmdb_env = None
    gc.collect()

def load_meta_entries():
    try:
        if not os.path.exists(meta_path):
            return []
        with meta_lock:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Meta read error: {e}")
        return []

def append_meta_entry(key_val, length, source_type, action_score):
    try:
        entry = {"key": int(key_val), "length": int(length), "timestamp": time.time(), "type": source_type, "action_score": float(action_score)}
        with meta_lock:
            data = load_meta_entries()
            data.append(entry)
            if len(data) > 20000:
                data = data[-20000:]
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
    except Exception as e:
        print(f"Meta append error: {e}")

def compute_action_score(actions):
    try:
        arr = np.asarray(actions, dtype=np.float32)
    except Exception:
        return 0.0
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    scores = []
    for row in arr:
        length = len(row)
        def val(idx):
            return float(row[idx]) if idx < length else 0.0
        delta_x = val(16)
        delta_y = val(17)
        movement = min(1.0, math.sqrt(delta_x * delta_x + delta_y * delta_y) / max(screen_w, screen_h))
        click = 1.0 if val(3) > 0.1 or val(4) > 0.1 else 0.0
        scroll_mag = min(1.0, abs(val(5)) / 120.0)
        hold_trace = 1.0 if any(abs(val(i)) > 0.0 for i in (7, 8, 10, 11, 13, 14, 18, 19)) else 0.0
        salience = 0.45 * movement + 0.35 * click + 0.15 * scroll_mag + 0.05 * hold_trace
        if salience < 0.02:
            salience = 0.0
        scores.append(salience)
    if not scores:
        return 0.0
    return float(np.mean(scores))

def append_lmdb_records(img_arr, act_arr, source_type="human"):
    global lmdb_counter, current_map_size_gb
    if len(img_arr) != len(act_arr):
        print("LMDB append skipped due to length mismatch")
        return
    if len(img_arr) == 0:
        return
    try:
        buf = io.BytesIO()
        np.savez_compressed(buf, image=img_arr, action=act_arr)
        payload = buf.getvalue()
    except Exception as e:
        print(f"Chunk pack error: {e}")
        return
    retry = 0
    while retry < 3:
        try:
            env = get_lmdb_env()
            with lmdb_lock:
                with env.begin(write=True) as txn:
                    key = lmdb_counter.to_bytes(8, "little")
                    txn.put(key, payload)
                    lmdb_counter += 1
                    txn.put(b"__counter__", lmdb_counter.to_bytes(8, "little"))
            score = compute_action_score(act_arr)
            append_meta_entry(lmdb_counter - 1, len(img_arr), source_type, score)
            return
        except lmdb.MapFullError:
            with lmdb_lock:
                current_map_size_gb = int(max(current_map_size_gb * 1.5, current_map_size_gb + 1))
                recreate_lmdb_env(current_map_size_gb)
            print(f"LMDB resized to {current_map_size_gb}GB")
            retry += 1
        except Exception as e:
            print(f"LMDB append error: {e}")
            return

def get_lmdb_length():
    try:
        env = get_lmdb_env()
        with env.begin() as txn:
            counter_bytes = txn.get(b"__counter__")
            start_bytes = txn.get(b"__start__")
            end_val = int.from_bytes(counter_bytes, "little") if counter_bytes else 0
            start_val = int.from_bytes(start_bytes, "little") if start_bytes else 0
            return max(0, end_val - start_val)
    except Exception as e:
        print(f"LMDB length error: {e}")
        return 0

def iterate_lmdb_entries(keys=None, return_keys=False):
    try:
        env = get_lmdb_env()
        if keys is None:
            with env.begin() as txn:
                start_bytes = txn.get(b"__start__")
                start_val = int.from_bytes(start_bytes, "little") if start_bytes else 0
            with env.begin() as txn:
                cursor = txn.cursor()
                seek_key = start_val.to_bytes(8, "little")
                if cursor.set_key(seek_key):
                    pass
                else:
                    cursor.first()
                for key, val in cursor:
                    if key in (b"__counter__", b"__start__"):
                        continue
                    yield (int.from_bytes(key, "little"), val) if return_keys else val
        else:
            with env.begin() as txn:
                for k in keys:
                    key_bytes = int(k).to_bytes(8, "little")
                    val = txn.get(key_bytes)
                    if val is not None:
                        yield (int(k), val) if return_keys else val
    except Exception as e:
        log_exception("LMDB iterate error", e)

def trim_lmdb(limit_bytes):
    try:
        if not os.path.exists(lmdb_path):
            return
        size = os.path.getsize(lmdb_path)
        if size <= limit_bytes:
            return
        env = get_lmdb_env()
        with lmdb_lock:
            with env.begin(write=True) as txn:
                start_bytes = txn.get(b"__start__")
                counter_bytes = txn.get(b"__counter__")
                start_val = int.from_bytes(start_bytes, "little") if start_bytes else 0
                end_val = int.from_bytes(counter_bytes, "little") if counter_bytes else 0
                keep = int(limit_bytes * 0.8)
                target_start = start_val
                while target_start < end_val and os.path.getsize(lmdb_path) > keep:
                    key = target_start.to_bytes(8, "little")
                    txn.delete(key)
                    target_start += 1
                txn.put(b"__start__", target_start.to_bytes(8, "little"))
                global lmdb_start
                lmdb_start = target_start
    except Exception as e:
        print(f"LMDB trim error: {e}")

def trim_binary_log(limit_bytes):
    try:
        if not os.path.exists(log_path):
            return
        entries = load_index_entries()
        if not entries:
            return
        total_bytes = sum(l + 8 for _, l, _ in entries)
        if total_bytes <= limit_bytes:
            return
        keep_limit = int(limit_bytes * 0.8)
        keep_entries = []
        size_acc = 0
        for offset, length, count in reversed(entries):
            if size_acc + length + 8 > keep_limit and keep_entries:
                break
            keep_entries.append((offset, length, count))
            size_acc += length + 8
        keep_entries = list(reversed(keep_entries))
        if not keep_entries:
            return
        start_offset = keep_entries[0][0]
        with open(log_path, "rb") as f:
            f.seek(start_offset)
            remaining = f.read()
        with log_lock:
            with open(log_path, "wb") as f:
                f.write(remaining)
            with open(index_path, "wb") as idx_f:
                base_offset = 0
                for _, length, count in keep_entries:
                    idx_f.write(struct.pack("<QQQ", base_offset, length, count))
                    base_offset += length + 8
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
        self.memory_unit = nn.GRU(self.model_dim, self.model_dim, batch_first=True)
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
                x = checkpoint(lambda t: block(t, mask), x, use_reentrant=False)
            else:
                x = block(x, mask)
        mem_out, hidden_out = self.memory_unit(x, hidden)
        action_token = mem_out[:, -1, :]
        grid_logits = self.action_head(action_token)
        button_logits = self.button_head(action_token)
        pred_features = self.feature_decoder(action_token)
        return grid_logits, pred_features, button_logits, hidden_out

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
                except Exception as e:
                    print(f"NVML shutdown warning: {e}")
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
        global dxcam_camera
        camera = dxcam_camera
        sct = None
        use_dx = os.name == "nt" and dxcam is not None
        while True:
            if capture_pause_event.is_set():
                if camera is not None:
                    try:
                        camera.stop()
                    except Exception as e:
                        print(f"Camera stop warning: {e}")
                    camera = dxcam_camera
                if sct is not None:
                    try:
                        sct.close()
                    except Exception as e:
                        print(f"Screen capture close warning: {e}")
                    sct = None
                time.sleep(0.1)
                continue
            if use_dx and camera is None:
                try:
                    dxcam_camera = dxcam_camera if dxcam_camera is not None else dxcam.create(output_idx=0)
                    camera = dxcam_camera
                    camera.start(target_fps=max(30, capture_freq * 2), video_mode=True, dup=False)
                except Exception:
                    camera = None
                    dxcam_camera = None
                    use_dx = False
            if use_dx and camera is not None:
                start = time.time()
                frame = camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                step_x = max(1, screen_w // target_w)
                step_y = max(1, screen_h // target_h)
                img_small = frame[::step_y, ::step_x]
                if img_small.shape[1] > target_w:
                    img_small = img_small[:, :target_w]
                if img_small.shape[0] > target_h:
                    img_small = img_small[:target_h, :]
                update_latest_frame(img_small, start)
                desired = max(30, capture_freq * 2)
                wait = (1.0 / desired) - (time.time() - start)
                if wait > 0:
                    time.sleep(wait)
                continue
            if sct is None:
                sct = mss.mss()
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

def release_mouse_outputs():
    try:
        ctrl = mouse.Controller()
        ctrl.release(mouse.Button.left)
        ctrl.release(mouse.Button.right)
    except Exception as e:
        log_exception("Mouse release failure", e)

def handle_training_escape(msg):
    global current_mode, stop_training_flag
    stop_training_flag = True
    release_mouse_outputs()
    flush_buffers()
    current_mode = MODE_LEARNING
    update_window_mode(current_mode)
    update_window_status(msg, "warn")
    input_allowed_event.set()

def on_press_key(key):
    global current_mode, stop_training_flag, user_stop_request_reason
    try:
        if key == keyboard.Key.esc:
            if current_mode == MODE_TRAINING:
                handle_training_escape("检测到ESC，暂停输出并返回学习模式")
    except Exception as e:
        log_exception("Key listener error", e)

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

def release_data_handle(data):
    try:
        if hasattr(data, "close"):
            data.close()
    except Exception as e:
        print(f"Data close warning: {e}")
    try:
        mmap_obj = getattr(data, "_mmap", None)
        if mmap_obj:
            mmap_obj.close()
    except Exception as e:
        print(f"mmap close warning: {e}")

def check_disk_space():
    try:
        total_size = 0
        if os.path.exists(lmdb_path):
            try:
                total_size += os.path.getsize(lmdb_path)
            except Exception as e:
                print(f"Disk size check warning for LMDB: {e}")
        limit = 20 * 1024 * 1024 * 1024
        if total_size > limit:
            trim_lmdb(limit)
    except Exception as e:
        print(f"Disk clean error: {e}")

def consolidate_data_files(max_frames=10000, min_commit=4000):
    try:
        trim_lmdb(20 * 1024 * 1024 * 1024)
    except Exception as e:
        print(f"Merge error: {e}")

def atomic_save_npz(fname, img_arr, act_arr, retries=5, backoff=0.25):
    last_error = None
    for attempt in range(retries):
        temp_path = None
        try:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            fd, temp_path = tempfile.mkstemp(prefix="tmp_", suffix=".npz", dir=temp_dir)
            os.close(fd)
            np.savez(temp_path, image=img_arr, action=act_arr)
            with file_write_lock:
                os.replace(temp_path, fname)
            return True
        except Exception as e:
            last_error = e
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as clean_err:
                    print(f"Temp cleanup warning for {temp_path}: {clean_err}")
            time.sleep(backoff * (attempt + 1))
    if last_error is not None:
        print(f"File save error: {last_error}")
    return False

def disk_writer_loop():
    check_counter = 0
    last_check = time.time()
    check_interval = 10
    check_time_window = 60.0
    while True:
        try:
            payload = save_queue.get()
            if payload is None or len(payload) < 2:
                continue
            img_arr, act_arr = payload[0], payload[1]
            source_type = payload[2] if len(payload) > 2 else "human"
            if len(img_arr) != len(act_arr) or len(img_arr) == 0:
                print("LMDB pipeline skipped invalid batch")
                continue
            try:
                append_lmdb_records(img_arr, act_arr, source_type)
            except Exception as e:
                print(f"LMDB pipeline error: {e}")
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

def unwrap_array(obj):
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        try:
            return obj.item()
        except Exception:
            return obj
    return obj

def prepare_saved_arrays(buffer_images, buffer_actions):
    imgs = []
    for img in buffer_images:
        arr = np.asarray(unwrap_array(img), dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[0] != target_h or arr.shape[1] != target_w:
            arr = cv2.resize(arr, (target_w, target_h))
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        imgs.append(arr)
    act_len = max((len(np.asarray(unwrap_array(a)).flatten()) for a in buffer_actions), default=0)
    acts = []
    for a in buffer_actions:
        flat = np.asarray(unwrap_array(a), dtype=np.float32).flatten()
        if len(flat) < act_len:
            pad = np.zeros(act_len - len(flat), dtype=np.float32)
            flat = np.concatenate([flat, pad])
        acts.append(flat)
    img_arr = np.stack(imgs, axis=0) if imgs else np.empty((0, target_h, target_w, 3), dtype=np.uint8)
    act_arr = np.stack(acts, axis=0) if acts else np.empty((0, act_len), dtype=np.float32)
    return img_arr, act_arr

def safe_action_array(entry):
    try:
        value = None
        if hasattr(entry, "dtype") and getattr(entry.dtype, "names", None) and "action" in entry.dtype.names:
            value = entry["action"]
        elif isinstance(entry, dict) and "action" in entry:
            value = entry["action"]
        elif isinstance(entry, (tuple, list)):
            if len(entry) >= 2:
                value = entry[1]
        else:
            value = entry
        if isinstance(value, np.ndarray) and value.ndim == 0:
            try:
                value = value.item()
            except Exception as e:
                print(f"Scalar extraction warning: {e}")
        if value is None:
            base = np.zeros(0, dtype=np.float32)
        else:
            base = np.asarray(value, dtype=np.float32).reshape(-1)
        min_len = mouse_feature_dim + 1
        if base.size < min_len:
            pad = np.zeros(min_len - base.size, dtype=np.float32)
            base = np.concatenate([base, pad], axis=0)
        return base
    except Exception:
        return np.zeros(mouse_feature_dim + 1, dtype=np.float32)


def safe_image_data(entry, structured=False):
    try:
        if isinstance(entry, np.ndarray) and entry.ndim == 3:
            return entry
        if structured:
            if hasattr(entry, "dtype") and getattr(entry.dtype, "names", None) and "image" in entry.dtype.names:
                value = entry["image"]
                if isinstance(value, np.ndarray) and value.ndim == 0:
                    value = value.item()
                return value
            if isinstance(entry, (tuple, list)) and len(entry) >= 1:
                return entry[0]
        if isinstance(entry, dict):
            img_data = entry.get("screen")
            if isinstance(img_data, bytes):
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            else:
                img = img_data
            if img is not None:
                if isinstance(img, np.ndarray) and img.ndim == 3:
                    return img
    except Exception as e:
        print(f"Image decode warning: {e}")
    return np.zeros((target_h, target_w, 3), dtype=np.uint8)

def build_entry_pair(data, idx, structured_npz, structured):
    try:
        if structured_npz:
            images = data.get('image') if isinstance(data, dict) else data['image']
            actions = data.get('action') if isinstance(data, dict) else data['action']
            if isinstance(images, np.ndarray) and images.ndim == 0:
                images = images.item()
            if isinstance(actions, np.ndarray) and actions.ndim == 0:
                actions = actions.item()
            if images is None or actions is None:
                return None
            try:
                pair_len = min(len(images), len(actions))
            except Exception:
                return None
            if pair_len <= 0:
                return None
            if idx < pair_len:
                return (images[idx], actions[idx])
            return None
        if structured:
            if idx < len(data):
                return data[idx]
            return None
        if isinstance(data, (list, tuple)) and len(data) == 2 and hasattr(data[0], "__len__") and hasattr(data[1], "__len__"):
            pair_len = min(len(data[0]), len(data[1]))
            if idx < pair_len:
                return (data[0][idx], data[1][idx])
            return None
        if idx < len(data):
            return data[idx]
    except Exception:
        return None
    return None

def dataset_length(data, structured, structured_npz):
    try:
        if isinstance(data, np.ndarray) and data.ndim == 0:
            try:
                data = data.item()
            except Exception:
                return 0
        if structured:
            if 'action' not in data or 'image' not in data:
                return 0
            actions = data['action']
            images = data['image']
            if isinstance(actions, np.ndarray) and actions.ndim == 0:
                actions = actions.item()
            if isinstance(images, np.ndarray) and images.ndim == 0:
                images = images.item()
            try:
                act_len = len(actions)
                img_len = len(images)
            except Exception:
                return 0
            return min(act_len, img_len)
        if isinstance(data, (list, tuple)) and len(data) == 2 and hasattr(data[0], "__len__") and hasattr(data[1], "__len__"):
            return min(len(data[0]), len(data[1]))
        return len(data)
    except Exception:
        return 0

class StreamingGameDataset(IterableDataset):
    def __init__(self, file_list, lmdb_keys=None):
        self.file_list = list(file_list)
        self.seq_len = seq_len
        self.lmdb_keys = lmdb_keys or {}

    def _prepare_item(self, slice_data, next_entry, structured, structured_npz):
        slice_len = len(slice_data)
        imgs = np.empty((slice_len, 3, target_h, target_w), dtype=np.uint8)
        m_ins = np.empty((slice_len, mouse_feature_dim), dtype=np.float32)

        def norm_coord(v, dim):
            return (2.0 * (v / dim)) - 1.0

        def load_image(item):
            return safe_image_data(item, structured)

        def normalize_image(img):
            arr = None
            try:
                arr = np.asarray(img) if img is not None else None
                if arr is None or arr.size == 0:
                    return np.zeros((3, target_h, target_w), dtype=np.uint8)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                elif arr.ndim == 3 and arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                elif arr.ndim < 3:
                    arr = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                if arr.ndim == 3 and arr.shape[2] > 3:
                    arr = arr[:, :, :3]
                if arr.shape[0] != target_h or arr.shape[1] != target_w:
                    arr = cv2.resize(arr, (target_w, target_h))
                if arr.ndim == 3 and arr.shape[2] == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
                if arr.ndim != 3 or arr.shape[2] < 3:
                    arr = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            except Exception as e:
                log_exception("Image normalize error", e)
                arr = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            return arr.transpose(2, 0, 1).astype(np.uint8, copy=False)

        for idx, item in enumerate(slice_data):
            if structured:
                raw_action = safe_action_array(item)
                action = np.atleast_1d(raw_action)
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
            imgs[idx] = img

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
            m_ins[idx] = np.asarray(m_vec, dtype=np.float32)

        if next_entry is not None:
            if structured:
                next_action = np.atleast_1d(safe_action_array(next_entry))
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
            next_img = np.zeros((3, target_h, target_w), dtype=np.uint8)

        return torch.from_numpy(imgs).float().div(255.0), torch.from_numpy(m_ins), torch.tensor(labels, dtype=torch.float32), torch.from_numpy(np.asarray(next_img, dtype=np.uint8)).float().div(255.0)

    def __iter__(self):
        queue_size = 2 if low_vram_mode or os.name == "nt" else 8
        output_queue = queue.Queue(maxsize=queue_size)
        stop_event = threading.Event()

        def should_stop():
            if stop_optimization_flag.is_set():
                gc.collect()
                return True
            return False

        def producer():
            try:
                for path in self.file_list:
                    if should_stop():
                        break
                    try:
                        if path == log_path:
                            for payload in iterate_binary_log(path):
                                if should_stop():
                                    break
                                try:
                                    buf = io.BytesIO(payload)
                                    data = np.load(buf, allow_pickle=True)
                                    data = unwrap_array(data)
                                    structured_npz = isinstance(data, np.lib.npyio.NpzFile)
                                    structured_array = hasattr(data, 'dtype') and data.dtype.names and 'action' in data.dtype.names
                                    structured = structured_npz or structured_array
                                    length = dataset_length(data, structured, structured_npz)
                                    if length <= 0:
                                        if structured_npz:
                                            data.close()
                                        continue
                                    max_start = max(1, length - self.seq_len)
                                    for start_idx in range(max_start):
                                        if should_stop():
                                            break
                                        slice_data = []
                                        for i in range(self.seq_len):
                                            idx_in_file = min(start_idx + i, length - 1)
                                            entry = build_entry_pair(data, idx_in_file, structured_npz, structured)
                                            if entry is None:
                                                slice_data = []
                                                break
                                            slice_data.append(entry)
                                        next_entry = None
                                        next_idx = start_idx + self.seq_len
                                        if next_idx < length:
                                            next_entry = build_entry_pair(data, next_idx, structured_npz, structured)
                                        if not slice_data:
                                            continue
                                        item = self._prepare_item(slice_data, next_entry, structured, structured_npz)
                                        output_queue.put(item)
                                    release_data_handle(data)
                                except Exception as e:
                                    log_exception("Binary data load error", e, f"path={path}")
                            continue
                        if path == lmdb_path:
                            try:
                                keys = self.lmdb_keys.get(lmdb_path)
                                for meta_key, payload in iterate_lmdb_entries(keys, return_keys=True):
                                    if should_stop():
                                        break
                                    try:
                                        buf = io.BytesIO(payload)
                                        data = np.load(buf, allow_pickle=True)
                                        data = unwrap_array(data)
                                        structured_npz = isinstance(data, np.lib.npyio.NpzFile)
                                        structured_array = hasattr(data, 'dtype') and data.dtype.names and 'action' in data.dtype.names
                                        structured = structured_npz or structured_array
                                        if structured_npz and (('image' not in data) or ('action' not in data)):
                                            release_data_handle(data)
                                            continue
                                        length = dataset_length(data, structured, structured_npz)
                                        if length <= 0:
                                            if structured_npz:
                                                data.close()
                                            continue
                                        max_start = max(1, length - self.seq_len)
                                        for start_idx in range(max_start):
                                            if should_stop():
                                                break
                                            slice_data = []
                                            for i in range(self.seq_len):
                                                idx_in_file = min(start_idx + i, length - 1)
                                                entry = build_entry_pair(data, idx_in_file, structured_npz, structured)
                                                if entry is None:
                                                    slice_data = []
                                                    break
                                                slice_data.append(entry)
                                            next_entry = None
                                            next_idx = start_idx + self.seq_len
                                            if next_idx < length:
                                                next_entry = build_entry_pair(data, next_idx, structured_npz, structured)
                                            if not slice_data:
                                                continue
                                            item = self._prepare_item(slice_data, next_entry, structured, structured_npz)
                                            output_queue.put(item)
                                        release_data_handle(data)
                                    except Exception as e:
                                        log_exception("LMDB data load error", e, f"key={meta_key}")
                            except Exception as e:
                                log_exception("LMDB data iteration failure", e)
                            continue
                        with file_read_lock:
                            data = np.load(path, allow_pickle=True, mmap_mode="r")
                        data = unwrap_array(data)
                        structured_npz = isinstance(data, np.lib.npyio.NpzFile)
                        structured_array = hasattr(data, 'dtype') and data.dtype.names and 'action' in data.dtype.names
                        structured = structured_npz or structured_array
                        length = dataset_length(data, structured, structured_npz)
                        if length <= 0:
                            if structured_npz:
                                data.close()
                            continue
                        max_start = max(1, length - self.seq_len)
                        for start_idx in range(max_start):
                            if should_stop():
                                break
                            slice_data = []
                            for i in range(self.seq_len):
                                idx_in_file = min(start_idx + i, length - 1)
                                entry = build_entry_pair(data, idx_in_file, structured_npz, structured)
                                if entry is None:
                                    slice_data = []
                                    break
                                slice_data.append(entry)
                            next_entry = None
                            next_idx = start_idx + self.seq_len
                            if next_idx < length:
                                next_entry = build_entry_pair(data, next_idx, structured_npz, structured)
                            if not slice_data:
                                continue
                            item = self._prepare_item(slice_data, next_entry, structured, structured_npz)
                            output_queue.put(item)
                        release_data_handle(data)
                    except Exception as e:
                        log_exception("Data load error", e, f"path={path}")
            except Exception as e:
                log_exception("Producer failure", e)
            finally:
                stop_event.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()
        idle_start = time.time()
        try:
            while True:
                try:
                    item = output_queue.get(timeout=0.5)
                    idle_start = time.time()
                    yield item
                except queue.Empty:
                    producer_alive = producer_thread.is_alive()
                    if stop_event.is_set() and output_queue.empty():
                        break
                    if not producer_alive and output_queue.empty():
                        break
                    if not producer_alive and time.time() - idle_start > 2.0:
                        break
                    continue
        finally:
            stop_event.set()
            producer_thread.join(timeout=1.0)
            try:
                while True:
                    output_queue.get_nowait()
            except queue.Empty:
                pass
            gc.collect()

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

def sample_actions_from_source(path, sample_cap=8):
    samples = []
    total_len = 0
    try:
        if path == lmdb_path:
            meta_entries = load_meta_entries()
            key_subset = [m.get("key") for m in sorted(meta_entries, key=lambda x: x.get("timestamp", 0), reverse=True)[:sample_cap] if "key" in m]
            for meta_key, payload in iterate_lmdb_entries(key_subset, return_keys=True):
                try:
                    buf = io.BytesIO(payload)
                    data = np.load(buf, allow_pickle=True)
                    data = unwrap_array(data)
                    structured_npz = isinstance(data, np.lib.npyio.NpzFile)
                    structured_array = hasattr(data, 'dtype') and getattr(data.dtype, "names", None) and 'action' in data.dtype.names
                    structured = structured_npz or structured_array
                    if structured_npz and (('image' not in data) or ('action' not in data)):
                        release_data_handle(data)
                        continue
                    length = dataset_length(data, structured, structured_npz)
                    total_len += max(0, int(length))
                    step = max(1, int(max(1, length) // max(1, sample_cap))) if length > 0 else 1
                    idx = 0
                    while idx < length and len(samples) < sample_cap:
                        entry = build_entry_pair(data, idx, structured_npz, structured)
                        if entry is None:
                            break
                        samples.append(safe_action_array(entry))
                        idx += step
                    release_data_handle(data)
                except Exception as e:
                    log_exception("LMDB sample error", e, f"key={meta_key}")
        elif path == log_path:
            entries = load_index_entries()
            if not entries:
                return 0, []
            known_total = sum(int(c) for _, _, c in entries if c > 0)
            unknown_remaining = len([1 for _, _, c in entries if c <= 0])
            unknown_indices = [i for i, e in enumerate(entries) if e[2] <= 0]
            pool = unknown_indices if unknown_indices else list(range(len(entries)))
            pick = min(len(pool), sample_cap)
            if pick <= 0:
                return known_total, []
            indices = [pool[i] for i in np.linspace(0, len(pool) - 1, pick, dtype=int)]
            measured = []
            for i in indices:
                offset, length_bytes, count = entries[i]
                payload = read_log_payload(offset, length_bytes)
                if payload is None:
                    continue
                buf = io.BytesIO(payload)
                data = np.load(buf, allow_pickle=True)
                data = unwrap_array(data)
                structured_npz = isinstance(data, np.lib.npyio.NpzFile)
                structured_array = hasattr(data, 'dtype') and getattr(data.dtype, "names", None) and 'action' in data.dtype.names
                structured = structured_npz or structured_array
                length = dataset_length(data, structured, structured_npz)
                if count <= 0:
                    total_len += max(0, int(length))
                    unknown_remaining = max(0, unknown_remaining - 1)
                    measured.append(max(0, int(length)))
                step = max(1, max(1, int(length // sample_cap))) if length > 0 else 1
                idx = 0
                while idx < length and len(samples) < sample_cap:
                    entry = (data['image'][idx], data['action'][idx]) if structured_npz else data[idx]
                    samples.append(safe_action_array(entry))
                    idx += step
                release_data_handle(data)
            total_len += known_total
            if unknown_remaining > 0 and measured:
                avg_len = float(sum(measured)) / float(len(measured))
                total_len += int(avg_len * unknown_remaining)
        else:
            with file_read_lock:
                data = np.load(path, allow_pickle=True, mmap_mode="r")
            data = unwrap_array(data)
            structured_npz = isinstance(data, np.lib.npyio.NpzFile)
            structured_array = hasattr(data, 'dtype') and getattr(data.dtype, "names", None) and 'action' in data.dtype.names
            structured = structured_npz or structured_array
            length = dataset_length(data, structured, structured_npz)
            total_len = max(0, int(length))
            step = max(1, max(1, int(length // sample_cap))) if length > 0 else 1
            idx = 0
            while idx < length and len(samples) < sample_cap:
                entry = (data['image'][idx], data['action'][idx]) if structured_npz else data[idx]
                samples.append(safe_action_array(entry))
                idx += step
            release_data_handle(data)
    except Exception as e:
        log_exception("Data probe error", e, f"path={path}")
    return total_len, samples


def build_sleep_file_mix(candidates, limit=20):
    stats = []
    lmdb_selection = {}
    for path in candidates:
        if path == lmdb_path:
            meta_entries = load_meta_entries()
            if not meta_entries:
                continue
            for m in meta_entries:
                length = int(m.get("length", 0))
                if length <= 0:
                    continue
                ts = float(m.get("timestamp", 0))
                human_score = 1.0 if m.get("type") == "human" else 0.3
                surprise_score = float(m.get("action_score", 0.0))
                steps_est = max(1, length - seq_len)
                stats.append((path, human_score, surprise_score, length, ts, steps_est, m.get("key")))
        else:
            length, samples = sample_actions_from_source(path)
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                mtime = 0
            if length <= 0 and not samples:
                continue
            human_score = 0.0
            surprise_score = 0.0
            if samples:
                flat_samples = []
                for a in samples:
                    try:
                        v = np.asarray(a, dtype=np.float32).reshape(-1)
                    except Exception:
                        v = safe_action_array(a)
                    flat_samples.append(v)
                human_vals = []
                surprise_vals = []
                for v in flat_samples:
                    size = int(getattr(v, "size", 0))
                    if size <= 0:
                        continue
                    if size > mouse_feature_dim:
                        human_vals.append(1.0 if v[mouse_feature_dim] < 0.5 else 0.0)
                    else:
                        human_vals.append(0.5)
                    if size > 18:
                        surprise_vals.append(float(np.mean(np.abs(v[16:18]))))
                    else:
                        surprise_vals.append(0.0)
                if human_vals:
                    human_score = float(np.mean(human_vals))
                if surprise_vals:
                    surprise_score = float(np.mean(surprise_vals))
            steps_est = max(1, int(length) - seq_len)
            stats.append((path, human_score, surprise_score, int(length), float(mtime), steps_est, None))
    if not stats:
        return [], [], {}
    if not stats:
        return [], [], {}
    now_ts = time.time()
    weights = []
    for path, human_score, surprise_score, length, ts, steps_est, key in stats:
        recency_bias = math.exp(-max(0.0, now_ts - ts) / (2 * 24 * 3600)) if ts > 0 else 0.05
        history_bonus = 0.08 if ts > 0 and (now_ts - ts) > (14 * 24 * 3600) else 0.0
        density = math.log1p(max(1.0, steps_est)) / math.log(2.0)
        weight = 0.55 * recency_bias + 0.2 * human_score + 0.2 * (1.0 + surprise_score) + 0.05 * density + history_bonus
        weights.append(max(weight, 0.01))
    sample_count = min(limit if limit else len(stats), len(stats))
    picks = random.choices(range(len(stats)), weights=weights, k=sample_count)
    selection = []
    selected_paths = set()
    oldest_idx = min(range(len(stats)), key=lambda i: stats[i][4])
    if oldest_idx not in picks and len(stats) > 1:
        picks[-1] = oldest_idx
    for idx in picks:
        path, _, _, _, ts, steps_est, key = stats[idx]
        if path == lmdb_path:
            lmdb_selection.setdefault(path, []).append(key)
        if path not in selected_paths:
            selection.append(path)
            selected_paths.add(path)
    return selection, stats, lmdb_selection


def update_meta_with_loss(key_plan, paths, loss_value):
    try:
        if loss_value <= 0:
            return
        targets = set()
        for p in paths:
            if p == lmdb_path:
                keys = key_plan.get(p)
                if keys:
                    targets.update(keys)
        if not targets:
            return
        with meta_lock:
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []
            changed = False
            now_ts = time.time()
            for entry in data:
                if entry.get("key") in targets:
                    old_score = float(entry.get("action_score", 0.0))
                    entry["action_score"] = 0.7 * old_score + 0.3 * min(5.0, float(loss_value))
                    entry["timestamp"] = max(float(entry.get("timestamp", now_ts)), now_ts - 0.1)
                    changed = True
            if changed:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
    except Exception as e:
        print(f"Meta loss feedback error: {e}")


def optimize_ai():
    global low_vram_mode, user_stop_request_reason
    try:
        if os.name == "nt":
            reset_lmdb_env()
        time.sleep(1.0)
        stop_optimization_flag.clear()
        user_stop_request_reason = None
        capture_pause_event.set()
        opt_start = time.time()
        force_memory_cleanup(3, 0.1)
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
            except Exception as e:
                print(f"CUDA stats reset warning: {e}")
        print("Starting Optimization...")
        update_window_status("启动睡眠优化...", "info")
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
        accumulation_steps = 8
        train_batch_size = 8
        low_vram_mode = False
        total_mem_gb = None
        if torch.cuda.is_available():
            try:
                total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_mem_gb <= 4:
                    accumulation_steps = 12
                    train_batch_size = 1
                    low_vram_mode = True
                elif total_mem_gb <= 6:
                    accumulation_steps = 10
                    train_batch_size = 3
                elif total_mem_gb > 10:
                    accumulation_steps = 10
                    train_batch_size = 12
            except Exception as e:
                print(f"GPU property probe warning: {e}")
        effective_batch = train_batch_size * accumulation_steps
        interrupted = False
        early_stop_reason = None

        def ensure_stop_reason(reason=None):
            nonlocal early_stop_reason
            if early_stop_reason is None:
                early_stop_reason = reason or user_stop_request_reason or "用户触发提前结束"
                if early_stop_reason:
                    update_window_status(early_stop_reason, "warn")

        def safe_save_model(final_reason=None, last_snapshot=None, steps_snapshot=None, total_snapshot=None, interrupted_flag=False):
            try:
                alpha = float(model.log_var_action.detach().item())
                beta = float(model.log_var_prediction.detach().item())
                gamma = float(model.log_var_energy.detach().item())
                temp_path = os.path.join(model_dir, "ai_model_temp.pth")
                torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "alpha": alpha, "beta": beta, "gamma": gamma, "last_loss": last_snapshot if last_snapshot is not None else 0.0, "steps": steps_snapshot if steps_snapshot is not None else 0, "total_steps": total_snapshot if total_snapshot is not None else 0, "interrupted": interrupted_flag, "timestamp": time.time(), "reason": final_reason or early_stop_reason or user_stop_request_reason}, temp_path)
                if os.path.exists(model_path):
                    try:
                        shutil.copy2(model_path, backup_path)
                    except Exception:
                        pass
                    try:
                        os.remove(model_path)
                    except Exception:
                        pass
                try:
                    os.rename(temp_path, model_path)
                except Exception:
                    shutil.copy2(temp_path, model_path)
                return True
            except Exception as e:
                print(f"Safe save warning: {e}")
                return False

        consolidate_data_files()

        candidates = []
        if os.path.exists(lmdb_path):
            candidates.append(lmdb_path)
        valid_candidates = []
        for c in candidates:
            try:
                if os.path.getsize(c) > 0:
                    valid_candidates.append(c)
            except Exception as e:
                log_exception("Data candidate check failed", e, f"path={c}")
        sleep_stats = []
        lmdb_key_plan = {}
        prioritized, sleep_stats, lmdb_key_plan = build_sleep_file_mix(valid_candidates, limit=25)
        target_files = prioritized if prioritized else valid_candidates
        files = []
        total_scan = len(target_files)
        scanned = 0
        print("Scanning data files...")
        torch.cuda.empty_cache()
        for f in target_files:
            if stop_optimization_flag.is_set():
                ensure_stop_reason()
                break
            scanned += 1
            progress_bar("数据扫描阶段", scanned, total_scan)
            try:
                if os.path.getsize(f) > 0:
                    files.append(f)
            except Exception as e:
                log_exception("Data scan failed", e, f"path={f}")
        if stop_optimization_flag.is_set():
            interrupted = True
            ensure_stop_reason()
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
                if stop_optimization_flag.is_set():
                    ensure_stop_reason()
                    break
                try:
                    if path == log_path:
                        entries = load_index_entries()
                        if not entries:
                            continue
                        known_steps = [max(1, int(c) - seq_len) for _, _, c in entries if c > 0]
                        total += sum(known_steps)
                        unknown_entries = [(i, e) for i, e in enumerate(entries) if e[2] <= 0]
                        if unknown_entries:
                            probe = min(len(unknown_entries), 5)
                            idx_candidates = np.linspace(0, len(unknown_entries) - 1, probe, dtype=int)
                            measured = []
                            remaining = len(unknown_entries)
                            for idx_val in idx_candidates:
                                if stop_optimization_flag.is_set():
                                    ensure_stop_reason()
                                    break
                                _, entry = unknown_entries[idx_val]
                                payload = read_log_payload(entry[0], entry[1])
                                if payload is None:
                                    remaining = max(0, remaining - 1)
                                    continue
                                buf = io.BytesIO(payload)
                                data = np.load(buf, allow_pickle=True)
                                data = unwrap_array(data)
                                structured_npz = isinstance(data, np.lib.npyio.NpzFile)
                                structured_array = hasattr(data, 'dtype') and data.dtype.names and 'action' in data.dtype.names
                                structured = structured_npz or structured_array
                                length = dataset_length(data, structured, structured_npz)
                                release_data_handle(data)
                                steps = max(1, int(length) - seq_len)
                                total += steps
                                measured.append(steps)
                                remaining = max(0, remaining - 1)
                            if remaining > 0 and measured:
                                avg_steps = float(sum(measured)) / float(len(measured))
                                total += int(avg_steps * remaining)
                        continue
                    if path == lmdb_path:
                        keys = lmdb_key_plan.get(path)
                        if keys:
                            for meta_key in keys:
                                try:
                                    meta_entries = [m for m in load_meta_entries() if m.get("key") == meta_key]
                                    if not meta_entries:
                                        continue
                                    length = int(meta_entries[0].get("length", 0))
                                    steps = max(1, length - seq_len)
                                    total += steps
                                except Exception as e:
                                    log_exception("LMDB meta parse failure", e, f"key={meta_key}")
                        else:
                            length = get_lmdb_length()
                            steps = max(1, length - seq_len)
                            total += steps
                        continue
                    with file_read_lock:
                        data = np.load(path, allow_pickle=True, mmap_mode="r")
                    data = unwrap_array(data)
                    structured_npz = isinstance(data, np.lib.npyio.NpzFile)
                    structured_array = hasattr(data, 'dtype') and data.dtype.names and 'action' in data.dtype.names
                    structured = structured_npz or structured_array
                    length = dataset_length(data, structured, structured_npz)
                    steps = max(1, length - seq_len)
                    total += steps
                    release_data_handle(data)
                except Exception as e:
                    log_exception("Step estimation error", e, f"path={path}")
            return total

        chunk_steps = []
        init_total = len(batch_files)
        init_done = 0
        print("Dataset Init")
        torch.cuda.empty_cache()
        for bf in batch_files:
            if stop_optimization_flag.is_set():
                interrupted = True
                ensure_stop_reason()
                break
            steps = estimate_steps(bf)
            chunk_steps.append(steps)
            init_done += 1
            progress_bar("Dataset Init", init_done, init_total, f"Chunks: {steps} steps")
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        total_steps = sum(chunk_steps)
        sleep_ratio = 0.618
        if total_steps > 0:
            target_steps = max(1, int(total_steps * sleep_ratio))
            combined = list(zip(batch_files, chunk_steps))
            random.shuffle(combined)
            sampled_batches = []
            sampled_steps = []
            acc_steps = 0
            for paths, steps in combined:
                if acc_steps >= target_steps:
                    break
                allowed = min(steps, target_steps - acc_steps)
                if allowed <= 0:
                    continue
                sampled_batches.append(paths)
                sampled_steps.append(allowed)
                acc_steps += allowed
            if acc_steps > 0:
                batch_files = sampled_batches
                chunk_steps = sampled_steps
                total_steps = sum(chunk_steps)
                print(f"睡眠采样比例 61.8%: 预计 {total_steps}/{target_steps} steps，批次数 {len(batch_files)}")
        max_sleep_steps = 15000
        if total_mem_gb is not None:
            if total_mem_gb <= 4:
                max_sleep_steps = 6000
            elif total_mem_gb <= 6:
                max_sleep_steps = 12000
            else:
                max_sleep_steps = 20000
        if total_steps > max_sleep_steps:
            trimmed = []
            trimmed_steps = []
            step_acc = 0
            for bf, steps in sorted(zip(batch_files, chunk_steps), key=lambda x: x[1], reverse=True):
                if step_acc >= max_sleep_steps:
                    break
                trimmed.append(bf)
                allowed = min(steps, max_sleep_steps - step_acc)
                trimmed_steps.append(allowed)
                step_acc += steps
            batch_files = trimmed
            chunk_steps = trimmed_steps
            total_steps = sum(chunk_steps)
            print(f"Sleep sampling启用: 预计 {total_steps}/{max_sleep_steps} steps，批次数 {len(batch_files)}")
        if total_steps <= 0 and not interrupted:
            print("No data to train.")
            torch.cuda.empty_cache()
            return

        if total_steps < 5000:
            epochs = 3
        elif total_steps < 20000:
            epochs = 2
        else:
            epochs = 1

        total_samples = total_steps * epochs
        current_step = 0
        total_chunks = len(batch_files)
        last_loss_value = 0.0
        plateau_counter = 0
        prev_loss = float('inf')
        finetune_enabled = False
        plateau_patience = 80
        best_loss = float('inf')
        improve_streak = 0
        early_stop_reason = None
        extend_done = False
        torch.cuda.empty_cache()
        for idx, bf in enumerate(batch_files):
            if stop_optimization_flag.is_set():
                interrupted = True
                ensure_stop_reason()
                break
            chunk_trained = False
            while not chunk_trained and not stop_optimization_flag.is_set():
                chunk_loss_acc = 0.0
                chunk_sample_count = 0
                dataset = StreamingGameDataset(bf, lmdb_key_plan)
                loader_workers = 0 if low_vram_mode or os.name == "nt" else 2
                loader = DataLoader(dataset, batch_size=train_batch_size, drop_last=False, num_workers=loader_workers, pin_memory=True, persistent_workers=loader_workers > 0)
                optimizer.zero_grad()
                attempt_steps = 0
                try:
                    dynamic_epochs = epochs
                    epoch_idx = 0
                    while epoch_idx < dynamic_epochs:
                        pending_steps = 0
                        for batch_idx, (imgs, mins, labels, next_frames) in enumerate(loader):
                            if stop_optimization_flag.is_set():
                                interrupted = True
                                ensure_stop_reason()
                                break
                            imgs = imgs.to(device)
                            mins = mins.to(device)
                            labels = labels.to(device)
                            next_frames = next_frames.to(device)
                            batch_sample = imgs.size(0)

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
                            loss_val = float(total_loss.item())
                            chunk_loss_acc += loss_val * batch_sample
                            chunk_sample_count += batch_sample
                            pending_steps += 1
                            attempt_steps += batch_sample
                            if pending_steps % accumulation_steps == 0:
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                                pending_steps = 0
                            last_loss_value = float(total_loss.item())
                            if abs(last_loss_value - prev_loss) < 1e-4:
                                plateau_counter += 1
                            else:
                                plateau_counter = 0
                            if last_loss_value < best_loss - 1e-4:
                                best_loss = last_loss_value
                                improve_streak += 1
                            else:
                                improve_streak = max(0, improve_streak - 1)
                            prev_loss = last_loss_value
                            current_step += batch_sample
                            if current_step > total_samples:
                                total_samples = current_step
                            if plateau_counter >= 3 and not finetune_enabled:
                                model.enable_backbone_finetune(0.25)
                                finetune_enabled = True
                            if plateau_counter >= plateau_patience and current_step > total_steps * 0.25:
                                early_stop_reason = f"Plateau保持 {plateau_counter} 次，提前结束"
                                stop_optimization_flag.set()
                                break
                            progress_bar("模型训练阶段", current_step, total_samples, f"Loss: {last_loss_value:.4f} | Chunk {idx+1}/{total_chunks}")
                            del imgs, mins, labels, next_frames, grid_logits, pred_feat, button_logits, target_grid, target_buttons, target_feat, total_loss
                            if low_vram_mode or batch_idx % 2 == 1:
                                torch.cuda.empty_cache()
                            if low_vram_mode:
                                gc.collect()
                                if os.name == "nt":
                                    force_memory_cleanup(1, 0.02)
                        if stop_optimization_flag.is_set() or early_stop_reason:
                            interrupted = True
                            ensure_stop_reason(early_stop_reason)
                            break
                        if pending_steps > 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                        if improve_streak >= 50 and not extend_done:
                            dynamic_epochs += 1
                            extra = chunk_steps[idx] if idx < len(chunk_steps) else chunk_steps[-1] if chunk_steps else 0
                            total_samples += extra
                            extend_done = True
                            print(f"延长训练: chunk {idx+1} 增加 1 个 epoch (当前 {dynamic_epochs})")
                        epoch_idx += 1
                        if stop_optimization_flag.is_set():
                            interrupted = True
                            ensure_stop_reason(early_stop_reason)
                            break
                    chunk_trained = True
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        current_step = max(0, current_step - attempt_steps)
                        torch.cuda.empty_cache()
                        gc.collect()
                        if train_batch_size > 1:
                            new_batch = max(1, train_batch_size // 2)
                            train_batch_size = new_batch
                            accumulation_steps = max(accumulation_steps, int(math.ceil(effective_batch / max(1, train_batch_size))))
                            print(f"OOM detected, batch -> {train_batch_size}, accumulation -> {accumulation_steps}")
                            continue
                        interrupted = True
                        stop_optimization_flag.set()
                        ensure_stop_reason("显存不足，提前结束睡眠优化")
                    else:
                        current_step = max(0, current_step - attempt_steps)
                        interrupted = True
                        print(f"Training error: {e}")
                        stop_optimization_flag.set()
                        ensure_stop_reason("训练过程中出现错误，提前停止")
                finally:
                    del loader
                    del dataset
                    gc.collect()
                    torch.cuda.empty_cache()
            if stop_optimization_flag.is_set():
                interrupted = True
                ensure_stop_reason(early_stop_reason)
                break
            if chunk_loss_acc > 0 and chunk_sample_count > 0:
                chunk_avg_loss = chunk_loss_acc / chunk_sample_count
                update_meta_with_loss(lmdb_key_plan, bf, chunk_avg_loss)

        print(f"Final Loss Snapshot: {last_loss_value:.6f} | Steps: {current_step}/{total_steps}")
        saved_ok = safe_save_model(early_stop_reason, last_loss_value, current_step, total_steps, interrupted)
        if interrupted:
            print("Optimization Interrupted (Safe Save).")
        else:
            print("Optimization Complete (Safe Save).")
        focus_weight = float(torch.exp(-model.log_var_action.detach()).item())
        curiosity_weight = float(torch.exp(-model.log_var_prediction.detach()).item())
        laziness_penalty = float(torch.exp(-model.log_var_energy.detach()).item())
        reason_text = early_stop_reason or user_stop_request_reason or f"Keyboard interrupt after {current_step}/{total_steps} steps"
        if interrupted:
            print("[Optimization Interrupted]")
            print(f"> Reason: {reason_text}")
            update_window_status(f"睡眠优化中断: {reason_text}", "warn")
        else:
            print("[Optimization Done]")
            update_window_status("睡眠优化完成，返回学习模式", "info")
        print(f"> Imitation Weight (Focus): {focus_weight:.3f}  <-- exp(-alpha)")
        print(f"> Prediction Weight (Curiosity): {curiosity_weight:.3f}  <-- exp(-beta)")
        print(f"> Energy Penalty (Laziness): {laziness_penalty:.3f}  <-- exp(-gamma)")
        print(f"> Last Loss: {last_loss_value:.6f}")
        print(f"> Model Saved To: {model_path}")
        used_file_set = {p for group in batch_files for p in group}
        train_file_count = len(used_file_set)
        estimated_samples = 0
        human_ratio = 0.0
        ai_ratio = 0.0
        recent_ratio = 0.0
        recent_cutoff = time.time() - 7 * 24 * 60 * 60
        if sleep_stats:
            used_stats = [s for s in sleep_stats if s[0] in used_file_set]
            if used_stats:
                total_steps_used = sum(max(1, s[5]) for s in used_stats)
                estimated_samples = total_steps_used
                human_weighted = sum(max(1, s[5]) * s[1] for s in used_stats)
                human_ratio = human_weighted / total_steps_used if total_steps_used > 0 else 0.0
                ai_ratio = 1.0 - human_ratio
                recent_weighted_len = sum(max(1, s[5]) for s in used_stats if s[4] >= recent_cutoff)
                recent_ratio = recent_weighted_len / total_steps_used if total_steps_used > 0 else 0.0
        print(f"> 训练文件数量: {train_file_count}")
        print(f"> 估计覆盖经验序列数: {estimated_samples}")
        if sleep_stats:
            print(f"> 人类/AI 样本比例(估计): {human_ratio:.3f} / {ai_ratio:.3f}")
            print(f"> 最近数据比例(近7天, 估计): {recent_ratio:.3f}")
        else:
            print("> 未获取到睡眠混合统计信息，使用原始文件集合。")
        print(f"> 有效 Epoch 数: {epochs}")
        print(f"> Batch Size: {train_batch_size}")
        print(f"> 实际使用文件数: {train_file_count}")
        print(f"> 实际训练步数: {current_step}")
        if estimated_samples > 0:
            human_steps = int(estimated_samples * human_ratio)
            ai_steps = int(estimated_samples * ai_ratio)
            print(f"> 估算人类样本步数: {human_steps} | AI样本步数: {ai_steps}")
        if early_stop_reason:
            print(f"> 提前结束提示: {early_stop_reason}")
        duration = time.time() - opt_start
        print(f"> 本轮训练用时: {duration:.2f} 秒")
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()
            except Exception as e:
                print(f"CUDA IPC cleanup warning: {e}")
        gc.collect()
    except Exception as e:
        print(f"Critical Optimization Error: {e}")
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as clean_e:
            print(f"Cleanup failure after critical error: {clean_e}")
    finally:
        try:
            if 'model' in locals() and model is not None:
                safe_save_model("异常提前终止", last_loss_value if 'last_loss_value' in locals() else None, locals().get("current_step"), locals().get("total_steps"), True)
        except Exception:
            pass
        stop_optimization_flag.clear()
        capture_pause_event.clear()
        torch.cuda.empty_cache()
        gc.collect()

def start_training_mode():
    global stop_training_flag, current_mode
    time.sleep(1.0)
    stop_optimization_flag.clear()
    stop_training_flag = False
    force_memory_cleanup(2, 0.05)
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    time.sleep(0.2)
    ctypes.windll.user32.ShowWindow(hwnd, 6)

    model_path = os.path.join(model_dir, "ai_model.pth")
    try:
        if not os.path.exists(model_path):
            print("No model found. Aborting training.")
            ctypes.windll.user32.ShowWindow(hwnd, 9)
            update_window_status("未找到模型，训练已取消。", "error")
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
        hidden_state = None
        try:
            model.share_memory()
        except Exception as e:
            print(f"Share memory warning: {e}")
        time.sleep(1.0)
        stop_training_flag = False
        train_start = time.time()
        action_steps = 0

        class PIDController:
            def __init__(self, kp, ki, kd):
                self.kp = kp
                self.ki = ki
                self.kd = kd
                self.integral_x = 0.0
                self.integral_y = 0.0
                self.prev_err_x = 0.0
                self.prev_err_y = 0.0
                self.prev_time = time.time()

            def step(self, target, current):
                now = time.time()
                dt = max(now - self.prev_time, 1e-3)
                err_x = target[0] - current[0]
                err_y = target[1] - current[1]
                self.integral_x += err_x * dt
                self.integral_y += err_y * dt
                der_x = (err_x - self.prev_err_x) / dt
                der_y = (err_y - self.prev_err_y) / dt
                self.prev_err_x = err_x
                self.prev_err_y = err_y
                self.prev_time = now
                out_x = (self.kp * err_x) + (self.ki * self.integral_x) + (self.kd * der_x)
                out_y = (self.kp * err_y) + (self.ki * self.integral_y) + (self.kd * der_y)
                return current[0] + out_x, current[1] + out_y

        pid = PIDController(0.6, 0.02, 0.3)
        jitter_amplitude = 1.5
        jitter_floor = 0.15
        jitter_decay = 0.995

        while not stop_training_flag:
            if stop_training_flag:
                break
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
                        grid_logits, pred_feat, button_logits, hidden_out = model(t_imgs, t_mins, hidden_state)
                    grid_probs = torch.softmax(grid_logits[0], dim=-1)
                    pred_cell = torch.argmax(grid_probs).item()
                    cell_x = pred_cell % grid_w
                    cell_y = pred_cell // grid_w
                    target_x = int(((cell_x + 0.5) / grid_w) * screen_w)
                    target_y = int(((cell_y + 0.5) / grid_h) * screen_h)
                    hidden_state = hidden_out.detach()
                    target_history.append((target_x, target_y))
                    avg_x = sum(p[0] for p in target_history) / len(target_history)
                    avg_y = sum(p[1] for p in target_history) / len(target_history)
                    current_mouse = mouse_ctrl.position
                    pid_target = (avg_x, avg_y)
                    pid_out = pid.step(pid_target, current_mouse)
                    filtered_x = (pid_out[0] * 0.8) + (current_mouse[0] * 0.2)
                    filtered_y = (pid_out[1] * 0.8) + (current_mouse[1] * 0.2)
                    jitter_x = random.uniform(-jitter_amplitude, jitter_amplitude)
                    jitter_y = random.uniform(-jitter_amplitude, jitter_amplitude)
                    jitter_amplitude = max(jitter_floor, jitter_amplitude * jitter_decay)
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
                action_steps += 1
                elapsed = time.time() - start_time
                wait = (1.0 / capture_freq) - elapsed
                if wait > 0:
                    time.sleep(wait)
            except Exception as e:
                print(f"Training Runtime Error: {e}")
                update_window_status(f"训练过程中出现异常: {e}", "error")
                break
        flush_buffers()
        ctypes.windll.user32.ShowWindow(hwnd, 9)
        current_mode = MODE_LEARNING
        duration = time.time() - train_start
        print(f"训练模式总结: 步数 {action_steps}, 人类样本 0, AI样本 {action_steps}, 用时 {duration:.2f} 秒")
        print("Exited Training Mode. Back to Learning.")
        update_window_status(f"训练结束，步数{action_steps}，返回学习模式。", "info")
    finally:
        current_mode = MODE_LEARNING
        input_allowed_event.set()

def record_data_loop():
    buffer_images = []
    buffer_actions = []
    last_pos = (0, 0)
    chunk_target = random.randint(60, 100)

    while True:
        if recording_pause_event.is_set():
            if flush_event.is_set():
                if buffer_images:
                    img_arr, act_arr = prepare_saved_arrays(buffer_images, buffer_actions)
                    source_type = "ai" if current_mode == MODE_TRAINING else "human"
                    save_queue.put((img_arr, act_arr, source_type))
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

                if len(buffer_images) >= chunk_target:
                    img_arr, act_arr = prepare_saved_arrays(buffer_images, buffer_actions)
                    source_type = "ai" if current_mode == MODE_TRAINING else "human"
                    save_queue.put((img_arr, act_arr, source_type))
                    buffer_images = []
                    buffer_actions = []
                    chunk_target = random.randint(60, 100)

                elapsed = time.time() - start_time
                wait = (1.0 / capture_freq) - elapsed
                if wait > 0:
                    time.sleep(wait)
            except Exception as e:
                print(f"Recording Error: {e}")
            if flush_event.is_set():
                if buffer_images:
                    img_arr, act_arr = prepare_saved_arrays(buffer_images, buffer_actions)
                    source_type = "ai" if current_mode == MODE_TRAINING else "human"
                    save_queue.put((img_arr, act_arr, source_type))
                    buffer_images = []
                    buffer_actions = []
                    save_queue.join()
                flush_event.clear()
                flush_done_event.set()
        else:
            if flush_event.is_set():
                if buffer_images:
                    img_arr, act_arr = prepare_saved_arrays(buffer_images, buffer_actions)
                    source_type = "ai" if current_mode == MODE_TRAINING else "human"
                    save_queue.put((img_arr, act_arr, source_type))
                    buffer_images = []
                    buffer_actions = []
                    save_queue.join()
                flush_event.clear()
                flush_done_event.set()
            time.sleep(1)

def request_sleep_mode():
    global current_mode, user_stop_request_reason
    if current_mode != MODE_LEARNING:
        msg = f"当前模式为{current_mode}，暂无法进入睡眠模式。"
        print(msg)
        update_window_status(msg, "warn")
        input_allowed_event.set()
        return
    input_allowed_event.clear()
    user_stop_request_reason = None
    def run():
        global current_mode
        try:
            update_window_status("检测到睡眠指令，正在准备进入睡眠模式...", "info")
            recording_pause_event.set()
            capture_pause_event.set()
            flush_buffers()
            cleanup_before_sleep()
            current_mode = MODE_SLEEP
            update_window_mode(current_mode)
            optimize_ai()
        except Exception as e:
            log_exception("Sleep mode failure", e)
        finally:
            current_mode = MODE_LEARNING
            recording_pause_event.clear()
            capture_pause_event.clear()
            input_allowed_event.set()
            update_window_mode(current_mode)
            update_window_status("Back to Learning.", "info")
    threading.Thread(target=run, daemon=True).start()

def request_training_mode():
    global current_mode, stop_training_flag
    if current_mode != MODE_LEARNING:
        msg = f"当前模式为{current_mode}，暂无法进入训练模式。"
        print(msg)
        update_window_status(msg, "warn")
        input_allowed_event.set()
        return
    input_allowed_event.clear()
    flush_buffers()
    current_mode = MODE_TRAINING
    update_window_mode(current_mode)
    update_window_status("检测到训练指令，进入训练模式...", "info")
    if window_ui is not None:
        window_ui.minimize()
    t_thread = threading.Thread(target=start_training_mode)
    t_thread.daemon = True
    t_thread.start()

def request_early_stop():
    global current_mode, user_stop_request_reason
    if current_mode == MODE_SLEEP:
        user_stop_request_reason = "用户点击早停按钮"
        stop_optimization_flag.set()
        msg = "早停请求已发出，正在停止优化并保存数据..."
        print(msg)
        update_window_status(msg, "warn")
    else:
        msg = f"当前模式为{current_mode}，早停请求未生效。"
        print(msg)
        update_window_status(msg, "warn")
        input_allowed_event.set()

def start_background_services():
    global mouse_listener, key_listener
    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
    mouse_listener.start()
    key_listener = keyboard.Listener(on_press=on_press_key)
    key_listener.start()
    t_res = threading.Thread(target=resource_monitor, daemon=True)
    t_res.start()
    t_frame = threading.Thread(target=frame_generator_loop, daemon=True)
    t_frame.start()
    t_save = threading.Thread(target=disk_writer_loop, daemon=True)
    t_save.start()
    t_rec = threading.Thread(target=record_data_loop, daemon=True)
    t_rec.start()
    print("System initialized. Mode: LEARNING")
    update_window_status("系统初始化完成，进入学习模式。", "info")
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


if __name__ == "__main__":
    ensure_initial_model()
    set_process_priority()
    init_window()
    update_window_mode(current_mode)
    update_window_status("System initializing...", "info")
    t_bg_logic = threading.Thread(target=start_background_services, daemon=True)
    t_bg_logic.start()
    if window_ui is not None:
        window_ui.run()
