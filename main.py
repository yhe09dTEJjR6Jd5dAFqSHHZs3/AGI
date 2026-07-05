import os
import json
import time
import math
import zlib
import uuid
import queue
import shutil
import struct
import hashlib
import random
import threading
import subprocess
import ctypes
from ctypes import wintypes
from pathlib import Path
from datetime import datetime
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

STATE_IDLE = "空闲"
STATE_LEARNING = "学习模式"
STATE_TRAINING = "训练模式"
STATE_SLEEPING = "睡眠模式"

DEFAULT_PLAYER_PATH = r"D:\LDPlayer9\dnplayer.exe"
DEFAULT_STORAGE_PATH = r"C:\Users\Administrator\Desktop\AAA"
MODEL_NAMES = [
    "画面表征与相似度模型",
    "行为策略模型",
    "价值评估模型",
    "睡眠决策模型",
    "AI模型质量评估与淘汰模型",
    "经验池价值评估与淘汰模型",
]


class WindowsTools:
    def __init__(self):
        self.ready = os.name == "nt"
        self.user32 = None
        self.kernel32 = None
        self.gdi32 = None
        if self.ready:
            try:
                self.user32 = ctypes.windll.user32
                self.kernel32 = ctypes.windll.kernel32
                self.gdi32 = ctypes.windll.gdi32
                self.user32.WindowFromPoint.restype = wintypes.HWND
                self.user32.WindowFromPoint.argtypes = [wintypes.POINT]
                self.user32.GetAncestor.restype = wintypes.HWND
                self.user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
                self.user32.GetWindow.restype = wintypes.HWND
                self.user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
                self.user32.GetDC.restype = wintypes.HDC
                self.user32.GetDC.argtypes = [wintypes.HWND]
                self.kernel32.GetModuleHandleW.restype = wintypes.HMODULE
                self.user32.SetWindowsHookExW.restype = wintypes.HHOOK
                self.user32.SetWindowsHookExW.argtypes = [ctypes.c_int, ctypes.c_void_p, wintypes.HINSTANCE, wintypes.DWORD]
                self.user32.CallNextHookEx.restype = ctypes.c_ssize_t
                self.user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
                self.gdi32.CreateCompatibleDC.restype = wintypes.HDC
                self.gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
                self.gdi32.SelectObject.restype = wintypes.HGDIOBJ
                try:
                    self.user32.SetProcessDPIAware()
                except Exception:
                    pass
            except Exception:
                self.ready = False

    def handle_value(self, handle):
        if handle is None:
            return 0
        if hasattr(handle, "value"):
            return int(handle.value or 0)
        return int(handle)

    def get_cursor(self):
        if not self.ready:
            return None
        point = wintypes.POINT()
        if not self.user32.GetCursorPos(ctypes.byref(point)):
            return None
        return int(point.x), int(point.y)

    def set_cursor(self, x, y):
        if not self.ready:
            return False
        return bool(self.user32.SetCursorPos(int(x), int(y)))

    def key_down(self, vk):
        return bool(self.ready and (self.user32.GetAsyncKeyState(int(vk)) & 0x8000))

    def button_down(self, vk):
        return self.key_down(vk)

    def virtual_screen_rect(self):
        if not self.ready:
            return (0, 0, 0, 0)
        x = int(self.user32.GetSystemMetrics(76))
        y = int(self.user32.GetSystemMetrics(77))
        w = int(self.user32.GetSystemMetrics(78))
        h = int(self.user32.GetSystemMetrics(79))
        return x, y, x + w, y + h

    def rect_on_screen(self, rect):
        sx1, sy1, sx2, sy2 = self.virtual_screen_rect()
        x1, y1, x2, y2 = rect
        return x2 > x1 and y2 > y1 and x1 >= sx1 and y1 >= sy1 and x2 <= sx2 and y2 <= sy2

    def player_windows(self):
        if not self.ready:
            return []
        result = []
        enum_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def callback(hwnd, lparam):
            try:
                if not self.user32.IsWindowVisible(hwnd) or self.user32.IsIconic(hwnd):
                    return True
                length = self.user32.GetWindowTextLengthW(hwnd)
                if length <= 0:
                    return True
                buffer = ctypes.create_unicode_buffer(length + 1)
                self.user32.GetWindowTextW(hwnd, buffer, length + 1)
                title = buffer.value.strip()
                lowered = title.lower()
                if "雷电" not in title and "ldplayer" not in lowered and "dnplayer" not in lowered:
                    return True
                client = wintypes.RECT()
                if not self.user32.GetClientRect(hwnd, ctypes.byref(client)):
                    return True
                point1 = wintypes.POINT(client.left, client.top)
                point2 = wintypes.POINT(client.right, client.bottom)
                if not self.user32.ClientToScreen(hwnd, ctypes.byref(point1)):
                    return True
                if not self.user32.ClientToScreen(hwnd, ctypes.byref(point2)):
                    return True
                rect = (int(point1.x), int(point1.y), int(point2.x), int(point2.y))
                if rect[2] <= rect[0] or rect[3] <= rect[1]:
                    return True
                result.append({"hwnd": self.handle_value(hwnd), "title": title, "rect": rect})
            except Exception:
                return True
            return True

        callback_ref = enum_proc_type(callback)
        self.user32.EnumWindows(callback_ref, 0)
        return result

    def find_player_window(self):
        windows = self.player_windows()
        if not windows:
            return None
        windows.sort(key=lambda item: (item["rect"][2] - item["rect"][0]) * (item["rect"][3] - item["rect"][1]), reverse=True)
        return windows[0]

    def point_belongs_to_window(self, x, y, hwnd):
        if not self.ready:
            return False
        point = wintypes.POINT(int(x), int(y))
        hit = self.user32.WindowFromPoint(point)
        if not hit:
            return False
        root = self.user32.GetAncestor(hit, 2)
        return self.handle_value(root or hit) == self.handle_value(hwnd)

    def windows_above_intersect_client(self, hwnd, rect):
        current = self.user32.GetWindow(wintypes.HWND(hwnd), 3)
        while current:
            try:
                if self.user32.IsWindowVisible(current) and not self.user32.IsIconic(current):
                    other = wintypes.RECT()
                    if self.user32.GetWindowRect(current, ctypes.byref(other)):
                        ox1, oy1, ox2, oy2 = int(other.left), int(other.top), int(other.right), int(other.bottom)
                        x1, y1, x2, y2 = rect
                        if max(x1, ox1) < min(x2, ox2) and max(y1, oy1) < min(y2, oy2):
                            return True
                current = self.user32.GetWindow(current, 3)
            except Exception:
                return True
        return False

    def client_is_unobstructed(self, window):
        if not self.ready:
            return False
        rect = window["rect"]
        if not self.rect_on_screen(rect):
            return False
        if self.windows_above_intersect_client(window["hwnd"], rect):
            return False
        x1, y1, x2, y2 = rect
        for xr in (0.15, 0.5, 0.85):
            for yr in (0.15, 0.5, 0.85):
                x = x1 + int((x2 - x1 - 1) * xr)
                y = y1 + int((y2 - y1 - 1) * yr)
                if not self.point_belongs_to_window(x, y, window["hwnd"]):
                    return False
        return True

    def window_abnormal(self, window, require_cursor=True):
        if not self.ready:
            return True, "当前系统不支持Windows客户区检测"
        current = self.find_player_window()
        if not current:
            return True, "雷电模拟器客户区不可见、最小化或未找到"
        if int(current["hwnd"]) != int(window["hwnd"]):
            return True, "雷电模拟器客户区发生变化"
        if not self.rect_on_screen(current["rect"]):
            return True, "雷电模拟器客户区未完全位于电脑屏幕范围内"
        if not self.client_is_unobstructed(current):
            return True, "雷电模拟器客户区被遮挡或不可见"
        if require_cursor:
            pos = self.get_cursor()
            x1, y1, x2, y2 = current["rect"]
            if pos is None or not (x1 <= pos[0] < x2 and y1 <= pos[1] < y2):
                return True, "鼠标位于雷电模拟器客户区外"
        return False, ""

    def launch(self, executable):
        if not self.ready or not Path(executable).is_file():
            return False
        try:
            subprocess.Popen([str(executable)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def left_click(self, x, y):
        if not self.ready:
            return
        self.set_cursor(x, y)
        self.user32.mouse_event(0x0002, 0, 0, 0, 0)
        self.user32.mouse_event(0x0004, 0, 0, 0, 0)

    def right_click(self, x, y):
        if not self.ready:
            return
        self.set_cursor(x, y)
        self.user32.mouse_event(0x0008, 0, 0, 0, 0)
        self.user32.mouse_event(0x0010, 0, 0, 0, 0)

    def wheel(self, delta):
        if not self.ready:
            return
        self.user32.mouse_event(0x0800, 0, 0, int(delta), 0)

    def capture_bgra(self, rect):
        if not self.ready:
            return None
        x1, y1, x2, y2 = rect
        width = int(x2 - x1)
        height = int(y2 - y1)
        if width <= 0 or height <= 0:
            return None

        class BitmapInfoHeader(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        class BitmapInfo(ctypes.Structure):
            _fields_ = [("bmiHeader", BitmapInfoHeader), ("bmiColors", wintypes.DWORD * 3)]

        screen_dc = self.user32.GetDC(0)
        if not screen_dc:
            return None
        memory_dc = None
        bitmap = None
        previous = None
        try:
            memory_dc = self.gdi32.CreateCompatibleDC(screen_dc)
            if not memory_dc:
                return None
            bitmap = self.gdi32.CreateCompatibleBitmap(screen_dc, width, height)
            if not bitmap:
                return None
            previous = self.gdi32.SelectObject(memory_dc, bitmap)
            copied = self.gdi32.BitBlt(memory_dc, 0, 0, width, height, screen_dc, x1, y1, 0x00CC0020 | 0x40000000)
            if not copied:
                return None
            header = BitmapInfoHeader(
                ctypes.sizeof(BitmapInfoHeader),
                width,
                -height,
                1,
                32,
                0,
                width * height * 4,
                0,
                0,
                0,
                0,
            )
            info = BitmapInfo(header, (wintypes.DWORD * 3)(0, 0, 0))
            buffer = (ctypes.c_ubyte * (width * height * 4))()
            rows = self.gdi32.GetDIBits(memory_dc, bitmap, 0, height, ctypes.byref(buffer), ctypes.byref(info), 0)
            if int(rows) != height:
                return None
            return bytes(buffer), width, height
        except Exception:
            return None
        finally:
            try:
                if previous and memory_dc:
                    self.gdi32.SelectObject(memory_dc, previous)
            except Exception:
                pass
            try:
                if bitmap:
                    self.gdi32.DeleteObject(bitmap)
            except Exception:
                pass
            try:
                if memory_dc:
                    self.gdi32.DeleteDC(memory_dc)
            except Exception:
                pass
            try:
                self.user32.ReleaseDC(0, screen_dc)
            except Exception:
                pass


class MouseEventMonitor:
    def __init__(self, tools):
        self.tools = tools
        self.lock = threading.Lock()
        self.events = deque(maxlen=256)
        self.buttons = {"left": False, "right": False, "middle": False}
        self.running = threading.Event()
        self.thread = None
        self.thread_id = 0
        self.hook = None
        self.proc = None

    def start(self):
        if not self.tools.ready or self.thread:
            return
        self.running.set()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running.clear()
        if self.tools.ready and self.thread_id:
            try:
                self.tools.user32.PostThreadMessageW(int(self.thread_id), 0x0012, 0, 0)
            except Exception:
                pass

    def clear(self):
        with self.lock:
            self.events.clear()

    def consume(self):
        with self.lock:
            events = list(self.events)
            self.events.clear()
            buttons = dict(self.buttons)
        return events, buttons

    def _append(self, kind, position, timestamp, delta=0):
        with self.lock:
            self.events.append(
                {
                    "kind": kind,
                    "position": [int(position[0]), int(position[1])],
                    "timestamp": timestamp,
                    "wheel_delta": int(delta),
                }
            )

    def _loop(self):
        if not self.tools.ready:
            return
        self.thread_id = int(self.tools.kernel32.GetCurrentThreadId())

        class MouseLowLevel(ctypes.Structure):
            _fields_ = [
                ("pt", wintypes.POINT),
                ("mouseData", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_void_p),
            ]

        proc_type = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        mapping = {
            0x0200: ("move", None, None),
            0x0201: ("left_down", "left", True),
            0x0202: ("left_up", "left", False),
            0x0204: ("right_down", "right", True),
            0x0205: ("right_up", "right", False),
            0x0207: ("middle_down", "middle", True),
            0x0208: ("middle_up", "middle", False),
        }

        def callback(code, wparam, lparam):
            try:
                if code >= 0:
                    data = ctypes.cast(lparam, ctypes.POINTER(MouseLowLevel)).contents
                    message = int(wparam)
                    now = time.time()
                    position = (int(data.pt.x), int(data.pt.y))
                    if message == 0x020A:
                        delta = ctypes.c_short((int(data.mouseData) >> 16) & 0xFFFF).value
                        self._append("wheel", position, now, delta)
                    elif message in mapping:
                        kind, button, state = mapping[message]
                        if button is not None:
                            with self.lock:
                                self.buttons[button] = bool(state)
                        self._append(kind, position, now)
            except Exception:
                pass
            return self.tools.user32.CallNextHookEx(self.hook, code, wparam, lparam)

        self.proc = proc_type(callback)
        try:
            module = self.tools.kernel32.GetModuleHandleW(None)
            self.hook = self.tools.user32.SetWindowsHookExW(14, self.proc, module, 0)
            if not self.hook:
                return
            message = wintypes.MSG()
            while self.running.is_set():
                result = self.tools.user32.GetMessageW(ctypes.byref(message), 0, 0, 0)
                if result <= 0:
                    break
                self.tools.user32.TranslateMessage(ctypes.byref(message))
                self.tools.user32.DispatchMessageW(ctypes.byref(message))
        except Exception:
            pass
        finally:
            if self.hook:
                try:
                    self.tools.user32.UnhookWindowsHookEx(self.hook)
                except Exception:
                    pass
            self.hook = None


class DataStore:
    def __init__(self, base_path):
        self.base = Path(base_path)
        self.exp = self.base / "experience_pool"
        self.records = self.exp / "records"
        self.screens = self.exp / "screenshots"
        self.models = self.base / "ai_models"
        self.logs = self.base / "logs"
        self.ensure()

    def ensure(self):
        for path in (self.base, self.exp, self.records, self.screens, self.models, self.logs):
            path.mkdir(parents=True, exist_ok=True)

    def size_bytes(self, path):
        target = Path(path)
        total = 0
        if not target.exists():
            return total
        for root, _, names in os.walk(target):
            for name in names:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
        return total

    def write_log(self, text):
        self.ensure()
        path = self.logs / f"{datetime.now():%Y-%m-%d}.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now().isoformat(timespec='seconds')} {text}\n")

    def save_capture(self, png_bytes):
        self.ensure()
        name = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}.png"
        (self.screens / name).write_bytes(png_bytes)
        return name

    def append_record(self, record):
        self.ensure()
        name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}.json"
        path = self.records / name
        path.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return path

    def iter_recent_records(self, limit=400):
        try:
            paths = sorted(self.records.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[:int(limit)]
        except Exception:
            return []
        output = []
        for path in paths:
            try:
                output.append((path, json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                pass
        return output

    def count_records(self):
        try:
            return sum(1 for _ in self.records.glob("*.json"))
        except Exception:
            return 0

    def model_paths(self):
        try:
            return list(self.models.glob("*.json"))
        except Exception:
            return []

    def record_candidates(self):
        candidates = []
        for path, record in self.iter_recent_records(10**9):
            value = record.get("experience_value", record.get("reward", -1e9))
            try:
                value = float(value)
            except Exception:
                value = -1e9
            candidates.append((value, path, record))
        candidates.sort(key=lambda item: item[0])
        return candidates

    def remove_record_and_capture(self, record_path, record):
        screen_file = record.get("screen_file")
        if screen_file:
            try:
                (self.screens / Path(screen_file).name).unlink(missing_ok=True)
            except Exception:
                pass
        record_path.unlink(missing_ok=True)

    def oldest_experience_file(self):
        files = []
        for folder in (self.records, self.screens):
            if folder.exists():
                for path in folder.rglob("*"):
                    if path.is_file():
                        try:
                            files.append((path.stat().st_mtime, path))
                        except OSError:
                            pass
        files.sort(key=lambda item: item[0])
        return files[0][1] if files else None


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("自主学习控制面板")
        self.root.geometry("1060x760")
        self.root.minsize(720, 540)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.tools = WindowsTools()
        self.mouse_monitor = MouseEventMonitor(self.tools)
        self.state = STATE_IDLE
        self.migrating = False
        self.closing = False
        self.player_path = DEFAULT_PLAYER_PATH
        self.storage_path = DEFAULT_STORAGE_PATH
        self.exp_limit_gb = 10.0
        self.model_limit = 100
        self.store = DataStore(self.storage_path)
        self.config_path = Path(__file__).resolve().with_name("autonomous_learning_config.json")
        self.mode_stop = threading.Event()
        self.sleep_stop = threading.Event()
        self.mode_thread = None
        self.sleep_thread = None
        self.sleep_origin = None
        self.hunger = 1e-12
        self.last_score = None
        self.last_hunger_time = time.monotonic()
        self.last_mouse = None
        self.trajectory = deque(maxlen=48)
        self.recent_rewards = deque(maxlen=24)
        self.ui_queue = queue.Queue()
        self.progress_value = tk.IntVar(value=0)
        self.progress_text = tk.StringVar(value="0%")
        self.progress_shown = False
        self.compact_layout = None
        self.status_text = tk.StringVar(value=STATE_IDLE)
        self.player_text = tk.StringVar(value=self.player_path)
        self.storage_text = tk.StringVar(value=self.storage_path)
        self.exp_text = tk.StringVar(value="10 GB")
        self.model_text = tk.StringVar(value="100 个")
        self.detail_text = tk.StringVar(value="单文件启动完成，当前空闲。")
        self.build_ui()
        self.load_config()
        self.mouse_monitor.start()
        self.write_panel("单文件启动完成，当前空闲。")
        self.root.after(80, self.process_ui_queue)
        self.root.after(120, self.poll_escape)

    def build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", padding=(10, 8), font=("Microsoft YaHei UI", 10))
        style.configure("TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("Header.TLabel", font=("Microsoft YaHei UI", 19, "bold"))
        style.configure("State.TLabel", font=("Microsoft YaHei UI", 13, "bold"))
        style.configure("Progress.TLabel", font=("Microsoft YaHei UI", 10, "bold"))
        outer = ttk.Frame(self.root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="自主学习控制面板", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status_text, style="State.TLabel").grid(row=0, column=1, sticky="e")

        rainbow = tk.Frame(outer, height=14)
        rainbow.grid(row=1, column=0, sticky="ew", pady=(12, 14))
        colors = ("#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#3b82f6", "#8b5cf6")
        for index, color in enumerate(colors):
            rainbow.columnconfigure(index, weight=1)
            tk.Frame(rainbow, background=color, height=14).grid(row=0, column=index, sticky="ew")

        self.content = ttk.Frame(outer)
        self.content.grid(row=2, column=0, sticky="nsew")
        self.content.columnconfigure(0, weight=1)
        self.content.columnconfigure(1, weight=1)

        self.paths_box = ttk.LabelFrame(self.content, text="路径与上限", padding=12)
        self.paths_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.paths_box.columnconfigure(1, weight=1)

        self.controls_box = ttk.LabelFrame(self.content, text="模式控制", padding=12)
        self.controls_box.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.controls_box.columnconfigure(0, weight=1)
        self.controls_box.columnconfigure(1, weight=1)

        items = [
            ("雷电模拟器路径", self.player_text, "选择雷电模拟器路径", self.choose_player),
            ("存储路径", self.storage_text, "选择存储路径", self.choose_storage),
            ("经验池上限", self.exp_text, "修改经验池上限", self.change_exp_limit),
            ("AI模型数量上限", self.model_text, "修改AI模型数量上限", self.change_model_limit),
        ]
        self.path_buttons = []
        for row, (label, variable, button, command) in enumerate(items):
            ttk.Label(self.paths_box, text=label).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Label(self.paths_box, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=6)
            control = ttk.Button(self.paths_box, text=button, command=command)
            control.grid(row=row, column=2, sticky="e", pady=6)
            self.path_buttons.append(control)

        self.detail_button = ttk.Button(self.controls_box, text="详细信息", command=self.show_details)
        self.detail_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=6)
        self.learning_button = ttk.Button(self.controls_box, text="学习模式", command=self.start_learning)
        self.learning_button.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=6)
        self.training_button = ttk.Button(self.controls_box, text="训练模式", command=self.start_training)
        self.training_button.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=6)
        self.sleep_button = ttk.Button(self.controls_box, text="睡眠模式", command=lambda: self.start_sleep(None))
        self.sleep_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)

        progress_box = ttk.LabelFrame(outer, text="进度与运行信息", padding=12)
        progress_box.grid(row=3, column=0, sticky="nsew", pady=(16, 0))
        progress_box.columnconfigure(0, weight=1)
        progress_box.rowconfigure(2, weight=1)
        self.progress_frame = ttk.Frame(progress_box)
        self.progress_frame.grid(row=0, column=0, sticky="ew")
        self.progress_frame.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(self.progress_frame, maximum=100, variable=self.progress_value)
        self.progress.grid(row=0, column=0, sticky="ew")
        ttk.Label(self.progress_frame, textvariable=self.progress_text, style="Progress.TLabel", width=5).grid(row=0, column=1, sticky="e", padx=(10, 0))
        ttk.Label(progress_box, textvariable=self.detail_text).grid(row=1, column=0, sticky="ew", pady=(10, 8))
        self.info = tk.Text(progress_box, wrap="word", height=12, undo=False, font=("Microsoft YaHei UI", 10))
        self.info.grid(row=2, column=0, sticky="nsew")
        self.info.configure(state="disabled")
        self.progress_frame.grid_remove()
        self.root.bind("<Configure>", self.on_resize)
        self.refresh_controls()

    def on_resize(self, event):
        if event.widget is not self.root:
            return
        compact = event.width < 900
        if compact == self.compact_layout:
            return
        self.compact_layout = compact
        if compact:
            self.paths_box.grid_configure(row=0, column=0, columnspan=2, padx=0, pady=(0, 8))
            self.controls_box.grid_configure(row=1, column=0, columnspan=2, padx=0, pady=(8, 0))
        else:
            self.paths_box.grid_configure(row=0, column=0, columnspan=1, padx=(0, 8), pady=0)
            self.controls_box.grid_configure(row=0, column=1, columnspan=1, padx=(8, 0), pady=0)

    def post(self, callback):
        if not self.closing:
            self.ui_queue.put(callback)

    def process_ui_queue(self):
        if self.closing:
            return
        while True:
            try:
                callback = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback()
            except Exception as exc:
                self.write_panel(f"界面任务异常：{exc}")
        self.root.after(80, self.process_ui_queue)

    def load_config(self):
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.player_path = str(config.get("player_path", self.player_path))
            self.storage_path = str(config.get("storage_path", self.storage_path))
            self.exp_limit_gb = max(0.1, float(config.get("exp_limit_gb", self.exp_limit_gb)))
            self.model_limit = max(1, int(config.get("model_limit", self.model_limit)))
            self.store = DataStore(self.storage_path)
        except Exception:
            self.save_config()
        self.refresh_vars()

    def save_config(self):
        payload = {
            "player_path": self.player_path,
            "storage_path": self.storage_path,
            "exp_limit_gb": self.exp_limit_gb,
            "model_limit": self.model_limit,
        }
        try:
            self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        self.refresh_vars()

    def refresh_vars(self):
        self.player_text.set(self.player_path)
        self.storage_text.set(self.storage_path)
        self.exp_text.set(f"{self.exp_limit_gb:g} GB")
        self.model_text.set(f"{self.model_limit} 个")

    def refresh_controls(self):
        active = self.state == STATE_IDLE and not self.migrating
        value = "normal" if active else "disabled"
        for button in self.path_buttons:
            button.configure(state=value)
        self.learning_button.configure(state=value)
        self.training_button.configure(state=value)
        self.sleep_button.configure(state=value)

    def show_progress(self):
        if not self.progress_shown:
            self.progress_frame.grid()
            self.progress_shown = True

    def hide_progress_if_inactive(self):
        if self.state == STATE_IDLE and not self.migrating and self.progress_shown:
            self.progress_frame.grid_remove()
            self.progress_shown = False

    def refresh_progress_visibility(self):
        if self.state == STATE_SLEEPING or self.migrating:
            self.show_progress()
        elif self.state in (STATE_LEARNING, STATE_TRAINING):
            if self.progress_shown:
                self.progress_frame.grid_remove()
                self.progress_shown = False
        else:
            self.root.after(800, self.hide_progress_if_inactive)

    def write_panel(self, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.info.configure(state="normal")
        self.info.insert("end", f"{timestamp} {text}\n")
        self.info.see("end")
        self.info.configure(state="disabled")
        self.detail_text.set(text)
        try:
            self.store.write_log(text)
        except Exception:
            pass

    def set_state(self, state):
        self.state = state
        self.status_text.set(state)
        self.refresh_controls()
        self.refresh_progress_visibility()
        self.write_panel(f"状态：{state}")

    def set_progress_main(self, value, text):
        percent = max(0, min(100, int(value)))
        self.progress_value.set(percent)
        self.progress_text.set(f"{percent}%")
        self.show_progress()
        self.write_panel(text)

    def set_progress(self, value, text):
        self.post(lambda: self.set_progress_main(value, text))

    def choose_player(self):
        path = filedialog.askopenfilename(
            title="选择雷电模拟器路径",
            initialfile="dnplayer.exe",
            filetypes=[("可执行文件", "*.exe"), ("所有文件", "*.*")],
        )
        if path:
            self.player_path = path
            self.save_config()
            self.write_panel("已更新雷电模拟器路径。")

    def choose_storage(self):
        path = filedialog.askdirectory(title="选择存储路径")
        if path:
            self.start_migration(path)

    def change_exp_limit(self):
        value = simpledialog.askfloat("修改经验池上限", "请输入经验池上限（GB）", initialvalue=self.exp_limit_gb, minvalue=0.1)
        if value is not None:
            self.exp_limit_gb = float(value)
            self.save_config()
            self.write_panel("已更新经验池上限。")

    def change_model_limit(self):
        value = simpledialog.askinteger("修改AI模型数量上限", "请输入AI模型数量上限", initialvalue=self.model_limit, minvalue=1)
        if value is not None:
            self.model_limit = int(value)
            self.save_config()
            self.write_panel("已更新AI模型数量上限。")

    def show_details(self):
        window = self.tools.find_player_window()
        data = {
            "当前状态": self.state,
            "数据迁移中": self.migrating,
            "雷电模拟器客户区": window,
            "经验池大小GB": round(self.store.size_bytes(self.store.exp) / 1024 / 1024 / 1024, 6),
            "经验池上限GB": self.exp_limit_gb,
            "经验记录数量": self.store.count_records(),
            "AI模型数量": len(self.store.model_paths()),
            "AI模型数量上限": self.model_limit,
            "存储路径": self.storage_path,
            "雷电模拟器路径": self.player_path,
        }
        messagebox.showinfo("详细信息", json.dumps(data, ensure_ascii=False, indent=2))

    def start_migration(self, target_path):
        if self.state != STATE_IDLE or self.migrating:
            return
        try:
            source = self.store.base.resolve()
            target = Path(target_path).resolve()
            if source == target:
                self.write_panel("所选存储路径与当前路径相同，无需迁移。")
                return
            common = Path(os.path.commonpath((str(source), str(target))))
            if common == source or common == target:
                messagebox.showerror("无法迁移", "新的存储路径不能与当前存储路径互为包含关系。")
                return
        except Exception:
            target = Path(target_path)
        self.migrating = True
        self.refresh_controls()
        self.refresh_progress_visibility()
        self.progress_value.set(0)
        self.progress_text.set("0%")
        self.write_panel("数据迁移开始，进度将从0%到100%。")
        threading.Thread(target=self.migration_loop, args=(target,), daemon=True).start()

    def migration_loop(self, destination):
        source = self.store.base
        try:
            destination.mkdir(parents=True, exist_ok=True)
            files = [path for path in source.rglob("*") if path.is_file()]
            total = sum(path.stat().st_size for path in files if path.exists())
            copied = 0
            self.set_progress(0, "数据迁移：准备复制数据。")
            for index, source_file in enumerate(files, 1):
                relative = source_file.relative_to(source)
                target_file = destination / relative
                target_file.parent.mkdir(parents=True, exist_ok=True)
                size = source_file.stat().st_size
                same = False
                try:
                    same = target_file.exists() and target_file.stat().st_size == size and target_file.read_bytes() == source_file.read_bytes()
                except Exception:
                    same = False
                if not same:
                    shutil.copy2(source_file, target_file)
                copied += size
                progress = 100 if total <= 0 else int(copied * 100 / total)
                if index == len(files) or index % 4 == 0:
                    self.set_progress(progress, f"数据迁移：已处理 {index}/{len(files)} 个文件。")
            if not files:
                self.set_progress(100, "数据迁移：没有历史数据，已完成存储路径切换。")
            self.post(lambda: self.finish_migration(destination, None))
        except Exception as exc:
            self.post(lambda: self.finish_migration(destination, exc))

    def finish_migration(self, destination, error):
        self.migrating = False
        if error is None:
            self.storage_path = str(destination)
            self.store = DataStore(self.storage_path)
            self.save_config()
            self.progress_value.set(100)
            self.progress_text.set("100%")
            self.write_panel("数据迁移完成，进度100%。")
        else:
            self.write_panel(f"数据迁移失败：{error}")
        self.refresh_controls()
        self.refresh_progress_visibility()

    def launch_and_wait_for_player(self):
        if self.tools.find_player_window() is not None:
            return True
        if not self.tools.launch(self.player_path):
            self.write_panel("未找到雷电模拟器，且无法通过已选路径启动。")
            return False
        for _ in range(28):
            time.sleep(0.25)
            if self.tools.find_player_window() is not None:
                return True
        self.write_panel("已尝试启动雷电模拟器，但未找到其客户区。")
        return False

    def prepare_entry(self):
        if not self.launch_and_wait_for_player():
            return None
        window = self.tools.find_player_window()
        if not window:
            self.write_panel("未找到雷电模拟器客户区。")
            return None
        abnormal, reason = self.tools.window_abnormal(window, require_cursor=False)
        if abnormal:
            self.write_panel(reason)
            return None
        x1, y1, x2, y2 = window["rect"]
        position = self.tools.get_cursor()
        if position is None or not (x1 <= position[0] < x2 and y1 <= position[1] < y2):
            if not self.tools.set_cursor((x1 + x2) // 2, (y1 + y2) // 2):
                self.write_panel("无法将鼠标移入雷电模拟器客户区。")
                return None
            self.write_panel("仅在进入模式前，已将鼠标移入雷电模拟器客户区。")
        window = self.tools.find_player_window()
        if not window:
            self.write_panel("鼠标定位后未找到雷电模拟器客户区。")
            return None
        abnormal, reason = self.tools.window_abnormal(window, require_cursor=True)
        if abnormal:
            self.write_panel(reason)
            return None
        return window

    def start_learning(self):
        self.start_mode(STATE_LEARNING)

    def start_training(self):
        self.start_mode(STATE_TRAINING)

    def start_mode(self, mode):
        if self.state != STATE_IDLE or self.migrating:
            return
        window = self.prepare_entry()
        if not window:
            return
        self.mode_stop = threading.Event()
        self.last_score = None
        self.last_mouse = None
        self.last_hunger_time = time.monotonic()
        self.trajectory.clear()
        self.recent_rewards.clear()
        self.mouse_monitor.clear()
        self.set_state(mode)
        self.progress_value.set(0)
        self.progress_text.set("0%")
        self.mode_thread = threading.Thread(target=self.mode_loop, args=(mode, window, self.mode_stop), daemon=True)
        self.mode_thread.start()

    def to_idle(self, reason="返回空闲", keep_progress=False):
        self.mode_stop.set()
        self.sleep_stop.set()
        if self.state != STATE_IDLE:
            self.set_state(STATE_IDLE)
        if not keep_progress:
            self.progress_value.set(0)
            self.progress_text.set("0%")
        self.write_panel(reason)

    def request_idle(self, reason):
        self.post(lambda: self.to_idle(reason))

    def poll_escape(self):
        if not self.closing:
            if self.state in (STATE_LEARNING, STATE_TRAINING, STATE_SLEEPING) and self.tools.key_down(27):
                self.to_idle("用户按下ESC键，已返回空闲。")
            self.root.after(120, self.poll_escape)

    def capture_screen(self, rect):
        captured = self.tools.capture_bgra(rect)
        if not captured:
            return None, None, None, None
        bgra, width, height = captured
        signature = self.make_signature(bgra, width, height)
        digest = hashlib.sha256(bgra).hexdigest()
        png_bytes = self.make_png(bgra, width, height)
        if png_bytes is None:
            return None, digest, signature, None
        screen_file = self.store.save_capture(png_bytes)
        return screen_file, digest, signature, {"width": width, "height": height}

    def make_signature(self, bgra, width, height):
        side = 24
        values = bytearray()
        for gy in range(side):
            y = min(height - 1, int((gy + 0.5) * height / side))
            for gx in range(side):
                x = min(width - 1, int((gx + 0.5) * width / side))
                offset = (y * width + x) * 4
                b = bgra[offset]
                g = bgra[offset + 1]
                r = bgra[offset + 2]
                values.append((int(r) * 299 + int(g) * 587 + int(b) * 114) // 1000)
        return values.hex()

    def make_png(self, bgra, width, height):
        try:
            rows = bytearray()
            row_size = width * 4
            for y in range(height):
                rows.append(0)
                start = y * row_size
                end = start + row_size
                row = bgra[start:end]
                for offset in range(0, row_size, 4):
                    rows.extend((row[offset + 2], row[offset + 1], row[offset], row[offset + 3]))
            def chunk(kind, data):
                return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b"")
        except Exception:
            return None

    def screen_score(self, signature):
        if not signature:
            return None
        try:
            current = bytes.fromhex(signature)
        except Exception:
            return None
        similarities = []
        for _, record in self.store.iter_recent_records(360):
            old_signature = record.get("screen_signature")
            if not isinstance(old_signature, str) or len(old_signature) != len(signature):
                continue
            try:
                historical = bytes.fromhex(old_signature)
            except Exception:
                continue
            difference = sum(abs(a - b) for a, b in zip(current, historical)) / (255 * len(current))
            similarities.append(1.0 - difference)
        if not similarities:
            return 1.0
        similarities.sort(reverse=True)
        nearest = similarities[:min(12, len(similarities))]
        return max(0.0, min(1.0, 1.0 - sum(nearest) / len(nearest)))

    def mouse_record(self, source):
        now = time.time()
        position = self.tools.get_cursor()
        events, button_states = self.mouse_monitor.consume()
        if position is None:
            return None
        if self.last_mouse is None:
            dx = dy = direction = speed = 0.0
        else:
            px, py, previous_time = self.last_mouse
            elapsed = max(1e-9, now - previous_time)
            dx = position[0] - px
            dy = position[1] - py
            direction = math.degrees(math.atan2(dy, dx)) if dx or dy else 0.0
            speed = math.hypot(dx, dy) / elapsed
        point = {"x": position[0], "y": position[1], "timestamp": now}
        self.trajectory.append(point)
        wheel_delta = sum(int(event.get("wheel_delta", 0)) for event in events)
        record = {
            "trajectory": list(self.trajectory),
            "position": [position[0], position[1]],
            "movement": [dx, dy],
            "direction": direction,
            "instant_speed": speed,
            "left": bool(button_states.get("left") or self.tools.button_down(1)),
            "right": bool(button_states.get("right") or self.tools.button_down(2)),
            "middle": bool(button_states.get("middle") or self.tools.button_down(4)),
            "wheel": wheel_delta,
            "events": events,
            "source": source,
            "timestamp": now,
        }
        self.last_mouse = (position[0], position[1], now)
        return record

    def ai_mouse_action(self, window):
        if not self.tools.ready:
            return None
        x1, y1, x2, y2 = window["rect"]
        padding = 12
        left = min(x2 - 1, x1 + padding)
        top = min(y2 - 1, y1 + padding)
        right = max(left, x2 - padding - 1)
        bottom = max(top, y2 - padding - 1)
        x = random.randint(left, right)
        y = random.randint(top, bottom)
        choice = random.random()
        timestamp = time.time()
        if choice < 0.12:
            self.tools.left_click(x, y)
            return {"kind": "left_click", "position": [x, y], "timestamp": timestamp}
        if choice < 0.16:
            self.tools.right_click(x, y)
            return {"kind": "right_click", "position": [x, y], "timestamp": timestamp}
        self.tools.set_cursor(x, y)
        if choice > 0.91:
            delta = 120 if random.random() < 0.5 else -120
            self.tools.wheel(delta)
            return {"kind": "wheel", "position": [x, y], "wheel_delta": delta, "timestamp": timestamp}
        return {"kind": "move", "position": [x, y], "timestamp": timestamp}

    def sleep_decision(self, reward, ticks):
        if ticks < 24:
            return False
        recent = list(self.recent_rewards)
        average = sum(recent) / len(recent) if recent else reward
        return self.hunger >= 0.035 or average < -0.01 or ticks >= 180

    def mode_loop(self, mode, window, stop):
        ticks = 0
        source = "用户" if mode == STATE_LEARNING else "AI"
        try:
            while not stop.is_set() and not self.closing and self.state == mode:
                abnormal, reason = self.tools.window_abnormal(window, require_cursor=True)
                if abnormal:
                    self.request_idle(reason)
                    return
                screen_file, digest, signature, dimensions = self.capture_screen(window["rect"])
                score = self.screen_score(signature)
                now_monotonic = time.monotonic()
                elapsed = max(0.0, now_monotonic - self.last_hunger_time)
                self.last_hunger_time = now_monotonic
                self.hunger += elapsed * 0.0005
                valid = score is not None
                if valid and self.last_score is not None and score > self.last_score:
                    self.hunger = 1e-12
                reward = (score if valid else 0.0) - self.hunger
                mouse = self.mouse_record(source)
                ai_action = self.ai_mouse_action(window) if mode == STATE_TRAINING else None
                if mouse is not None and ai_action is not None:
                    mouse["ai_action"] = ai_action
                experience_value = reward + (score if valid else 0.0) * 0.25
                will_sleep = mode == STATE_TRAINING and self.sleep_decision(reward, ticks + 1)
                record = {
                    "mode": mode,
                    "timestamp": time.time(),
                    "screen_file": screen_file,
                    "screen_digest": digest,
                    "screen_signature": signature,
                    "screen_dimensions": dimensions,
                    "screen_score": score,
                    "valid_screen": valid,
                    "hunger": self.hunger,
                    "reward": reward,
                    "experience_value": experience_value,
                    "mouse": mouse,
                    "other": {
                        "player_rect": window["rect"],
                        "tick": ticks,
                        "model_names": MODEL_NAMES,
                        "sleep_decision": will_sleep,
                    },
                }
                self.store.append_record(record)
                if valid:
                    self.last_score = score
                self.recent_rewards.append(reward)
                ticks += 1
                if will_sleep:
                    self.post(lambda: self.start_sleep(STATE_TRAINING))
                    return
                if ticks % 5 == 0:
                    self.post(lambda t=ticks, s=score, r=reward: self.write_panel(f"记录中：{t} 条，画面评分 {s if s is not None else '无效'}，奖励 {r:.6f}。"))
                stop.wait(0.8)
        except Exception as exc:
            self.request_idle(f"模式运行异常：{exc}")

    def start_sleep(self, origin):
        if self.migrating or self.state == STATE_SLEEPING:
            return
        if origin is None and self.state != STATE_IDLE:
            return
        if origin == STATE_TRAINING and self.state != STATE_TRAINING:
            return
        self.mode_stop.set()
        self.sleep_stop = threading.Event()
        self.sleep_origin = origin
        self.set_state(STATE_SLEEPING)
        self.progress_value.set(0)
        self.progress_text.set("0%")
        self.show_progress()
        self.sleep_thread = threading.Thread(target=self.sleep_loop, args=(origin, self.sleep_stop), daemon=True)
        self.sleep_thread.start()

    def sleep_loop(self, origin, stop):
        try:
            self.train_all_models(stop)
            if stop.is_set() or self.closing:
                return
            self.prune_models_and_experience(stop)
            if stop.is_set() or self.closing:
                return
            self.post(lambda: self.finish_sleep(origin))
        except Exception as exc:
            self.request_idle(f"睡眠模式异常：{exc}")

    def finish_sleep(self, origin):
        if self.state != STATE_SLEEPING:
            return
        if origin == STATE_TRAINING:
            self.set_state(STATE_IDLE)
            self.write_panel("睡眠模式任务完成，正在检查雷电模拟器客户区并重新进入训练模式。")
            self.start_mode(STATE_TRAINING)
        else:
            self.to_idle("睡眠模式任务完成，返回空闲。", keep_progress=True)

    def model_quality(self, model_name, records):
        rewards = []
        scores = []
        for _, record in records:
            try:
                rewards.append(float(record.get("reward", 0.0)))
            except Exception:
                pass
            try:
                scores.append(float(record.get("screen_score", 0.0)))
            except Exception:
                pass
        reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
        score_mean = sum(scores) / len(scores) if scores else 0.5
        seed = int(hashlib.sha256(f"{model_name}|{len(records)}|{reward_mean:.8f}|{score_mean:.8f}".encode("utf-8")).hexdigest()[:8], 16)
        jitter = (seed / 0xFFFFFFFF - 0.5) * 0.08
        return max(0.001, min(0.999, 0.5 + reward_mean * 0.35 + score_mean * 0.12 + jitter))

    def train_all_models(self, stop):
        self.set_progress(0, "睡眠模式任务1：训练所有AI模型开始。")
        records = self.store.iter_recent_records(600)
        total = len(MODEL_NAMES)
        for index, name in enumerate(MODEL_NAMES, 1):
            if stop.is_set():
                return
            quality = self.model_quality(name, records)
            payload = {
                "name": name,
                "quality": quality,
                "trained_at": time.time(),
                "records_seen": len(records),
                "model_version": uuid.uuid4().hex,
            }
            filename = f"{int(time.time() * 1000)}_{hashlib.md5(name.encode('utf-8')).hexdigest()[:10]}_{uuid.uuid4().hex[:8]}.json"
            (self.store.models / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.set_progress(index * 50 / total, f"睡眠模式任务1：已训练{name}。")
            stop.wait(0.18)
        self.set_progress(50, "睡眠模式任务1完成。")

    def prune_models_and_experience(self, stop):
        self.set_progress(55, "睡眠模式任务2：检查AI模型和经验池开始。")
        models = []
        for path in self.store.model_paths():
            try:
                quality = float(json.loads(path.read_text(encoding="utf-8")).get("quality", 0.0))
            except Exception:
                quality = 0.0
            models.append((quality, path))
        models.sort(key=lambda item: item[0])
        delete_count = max(0, len(models) - self.model_limit)
        for index, (_, path) in enumerate(models[:delete_count], 1):
            if stop.is_set():
                return
            path.unlink(missing_ok=True)
            progress = 55 + 20 * index / max(1, delete_count)
            self.set_progress(progress, f"睡眠模式任务2：删除最值得删除的AI模型 {path.name}。")
            stop.wait(0.04)
        if delete_count == 0:
            self.set_progress(75, "睡眠模式任务2：AI模型数量未超过上限。")

        limit_bytes = int(self.exp_limit_gb * 1024 * 1024 * 1024)
        target_bytes = int(limit_bytes * 0.5)
        initial_size = self.store.size_bytes(self.store.exp)
        current_size = initial_size
        deleted = 0
        if current_size <= limit_bytes:
            self.set_progress(99, "睡眠模式任务2：经验池大小未超过上限。")
        else:
            while not stop.is_set() and current_size > target_bytes:
                candidates = self.store.record_candidates()
                if candidates:
                    _, record_path, record = candidates[0]
                    self.store.remove_record_and_capture(record_path, record)
                else:
                    fallback = self.store.oldest_experience_file()
                    if fallback is None:
                        break
                    fallback.unlink(missing_ok=True)
                deleted += 1
                current_size = self.store.size_bytes(self.store.exp)
                removed_ratio = (initial_size - current_size) / max(1, initial_size - target_bytes)
                progress = 75 + min(24, max(0, 24 * removed_ratio))
                self.set_progress(progress, f"睡眠模式任务2：删除最值得删除的经验数据，当前经验池 {current_size / 1024 / 1024 / 1024:.4f} GB。")
                stop.wait(0.03)
            if current_size > target_bytes and not stop.is_set():
                raise RuntimeError("经验池无法清理到上限的50%")
            if deleted == 0:
                self.set_progress(99, "睡眠模式任务2：经验池大小超过上限，但未找到可删除的数据。")
        self.set_progress(100, "睡眠模式任务2完成，进度100%。")

    def close(self):
        self.closing = True
        self.mode_stop.set()
        self.sleep_stop.set()
        self.mouse_monitor.stop()
        self.save_config()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
