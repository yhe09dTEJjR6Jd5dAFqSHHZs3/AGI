import ctypes
import concurrent.futures
import ast
import collections
import heapq
import hashlib
import io
import datetime
import shutil
import re
from bisect import bisect_right
from dataclasses import dataclass, field
import json
import math
import os
import queue
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import tokenize
import uuid
import zlib
from ctypes import wintypes
from pathlib import Path
from tkinter import Tk, Toplevel, StringVar, DoubleVar, Canvas, Frame, Label, Button, Entry, filedialog, messagebox, simpledialog
from tkinter import ttk

IS_WINDOWS = os.name == "nt"

class NullWinFunction:
    def __init__(self, name=""):
        self.name = str(name)
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return 0

class NullWinLibrary:
    def __init__(self):
        self._functions = {}

    def __getattr__(self, name):
        if name not in self._functions:
            self._functions[name] = NullWinFunction(name)
        return self._functions[name]

class NullWindll:
    def __getattr__(self, name):
        return NullWinLibrary()

if not hasattr(ctypes, "WINFUNCTYPE"):
    ctypes.WINFUNCTYPE = ctypes.CFUNCTYPE
if not hasattr(ctypes, "get_last_error"):
    ctypes.get_last_error = lambda: 0
if not hasattr(ctypes, "windll"):
    ctypes.windll = NullWindll()

if IS_WINDOWS:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    try:
        dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
    except OSError:
        dwmapi = None
else:
    user32 = NullWinLibrary()
    gdi32 = NullWinLibrary()
    kernel32 = NullWinLibrary()
    psapi = NullWinLibrary()
    dwmapi = None

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

WH_MOUSE_LL = 14
WH_KEYBOARD_LL = 13
HC_ACTION = 0
WM_QUIT = 0x0012
SW_RESTORE = 9
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_MOUSEWHEEL = 0x020A
WM_MOUSEHWHEEL = 0x020E
WM_XBUTTONDOWN = 0x020B
WM_XBUTTONUP = 0x020C
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
PM_NOREMOVE = 0
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x01000
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
AI_MOUSE_MARKER = 0x4C445F41495F4D31 if ctypes.sizeof(ctypes.c_void_p) >= 8 else 0x495F4D31
AI_INPUT_SERIAL_LOCK = threading.RLock()
LLMHF_INJECTED = 0x00000001
LLMHF_LOWER_IL_INJECTED = 0x00000002
GA_ROOT = 2
GW_HWNDPREV = 3
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0
TH32CS_SNAPPROCESS = 0x00000002
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
VK_ESCAPE = 0x1B
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_VM_READ = 0x0010
MONITOR_DEFAULTTONULL = 0
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
DWMWA_CLOAKED = 14
EVENT_SYSTEM_FOREGROUND = 0x0003
EVENT_SYSTEM_MINIMIZESTART = 0x0016
EVENT_OBJECT_LOCATIONCHANGE = 0x800B
WINEVENT_OUTOFCONTEXT = 0
OWN_TRANSPARENT_OVERLAY_WINDOWS = set()
OWN_OVERLAY_LOCK = threading.Lock()
STRICT_EXCEPTION_COUNTS = collections.Counter()
STRICT_EXCEPTION_LAST = {}
STRICT_EXCEPTION_RING = collections.deque(maxlen=200)
REWARD_DEFINITION_VERSION = "screen_score_only"
CLIENT_RECORDING_FAILURE_LIMIT = 1
CLIENT_RECORDING_FAILURE_SECONDS = 0.0
STATE_TRANSITION_EVENTS = {
    "idle_click_learning": {("idle", "learning")},
    "idle_click_training": {("idle", "training")},
    "manual_sleep": {("idle", "sleep")},
    "auto_sleep_worth": {("training", "stopping"), ("stopping", "sleep")},
    "esc": {("learning", "stopping"), ("training", "stopping"), ("sleep", "stopping"), ("stopping", "idle")},
    "mouse_outside": {("learning", "stopping"), ("training", "stopping"), ("stopping", "idle")},
    "client_invalid": {("learning", "stopping"), ("training", "stopping"), ("stopping", "idle")},
    "manual_sleep_task2_done": {("sleep", "idle")},
    "auto_sleep_task2_done_resume_training": {("sleep", "training")},
}

def assert_transition(event, source, target):
    event = str(event or "")
    source = str(source or "")
    target = str(target or "")
    if source == target:
        return True
    if event not in STATE_TRANSITION_EVENTS:
        raise AssertionError("unknown_state_event:{}".format(event))
    if (source, target) not in STATE_TRANSITION_EVENTS.get(event, set()):
        raise AssertionError("illegal_state_transition:{}:{}->{}".format(event, source, target))
    if source == "sleep" and target == "idle" and event != "manual_sleep_task2_done":
        raise AssertionError("sleep_idle_requires_manual_task2")
    if event == "auto_sleep_task2_done_resume_training" and (source, target) != ("sleep", "training"):
        raise AssertionError("auto_sleep_task2_must_resume_training")
    return True


APP_NAME = "LDTrainingPanel"

def default_storage_path():
    if sys.platform.startswith("win"):
        return r"C:\Users\Administrator\Desktop\AAA"
    return str(Path.home() / "Desktop" / "AAA")

def _path_has_symlink(path):
    try:
        current = Path(path).expanduser()
        parts = current.parts
        if not parts:
            return False
        probe = Path(parts[0])
        start = 1
        if current.is_absolute() and str(probe) == os.sep:
            start = 1
        for part in parts[start:]:
            probe = probe / part
            try:
                if probe.exists() and probe.is_symlink():
                    return True
            except OSError:
                return True
        return False
    except Exception:
        return True

def _reject_symlink_ancestor(path):
    current = Path(path).expanduser()
    candidates = []
    while True:
        candidates.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for item in reversed(candidates):
        try:
            if item.exists() and (item.is_symlink() or _is_windows_reparse_point(item)):
                raise OSError("存储路径祖先链包含符号链接或重解析点：" + str(item))
        except OSError:
            raise
        except Exception as error:
            raise OSError("无法校验存储路径祖先链：" + str(error))

def _atomic_write_bytes(path, payload, storage_root=None):
    target = Path(path).expanduser()
    root = Path(storage_root).expanduser() if storage_root is not None else target.parent
    checked_storage_path(target, root)
    checked_storage_path(target.parent, root).mkdir(parents=True, exist_ok=True)
    tmp = checked_storage_path(target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp"), root)
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)
    checked_storage_path(target, root)
    try:
        fd = os.open(str(target.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        if os.name == "nt":
            handle = kernel32.CreateFileW(str(target.parent), 0x80000000, 0x00000001 | 0x00000002 | 0x00000004, None, 3, 0x02000000, None)
            if handle and handle != INVALID_HANDLE_VALUE:
                try:
                    kernel32.FlushFileBuffers(handle)
                finally:
                    kernel32.CloseHandle(handle)
    return target

class PlatformBackend:
    name = "null"
    learning_training_enabled = False
    disabled_reason = "当前平台暂不支持完整客户区截图、鼠标绑定和输入注入，已禁用学习/训练模式。"

    def list_windows(self):
        return []

    def activate_window(self, hwnd):
        return False

    def client_rect(self, hwnd):
        return None

    def move_cursor_into_client(self, hwnd, rect):
        return False

    def capture_client(self, hwnd, rect):
        return None

    def install_mouse_hook(self):
        return False

    def install_keyboard_hook(self):
        return False

    def inject_mouse(self, action):
        return False

    def resource_snapshot(self):
        return {"platform": self.name, "learning_training_enabled": self.learning_training_enabled, "disabled_reason": self.disabled_reason}

    def mode_unavailable_reason(self):
        return self.disabled_reason

class WindowsBackend(PlatformBackend):
    name = "windows"
    learning_training_enabled = True
    disabled_reason = ""

    def list_windows(self):
        return find_emulator_window_candidates("") if "find_emulator_window_candidates" in globals() else []

    def activate_window(self, hwnd):
        return activate_root_window(hwnd) if "activate_root_window" in globals() else False

    def client_rect(self, hwnd):
        return client_rect(hwnd) if "client_rect" in globals() else None

    def move_cursor_into_client(self, hwnd, rect):
        if not rect:
            return False
        x = int((rect[0] + rect[2]) / 2)
        y = int((rect[1] + rect[3]) / 2)
        return bool(user32.SetCursorPos(x, y))

    def capture_client(self, hwnd, rect):
        if not rect or "capture_client" not in globals():
            return None
        return capture_client(hwnd, rect[2] - rect[0], rect[3] - rect[1])

    def install_mouse_hook(self):
        return True

    def install_keyboard_hook(self):
        return True

    def inject_mouse(self, action):
        return bool(action)

class LinuxBackend(PlatformBackend):
    name = "linux"

    def __init__(self):
        self.disabled_reason = ""
        self.display = None
        self.root = 0
        self.x11 = None
        self.xtst = None
        self._mouse_threads = []
        self._keyboard_threads = []
        self._last_buttons = 0
        self._init_x11()

    def _init_x11(self):
        session_type = str(os.environ.get("XDG_SESSION_TYPE", "")).lower()
        if session_type == "wayland" and not os.environ.get("DISPLAY"):
            self.learning_training_enabled = False
            self.disabled_reason = "Linux Wayland 当前不可用：Wayland 不允许普通进程全局截图、监听和输入注入；请使用 X11 会话或实现门户/辅助服务。"
            return False
        try:
            self.x11 = ctypes.CDLL("libX11.so.6")
            self.xtst = ctypes.CDLL("libXtst.so.6")
            self.x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
            self.x11.XOpenDisplay.restype = ctypes.c_void_p
            self.x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
            self.x11.XDefaultRootWindow.restype = ctypes.c_ulong
            self.x11.XFlush.argtypes = [ctypes.c_void_p]
            self.x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
            self.display = self.x11.XOpenDisplay(None)
            if not self.display:
                raise RuntimeError("无法打开 X11 DISPLAY")
            self.root = int(self.x11.XDefaultRootWindow(self.display))
            self.learning_training_enabled = True
            self.disabled_reason = ""
            return True
        except Exception as error:
            self.learning_training_enabled = False
            self.disabled_reason = "Linux X11 后端不可用：" + str(error)
            return False

    def _atom(self, name):
        self.x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self.x11.XInternAtom.restype = ctypes.c_ulong
        return int(self.x11.XInternAtom(self.display, str(name).encode("utf-8"), 1))

    def _window_property(self, hwnd, name):
        atom = self._atom(name)
        if not atom:
            return None
        actual_type = ctypes.c_ulong()
        actual_format = ctypes.c_int()
        nitems = ctypes.c_ulong()
        bytes_after = ctypes.c_ulong()
        prop = ctypes.c_void_p()
        self.x11.XGetWindowProperty.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_void_p)]
        status = self.x11.XGetWindowProperty(self.display, ctypes.c_ulong(int(hwnd)), ctypes.c_ulong(atom), 0, 65536, 0, 0, ctypes.byref(actual_type), ctypes.byref(actual_format), ctypes.byref(nitems), ctypes.byref(bytes_after), ctypes.byref(prop))
        if status != 0 or not prop.value:
            return None
        try:
            count = int(nitems.value)
            fmt = int(actual_format.value)
            if fmt == 8:
                return ctypes.string_at(prop, count)
            if fmt == 32:
                array_type = ctypes.c_ulong * count
                return list(array_type.from_address(prop.value))
            if fmt == 16:
                array_type = ctypes.c_ushort * count
                return list(array_type.from_address(prop.value))
            return None
        finally:
            try:
                self.x11.XFree(prop)
            except Exception:
                pass

    def _title(self, hwnd):
        for name in ("_NET_WM_NAME", "WM_NAME"):
            value = self._window_property(hwnd, name)
            if isinstance(value, bytes):
                try:
                    return value.decode("utf-8", "ignore").strip("\x00")
                except Exception:
                    pass
        return ""

    def _pid(self, hwnd):
        value = self._window_property(hwnd, "_NET_WM_PID")
        if isinstance(value, list) and value:
            return int(value[0])
        return 0

    def _rect(self, hwnd):
        class XWindowAttributes(ctypes.Structure):
            _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int), ("width", ctypes.c_int), ("height", ctypes.c_int), ("border_width", ctypes.c_int), ("depth", ctypes.c_int), ("visual", ctypes.c_void_p), ("root", ctypes.c_ulong), ("class", ctypes.c_int), ("bit_gravity", ctypes.c_int), ("win_gravity", ctypes.c_int), ("backing_store", ctypes.c_int), ("backing_planes", ctypes.c_ulong), ("backing_pixel", ctypes.c_ulong), ("save_under", ctypes.c_int), ("colormap", ctypes.c_ulong), ("map_installed", ctypes.c_int), ("map_state", ctypes.c_int), ("all_event_masks", ctypes.c_long), ("your_event_mask", ctypes.c_long), ("do_not_propagate_mask", ctypes.c_long), ("override_redirect", ctypes.c_int), ("screen", ctypes.c_void_p)]
        attrs = XWindowAttributes()
        self.x11.XGetWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(XWindowAttributes)]
        if not self.x11.XGetWindowAttributes(self.display, ctypes.c_ulong(int(hwnd)), ctypes.byref(attrs)):
            return None
        if attrs.width <= 0 or attrs.height <= 0 or int(attrs.map_state) == 0:
            return None
        rx = ctypes.c_int()
        ry = ctypes.c_int()
        child = ctypes.c_ulong()
        self.x11.XTranslateCoordinates.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong)]
        if not self.x11.XTranslateCoordinates(self.display, ctypes.c_ulong(int(hwnd)), ctypes.c_ulong(self.root), 0, 0, ctypes.byref(rx), ctypes.byref(ry), ctypes.byref(child)):
            return None
        return (int(rx.value), int(ry.value), int(rx.value + attrs.width), int(ry.value + attrs.height))

    def list_windows(self):
        if not self.learning_training_enabled:
            return []
        client_list = self._window_property(self.root, "_NET_CLIENT_LIST") or self._window_property(self.root, "_NET_CLIENT_LIST_STACKING") or []
        result = []
        for hwnd in client_list:
            rect = self._rect(hwnd)
            title = self._title(hwnd)
            if rect is None or rect[2] - rect[0] < 96 or rect[3] - rect[1] < 96:
                continue
            pid = self._pid(hwnd)
            path = ""
            if pid:
                try:
                    path = os.readlink("/proc/{}/exe".format(pid))
                except Exception:
                    path = ""
            if title or path:
                result.append({"hwnd": int(hwnd), "title": title or Path(path).name, "pid": int(pid), "path": path, "rect": rect, "area": (rect[2]-rect[0])*(rect[3]-rect[1]), "class": "X11"})
        return sorted(result, key=lambda item: (item["title"].lower(), item["pid"]))

    def activate_window(self, hwnd):
        try:
            self.x11.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            self.x11.XSetInputFocus.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            self.x11.XRaiseWindow(self.display, ctypes.c_ulong(int(hwnd)))
            self.x11.XSetInputFocus(self.display, ctypes.c_ulong(int(hwnd)), 1, 0)
            self.x11.XFlush(self.display)
            return True
        except Exception as error:
            self.disabled_reason = "X11 激活窗口失败：" + str(error)
            return False

    def client_rect(self, hwnd):
        return self._rect(hwnd)

    def cursor_position(self):
        root_return = ctypes.c_ulong()
        child_return = ctypes.c_ulong()
        root_x = ctypes.c_int()
        root_y = ctypes.c_int()
        win_x = ctypes.c_int()
        win_y = ctypes.c_int()
        mask = ctypes.c_uint()
        self.x11.XQueryPointer.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint)]
        if self.display and self.x11.XQueryPointer(self.display, ctypes.c_ulong(self.root), ctypes.byref(root_return), ctypes.byref(child_return), ctypes.byref(root_x), ctypes.byref(root_y), ctypes.byref(win_x), ctypes.byref(win_y), ctypes.byref(mask)):
            return (int(root_x.value), int(root_y.value))
        return (0, 0)

    def pointer_state(self):
        root_return = ctypes.c_ulong()
        child_return = ctypes.c_ulong()
        root_x = ctypes.c_int()
        root_y = ctypes.c_int()
        win_x = ctypes.c_int()
        win_y = ctypes.c_int()
        mask = ctypes.c_uint()
        if self.display and self.x11.XQueryPointer(self.display, ctypes.c_ulong(self.root), ctypes.byref(root_return), ctypes.byref(child_return), ctypes.byref(root_x), ctypes.byref(root_y), ctypes.byref(win_x), ctypes.byref(win_y), ctypes.byref(mask)):
            return int(root_x.value), int(root_y.value), int(mask.value)
        return 0, 0, 0

    def escape_pressed(self):
        try:
            self.x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            self.x11.XKeysymToKeycode.restype = ctypes.c_uint
            code = int(self.x11.XKeysymToKeycode(self.display, 0xff1b))
            if code <= 0:
                return False
            buffer = (ctypes.c_char * 32)()
            self.x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
            self.x11.XQueryKeymap(self.display, buffer)
            value = buffer[code // 8]
            value = value[0] if isinstance(value, bytes) else int(value)
            return bool(value & (1 << (code % 8)))
        except Exception:
            return False

    def move_cursor_into_client(self, hwnd, rect):
        if not rect:
            return False
        return self.inject_mouse({"action": "move", "x": int((rect[0]+rect[2])//2), "y": int((rect[1]+rect[3])//2)})

    def capture_client(self, hwnd, rect):
        if not rect:
            return None
        try:
            width = max(1, int(rect[2]) - int(rect[0]))
            height = max(1, int(rect[3]) - int(rect[1]))
            capture_started_ns = time.monotonic_ns()
            capture_started = time.time()
            self.x11.XGetImage.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_int]
            self.x11.XGetImage.restype = ctypes.c_void_p
            image_ptr = self.x11.XGetImage(self.display, ctypes.c_ulong(self.root), int(rect[0]), int(rect[1]), width, height, 0xFFFFFFFF, 2)
            if not image_ptr:
                raise RuntimeError("XGetImage 返回空")
            class XImage(ctypes.Structure):
                _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int), ("xoffset", ctypes.c_int), ("format", ctypes.c_int), ("data", ctypes.c_char_p), ("byte_order", ctypes.c_int), ("bitmap_unit", ctypes.c_int), ("bitmap_bit_order", ctypes.c_int), ("bitmap_pad", ctypes.c_int), ("depth", ctypes.c_int), ("bytes_per_line", ctypes.c_int), ("bits_per_pixel", ctypes.c_int), ("red_mask", ctypes.c_ulong), ("green_mask", ctypes.c_ulong), ("blue_mask", ctypes.c_ulong)]
            try:
                img = ctypes.cast(image_ptr, ctypes.POINTER(XImage)).contents
                stride = int(img.bytes_per_line)
                bpp = max(1, int(img.bits_per_pixel) // 8)
                raw = ctypes.string_at(img.data, stride * height)
                bgra = bytearray(width * height * 4)
                for y in range(height):
                    src_row = y * stride
                    dst_row = y * width * 4
                    for x in range(width):
                        src = src_row + x * bpp
                        dst = dst_row + x * 4
                        if bpp >= 4:
                            bgra[dst:dst+4] = raw[src:src+4]
                        elif bpp >= 3:
                            bgra[dst:dst+3] = raw[src:src+3]
                            bgra[dst+3] = 0
                finished_ns = time.monotonic_ns()
                return {"width": width, "height": height, "bgra": bytes(bgra), "pixel_format": "BGRA", "capture_started_monotonic_ns": capture_started_ns, "capture_finished_monotonic_ns": finished_ns, "capture_started": capture_started, "capture_finished": time.time(), "capture_backend": "x11", "capture_elapsed_ms": (finished_ns-capture_started_ns)/1000000.0, "capture_complete": 1}
            finally:
                try:
                    self.x11.XDestroyImage(ctypes.c_void_p(image_ptr))
                except Exception:
                    pass
        except Exception as error:
            self.disabled_reason = "X11 客户区截图失败：" + str(error)
            return None

    def install_mouse_hook(self):
        return self.learning_training_enabled

    def install_keyboard_hook(self):
        return self.learning_training_enabled

    def inject_mouse(self, action):
        try:
            action = action or {}
            kind = str(action.get("action", "move"))
            x = int(action.get("x", 0) or 0)
            y = int(action.get("y", 0) or 0)
            self.xtst.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
            self.xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
            if kind == "move":
                self.xtst.XTestFakeMotionEvent(self.display, -1, x, y, 0)
            elif kind == "left_click":
                self.xtst.XTestFakeButtonEvent(self.display, 1, 1, 0); self.xtst.XTestFakeButtonEvent(self.display, 1, 0, 0)
            elif kind == "right_click":
                self.xtst.XTestFakeButtonEvent(self.display, 3, 1, 0); self.xtst.XTestFakeButtonEvent(self.display, 3, 0, 0)
            elif kind in ("wheel", "wheel_horizontal"):
                delta = int(action.get("delta", 0) or 0)
                if delta == 0:
                    return True
                button = 4 if kind == "wheel" and delta > 0 else 5 if kind == "wheel" else 6 if delta > 0 else 7
                steps = max(1, min(10, abs(delta) // 120 if abs(delta) >= 120 else 1))
                for _ in range(steps):
                    self.xtst.XTestFakeButtonEvent(self.display, button, 1, 0)
                    self.xtst.XTestFakeButtonEvent(self.display, button, 0, 0)
            self.x11.XFlush(self.display)
            return True
        except Exception as error:
            self.disabled_reason = "XTest 输入注入失败：" + str(error)
            return False

class CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

class CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]

class CGRect(ctypes.Structure):
    _fields_ = [("origin", CGPoint), ("size", CGSize)]

class MacOSBackend(PlatformBackend):
    name = "macos"

    def __init__(self):
        self.cg = None
        self.cf = None
        self.learning_training_enabled = False
        self.disabled_reason = "macOS 后端需要辅助功能和屏幕录制权限。"
        self._keys = {}
        self._window_cache = {}
        self._init_quartz()

    def _init_quartz(self):
        try:
            self.cg = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
            self.cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            self.cg.CGWindowListCopyWindowInfo.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
            self.cg.CGWindowListCopyWindowInfo.restype = ctypes.c_void_p
            self.cg.CGWindowListCreateImage.argtypes = [CGRect, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
            self.cg.CGWindowListCreateImage.restype = ctypes.c_void_p
            self.cg.CGImageGetWidth.argtypes = [ctypes.c_void_p]
            self.cg.CGImageGetWidth.restype = ctypes.c_size_t
            self.cg.CGImageGetHeight.argtypes = [ctypes.c_void_p]
            self.cg.CGImageGetHeight.restype = ctypes.c_size_t
            self.cg.CGImageGetBytesPerRow.argtypes = [ctypes.c_void_p]
            self.cg.CGImageGetBytesPerRow.restype = ctypes.c_size_t
            self.cg.CGImageGetDataProvider.argtypes = [ctypes.c_void_p]
            self.cg.CGImageGetDataProvider.restype = ctypes.c_void_p
            self.cg.CGDataProviderCopyData.argtypes = [ctypes.c_void_p]
            self.cg.CGDataProviderCopyData.restype = ctypes.c_void_p
            self.cg.CGEventCreate.argtypes = [ctypes.c_void_p]
            self.cg.CGEventCreate.restype = ctypes.c_void_p
            self.cg.CGEventGetLocation.argtypes = [ctypes.c_void_p]
            self.cg.CGEventGetLocation.restype = CGPoint
            self.cg.CGEventCreateMouseEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, CGPoint, ctypes.c_uint32]
            self.cg.CGEventCreateMouseEvent.restype = ctypes.c_void_p
            self.cg.CGEventCreateScrollWheelEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32]
            self.cg.CGEventCreateScrollWheelEvent.restype = ctypes.c_void_p
            self.cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
            self.cg.CGEventSourceButtonState.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
            self.cg.CGEventSourceButtonState.restype = ctypes.c_bool
            self.cg.CGEventSourceKeyState.argtypes = [ctypes.c_uint32, ctypes.c_uint16]
            self.cg.CGEventSourceKeyState.restype = ctypes.c_bool
            self.cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
            self.cf.CFArrayGetCount.restype = ctypes.c_long
            self.cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
            self.cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
            self.cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.cf.CFDictionaryGetValue.restype = ctypes.c_void_p
            self.cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
            self.cf.CFStringCreateWithCString.restype = ctypes.c_void_p
            self.cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
            self.cf.CFStringGetCString.restype = ctypes.c_bool
            self.cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            self.cf.CFNumberGetValue.restype = ctypes.c_bool
            self.cf.CFDataGetLength.argtypes = [ctypes.c_void_p]
            self.cf.CFDataGetLength.restype = ctypes.c_long
            self.cf.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
            self.cf.CFDataGetBytePtr.restype = ctypes.POINTER(ctypes.c_ubyte)
            self.cf.CFRelease.argtypes = [ctypes.c_void_p]
            self.learning_training_enabled = True
            self.disabled_reason = ""
            return True
        except Exception as error:
            self.learning_training_enabled = False
            self.disabled_reason = "macOS Quartz 后端不可用：" + str(error) + "；请授予辅助功能和屏幕录制权限。"
            return False

    def _cf_key(self, name):
        if name not in self._keys:
            self._keys[name] = self.cf.CFStringCreateWithCString(None, str(name).encode("utf-8"), 0x08000100)
        return self._keys[name]

    def _dict_value(self, dictionary, name):
        if not dictionary:
            return None
        return self.cf.CFDictionaryGetValue(dictionary, self._cf_key(name))

    def _number(self, value, default=0):
        if not value:
            return default
        integer = ctypes.c_longlong()
        if self.cf.CFNumberGetValue(value, 4, ctypes.byref(integer)):
            return int(integer.value)
        floating = ctypes.c_double()
        if self.cf.CFNumberGetValue(value, 6, ctypes.byref(floating)):
            return float(floating.value)
        return default

    def _string(self, value):
        if not value:
            return ""
        buffer = ctypes.create_string_buffer(4096)
        if self.cf.CFStringGetCString(value, buffer, len(buffer), 0x08000100):
            return buffer.value.decode("utf-8", "ignore")
        return ""

    def _bounds(self, value):
        if not value:
            return None
        x = self._number(self._dict_value(value, "X"), 0)
        y = self._number(self._dict_value(value, "Y"), 0)
        width = self._number(self._dict_value(value, "Width"), 0)
        height = self._number(self._dict_value(value, "Height"), 0)
        if width <= 0 or height <= 0:
            return None
        return (int(round(x)), int(round(y)), int(round(x + width)), int(round(y + height)))

    def list_windows(self):
        if not self.learning_training_enabled:
            return []
        info = self.cg.CGWindowListCopyWindowInfo(1, 0)
        if not info:
            self.disabled_reason = "macOS 未返回可见窗口；请授予屏幕录制权限。"
            return []
        result = []
        try:
            count = int(self.cf.CFArrayGetCount(info))
            for index in range(count):
                item = self.cf.CFArrayGetValueAtIndex(info, index)
                if not item:
                    continue
                layer = self._number(self._dict_value(item, "kCGWindowLayer"), 0)
                if int(layer) != 0:
                    continue
                hwnd = self._number(self._dict_value(item, "kCGWindowNumber"), 0)
                pid = self._number(self._dict_value(item, "kCGWindowOwnerPID"), 0)
                title = self._string(self._dict_value(item, "kCGWindowName")) or self._string(self._dict_value(item, "kCGWindowOwnerName"))
                rect = self._bounds(self._dict_value(item, "kCGWindowBounds"))
                if not hwnd or rect is None or rect[2] - rect[0] < 96 or rect[3] - rect[1] < 96:
                    continue
                path = ""
                if pid:
                    try:
                        command = 'tell application "System Events" to get POSIX path of application file of first process whose unix id is {}'.format(int(pid))
                        path = subprocess.run(["osascript", "-e", command], capture_output=True, text=True, timeout=2).stdout.strip()
                    except Exception:
                        path = ""
                record = {"hwnd": int(hwnd), "title": title or Path(path).name, "pid": int(pid), "path": path, "rect": rect, "area": (rect[2]-rect[0])*(rect[3]-rect[1]), "class": "Quartz"}
                result.append(record)
                self._window_cache[int(hwnd)] = record
        finally:
            self.cf.CFRelease(info)
        return sorted(result, key=lambda row: (0 if is_ld_window_candidate(row.get("path", ""), row.get("title", "")) else 1, -int(row.get("area", 0)), str(row.get("title", "")).lower()))

    def activate_window(self, hwnd):
        item = self._window_cache.get(int(hwnd or 0))
        if item is None:
            for row in self.list_windows():
                if int(row.get("hwnd", 0) or 0) == int(hwnd or 0):
                    item = row
                    break
        pid = int((item or {}).get("pid", 0) or 0)
        if pid <= 0:
            self.disabled_reason = "macOS 无法定位目标窗口所属进程。"
            return False
        try:
            command = 'tell application "System Events" to set frontmost of first process whose unix id is {} to true'.format(pid)
            completed = subprocess.run(["osascript", "-e", command], capture_output=True, text=True, timeout=3)
            ok = completed.returncode == 0
            self.disabled_reason = "" if ok else "macOS 激活窗口失败：" + (completed.stderr.strip() or completed.stdout.strip())
            return ok
        except Exception as error:
            self.disabled_reason = "macOS 激活窗口失败：" + str(error)
            return False

    def client_rect(self, hwnd):
        item = self._window_cache.get(int(hwnd or 0))
        if item is not None:
            return item.get("rect")
        for row in self.list_windows():
            if int(row.get("hwnd", 0) or 0) == int(hwnd or 0):
                return row.get("rect")
        return None

    def cursor_position(self):
        event = self.cg.CGEventCreate(None)
        if not event:
            return (0, 0)
        try:
            point = self.cg.CGEventGetLocation(event)
            return (int(round(point.x)), int(round(point.y)))
        finally:
            self.cf.CFRelease(event)

    def pointer_state(self):
        x, y = self.cursor_position()
        mask = 0
        try:
            if self.cg.CGEventSourceButtonState(1, 0):
                mask |= 1 << 8
            if self.cg.CGEventSourceButtonState(1, 1):
                mask |= 1 << 10
        except Exception:
            pass
        return x, y, mask

    def escape_pressed(self):
        try:
            return bool(self.cg.CGEventSourceKeyState(1, 53))
        except Exception:
            return False

    def move_cursor_into_client(self, hwnd, rect):
        if not rect:
            return False
        return self.inject_mouse({"action": "move", "x": int((rect[0]+rect[2])//2), "y": int((rect[1]+rect[3])//2)})

    def capture_client(self, hwnd, rect):
        if not rect:
            return None
        try:
            width = max(1, int(rect[2]) - int(rect[0]))
            height = max(1, int(rect[3]) - int(rect[1]))
            capture_started_ns = time.monotonic_ns()
            capture_started = time.time()
            cg_rect = CGRect(CGPoint(float(rect[0]), float(rect[1])), CGSize(float(width), float(height)))
            image = self.cg.CGWindowListCreateImage(cg_rect, 8, int(hwnd), 1)
            if not image:
                raise RuntimeError("CGWindowListCreateImage 返回空；请授予屏幕录制权限")
            data = None
            try:
                actual_width = int(self.cg.CGImageGetWidth(image))
                actual_height = int(self.cg.CGImageGetHeight(image))
                stride = int(self.cg.CGImageGetBytesPerRow(image))
                provider = self.cg.CGImageGetDataProvider(image)
                data = self.cg.CGDataProviderCopyData(provider) if provider else None
                if not data:
                    raise RuntimeError("CGDataProviderCopyData 返回空")
                length = int(self.cf.CFDataGetLength(data))
                pointer = self.cf.CFDataGetBytePtr(data)
                raw = ctypes.string_at(pointer, length)
                bgra = bytearray(actual_width * actual_height * 4)
                for y in range(actual_height):
                    source_offset = y * stride
                    target_offset = y * actual_width * 4
                    bgra[target_offset:target_offset + actual_width * 4] = raw[source_offset:source_offset + actual_width * 4]
                finished_ns = time.monotonic_ns()
                return {"width": actual_width, "height": actual_height, "bgra": bytes(bgra), "pixel_format": "BGRA", "capture_started_monotonic_ns": capture_started_ns, "capture_finished_monotonic_ns": finished_ns, "capture_started": capture_started, "capture_finished": time.time(), "capture_backend": "quartz", "capture_elapsed_ms": (finished_ns-capture_started_ns)/1000000.0, "capture_complete": 1}
            finally:
                if data:
                    self.cf.CFRelease(data)
                self.cf.CFRelease(image)
        except Exception as error:
            self.disabled_reason = "macOS 客户区截图失败：" + str(error)
            return None

    def install_mouse_hook(self):
        return self.learning_training_enabled

    def install_keyboard_hook(self):
        return self.learning_training_enabled

    def inject_mouse(self, action):
        try:
            action = action or {}
            kind = str(action.get("action", "move"))
            x = int(action.get("x", 0) or 0)
            y = int(action.get("y", 0) or 0)
            if kind == "move":
                event = self.cg.CGEventCreateMouseEvent(None, 5, CGPoint(float(x), float(y)), 0)
                self.cg.CGEventPost(0, event)
                self.cf.CFRelease(event)
                return True
            if kind == "left_click":
                x, y = self.cursor_position()
                for event_type in (1, 2):
                    event = self.cg.CGEventCreateMouseEvent(None, event_type, CGPoint(float(x), float(y)), 0)
                    self.cg.CGEventPost(0, event)
                    self.cf.CFRelease(event)
                return True
            if kind == "right_click":
                x, y = self.cursor_position()
                for event_type in (3, 4):
                    event = self.cg.CGEventCreateMouseEvent(None, event_type, CGPoint(float(x), float(y)), 1)
                    self.cg.CGEventPost(0, event)
                    self.cf.CFRelease(event)
                return True
            if kind in ("wheel", "wheel_horizontal") and hasattr(self.cg, "CGEventCreateScrollWheelEvent"):
                delta = int(action.get("delta", 0) or 0)
                if delta == 0:
                    return True
                units = max(-10, min(10, int(round(delta / 120.0)) or (1 if delta > 0 else -1)))
                event = self.cg.CGEventCreateScrollWheelEvent(None, 0, 2, units if kind == "wheel" else 0, units if kind == "wheel_horizontal" else 0)
                if not event:
                    return False
                self.cg.CGEventPost(0, event)
                self.cf.CFRelease(event)
                return True
            return False
        except Exception as error:
            self.disabled_reason = "CGEventPost 输入注入失败：" + str(error) + "；请授予辅助功能权限。"
            return False

def make_platform_backend():
    if sys.platform.startswith("win"):
        return WindowsBackend()
    if sys.platform.startswith("linux"):
        return LinuxBackend()
    if sys.platform == "darwin":
        return MacOSBackend()
    return PlatformBackend()

PLATFORM_BACKEND = make_platform_backend()

def note_strict_exception(area, error, payload=None):
    try:
        key = str(area or "unknown")
        item = {"area": key, "exception_type": type(error).__name__, "message": str(error), "payload": dict(payload or {}), "created": time.time()}
        STRICT_EXCEPTION_COUNTS[key] += 1
        STRICT_EXCEPTION_LAST[key] = item
        STRICT_EXCEPTION_RING.appendleft(item)
        return item
    except Exception:
        return None

ULONG_PTR = ctypes.c_size_t
LRESULT = ctypes.c_ssize_t

class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG), ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

class MSG(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND), ("message", wintypes.UINT), ("wParam", wintypes.WPARAM), ("lParam", wintypes.LPARAM), ("time", wintypes.DWORD), ("pt", POINT), ("lPrivate", wintypes.DWORD)]

class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]

class RGBQUAD(ctypes.Structure):
    _fields_ = [("rgbBlue", ctypes.c_ubyte), ("rgbGreen", ctypes.c_ubyte), ("rgbRed", ctypes.c_ubyte), ("rgbReserved", ctypes.c_ubyte)]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", RGBQUAD)]

class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("pt", POINT), ("mouseData", wintypes.DWORD), ("flags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

class INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]

class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [("type", wintypes.DWORD), ("union", INPUTUNION)]

class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ULONG_PTR), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT), ("rcWork", RECT), ("dwFlags", wintypes.DWORD)]

class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]

PDH_FMT_DOUBLE = 0x00000200

LowLevelMouseProc = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
LowLevelKeyboardProc = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
MonitorEnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(RECT), wintypes.LPARAM)
WinEventProc = ctypes.WINFUNCTYPE(None, wintypes.HANDLE, wintypes.DWORD, wintypes.HWND, ctypes.c_long, ctypes.c_long, wintypes.DWORD, wintypes.DWORD)

user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
user32.GetCursorPos.restype = wintypes.BOOL
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = wintypes.BOOL
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
user32.GetClientRect.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(POINT)]
user32.ClientToScreen.restype = wintypes.BOOL
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsIconic.restype = wintypes.BOOL
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL
WindowFromPoint = user32.WindowFromPoint
WindowFromPoint.argtypes = [POINT]
WindowFromPoint.restype = wintypes.HWND
user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetAncestor.restype = wintypes.HWND
user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetWindow.restype = wintypes.HWND
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = wintypes.LONG
user32.EnumWindows.argtypes = [EnumWindowsProc, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, ctypes.c_void_p, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
user32.UnhookWindowsHookEx.restype = wintypes.BOOL
user32.SetWinEventHook.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.HMODULE, WinEventProc, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD]
user32.SetWinEventHook.restype = wintypes.HANDLE
user32.UnhookWinEvent.argtypes = [wintypes.HANDLE]
user32.UnhookWinEvent.restype = wintypes.BOOL
user32.GetMessageW.argtypes = [ctypes.POINTER(MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
user32.GetMessageW.restype = ctypes.c_int
user32.PeekMessageW.argtypes = [ctypes.POINTER(MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
user32.PeekMessageW.restype = wintypes.BOOL
user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostThreadMessageW.restype = wintypes.BOOL
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.ReleaseDC.restype = ctypes.c_int
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT

kernel32.GetCurrentProcess.argtypes = []
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME)]
kernel32.GetProcessTimes.restype = wintypes.BOOL
kernel32.GetLastError.argtypes = []
kernel32.GetLastError.restype = wintypes.DWORD
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE
kernel32.GetCurrentThreadId.argtypes = []
kernel32.GetCurrentThreadId.restype = wintypes.DWORD
kernel32.GetSystemTimes.argtypes = [ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME)]
kernel32.GetSystemTimes.restype = wintypes.BOOL
kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.Process32FirstW.restype = wintypes.BOOL
kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.Process32NextW.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
kernel32.CreateFileW.restype = wintypes.HANDLE
kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
kernel32.FlushFileBuffers.restype = wintypes.BOOL
user32.EnumDisplayMonitors.argtypes = [wintypes.HDC, ctypes.POINTER(RECT), MonitorEnumProc, wintypes.LPARAM]
user32.EnumDisplayMonitors.restype = wintypes.BOOL
user32.MonitorFromRect.argtypes = [ctypes.POINTER(RECT), wintypes.DWORD]
user32.MonitorFromRect.restype = wintypes.HMONITOR
user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]
user32.GetMonitorInfoW.restype = wintypes.BOOL
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
if dwmapi is not None:
    dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long

gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteObject.restype = wintypes.BOOL
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteDC.restype = wintypes.BOOL
gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
gdi32.BitBlt.restype = wintypes.BOOL
gdi32.StretchBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
gdi32.StretchBlt.restype = wintypes.BOOL
gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.POINTER(BITMAPINFO), wintypes.UINT]
gdi32.GetDIBits.restype = ctypes.c_int


class ByteBudgetSemaphore:
    def __init__(self, capacity_bytes):
        self.lock = threading.Condition()
        self.capacity = max(1, int(capacity_bytes))
        self.used = 0

    def resize(self, capacity_bytes):
        with self.lock:
            self.capacity = max(1, int(capacity_bytes))
            self.lock.notify_all()

    def acquire(self, amount, timeout=0.0):
        amount = max(0, int(amount))
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self.lock:
            while self.used + amount > self.capacity:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.lock.wait(remaining)
            self.used += amount
            return True

    def release(self, amount):
        amount = max(0, int(amount))
        with self.lock:
            self.used = max(0, self.used - amount)
            self.lock.notify_all()

    def snapshot(self):
        with self.lock:
            return {"capacity": self.capacity, "used": self.used, "available": max(0, self.capacity - self.used)}

def feature_bytes(value, expected=None):
    if value is None:
        return b""
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        result = value
    elif isinstance(value, str):
        try:
            result = bytes.fromhex(value)
        except ValueError:
            result = b""
    else:
        result = b""
    if expected is not None and len(result) != int(expected):
        return b""
    return result

def feature_hex(value):
    return feature_bytes(value).hex()

def histogram_values(value):
    if value is None:
        return []
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        if len(value) == 24 * 4:
            try:
                return list(struct.unpack("<24I", value))
            except struct.error:
                return []
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [max(0, int(item)) for item in parsed] if isinstance(parsed, list) else []
        except Exception:
            return []
    if isinstance(value, (list, tuple)):
        return [max(0, int(item)) for item in value]
    return []

def histogram_blob(value):
    items = histogram_values(value)
    if len(items) != 24:
        items = (items + [0] * 24)[:24]
    return struct.pack("<24I", *items)

def histogram_text(value):
    return json.dumps(histogram_values(value), separators=(",", ":"))

def local_visual_descriptor(gray_value, color_histogram, edge_density, rx, ry, radius=3):
    gray = feature_bytes(gray_value, 32 * 18)
    if not gray:
        return {"gray": "", "edge": 0.0, "color": histogram_values(color_histogram), "radius": int(radius), "rx": float(rx), "ry": float(ry)}
    cx = max(0, min(31, int(round(float(rx) * 31))))
    cy = max(0, min(17, int(round(float(ry) * 17))))
    radius = max(1, min(6, int(radius)))
    values = []
    edges = 0.0
    comparisons = 0
    for y in range(max(0, cy - radius), min(18, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(32, cx + radius + 1)):
            value = gray[y * 32 + x]
            values.append(value)
            if x + 1 < 32:
                edges += abs(value - gray[y * 32 + x + 1])
                comparisons += 1
            if y + 1 < 18:
                edges += abs(value - gray[(y + 1) * 32 + x])
                comparisons += 1
    return {"gray": bytes(values).hex(), "edge": max(0.0, min(1.0, edges / max(1.0, comparisons * 255.0))), "color": histogram_values(color_histogram), "radius": radius, "rx": float(rx), "ry": float(ry)}

def local_visual_distance(current, expected):
    current_gray = feature_bytes((current or {}).get("gray"), None)
    expected_gray = feature_bytes((expected or {}).get("gray"), None)
    gray_distance = 1.0
    if current_gray and expected_gray and len(current_gray) == len(expected_gray):
        gray_distance = sum(abs(left - right) for left, right in zip(current_gray, expected_gray)) / (255.0 * len(current_gray))
    try:
        edge_distance = min(1.0, abs(float((current or {}).get("edge", 0.0)) - float((expected or {}).get("edge", 0.0))) * 2.0)
    except (TypeError, ValueError):
        edge_distance = 1.0
    current_hist = histogram_values((current or {}).get("color"))
    expected_hist = histogram_values((expected or {}).get("color"))
    color_distance = 1.0
    if current_hist and expected_hist and len(current_hist) == len(expected_hist) and sum(current_hist) > 0 and sum(expected_hist) > 0:
        left_total = float(sum(current_hist))
        right_total = float(sum(expected_hist))
        color_distance = min(1.0, sum(abs(left / left_total - right / right_total) for left, right in zip(current_hist, expected_hist)) / 2.0)
    return max(0.0, min(1.0, 0.58 * gray_distance + 0.22 * edge_distance + 0.20 * color_distance))

def packet_byte_cost(packet):
    image = packet.get("image", {}) if isinstance(packet, dict) else {}
    total = 2048
    for key in ("bgra", "rgb", "png", "gray32x18", "color_histogram"):
        value = image.get(key) if isinstance(image, dict) else None
        if isinstance(value, memoryview):
            total += len(value)
        elif isinstance(value, (bytes, bytearray, str)):
            total += len(value)
    return max(4096, total + 8192)

def _is_windows_reparse_point(path):
    if os.name != "nt":
        return False
    try:
        attrs = kernel32.GetFileAttributesW(str(Path(path)))
        if attrs == 0xFFFFFFFF:
            return False
        return bool(int(attrs) & 0x0400)
    except Exception:
        return False

def _existing_storage_ancestors(path):
    current = Path(path).expanduser()
    result = []
    while True:
        if current.exists():
            result.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return list(reversed(result))

def storage_path_allowed(path, storage_root):
    try:
        root_input = Path(storage_root).expanduser()
        candidate_input = Path(path).expanduser()
        _reject_symlink_ancestor(root_input)
        for item in _existing_storage_ancestors(candidate_input):
            if item.is_symlink() or _is_windows_reparse_point(item):
                return False
        root = root_input.resolve()
        candidate = candidate_input.resolve()
        if os.path.commonpath([str(root), str(candidate)]) != str(root):
            return False
        if root.exists() and (root.is_symlink() or _is_windows_reparse_point(root)):
            return False
        return True
    except Exception:
        return False

def checked_storage_path(path, storage_root):
    candidate = Path(path).expanduser().resolve()
    if not storage_path_allowed(path, storage_root):
        raise OSError("落盘路径越界或包含 Windows reparse point：{} 不在 {} 内".format(candidate, Path(storage_root).expanduser().resolve()))
    return candidate

@dataclass
class PipelineContext:
    token: int
    session_id: str
    capture_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=32))
    feature_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=32))
    persist_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=32))
    stop_event: threading.Event = field(default_factory=threading.Event)
    capture_done: threading.Event = field(default_factory=threading.Event)
    feature_done: threading.Event = field(default_factory=threading.Event)
    encode_done: threading.Event = field(default_factory=threading.Event)
    persist_done: threading.Event = field(default_factory=threading.Event)
    exact_done: threading.Event = field(default_factory=threading.Event)
    drain_complete: threading.Event = field(default_factory=threading.Event)
    threads: list = field(default_factory=list)
    queue_wait_timeout_seconds: float = 1.0
    queue_capacity: int = 32
    byte_budget: ByteBudgetSemaphore = field(default_factory=lambda: ByteBudgetSemaphore(64 * 1024 * 1024))
    accepting: bool = True
    draining: bool = False
    closed: bool = False
@dataclass
class ResourceBudget:
    allowed: bool
    next_interval: float
    max_batch: int
    cpu_workers: int
    gpu: str
    gpu_batch_size: int
    max_capture_resolution: tuple
    must_pause: bool
    pause_reason: str
    state: str = "正常"
    retrieval_candidate_limit: int = 64
    database_batch_size: int = 32
    training_block_size: int = 32
    inference_concurrency: int = 1
    retrieval_deadline_seconds: float = 0.12
    queue_fill_ratio: float = 0.0

class ResourceBudgetBusy(Exception):
    pass

class ResourceBudgetSlot:
    def __init__(self, governor, name, redline_ok=False):
        self.governor = governor
        self.name = str(name)
        self.redline_ok = bool(redline_ok)
        self.entered = False

    def __enter__(self):
        if not self.governor.try_enter_budget(self.name, self.redline_ok):
            raise ResourceBudgetBusy(self.name)
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.entered:
            self.governor.leave_budget(self.name)
            self.entered = False
        return False

POLICY_ACTION_TYPES = ("移动", "左键", "右键", "滚轮", "水平滚轮")
POLICY_GRID_WIDTH = 10
POLICY_GRID_HEIGHT = 6
POLICY_GRAY_SIZE = 32 * 18
POLICY_INPUT_SIZE = POLICY_GRAY_SIZE + 24 + 1 + 4
POLICY_OUTPUT_SIZE = len(POLICY_ACTION_TYPES) + POLICY_GRID_WIDTH * POLICY_GRID_HEIGHT + 3
CAPTURE_BACKEND_CACHE = {}
CAPTURE_BACKEND_CACHE_LOCK = threading.RLock()

class HardwareProbe:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_probe = 0.0
        self.last_runtime_probe = 0.0
        self.gpus = []
        self.runtime = {"available": False, "source": "GPU 指标尚未初始化", "program_gpu": 0.0, "ldplayer_gpu": 0.0, "gpu_engine": 0.0, "dedicated_used": 0, "dedicated_total": None, "dedicated_free": None}
        self.backend = "CPU 表格策略"
        self._pdh = None
        self._pdh_query = None
        self._pdh_engine = []
        self._pdh_memory = []
        self._pdh_loaded = False
        self._pdh_ready = False
        self._pdh_error = ""
        self._adapter_total = None
        self._adapter_total_checked = 0.0

    def probe(self):
        now = time.monotonic()
        with self.lock:
            if now - self.last_probe < 5.0 and self.gpus:
                return [dict(item) for item in self.gpus]
        runtime = self._collect_pdh(set(), set())
        gpus = self._probe_windows_gpus(runtime)
        with self.lock:
            self.gpus = gpus or [{"name": "Windows GPU 性能计数器", "adapter_type": "未知", "hardware": bool(runtime.get("available")), "software": False, "dedicated_total": runtime.get("dedicated_total"), "dedicated_used": runtime.get("dedicated_used"), "dedicated_free": runtime.get("dedicated_free"), "utilization": runtime.get("gpu_engine"), "engine_utilization": runtime.get("gpu_engine"), "ldplayer": runtime.get("ldplayer_gpu"), "program": runtime.get("program_gpu"), "sampling_source": runtime.get("source")}]
            self.backend = "CPU 表格策略"
            self.last_probe = now
            return [dict(item) for item in self.gpus]

    def snapshot_gpus(self):
        with self.lock:
            return [dict(item) for item in self.gpus]

    def _expand_pdh_paths(self, wildcard):
        if self._pdh is None:
            return []
        size = wintypes.DWORD(0)
        status = self._pdh.PdhExpandWildCardPathW(None, wildcard, None, ctypes.byref(size), 0)
        if status not in (0, 0x800007D2) or size.value <= 1:
            return []
        buffer = ctypes.create_unicode_buffer(size.value)
        status = self._pdh.PdhExpandWildCardPathW(None, wildcard, buffer, ctypes.byref(size), 0)
        if status != 0:
            return []
        return [part for part in buffer[:size.value].split("\x00") if part]

    def _init_pdh(self):
        if self._pdh_ready:
            return True
        if self._pdh_loaded:
            return False
        try:
            pdh = ctypes.WinDLL("pdh", use_last_error=True)
            pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
            pdh.PdhOpenQueryW.restype = wintypes.DWORD
            pdh.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
            pdh.PdhAddEnglishCounterW.restype = wintypes.DWORD
            pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
            pdh.PdhCollectQueryData.restype = wintypes.DWORD
            pdh.PdhGetFormattedCounterValue.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(PDH_FMT_COUNTERVALUE)]
            pdh.PdhGetFormattedCounterValue.restype = wintypes.DWORD
            pdh.PdhExpandWildCardPathW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
            pdh.PdhExpandWildCardPathW.restype = wintypes.DWORD
            query = wintypes.HANDLE()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
                raise OSError("PdhOpenQueryW 失败")
            self._pdh = pdh
            self._pdh_loaded = True
            self._pdh_query = query
            for path in self._expand_pdh_paths(r"\GPU Engine(*)\Utilization Percentage"):
                counter = wintypes.HANDLE()
                if pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(counter)) == 0:
                    self._pdh_engine.append((path, counter))
            for path in self._expand_pdh_paths(r"\GPU Adapter Memory(*)\Dedicated Usage"):
                counter = wintypes.HANDLE()
                if pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(counter)) == 0:
                    self._pdh_memory.append((path, counter))
            self._pdh_ready = bool(self._pdh_engine or self._pdh_memory)
            if self._pdh_ready:
                pdh.PdhCollectQueryData(query)
            else:
                self._pdh_error = "未发现 GPU Engine 或 GPU Adapter Memory 性能计数器"
            return self._pdh_ready
        except Exception as error:
            self._pdh_error = str(error)
            self._pdh = None
            self._pdh_loaded = False
            self._pdh_ready = False
            return False

    def _counter_value(self, counter):
        typ = wintypes.DWORD()
        value = PDH_FMT_COUNTERVALUE()
        if self._pdh.PdhGetFormattedCounterValue(counter, PDH_FMT_DOUBLE, ctypes.byref(typ), ctypes.byref(value)) != 0:
            return 0.0
        return max(0.0, float(value.doubleValue))

    def _dedicated_total_bytes(self):
        now = time.monotonic()
        if now - self._adapter_total_checked < 60.0:
            return self._adapter_total
        self._adapter_total_checked = now
        try:
            command = "Get-CimInstance Win32_VideoController | ForEach-Object { $_.AdapterRAM }"
            completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True, timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            values = [int(item.strip()) for item in completed.stdout.splitlines() if item.strip().isdigit() and int(item.strip()) > 0]
            self._adapter_total = max(values) if values else None
        except Exception:
            self._adapter_total = None
        return self._adapter_total

    def _collect_pdh(self, program_pids, ldplayer_pids):
        program_pids = {int(pid) for pid in (program_pids or set())}
        ldplayer_pids = {int(pid) for pid in (ldplayer_pids or set())}
        if not self._init_pdh():
            return {"available": False, "source": "Windows PDH GPU 性能计数器不可用：" + (self._pdh_error or "未知原因"), "program_gpu": 0.0, "ldplayer_gpu": 0.0, "gpu_engine": 0.0, "dedicated_used": 0, "dedicated_total": None, "dedicated_free": None}
        try:
            self._pdh.PdhCollectQueryData(self._pdh_query)
            time.sleep(0.02)
            self._pdh.PdhCollectQueryData(self._pdh_query)
            program = 0.0
            ldplayer = 0.0
            total = 0.0
            for path, counter in self._pdh_engine:
                value = self._counter_value(counter)
                total += value
                match = re.search(r"pid_(\d+)", path, re.I)
                pid = int(match.group(1)) if match else 0
                if pid in program_pids:
                    program += value
                if pid in ldplayer_pids:
                    ldplayer += value
            dedicated = sum(self._counter_value(counter) for _, counter in self._pdh_memory)
            dedicated_total = self._dedicated_total_bytes()
            dedicated_free = max(0, int(dedicated_total) - int(dedicated)) if dedicated_total is not None else None
            return {"available": True, "source": "Windows PDH GPU Engine / GPU Adapter Memory；AdapterRAM 来自 Win32_VideoController", "program_gpu": min(100.0, program), "ldplayer_gpu": min(100.0, ldplayer), "gpu_engine": min(100.0, total), "dedicated_used": int(dedicated), "dedicated_total": dedicated_total, "dedicated_free": dedicated_free}
        except Exception as error:
            return {"available": False, "source": "Windows PDH GPU 采样失败：" + str(error), "program_gpu": 0.0, "ldplayer_gpu": 0.0, "gpu_engine": 0.0, "dedicated_used": 0, "dedicated_total": None, "dedicated_free": None}

    def _probe_windows_gpus(self, runtime=None):
        runtime = runtime or {}
        if not runtime.get("available"):
            return []
        return [{"name": "Windows GPU 性能计数器", "adapter_type": "PDH", "hardware": True, "software": False, "dedicated_total": runtime.get("dedicated_total"), "dedicated_used": runtime.get("dedicated_used"), "dedicated_free": runtime.get("dedicated_free"), "utilization": runtime.get("gpu_engine"), "engine_utilization": runtime.get("gpu_engine"), "ldplayer": runtime.get("ldplayer_gpu"), "program": runtime.get("program_gpu"), "sampling_source": runtime.get("source")}]

    def runtime_metrics(self, program_pid, ldplayer_pids):
        now = time.monotonic()
        with self.lock:
            if now - self.last_runtime_probe < 1.0:
                return dict(self.runtime)
        result = self._collect_pdh({int(program_pid)}, set(ldplayer_pids or set()))
        result["source"] = str(result.get("source") or "GPU 指标不可用")
        with self.lock:
            self.runtime = result
            self.last_runtime_probe = now
            return dict(result)

    def choose_gpu(self, metrics):
        metrics = metrics or {}
        if not metrics.get("gpu_metrics_available", metrics.get("available", False)):
            return "CPU"
        gpu_p95 = float(metrics.get("gpu_engine_p95", metrics.get("gpu_engine", metrics.get("gpu", 100.0))) or 0.0)
        program_gpu = float(metrics.get("program_gpu_p95", metrics.get("program_gpu", 0.0)) or 0.0)
        free = metrics.get("gpu_dedicated_free", metrics.get("dedicated_free"))
        if gpu_p95 >= 80.0 or program_gpu >= 70.0:
            return "CPU"
        if free is not None and int(free) < 512 * 1024 * 1024:
            return "CPU"
        return "GPU"

class ComputeBackend:
    def __init__(self, probe):
        self.probe = probe
        self.lock = threading.RLock()
        self.executor = None
        self.executor_workers = 0
        self.last_backend = "CPU 表格策略"
        self.runtime_provider = None
        self.runtime_ready = False
        self.model_session = None
        self.encode_slots = threading.Semaphore(1)
        self.encode_metrics_sink = None
        self._png_lock = threading.Lock()
        self._png_active = 0
        self.gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="GpuPolicy")
        self.gpu_inflight = threading.Semaphore(1)
        self.gpu_model_path = None
        self.gpu_warmup_started = 0.0
        self.gpu_retry_after = 0.0
        self.gpu_failure_reason = ""
        self._probe_runtime_provider()

    def _probe_runtime_provider(self):
        try:
            import onnxruntime as ort
            providers = list(ort.get_available_providers())
            self.runtime_provider = next((name for name in ("CUDAExecutionProvider", "DmlExecutionProvider") if name in providers), None)
        except Exception:
            self.runtime_provider = None
        self.runtime_ready = False
        self.model_session = None

    def can_use_gpu(self):
        return bool(self.runtime_ready and self.runtime_provider and self.model_session is not None and time.monotonic() - self.gpu_warmup_started >= 30.0)

    def try_enable_gpu_model(self, model_path, metrics):
        now = time.monotonic()
        if now < self.gpu_retry_after or not self.runtime_provider:
            return False
        path = Path(model_path) if model_path else None
        free = metrics.get("gpu_dedicated_free") if isinstance(metrics, dict) else None
        if free is not None and int(free) < 512 * 1024 * 1024:
            self.gpu_failure_reason = "GPU 可用显存不足"
            self.gpu_retry_after = now + 300.0
            return False
        if path is None or not path.exists() or path.suffix.lower() != ".onnx":
            return False
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(path), providers=[self.runtime_provider, "CPUExecutionProvider"])
            inputs = session.get_inputs()
            if not inputs:
                raise RuntimeError("ONNX 模型没有输入")
            sample = inputs[0]
            shape = [1 if not isinstance(dim, int) or dim <= 0 else min(8, dim) for dim in sample.shape]
            import numpy as np
            session.run(None, {sample.name: np.zeros(shape, dtype=np.float32)})
            with self.lock:
                self.model_session = session
                self.gpu_model_path = str(path)
                self.gpu_warmup_started = now
                self.runtime_ready = False
                self.gpu_failure_reason = ""
            return True
        except Exception as error:
            self.disable_gpu("GPU 模型加载或 warm-up 失败：" + str(error), 300.0)
            return False

    def refresh_gpu_stability(self, metrics):
        with self.lock:
            if self.model_session is None:
                return False
            if time.monotonic() < self.gpu_retry_after:
                self.runtime_ready = False
                return False
            free = metrics.get("gpu_dedicated_free") if isinstance(metrics, dict) else None
            if free is not None and int(free) < 384 * 1024 * 1024:
                self.disable_gpu("GPU 显存余量不足", 300.0)
                return False
            if time.monotonic() - self.gpu_warmup_started >= 30.0:
                self.runtime_ready = True
            return self.runtime_ready

    def _vision_tensor(self, features):
        gray = feature_bytes((features or {}).get("gray32x18"), 32 * 18)
        if not gray:
            gray = bytes([128] * (32 * 18))
        values = [max(0.0, min(1.0, item / 255.0)) for item in gray]
        return [[[[values[y * 32 + x] for x in range(32)] for y in range(18)]]]

    def _policy_tensor(self, features):
        return [visual_policy_input_vector(features)]

    def run_gpu_features(self, features, metrics, timeout=0.06):
        if not self.refresh_gpu_stability(metrics):
            return None
        if not self.gpu_inflight.acquire(timeout=0.0):
            return None
        try:
            import numpy as np
            with self.lock:
                session = self.model_session
                if self.gpu_executor is None:
                    self.gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="GpuPolicy")
                executor = self.gpu_executor
            inputs = session.get_inputs()
            sample = inputs[0]
            rank = len(sample.shape or [])
            array = np.asarray(self._vision_tensor(features) if rank == 4 else self._policy_tensor(features), dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            feed = {sample.name: array}
            for extra in inputs[1:]:
                shape = [1 if not isinstance(dim, int) or dim <= 0 else int(dim) for dim in (extra.shape or [1, 1])]
                feed[extra.name] = np.zeros(shape, dtype=np.float32)
            future = executor.submit(session.run, None, feed)
            return future.result(timeout=max(0.01, float(timeout)))
        except concurrent.futures.TimeoutError:
            self.disable_gpu("GPU 推理超时", 180.0)
            return None
        except Exception as error:
            self.disable_gpu("GPU 推理失败：" + str(error), 300.0)
            return None
        finally:
            self.gpu_inflight.release()

    def _policy_vector(self, features):
        action_type = str((features or {}).get("action_type", "移动"))
        samples = max(0.0, min(1.0, float((features or {}).get("samples", 0) or 0) / 256.0))
        distance = max(0.0, min(1.0, float((features or {}).get("state_match_distance", 1.0) or 1.0)))
        uncertainty = max(0.0, min(1.0, float((features or {}).get("uncertainty", 1.0) or 1.0)))
        available = 1.0 if (features or {}).get("model_available") else 0.0
        return [[1.0 if action_type == "移动" else 0.0, samples, distance, uncertainty, available]]

    def disable_gpu(self, reason, retry_seconds=300.0):
        with self.lock:
            self.runtime_ready = False
            self.model_session = None
            self.gpu_failure_reason = str(reason)
            self.gpu_retry_after = time.monotonic() + max(30.0, float(retry_seconds))
            self.last_backend = "CPU 表格策略；GPU 回退"

    def _ensure_executor(self, workers):
        workers = max(1, min(4, int(workers)))
        old_executor = None
        with self.lock:
            if self.executor is None or int(self.executor_workers or 0) != workers:
                old_executor = self.executor
                self.executor_workers = workers
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="PngEncode")
            executor = self.executor
        if old_executor is not None:
            old_executor.shutdown(wait=False, cancel_futures=True)
        return executor

    def _encode_parallelism(self, workers):
        return max(1, min(4, int(workers)))

    def set_encode_metrics_sink(self, sink):
        with self.lock:
            self.encode_metrics_sink = sink

    def name(self):
        with self.lock:
            return self.last_backend

    def snapshot(self):
        with self.lock:
            warmup_age = 0.0 if not self.gpu_warmup_started else max(0.0, time.monotonic() - float(self.gpu_warmup_started))
            return {"runtime_provider": self.runtime_provider or "不可用", "runtime_ready": bool(self.runtime_ready), "gpu_model_path": self.gpu_model_path or "", "gpu_failure_reason": self.gpu_failure_reason or "", "last_backend": self.last_backend, "warmup_age": warmup_age, "warmup_complete": warmup_age >= 30.0, "retry_after_seconds": max(0.0, float(self.gpu_retry_after) - time.monotonic())}

    def shutdown(self):
        with self.lock:
            executor = self.executor
            gpu_executor = self.gpu_executor
            self.executor = None
            self.executor_workers = 0
            self.gpu_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if gpu_executor is not None:
            gpu_executor.shutdown(wait=False, cancel_futures=True)

    def _encode_png_one(self, frame, queued_at=None, compression_level=6):
        started = time.monotonic()
        with self._png_lock:
            self._png_active += 1
            active = self._png_active
        try:
            image = dict(frame)
            if image.get("bgra") is not None:
                image["png"] = encode_png_bgra(int(image["width"]), int(image["height"]), image.pop("bgra"), compression_level)
            else:
                image["png"] = encode_png(int(image["width"]), int(image["height"]), image.pop("rgb"), compression_level)
            image["png_compression_level"] = int(compression_level)
            image["compute_backend"] = "CPU PNG 后台编码"
            return image
        finally:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            with self._png_lock:
                self._png_active = max(0, self._png_active - 1)
                active_after = self._png_active
            sink = self.encode_metrics_sink
            if sink is not None:
                try:
                    sink(elapsed_ms, active, active_after, max(0.0, time.monotonic() - float(queued_at or started)))
                except Exception:
                    pass

    def encode_frames(self, frames, budget):
        items = list(frames or [])[:max(1, int(budget.max_batch))]
        if not items:
            return []
        slots = max(1, min(4, int(budget.cpu_workers), len(items)))
        executor = self._ensure_executor(slots)
        queued_at = time.monotonic()
        compression_level = 1 if str(getattr(budget, "state", "正常")) != "正常" or float(getattr(budget, "queue_fill_ratio", 0.0) or 0.0) >= 0.70 else 7
        encoded = []
        for start in range(0, len(items), slots):
            batch = items[start:start + slots]
            futures = [executor.submit(self._encode_png_one, item, queued_at, compression_level) for item in batch]
            for future in futures:
                encoded.append(future.result())
        with self.lock:
            self.last_backend = "CPU 表格策略；PNG 并发由每任务预算令牌限制"
        return encoded

    def infer_policy(self, features, budget):
        features = features or {}
        action_type = str(features.get("action_type", "移动"))
        samples = max(0, int(features.get("samples", 0) or 0))
        distance = max(0.0, min(1.0, float(features.get("state_match_distance", 1.0) or 1.0)))
        uncertainty = max(0.0, min(1.0, float(features.get("uncertainty", 1.0) or 1.0)))
        model_available = bool(features.get("model_available"))
        if action_type == "移动":
            confidence = 0.58 if not model_available else 0.52 + 0.18 * (1.0 - distance) + min(0.12, samples / 500.0)
            uncertainty = max(0.08, min(0.65, 0.45 * distance + (0.15 if not model_available else uncertainty * 0.35)))
        else:
            confidence = max(0.0, min(1.0, float(features.get("confidence_probability", features.get("confidence", 0.0)) or 0.0)))
            confidence *= max(0.0, 1.0 - distance * 0.45)
        backend = "CPU 表格策略"
        metrics = features.get("resource_metrics") or {}
        timeout = min(0.18, max(0.02, float(getattr(budget, "retrieval_deadline_seconds", 0.08) or 0.08)))
        gpu_result = self.run_gpu_features(features, metrics, timeout=timeout)
        if gpu_result is not None:
            try:
                import numpy as np
                values = np.asarray(gpu_result[0], dtype=np.float32).reshape(-1)
                if values.size >= POLICY_OUTPUT_SIZE:
                    parsed = parse_visual_policy_output(values)
                    gpu_confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
                    gpu_uncertainty = max(0.02, min(0.95, float(parsed.get("uncertainty", 1.0))))
                    confidence = max(0.0, min(1.0, 0.30 * confidence + 0.70 * gpu_confidence))
                    uncertainty = max(0.02, min(0.95, 0.30 * uncertainty + 0.70 * gpu_uncertainty))
                    backend = "GPU ONNX 多头视觉策略"
                elif values.size >= 2 and math.isfinite(float(values[0])) and math.isfinite(float(values[1])):
                    gpu_confidence = max(0.0, min(1.0, float(values[0])))
                    gpu_uncertainty = max(0.02, min(0.95, float(values[1])))
                    if action_type == "移动":
                        confidence = max(0.0, min(1.0, 0.35 * confidence + 0.65 * gpu_confidence))
                        uncertainty = max(0.02, min(0.95, 0.35 * uncertainty + 0.65 * gpu_uncertainty))
                    else:
                        confidence = min(confidence, gpu_confidence)
                        uncertainty = max(uncertainty, gpu_uncertainty)
                    backend = "GPU ONNX 置信度回退策略"
            except Exception as error:
                self.disable_gpu("GPU 输出解析失败：" + str(error), 300.0)
                backend = "CPU 表格策略；GPU 输出回退"
        with self.lock:
            self.last_backend = backend
        return {"backend": backend, "confidence": max(0.0, min(1.0, confidence)), "uncertainty": uncertainty, "executed": True}

    def suggest_visual_action(self, features, metrics, timeout=0.06):
        gpu_result = self.run_gpu_features(features, metrics, timeout=timeout)
        if gpu_result is None:
            return None
        try:
            import numpy as np
            values = np.asarray(gpu_result[0], dtype=np.float32).reshape(-1)
            if values.size < POLICY_OUTPUT_SIZE:
                return None
            parsed = parse_visual_policy_output(values)
            with self.lock:
                self.last_backend = "GPU ONNX 多头视觉策略"
            return parsed
        except Exception as error:
            self.disable_gpu("GPU 多头策略输出解析失败：" + str(error), 300.0)
            return None

class GpuScheduler:
    def __init__(self, probe):
        self.probe = probe

    def assign(self, metrics, backend=None):
        if backend is None or not backend.can_use_gpu():
            return "CPU"
        return self.probe.choose_gpu(metrics)


class ModelRuntime:
    def __init__(self, backend):
        self.backend = backend

    def confidence_band(self, confidence, pressure):
        if pressure:
            return "pressure"
        if confidence >= 0.65:
            return "high"
        if confidence >= 0.35:
            return "medium"
        return "low"

class ResourceGovernor:
    def __init__(self, storage_path):
        self.lock = threading.RLock()
        self.storage_path = Path(storage_path).expanduser().resolve()
        self.emulator_path = ""
        self.emulator_pid = 0
        self.previous = None
        self.process_previous = {}
        self.process_last_sample = {}
        self.last_sample = 0.0
        self.last_disk_probe = 0.0
        self.window = []
        self.resource_green_level = 0
        self.resource_green_since = None
        self.metrics = {"cpu": 0.0, "process_cpu": 0.0, "process_memory": 0, "ldplayer_cpu": 0.0, "ldplayer_memory": 0, "memory": 0.0, "avail_memory": 0, "commit_free": 0, "disk_free": 0, "disk_write_latency": None, "sqlite_latency": 0.0, "capture_latency": 0.0, "queue": 0, "queue_age": 0.0, "pipeline_queue": 0, "pipeline_queue_age": 0.0, "pipeline_queue_capacity": 96, "pipeline_queue_ratio": 0.0, "gpu": None, "gpu_dedicated_total": None, "gpu_dedicated_used": None, "gpu_dedicated_free": None, "gpu_engine": None, "ldplayer_gpu": None, "program_gpu": None, "gpu_metrics_available": False, "gpu_sampling_source": "Windows GPU 性能计数器不可用", "last_user_input": time.time(), "capture_failure_rate": 0.0, "png_encode_ms": 0.0, "png_active": 0, "png_queue_age": 0.0, "exact_score_backlog": 0, "exact_score_oldest_age": 0.0, "wal_bytes": 0, "wal_checkpoint_ms": 0.0, "sqlite_transaction_ms": 0.0, "ui_heartbeat_jitter_ms": 0.0, "metric_sources": {"本程序 CPU": "GetProcessTimes", "目标进程 CPU": "GetProcessTimes（绑定雷电进程树）", "本程序 GPU 引擎": "Windows GPU 性能计数器", "目标进程 GPU 引擎": "Windows GPU 性能计数器", "可用显存": "Windows GPU Adapter Memory 性能计数器", "磁盘写入延迟": "fsync 探针", "SQLite 写入延迟": "实际 SQLite 事务计时", "队列年龄": "队列记录时间戳", "UI 心跳抖动": "Tk after 心跳延迟"}}
        self.probe = HardwareProbe()
        self.backend = ComputeBackend(self.probe)
        self.backend.set_encode_metrics_sink(self.update_encode_metrics)
        self.gpu_scheduler = GpuScheduler(self.probe)
        self.resource_state = "正常"
        self.resource_state_since = time.monotonic()
        self.resource_decisions = []
        self.device_baseline_started = time.monotonic()
        self.device_baseline_samples = []
        self.device_baseline = {}
        self.device_baseline_ready = False
        self.runtime = ModelRuntime(self.backend)
        self.inflight_limits = {"capture": 1, "png": 4, "sqlite": 1, "exact": 1, "training": 1}
        self.inflight_counts = collections.Counter()
        self.levels = {}
        self.last_pressure = 0.0
        self.healthy_since = time.monotonic()
        self.red_latched = False
        self.red_recovery_since = None
        self.pressure_reasons = []
        self._slow_stop = threading.Event()
        self._slow_thread = None
        checked_storage_path(self.storage_path, self.storage_path)
        self.sample()
        self._slow_thread = threading.Thread(target=self._slow_sample_loop, name="ResourceSlowProbe", daemon=True)
        self._slow_thread.start()
        self._fast_stop = threading.Event()
        self._fast_budget = ResourceBudget(True, 1.0, 1, 1, "CPU", 0, (640, 360), False, "", "正常")
        self._fast_thread = threading.Thread(target=self._fast_budget_loop, name="ResourceBudgetSnapshot", daemon=True)
        self._fast_thread.start()

    def set_storage_path(self, path):
        resolved = Path(path).expanduser().resolve()
        checked_storage_path(resolved, resolved)
        with self.lock:
            self.storage_path = resolved

    def set_emulator_path(self, path):
        with self.lock:
            self.emulator_path = str(path or "")

    def set_emulator_pid(self, pid):
        with self.lock:
            self.emulator_pid = max(0, int(pid or 0))

    def _fast_budget_loop(self):
        while not self._fast_stop.is_set():
            try:
                budget = self.acquire("capture")
                with self.lock:
                    self._fast_budget = budget
            except Exception:
                pass
            self._fast_stop.wait(0.20)

    def capture_snapshot(self):
        with self.lock:
            return self._fast_budget

    def update_queue(self, length, age=0.0):
        with self.lock:
            self.metrics["queue"] = int(length)
            self.metrics["queue_age"] = float(age)

    def update_pipeline_queue(self, length, age=0.0, capacity=96):
        with self.lock:
            capacity = max(1, int(capacity))
            self.metrics["pipeline_queue"] = int(length)
            self.metrics["pipeline_queue_capacity"] = capacity
            self.metrics["pipeline_queue_ratio"] = max(0.0, min(1.0, float(length) / capacity))
            self.metrics["pipeline_queue_age"] = float(age)
            self.metrics["queue_age"] = max(float(self.metrics.get("queue_age", 0.0)), float(age))

    def update_capture_metrics(self, elapsed_ms=None, failed=False):
        with self.lock:
            if elapsed_ms is not None:
                self.metrics["capture_latency"] = float(elapsed_ms)
            old = float(self.metrics.get("capture_failure_rate", 0.0))
            self.metrics["capture_failure_rate"] = old * 0.95 + (0.05 if failed else 0.0)

    def update_encode_metrics(self, elapsed_ms, active_before, active_after, queue_age):
        with self.lock:
            self.metrics["png_encode_ms"] = float(elapsed_ms)
            self.metrics["png_active"] = max(0, int(active_after))
            self.metrics["png_queue_age"] = max(0.0, float(queue_age))

    def update_ui_heartbeat(self, jitter_ms):
        with self.lock:
            self.metrics["ui_heartbeat_jitter_ms"] = max(0.0, float(jitter_ms))

    def update_exact_score_metrics(self, pending, oldest_age):
        with self.lock:
            self.metrics["exact_score_backlog"] = max(0, int(pending))
            self.metrics["exact_score_oldest_age"] = max(0.0, float(oldest_age))

    def update_database_metrics(self, wal_bytes=None, checkpoint_ms=None, transaction_ms=None):
        with self.lock:
            if wal_bytes is not None:
                self.metrics["wal_bytes"] = max(0, int(wal_bytes))
            if checkpoint_ms is not None:
                self.metrics["wal_checkpoint_ms"] = max(0.0, float(checkpoint_ms))
            if transaction_ms is not None:
                self.metrics["sqlite_transaction_ms"] = max(0.0, float(transaction_ms))

    def pop_resource_decisions(self):
        with self.lock:
            rows = list(self.resource_decisions)
            self.resource_decisions = []
        return rows

    def _set_resource_state_locked(self, state, reasons):
        now = time.monotonic()
        if state == self.resource_state:
            return
        self.resource_decisions.append({"from": self.resource_state, "to": state, "triggered_at": time.time(), "duration_seconds": max(0.0, now - self.resource_state_since), "reasons": list(reasons), "metrics": {"cpu": float(self.metrics.get("cpu", 0.0)), "capture_failure_rate": float(self.metrics.get("capture_failure_rate", 0.0)), "png_encode_ms": float(self.metrics.get("png_encode_ms", 0.0)), "pipeline_queue_ratio": float(self.metrics.get("pipeline_queue_ratio", 0.0)), "exact_score_backlog": int(self.metrics.get("exact_score_backlog", 0)), "wal_bytes": int(self.metrics.get("wal_bytes", 0))}})
        self.resource_state = state
        self.resource_state_since = now

    def update_sqlite_latency(self, elapsed_ms):
        with self.lock:
            self.metrics["sqlite_latency"] = float(elapsed_ms)

    def update_user_input(self):
        with self.lock:
            self.metrics["last_user_input"] = time.time()

    def _cpu_for_pid(self, pid, now):
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, int(pid))
        if not handle:
            return 0.0, 0
        try:
            creation = FILETIME(); exit_time = FILETIME(); kernel_time = FILETIME(); user_time = FILETIME()
            cpu = 0.0
            if kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel_time), ctypes.byref(user_time)):
                total = filetime_value(kernel_time) + filetime_value(user_time)
                with self.lock:
                    previous = self.process_previous.get(int(pid))
                    previous_time = self.process_last_sample.get(int(pid), now)
                    self.process_previous[int(pid)] = total
                    self.process_last_sample[int(pid)] = now
                if previous is not None:
                    elapsed = max(0.001, now - previous_time)
                    cpu = max(0.0, min(100.0, (total - previous) / 10000000.0 / elapsed * 100.0 / max(1, os.cpu_count() or 1)))
            counters = PROCESS_MEMORY_COUNTERS(); counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            memory = int(counters.WorkingSetSize) if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb) else 0
            return cpu, memory
        finally:
            kernel32.CloseHandle(handle)

    def _process_metrics(self, now):
        program_cpu, program_memory = self._cpu_for_pid(os.getpid(), now)
        with self.lock:
            emulator_pid = int(self.emulator_pid or 0)
            name = Path(self.emulator_path).name.lower() if self.emulator_path else "dnplayer.exe"
        roots = {emulator_pid} if emulator_pid else processes_for_name(name)
        ld_pids = process_tree(roots)
        ld_cpu = 0.0
        ld_memory = 0
        for pid in ld_pids:
            item_cpu, item_memory = self._cpu_for_pid(pid, now)
            ld_cpu += item_cpu
            ld_memory += item_memory
        return program_cpu, program_memory, ld_cpu, ld_memory

    def _disk_probe_latency(self, now):
        if now - self.last_disk_probe < 5.0:
            return self.metrics.get("disk_write_latency")
        self.last_disk_probe = now
        probe_path = checked_storage_path(Path(self.storage_path) / "experience_pool" / ".write_latency_probe.tmp", self.storage_path)
        try:
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            with probe_path.open("wb") as handle:
                handle.write(b"0" * 4096)
                handle.flush()
                os.fsync(handle.fileno())
            if storage_path_allowed(probe_path, self.storage_path) and not _is_windows_reparse_point(probe_path):
                probe_path.unlink(missing_ok=True)
            return (time.perf_counter() - started) * 1000.0
        except OSError:
            return None

    def _slow_sample_loop(self):
        while not self._slow_stop.is_set():
            with self.lock:
                emulator_name = Path(self.emulator_path).name.lower() if self.emulator_path else "dnplayer.exe"
                emulator_pid = int(self.emulator_pid or 0)
            roots = {emulator_pid} if emulator_pid else processes_for_name(emulator_name)
            runtime = self.probe.runtime_metrics(os.getpid(), process_tree(roots))
            self.backend.refresh_gpu_stability(runtime)
            self.probe.probe()
            disk_latency = self._disk_probe_latency(time.monotonic())
            with self.lock:
                self.metrics.update({"gpu": runtime.get("gpu_engine"), "gpu_dedicated_total": runtime.get("dedicated_total"), "gpu_dedicated_used": runtime.get("dedicated_used"), "gpu_dedicated_free": runtime.get("dedicated_free"), "gpu_engine": runtime.get("gpu_engine"), "program_gpu": runtime.get("program_gpu"), "ldplayer_gpu": runtime.get("ldplayer_gpu"), "gpu_metrics_available": bool(runtime.get("available")), "gpu_sampling_source": runtime.get("source"), "disk_write_latency": disk_latency})
            self._slow_stop.wait(1.0)

    def shutdown(self):
        self._slow_stop.set()
        self._fast_stop.set()
        for thread in (self._slow_thread, self._fast_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(5.0)
        self.backend.shutdown()

    def sample(self):
        now = time.monotonic()
        with self.lock:
            if now - self.last_sample < 1.0 and self.window:
                return self.summary_locked()
            storage_path = self.storage_path
            cpu = self.metrics.get("cpu", 0.0)
            previous = self.previous
            memory = self.metrics.get("memory", 0.0)
            avail_memory = self.metrics.get("avail_memory", 0)
            commit_free = self.metrics.get("commit_free", 0)
        idle = FILETIME(); kernel = FILETIME(); user = FILETIME()
        current = None
        if kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            current = tuple((item.dwHighDateTime << 32) | item.dwLowDateTime for item in (idle, kernel, user))
            if previous is not None:
                idle_delta = current[0] - previous[0]
                total_delta = current[1] + current[2] - previous[1] - previous[2]
                if total_delta > 0:
                    cpu = max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))
        status = MEMORYSTATUSEX(); status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            memory = float(status.dwMemoryLoad)
            avail_memory = int(status.ullAvailPhys)
            commit_free = max(0, int(status.ullAvailPageFile))
        try:
            disk_free = int(shutil.disk_usage(storage_path).free)
        except Exception:
            disk_free = 0
        process_cpu, process_memory, ld_cpu, ld_memory = self._process_metrics(now)
        with self.lock:
            if current is not None:
                self.previous = current
            self.metrics.update({"cpu": cpu, "memory": memory, "avail_memory": avail_memory, "commit_free": commit_free, "disk_free": disk_free, "process_cpu": process_cpu, "process_memory": process_memory, "ldplayer_cpu": ld_cpu, "ldplayer_memory": ld_memory})
            point = dict(self.metrics, t=now)
            self.window.append(point)
            self.window = [p for p in self.window if now - p["t"] <= 30.0]
            self._update_device_baseline_locked(point, now)
            self.last_sample = now
            return self.summary_locked()

    def summary_locked(self):
        result = dict(self.metrics)
        for key in list(result):
            values = [float(p.get(key, 0.0)) for p in self.window if isinstance(p.get(key), (int, float))]
            if values:
                ordered = sorted(values)
                result[key + "_avg"] = sum(values) / len(values)
                result[key + "_p95"] = ordered[min(len(ordered) - 1, int(math.ceil(len(ordered) * 0.95)) - 1)]
                result[key + "_trend"] = values[-1] - values[0]
        result["io_latency"] = max(float(result.get("capture_latency_p95", result.get("capture_latency", 0.0)) or 0.0) / 1000.0, float(result.get("sqlite_latency_p95", result.get("sqlite_latency", 0.0)) or 0.0) / 1000.0, float(result.get("queue_age", 0.0) or 0.0))
        result["backend"] = self.backend.name()
        result["gpus"] = self.probe.snapshot_gpus()
        result["resource_state"] = self.resource_state
        result["pause_reason"] = "；".join(self.pressure_reasons)
        result["device_baseline_ready"] = bool(self.device_baseline_ready)
        result["device_baseline"] = dict(self.device_baseline)
        return result

    def _percentile_value(self, values, percentile):
        items = sorted(float(item) for item in values if isinstance(item, (int, float)) and math.isfinite(float(item)))
        if not items:
            return 0.0
        return items[min(len(items) - 1, max(0, int(math.ceil(len(items) * float(percentile))) - 1))]

    def _update_device_baseline_locked(self, point, now):
        if self.device_baseline_ready:
            return
        elapsed = max(0.0, now - float(self.device_baseline_started))
        sample = {
            "capture_latency": float(point.get("capture_latency", 0.0) or 0.0),
            "sqlite_latency": float(point.get("sqlite_latency", 0.0) or 0.0),
            "png_encode_ms": float(point.get("png_encode_ms", 0.0) or 0.0),
            "ui_heartbeat_jitter_ms": float(point.get("ui_heartbeat_jitter_ms", 0.0) or 0.0),
            "disk_write_latency": float(point.get("disk_write_latency", 0.0) or 0.0),
            "cpu_busy": float(point.get("cpu", 0.0) or 0.0),
            "cpu_idle": max(0.0, 100.0 - float(point.get("cpu", 0.0) or 0.0)),
        }
        self.device_baseline_samples.append(sample)
        self.device_baseline_samples = self.device_baseline_samples[-120:]
        if elapsed < 60.0 and len(self.device_baseline_samples) < 45:
            return
        baseline = {}
        for key in ("capture_latency", "sqlite_latency", "png_encode_ms", "ui_heartbeat_jitter_ms", "disk_write_latency", "cpu_busy"):
            baseline[key + "_p95"] = self._percentile_value([item.get(key, 0.0) for item in self.device_baseline_samples], 0.95)
        baseline["cpu_idle_p05"] = self._percentile_value([100.0 - item.get("cpu_idle", 0.0) for item in self.device_baseline_samples], 0.05)
        baseline["sample_count"] = len(self.device_baseline_samples)
        baseline["elapsed_seconds"] = elapsed
        self.device_baseline = baseline
        self.device_baseline_ready = True

    def _task_profile(self, task):
        profiles = {
            "capture": {"cpu": 1.00, "io": 1.00, "capture": 1.00, "sqlite": 1.00, "png": 1.00, "ui": 1.00, "interval": 1.00, "batch": 1.00},
            "ai_inference": {"cpu": 1.08, "io": 1.10, "capture": 1.20, "sqlite": 1.10, "png": 1.15, "ui": 1.05, "interval": 0.92, "batch": 0.85},
            "maintenance": {"cpu": 0.92, "io": 0.82, "capture": 1.45, "sqlite": 0.85, "png": 1.40, "ui": 0.90, "interval": 1.20, "batch": 0.65},
            "sleep_training": {"cpu": 1.15, "io": 1.05, "capture": 1.35, "sqlite": 1.15, "png": 1.20, "ui": 1.10, "interval": 0.78, "batch": 1.20},
        }
        return dict(profiles.get(str(task), profiles["capture"]))

    def _adaptive_threshold(self, metric, absolute, multiple, sample, cap=None):
        key = metric + "_p95"
        value = float((self.device_baseline or {}).get(key, 0.0) or 0.0)
        if metric == "cpu_busy" and not value:
            value = float(sample.get("cpu_avg", sample.get("cpu", 0.0)) or 0.0)
        threshold = max(float(absolute), value * float(multiple))
        if cap is not None:
            threshold = min(float(cap), threshold)
        return threshold

    def try_enter_budget(self, name, redline_ok=False):
        key = str(name)
        with self.lock:
            red = bool(self.red_latched)
            if red and not redline_ok:
                return False
            if self.inflight_counts[key] >= int(self.inflight_limits.get(key, 1)):
                return False
            self.inflight_counts[key] += 1
            return True

    def leave_budget(self, name):
        key = str(name)
        with self.lock:
            self.inflight_counts[key] = max(0, int(self.inflight_counts.get(key, 0)) - 1)

    def inflight_snapshot(self):
        with self.lock:
            return dict(self.inflight_counts)

    def budget_slot(self, name, redline_ok=False):
        return ResourceBudgetSlot(self, name, redline_ok)

    def acquire(self, task):
        sample = self.sample()
        now = time.monotonic()
        task = str(task)
        profile = self._task_profile(task)
        reasons = []
        yellow = []
        cpu_p95 = float(sample.get("cpu_p95", sample.get("cpu", 0.0)) or 0.0)
        disk_p95 = float(sample.get("disk_write_latency_p95", sample.get("disk_write_latency", 0.0)) or 0.0)
        queue_age = float(sample.get("queue_age", 0.0) or 0.0)
        queue_ratio = float(sample.get("pipeline_queue_ratio", 0.0) or 0.0)
        capture_failure_rate = float(sample.get("capture_failure_rate", 0.0) or 0.0)
        png_p95 = float(sample.get("png_encode_ms_p95", sample.get("png_encode_ms", 0.0)) or 0.0)
        wal_bytes = int(sample.get("wal_bytes", 0) or 0)
        checkpoint_p95 = float(sample.get("wal_checkpoint_ms_p95", sample.get("wal_checkpoint_ms", 0.0)) or 0.0)
        exact_backlog = int(sample.get("exact_score_backlog", 0) or 0)
        ui_jitter_p95 = float(sample.get("ui_heartbeat_jitter_ms_p95", sample.get("ui_heartbeat_jitter_ms", 0.0)) or 0.0)
        sqlite_p95 = float(sample.get("sqlite_latency_p95", sample.get("sqlite_latency", 0.0)) or 0.0)
        capture_p95 = float(sample.get("capture_latency_p95", sample.get("capture_latency", 0.0)) or 0.0)
        gpu_p95 = float(sample.get("gpu_engine_p95", sample.get("gpu_engine", 0.0)) or 0.0)
        cpu_yellow_threshold = self._adaptive_threshold("cpu_busy", 85.0, profile["cpu"], sample, 97.0)
        cpu_red_threshold = self._adaptive_threshold("cpu_busy", 95.0, profile["cpu"] * 1.18, sample, 99.0)
        disk_yellow_threshold = self._adaptive_threshold("disk_write_latency", 100.0, 2.5 * profile["io"], sample, 300.0)
        disk_red_threshold = self._adaptive_threshold("disk_write_latency", 150.0, 3.5 * profile["io"], sample, 650.0)
        capture_yellow_threshold = self._adaptive_threshold("capture_latency", 150.0, 2.2 * profile["capture"], sample, 500.0)
        capture_soft_threshold = self._adaptive_threshold("capture_latency", 120.0, 1.8 * profile["capture"], sample, 420.0)
        sqlite_yellow_threshold = self._adaptive_threshold("sqlite_latency", 100.0, 2.4 * profile["sqlite"], sample, 500.0)
        sqlite_red_threshold = self._adaptive_threshold("sqlite_latency", 2000.0, 9.0 * profile["sqlite"], sample, 5000.0)
        png_yellow_threshold = self._adaptive_threshold("png_encode_ms", 120.0, 2.2 * profile["png"], sample, 450.0)
        png_hard_threshold = self._adaptive_threshold("png_encode_ms", 250.0, 4.0 * profile["png"], sample, 900.0)
        png_pause_threshold = self._adaptive_threshold("png_encode_ms", 500.0, 7.0 * profile["png"], sample, 1500.0)
        ui_yellow_threshold = self._adaptive_threshold("ui_heartbeat_jitter_ms", 200.0, 2.2 * profile["ui"], sample, 650.0)
        ui_red_threshold = self._adaptive_threshold("ui_heartbeat_jitter_ms", 500.0, 4.0 * profile["ui"], sample, 1500.0)
        red_now = False
        if cpu_p95 >= cpu_red_threshold:
            red_now = True
            reasons.append("系统 CPU P95 ≥ {:.0f}%".format(cpu_red_threshold))
        if int(sample.get("avail_memory", 0) or 0) < 384 * 1024 * 1024:
            red_now = True
            reasons.append("可用内存不足 384 MB")
        if queue_age > 2.0:
            red_now = True
            reasons.append("高优先级队列等待超过 2 秒")
        if disk_p95 > disk_red_threshold:
            red_now = True
            reasons.append("磁盘写入 P95 超过 {:.0f} ms".format(disk_red_threshold))
        if int(sample.get("disk_free", 0) or 0) < 1024 * 1024 * 1024:
            red_now = True
            reasons.append("磁盘剩余空间不足 1 GB")
        if capture_failure_rate >= 0.35:
            red_now = True
            reasons.append("截图连续失败率过高")
        if checkpoint_p95 > 2000.0 or sqlite_p95 > sqlite_red_threshold:
            red_now = True
            reasons.append("SQLite 写入或 WAL checkpoint P95 过高")
        if ui_jitter_p95 > ui_red_threshold:
            red_now = True
            reasons.append("控制面板 UI 心跳 P95 超过 {:.0f} ms".format(ui_red_threshold))
        if cpu_p95 >= cpu_yellow_threshold:
            yellow.append("系统 CPU P95 ≥ {:.0f}%".format(cpu_yellow_threshold))
        if float(sample.get("process_cpu", 0.0) or 0.0) >= 60.0:
            yellow.append("本程序 CPU ≥ 60%")
        if float(sample.get("ldplayer_cpu", 0.0) or 0.0) >= 70.0:
            yellow.append("目标进程 CPU ≥ 70%")
        if float(sample.get("memory", 0.0) or 0.0) >= 88.0:
            yellow.append("内存占用 ≥ 88%")
        if sample.get("gpu_metrics_available") and gpu_p95 >= 80.0:
            yellow.append("GPU 引擎 P95 ≥ 80%")
            if self.backend.can_use_gpu():
                self.backend.disable_gpu("GPU 引擎 P95 过高", 300.0)
        if capture_p95 > capture_yellow_threshold:
            yellow.append("截图 P95 延迟过高")
        if sqlite_p95 > sqlite_yellow_threshold:
            yellow.append("SQLite 写入 P95 延迟过高")
        if disk_p95 > disk_yellow_threshold:
            yellow.append("磁盘写入延迟升高")
        if queue_age > 1.0:
            yellow.append("队列等待超过 1 秒")
        if queue_ratio >= 0.70:
            yellow.append("流水线队列达到 70%")
        if capture_failure_rate >= 0.08:
            yellow.append("截图失败率升高")
        if png_p95 > png_yellow_threshold:
            yellow.append("PNG 编码 P95 延迟过高")
        if checkpoint_p95 > 500.0:
            yellow.append("SQLite WAL checkpoint P95 延迟过高")
        if wal_bytes > 512 * 1024 * 1024:
            yellow.append("SQLite WAL 文件增长过大")
        if exact_backlog >= 96:
            yellow.append("精确评分积压过多")
        if ui_jitter_p95 > ui_yellow_threshold:
            yellow.append("控制面板 UI 心跳 P95 超过 {:.0f} ms".format(ui_yellow_threshold))
        avail_memory = int(sample.get("avail_memory", 0) or 0)
        memory_error = max(0.0, (768 * 1024 * 1024 - avail_memory) / float(768 * 1024 * 1024))
        errors = [
            max(0.0, queue_age / 0.25 - 1.0),
            max(0.0, sqlite_p95 / max(1.0, sqlite_yellow_threshold) - 1.0),
            max(0.0, capture_p95 / max(1.0, capture_yellow_threshold) - 1.0),
            max(0.0, cpu_p95 / max(1.0, cpu_yellow_threshold) - 1.0),
            max(0.0, gpu_p95 / 80.0 - 1.0) if sample.get("gpu_metrics_available") else 0.0,
            max(0.0, ui_jitter_p95 / max(1.0, ui_yellow_threshold) - 1.0),
            memory_error,
        ]
        control_error = max(errors)
        if control_error > 0.0 and not red_now:
            yellow.append("闭环控制误差 {:.2f}".format(control_error))
        with self.lock:
            if red_now:
                self.red_latched = True
                self.red_recovery_since = None
            elif self.red_latched:
                if self.red_recovery_since is None:
                    self.red_recovery_since = now
                elif now - self.red_recovery_since >= 20.0 and queue_age <= 0.25 and sqlite_p95 <= sqlite_yellow_threshold and capture_p95 <= capture_yellow_threshold and cpu_p95 <= cpu_yellow_threshold and ui_jitter_p95 <= ui_yellow_threshold and (not sample.get("gpu_metrics_available") or gpu_p95 <= 80.0) and avail_memory >= 768 * 1024 * 1024:
                    self.red_latched = False
                    self.red_recovery_since = None
            red_pause = bool(self.red_latched)
            pressure = red_pause or bool(yellow)
            current_level = float(self.levels.get(task, 4.0))
            if red_pause:
                next_level = max(1.0, current_level * 0.70)
                self.last_pressure = now
                self.healthy_since = now
            else:
                gain = 0.18 if pressure else -0.06
                next_level = current_level * math.exp(-gain * min(3.0, control_error))
                if not pressure and now - self.healthy_since >= 2.0:
                    next_level = min(16.0, next_level + 0.20)
                    self.healthy_since = now
                if pressure:
                    self.last_pressure = now
                    self.healthy_since = now
            self.levels[task] = max(1.0, min(16.0, next_level))
            level = int(max(1, round(self.levels.get(task, 4.0))))
            recovery_note = "红色条件已消失，恢复观察中" if red_pause and not red_now else ""
            self.pressure_reasons = reasons + yellow + ([recovery_note] if recovery_note else [])
        cores = max(1, os.cpu_count() or 1)
        workers_float = max(1.0, min(float(max(1, cores - max(1, math.ceil(cores * 0.25)))), self.levels.get(task, float(level))))
        workers = 1 if task in ("maintenance", "ai_inference") else max(1, int(round(workers_float)))
        interval_base = {"capture": 1.0, "ai_inference": 0.8, "sleep_training": 0.05, "maintenance": 1.0}.get(task, 1.0) * profile["interval"]
        interval = max(0.05, interval_base * max(0.35, 4.0 / max(1.0, self.levels.get(task, float(level)))))
        interval *= max(1.0, 1.0 + min(2.5, control_error * 0.85))
        if png_p95 > png_yellow_threshold:
            workers = 1
        if control_error <= 0.10 and png_p95 <= png_yellow_threshold and capture_p95 <= capture_soft_threshold:
            resolution = (640, 360)
        elif control_error <= 0.65:
            resolution = (426, 240)
        else:
            resolution = (320, 180)
        if png_p95 > png_hard_threshold:
            resolution = (426, 240)
        if png_p95 > png_pause_threshold:
            interval *= 2.0
        if queue_ratio >= 0.70:
            interval *= 1.0 + min(2.0, (queue_ratio - 0.70) * 6.0)
        if task == "capture":
            if queue_ratio >= 0.75:
                interval = max(interval, interval_base * 4.0)
                resolution = (320, 180)
            elif queue_ratio >= 0.50:
                interval = max(interval, interval_base * 2.0)
                resolution = (426, 240)
        sqlite_transaction_p95 = float(sample.get("sqlite_transaction_ms_p95", sample.get("sqlite_transaction_ms", 0.0)) or 0.0)
        pipeline_age = float(sample.get("pipeline_queue_age", queue_age) or 0.0)
        green = cpu_p95 < min(55.0, cpu_yellow_threshold * 0.70) and float(sample.get("process_cpu", 0.0) or 0.0) < 35.0 and avail_memory > 2 * 1024 * 1024 * 1024 and disk_p95 < min(20.0, disk_yellow_threshold * 0.50) and sqlite_transaction_p95 < 30.0 and pipeline_age < 0.100 and capture_failure_rate < 0.01 and not pressure and not red_pause
        with self.lock:
            if red_pause:
                self.resource_green_level = 0
                self.resource_green_since = None
            elif pressure:
                self.resource_green_level = max(0, int(self.resource_green_level) - 1)
                self.resource_green_since = None
            elif green:
                if self.resource_green_since is None:
                    self.resource_green_since = now
                elif now - self.resource_green_since >= 20.0:
                    self.resource_green_level = min(2, int(self.resource_green_level) + 1)
                    self.resource_green_since = now
            else:
                self.resource_green_since = None
            green_level = int(self.resource_green_level)
        draining = queue_ratio >= 0.85 or exact_backlog >= 128 or queue_age > 1.0
        if draining:
            workers = 1
            resolution = (320, 180)
            interval *= 1.0 + min(3.0, max(queue_age / 0.25, exact_backlog / 128.0))
        if red_pause:
            workers = 1
            resolution = (320, 180)
            interval = max(interval, 1.0)
        if not draining and not pressure and not red_pause and task == "capture":
            green_resolutions = ((640, 360), (854, 480), (1280, 720))
            resolution = green_resolutions[min(green_level, len(green_resolutions) - 1)]
            interval *= (1.0, 0.82, 0.68)[min(green_level, 2)]
        exact_batch_factor = (0.35 if control_error >= 1.0 else 0.65 if control_error >= 0.35 else (1.0 + 0.15 * min(green_level, 2))) * profile["batch"]
        max_batch = 1 if draining else max(1, min(64, int(round(level * (2 if task == "sleep_training" else 1) * exact_batch_factor))))
        maintenance_task = task == "maintenance"
        extreme_memory = int(sample.get("avail_memory", 0) or 0) < 128 * 1024 * 1024 or int(sample.get("commit_free", 0) or 0) < 128 * 1024 * 1024
        extreme_disk = int(sample.get("disk_free", 0) or 0) < 256 * 1024 * 1024
        sqlite_error = sqlite_p95 >= sqlite_red_threshold or checkpoint_p95 >= 5000.0 or float(sample.get("sqlite_transaction_ms_p95", sample.get("sqlite_transaction_ms", 0.0)) or 0.0) >= 2000.0
        if maintenance_task:
            hard_reasons = []
            if extreme_memory:
                hard_reasons.append("维护暂停：内存极低")
            if extreme_disk:
                hard_reasons.append("维护暂停：磁盘极低")
            if sqlite_error:
                hard_reasons.append("维护暂停：SQLite 错误或严重阻塞")
            pause = bool(hard_reasons)
            if hard_reasons:
                self.pressure_reasons = hard_reasons + [item for item in self.pressure_reasons if item not in hard_reasons]
            workers = 1
            max_batch = max(1, min(8, max_batch))
            interval = max(0.25, min(5.0, interval))
            resolution = (320, 180)
        else:
            pause = red_pause
        state = "暂停" if pause else "排空" if draining else "降速" if pressure else "正常"
        with self.lock:
            self._set_resource_state_locked(state, self.pressure_reasons)
        deadline = max(0.03, min(0.25, interval * 0.35))
        gpu_assignment = self.gpu_scheduler.assign(sample, self.backend)
        gpu_batch = max_batch if gpu_assignment == "GPU" else 0
        return ResourceBudget(not pause, interval, max_batch, workers, gpu_assignment, gpu_batch, resolution, pause, "；".join(self.pressure_reasons), state, max(8, min(512, level * 16)), max(1, min(8, max_batch if maintenance_task else level * 8)), max(8, min(256, level * 16)), max(1, min(workers, 4)), deadline, queue_ratio)


class Settings:
    DEFAULT_STORAGE_PATH = default_storage_path()

    def __init__(self):
        self.data = {"emulator_path": r"D:\LDPlayer9\dnplayer.exe", "storage_path": self.DEFAULT_STORAGE_PATH, "experience_limit": 10 * 1024 * 1024 * 1024, "model_limit": 100, "transaction_reserve_bytes": 8 * 1024 * 1024, "emulator_pid": 0, "emulator_title": ""}
        self.config_errors = []
        self._load_bootstrap_pointer()
        self.path = self._path_for_storage(self.data["storage_path"])
        self._load_storage_config()

    def _path_for_storage(self, storage_path):
        root = Path(storage_path).expanduser().resolve()
        return checked_storage_path(root / "config" / "settings.json", root)

    def _bootstrap_path(self):
        if sys.platform.startswith("win"):
            base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
        return base / APP_NAME / "bootstrap.json"

    def _load_bootstrap_pointer(self):
        try:
            pointer = self._bootstrap_path()
            if not pointer.exists():
                return False
            loaded = json.loads(pointer.read_text(encoding="utf-8"))
            storage_path = loaded.get("storage_path") if isinstance(loaded, dict) else None
            if not isinstance(storage_path, str) or not storage_path.strip():
                return False
            root = Path(storage_path).expanduser().resolve()
            checked_storage_path(root, root)
            self.data["storage_path"] = str(root)
            return True
        except Exception as error:
            self.config_errors.append("bootstrap 指针读取失败:" + str(error))
            return False

    def _write_bootstrap_pointer(self):
        pointer = self._bootstrap_path()
        pointer.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"storage_path": self.data.get("storage_path", self.DEFAULT_STORAGE_PATH), "updated": time.time()}, ensure_ascii=False, separators=(",", ":"))
        temp = pointer.with_name(pointer.name + "." + uuid.uuid4().hex + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(pointer)
        try:
            directory = os.open(str(pointer.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except Exception:
            pass

    def _apply_loaded(self, loaded, allow_storage_path=False):
        if not isinstance(loaded, dict):
            self.config_errors.append("配置根对象不是字典")
            return False
        string_keys = ("emulator_path", "emulator_title") + (("storage_path",) if allow_storage_path else ())
        for key in string_keys:
            if key in loaded:
                if isinstance(loaded[key], str):
                    self.data[key] = loaded[key]
                else:
                    self.config_errors.append(key + " 类型无效")
        for key, minimum, maximum in (("experience_limit", int(0.1*1024*1024*1024), 4096*1024*1024*1024), ("transaction_reserve_bytes", 1*1024*1024, 512*1024*1024), ("model_limit", 1, 100000), ("emulator_pid", 0, 2**31-1)):
            if key in loaded:
                value = loaded[key]
                if isinstance(value, int) and minimum <= value <= maximum:
                    self.data[key] = value
                else:
                    self.config_errors.append(key + " 超出范围或类型无效")
        return True

    def _load_config_from_path(self, source):
        try:
            return json.loads(Path(source).read_text(encoding="utf-8"))
        except Exception as error:
            self.config_errors.append("配置读取失败:" + str(error))
            note_strict_exception("settings_load", error, {"path": str(source)})
            return None

    def _load_storage_config(self):
        self.config_errors = []
        try:
            checked_storage_path(self.data["storage_path"], self.data["storage_path"])
        except Exception:
            self.data["storage_path"] = self.DEFAULT_STORAGE_PATH
        self.data["storage_path"] = str(Path(self.data.get("storage_path", self.DEFAULT_STORAGE_PATH)).expanduser().resolve())
        self.path = self._path_for_storage(self.data["storage_path"])
        loaded = self._load_config_from_path(self.path) if self.path.exists() else None
        if loaded is not None:
            self._apply_loaded(loaded, False)
            self.data["storage_path"] = str(Path(self.data.get("storage_path", self.DEFAULT_STORAGE_PATH)).expanduser().resolve())
            self.path = self._path_for_storage(self.data["storage_path"])
        checked_storage_path(self.path.parent, self.data["storage_path"]).mkdir(parents=True, exist_ok=True)
        self.save()

    def load(self):
        self.config_errors = []
        loaded = self._load_config_from_path(self.path)
        if loaded is None:
            return
        current_root = str(Path(self.data.get("storage_path", self.DEFAULT_STORAGE_PATH)).expanduser().resolve())
        self._apply_loaded(loaded, False)
        self.data["storage_path"] = current_root
        self.path = self._path_for_storage(self.data["storage_path"])

    def migrate_storage_path(self, storage_path):
        old_path = self.path
        old_data = dict(self.data)
        try:
            new_root = Path(storage_path).expanduser().resolve()
            checked_storage_path(new_root, new_root)
            self.data["storage_path"] = str(new_root)
            self.path = self._path_for_storage(self.data["storage_path"])
            if old_path != self.path and old_path.exists():
                backup_root = checked_storage_path(new_root / "config" / "migration_backup", new_root)
                backup_root.mkdir(parents=True, exist_ok=True)
                backup_name = old_path.name + "." + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
                backup_path = checked_storage_path(backup_root / backup_name, new_root)
                backup_path.write_bytes(old_path.read_bytes())
                self._fsync_directory_for_path(backup_path.parent)
            self.save()
        except Exception:
            self.data.update(old_data)
            self.path = old_path
            try:
                self.save()
            except Exception:
                pass
            raise

    def _fsync_directory_for_path(self, path):
        try:
            directory = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            if os.name == "nt":
                handle = kernel32.CreateFileW(str(path), 0x80000000, 0x00000001 | 0x00000002 | 0x00000004, None, 3, 0x02000000, None)
                if handle and handle != INVALID_HANDLE_VALUE:
                    try:
                        kernel32.FlushFileBuffers(handle)
                    finally:
                        kernel32.CloseHandle(handle)

    def save(self):
        self.data["storage_path"] = str(Path(self.data.get("storage_path", self.DEFAULT_STORAGE_PATH)).expanduser().resolve())
        self.path = self._path_for_storage(self.data["storage_path"])
        checked_storage_path(self.path, self.data["storage_path"])
        checked_storage_path(self.path.parent, self.data["storage_path"]).mkdir(parents=True, exist_ok=True)
        temp = checked_storage_path(self.path.with_suffix(".tmp"), self.data["storage_path"])
        payload = json.dumps(self.data, ensure_ascii=False, indent=2)
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(self.path)
        self._fsync_directory_for_path(self.path.parent)
        self._write_bootstrap_pointer()


class PoolCapacityBlocked(RuntimeError):
    pass

class DataStore:
    SCHEMA_VERSION = 8

    def __init__(self):
        self.root = None
        self.pool = None
        self.models = None
        self.screens = None
        self.journal = None
        self.conn = None
        self.lock = threading.RLock()
        self._database_bytes_cached = 0
        self._database_bytes_checked = 0.0
        self.transaction_reserve_bytes = 8 * 1024 * 1024
        self._recent_png_sizes = []
        self._capacity_write_count = 0
        self.faults = {}
        self.exact_score_lock = threading.Lock()
        self.last_wal_metrics = {"wal_bytes": 0, "checkpoint_ms": 0.0, "transaction_ms": 0.0}
        self.last_prune_coverage_loss = {"before": 0, "after": 0, "loss": 0, "ratio": 0.0, "paused": False}
        self._journal_lock = threading.RLock()
        self._journal_mouse_bytes = 0
        self._journal_mouse_last_fsync = 0.0
        self._sqlite_busy_count = 0
        self.db_writer = None

    def attach_db_writer(self, writer):
        self.db_writer = writer

    def _dispatch_write_to_writer(self, method_name, *args, **kwargs):
        writer = self.db_writer
        if writer is not None and not writer.is_writer_thread():
            return True, writer.call(lambda: getattr(self, method_name)(*args, **kwargs), timeout=60.0)
        return False, None

    def set_fault_injection(self, stage, count=1):
        with self.lock:
            self.faults[str(stage)] = max(0, int(count))

    def _inject_fault(self, stage):
        with self.lock:
            remaining = int(self.faults.get(str(stage), 0) or 0)
            if remaining <= 0:
                return
            self.faults[str(stage)] = remaining - 1
        raise OSError("故障注入：" + str(stage))

    def _commit_critical_locked(self):
        if self.conn is None:
            return
        try:
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.commit()
            try:
                self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass
        finally:
            try:
                self.conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.Error:
                pass

    def _fsync_directory(self, directory):
        try:
            directory = self._assert_storage_path(directory, "fsync_directory")
            if os.name == "nt":
                handle = kernel32.CreateFileW(str(directory), 0x80000000, 0x00000001 | 0x00000002 | 0x00000004, None, 3, 0x02000000 | 0x04000000, None)
                if handle and handle != INVALID_HANDLE_VALUE:
                    try:
                        kernel32.FlushFileBuffers(handle)
                    finally:
                        kernel32.CloseHandle(handle)
            else:
                fd = os.open(str(directory), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except Exception:
            pass

    def set_transaction_reserve(self, value):
        with self.lock:
            self.transaction_reserve_bytes = max(1 * 1024 * 1024, min(512 * 1024 * 1024, int(value)))

    def _critical_path_event(self, path, operation):
        payload = {"operation": str(operation), "path": str(path), "root": str(self.root or ""), "time": time.time()}
        try:
            if self.conn is not None:
                self.conn.execute("INSERT INTO system_events(id, session_id, created, kind, payload) VALUES (?, NULL, ?, 'critical_path_violation', ?)", (uuid.uuid4().hex, time.time(), json.dumps(payload, ensure_ascii=False)))
                self._trim_system_events_locked()
                self.conn.commit()
        except Exception:
            pass

    def _assert_storage_path(self, path, operation="write"):
        if self.root is None:
            raise OSError("存储根路径尚未初始化")
        try:
            return checked_storage_path(path, self.root)
        except OSError:
            self._critical_path_event(path, operation)
            raise

    def ensure(self, location):
        root = Path(location).expanduser().resolve()
        pool = root / "experience_pool"
        models = root / "models"
        screens = pool / "screens"
        journal = pool / "journal"
        for candidate in (root, pool, models, screens, journal, journal / "frames", pool / "records.sqlite3"):
            checked_storage_path(candidate, root)
        with self.lock:
            if self.root == root and self.conn is not None:
                return
            self.close()
            checked_storage_path(root, root).mkdir(parents=True, exist_ok=True)
            checked_storage_path(root, root)
            checked_storage_path(pool, root).mkdir(parents=True, exist_ok=True)
            checked_storage_path(models, root).mkdir(parents=True, exist_ok=True)
            checked_storage_path(screens, root).mkdir(parents=True, exist_ok=True)
            checked_storage_path(journal, root).mkdir(parents=True, exist_ok=True)
            checked_storage_path(journal / "frames", root).mkdir(parents=True, exist_ok=True)
            for made in (root, pool, models, screens, journal, journal / "frames"):
                checked_storage_path(made, root)
            self.root = root
            self.pool = pool
            self.models = models
            self.screens = screens
            self.journal = journal
            self._database_bytes_cached = 0
            self._database_bytes_checked = 0.0
            try:
                self.conn = sqlite3.connect(str(checked_storage_path(pool / "records.sqlite3", root)), check_same_thread=False, timeout=30)
                self.conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA synchronous=NORMAL")
                self.conn.execute("PRAGMA foreign_keys=ON")
                self._ensure_schema_core()
                self._migrate()
                self.conn.commit()
                self.compact_accepted_journal()
                self.replay_accepted_journal()
                self.recover_ingestions()
                self.recover_deletions()
            except Exception:
                try:
                    if self.conn is not None:
                        self.conn.rollback()
                        self.conn.close()
                except Exception:
                    pass
                self.conn = None
                self.root = None
                self.pool = None
                self.models = None
                self.screens = None
                self.journal = None
                raise

    def _ensure_schema_core(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                started REAL NOT NULL,
                ended REAL,
                frame_count INTEGER NOT NULL DEFAULT 0,
                mouse_count INTEGER NOT NULL DEFAULT 0,
                reason TEXT,
                trainable INTEGER NOT NULL DEFAULT 1,
                training_exclusion_reason TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS frames (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                created REAL NOT NULL,
                screenshot_path TEXT NOT NULL,
                phash TEXT NOT NULL,
                created_monotonic_ns INTEGER NOT NULL DEFAULT 0,
                capture_backend TEXT NOT NULL DEFAULT 'gdi',
                capture_elapsed_ms REAL NOT NULL DEFAULT 0,
                capture_complete INTEGER NOT NULL DEFAULT 1,
                brightness REAL NOT NULL DEFAULT 0,
                variance REAL NOT NULL DEFAULT 0,
                gray32x18 BLOB,
                edge_density REAL NOT NULL DEFAULT 0,
                color_histogram BLOB,
                capture_failure_reason TEXT NOT NULL DEFAULT '',
                capture_hash_delta REAL NOT NULL DEFAULT 64,
                capture_fallback INTEGER NOT NULL DEFAULT 0,
                dhash64 TEXT,
                score REAL,
                online_score REAL,
                hunger REAL NOT NULL,
                reward REAL,
                raw_score REAL,
                raw_reward REAL,
                reward_source TEXT NOT NULL DEFAULT 'screen_score_only',
                score_valid_for_training INTEGER NOT NULL DEFAULT 0,
                score_status TEXT NOT NULL DEFAULT 'invalid',
                score_generation TEXT NOT NULL DEFAULT 'online',
                history_boundary_frame_id TEXT,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                novelty REAL NOT NULL DEFAULT 0,
                action_result REAL NOT NULL DEFAULT 0,
                coverage REAL NOT NULL DEFAULT 0,
                model_refs INTEGER NOT NULL DEFAULT 0,
                retain_value REAL NOT NULL DEFAULT 0,
                retain_version INTEGER NOT NULL DEFAULT 1,
                last_used REAL NOT NULL DEFAULT 0,
                bucket0 INTEGER NOT NULL DEFAULT 0,
                bucket1 INTEGER NOT NULL DEFAULT 0,
                bucket2 INTEGER NOT NULL DEFAULT 0,
                bucket3 INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_frames_created ON frames(created DESC);
            CREATE INDEX IF NOT EXISTS idx_frames_reward ON frames(reward ASC, created ASC);
            CREATE INDEX IF NOT EXISTS idx_frames_buckets ON frames(bucket0, bucket1, bucket2, bucket3);
            CREATE TABLE IF NOT EXISTS mouse_loss_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                created REAL NOT NULL,
                started REAL NOT NULL,
                ended REAL NOT NULL,
                lost_count INTEGER NOT NULL,
                rule TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mouse_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                created REAL NOT NULL,
                created_monotonic_ns INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL,
                event_type TEXT NOT NULL,
                button TEXT NOT NULL,
                wheel INTEGER NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                relative_x REAL,
                relative_y REAL,
                dx REAL NOT NULL,
                dy REAL NOT NULL,
                direction REAL NOT NULL,
                speed REAL NOT NULL,
                before_frame_id TEXT,
                after_frame_id TEXT,
                action_time INTEGER NOT NULL DEFAULT 0,
                post_action_delay_ms REAL NOT NULL DEFAULT 0,
                score_delta REAL NOT NULL DEFAULT 0,
                reward_delta REAL NOT NULL DEFAULT 0,
                outcome_valid INTEGER NOT NULL DEFAULT 0,
                behavior_probability REAL
            );
            CREATE INDEX IF NOT EXISTS idx_mouse_session ON mouse_events(session_id, created);
            CREATE INDEX IF NOT EXISTS idx_mouse_session_mono ON mouse_events(session_id, created_monotonic_ns);
            CREATE INDEX IF NOT EXISTS idx_frames_session_mono ON frames(session_id, created_monotonic_ns);
            CREATE INDEX IF NOT EXISTS idx_mouse_session_type_mono ON mouse_events(session_id, event_type, created_monotonic_ns);
            CREATE TABLE IF NOT EXISTS state_clusters (cluster_id TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS frame_lsh (key INTEGER NOT NULL, frame_id TEXT NOT NULL, PRIMARY KEY(key, frame_id));
            CREATE INDEX IF NOT EXISTS idx_frame_lsh_key ON frame_lsh(key);
            CREATE TABLE IF NOT EXISTS ingestion_journal (id TEXT PRIMARY KEY, object_type TEXT NOT NULL, object_id TEXT NOT NULL, path TEXT, stage TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, error TEXT);
            CREATE TABLE IF NOT EXISTS action_outcomes (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), mouse_event_id TEXT NOT NULL, before_frame_id TEXT NOT NULL, after_frame_id TEXT NOT NULL, action_time INTEGER NOT NULL, post_action_delay_ms REAL NOT NULL, score_delta REAL NOT NULL, reward_delta REAL NOT NULL, outcome_valid INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS system_events (
                id TEXT PRIMARY KEY,
                session_id TEXT REFERENCES sessions(id),
                created REAL NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mode_transitions (
                id TEXT PRIMARY KEY,
                created REAL NOT NULL,
                from_mode TEXT NOT NULL,
                to_mode TEXT NOT NULL,
                reason TEXT NOT NULL,
                trigger TEXT NOT NULL,
                session_id TEXT,
                window_state TEXT NOT NULL,
                cursor_x INTEGER,
                cursor_y INTEGER,
                resource_state TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mode_transitions_created ON mode_transitions(created);
            CREATE TABLE IF NOT EXISTS deletion_journal (
                id TEXT PRIMARY KEY,
                object_type TEXT NOT NULL,
                object_id TEXT NOT NULL,
                path TEXT,
                stage TEXT NOT NULL,
                created REAL NOT NULL,
                updated REAL NOT NULL,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS pool_meta (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_frame_refs (
                model_id TEXT NOT NULL,
                frame_id TEXT NOT NULL REFERENCES frames(id),
                role TEXT NOT NULL,
                PRIMARY KEY(model_id, frame_id, role)
            );
            CREATE TABLE IF NOT EXISTS model_metadata (
                id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL UNIQUE,
                created REAL NOT NULL,
                quality REAL NOT NULL DEFAULT 0,
                validation_quality REAL NOT NULL DEFAULT 0,
                champion INTEGER NOT NULL DEFAULT 0,
                updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mouse_compression_segments (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                source TEXT NOT NULL,
                started REAL NOT NULL,
                ended REAL NOT NULL,
                started_monotonic_ns INTEGER NOT NULL,
                ended_monotonic_ns INTEGER NOT NULL,
                start_x INTEGER NOT NULL,
                start_y INTEGER NOT NULL,
                end_x INTEGER NOT NULL,
                end_y INTEGER NOT NULL,
                original_count INTEGER NOT NULL,
                max_speed REAL NOT NULL,
                path_length REAL NOT NULL,
                trajectory_blob BLOB NOT NULL DEFAULT X'',
                trajectory_codec TEXT NOT NULL DEFAULT 'varint-zigzag-dtxy-v1',
                rule TEXT NOT NULL,
                average_speed REAL NOT NULL DEFAULT 0,
                direction_change_count INTEGER NOT NULL DEFAULT 0,
                click_pre_dwell_ms REAL NOT NULL DEFAULT 0,
                crossed_client_boundary INTEGER NOT NULL DEFAULT 0,
                raw_point_hash TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_mouse_segments_session_ended ON mouse_compression_segments(session_id, ended_monotonic_ns);
            CREATE TABLE IF NOT EXISTS deferred_exact_scores (
                frame_id TEXT PRIMARY KEY REFERENCES frames(id),
                dhash64 TEXT NOT NULL,
                created REAL NOT NULL,
                updated REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                state TEXT NOT NULL DEFAULT 'pending',
                last_error TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_deferred_exact_scores_state ON deferred_exact_scores(state, created);
            CREATE TABLE IF NOT EXISTS sleep_decision_samples (
                id TEXT PRIMARY KEY,
                created REAL NOT NULL,
                features TEXT NOT NULL,
                expected_gain REAL NOT NULL DEFAULT 0,
                actual_quality_delta REAL,
                cleanup_bytes INTEGER,
                duration_seconds REAL,
                restored_training_gain REAL,
                training_status TEXT NOT NULL DEFAULT '',
                training_reason TEXT NOT NULL DEFAULT '',
                training_samples INTEGER NOT NULL DEFAULT 0,
                validation_samples INTEGER NOT NULL DEFAULT 0,
                outcome_ready INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS frame_capture_audit (
                frame_id TEXT PRIMARY KEY REFERENCES frames(id) ON DELETE CASCADE,
                created REAL NOT NULL,
                session_id TEXT NOT NULL,
                audit_json TEXT NOT NULL,
                validation_before TEXT NOT NULL DEFAULT '{}',
                validation_after TEXT NOT NULL DEFAULT '{}',
                capture_backend TEXT NOT NULL DEFAULT '',
                fallback_reason TEXT NOT NULL DEFAULT '',
                capture_hash_delta REAL NOT NULL DEFAULT 64,
                black_ratio REAL NOT NULL DEFAULT 1,
                variance REAL NOT NULL DEFAULT 0,
                monitor_coverage REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS pipeline_loss_events (
                id TEXT PRIMARY KEY,
                session_id TEXT REFERENCES sessions(id),
                created REAL NOT NULL,
                started REAL NOT NULL,
                ended REAL NOT NULL,
                lost_count INTEGER NOT NULL,
                stage TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS capture_contract_ticks (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                created REAL NOT NULL,
                created_monotonic_ns INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                phash TEXT NOT NULL,
                dhash64 TEXT NOT NULL,
                gray32x18 BLOB,
                edge_density REAL NOT NULL DEFAULT 0,
                color_histogram BLOB,
                online_score REAL,
                score_status TEXT NOT NULL DEFAULT '',
                mouse_x INTEGER,
                mouse_y INTEGER,
                mouse_inside INTEGER NOT NULL DEFAULT 0,
                persisted_png INTEGER NOT NULL DEFAULT 0,
                frame_id TEXT,
                reason TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_capture_contract_ticks_session ON capture_contract_ticks(session_id, created_monotonic_ns);
            """)


    def _migrate(self):
        user_version = int(self.conn.execute("PRAGMA user_version").fetchone()[0] or 0)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS system_events (
            id TEXT PRIMARY KEY,
            session_id TEXT REFERENCES sessions(id),
            created REAL NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS deletion_journal (
            id TEXT PRIMARY KEY,
            object_type TEXT NOT NULL,
            object_id TEXT NOT NULL,
            path TEXT,
            stage TEXT NOT NULL,
            created REAL NOT NULL,
            updated REAL NOT NULL,
            error TEXT
        )
        """)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS mode_transitions (
            id TEXT PRIMARY KEY,
            created REAL NOT NULL,
            from_mode TEXT NOT NULL,
            to_mode TEXT NOT NULL,
            reason TEXT NOT NULL,
            trigger TEXT NOT NULL,
            session_id TEXT,
            window_state TEXT NOT NULL,
            cursor_x INTEGER,
            cursor_y INTEGER,
            resource_state TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mode_transitions_created ON mode_transitions(created)")
        frame_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(frames)").fetchall()}
        additions = dict([
            ("size_bytes", "INTEGER NOT NULL DEFAULT 0"), ("online_score", "REAL"), ("raw_score", "REAL"), ("raw_reward", "REAL"),
            ("reward_source", "TEXT NOT NULL DEFAULT 'screen_score_only'"), ("score_valid_for_training", "INTEGER NOT NULL DEFAULT 0"),
            ("score_status", "TEXT NOT NULL DEFAULT 'invalid'"), ("novelty", "REAL NOT NULL DEFAULT 0"), ("action_result", "REAL NOT NULL DEFAULT 0"),
            ("coverage", "REAL NOT NULL DEFAULT 0"), ("model_refs", "INTEGER NOT NULL DEFAULT 0"), ("retain_value", "REAL NOT NULL DEFAULT 0"),
            ("retain_version", "INTEGER NOT NULL DEFAULT 1"), ("last_used", "REAL NOT NULL DEFAULT 0"), ("dhash64", "TEXT"),
            ("bucket0", "INTEGER NOT NULL DEFAULT 0"), ("bucket1", "INTEGER NOT NULL DEFAULT 0"), ("bucket2", "INTEGER NOT NULL DEFAULT 0"),
            ("bucket3", "INTEGER NOT NULL DEFAULT 0"), ("state_cluster_id", "TEXT"), ("state_support_count", "INTEGER NOT NULL DEFAULT 1"),
            ("action_outcome_information", "REAL NOT NULL DEFAULT 0"), ("model_dependency_count", "INTEGER NOT NULL DEFAULT 0"),
            ("validation_last_used", "REAL NOT NULL DEFAULT 0"), ("created_monotonic_ns", "INTEGER NOT NULL DEFAULT 0"),
            ("capture_backend", "TEXT NOT NULL DEFAULT 'gdi'"), ("capture_elapsed_ms", "REAL NOT NULL DEFAULT 0"),
            ("capture_complete", "INTEGER NOT NULL DEFAULT 1"), ("brightness", "REAL NOT NULL DEFAULT 0"), ("variance", "REAL NOT NULL DEFAULT 0"),
            ("gray32x18", "BLOB"), ("edge_density", "REAL NOT NULL DEFAULT 0"), ("color_histogram", "BLOB"),
            ("asset_ref_count", "INTEGER NOT NULL DEFAULT 1"), ("score_candidate_count", "INTEGER NOT NULL DEFAULT 0"),
            ("score_top_k_distance", "REAL NOT NULL DEFAULT 64"), ("score_retrieval_fallback", "INTEGER NOT NULL DEFAULT 0"),
            ("score_retrieval_mode", "TEXT NOT NULL DEFAULT 'warmup'"), ("score_exact_or_approx", "TEXT NOT NULL DEFAULT 'exact'"),
            ("score_recall_guard", "INTEGER NOT NULL DEFAULT 0"), ("score_valid", "INTEGER NOT NULL DEFAULT 0"), ("accepted_journal_id", "TEXT"),
            ("capture_started_monotonic_ns", "INTEGER NOT NULL DEFAULT 0"), ("capture_finished_monotonic_ns", "INTEGER NOT NULL DEFAULT 0"),
            ("capture_started", "REAL NOT NULL DEFAULT 0"), ("capture_finished", "REAL NOT NULL DEFAULT 0"),
            ("capture_failure_reason", "TEXT NOT NULL DEFAULT ''"), ("capture_hash_delta", "REAL NOT NULL DEFAULT 64"),
            ("capture_fallback", "INTEGER NOT NULL DEFAULT 0"), ("capture_audit_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("score_generation", "TEXT NOT NULL DEFAULT 'online'"), ("history_boundary_frame_id", "TEXT")
        ])
        for name, definition in additions.items():
            if name not in frame_columns:
                self.conn.execute(f"ALTER TABLE frames ADD COLUMN {name} {definition}")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_buckets ON frames(bucket0, bucket1, bucket2, bucket3)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_cluster ON frames(state_cluster_id, state_support_count)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_session_mono ON frames(session_id, created_monotonic_ns)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_session_score_ready ON frames(session_id, capture_complete, score_valid, score_status, capture_finished_monotonic_ns)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_session_finished_id ON frames(session_id, capture_finished_monotonic_ns, id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_finished_id ON frames(capture_finished_monotonic_ns, id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_score_pending ON frames(score_valid, score_status, capture_finished_monotonic_ns)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_history_boundary ON frames(history_boundary_frame_id, score_generation)")
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_frames_accepted_journal ON frames(accepted_journal_id) WHERE accepted_journal_id IS NOT NULL AND accepted_journal_id!=''")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_prune_refs ON frames(model_dependency_count, model_refs, asset_ref_count, retain_value, created)")
        session_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if "trainable" not in session_columns:
            self.conn.execute("ALTER TABLE sessions ADD COLUMN trainable INTEGER NOT NULL DEFAULT 1")
        if "training_exclusion_reason" not in session_columns:
            self.conn.execute("ALTER TABLE sessions ADD COLUMN training_exclusion_reason TEXT NOT NULL DEFAULT ''")
        mouse_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(mouse_events)").fetchall()}
        if "behavior_probability" not in mouse_columns:
            self.conn.execute("ALTER TABLE mouse_events ADD COLUMN behavior_probability REAL")
        for name in ("before_frame_id", "after_frame_id"):
            if name not in mouse_columns:
                self.conn.execute("ALTER TABLE mouse_events ADD COLUMN {} TEXT".format(name))
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mouse_session_mono ON mouse_events(session_id, created_monotonic_ns)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mouse_session_type_mono ON mouse_events(session_id, event_type, created_monotonic_ns)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mouse_after_frame_pending ON mouse_events(session_id, after_frame_id, created_monotonic_ns)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS mouse_loss_events (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, created REAL NOT NULL, started REAL NOT NULL, ended REAL NOT NULL, lost_count INTEGER NOT NULL, rule TEXT NOT NULL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS mouse_compression_segments (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source TEXT NOT NULL, started REAL NOT NULL, ended REAL NOT NULL, started_monotonic_ns INTEGER NOT NULL, ended_monotonic_ns INTEGER NOT NULL, start_x INTEGER NOT NULL, start_y INTEGER NOT NULL, end_x INTEGER NOT NULL, end_y INTEGER NOT NULL, original_count INTEGER NOT NULL, max_speed REAL NOT NULL, path_length REAL NOT NULL, trajectory_blob BLOB NOT NULL DEFAULT X'', trajectory_codec TEXT NOT NULL DEFAULT 'varint-zigzag-dtxy-v1', rule TEXT NOT NULL)")
        segment_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(mouse_compression_segments)").fetchall()}
        if "trajectory_blob" not in segment_columns:
            self.conn.execute("ALTER TABLE mouse_compression_segments ADD COLUMN trajectory_blob BLOB")
        if "trajectory_codec" not in segment_columns:
            self.conn.execute("ALTER TABLE mouse_compression_segments ADD COLUMN trajectory_codec TEXT NOT NULL DEFAULT 'varint-zigzag-dtxy-v1'")
        for name, definition in (("client_left", "INTEGER"), ("client_top", "INTEGER"), ("client_right", "INTEGER"), ("client_bottom", "INTEGER"), ("average_speed", "REAL NOT NULL DEFAULT 0"), ("direction_change_count", "INTEGER NOT NULL DEFAULT 0"), ("click_pre_dwell_ms", "REAL NOT NULL DEFAULT 0"), ("crossed_client_boundary", "INTEGER NOT NULL DEFAULT 0"), ("raw_point_hash", "TEXT NOT NULL DEFAULT ''")):
            if name not in segment_columns:
                self.conn.execute("ALTER TABLE mouse_compression_segments ADD COLUMN {} {}".format(name, definition))
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mouse_segments_session_ended ON mouse_compression_segments(session_id, ended_monotonic_ns)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS deferred_exact_scores (frame_id TEXT PRIMARY KEY REFERENCES frames(id), dhash64 TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'pending', last_error TEXT NOT NULL DEFAULT '')")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_deferred_exact_scores_state ON deferred_exact_scores(state, created)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS capture_contract_ticks (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, created REAL NOT NULL, created_monotonic_ns INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, phash TEXT NOT NULL, dhash64 TEXT NOT NULL, gray32x18 BLOB, edge_density REAL NOT NULL DEFAULT 0, color_histogram BLOB, online_score REAL, score_status TEXT NOT NULL DEFAULT '', mouse_x INTEGER, mouse_y INTEGER, mouse_inside INTEGER NOT NULL DEFAULT 0, persisted_png INTEGER NOT NULL DEFAULT 0, frame_id TEXT, reason TEXT NOT NULL DEFAULT '')")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_capture_contract_ticks_session ON capture_contract_ticks(session_id, created_monotonic_ns)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS sleep_decision_samples (id TEXT PRIMARY KEY, created REAL NOT NULL, features TEXT NOT NULL, expected_gain REAL NOT NULL DEFAULT 0, actual_quality_delta REAL, cleanup_bytes INTEGER, duration_seconds REAL, restored_training_gain REAL, training_status TEXT NOT NULL DEFAULT '', training_reason TEXT NOT NULL DEFAULT '', training_samples INTEGER NOT NULL DEFAULT 0, validation_samples INTEGER NOT NULL DEFAULT 0, outcome_ready INTEGER NOT NULL DEFAULT 0)")
        sleep_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(sleep_decision_samples)").fetchall()}
        for name, definition in (("training_status", "TEXT NOT NULL DEFAULT ''"), ("training_reason", "TEXT NOT NULL DEFAULT ''"), ("training_samples", "INTEGER NOT NULL DEFAULT 0"), ("validation_samples", "INTEGER NOT NULL DEFAULT 0")):
            if name not in sleep_columns:
                self.conn.execute("ALTER TABLE sleep_decision_samples ADD COLUMN {} {}".format(name, definition))
        self.conn.execute("CREATE TABLE IF NOT EXISTS frame_capture_audit (frame_id TEXT PRIMARY KEY REFERENCES frames(id) ON DELETE CASCADE, created REAL NOT NULL, session_id TEXT NOT NULL, audit_json TEXT NOT NULL, validation_before TEXT NOT NULL DEFAULT '{}', validation_after TEXT NOT NULL DEFAULT '{}', capture_backend TEXT NOT NULL DEFAULT '', fallback_reason TEXT NOT NULL DEFAULT '', capture_hash_delta REAL NOT NULL DEFAULT 64, black_ratio REAL NOT NULL DEFAULT 1, variance REAL NOT NULL DEFAULT 0, monitor_coverage REAL NOT NULL DEFAULT 0)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS pipeline_loss_events (id TEXT PRIMARY KEY, session_id TEXT, created REAL NOT NULL, started REAL NOT NULL, ended REAL NOT NULL, lost_count INTEGER NOT NULL, stage TEXT NOT NULL, reason TEXT NOT NULL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS deletion_journal (id TEXT PRIMARY KEY, object_type TEXT NOT NULL, object_id TEXT NOT NULL, path TEXT, stage TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, error TEXT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS pool_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
        self.conn.execute("INSERT OR IGNORE INTO pool_meta(key, value) VALUES ('total_asset_bytes', COALESCE((SELECT SUM(size_bytes) FROM frames), 0))")
        self.conn.execute("INSERT OR IGNORE INTO pool_meta(key, value) VALUES ('asset_bytes', COALESCE((SELECT SUM(size_bytes) FROM frames), 0))")
        for key in ("reserved_asset_bytes", "database_bytes", "transient_bytes", "other_bytes", "last_reconciled_at", "pool_capacity_blocked", "pool_capacity_target", "pool_capacity_remaining", "pool_capacity_updated"):
            self.conn.execute("INSERT OR IGNORE INTO pool_meta(key, value) VALUES (?, 0)", (key,))
        self.conn.execute("UPDATE pool_meta SET value=COALESCE((SELECT value FROM pool_meta WHERE key='total_asset_bytes'), value) WHERE key='asset_bytes' AND value=0")
        self.conn.execute("CREATE TABLE IF NOT EXISTS state_clusters (cluster_id TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS frame_lsh (key INTEGER NOT NULL, frame_id TEXT NOT NULL, PRIMARY KEY(key, frame_id))")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frame_lsh_key ON frame_lsh(key)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS ingestion_journal (id TEXT PRIMARY KEY, object_type TEXT NOT NULL, object_id TEXT NOT NULL, path TEXT, stage TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, error TEXT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS action_outcomes (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), mouse_event_id TEXT NOT NULL, before_frame_id TEXT NOT NULL, after_frame_id TEXT NOT NULL, action_time INTEGER NOT NULL, post_action_delay_ms REAL NOT NULL, score_delta REAL NOT NULL, reward_delta REAL NOT NULL, outcome_valid INTEGER NOT NULL)")
        action_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(action_outcomes)").fetchall()}
        action_additions = {"action_id": "TEXT", "split_role": "TEXT NOT NULL DEFAULT 'unknown'", "hunger_delta_expected": "REAL NOT NULL DEFAULT 0", "baseline_score_delta": "REAL NOT NULL DEFAULT 0", "expected_no_action_reward_delta": "REAL NOT NULL DEFAULT 0", "action_advantage": "REAL NOT NULL DEFAULT 0", "stability": "REAL NOT NULL DEFAULT 0", "baseline_count": "INTEGER NOT NULL DEFAULT 0"}
        for name, definition in action_additions.items():
            if name not in action_columns:
                self.conn.execute(f"ALTER TABLE action_outcomes ADD COLUMN {name} {definition}")
        self.conn.execute("UPDATE action_outcomes SET action_id=COALESCE(action_id, mouse_event_id) WHERE action_id IS NULL OR action_id='' ")
        self.conn.execute("DELETE FROM action_outcomes WHERE rowid NOT IN (SELECT MAX(rowid) FROM action_outcomes GROUP BY action_id)")
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_action_outcomes_action_id ON action_outcomes(action_id)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS model_frame_refs (model_id TEXT NOT NULL, frame_id TEXT NOT NULL REFERENCES frames(id), role TEXT NOT NULL, PRIMARY KEY(model_id, frame_id, role))")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_model_frame_refs_frame ON model_frame_refs(frame_id)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS model_metadata (id TEXT PRIMARY KEY, file_name TEXT NOT NULL UNIQUE, created REAL NOT NULL, quality REAL NOT NULL DEFAULT 0, validation_quality REAL NOT NULL DEFAULT 0, champion INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL)")
        current_champion = self.conn.execute("SELECT id FROM model_metadata WHERE champion=1 ORDER BY validation_quality DESC, quality DESC, updated DESC, id DESC LIMIT 1").fetchone()
        if current_champion:
            self.conn.execute("UPDATE model_metadata SET champion=CASE WHEN id=? THEN 1 ELSE 0 END", (current_champion[0],))
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_model_metadata_one_champion ON model_metadata(champion) WHERE champion=1")
        missing_lsh = self.conn.execute("SELECT COUNT(*) FROM frames WHERE (dhash64 IS NOT NULL OR phash IS NOT NULL) AND id NOT IN (SELECT frame_id FROM frame_lsh)").fetchone()[0]
        if missing_lsh:
            for fid, dhash, phash in self.conn.execute("SELECT id, dhash64, phash FROM frames WHERE (dhash64 IS NOT NULL OR phash IS NOT NULL) AND id NOT IN (SELECT frame_id FROM frame_lsh)").fetchall():
                value = dhash or phash
                self.conn.executemany("INSERT OR IGNORE INTO frame_lsh(key, frame_id) VALUES (?, ?)", [(key, fid) for key in self._hash_buckets(value)])
        self.conn.execute("UPDATE frames SET score_status='exact' WHERE score_valid=1 AND score IS NOT NULL AND score_status!='exact'")
        self.conn.execute("UPDATE frames SET score_status='provisional' WHERE score_valid=0 AND (score_status IN ('pending_exact','unknown','valid') OR score_status IS NULL) AND (online_score IS NOT NULL OR dhash64 IS NOT NULL OR phash IS NOT NULL)")
        self.conn.execute("UPDATE frames SET score_status='invalid' WHERE score_valid=0 AND (score_status IN ('warmup_exact','unknown') OR score_status IS NULL) AND online_score IS NULL")
        self.conn.execute("UPDATE frames SET hunger=0.0, reward=score, raw_score=score, raw_reward=score, reward_source='screen_score_only', score_valid_for_training=CASE WHEN score_status='exact' THEN 1 ELSE 0 END WHERE score_valid=1 AND score IS NOT NULL")
        self.conn.execute("UPDATE frames SET hunger=0.0, reward=NULL, raw_reward=NULL, reward_source='screen_score_only', score_valid_for_training=0 WHERE score_valid=0 OR score IS NULL")
        self._sync_model_metadata_locked()
        self._recalculate_model_refs_locked([row[0] for row in self.conn.execute("SELECT id FROM frames WHERE model_dependency_count!=0 OR model_refs!=0").fetchall()])
        self.conn.execute("PRAGMA user_version={}".format(self.SCHEMA_VERSION))

    def close(self):
        with self.lock:
            if self.conn is not None:
                try:
                    try:
                        self.conn.interrupt()
                    except Exception:
                        pass
                    self.conn.commit()
                    self.conn.close()
                except Exception:
                    pass
            self.conn = None
            self._database_bytes_cached = 0
            self._database_bytes_checked = 0.0
            self.root = None
            self.pool = None
            self.models = None
            self.screens = None

    def create_session(self, mode):
        delegated, result = self._dispatch_write_to_writer("create_session", mode)
        if delegated:
            return result
        identifier = uuid.uuid4().hex
        now = time.time()
        with self.lock:
            self.conn.execute("INSERT INTO sessions(id, mode, started) VALUES (?, ?, ?)", (identifier, mode, now))
            self.conn.commit()
        return identifier

    def _trim_system_events_locked(self):
        self.conn.execute("DELETE FROM system_events WHERE id IN (SELECT id FROM system_events ORDER BY created DESC, id DESC LIMIT -1 OFFSET 500)")

    def mark_session_untrainable(self, session_id, reason):
        delegated, result = self._dispatch_write_to_writer("mark_session_untrainable", session_id, reason)
        if delegated:
            return result
        with self.lock:
            if self.conn is None or not session_id:
                return
            self.conn.execute("UPDATE sessions SET trainable=0, training_exclusion_reason=? WHERE id=?", (str(reason), str(session_id)))
            self.conn.execute("INSERT INTO system_events(id, session_id, created, kind, payload) VALUES (?, ?, ?, ?, ?)", (uuid.uuid4().hex, str(session_id), time.time(), "session_forced_untrainable", json.dumps({"reason": str(reason)}, ensure_ascii=False)))
            self._trim_system_events_locked()
            self.conn.commit()

    def _accepted_journal_path(self):
        return self.journal / "accepted.jsonl" if self.journal is not None else None

    def _append_accepted_journal_line(self, item, durable=True):
        path = self._accepted_journal_path()
        if path is None:
            return None
        path = self._assert_storage_path(path, "accepted_journal")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(item or {})
        payload["id"] = str(payload.get("id") or uuid.uuid4().hex)
        payload["state"] = str(payload.get("state") or "accepted")
        payload["created"] = float(payload.get("created") or time.time())
        payload["attempts"] = max(0, int(payload.get("attempts", 0) or 0))
        payload["last_error"] = str(payload.get("last_error", payload.get("error", "")) or "")
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        size = len(line.encode("utf-8", errors="ignore"))
        now = time.monotonic()
        with self._journal_lock:
            self._journal_mouse_bytes += size
            need_fsync = bool(durable) or self._journal_mouse_bytes >= 32768 or now - self._journal_mouse_last_fsync >= 0.10
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                if need_fsync:
                    os.fsync(handle.fileno())
                    self._journal_mouse_bytes = 0
                    self._journal_mouse_last_fsync = now
        return payload.get("id")

    def _journal_json_value(self, value):
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bytes):
            return {"__bytes_hex__": value.hex()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, tuple):
            return [self._journal_json_value(item) for item in value]
        if isinstance(value, list):
            return [self._journal_json_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._journal_json_value(item) for key, item in value.items()}
        return str(value)

    def _journal_restore_value(self, value):
        if isinstance(value, dict) and set(value.keys()) == {"__bytes_hex__"}:
            try:
                return bytes.fromhex(str(value.get("__bytes_hex__") or ""))
            except ValueError:
                return b""
        if isinstance(value, dict):
            return {str(key): self._journal_restore_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._journal_restore_value(item) for item in value]
        return value

    def _journal_frame_metadata(self, image):
        return {str(key): self._journal_json_value(value) for key, value in dict(image or {}).items() if key not in ("png", "bgra", "rgb")}

    def journal_frame_packet(self, packet):
        if self.journal is None or not isinstance(packet, dict) or not isinstance(packet.get("image"), dict):
            return None
        image = packet.get("image")
        png = image.get("png")
        if isinstance(png, memoryview):
            png = png.tobytes()
        if isinstance(png, bytearray):
            png = bytes(png)
        if not isinstance(png, bytes) or not png:
            return None
        jid = packet.get("accepted_journal_id") or image.get("accepted_journal_id") or uuid.uuid4().hex
        png_path = self._assert_storage_path(self.journal / "frames" / (jid + ".png"), "journal_frame_png")
        meta_path = self._assert_storage_path(self.journal / "frames" / (jid + ".json"), "journal_frame_json")
        payload = {"session_id": packet.get("session_id"), "image": self._journal_frame_metadata(image), "phash": image.get("phash"), "online_score": packet.get("online_score"), "exact_score": packet.get("exact_score"), "score": packet.get("exact_score")}
        _atomic_write_bytes(png_path, png, self.root)
        _atomic_write_bytes(meta_path, json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), self.root)
        self._append_accepted_journal_line({"id": jid, "kind": "frame", "state": "accepted", "created": time.time(), "png_path": str(png_path.relative_to(self.pool)), "meta_path": str(meta_path.relative_to(self.pool)), "session_id": str(packet.get("session_id") or "")}, durable=True)
        packet["accepted_journal_id"] = jid
        image["accepted_journal_id"] = jid
        return jid

    def journal_mouse_record(self, record, durable=None):
        if self.journal is None or not isinstance(record, dict):
            return None
        jid = record.get("accepted_journal_id") or uuid.uuid4().hex
        record["accepted_journal_id"] = jid
        critical = str(record.get("event_type") or "") != "move" or bool(record.get("button")) or bool(record.get("wheel"))
        durable = critical if durable is None else bool(durable)
        self._append_accepted_journal_line({"id": jid, "kind": "mouse", "state": "accepted", "created": time.time(), "record": record}, durable=durable)
        return jid

    def journal_mouse_segment(self, segment, durable=True):
        if self.journal is None or not isinstance(segment, dict):
            return None
        jid = segment.get("accepted_journal_id") or uuid.uuid4().hex
        segment["accepted_journal_id"] = jid
        self._append_accepted_journal_line({"id": jid, "kind": "mouse_segment", "state": "accepted", "created": time.time(), "segment": self._journal_json_value(segment)}, durable=durable)
        return jid

    def complete_accepted_journal(self, jid):
        if not jid:
            return
        self._append_accepted_journal_line({"id": str(jid), "state": "completed", "created": time.time()})
        if self.journal is None:
            return
        for suffix in (".png", ".json", ".pickle"):
            try:
                self._safe_unlink_storage(self.journal / "frames" / (str(jid) + suffix), "journal_complete_unlink")
            except OSError:
                pass

    def _journal_mark_replayed(self, jid, attempts=0):
        if jid:
            self._append_accepted_journal_line({"id": str(jid), "state": "replayed", "created": time.time(), "attempts": max(0, int(attempts or 0))}, durable=True)

    def _journal_mark_failed(self, jid, error, attempts):
        state = "quarantined" if int(attempts or 0) >= 3 else "failed"
        self._append_accepted_journal_line({"id": str(jid), "state": state, "created": time.time(), "attempts": max(0, int(attempts or 0)), "last_error": str(error)}, durable=True)
        try:
            self.add_system_event(None, "accepted_journal_" + state, {"journal_id": str(jid), "attempts": int(attempts or 0), "error": str(error)})
        except Exception:
            pass

    def _journal_payload_path(self, relative, operation):
        if self.pool is None or not relative:
            return None
        try:
            candidate = self._assert_storage_path((self.pool / str(relative)).resolve(), operation)
            if self.pool.resolve() not in candidate.parents:
                return None
            return candidate
        except OSError:
            return None

    def _accepted_journal_reduce(self):
        path = self._accepted_journal_path()
        states = {}
        entries = {}
        if path is None or not path.exists():
            return states, entries
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                item = json.loads(line)
            except Exception:
                continue
            jid = str(item.get("id") or "")
            if not jid:
                continue
            state = str(item.get("state") or "accepted")
            previous = states.get(jid, {})
            merged = dict(previous)
            merged.update(item)
            merged["state"] = state
            merged["attempts"] = max(int(previous.get("attempts", 0) or 0), int(item.get("attempts", 0) or 0))
            if item.get("last_error") or item.get("error"):
                merged["last_error"] = str(item.get("last_error", item.get("error", "")) or "")
            states[jid] = merged
            if item.get("kind"):
                entries[jid] = dict(item)
        return states, entries

    def compact_accepted_journal(self):
        path = self._accepted_journal_path()
        if path is None or not path.exists():
            return 0
        states, entries = self._accepted_journal_reduce()
        keep = []
        for jid, state_item in sorted(states.items(), key=lambda pair: float(pair[1].get("created", 0) or 0)):
            state = str(state_item.get("state") or "")
            if state in {"completed", "complete", "replayed"}:
                continue
            item = dict(entries.get(jid, {}))
            item.update(state_item)
            if item.get("kind") or state in {"failed", "quarantined", "accepted"}:
                keep.append(item)
        payload = "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in keep).encode("utf-8")
        _atomic_write_bytes(path, payload, self.root)
        return len(keep)

    def replay_accepted_journal(self, experience_limit=None, kinds=None, max_items=None):
        path = self._accepted_journal_path()
        if path is None or not path.exists():
            return 0
        wanted = {str(item) for item in kinds} if kinds else None
        states, entries = self._accepted_journal_reduce()
        restored = 0
        for jid, item in list(entries.items()):
            state_item = states.get(jid, {})
            state = str(state_item.get("state") or item.get("state") or "")
            attempts = max(0, int(state_item.get("attempts", item.get("attempts", 0)) or 0))
            if state in {"completed", "complete", "replayed", "quarantined"}:
                continue
            if wanted is not None and str(item.get("kind") or "") not in wanted:
                continue
            if max_items is not None and restored >= int(max_items):
                break
            try:
                if item.get("kind") == "mouse":
                    record = item.get("record") or {}
                    if record:
                        self.save_mouse_batch([record])
                        self._journal_mark_replayed(jid, attempts)
                        self.complete_accepted_journal(jid)
                        restored += 1
                elif item.get("kind") == "mouse_segment":
                    segment = self._journal_restore_value(item.get("segment") or {})
                    if segment:
                        self.record_mouse_compression(segment)
                        self._journal_mark_replayed(jid, attempts)
                        self.complete_accepted_journal(jid)
                        restored += 1
                elif item.get("kind") == "frame":
                    with self.lock:
                        exists = self.conn.execute("SELECT id FROM frames WHERE accepted_journal_id=?", (jid,)).fetchone() if self.conn is not None else None
                    if exists:
                        self.complete_accepted_journal(jid)
                        continue
                    legacy = str(item.get("path") or "")
                    if legacy.lower().endswith(".pickle"):
                        self._journal_mark_failed(jid, "legacy_pickle_journal_ignored", 3)
                        continue
                    meta_path = self._journal_payload_path(item.get("meta_path"), "journal_replay_json")
                    png_path = self._journal_payload_path(item.get("png_path"), "journal_replay_png")
                    if meta_path is None or png_path is None or not meta_path.exists() or not png_path.exists():
                        self._journal_mark_failed(jid, "journal_payload_missing", attempts + 1)
                        continue
                    payload = json.loads(meta_path.read_text(encoding="utf-8"))
                    image = self._journal_restore_value(payload.get("image") or {})
                    image["png"] = png_path.read_bytes()
                    image["accepted_journal_id"] = jid
                    if image.get("png"):
                        self.save_frame(payload.get("session_id"), image, image.get("phash") or payload.get("phash") or "", online_score=payload.get("online_score"), exact_score=payload.get("exact_score"), hunger=0.0 if payload.get("exact_score") is not None else None, reward=payload.get("exact_score"), experience_limit=experience_limit)
                        self._journal_mark_replayed(jid, attempts)
                        self.complete_accepted_journal(jid)
                        restored += 1
            except Exception as error:
                self._journal_mark_failed(jid, error, attempts + 1)
        try:
            self.compact_accepted_journal()
        except Exception:
            pass
        return restored

    def accepted_journal_backlog(self):
        result = {"total": 0, "frame": 0, "mouse": 0, "mouse_segment": 0, "failed": 0, "quarantined": 0}
        try:
            states, entries = self._accepted_journal_reduce()
        except Exception:
            return result
        for jid, state_item in states.items():
            state = str(state_item.get("state") or "")
            if state in {"completed", "complete", "replayed"}:
                continue
            kind = str((entries.get(jid) or state_item).get("kind") or "")
            result["total"] += 1
            if kind in result:
                result[kind] += 1
            if state == "failed":
                result["failed"] += 1
            if state == "quarantined":
                result["quarantined"] += 1
        return result

    def recover_after_forced_stop(self):
        try:
            self.compact_accepted_journal()
            self.replay_accepted_journal()
            self.recover_ingestions()
            self.recover_deletions()
            self._cleanup_pool_files()
            self.reconcile_pool_ledger()
            return self.validate_consistency()
        except Exception as error:
            return False, str(error)

    def preflight_persistence_check(self, mode="learning"):
        delegated, result = self._dispatch_write_to_writer("preflight_persistence_check", mode)
        if delegated:
            return result
        if self.conn is None or self.pool is None or self.screens is None or self.journal is None:
            raise RuntimeError("存储未打开")
        stage = "init"
        sid = "preflight_" + uuid.uuid4().hex
        frame_id = None
        root = self.root
        png = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c63f8ffff3f0005fe02fe41e28cd50000000049454e44ae426082")
        now = time.time()
        mono = time.monotonic_ns()
        image = {"width": 1, "height": 1, "png": png, "phash": "0" * 16, "dhash64": "0" * 16, "capture_started_monotonic_ns": mono - 1000000, "capture_finished_monotonic_ns": mono, "capture_started": now - 0.001, "capture_finished": now, "capture_backend": "preflight", "capture_elapsed_ms": 0.1, "capture_complete": 1, "content_valuable": 1, "brightness": 128.0, "variance": 1.0, "gray32x18": bytes([128] * (32 * 18)), "edge_density": 0.0, "color_histogram": struct.pack("<24I", *([1] * 24)), "score_candidate_count": 0, "score_top_k_distance": 64.0, "score_retrieval_fallback": 0, "score_retrieval_mode": "preflight_exact", "score_exact_or_approx": "exact", "score_recall_guard": 1, "score_valid": True, "score_provisional": False, "score_status": "exact", "score_valid_for_training": 1, "reward_source": "screen_score_only"}
        mouse = {"session_id": sid, "created": now, "created_monotonic_ns": mono - 1, "source": "preflight", "event_type": "move", "button": "", "wheel": 0, "x": 0, "y": 0, "relative_x": 0.0, "relative_y": 0.0, "dx": 0.0, "dy": 0.0, "direction": 0.0, "speed": 0.0, "behavior_probability": None, "before_frame_id": None, "after_frame_id": None}
        try:
            stage = "create_session"
            with self.lock:
                self.conn.execute("INSERT INTO sessions(id, mode, started, trainable, training_exclusion_reason) VALUES (?, ?, ?, 0, 'preflight')", (sid, str(mode), now))
                self.conn.commit()
            stage = "journal_mouse"
            self.journal_mouse_record(dict(mouse), durable=True)
            stage = "write_mouse"
            self.save_mouse_batch([mouse])
            stage = "journal_frame"
            packet = {"token": 0, "session_id": sid, "image": dict(image), "online_score": 0.5, "exact_score": 0.5, "hunger": 0.0, "reward": 0.5}
            jid = self.journal_frame_packet(packet)
            if not jid:
                raise RuntimeError("accepted journal 未写入")
            stage = "close_reopen_replay"
            self.close()
            self.ensure(root)
            replayed = self.replay_accepted_journal(kinds=("frame",), max_items=16)
            stage = "validate_replay"
            with self.lock:
                frame = self.conn.execute("SELECT id, score, reward, reward_source, score_status, score_valid, score_valid_for_training, accepted_journal_id FROM frames WHERE session_id=? ORDER BY created DESC LIMIT 1", (sid,)).fetchone()
                if not frame:
                    raise RuntimeError("frame 未通过 accepted journal replay 落库")
                frame_id = str(frame[0])
                self.bind_mouse_events_after_frame(sid, frame_id, mono)
                bound = int(self.conn.execute("SELECT COUNT(*) FROM mouse_events WHERE session_id=? AND after_frame_id=?", (sid, frame_id)).fetchone()[0] or 0)
                session_row = self.conn.execute("SELECT frame_count, mouse_count, trainable FROM sessions WHERE id=?", (sid,)).fetchone()
                ledger = self._ledger_values_locked()
            if abs(float(frame[1]) - 0.5) > 1e-9 or abs(float(frame[2]) - 0.5) > 1e-9 or str(frame[3]) != "screen_score_only" or str(frame[4]) != "exact" or int(frame[5]) != 1 or int(frame[6]) != 1:
                raise RuntimeError("frame、score、reward ledger 不一致")
            if bound <= 0:
                raise RuntimeError("mouse 未绑定到 frame")
            if not session_row or int(session_row[0]) < 1 or int(session_row[1]) < 1:
                raise RuntimeError("session ledger 不一致")
            if int(ledger.get("asset_bytes", 0)) < 0 or int(ledger.get("reserved_asset_bytes", 0)) < 0:
                raise RuntimeError("capacity ledger 不一致")
            stage = "cleanup"
            self._cleanup_preflight_data(sid)
            return True, "完整流水线持久化预检通过"
        except Exception as error:
            try:
                if self.conn is not None:
                    self.conn.rollback()
            except Exception:
                pass
            try:
                if self.conn is None and root is not None:
                    self.ensure(root)
                self._cleanup_preflight_data(sid)
            except Exception:
                pass
            try:
                self.add_critical_exception("DataStore", "preflight_persistence_check", error, payload={"stage": stage, "session_id": sid, "frame_id": frame_id})
            except Exception:
                pass
            raise RuntimeError("进入学习/训练模式前完整流水线持久化预检失败：{}：{}".format(stage, error))

    def _cleanup_preflight_data(self, session_id):
        if self.conn is None or self.pool is None or not session_id:
            return
        with self.lock:
            paths = [row[0] for row in self.conn.execute("SELECT screenshot_path FROM frames WHERE session_id=?", (session_id,)).fetchall()]
            frame_ids = [row[0] for row in self.conn.execute("SELECT id FROM frames WHERE session_id=?", (session_id,)).fetchall()]
            for frame_id in frame_ids:
                self.conn.execute("DELETE FROM frame_lsh WHERE frame_id=?", (frame_id,))
                self.conn.execute("DELETE FROM deferred_exact_scores WHERE frame_id=?", (frame_id,))
                self.conn.execute("DELETE FROM frame_capture_audit WHERE frame_id=?", (frame_id,))
            self.conn.execute("DELETE FROM mouse_events WHERE session_id=?", (session_id,))
            self.conn.execute("DELETE FROM frames WHERE session_id=?", (session_id,))
            self.conn.execute("DELETE FROM ingestion_journal WHERE object_id IN ({})".format(",".join("?" for _ in frame_ids)) if frame_ids else "DELETE FROM ingestion_journal WHERE 0", tuple(frame_ids))
            self.conn.execute("DELETE FROM system_events WHERE session_id=?", (session_id,))
            self.conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self.conn.commit()
        for relative in paths:
            try:
                if relative:
                    self._safe_unlink_storage(self.pool / str(relative), "preflight_cleanup_frame")
            except Exception:
                pass
        try:
            folder = self._assert_storage_path(self.screens / session_id, "preflight_cleanup_folder")
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()
        except Exception:
            pass
        try:
            self.reconcile_pool_ledger()
        except Exception:
            pass

    def close_session(self, session_id, reason):
        delegated, result = self._dispatch_write_to_writer("close_session", session_id, reason)
        if delegated:
            return result
        with self.lock:
            if self.conn is None or not session_id:
                return
            self.conn.execute("UPDATE sessions SET ended=?, reason=? WHERE id=?", (time.time(), reason, session_id))
            self.conn.commit()

    def add_system_event(self, session_id, kind, payload):
        delegated, result = self._dispatch_write_to_writer("add_system_event", session_id, kind, payload)
        if delegated:
            return result
        with self.lock:
            if self.conn is None:
                return
            self.conn.execute("INSERT INTO system_events(id, session_id, created, kind, payload) VALUES (?, ?, ?, ?, ?)", (uuid.uuid4().hex, session_id, time.time(), kind, json.dumps(payload, ensure_ascii=False)))
            self._trim_system_events_locked()
            self.conn.commit()

    def record_mode_transition(self, session_id, source, target, reason, trigger, window_state, cursor, resource_state, payload):
        delegated, result = self._dispatch_write_to_writer("record_mode_transition", session_id, source, target, reason, trigger, window_state, cursor, resource_state, payload)
        if delegated:
            return result
        with self.lock:
            if self.conn is None:
                return
            x, y = cursor if isinstance(cursor, tuple) and len(cursor) == 2 else (None, None)
            self.conn.execute("INSERT INTO mode_transitions(id, created, from_mode, to_mode, reason, trigger, session_id, window_state, cursor_x, cursor_y, resource_state, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, time.time(), str(source), str(target), str(reason or ""), str(trigger or ""), session_id, str(window_state or ""), x, y, str(resource_state or ""), json.dumps(payload or {}, ensure_ascii=False)))
            self._commit_critical_locked()

    def record_exception_event(self, session_id, kind, error, payload=None):
        delegated, result = self._dispatch_write_to_writer("record_exception_event", session_id, kind, error, payload)
        if delegated:
            return result
        data = dict(payload or {})
        data["error"] = str(error)
        data["error_type"] = type(error).__name__
        self.add_system_event(session_id, kind, data)

    def add_critical_exception(self, module, function, error, session_id=None, token=None, resource_state=None, payload=None):
        delegated, result = self._dispatch_write_to_writer("add_critical_exception", module, function, error, session_id, token, resource_state, payload)
        if delegated:
            return result
        data = dict(payload or {})
        message = str(error)
        if isinstance(error, sqlite3.OperationalError) and ("locked" in message.lower() or "busy" in message.lower()):
            self._sqlite_busy_count += 1
        data.update({"module": str(module), "function": str(function), "exception_type": type(error).__name__, "message": message, "session_id": session_id, "token": token, "resource_state": resource_state, "time": time.time(), "errno": getattr(error, "errno", None), "winerror": getattr(error, "winerror", None), "filename": str(getattr(error, "filename", "") or ""), "filename2": str(getattr(error, "filename2", "") or ""), "sqlite_busy_count": int(self._sqlite_busy_count)})
        try:
            if self.root is not None:
                data["disk_free"] = int(shutil.disk_usage(self.root).free)
        except Exception:
            pass
        try:
            if self.conn is not None:
                data["wal"] = self.wal_metrics()
                data["capacity"] = self.capacity_status()
                data["journal_backlog"] = self.accepted_journal_backlog()
        except Exception:
            pass
        self.add_system_event(session_id, "critical_exception", data)

    def recent_critical_errors(self, limit=5):
        with self.lock:
            if self.conn is None:
                return []
            rows = self.conn.execute("SELECT created, kind, payload FROM system_events WHERE kind='critical_exception' OR kind LIKE '%failed%' OR payload LIKE '%exception_type%' OR payload LIKE '%error_type%' ORDER BY created DESC LIMIT ?", (max(1, int(limit)),)).fetchall()
        result = []
        for created, kind, payload in rows:
            try:
                data = json.loads(payload)
            except Exception:
                data = {"message": str(payload)}
            result.append({"created": float(created or 0.0), "kind": str(kind), "payload": data})
        return result

    def assign_state_cluster(self, dhash, width=None, height=None, gray32x18=None, edge_density=0.0):
        buckets = self._hash_buckets(dhash)
        aspect = 0.0 if not width or not height else round(float(width) / max(1, float(height)), 1)
        edge = int(max(0, min(9, round(float(edge_density or 0.0) * 10))))
        return "{:04x}_{:04x}_{:.1f}_{}".format(buckets[0] & 0xFFFF, buckets[2] & 0xFFFF, aspect, edge)

    def _hash_buckets(self, phash):
        value = int(phash, 16)
        parts = tuple((value >> shift) & 0xFFFF for shift in (48, 32, 16, 0))
        return tuple((index << 16) | part for index, part in enumerate(parts))

    def _bucket_variants(self, part, radius):
        values = [part]
        if radius <= 0:
            return values
        bits = range(16)
        if radius >= 1:
            for bit in bits:
                values.append(part ^ (1 << bit))
        if radius >= 2:
            for first in range(16):
                for second in range(first + 1, 16):
                    values.append(part ^ (1 << first) ^ (1 << second))
        if radius >= 3:
            for first in range(16):
                for second in range(first + 1, 16):
                    for third in range(second + 1, 16):
                        values.append(part ^ (1 << first) ^ (1 << second) ^ (1 << third))
        return values

    def _read_connection(self):
        with self.lock:
            if self.pool is None or self.conn is None:
                return None
            path = self.pool / "records.sqlite3"
        try:
            return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0)
        except (OSError, sqlite3.Error):
            return None

    def _rank_hash_rows(self, current, rows, limit):
        heap = []
        seen = set()
        for row in rows:
            stored = row[1] or row[2] if len(row) >= 3 else row[0]
            frame_id = row[0] if len(row) >= 3 else stored
            if not stored or frame_id in seen:
                continue
            try:
                distance = bit_count(current ^ int(stored, 16))
            except Exception:
                continue
            seen.add(frame_id)
            item = (-distance, str(frame_id), row)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif distance < -heap[0][0]:
                heapq.heapreplace(heap, item)
        ranked = sorted([item[2] for item in heap], key=lambda value: (bit_count(current ^ int((value[1] or value[2]), 16)), str(value[0])))
        return ranked, len(seen)

    def _frame_composite_entry(self, current_dhash, current_features, row):
        current_phash = str((current_features or {}).get("phash") or "")
        current_gray = feature_bytes((current_features or {}).get("gray32x18"), 32 * 18)
        try:
            current_edge = float((current_features or {}).get("edge_density", 0.0) or 0.0)
        except (TypeError, ValueError):
            current_edge = 0.0
        current_hist = histogram_values((current_features or {}).get("color_histogram"))
        frame_id, stored_dhash, stored_phash = row[0], row[1], row[2]
        stored = stored_dhash or stored_phash
        if not stored:
            return None
        try:
            d_distance = bit_count(int(current_dhash, 16) ^ int(stored, 16)) / 64.0
        except Exception:
            return None
        d_similarity = 1.0 - d_distance
        p_similarity = d_similarity
        if current_phash and stored_phash:
            try:
                p_similarity = 1.0 - bit_count(int(current_phash, 16) ^ int(stored_phash, 16)) / 64.0
            except Exception:
                p_similarity = d_similarity
        gray_similarity = d_similarity
        stored_gray = feature_bytes(row[3] if len(row) > 3 else None, 32 * 18)
        if current_gray and stored_gray and len(current_gray) == len(stored_gray):
            gray_similarity = 1.0 - sum(abs(a - b) for a, b in zip(current_gray, stored_gray)) / (255.0 * len(current_gray))
        edge_similarity = 1.0
        if len(row) > 4:
            try:
                edge_similarity = max(0.0, 1.0 - min(1.0, abs(current_edge - float(row[4] or 0.0)) * 2.0))
            except (TypeError, ValueError):
                edge_similarity = d_similarity
        color_similarity = d_similarity
        stored_hist = histogram_values(row[5] if len(row) > 5 else None)
        if current_hist and stored_hist and len(stored_hist) == len(current_hist) and sum(current_hist) > 0 and sum(stored_hist) > 0:
            left_total = float(sum(current_hist))
            right_total = float(sum(stored_hist))
            color_similarity = max(0.0, 1.0 - min(1.0, sum(abs(a / left_total - b / right_total) for a, b in zip(current_hist, stored_hist)) / 2.0))
        similarity = max(0.0, min(1.0, 0.34 * d_similarity + 0.24 * p_similarity + 0.24 * gray_similarity + 0.09 * edge_similarity + 0.09 * color_similarity))
        return {"frame_id": str(frame_id), "hash": str(stored), "similarity": similarity, "distance": d_distance * 64.0, "row": row}

    def _frame_similarity_entries(self, current_dhash, current_features, rows, limit):
        entries = []
        seen = set()
        for row in rows:
            try:
                entry = self._frame_composite_entry(current_dhash, current_features, row)
                if entry is None or entry["frame_id"] in seen:
                    continue
                seen.add(entry["frame_id"])
                entries.append(entry)
            except Exception:
                continue
        entries.sort(key=lambda item: (-item["similarity"], item["distance"], item["frame_id"]))
        return entries[:max(1, int(limit))]

    def _frame_similarity_rows(self, current_dhash, current_features, rows, limit):
        return [(entry["similarity"], entry["hash"], entry["distance"]) for entry in self._frame_similarity_entries(current_dhash, current_features, rows, limit)]

    def _lsh_candidate_rows(self, connection, keys, cap, exclude_frame_id, before_capture_finished_ns):
        rows = []
        truncated = False
        for start in range(0, len(keys), 900):
            chunk = keys[start:start + 900]
            if not chunk:
                continue
            marks = ",".join("?" for _ in chunk)
            fetched = connection.execute(
                "SELECT DISTINCT frames.id, frames.dhash64, frames.phash, frames.gray32x18, frames.edge_density, frames.color_histogram FROM frame_lsh "
                "JOIN frames ON frames.id=frame_lsh.frame_id "
                "WHERE frame_lsh.key IN ({}) AND frames.id <> ? AND frames.capture_finished_monotonic_ns < ? "
                "ORDER BY frames.capture_finished_monotonic_ns DESC, frames.id DESC LIMIT ?".format(marks),
                tuple(chunk) + (str(exclude_frame_id or ""), int(before_capture_finished_ns or 0), max(1, cap - len(rows) + 1)),
            ).fetchall()
            rows.extend(fetched)
            if len(rows) > cap:
                truncated = True
                rows = rows[:cap]
                break
            if len(rows) >= cap:
                truncated = True
                break
        return rows, truncated

    def _online_candidate_rows(self, connection, current, candidate_cap, exclude_frame_id, before_capture_finished_ns, lsh_radius=1):
        candidate_cap = max(8, min(2048, int(candidate_cap)))
        radius = max(0, min(2, int(lsh_radius or 0)))
        parts = tuple((current >> shift) & 0xFFFF for shift in (48, 32, 16, 0))
        keys = []
        for index, part in enumerate(parts):
            for variant in self._bucket_variants(part, radius):
                keys.append((index << 16) | variant)
        lsh_rows, truncated = self._lsh_candidate_rows(connection, keys, max(8, min(candidate_cap, candidate_cap * 3 // 4)), exclude_frame_id, before_capture_finished_ns)
        base_sql = "SELECT id, dhash64, phash, gray32x18, edge_density, color_histogram FROM frames WHERE (dhash64 IS NOT NULL OR phash IS NOT NULL) AND id<>? AND capture_finished_monotonic_ns<? "
        window = max(8, candidate_cap // 4)
        recent_rows = connection.execute(base_sql + "ORDER BY capture_finished_monotonic_ns DESC, id DESC LIMIT ?", (str(exclude_frame_id or ""), int(before_capture_finished_ns or 0), window)).fetchall()
        value_rows = connection.execute(base_sql + "ORDER BY (coverage + action_outcome_information + model_dependency_count * 0.25 + model_refs * 0.25) DESC, validation_last_used DESC, capture_finished_monotonic_ns DESC LIMIT ?", (str(exclude_frame_id or ""), int(before_capture_finished_ns or 0), window)).fetchall()
        rows = []
        seen = set()
        for row in lsh_rows + recent_rows + value_rows:
            if row[0] not in seen:
                rows.append(row)
                seen.add(row[0])
            if len(rows) >= candidate_cap:
                break
        return rows, bool(truncated), {"lsh": len(lsh_rows), "recent": len(recent_rows), "high_value": len(value_rows), "lsh_radius": radius}

    def _full_exact_hashes(self, connection, current_dhash, current_features, limit, chunk_size, exclude_frame_id, before_capture_finished_ns, deadline=None, cancelled=None, yield_if_pressure=None):
        cursor = connection.execute(
            "SELECT id, dhash64, phash, gray32x18, edge_density, color_histogram FROM frames "
            "WHERE (frames.dhash64 IS NOT NULL OR frames.phash IS NOT NULL) AND frames.id <> ? AND frames.capture_finished_monotonic_ns < ? "
            "ORDER BY capture_finished_monotonic_ns ASC, id ASC",
            (str(exclude_frame_id or ""), int(before_capture_finished_ns or 0)),
        )
        heap = []
        scanned = 0
        status = "complete"
        wanted = max(1, int(limit))
        while True:
            if cancelled is not None and cancelled():
                status = "cancelled"
                break
            if deadline is not None and time.monotonic() >= deadline:
                status = "deadline"
                break
            if yield_if_pressure is not None and not yield_if_pressure():
                status = "pressure"
                break
            rows = cursor.fetchmany(max(128, min(2048, int(chunk_size or 512))))
            if not rows:
                break
            for row in rows:
                entry = self._frame_composite_entry(current_dhash, current_features, row)
                if entry is None:
                    continue
                scanned += 1
                try:
                    tie = -int(entry["frame_id"], 16)
                except ValueError:
                    tie = -sum(ord(part) for part in entry["frame_id"])
                priority = (float(entry["similarity"]), -float(entry["distance"]), tie)
                item = (priority, entry)
                if len(heap) < wanted:
                    heapq.heappush(heap, item)
                elif priority > heap[0][0]:
                    heapq.heapreplace(heap, item)
            time.sleep(0)
        entries = [item[1] for item in heap]
        entries.sort(key=lambda item: (-item["similarity"], item["distance"], item["frame_id"]))
        return entries, scanned, status

    def nearest_hashes(self, dhash, limit=8, strict=True, candidate_limit=None, deadline=None, cancelled=None, yield_if_pressure=None, force_exact=False, current_features=None, exclude_frame_id=None, before_capture_finished_ns=None, lsh_radius=1):
        try:
            current = int(dhash, 16)
        except Exception:
            return {"hashes": [], "frame_ids": [], "similarities": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "invalid_hash", "exact_or_approx": "unknown", "recall_guard": False, "total_history": 0, "score_valid": False, "provisional": False}
        if before_capture_finished_ns is None:
            return {"hashes": [], "frame_ids": [], "similarities": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "missing_history_boundary", "exact_or_approx": "unknown", "recall_guard": False, "total_history": 0, "score_valid": False, "provisional": False}
        limit = max(1, int(limit))
        candidate_cap = max(limit, min(2048, int(candidate_limit or 256)))
        connection = self._read_connection()
        if connection is None:
            return {"hashes": [], "frame_ids": [], "similarities": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "store_closed", "exact_or_approx": "unknown", "recall_guard": False, "total_history": 0, "score_valid": False, "provisional": False}
        def response(entries, candidate_count, mode, exact, guard, provisional, sources):
            return {
                "hashes": [item["hash"] for item in entries],
                "frame_ids": [item["frame_id"] for item in entries],
                "similarities": [float(item["similarity"]) for item in entries],
                "candidate_count": int(candidate_count),
                "top_k_distance": float(max((item["distance"] for item in entries), default=64.0)),
                "retrieval_fallback": False,
                "retrieval_mode": mode,
                "exact_or_approx": exact,
                "recall_guard": bool(guard),
                "total_history": total,
                "score_valid": bool(entries) and bool(guard),
                "provisional": bool(provisional),
                "candidate_sources": sources,
            }
        try:
            total = int(connection.execute("SELECT COUNT(*) FROM frames WHERE (dhash64 IS NOT NULL OR phash IS NOT NULL) AND frames.id <> ? AND frames.capture_finished_monotonic_ns < ?", (str(exclude_frame_id or ""), int(before_capture_finished_ns))).fetchone()[0] or 0)
            if total <= 0:
                return {"hashes": [], "frame_ids": [], "similarities": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "warmup_no_history", "exact_or_approx": "exact", "recall_guard": True, "total_history": 0, "score_valid": False, "provisional": False}
            if not force_exact:
                if deadline is not None and time.monotonic() >= deadline:
                    return {"hashes": [], "frame_ids": [], "similarities": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "online_deadline_before_retrieval", "exact_or_approx": "approximate", "recall_guard": False, "total_history": total, "score_valid": False, "provisional": True}
                rows, truncated, sources = self._online_candidate_rows(connection, current, candidate_cap, exclude_frame_id, before_capture_finished_ns, lsh_radius)
                entries = self._frame_similarity_entries(dhash, current_features, rows, limit)
                if not entries:
                    return {"hashes": [], "frame_ids": [], "similarities": [], "candidate_count": len(rows), "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "online_candidate_empty", "exact_or_approx": "approximate", "recall_guard": False, "total_history": total, "score_valid": False, "provisional": True, "candidate_sources": sources}
                return response(entries, len(rows), "two_stage_lsh_recall_exact_distance_r{}__recent_high_value".format(int(sources.get("lsh_radius", 0))) + ("_capped" if truncated else ""), "approximate", False, True, sources)
            entries, scanned, status = self._full_exact_hashes(connection, dhash, current_features, limit, candidate_limit or 512, exclude_frame_id, before_capture_finished_ns, deadline, cancelled, yield_if_pressure)
            complete = status == "complete" and len(entries) >= min(limit, total)
            if not complete:
                return {"hashes": [item["hash"] for item in entries], "frame_ids": [item["frame_id"] for item in entries], "similarities": [float(item["similarity"]) for item in entries], "candidate_count": scanned, "top_k_distance": float(max((item["distance"] for item in entries), default=64.0)), "retrieval_fallback": True, "retrieval_mode": "sleep_exact_deferred_{}".format(status), "exact_or_approx": "unknown", "recall_guard": False, "total_history": total, "score_valid": False, "provisional": False, "candidate_sources": {"exact": scanned}}
            return response(entries, scanned, "two_stage_sleep_full_exact_composite", "exact", True, False, {"exact": scanned})
        except sqlite3.Error:
            return {"hashes": [], "frame_ids": [], "similarities": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "read_error", "exact_or_approx": "unknown", "recall_guard": False, "total_history": 0, "score_valid": False, "provisional": False}
        finally:
            connection.close()

    def _encode_trajectory(self, points):
        def write_varint(output, value):
            value = int(value)
            while value >= 0x80:
                output.append((value & 0x7F) | 0x80)
                value >>= 7
            output.append(value)
        output = bytearray()
        for dt, dx, dy in points or []:
            write_varint(output, max(0, int(dt)))
            write_varint(output, (int(dx) << 1) ^ (int(dx) >> 63))
            write_varint(output, (int(dy) << 1) ^ (int(dy) >> 63))
        return bytes(output)

    def _decode_trajectory(self, payload):
        data = memoryview(payload or b"")
        index = 0

        def read_varint():
            nonlocal index
            value = 0
            shift = 0
            while index < len(data):
                part = int(data[index])
                index += 1
                value |= (part & 0x7F) << shift
                if not (part & 0x80):
                    return value
                shift += 7
                if shift > 70:
                    raise ValueError("轨迹 varint 过长")
            raise ValueError("轨迹 varint 截断")

        points = []
        while index < len(data):
            dt = read_varint()
            zx = read_varint()
            zy = read_varint()
            dx = (zx >> 1) ^ -(zx & 1)
            dy = (zy >> 1) ^ -(zy & 1)
            points.append((int(dt), int(dx), int(dy)))
        return points

    def record_mouse_loss(self, session_id, started, ended, count, rule):
        delegated, result = self._dispatch_write_to_writer("record_mouse_loss", session_id, started, ended, count, rule)
        if delegated:
            return result
        with self.lock:
            if self.conn is None or not session_id or count <= 0:
                return
            self.conn.execute("INSERT INTO mouse_loss_events(id, session_id, created, started, ended, lost_count, rule) VALUES (?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, session_id, time.time(), started, ended, int(count), rule))
            self.conn.commit()

    def record_mouse_compression(self, segment):
        delegated, result = self._dispatch_write_to_writer("record_mouse_compression", segment)
        if delegated:
            return result
        if not segment or not segment.get("session_id"):
            raise RuntimeError("无效的无损轨迹段")
        payload = self._encode_trajectory(segment.get("points", []))
        if not payload and int(segment.get("count", 0)) > 0:
            raise RuntimeError("无损轨迹编码为空")
        rect = segment.get("client_rect") or (None, None, None, None)
        with self.lock:
            if self.conn is None:
                raise RuntimeError("存储未打开")
            point_hash = hashlib.sha256(payload).hexdigest() if payload else hashlib.sha256(json.dumps(segment.get("points", []), separators=(",", ":")).encode("utf-8")).hexdigest()
            self.conn.execute("INSERT INTO mouse_compression_segments(id, session_id, source, started, ended, started_monotonic_ns, ended_monotonic_ns, start_x, start_y, end_x, end_y, original_count, max_speed, path_length, trajectory_blob, trajectory_codec, rule, client_left, client_top, client_right, client_bottom, average_speed, direction_change_count, click_pre_dwell_ms, crossed_client_boundary, raw_point_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, segment["session_id"], segment.get("source", "user"), float(segment["started"]), float(segment["ended"]), int(segment["started_ns"]), int(segment["ended_ns"]), int(segment["start_x"]), int(segment["start_y"]), int(segment["end_x"]), int(segment["end_y"]), int(segment["count"]), float(segment.get("max_speed", 0.0)), float(segment.get("path_length", 0.0)), sqlite3.Binary(payload), "varint-zigzag-dtxy-v1", str(segment.get("rule", "无损移动轨迹压缩")), *[None if value is None else int(value) for value in rect], float(segment.get("average_speed", 0.0)), int(segment.get("direction_change_count", 0)), float(segment.get("click_pre_dwell_ms", 0.0)), 1 if segment.get("crossed_client_boundary") else 0, point_hash))
            self.conn.commit()
        if segment.get("accepted_journal_id"):
            self.complete_accepted_journal(segment.get("accepted_journal_id"))
        return True

    def record_pipeline_loss(self, session_id, started, ended, count, stage, reason):
        delegated, result = self._dispatch_write_to_writer("record_pipeline_loss", session_id, started, ended, count, stage, reason)
        if delegated:
            return result
        if count <= 0:
            return
        with self.lock:
            if self.conn is None:
                return
            self.conn.execute("INSERT INTO pipeline_loss_events(id, session_id, created, started, ended, lost_count, stage, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, session_id, time.time(), float(started), float(ended), int(count), str(stage), str(reason)))
            self.conn.commit()

    def _database_bytes_precise_locked(self):
        if self.conn is None or self.pool is None:
            return 0
        try:
            page_count = int(self.conn.execute("PRAGMA page_count").fetchone()[0] or 0)
            page_size = int(self.conn.execute("PRAGMA page_size").fetchone()[0] or 0)
            total = page_count * page_size
        except sqlite3.Error:
            total = 0
        for name in ("records.sqlite3-wal", "records.sqlite3-shm", "records.sqlite3-journal"):
            try:
                total += int((self.pool / name).stat().st_size)
            except OSError:
                pass
        self._database_bytes_cached = max(0, int(total))
        self._database_bytes_checked = time.monotonic()
        return self._database_bytes_cached

    def _database_bytes_estimate_locked(self):
        if self._database_bytes_cached <= 0 or time.monotonic() - self._database_bytes_checked >= 1.0:
            return self._database_bytes_precise_locked()
        return max(0, int(self._database_bytes_cached))

    def _capacity_calibration_due_locked(self, limit, projected):
        if self._database_bytes_cached <= 0:
            return True
        if time.monotonic() - self._database_bytes_checked >= 1.0:
            return True
        if self._capacity_write_count % 64 == 0:
            return True
        return bool(limit and projected >= int(limit * 0.80))

    def wal_metrics(self):
        with self.lock:
            pool = self.pool
        wal_bytes = 0
        if pool is not None:
            try:
                wal_bytes = int((pool / "records.sqlite3-wal").stat().st_size)
            except OSError:
                wal_bytes = 0
        with self.lock:
            result = dict(self.last_wal_metrics)
            result["wal_bytes"] = wal_bytes
            self.last_wal_metrics = dict(result)
        return result

    def _other_pool_bytes_locked(self):
        if self.pool is None:
            return 0, 0
        transient = 0
        other = 0
        for item in self.pool.iterdir():
            try:
                if item.name.lower().startswith("records.sqlite3"):
                    continue
                if item.name == "screens":
                    continue
                if item.name == "trash" or item.name.endswith(".tmp") or item.name.startswith(".write_latency_probe"):
                    transient += sum(int(child.stat().st_size) for child in self._iter_storage_files(item, None, "pool_transient_walk")) if item.is_dir() else int(item.stat().st_size)
                elif item.is_file():
                    other += int(item.stat().st_size)
                elif item.is_dir():
                    other += sum(int(child.stat().st_size) for child in self._iter_storage_files(item, None, "pool_other_walk"))
            except OSError:
                pass
        return transient, other

    def _capacity_snapshot_locked(self, precise=True):
        asset = int(self.conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM frames").fetchone()[0] or 0) if self.conn is not None else 0
        database = self._database_bytes_precise_locked() if precise else self._database_bytes_estimate_locked()
        transient, other = self._other_pool_bytes_locked()
        reserved = self._ledger_values_locked().get("reserved_asset_bytes", 0) if self.conn is not None else 0
        return {"asset_bytes": max(0, asset), "reserved_asset_bytes": max(0, int(reserved)), "database_bytes": database, "transient_bytes": transient, "other_bytes": other, "total": max(0, asset + database + transient + other)}

    def _png_reservation_locked(self, declared_size):
        rows = self.conn.execute("SELECT size_bytes FROM frames WHERE size_bytes>0 ORDER BY capture_finished_monotonic_ns DESC LIMIT 256").fetchall()
        sizes = sorted(max(0, int(row[0] or 0)) for row in rows)
        p95 = sizes[min(len(sizes) - 1, max(0, int(math.ceil(len(sizes) * 0.95)) - 1))] if sizes else int(declared_size)
        predicted = max(int(declared_size), int(p95))
        return int(math.ceil(predicted * 1.35)) + int(self.transaction_reserve_bytes)

    def _write_ledger_locked(self, snapshot, reserved):
        values = (
            ("asset_bytes", int(snapshot["asset_bytes"])),
            ("total_asset_bytes", int(snapshot["asset_bytes"])),
            ("reserved_asset_bytes", max(0, int(reserved))),
            ("database_bytes", int(snapshot["database_bytes"])),
            ("transient_bytes", int(snapshot["transient_bytes"])),
            ("other_bytes", int(snapshot["other_bytes"])),
            ("last_reconciled_at", int(time.time())),
        )
        self.conn.executemany("INSERT OR REPLACE INTO pool_meta(key, value) VALUES (?, ?)", values)

    def _capture_audit_payload(self, image):
        image = image or {}
        payload = {
            "validation_before": image.get("validation_before") or {},
            "validation_after": image.get("validation_after") or {},
            "obstruction": image.get("obstruction") or image.get("occlusion") or {},
            "capture_backend": str(image.get("capture_backend", "")),
            "fallback_reason": str(image.get("fallback_reason") or image.get("capture_failure_reason") or ""),
            "capture_hash_delta": float(image.get("capture_hash_delta", 64.0) or 64.0),
            "black_ratio": float(image.get("black_ratio", 1.0) or 0.0),
            "variance": float(image.get("variance", 0.0) or 0.0),
            "monitor_coverage": float(image.get("monitor_coverage", 0.0) or 0.0),
            "capture_complete": 1 if image.get("capture_complete") else 0,
            "capture_fallback": 1 if image.get("capture_fallback") else 0,
            "capture_generation": int(image.get("capture_generation", 0) or 0),
        }
        return payload

    def record_capture_contract_tick(self, session_id, image, mouse_position=None, mouse_inside=False, persisted_png=False, frame_id=None, reason=""):
        delegated, result = self._dispatch_write_to_writer("record_capture_contract_tick", session_id, image, mouse_position, mouse_inside, persisted_png, frame_id, reason)
        if delegated:
            return result
        if not session_id or not isinstance(image, dict):
            return None
        identifier = str(image.get("contract_tick_id") or uuid.uuid4().hex)
        image["contract_tick_id"] = identifier
        x, y = mouse_position if isinstance(mouse_position, tuple) and len(mouse_position) == 2 else (None, None)
        with self.lock:
            if self.conn is None:
                return identifier
            self.conn.execute("""
                INSERT INTO capture_contract_ticks(id, session_id, created, created_monotonic_ns, width, height, phash, dhash64, gray32x18, edge_density, color_histogram, online_score, score_status, mouse_x, mouse_y, mouse_inside, persisted_png, frame_id, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET online_score=excluded.online_score, score_status=excluded.score_status, mouse_x=excluded.mouse_x, mouse_y=excluded.mouse_y, mouse_inside=excluded.mouse_inside, persisted_png=MAX(capture_contract_ticks.persisted_png, excluded.persisted_png), frame_id=COALESCE(excluded.frame_id, capture_contract_ticks.frame_id), reason=excluded.reason
            """, (identifier, str(session_id), float(image.get("capture_finished", time.time()) or time.time()), int(image.get("capture_finished_monotonic_ns", time.monotonic_ns()) or 0), int(image.get("width", 0) or 0), int(image.get("height", 0) or 0), str(image.get("phash") or ""), str(image.get("dhash64") or ""), sqlite3.Binary(feature_bytes(image.get("gray32x18"), 32 * 18)), float(image.get("edge_density", 0.0) or 0.0), sqlite3.Binary(histogram_blob(image.get("color_histogram"))), image.get("online_score"), str(image.get("score_status") or ""), x, y, 1 if mouse_inside else 0, 1 if persisted_png else 0, frame_id, str(reason or "")))
            self.conn.commit()
        return identifier

    def mark_capture_contract_persisted(self, tick_id, frame_id):
        delegated, result = self._dispatch_write_to_writer("mark_capture_contract_persisted", tick_id, frame_id)
        if delegated:
            return result
        if not tick_id or not frame_id:
            return
        with self.lock:
            if self.conn is None:
                return
            self.conn.execute("UPDATE capture_contract_ticks SET persisted_png=1, frame_id=? WHERE id=?", (str(frame_id), str(tick_id)))
            self.conn.commit()

    def save_frame(self, session_id, image, phash, online_score=None, exact_score=None, hunger=None, reward=None, experience_limit=None):
        delegated, result = self._dispatch_write_to_writer("save_frame", session_id, image, phash, online_score, exact_score, hunger, reward, experience_limit)
        if delegated:
            return result
        if not session_id or not isinstance(image, dict) or not image.get("png"):
            raise RuntimeError("无效截图写入请求")
        accepted_journal_id = str(image.get("accepted_journal_id") or "")
        if accepted_journal_id:
            with self.lock:
                if self.conn is not None:
                    existing = self.conn.execute("SELECT id FROM frames WHERE accepted_journal_id=?", (accepted_journal_id,)).fetchone()
                    if existing:
                        return existing[0]
        identifier = uuid.uuid4().hex
        journal_id = uuid.uuid4().hex
        moment = float(image.get("capture_finished", time.time()))
        mono = int(image.get("capture_finished_monotonic_ns", time.monotonic_ns()))
        capture_started_mono = int(image.get("capture_started_monotonic_ns", mono))
        capture_started_wall = float(image.get("capture_started", moment))
        png = image["png"]
        declared_size = int(len(png))
        limit = int(experience_limit) if experience_limit is not None else 0
        dhash = image.get("dhash64") or phash
        buckets = self._hash_buckets(dhash)
        state_cluster_id = self.assign_state_cluster(dhash, image.get("width"), image.get("height"), image.get("gray32x18"), image.get("edge_density", 0.0))
        relative = Path("screens") / session_id / (identifier + ".png")
        reservation = int(declared_size + self.transaction_reserve_bytes)
        final_path = None
        temporary = None
        reserved = False
        try:
            with self.lock:
                if self.conn is None or self.pool is None or self.screens is None:
                    raise RuntimeError("存储未打开")
                self._inject_fault("sqlite_begin")
                self.conn.execute("BEGIN IMMEDIATE")
                snapshot = self._capacity_snapshot_locked(False)
                reservation = self._png_reservation_locked(declared_size)
                projected = snapshot["total"] + snapshot["reserved_asset_bytes"] + reservation
                if self._capacity_calibration_due_locked(limit, projected):
                    snapshot = self._capacity_snapshot_locked(True)
                    projected = snapshot["total"] + snapshot["reserved_asset_bytes"] + reservation
                usage = 0.0 if limit <= 0 else projected / float(limit)
                tier = 100 if usage >= 1.0 else 95 if usage >= 0.95 else 90 if usage >= 0.90 else 80 if usage >= 0.80 else 0
                if limit > 0 and projected > limit:
                    target, _target_meta = self._dynamic_prune_target(limit, snapshot["total"])
                    values = (("pool_capacity_blocked", 1), ("pool_capacity_target", target), ("pool_capacity_remaining", int(snapshot["total"])), ("pool_capacity_updated", int(time.time())), ("pool_capacity_tier", 100), ("pool_capacity_transaction_reserve", int(self.transaction_reserve_bytes)))
                    self.conn.executemany("INSERT OR REPLACE INTO pool_meta(key, value) VALUES (?, ?)", values)
                    self.conn.commit()
                    raise PoolCapacityBlocked("经验池容量硬拒绝：实际 {} 字节，PNG 与事务预留 {} 字节，上限 {} 字节".format(snapshot["total"], reservation, limit))
                self._write_ledger_locked(snapshot, snapshot["reserved_asset_bytes"] + reservation)
                self.conn.executemany("INSERT OR REPLACE INTO pool_meta(key, value) VALUES (?, ?)", (("pool_capacity_tier", tier), ("pool_capacity_transaction_reserve", int(self.transaction_reserve_bytes))))
                self.conn.execute("INSERT INTO ingestion_journal(id, object_type, object_id, path, stage, created, updated, error) VALUES (?, ?, ?, ?, ?, ?, ?, '')", (journal_id, "frame", identifier, str(relative), "reserved", moment, moment))
                self.conn.commit()
                reserved = True
                pool = self.pool
                screens = self.screens
            folder = self._assert_storage_path(screens / session_id, "frame_folder")
            folder.mkdir(parents=True, exist_ok=True)
            final_path = self._assert_storage_path(pool / relative, "frame_png")
            temporary = self._assert_storage_path(final_path.with_suffix(".tmp"), "frame_png_tmp")
            self._inject_fault("png_write")
            _atomic_write_bytes(final_path, png, self.root)
            actual_size = int(self._assert_storage_path(final_path, "frame_png_post_replace").stat().st_size)
            with self.lock:
                self._inject_fault("sqlite_write")
                self.conn.execute("BEGIN IMMEDIATE")
                self.conn.execute("UPDATE ingestion_journal SET stage='file_ready', updated=?, error='' WHERE id=?", (time.time(), journal_id))
                support = self.conn.execute("SELECT COUNT(*) + 1 FROM frames WHERE state_cluster_id=?", (state_cluster_id,)).fetchone()[0]
                score_valid = exact_score is not None and bool(image.get("score_valid")) and not bool(image.get("score_provisional"))
                online_value = float(online_score) if online_score is not None else None
                score_value = float(exact_score) if score_valid else None
                hunger_value = 0.0
                reward_value = score_value if score_valid else None
                requested_status = str(image.get("score_status") or "")
                score_status = "exact" if score_valid else "provisional_cache" if requested_status == "provisional_cache" else "provisional" if online_value is not None or dhash else "invalid"
                boundary_row = self.conn.execute("SELECT id FROM frames WHERE capture_finished_monotonic_ns<? ORDER BY capture_finished_monotonic_ns DESC, id DESC LIMIT 1", (mono,)).fetchone()
                history_boundary_frame_id = str(boundary_row[0]) if boundary_row else None
                score_generation = str(image.get("score_generation") or ("online" if score_status != "invalid" else "online"))
                audit_payload = self._capture_audit_payload(image)
                audit_json = json.dumps(audit_payload, ensure_ascii=False, separators=(",", ":"))
                frame_values = (identifier, session_id, moment, mono, capture_started_mono, mono, capture_started_wall, moment, str(relative), phash, dhash, score_value, online_value, hunger_value, reward_value, score_value, reward_value, score_status, score_generation, history_boundary_frame_id, image["width"], image["height"], actual_size, 0.0, 0.0, 0.0, 0, moment, *buckets, state_cluster_id, support, 0.0, 0, moment, 1, image.get("capture_backend", "gdi"), image.get("capture_elapsed_ms", 0.0), image.get("capture_complete", 1), image.get("brightness", 0.0), image.get("variance", 0.0), sqlite3.Binary(feature_bytes(image.get("gray32x18"), 32 * 18)), image.get("edge_density", 0.0), sqlite3.Binary(histogram_blob(image.get("color_histogram"))), str(image.get("capture_failure_reason", "")), float(image.get("capture_hash_delta", 64.0)), 1 if image.get("capture_fallback") else 0, audit_json, int(image.get("score_candidate_count", 0)), float(image.get("score_top_k_distance", 64.0)), int(image.get("score_retrieval_fallback", 0)))
                self.conn.execute("INSERT INTO frames(id, session_id, created, created_monotonic_ns, capture_started_monotonic_ns, capture_finished_monotonic_ns, capture_started, capture_finished, screenshot_path, phash, dhash64, score, online_score, hunger, reward, raw_score, raw_reward, score_status, score_generation, history_boundary_frame_id, width, height, size_bytes, novelty, action_result, coverage, model_refs, last_used, bucket0, bucket1, bucket2, bucket3, state_cluster_id, state_support_count, action_outcome_information, model_dependency_count, validation_last_used, asset_ref_count, capture_backend, capture_elapsed_ms, capture_complete, brightness, variance, gray32x18, edge_density, color_histogram, capture_failure_reason, capture_hash_delta, capture_fallback, capture_audit_json, score_candidate_count, score_top_k_distance, score_retrieval_fallback) VALUES ({})".format(",".join("?" for _ in frame_values)), frame_values)
                self.conn.execute("INSERT OR REPLACE INTO frame_capture_audit(frame_id, created, session_id, audit_json, validation_before, validation_after, capture_backend, fallback_reason, capture_hash_delta, black_ratio, variance, monitor_coverage) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (identifier, moment, session_id, audit_json, json.dumps(audit_payload.get("validation_before") or {}, ensure_ascii=False, separators=(",", ":")), json.dumps(audit_payload.get("validation_after") or {}, ensure_ascii=False, separators=(",", ":")), audit_payload.get("capture_backend", ""), audit_payload.get("fallback_reason", ""), audit_payload.get("capture_hash_delta", 64.0), audit_payload.get("black_ratio", 1.0), audit_payload.get("variance", 0.0), audit_payload.get("monitor_coverage", 0.0)))
                self.conn.execute("UPDATE frames SET score_retrieval_mode=?, score_exact_or_approx=?, score_recall_guard=?, score_valid=?, accepted_journal_id=?, reward_source='screen_score_only', score_valid_for_training=? WHERE id=?", (str(image.get("score_retrieval_mode", "online_pending")), "exact" if score_valid else ("provisional_cache" if score_status == "provisional_cache" else "approximate" if score_status == "provisional" else "invalid"), 1 if score_valid else 0, 1 if score_valid else 0, accepted_journal_id or None, 1 if score_valid else 0, identifier))
                if not score_valid and dhash:
                    self.conn.execute("INSERT OR REPLACE INTO deferred_exact_scores(frame_id, dhash64, created, updated, attempts, state, last_error) VALUES (?, ?, ?, ?, 0, 'pending', '')", (identifier, dhash, moment, moment))
                self.conn.execute("INSERT OR IGNORE INTO state_clusters(cluster_id, count, updated_at) VALUES (?, 0, ?)", (state_cluster_id, moment))
                self.conn.execute("UPDATE state_clusters SET count=count+1, updated_at=? WHERE cluster_id=?", (moment, state_cluster_id))
                rarity = 1.0 / max(1.0, float(support))
                quality = 0.20 if image.get("capture_complete", 0) else 0.0
                contrast = min(0.15, max(0.0, float(image.get("variance", 0.0) or 0.0) / 2550.0))
                retain_value = rarity + quality + contrast
                self.conn.execute("UPDATE frames SET retain_value=?, retain_version=2 WHERE id=?", (retain_value, identifier))
                self.conn.executemany("INSERT OR IGNORE INTO frame_lsh(key, frame_id) VALUES (?, ?)", [(key, identifier) for key in self._hash_buckets(dhash)])
                self.conn.execute("UPDATE sessions SET frame_count=frame_count+1 WHERE id=?", (session_id,))
                snapshot_after = self._capacity_snapshot_locked(False)
                remaining_reserved = max(0, snapshot_after["reserved_asset_bytes"] - reservation)
                final_projected = snapshot_after["total"]
                if self._capacity_calibration_due_locked(limit, final_projected):
                    snapshot_after = self._capacity_snapshot_locked(True)
                    final_projected = snapshot_after["total"]
                    remaining_reserved = max(0, snapshot_after["reserved_asset_bytes"] - reservation)
                if limit > 0 and final_projected > limit:
                    raise PoolCapacityBlocked("提交前容量校验失败：实际目录与 SQLite/WAL 为 {} 字节，上限 {} 字节".format(final_projected, limit))
                self._write_ledger_locked(snapshot_after, remaining_reserved)
                usage_after = 0.0 if limit <= 0 else final_projected / float(limit)
                tier_after = 100 if usage_after >= 1.0 else 95 if usage_after >= 0.95 else 90 if usage_after >= 0.90 else 80 if usage_after >= 0.80 else 0
                self.conn.executemany("INSERT OR REPLACE INTO pool_meta(key, value) VALUES (?, ?)", (
                    ("pool_capacity_blocked", 0), ("pool_capacity_target", int(limit or 0)),
                    ("pool_capacity_remaining", int(final_projected)), ("pool_capacity_updated", int(time.time())),
                    ("pool_capacity_tier", tier_after), ("pool_capacity_transaction_reserve", int(self.transaction_reserve_bytes))))
                self.conn.execute("UPDATE ingestion_journal SET stage='complete', updated=?, error='' WHERE id=?", (time.time(), journal_id))
                self._capacity_write_count += 1
                self.conn.execute("INSERT OR REPLACE INTO pool_meta(key, value) VALUES ('database_bytes', ?)", (int(snapshot_after["database_bytes"]),))
                self.conn.commit()
                reserved = False
            return identifier
        except Exception as error:
            try:
                self.add_critical_exception("DataStore", "save_frame", error, session_id=session_id, payload={"frame_id": identifier, "stage": "frame_write", "limit": limit, "reservation": reservation})
            except Exception:
                pass
            if temporary is not None:
                try:
                    self._safe_unlink_storage(temporary, "save_frame_tmp_cleanup")
                except OSError:
                    pass
            if final_path is not None:
                try:
                    self._safe_unlink_storage(final_path, "save_frame_final_cleanup")
                except OSError:
                    pass
            if reserved:
                with self.lock:
                    try:
                        self.conn.rollback()
                    except Exception:
                        pass
                    try:
                        self.conn.execute("BEGIN IMMEDIATE")
                        ledger = self._ledger_values_locked()
                        self.conn.execute("INSERT OR REPLACE INTO pool_meta(key, value) VALUES ('reserved_asset_bytes', ?)", (max(0, ledger["reserved_asset_bytes"] - reservation),))
                        self.conn.execute("UPDATE ingestion_journal SET stage='failed', updated=?, error=? WHERE id=?", (time.time(), str(error), journal_id))
                        self.conn.commit()
                    except Exception:
                        try:
                            self.conn.rollback()
                        except Exception:
                            pass
            raise

    def save_mouse_batch(self, records):
        delegated, result = self._dispatch_write_to_writer("save_mouse_batch", records)
        if delegated:
            return result
        if not records:
            return
        values = []
        counts = {}
        for record in records:
            values.append((uuid.uuid4().hex, record["session_id"], record["created"], record.get("created_monotonic_ns", 0), record["source"], record["event_type"], record["button"], record["wheel"], record["x"], record["y"], record["relative_x"], record["relative_y"], record["dx"], record["dy"], record["direction"], record["speed"], record.get("behavior_probability"), record.get("before_frame_id"), record.get("after_frame_id")))
            counts[record["session_id"]] = counts.get(record["session_id"], 0) + 1
        with self.lock:
            if self.conn is None:
                return
            self.conn.executemany("INSERT INTO mouse_events(id, session_id, created, created_monotonic_ns, source, event_type, button, wheel, x, y, relative_x, relative_y, dx, dy, direction, speed, behavior_probability, before_frame_id, after_frame_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
            self.conn.executemany("UPDATE sessions SET mouse_count=mouse_count+? WHERE id=?", [(value, key) for key, value in counts.items()])
            self.conn.commit()
        for record in records:
            if record.get("accepted_journal_id"):
                self.complete_accepted_journal(record.get("accepted_journal_id"))

    def bind_mouse_events_after_frame(self, session_id, frame_id, frame_finished_ns):
        delegated, result = self._dispatch_write_to_writer("bind_mouse_events_after_frame", session_id, frame_id, frame_finished_ns)
        if delegated:
            return result
        if not session_id or not frame_id:
            return 0
        before = int(self.conn.total_changes) if self.conn is not None else 0
        with self.lock:
            if self.conn is None:
                return 0
            self.conn.execute("""
                UPDATE mouse_events
                   SET after_frame_id=?
                 WHERE session_id=?
                   AND after_frame_id IS NULL
                   AND created_monotonic_ns<=?
                   AND (before_frame_id IS NULL OR before_frame_id!=?)
            """, (frame_id, session_id, int(frame_finished_ns or 0), frame_id))
            changed = max(0, int(self.conn.total_changes) - before)
            self.conn.commit()
            return changed

    def training_readiness(self, minimum_actions=30, minimum_sessions=1):
        with self.lock:
            eligible = """
                SELECT s.id FROM sessions s
                WHERE COALESCE(s.trainable, 1)=1
                  AND NOT EXISTS (SELECT 1 FROM pipeline_loss_events p WHERE p.session_id=s.id)
                  AND NOT EXISTS (SELECT 1 FROM mouse_loss_events m WHERE m.session_id=s.id AND (m.lost_count>0 OR instr(m.rule, '关键事件丢失')>0))
                  AND NOT EXISTS (SELECT 1 FROM deferred_exact_scores d JOIN frames f ON f.id=d.frame_id WHERE f.session_id=s.id AND d.state!='complete')
            """
            sessions = int(self.conn.execute("SELECT COUNT(*) FROM (" + eligible + ")").fetchone()[0] or 0)
            frames = int(self.conn.execute("SELECT COUNT(*) FROM frames WHERE session_id IN (" + eligible + ") AND capture_complete=1 AND score_valid=1 AND score_status='exact'").fetchone()[0] or 0)
            event_actions = int(self.conn.execute("SELECT COUNT(*) FROM mouse_events WHERE session_id IN (" + eligible + ") AND event_type IN ('button_up','wheel','move')").fetchone()[0] or 0)
            segment_actions = int(self.conn.execute("SELECT COALESCE(SUM(original_count), 0) FROM mouse_compression_segments WHERE session_id IN (" + eligible + ")").fetchone()[0] or 0)
        actions = event_actions + segment_actions
        return {"ready": sessions >= minimum_sessions and frames >= 24 and actions >= minimum_actions, "sessions": sessions, "actions": actions, "frames": frames, "compressed_trajectory_points": segment_actions}

    def training_data_fingerprint(self):
        with self.lock:
            frame_count, frame_max, valid_count = self.conn.execute("SELECT COUNT(*), COALESCE(MAX(capture_finished_monotonic_ns), 0), COALESCE(SUM(score_valid), 0) FROM frames WHERE capture_complete=1").fetchone()
            mouse_count, mouse_max = self.conn.execute("SELECT COUNT(*), COALESCE(MAX(created_monotonic_ns), 0) FROM mouse_events").fetchone()
            segment_count, segment_points, segment_max = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(original_count), 0), COALESCE(MAX(ended_monotonic_ns), 0) FROM mouse_compression_segments").fetchone()
        return "{}:{}:{}:{}:{}:{}:{}:{}".format(int(frame_count or 0), int(valid_count or 0), int(frame_max or 0), int(mouse_count or 0), int(mouse_max or 0), int(segment_count or 0), int(segment_points or 0), int(segment_max or 0))

    def model_files(self):
        if self.models is None:
            return []
        return sorted(self.models.glob("model_*.json"))

    def model_summaries(self):
        result = []
        for path in self.model_files():
            try:
                content = json.loads(path.read_text(encoding="utf-8"))
                rank = float(content.get("validation_quality", content.get("quality", 0.0)))
                coverage = float(content.get("coverage_states", content.get("diversity", 0.0)))
                samples = int(content.get("validation_samples", content.get("frame_count", 0)))
                result.append((path, rank + coverage * 0.001 + min(samples, 1000000) * 0.000000001, float(content.get("validated_at", content.get("trained_at", 0.0)))))
            except Exception:
                result.append((path, -999999.0, 0.0))
        return result

    def best_model(self):
        with self.lock:
            if self.conn is None or self.models is None:
                return None
            row = self.conn.execute("SELECT file_name, champion FROM model_metadata WHERE champion=1 ORDER BY validation_quality DESC, updated DESC LIMIT 1").fetchone()
            if row is None:
                row = self.conn.execute("SELECT file_name, champion FROM model_metadata ORDER BY validation_quality DESC, quality DESC, updated DESC LIMIT 1").fetchone()
            path = self.models / str(row[0]) if row else None
            champion = bool(row and int(row[1] or 0))
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["champion"] = champion
            return payload
        except Exception:
            return None

    def recent_action_benefit(self, action_type, limit=8):
        action_type = str(action_type or "")
        limit = max(4, min(32, int(limit or 8)))
        if action_type == "左键":
            predicate = "me.event_type='button_up' AND me.button='left'"
        elif action_type == "右键":
            predicate = "me.event_type='button_up' AND me.button='right'"
        elif action_type == "滚轮":
            predicate = "me.event_type='wheel' AND me.button='vertical'"
        elif action_type == "水平滚轮":
            predicate = "me.event_type='wheel' AND me.button='horizontal'"
        else:
            return {"count": 0, "mean": 0.0, "stdev": 0.0, "unstable": False}
        with self.lock:
            rows = self.conn.execute("""SELECT ao.score_delta FROM action_outcomes ao
                JOIN mouse_events me ON me.id=ao.mouse_event_id
                WHERE ao.outcome_valid=1 AND me.source='ai' AND {}
                ORDER BY ao.action_time DESC LIMIT ?""".format(predicate), (limit,)).fetchall()
        values = [float(row[0] or 0.0) for row in rows]
        if not values:
            return {"count": 0, "mean": 0.0, "stdev": 0.0, "unstable": False}
        mean = sum(values) / len(values)
        variance = sum((item - mean) ** 2 for item in values) / max(1, len(values))
        stdev = math.sqrt(max(0.0, variance))
        unstable = len(values) >= max(4, limit // 2) and (mean < 0.0 or stdev > max(0.08, abs(mean) * 2.5))
        return {"count": len(values), "mean": mean, "stdev": stdev, "unstable": unstable}

    def collect_training_data(self):
        with self.lock:
            sessions = [row[0] for row in self.conn.execute("""
                SELECT s.id FROM sessions s
                WHERE COALESCE(s.trainable, 1)=1
                  AND NOT EXISTS (SELECT 1 FROM pipeline_loss_events p WHERE p.session_id=s.id)
                  AND NOT EXISTS (SELECT 1 FROM mouse_loss_events m WHERE m.session_id=s.id AND (m.lost_count>0 OR instr(m.rule, '关键事件丢失')>0))
                  AND NOT EXISTS (SELECT 1 FROM deferred_exact_scores d JOIN frames f ON f.id=d.frame_id WHERE f.session_id=s.id AND d.state NOT IN ('complete','completed'))
                  AND NOT EXISTS (SELECT 1 FROM mouse_events me WHERE me.session_id=s.id AND ((s.mode='learning' AND me.source!='user') OR (s.mode='training' AND me.source!='ai') OR me.source='external_injected'))
                  AND NOT EXISTS (SELECT 1 FROM mouse_compression_segments ms WHERE ms.session_id=s.id AND ((s.mode='learning' AND ms.source!='user') OR (s.mode='training' AND ms.source!='ai') OR ms.source='external_injected'))
                ORDER BY s.started DESC LIMIT 32
            """).fetchall()]
            frames = {}
            mouse = {}
            for session_id in sessions:
                frame_rows = self.conn.execute("""
                    SELECT id, session_id, created_monotonic_ns, created, dhash64, phash, score, COALESCE(raw_score, score), 0.0, width, height, gray32x18, edge_density, color_histogram, capture_started_monotonic_ns, capture_finished_monotonic_ns, score_valid
                    FROM frames
                    WHERE session_id=? AND capture_complete=1 AND score_valid=1 AND score_status='exact' AND score IS NOT NULL AND created_monotonic_ns>0
                    ORDER BY capture_finished_monotonic_ns ASC, id ASC
                    LIMIT 9000
                """, (session_id,)).fetchall()
                frame_rows = [tuple(list(row[:11]) + [feature_hex(row[11])] + list(row[12:13]) + [histogram_text(row[13])] + list(row[14:])) for row in frame_rows]
                mouse_rows = list(self.conn.execute("""
                    SELECT id, session_id, created_monotonic_ns, created, event_type, source, relative_x, relative_y, speed, dx, dy, direction, button, wheel, behavior_probability
                    FROM mouse_events
                    WHERE session_id=? AND created_monotonic_ns>0 AND source=CASE (SELECT mode FROM sessions WHERE id=?) WHEN 'learning' THEN 'user' WHEN 'training' THEN 'ai' ELSE source END AND source!='external_injected'
                    ORDER BY created_monotonic_ns ASC, id ASC
                    LIMIT 24000
                """, (session_id, session_id)).fetchall())
                segments = self.conn.execute("""
                    SELECT id, source, started, started_monotonic_ns, start_x, start_y, trajectory_blob, client_left, client_top, client_right, client_bottom
                    FROM mouse_compression_segments
                    WHERE session_id=? AND ended_monotonic_ns>0 AND source=CASE (SELECT mode FROM sessions WHERE id=?) WHEN 'learning' THEN 'user' WHEN 'training' THEN 'ai' ELSE source END AND source!='external_injected'
                    ORDER BY ended_monotonic_ns ASC, id ASC
                    LIMIT 4096
                """, (session_id, session_id)).fetchall()
                decoded = []
                for segment_id, source, started, start_ns, start_x, start_y, blob, left, top, right, bottom in segments:
                    try:
                        x = int(start_x)
                        y = int(start_y)
                        ns = int(start_ns)
                        previous_speed = 0.0
                        for index, (dt, dx, dy) in enumerate(self._decode_trajectory(blob)):
                            ns += int(dt)
                            x += int(dx)
                            y += int(dy)
                            seconds = max(1e-9, float(dt) / 1_000_000_000.0)
                            speed = math.hypot(dx, dy) / seconds
                            direction = math.atan2(dy, dx) if dx or dy else 0.0
                            rx = (x - int(left)) / max(1, int(right) - int(left)) if None not in (left, top, right, bottom) else None
                            ry = (y - int(top)) / max(1, int(bottom) - int(top)) if None not in (left, top, right, bottom) else None
                            decoded.append(("segment:{}:{}".format(segment_id, index), session_id, ns, float(started) + (ns - int(start_ns)) / 1_000_000_000.0, "move", source, rx, ry, speed, float(dx), float(dy), direction, "", 0, None))
                            previous_speed = speed
                    except Exception:
                        continue
                if decoded:
                    mouse_rows.extend(decoded)
                    mouse_rows.sort(key=lambda row: (int(row[2]), str(row[0])))
                    mouse_rows = mouse_rows[-24000:]
                if frame_rows:
                    frames[session_id] = frame_rows
                if mouse_rows:
                    mouse[session_id] = mouse_rows
            sampled = [row[0] for rows in frames.values() for row in rows[-256:]]
            if sampled:
                self.conn.executemany("UPDATE frames SET validation_last_used=? WHERE id=?", [(time.time(), item) for item in sampled])
            self.conn.commit()
        return frames, mouse

    def save_action_outcomes(self, outcomes):
        delegated, result = self._dispatch_write_to_writer("save_action_outcomes", outcomes)
        if delegated:
            return result
        if not outcomes:
            return
        rows = []
        for item in outcomes:
            score_delta = float(item.get("score_delta", 0.0) or 0.0)
            baseline_score_delta = float(item.get("baseline_score_delta", 0.0) or 0.0)
            rows.append((item.get("action_id") or item["mouse_event_id"], uuid.uuid4().hex, item["session_id"], item["mouse_event_id"], item["before_frame_id"], item["after_frame_id"], item["action_time"], item["post_action_delay_ms"], score_delta, score_delta, 0.0, baseline_score_delta, baseline_score_delta, float(item.get("action_advantage", 0.0)), float(item.get("stability", 0.0)), int(item.get("baseline_count", 0)), 1 if item.get("outcome_valid") else 0, item.get("split_role", "unknown")))
        with self.lock:
            self.conn.executemany("INSERT INTO action_outcomes(action_id, id, session_id, mouse_event_id, before_frame_id, after_frame_id, action_time, post_action_delay_ms, score_delta, reward_delta, hunger_delta_expected, baseline_score_delta, expected_no_action_reward_delta, action_advantage, stability, baseline_count, outcome_valid, split_role) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(action_id) DO UPDATE SET session_id=excluded.session_id, mouse_event_id=excluded.mouse_event_id, before_frame_id=excluded.before_frame_id, after_frame_id=excluded.after_frame_id, action_time=excluded.action_time, post_action_delay_ms=excluded.post_action_delay_ms, score_delta=excluded.score_delta, reward_delta=excluded.reward_delta, hunger_delta_expected=excluded.hunger_delta_expected, baseline_score_delta=excluded.baseline_score_delta, expected_no_action_reward_delta=excluded.expected_no_action_reward_delta, action_advantage=excluded.action_advantage, stability=excluded.stability, baseline_count=excluded.baseline_count, outcome_valid=excluded.outcome_valid, split_role=excluded.split_role", rows)
            self.conn.executemany("UPDATE mouse_events SET before_frame_id=?, after_frame_id=?, action_time=?, post_action_delay_ms=?, score_delta=?, reward_delta=?, outcome_valid=? WHERE id=?", [(item["before_frame_id"], item["after_frame_id"], item["action_time"], item["post_action_delay_ms"], float(item.get("score_delta", 0.0) or 0.0), float(item.get("score_delta", 0.0) or 0.0), 1 if item.get("outcome_valid") else 0, item["mouse_event_id"]) for item in outcomes])
            frame_ids = [frame_id for item in outcomes for frame_id in (item.get("before_frame_id"), item.get("after_frame_id")) if frame_id]
            if frame_ids:
                self.conn.executemany("UPDATE frames SET action_outcome_information=MAX(action_outcome_information, ?) WHERE id=?", [(min(1.0, abs(float(item.get("action_advantage", 0.0))) + (0.25 if item.get("outcome_valid") else 0.0)), frame_id) for item in outcomes for frame_id in (item.get("before_frame_id"), item.get("after_frame_id")) if frame_id])
                self._refresh_retain_values_locked(frame_ids)
            self.conn.commit()

    def _recalculate_model_refs_locked(self, frame_ids):
        unique = set(frame_ids or [])
        for frame_id in unique:
            self.conn.execute("UPDATE frames SET model_dependency_count=(SELECT COUNT(DISTINCT model_id) FROM model_frame_refs WHERE model_frame_refs.frame_id=frames.id), model_refs=(SELECT COUNT(DISTINCT model_id) FROM model_frame_refs WHERE model_frame_refs.frame_id=frames.id) WHERE id=?", (frame_id,))
        self._refresh_retain_values_locked(unique)

    def _read_model_payload(self, path):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            identifier = str(payload.get("id") or Path(path).stem)
            return identifier, payload
        except Exception:
            return Path(path).stem, {}

    def _sync_model_metadata_locked(self):
        if self.models is None:
            return
        existing = {str(row[0]): int(row[1] or 0) for row in self.conn.execute("SELECT id, champion FROM model_metadata").fetchall()}
        seen = set()
        for path in self.model_files():
            identifier, payload = self._read_model_payload(path)
            seen.add(identifier)
            champion = 1 if existing.get(identifier, 0) else 0
            self.conn.execute("INSERT INTO model_metadata(id, file_name, created, quality, validation_quality, champion, updated) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET file_name=excluded.file_name, created=excluded.created, quality=excluded.quality, validation_quality=excluded.validation_quality, champion=model_metadata.champion, updated=excluded.updated", (identifier, path.name, float(payload.get("trained_at", payload.get("validated_at", 0.0)) or 0.0), float(payload.get("quality", 0.0) or 0.0), float(payload.get("validation_quality", payload.get("quality", 0.0)) or 0.0), champion, time.time()))
        stale = [row[0] for row in self.conn.execute("SELECT id FROM model_metadata").fetchall() if row[0] not in seen]
        if stale:
            marks = ",".join("?" for _ in stale)
            frame_ids = [row[0] for row in self.conn.execute("SELECT DISTINCT frame_id FROM model_frame_refs WHERE model_id IN (" + marks + ")", stale).fetchall()]
            self.conn.execute("DELETE FROM model_frame_refs WHERE model_id IN (" + marks + ")", stale)
            self.conn.execute("DELETE FROM model_metadata WHERE id IN (" + marks + ")", stale)
            self._recalculate_model_refs_locked(frame_ids)
        rows = self.conn.execute("SELECT id FROM model_metadata WHERE champion=1 ORDER BY validation_quality DESC, quality DESC, updated DESC, id DESC").fetchall()
        if len(rows) > 1:
            winner = str(rows[0][0])
            self.conn.execute("UPDATE model_metadata SET champion=CASE WHEN id=? THEN 1 ELSE 0 END WHERE champion=1", (winner,))

    def sync_model_metadata(self):
        delegated, result = self._dispatch_write_to_writer("sync_model_metadata")
        if delegated:
            return result
        with self.lock:
            if self.conn is None:
                return
            self._sync_model_metadata_locked()
            self.conn.commit()

    def register_model_metadata(self, model_id, path, payload, outcomes=None, validation_outcomes=None):
        delegated, result = self._dispatch_write_to_writer("register_model_metadata", model_id, path, payload, outcomes, validation_outcomes)
        if delegated:
            return result
        refs = []
        for role, items in (("train", outcomes or ()), ("validation", validation_outcomes or ())):
            for item in items:
                example = item[-1] if isinstance(item, tuple) else item
                for field in ("before_frame_id", "after_frame_id"):
                    frame_id = example.get(field) if isinstance(example, dict) else None
                    if frame_id:
                        refs.append((str(model_id), frame_id, role))
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            if payload.get("champion", False):
                self.conn.execute("UPDATE model_metadata SET champion=0")
            self.conn.execute("INSERT INTO model_metadata(id, file_name, created, quality, validation_quality, champion, updated) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET file_name=excluded.file_name, created=excluded.created, quality=excluded.quality, validation_quality=excluded.validation_quality, champion=excluded.champion, updated=excluded.updated", (str(model_id), Path(path).name, float(payload.get("trained_at", time.time())), float(payload.get("quality", 0.0)), float(payload.get("validation_quality", payload.get("quality", 0.0))), 1 if payload.get("champion", False) else 0, time.time()))
            if refs:
                self.conn.executemany("INSERT OR IGNORE INTO model_frame_refs(model_id, frame_id, role) VALUES (?, ?, ?)", refs)
                self._recalculate_model_refs_locked([item[1] for item in refs])
            self._commit_critical_locked()

    def save_model_frame_refs(self, model_id, outcomes, validation_outcomes):
        delegated, result = self._dispatch_write_to_writer("save_model_frame_refs", model_id, outcomes, validation_outcomes)
        if delegated:
            return result
        refs = []
        for role, items in (("train", outcomes), ("validation", validation_outcomes)):
            for item in items:
                example = item[-1] if isinstance(item, tuple) else item
                for field in ("before_frame_id", "after_frame_id"):
                    frame_id = example.get(field) if isinstance(example, dict) else None
                    if frame_id:
                        refs.append((model_id, frame_id, role))
        if not refs:
            return
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.executemany("INSERT OR IGNORE INTO model_frame_refs(model_id, frame_id, role) VALUES (?, ?, ?)", refs)
            self._recalculate_model_refs_locked([item[1] for item in refs])
            self.conn.commit()

    def record_sleep_decision(self, features, expected_gain):
        delegated, result = self._dispatch_write_to_writer("record_sleep_decision", features, expected_gain)
        if delegated:
            return result
        identifier = uuid.uuid4().hex
        with self.lock:
            if self.conn is None:
                return None
            self.conn.execute("INSERT INTO sleep_decision_samples(id, created, features, expected_gain) VALUES (?, ?, ?, ?)", (identifier, time.time(), json.dumps(features, ensure_ascii=False), float(expected_gain)))
            self.conn.commit()
        return identifier

    def finalize_sleep_decision(self, identifier, training_result, cleanup_bytes, duration_seconds, restored_training_gain):
        if not identifier:
            return
        result = dict(training_result or {})
        status = str(result.get("status", "failed"))
        reason = str(result.get("reason", ""))
        trained = status == "trained" and bool(result.get("champion_persisted"))
        quality_delta = float(result.get("quality_delta", 0.0)) if trained else None
        restored = float(restored_training_gain or 0.0) if trained else 0.0
        with self.lock:
            if self.conn is None:
                return
            self.conn.execute(
                "UPDATE sleep_decision_samples SET actual_quality_delta=?, cleanup_bytes=?, duration_seconds=?, restored_training_gain=?, training_status=?, training_reason=?, training_samples=?, validation_samples=?, outcome_ready=1 WHERE id=?",
                (quality_delta, int(cleanup_bytes or 0), float(duration_seconds or 0.0), restored, status, reason, int(result.get("training_samples", 0) or 0), int(result.get("validation_samples", 0) or 0), identifier),
            )
            self._commit_critical_locked()

    def sleep_decision_history(self, limit=128):
        with self.lock:
            if self.conn is None:
                return []
            rows = self.conn.execute("SELECT features, expected_gain, actual_quality_delta, cleanup_bytes, duration_seconds, restored_training_gain, training_status, training_reason, training_samples, validation_samples FROM sleep_decision_samples WHERE outcome_ready=1 ORDER BY created DESC LIMIT ?", (int(limit),)).fetchall()
        result = []
        for features, expected_gain, quality_delta, cleanup, duration, restored, status, reason, training_samples, validation_samples in rows:
            try:
                item = json.loads(features)
                item.update({"expected_gain": float(expected_gain or 0.0), "actual_quality_delta": float(quality_delta or 0.0), "cleanup_bytes": int(cleanup or 0), "duration_seconds": float(duration or 0.0), "restored_training_gain": float(restored or 0.0), "training_status": str(status or ""), "training_reason": str(reason or ""), "training_samples": int(training_samples or 0), "validation_samples": int(validation_samples or 0)})
                result.append(item)
            except Exception:
                pass
        return result

    def _immediate_exact_reward_locked(self, frame_id, score, session_id=None):
        reward = float(score)
        self.conn.execute("UPDATE frames SET hunger=0.0, reward=?, raw_score=?, raw_reward=?, reward_source='screen_score_only', score_valid_for_training=1 WHERE id=?", (reward, reward, reward, frame_id))
        self._refresh_retain_values_locked([frame_id])

    def _recalculate_session_scores_locked(self, session_id=None):
        rows = self.conn.execute("""
            SELECT id, score, score_valid
            FROM frames WHERE (? IS NULL OR session_id=?)
            ORDER BY capture_finished_monotonic_ns ASC, id ASC
        """, (session_id, session_id)).fetchall()
        updates = []
        for frame_id, score, score_valid in rows:
            if not score_valid or score is None:
                continue
            reward = float(score)
            updates.append((reward, reward, reward, frame_id))
        if updates:
            self.conn.executemany("UPDATE frames SET hunger=0.0, reward=?, raw_score=?, raw_reward=?, reward_source='screen_score_only', score_valid_for_training=1, score_generation=CASE WHEN score_generation='deferred_exact' THEN score_generation ELSE 'recomputed' END WHERE id=?", updates)
            self._refresh_retain_values_locked([item[-1] for item in updates])

    def _refresh_retain_values_locked(self, frame_ids=None):
        where = ""
        params = ()
        if frame_ids:
            unique = sorted({str(item) for item in frame_ids if item})
            if not unique:
                return
            where = "WHERE f.id IN ({})".format(",".join("?" for _ in unique))
            params = tuple(unique)
        rows = self.conn.execute("""
            SELECT f.id, f.state_support_count, COALESCE(f.score, f.raw_score), f.capture_complete, f.variance,
                   f.validation_last_used, f.model_dependency_count, f.model_refs,
                   f.created, f.session_id,
                   COALESCE((SELECT MAX(a.action_advantage) FROM action_outcomes a WHERE a.before_frame_id=f.id OR a.after_frame_id=f.id), 0),
                   COALESCE((SELECT COUNT(*) FROM action_outcomes a WHERE a.before_frame_id=f.id OR a.after_frame_id=f.id), 0),
                   COALESCE((SELECT COUNT(*) FROM frames later WHERE later.session_id=f.session_id AND later.dhash64=f.dhash64 AND later.created>f.created), 0)
            FROM frames f
        """ + where, params).fetchall()
        now = time.time()
        updates = []
        for frame_id, support, reward, complete, variance, validation_used, dependency_count, model_refs, created, session_id, action_info, action_refs, duplicates in rows:
            rarity = 1.0 / max(1.0, float(support or 1))
            action_value = min(0.85, max(0.0, float(action_info or 0.0)) * 2.0 + min(0.25, float(action_refs or 0) * 0.08))
            model_value = min(0.90, float(dependency_count or 0) * 0.20 + float(model_refs or 0) * 0.10)
            validation_value = 0.45 if float(validation_used or 0.0) > 0 else 0.0
            coverage_value = 0.25 if float(created or 0.0) <= now - 60.0 else 0.10
            reward_number = float(reward or 0.0)
            reward_value = min(0.55, max(0.0, reward_number))
            negative_penalty = min(0.75, max(0.0, -reward_number) + max(0.0, -float(action_info or 0.0)) * 2.0)
            quality_value = (0.20 if int(complete or 0) else 0.0) + min(0.15, max(0.0, float(variance or 0.0) / 2550.0))
            duplicate_penalty = min(0.80, float(duplicates or 0) * 0.10)
            provisional_penalty = 0.20 if reward is None and float(created or 0.0) < now - 3600.0 else 0.0
            retain = max(0.0, rarity + action_value + model_value + validation_value + coverage_value + reward_value + quality_value - duplicate_penalty - negative_penalty - provisional_penalty)
            updates.append((retain, frame_id))
        if updates:
            self.conn.executemany("UPDATE frames SET retain_value=?, retain_version=2 WHERE id=?", updates)

    def session_score_summary(self, session_id):
        with self.lock:
            if self.conn is None or not session_id:
                return {"valid_frames": 0, "scores": [], "latest": None}
            count = int(self.conn.execute("SELECT COUNT(*) FROM frames WHERE session_id=? AND score_valid=1 AND score_status='exact' AND score IS NOT NULL", (session_id,)).fetchone()[0] or 0)
            scores = [float(row[0]) for row in self.conn.execute("SELECT score FROM frames WHERE session_id=? AND score_valid=1 AND score_status='exact' AND score IS NOT NULL ORDER BY capture_finished_monotonic_ns DESC, id DESC LIMIT 120", (session_id,)).fetchall()]
            latest = self.conn.execute("SELECT id, score, hunger, reward, capture_finished_monotonic_ns FROM frames WHERE session_id=? AND score_valid=1 AND score_status='exact' AND score IS NOT NULL ORDER BY capture_finished_monotonic_ns DESC, id DESC LIMIT 1", (session_id,)).fetchone()
        scores.reverse()
        return {"valid_frames": count, "scores": scores, "latest": latest}

    def _exact_entries_from_snapshot(self, current_dhash, current_features, history_rows, limit, cancelled=None, cooperative=None):
        heap = []
        scanned = 0
        wanted = max(1, int(limit))
        for index, row in enumerate(history_rows or []):
            if cancelled is not None and cancelled():
                return [item[1] for item in sorted(heap, key=lambda value: value[0], reverse=True)], scanned, "cancelled"
            if cooperative is not None and index % 256 == 0 and not cooperative():
                return [item[1] for item in sorted(heap, key=lambda value: value[0], reverse=True)], scanned, "pressure"
            entry = self._frame_composite_entry(current_dhash, current_features, row)
            if entry is None:
                continue
            scanned += 1
            try:
                tie = -int(entry["frame_id"], 16)
            except ValueError:
                tie = -sum(ord(part) for part in entry["frame_id"])
            priority = (float(entry["similarity"]), -float(entry["distance"]), tie)
            item = (priority, entry)
            if len(heap) < wanted:
                heapq.heappush(heap, item)
            elif priority > heap[0][0]:
                heapq.heapreplace(heap, item)
        entries = [item[1] for item in heap]
        entries.sort(key=lambda item: (-item["similarity"], item["distance"], item["frame_id"]))
        return entries, scanned, "complete"

    def _exact_entries_from_database_full(self, current_dhash, current_features, finished_ns, limit, cancelled=None, cooperative=None, chunk_size=512):
        heap = []
        scanned = 0
        wanted = max(1, int(limit))
        chunk_size = max(128, min(1024, int(chunk_size or 512)))
        last_ns = -1
        last_id = ""
        while True:
            if cancelled is not None and cancelled():
                return [item[1] for item in sorted(heap, key=lambda value: value[0], reverse=True)], scanned, "cancelled"
            if cooperative is not None and not cooperative():
                return [item[1] for item in sorted(heap, key=lambda value: value[0], reverse=True)], scanned, "pressure"
            with self.lock:
                if self.conn is None:
                    return [item[1] for item in sorted(heap, key=lambda value: value[0], reverse=True)], scanned, "closed"
                rows = self.conn.execute(
                    "SELECT id, dhash64, phash, gray32x18, edge_density, color_histogram, capture_finished_monotonic_ns FROM frames "
                    "WHERE (dhash64 IS NOT NULL OR phash IS NOT NULL) AND capture_finished_monotonic_ns < ? "
                    "AND (capture_finished_monotonic_ns > ? OR (capture_finished_monotonic_ns = ? AND id > ?)) "
                    "ORDER BY capture_finished_monotonic_ns ASC, id ASC LIMIT ?",
                    (int(finished_ns), int(last_ns), int(last_ns), str(last_id), int(chunk_size)),
                ).fetchall()
            if not rows:
                break
            for index, row in enumerate(rows):
                if cancelled is not None and cancelled():
                    return [item[1] for item in sorted(heap, key=lambda value: value[0], reverse=True)], scanned, "cancelled"
                if cooperative is not None and index % 128 == 0 and not cooperative():
                    return [item[1] for item in sorted(heap, key=lambda value: value[0], reverse=True)], scanned, "pressure"
                last_id = str(row[0])
                last_ns = int(row[6] or 0)
                entry = self._frame_composite_entry(current_dhash, current_features, row[:6])
                if entry is None:
                    continue
                scanned += 1
                try:
                    tie = -int(entry["frame_id"], 16)
                except ValueError:
                    tie = -sum(ord(part) for part in entry["frame_id"])
                priority = (float(entry["similarity"]), -float(entry["distance"]), tie)
                item = (priority, entry)
                if len(heap) < wanted:
                    heapq.heappush(heap, item)
                elif priority > heap[0][0]:
                    heapq.heapreplace(heap, item)
            if len(rows) < chunk_size:
                break
        entries = [item[1] for item in heap]
        entries.sort(key=lambda item: (-item["similarity"], item["distance"], item["frame_id"]))
        return entries, scanned, "complete"

    def _exact_entries_from_database(self, current_dhash, current_features, finished_ns, limit, cancelled=None, cooperative=None, chunk_size=512):
        wanted = max(1, int(limit))
        return self._exact_entries_from_database_full(current_dhash, current_features, finished_ns, wanted, cancelled, cooperative, chunk_size)

    def process_deferred_exact_scores(self, cancelled=None, cooperative=None, maximum=1, session_id=None):
        delegated, result = self._dispatch_write_to_writer("process_deferred_exact_scores", cancelled, cooperative, maximum, session_id)
        if delegated:
            return result
        if not self.exact_score_lock.acquire(blocking=False):
            return 0
        try:
            maximum = max(1, int(maximum))
            with self.lock:
                if self.conn is None:
                    return 0
                predicate = ""
                params = []
                if session_id:
                    predicate = " AND f.session_id=?"
                    params.append(str(session_id))
                rows = self.conn.execute(
                    "SELECT d.frame_id, d.dhash64, f.session_id, f.phash, f.gray32x18, f.edge_density, f.color_histogram, f.capture_finished_monotonic_ns FROM deferred_exact_scores d JOIN frames f ON f.id=d.frame_id "
                    "WHERE d.state='pending'{} ORDER BY f.capture_finished_monotonic_ns ASC, f.id ASC LIMIT ?".format(predicate),
                    tuple(params + [maximum]),
                ).fetchall()
            if not rows:
                return 0
            resolved = 0
            valid_updates = []
            invalid_updates = []
            chunk_size = 512
            for frame_id, dhash, session_key, phash, gray, edge, histogram, finished_ns in rows:
                if cancelled is not None and cancelled():
                    break
                finished_ns = int(finished_ns or 0)
                current_features = {"phash": phash, "gray32x18": gray, "edge_density": edge, "color_histogram": histogram}
                with self.lock:
                    history_count = int(self.conn.execute("SELECT COUNT(*) FROM frames WHERE (dhash64 IS NOT NULL OR phash IS NOT NULL) AND capture_finished_monotonic_ns < ?", (finished_ns,)).fetchone()[0] or 0)
                if history_count <= 0:
                    result = {"hashes": [], "frame_ids": [], "similarities": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "warmup_no_history", "exact_or_approx": "exact", "recall_guard": True, "total_history": 0, "score_valid": True, "provisional": False}
                    meta_status = "complete"
                else:
                    entries, scanned, meta_status = self._exact_entries_from_database(dhash, current_features, finished_ns, 8, cancelled, cooperative, chunk_size)
                    complete = meta_status == "complete" and len(entries) >= min(8, history_count)
                    result = {"hashes": [item["hash"] for item in entries], "frame_ids": [item["frame_id"] for item in entries], "similarities": [float(item["similarity"]) for item in entries], "candidate_count": scanned, "top_k_distance": float(max((item["distance"] for item in entries), default=64.0)), "retrieval_fallback": False, "retrieval_mode": "chunked_exact_composite" if complete else "chunked_exact_composite_" + meta_status, "exact_or_approx": "exact" if complete else "unknown", "recall_guard": bool(complete), "total_history": history_count, "score_valid": bool(complete), "provisional": False, "candidate_sources": {"chunked_scan": scanned, "chunk_size": chunk_size}}
                if meta_status != "complete":
                    break
                score, meta = frame_score(dhash, result, current_features)
                if not meta.get("score_valid") or score is None:
                    invalid_updates.append((time.time(), str(meta.get("retrieval_mode", "deferred")), frame_id))
                    continue
                valid_updates.append((float(score), float(score), str(meta.get("retrieval_mode", "chunked_exact_composite")), frame_id, str(session_key)))
            if valid_updates or invalid_updates:
                sessions = sorted({item[4] for item in valid_updates if item[4]})
                with self.lock:
                    self.conn.execute("BEGIN IMMEDIATE")
                    for score, raw_score, mode, frame_id, session_key in valid_updates:
                        self.conn.execute("UPDATE frames SET score=?, raw_score=?, score_status='exact', score_generation='deferred_exact', score_valid=1, score_valid_for_training=1, reward_source='screen_score_only', score_retrieval_mode=?, score_exact_or_approx='exact', score_recall_guard=1 WHERE id=?", (score, raw_score, mode, frame_id))
                        self._immediate_exact_reward_locked(frame_id, score, session_key)
                        self.conn.execute("UPDATE deferred_exact_scores SET state='complete', updated=?, attempts=attempts+1, last_error='' WHERE frame_id=?", (time.time(), frame_id))
                        resolved += 1
                    for updated, error, frame_id in invalid_updates:
                        self.conn.execute("UPDATE deferred_exact_scores SET attempts=attempts+1, updated=?, state=CASE WHEN attempts+1>=3 THEN 'failed' ELSE 'pending' END, last_error=? WHERE frame_id=?", (updated, error, frame_id))
                    for session_key in sessions:
                        self._recalculate_session_scores_locked(session_key)
                    self.conn.commit()
            return resolved
        finally:
            self.exact_score_lock.release()

    def compute_exact_score_for_image(self, image, cancelled=None, cooperative=None, deadline=None, limit=8):
        if not isinstance(image, dict):
            return None, {"score_valid": False, "retrieval_mode": "invalid_image"}
        if deadline is not None and time.monotonic() >= float(deadline):
            return None, {"score_valid": False, "retrieval_mode": "fast_exact_deadline"}
        dhash = str(image.get("dhash64") or image.get("phash") or "")
        if not dhash:
            return None, {"score_valid": False, "retrieval_mode": "fast_exact_missing_hash"}
        finished_ns = int(image.get("capture_finished_monotonic_ns", 0) or 0)
        current_features = {"phash": image.get("phash"), "gray32x18": image.get("gray32x18"), "edge_density": image.get("edge_density", 0.0), "color_histogram": image.get("color_histogram")}
        with self.lock:
            if self.conn is None:
                return None, {"score_valid": False, "retrieval_mode": "fast_exact_storage_closed"}
            history_count = int(self.conn.execute("SELECT COUNT(*) FROM frames WHERE (dhash64 IS NOT NULL OR phash IS NOT NULL) AND capture_finished_monotonic_ns < ?", (finished_ns,)).fetchone()[0] or 0)
        if history_count <= 0:
            result = {"hashes": [], "frame_ids": [], "similarities": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "warmup_no_history", "exact_or_approx": "exact", "recall_guard": True, "total_history": 0, "score_valid": True, "provisional": False}
        else:
            entries, scanned, status = self._exact_entries_from_database(dhash, current_features, finished_ns, limit, cancelled, cooperative, 256)
            complete = status == "complete" and len(entries) >= min(int(limit), history_count)
            if not complete:
                return None, {"score_valid": False, "retrieval_mode": "fast_exact_" + status, "candidate_count": scanned, "provisional": False}
            result = {"hashes": [item["hash"] for item in entries], "frame_ids": [item["frame_id"] for item in entries], "similarities": [float(item["similarity"]) for item in entries], "candidate_count": scanned, "top_k_distance": float(max((item["distance"] for item in entries), default=64.0)), "retrieval_fallback": False, "retrieval_mode": "training_fast_path_exact", "exact_or_approx": "exact", "recall_guard": True, "total_history": history_count, "score_valid": True, "provisional": False, "candidate_sources": {"fast_path": scanned}}
        score, meta = frame_score(dhash, result, current_features)
        meta["score_status"] = "exact" if meta.get("score_valid") and score is not None else "invalid"
        meta["provisional"] = False
        return score, meta

    def mark_deferred_exact_timeout(self, session_id, timeout_seconds, limit=100000):
        delegated, result = self._dispatch_write_to_writer("mark_deferred_exact_timeout", session_id, timeout_seconds, limit)
        if delegated:
            return result
        with self.lock:
            if self.conn is None or not session_id:
                return 0
            rows = self.conn.execute("""
                SELECT d.frame_id FROM deferred_exact_scores d
                JOIN frames f ON f.id=d.frame_id
                WHERE f.session_id=? AND d.state='pending'
                ORDER BY d.created ASC LIMIT ?
            """, (str(session_id), int(limit))).fetchall()
            ids = [str(row[0]) for row in rows]
            if not ids:
                return 0
            self.conn.executemany("UPDATE deferred_exact_scores SET updated=?, attempts=attempts+1, state='exact_timeout', last_error=? WHERE frame_id=?", [(time.time(), "停止排空超过 {:.1f} 秒".format(float(timeout_seconds)), item) for item in ids])
            self.conn.execute("UPDATE sessions SET trainable=0, training_exclusion_reason=? WHERE id=?", ("存在停止排空 exact_timeout 帧", str(session_id)))
            self.conn.execute("INSERT INTO system_events(id, session_id, created, kind, payload) VALUES (?, ?, ?, ?, ?)", (uuid.uuid4().hex, str(session_id), time.time(), "exact_score_drain_timeout", json.dumps({"timeout_seconds": float(timeout_seconds), "pending": len(ids)}, ensure_ascii=False)))
            self.conn.commit()
            return len(ids)

    def deferred_score_status(self, session_id=None):
        with self.lock:
            if self.conn is None:
                return {"pending": 0, "failed": 0, "oldest": None}
            predicate = ""
            params = ()
            if session_id:
                predicate = " JOIN frames f ON f.id=d.frame_id WHERE f.session_id=? AND "
                params = (str(session_id),)
                pending, oldest = self.conn.execute("SELECT COUNT(*), MIN(d.created) FROM deferred_exact_scores d{}d.state='pending'".format(predicate), params).fetchone()
                failed = self.conn.execute("SELECT COUNT(*) FROM deferred_exact_scores d{}d.state='failed'".format(predicate), params).fetchone()[0]
            else:
                pending, oldest = self.conn.execute("SELECT COUNT(*), MIN(created) FROM deferred_exact_scores WHERE state='pending'").fetchone()
                failed = self.conn.execute("SELECT COUNT(*) FROM deferred_exact_scores WHERE state='failed'").fetchone()[0]
        return {"pending": int(pending or 0), "failed": int(failed or 0), "oldest": float(oldest) if oldest is not None else None}

    def exclude_failed_deferred_sessions(self):
        delegated, result = self._dispatch_write_to_writer("exclude_failed_deferred_sessions")
        if delegated:
            return result
        with self.lock:
            rows = self.conn.execute("""
                SELECT DISTINCT f.session_id FROM deferred_exact_scores d
                JOIN frames f ON f.id=d.frame_id WHERE d.state='failed'
            """).fetchall()
            sessions = [str(row[0]) for row in rows]
            for session_id in sessions:
                details = self.conn.execute("""
                    SELECT f.id, d.last_error FROM deferred_exact_scores d JOIN frames f ON f.id=d.frame_id
                    WHERE d.state='failed' AND f.session_id=? ORDER BY f.capture_finished_monotonic_ns ASC, f.id ASC
                """, (session_id,)).fetchall()
                reason = "延迟精确评分失败三次：" + "; ".join("{}:{}".format(fid, err) for fid, err in details[:8])
                self.conn.execute("UPDATE sessions SET trainable=0, training_exclusion_reason=? WHERE id=?", (reason, session_id))
                self.conn.execute("INSERT INTO system_events(id, session_id, created, kind, payload) VALUES (?, ?, ?, ?, ?)", (uuid.uuid4().hex, session_id, time.time(), "deferred_score_failed_training_excluded", json.dumps({"reason": reason}, ensure_ascii=False)))
            self._trim_system_events_locked()
            self.conn.commit()
        return sessions

    def _ledger_values_locked(self):
        keys = ("asset_bytes", "reserved_asset_bytes", "database_bytes", "transient_bytes", "other_bytes", "last_reconciled_at")
        rows = dict(self.conn.execute("SELECT key, value FROM pool_meta WHERE key IN ({})".format(",".join("?" for _ in keys)), keys).fetchall())
        return {key: max(0, int(rows.get(key, 0) or 0)) for key in keys}

    def reconcile_pool_ledger(self):
        delegated, result = self._dispatch_write_to_writer("reconcile_pool_ledger")
        if delegated:
            return result
        result = {"frame_asset_bytes": 0, "reserved_asset_bytes": 0, "database_bytes": 0, "transient_bytes": 0, "other_bytes": 0, "experience_total_bytes": 0, "last_reconciled_at": int(time.time())}
        if self.pool is None:
            return result
        try:
            for item in self._iter_storage_files(self.pool, None, "reconcile_pool_walk"):
                try:
                    if not item.is_file():
                        continue
                    size = int(item.stat().st_size)
                    relative = item.relative_to(self.pool)
                    name = item.name.lower()
                    result["experience_total_bytes"] += size
                    if relative.parts and relative.parts[0].lower() == "screens":
                        result["frame_asset_bytes"] += size
                    elif name.startswith("records.sqlite3") or name in ("records.sqlite3-wal", "records.sqlite3-shm", "records.sqlite3-journal"):
                        result["database_bytes"] += size
                    elif "trash" in {part.lower() for part in relative.parts} or name.endswith(".tmp") or name.startswith(".write_latency_probe"):
                        result["transient_bytes"] += size
                    else:
                        result["other_bytes"] += size
                except OSError:
                    pass
        except OSError:
            pass
        with self.lock:
            if self.conn is not None:
                values = (
                    ("asset_bytes", int(result["frame_asset_bytes"])),
                    ("total_asset_bytes", int(result["frame_asset_bytes"])),
                    ("reserved_asset_bytes", 0),
                    ("database_bytes", int(result["database_bytes"])),
                    ("transient_bytes", int(result["transient_bytes"])),
                    ("other_bytes", int(result["other_bytes"])),
                    ("last_reconciled_at", int(result["last_reconciled_at"])),
                )
                self.conn.executemany("INSERT OR REPLACE INTO pool_meta(key, value) VALUES (?, ?)", values)
                self.conn.commit()
        return result

    def pool_breakdown(self, reconcile=False):
        if reconcile:
            return self.reconcile_pool_ledger()
        with self.lock:
            if self.conn is None:
                return {"frame_asset_bytes": 0, "reserved_asset_bytes": 0, "database_bytes": 0, "transient_bytes": 0, "other_bytes": 0, "experience_total_bytes": 0, "last_reconciled_at": 0}
            values = self._ledger_values_locked()
        total = values["asset_bytes"] + values["database_bytes"] + values["transient_bytes"] + values["other_bytes"]
        return {"frame_asset_bytes": values["asset_bytes"], "reserved_asset_bytes": values["reserved_asset_bytes"], "database_bytes": values["database_bytes"], "transient_bytes": values["transient_bytes"], "other_bytes": values["other_bytes"], "experience_total_bytes": total, "last_reconciled_at": values["last_reconciled_at"]}

    def capacity_status(self):
        with self.lock:
            if self.conn is None:
                return {"blocked": False, "target": 0, "remaining": 0}
            rows = dict(self.conn.execute("SELECT key, value FROM pool_meta WHERE key IN ('pool_capacity_blocked','pool_capacity_target','pool_capacity_remaining','pool_capacity_updated','pool_capacity_tier','pool_capacity_transaction_reserve')").fetchall())
        return {"blocked": bool(rows.get("pool_capacity_blocked", 0)), "target": int(rows.get("pool_capacity_target", 0)), "remaining": int(rows.get("pool_capacity_remaining", 0)), "updated": int(rows.get("pool_capacity_updated", 0)), "tier": int(rows.get("pool_capacity_tier", 0)), "transaction_reserve": int(rows.get("pool_capacity_transaction_reserve", self.transaction_reserve_bytes))}

    def _set_capacity_status(self, blocked, target, remaining):
        with self.lock:
            values = (("pool_capacity_blocked", 1 if blocked else 0), ("pool_capacity_target", int(target)), ("pool_capacity_remaining", int(remaining)), ("pool_capacity_updated", int(time.time())))
            self.conn.executemany("INSERT OR REPLACE INTO pool_meta(key, value) VALUES (?, ?)", values)
            self.conn.commit()


    def _safe_unlink_storage(self, path, operation):
        try:
            target = self._assert_storage_path(path, operation)
            if _is_windows_reparse_point(target):
                self._critical_path_event(target, operation)
                return False
            target.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def _iter_storage_files(self, root, suffix=None, operation="walk"):
        try:
            base = self._assert_storage_path(root, operation)
            if not base.exists() or _is_windows_reparse_point(base):
                return []
            suffix = str(suffix or "")
            result = []
            for current, dirs, files in os.walk(str(base), topdown=True, followlinks=False):
                current_path = Path(current)
                try:
                    self._assert_storage_path(current_path, operation)
                    if _is_windows_reparse_point(current_path):
                        dirs[:] = []
                        continue
                except OSError:
                    dirs[:] = []
                    continue
                kept = []
                for name in dirs:
                    child = current_path / name
                    try:
                        self._assert_storage_path(child, operation)
                        if not _is_windows_reparse_point(child):
                            kept.append(name)
                    except OSError:
                        self._critical_path_event(child, operation)
                dirs[:] = kept
                for name in files:
                    item = current_path / name
                    if suffix and not item.name.lower().endswith(suffix.lower()):
                        continue
                    try:
                        self._assert_storage_path(item, operation)
                        result.append(item)
                    except OSError:
                        self._critical_path_event(item, operation)
            return result
        except OSError:
            return []


    def _cleanup_pool_files(self):
        if self.pool is None:
            return
        referenced = set()
        with self.lock:
            if self.conn is not None:
                referenced = {str(row[0]) for row in self.conn.execute("SELECT screenshot_path FROM frames").fetchall()}
        for item in self._iter_storage_files(self.pool, ".tmp", "cleanup_tmp_walk"):
            try:
                self._safe_unlink_storage(item, "cleanup_tmp")
            except OSError:
                pass
        if self.screens and self.screens.exists():
            for item in self._iter_storage_files(self.screens, ".png", "cleanup_screens_walk"):
                try:
                    if str(item.relative_to(self.pool)) not in referenced:
                        self._safe_unlink_storage(item, "cleanup_orphan_png")
                except OSError:
                    pass
        trash = self.pool / "trash"
        if trash.exists():
            for item in self._iter_storage_files(trash, None, "cleanup_trash_walk"):
                try:
                    if item.is_file():
                        self._safe_unlink_storage(item, "cleanup_trash_file")
                except OSError:
                    pass

    def _database_bytes_fast(self):
        now = time.monotonic()
        with self.lock:
            if self.pool is None:
                return 0
            if now - self._database_bytes_checked < 5.0:
                return int(self._database_bytes_cached)
        return self._database_bytes_now()

    def _database_bytes_now(self):
        with self.lock:
            if self.pool is None or self.conn is None:
                return 0
            total = self._database_bytes_precise_locked()
            self._database_bytes_cached = total
            self._database_bytes_checked = time.monotonic()
            self.conn.execute("INSERT OR REPLACE INTO pool_meta(key, value) VALUES ('database_bytes', ?)", (int(total),))
            self.conn.commit()
            return total

    def pool_size_fast(self):
        return int(self.pool_breakdown(False).get("experience_total_bytes", 0))

    def pool_size(self):
        return int(self.pool_breakdown(True).get("experience_total_bytes", 0))

    def _reconcile_asset_bytes(self):
        return self.reconcile_pool_ledger()

    def prune_models(self, maximum, cancelled=None, cooperative=None, batch_size=8):
        delegated, result = self._dispatch_write_to_writer("prune_models", maximum, cancelled, cooperative, batch_size)
        if delegated:
            return result
        maximum = max(1, int(maximum))
        self.sync_model_metadata()
        with self.lock:
            rows = self.conn.execute("""
                SELECT m.id, m.file_name, m.validation_quality, m.quality, m.champion, m.updated,
                       COALESCE((SELECT COUNT(DISTINCT frame_id) FROM model_frame_refs r WHERE r.model_id=m.id), 0)
                FROM model_metadata m
            """).fetchall()
            champion_rows = self.conn.execute("SELECT id FROM model_metadata WHERE champion=1").fetchall()
            champion_ids = {str(row[0]) for row in champion_rows}
            champion_clusters = set()
            if champion_ids:
                marks = ",".join("?" for _ in champion_ids)
                champion_clusters = {str(row[0]) for row in self.conn.execute("""SELECT DISTINCT frames.state_cluster_id FROM model_frame_refs JOIN frames ON frames.id=model_frame_refs.frame_id WHERE model_frame_refs.model_id IN ({}) AND frames.state_cluster_id IS NOT NULL AND frames.state_cluster_id!=''""".format(marks), tuple(champion_ids)).fetchall()}
            model_clusters = {}
            for model_id, *_ in rows:
                model_clusters[str(model_id)] = {str(row[0]) for row in self.conn.execute("""SELECT DISTINCT frames.state_cluster_id FROM model_frame_refs JOIN frames ON frames.id=model_frame_refs.frame_id WHERE model_frame_refs.model_id=? AND frames.state_cluster_id IS NOT NULL AND frames.state_cluster_id!=''""", (str(model_id),)).fetchall()}
        initial = len(rows)
        with self.lock:
            normalized_champions = int(self.conn.execute("SELECT COUNT(*) FROM model_metadata WHERE champion=1").fetchone()[0] or 0)
        champion_normalized = (normalized_champions == 1) if initial > 0 else (normalized_champions == 0)
        if initial <= maximum:
            self.last_model_prune_result = {"initial": initial, "removed": 0, "target": initial, "remaining": initial, "champions": normalized_champions, "champion_normalized": champion_normalized, "success": champion_normalized}
            if not champion_normalized and not (cancelled is not None and cancelled()):
                self.add_system_event(None, "model_champion_normalization_failed", dict(self.last_model_prune_result))
            return 0
        target = max(1, int(math.floor(maximum * 0.5)))
        ranked = []
        for row in rows:
            model_id, file_name, validation_quality, quality, champion, updated, ref_count = row
            clusters = model_clusters.get(str(model_id), set())
            overlap = len(clusters & champion_clusters) / max(1, len(clusters)) if champion_clusters else 1.0
            unique_count = len(clusters - champion_clusters) if champion_clusters else 0
            validation_value = float(validation_quality or 0.0)
            quality_value = float(quality or 0.0)
            refs = int(ref_count or 0)
            recent_penalty = min(1.0, max(0.0, (time.time() - float(updated or 0.0)) / (30.0 * 86400.0)))
            learned_delete_score = overlap * 1.8 + recent_penalty * 0.35 - min(1.0, max(-1.0, validation_value)) * 0.75 - min(1.0, max(-1.0, quality_value)) * 0.55 - min(1.0, refs / 128.0) * 0.65 - min(1.0, unique_count / 16.0) * 0.80
            ranked.append((int(champion or 0), -learned_delete_score, validation_value, quality_value, refs, -overlap, unique_count, float(updated or 0.0), str(model_id), row, overlap, unique_count))
        ranked.sort()
        removed = 0
        batch_size = max(1, min(8, int(batch_size or 1)))
        trash_root = self._assert_storage_path(self.models / ".trash", "model_trash")
        trash_root.mkdir(parents=True, exist_ok=True)
        removed_overlap = []
        for index, item in enumerate(ranked):
            champion, _, _, _, _, _, _, _, _, row, overlap, unique_count = item
            model_id, file_name, validation_quality, quality, champion_value, updated, ref_count = row
            if initial - removed <= target or removed >= batch_size:
                break
            if champion or int(champion_value):
                continue
            if cancelled is not None and cancelled():
                break
            if index % 8 == 0 and cooperative is not None and not cooperative():
                break
            path = self._assert_storage_path(self.models / str(file_name), "model_prune")
            trash = self._assert_storage_path(trash_root / (uuid.uuid4().hex + ".json"), "model_prune_trash")
            onnx_path = None
            try:
                payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                onnx_value = payload.get("onnx_path") if isinstance(payload, dict) else None
                if onnx_value:
                    onnx_path = Path(str(onnx_value))
                    if not onnx_path.is_absolute():
                        onnx_path = self.models / onnx_path
                    onnx_path = self._assert_storage_path(onnx_path, "model_prune_onnx")
            except Exception:
                onnx_path = None
            if not path.exists():
                try:
                    with self.lock:
                        self.conn.execute("BEGIN IMMEDIATE")
                        frame_ids = [row[0] for row in self.conn.execute("SELECT DISTINCT frame_id FROM model_frame_refs WHERE model_id=?", (model_id,)).fetchall()]
                        self.conn.execute("DELETE FROM model_frame_refs WHERE model_id=?", (model_id,))
                        self.conn.execute("DELETE FROM model_metadata WHERE id=? AND champion=0", (model_id,))
                        self._recalculate_model_refs_locked(frame_ids)
                        self.conn.commit()
                    removed += 1
                    removed_overlap.append({"model_id": str(model_id), "overlap": float(overlap), "unique_states": int(unique_count), "refs": int(ref_count or 0), "validation_quality": float(validation_quality or 0.0), "missing_file": True})
                    self.add_system_event(None, "model_file_missing_pruned", {"model_id": str(model_id), "file_name": str(file_name), "time": time.time()})
                except Exception as error:
                    with self.lock:
                        try:
                            self.conn.rollback()
                        except Exception:
                            pass
                    self.record_exception_event(None, "model_file_missing_prune_failed", error, {"model_id": str(model_id), "file_name": str(file_name)})
                continue
            try:
                path.replace(trash)
            except OSError as error:
                self.record_exception_event(None, "model_prune_file_move_failed", error, {"model_id": str(model_id), "file_name": str(file_name)})
                continue
            try:
                with self.lock:
                    self.conn.execute("BEGIN IMMEDIATE")
                    frame_ids = [row[0] for row in self.conn.execute("SELECT DISTINCT frame_id FROM model_frame_refs WHERE model_id=?", (model_id,)).fetchall()]
                    self.conn.execute("DELETE FROM model_frame_refs WHERE model_id=?", (model_id,))
                    self.conn.execute("DELETE FROM model_metadata WHERE id=? AND champion=0", (model_id,))
                    self._recalculate_model_refs_locked(frame_ids)
                    self.conn.commit()
                self._safe_unlink_storage(trash, "model_prune_trash_complete")
                if onnx_path is not None and onnx_path.exists():
                    self._safe_unlink_storage(onnx_path, "model_prune_onnx_complete")
                removed += 1
                removed_overlap.append({"model_id": str(model_id), "overlap": float(overlap), "unique_states": int(unique_count), "refs": int(ref_count or 0), "validation_quality": float(validation_quality or 0.0)})
            except Exception:
                with self.lock:
                    try:
                        self.conn.rollback()
                    except Exception:
                        pass
                try:
                    trash.replace(path)
                except OSError:
                    pass
        with self.lock:
            self._sync_model_metadata_locked()
            remaining = int(self.conn.execute("SELECT COUNT(*) FROM model_metadata").fetchone()[0] or 0)
            champions = int(self.conn.execute("SELECT COUNT(*) FROM model_metadata WHERE champion=1").fetchone()[0] or 0)
            self.conn.commit()
        champion_normalized = (champions == 1) if remaining > 0 else (champions == 0)
        success = remaining <= target and champion_normalized
        self.last_model_prune_result = {"initial": initial, "removed": removed, "target": target, "remaining": remaining, "champions": champions, "champion_normalized": champion_normalized, "success": success, "learned_deletion_scorer": "state_cluster_overlap+recent_use+exact_score+model_refs+training_gain_proxy+coverage_loss_guard", "coverage_overlap_used": True, "removed_overlap": removed_overlap[-16:]}
        if not success and not (cancelled is not None and cancelled()):
            self.add_system_event(None, "model_prune_incomplete", dict(self.last_model_prune_result))
        return removed

    def _safe_screen_path(self, stored):
        candidate = (self.pool / stored).resolve()
        base = self.pool.resolve()
        if candidate == base or base not in candidate.parents or not storage_path_allowed(candidate, self.root):
            self._critical_path_event(candidate, "safe_screen_path")
            return None
        return candidate

    def _trash_path(self, journal_id):
        path = self._assert_storage_path(self.pool / "trash" / journal_id, "frame_trash")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _restore_trash(self, journal_id, stored):
        source = self._trash_path(journal_id)
        target = self._safe_screen_path(stored)
        if target is None or not source.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)

    def _delete_frame_batch(self, rows):
        if not rows:
            return 0
        moved = []
        now = time.time()
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            for row in rows:
                identifier, stored, size_bytes = row[:3]
                retain_value = float(row[3] or 0.0) if len(row) > 3 else 0.0
                deletion_reason = str(row[4]) if len(row) > 4 else "retention"
                journal_id = uuid.uuid4().hex
                self.conn.execute("INSERT INTO deletion_journal(id, object_type, object_id, path, stage, created, updated, error) VALUES (?, 'frame', ?, ?, 'pending', ?, ?, '')", (journal_id, identifier, stored, now, now))
                self.conn.execute("INSERT INTO system_events(id, session_id, created, kind, payload) SELECT ?, session_id, ?, 'frame_pruned', ? FROM frames WHERE id=?", (uuid.uuid4().hex, now, json.dumps({"reason": deletion_reason, "retain_value": retain_value, "retain_version": 1}, ensure_ascii=False), identifier))
                self._trim_system_events_locked()
                moved.append((journal_id, identifier, stored, int(size_bytes or 0)))
            self._commit_critical_locked()
        for journal_id, identifier, stored, size_bytes in moved:
            source = self._safe_screen_path(stored)
            trash = self._trash_path(journal_id)
            if source is not None and source.exists():
                trash.parent.mkdir(parents=True, exist_ok=True)
                source.replace(trash)
            with self.lock:
                self.conn.execute("BEGIN IMMEDIATE")
                info = self.conn.execute("SELECT state_cluster_id FROM frames WHERE id=?", (identifier,)).fetchone()
                self.conn.execute("DELETE FROM model_frame_refs WHERE frame_id=?", (identifier,))
                self.conn.execute("DELETE FROM frame_lsh WHERE frame_id=?", (identifier,))
                self.conn.execute("DELETE FROM action_outcomes WHERE before_frame_id=? OR after_frame_id=?", (identifier, identifier))
                self.conn.execute("UPDATE mouse_events SET before_frame_id=NULL WHERE before_frame_id=?", (identifier,))
                self.conn.execute("UPDATE mouse_events SET after_frame_id=NULL WHERE after_frame_id=?", (identifier,))
                self.conn.execute("DELETE FROM frames WHERE id=?", (identifier,))
                if info and info[0]:
                    self.conn.execute("UPDATE state_clusters SET count=MAX(0, count-1), updated_at=? WHERE cluster_id=?", (time.time(), info[0]))
                    self.conn.execute("DELETE FROM state_clusters WHERE cluster_id=? AND count<=0", (info[0],))
                self.conn.execute("UPDATE deletion_journal SET stage='db_deleted', updated=?, error='' WHERE id=?", (time.time(), journal_id))
                self.conn.execute("INSERT OR REPLACE INTO pool_meta(key, value) VALUES ('asset_bytes', MAX(0, COALESCE((SELECT value FROM pool_meta WHERE key='asset_bytes'), 0) - ?))", (size_bytes,))
                self.conn.execute("INSERT OR REPLACE INTO pool_meta(key, value) VALUES ('total_asset_bytes', COALESCE((SELECT value FROM pool_meta WHERE key='asset_bytes'), 0))")
                self._commit_critical_locked()
            self._safe_unlink_storage(trash, "frame_trash_complete")
            with self.lock:
                self.conn.execute("UPDATE deletion_journal SET stage='complete', updated=? WHERE id=?", (time.time(), journal_id))
                self._commit_critical_locked()
        return len(moved)


    def recover_ingestions(self):
        delegated, result = self._dispatch_write_to_writer("recover_ingestions")
        if delegated:
            return result
        if self.conn is None or self.pool is None:
            return
        if self.pool.exists():
            for item in self._iter_storage_files(self.pool, ".tmp", "recover_tmp_walk"):
                try:
                    self._safe_unlink_storage(item, "recover_tmp")
                except OSError:
                    pass
        with self.lock:
            journals = self.conn.execute("SELECT id, object_id, path, stage FROM ingestion_journal WHERE object_type='frame' AND stage!='complete'").fetchall()
        missing_frames = []
        for journal_id, object_id, stored, stage in journals:
            path = self._safe_screen_path(stored) if stored else None
            with self.lock:
                frame_exists = self.conn.execute("SELECT 1 FROM frames WHERE id=?", (object_id,)).fetchone() is not None
            file_exists = bool(path is not None and path.exists())
            if frame_exists and file_exists:
                with self.lock:
                    self.conn.execute("UPDATE ingestion_journal SET stage='complete', updated=?, error='' WHERE id=?", (time.time(), journal_id))
                    self.conn.commit()
                continue
            if file_exists and not frame_exists:
                try:
                    self._safe_unlink_storage(path, "recover_ingestion_orphan")
                except OSError:
                    pass
            if frame_exists and not file_exists:
                missing_frames.append((object_id, stored, 0))
            with self.lock:
                self.conn.execute("UPDATE ingestion_journal SET stage='recovered', updated=?, error=? WHERE id=?", (time.time(), "已释放预留并清理未完成文件", journal_id))
                self.conn.commit()
        if missing_frames:
            self._delete_frame_batch(missing_frames)
        if self.screens and self.screens.exists():
            with self.lock:
                referenced = {str(row[0]) for row in self.conn.execute("SELECT screenshot_path FROM frames").fetchall()}
            for item in self._iter_storage_files(self.screens, ".png", "recover_ingestion_screens_walk"):
                try:
                    rel = str(item.relative_to(self.pool))
                    if rel not in referenced:
                        self._safe_unlink_storage(item, "recover_ingestion_orphan_png")
                except OSError:
                    pass
        self.reconcile_pool_ledger()

    def recover_deletions(self):
        delegated, result = self._dispatch_write_to_writer("recover_deletions")
        if delegated:
            return result
        if self.conn is None or self.pool is None:
            return
        with self.lock:
            journals = self.conn.execute("SELECT id, path, stage FROM deletion_journal WHERE object_type='frame' AND stage!='complete'").fetchall()
        for journal_id, stored, stage in journals:
            try:
                if stage == "pending":
                    self._restore_trash(journal_id, stored)
                    with self.lock:
                        self.conn.execute("UPDATE deletion_journal SET stage='complete', updated=?, error='' WHERE id=?", (time.time(), journal_id))
                        self._commit_critical_locked()
                elif stage == "db_deleted":
                    self._safe_unlink_storage(self._trash_path(journal_id), "recover_deletion_trash")
                    with self.lock:
                        self.conn.execute("UPDATE deletion_journal SET stage='complete', updated=?, error='' WHERE id=?", (time.time(), journal_id))
                        self._commit_critical_locked()
            except OSError as error:
                with self.lock:
                    self.conn.execute("UPDATE deletion_journal SET updated=?, error=? WHERE id=?", (time.time(), str(error), journal_id))
                    self._commit_critical_locked()
        referenced = set()
        with self.lock:
            rows = self.conn.execute("SELECT id, screenshot_path, size_bytes FROM frames").fetchall()
        missing = []
        for identifier, stored, size_bytes in rows:
            path = self._safe_screen_path(stored)
            if path is not None:
                referenced.add(path.resolve())
            if path is None or not path.exists():
                missing.append((identifier, stored, size_bytes))
        if missing:
            self._delete_frame_batch(missing)
        if self.screens and self.screens.exists():
            for item in self._iter_storage_files(self.screens, ".png", "recover_deletion_screens_walk"):
                try:
                    if item.resolve() not in referenced:
                        self._safe_unlink_storage(item, "recover_deletion_orphan_png")
                except OSError:
                    pass
        with self.lock:
            self.conn.execute("DELETE FROM system_events WHERE session_id IS NOT NULL AND session_id NOT IN (SELECT id FROM sessions)")
            self.conn.commit()
            self._reconcile_asset_bytes()

    def validate_consistency(self):
        if self.conn is None or self.pool is None:
            return True, "存储未打开"
        self.sync_model_metadata()
        with self.lock:
            rows = self.conn.execute("SELECT id, screenshot_path FROM frames").fetchall()
            bad_mouse = self.conn.execute("SELECT COUNT(*) FROM mouse_events WHERE session_id NOT IN (SELECT id FROM sessions)").fetchone()[0]
            bad_lsh = self.conn.execute("SELECT COUNT(*) FROM frame_lsh WHERE frame_id NOT IN (SELECT id FROM frames)").fetchone()[0]
            bad_outcomes = self.conn.execute("SELECT COUNT(*) FROM action_outcomes WHERE before_frame_id NOT IN (SELECT id FROM frames) OR after_frame_id NOT IN (SELECT id FROM frames)").fetchone()[0]
            bad_model_refs = self.conn.execute("SELECT COUNT(*) FROM model_frame_refs WHERE frame_id NOT IN (SELECT id FROM frames) OR model_id NOT IN (SELECT id FROM model_metadata)").fetchone()[0]
            bad_counts = self.conn.execute("SELECT COUNT(*) FROM frames WHERE model_dependency_count!=(SELECT COUNT(DISTINCT model_id) FROM model_frame_refs WHERE model_frame_refs.frame_id=frames.id) OR model_refs!=(SELECT COUNT(DISTINCT model_id) FROM model_frame_refs WHERE model_frame_refs.frame_id=frames.id)").fetchone()[0]
        missing = [identifier for identifier, stored in rows if (self._safe_screen_path(stored) is None or not self._safe_screen_path(stored).exists())][:20]
        if missing:
            return False, "数据库引用了缺失截图 {} 条".format(len(missing))
        if bad_lsh or bad_outcomes or bad_model_refs or bad_counts or bad_mouse:
            return False, "索引或引用不一致：LSH {}，动作结果 {}，模型引用 {}，模型计数 {}，鼠标 {}".format(bad_lsh, bad_outcomes, bad_model_refs, bad_counts, bad_mouse)
        return True, "一致"

    def _compact_database(self, cooperative=None):
        started = time.monotonic()
        with self.lock:
            conn = self.conn
            if conn is None:
                self.last_wal_metrics["checkpoint_ms"] = 0.0
                return False
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                freelist = int(conn.execute("PRAGMA freelist_count").fetchone()[0] or 0)
                while freelist > 0:
                    if cooperative is not None and not cooperative():
                        break
                    conn.execute("PRAGMA incremental_vacuum({})".format(min(4096, freelist)))
                    freelist = int(conn.execute("PRAGMA freelist_count").fetchone()[0] or 0)
                conn.commit()
                return True
            except sqlite3.Error as error:
                note_strict_exception("database_compact", error, {"root": str(self.root or "")})
                return False
            finally:
                self.last_wal_metrics["checkpoint_ms"] = (time.monotonic() - started) * 1000.0

    def _hard_compact_database(self, cooperative=None):
        started = time.monotonic()
        with self.lock:
            conn = self.conn
            if conn is None:
                self.last_wal_metrics["checkpoint_ms"] = 0.0
                return False
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.commit()
                if cooperative is None or cooperative():
                    conn.execute("VACUUM")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.commit()
                return True
            except sqlite3.Error as error:
                try:
                    self.add_system_event(None, "database_hard_compact_failed", {"error": str(error)})
                except Exception:
                    note_strict_exception("database_hard_compact_failed", error, {"root": str(self.root or "")})
                return False
            finally:
                self.last_wal_metrics["checkpoint_ms"] = (time.monotonic() - started) * 1000.0

    def _release_model_frame_refs(self, champion_only):
        with self.lock:
            if self.conn is None:
                return 0
            champion_rows = self.conn.execute("SELECT id FROM model_metadata WHERE champion=1").fetchall()
            champions = {str(row[0]) for row in champion_rows}
            if champion_only:
                if not champions:
                    return 0
                marks = ",".join("?" for _ in champions)
                frame_rows = self.conn.execute("SELECT DISTINCT frame_id FROM model_frame_refs WHERE model_id IN ({})".format(marks), tuple(champions)).fetchall()
                self.conn.execute("DELETE FROM model_frame_refs WHERE model_id IN ({})".format(marks), tuple(champions))
            else:
                if champions:
                    marks = ",".join("?" for _ in champions)
                    frame_rows = self.conn.execute("SELECT DISTINCT frame_id FROM model_frame_refs WHERE model_id NOT IN ({})".format(marks), tuple(champions)).fetchall()
                    self.conn.execute("DELETE FROM model_frame_refs WHERE model_id NOT IN ({})".format(marks), tuple(champions))
                else:
                    frame_rows = self.conn.execute("SELECT DISTINCT frame_id FROM model_frame_refs").fetchall()
                    self.conn.execute("DELETE FROM model_frame_refs")
            self._recalculate_model_refs_locked([row[0] for row in frame_rows])
            self.conn.commit()
            return len(frame_rows)

    def _asset_bytes_locked(self):
        row = self.conn.execute("SELECT value FROM pool_meta WHERE key='asset_bytes'").fetchone()
        return max(0, int(row[0] or 0)) if row else 0

    def _prune_metadata_before_assets(self, cooperative=None):
        with self.lock:
            if self.conn is None:
                return
            cutoff = time.time() - 30 * 86400
            self.conn.execute("DELETE FROM system_events WHERE created<? AND kind NOT IN ('pool_prune_incomplete_after_reference_release','client_validation_failure','capacity_rejection')", (cutoff,))
            self.conn.execute("DELETE FROM ingestion_journal WHERE stage='complete' AND updated<?", (cutoff,))
            self.conn.execute("DELETE FROM deletion_journal WHERE stage='complete' AND updated<?", (cutoff,))
            self.conn.commit()
        self._compact_database(cooperative)

    def _dynamic_prune_target(self, maximum, initial=None):
        maximum = max(1, int(maximum))
        disk_free = 0
        try:
            root_path = self.root if self.root is not None else Path.cwd()
            disk_free = int(shutil.disk_usage(root_path).free)
        except Exception as error:
            note_strict_exception("pool_prune_disk_probe", error, {"root": str(self.root or "")})
        wal_bytes = int((self.last_wal_metrics or {}).get("wal_bytes", 0) or 0)
        recent_rate = 0.0
        cluster_gap = False
        try:
            if self.conn is not None:
                now_value = time.time()
                recent_count = self.conn.execute("SELECT COUNT(*) FROM frames WHERE created>=?", (now_value - 300.0,)).fetchone()[0]
                recent_rate = float(recent_count) / 300.0
                thin_clusters = self.conn.execute("SELECT COUNT(*) FROM (SELECT state_cluster_id, COUNT(*) AS c FROM frames WHERE state_cluster_id IS NOT NULL AND state_cluster_id!='' GROUP BY state_cluster_id HAVING c<3)").fetchone()[0]
                cluster_gap = int(thin_clusters or 0) > 0
        except Exception as error:
            note_strict_exception("pool_prune_target_probe", error, {})
        if disk_free and (disk_free < 2 * 1024 * 1024 * 1024 or wal_bytes > 768 * 1024 * 1024):
            ratio = 0.45 if disk_free < 1024 * 1024 * 1024 or wal_bytes > 1024 * 1024 * 1024 else 0.50
            tier = "tight"
        elif disk_free > max(20 * 1024 * 1024 * 1024, maximum * 3):
            ratio = 0.70
            tier = "high_free"
        else:
            ratio = 0.60
            tier = "normal"
        if tier != "tight" and (recent_rate > 0.5 or cluster_gap):
            ratio = min(0.72, ratio + 0.05)
        target = int(math.floor(maximum * ratio))
        target = max(1, min(maximum - 1 if maximum > 1 else 1, target))
        return target, {"ratio": ratio, "tier": tier, "disk_free": disk_free, "wal_bytes": wal_bytes, "recent_rate": recent_rate, "cluster_gap": cluster_gap, "initial": int(initial or 0)}

    def prune_experience(self, maximum, cancelled, progress, cooperative=None, batch_size=8):
        maximum = max(1, int(maximum))
        target_total, dynamic_target_info = self._dynamic_prune_target(maximum)
        batch_size = max(1, min(8, int(batch_size or 1)))
        self.recover_deletions()
        self.sync_model_metadata()
        self._prune_metadata_before_assets(cooperative)
        with self.lock:
            self._compact_database(cooperative)
        initial_assets = self._asset_bytes_locked()
        with self.lock:
            initial_clusters = {str(row[0]) for row in self.conn.execute("SELECT DISTINCT state_cluster_id FROM frames WHERE state_cluster_id IS NOT NULL AND state_cluster_id!=''").fetchall()}
        initial_metadata = self._database_bytes_now()
        initial = self.pool_size()
        asset_target = max(0, target_total - initial_metadata)
        if initial_metadata > target_total:
            self._hard_compact_database(cooperative)
            initial_metadata = self._database_bytes_now()
            initial = self.pool_size()
            asset_target = max(0, target_total - initial_metadata)
            self.add_system_event(None, "pool_prune_database_hard_compact", {"target": target_total, "metadata_bytes": initial_metadata, "remaining": initial, "asset_target": asset_target})
        if initial <= maximum:
            self._set_capacity_status(False, target_total, initial)
            self.last_experience_prune_result = {"initial": initial, "removed": 0, "target": target_total, "asset_target": asset_target, "remaining": initial, "metadata_bytes": initial_metadata, "dynamic_target": dict(dynamic_target_info), "success": initial <= maximum}
            return 0, initial
        removed = 0
        current_assets = initial_assets

        def delete_until(predicate, params, progress_start, progress_end):
            nonlocal current_assets, removed
            while current_assets > asset_target and not cancelled():
                if cooperative is not None and not cooperative():
                    time.sleep(0.25)
                    continue
                with self.lock:
                    rows = self.conn.execute("""SELECT frames.id, frames.screenshot_path, frames.size_bytes, frames.retain_value,
                        CASE WHEN COALESCE(state_clusters.count, frames.state_support_count, 1)>1 THEN 'redundant_cluster' ELSE 'minimum_cluster_guard' END
                        FROM frames LEFT JOIN state_clusters ON state_clusters.cluster_id=frames.state_cluster_id
                        WHERE {} ORDER BY
                        (frames.retain_value * 0.45 + frames.model_dependency_count * 0.65 + frames.model_refs * 0.55 + frames.validation_last_used / 10000000000.0 + CASE WHEN frames.score_status='exact' THEN 0.20 ELSE 0.0 END - CASE WHEN COALESCE(state_clusters.count, frames.state_support_count, 1)>1 THEN 0.60 ELSE 0.0 END + CASE WHEN frames.score_valid=1 THEN 0.15 ELSE 0.0 END) ASC,
                        frames.created ASC
                        LIMIT ?""".format(predicate), tuple(params()) + (batch_size,)).fetchall()
                if not rows:
                    break
                removed += self._delete_frame_batch(rows)
                with self.lock:
                    current_assets = self._asset_bytes_locked()
                    remaining_clusters = {str(row[0]) for row in self.conn.execute("SELECT DISTINCT state_cluster_id FROM frames WHERE state_cluster_id IS NOT NULL AND state_cluster_id!=''").fetchall()}
                loss = max(0, len(initial_clusters) - len(remaining_clusters))
                ratio = loss / max(1, len(initial_clusters))
                self.last_prune_coverage_loss = {"before": len(initial_clusters), "after": len(remaining_clusters), "loss": loss, "ratio": ratio, "paused": False, "priority_only": True}
                if ratio > 0.10:
                    self.add_system_event(None, "pool_prune_coverage_priority_only", dict(self.last_prune_coverage_loss))
                span = max(0.0, progress_end - progress_start)
                progress(min(progress_end, progress_start + span * min(1.0, max(0.0, initial_assets - current_assets) / max(1, initial_assets - asset_target))))
            return current_assets <= asset_target

        def delete_stratified(progress_start, progress_end):
            nonlocal current_assets, removed
            while current_assets > asset_target and not cancelled():
                if cooperative is not None and not cooperative():
                    time.sleep(0.25)
                    continue
                now_value = time.time()
                with self.lock:
                    cluster_rows = self.conn.execute("""
                        SELECT frames.state_cluster_id, COUNT(*), MAX(frames.model_dependency_count + frames.model_refs), MAX(frames.retain_value), MAX(frames.validation_last_used)
                        FROM frames
                        WHERE frames.state_cluster_id IS NOT NULL AND frames.state_cluster_id!=''
                        GROUP BY frames.state_cluster_id
                    """).fetchall()
                    floors = {}
                    counts = {}
                    for cluster_id, count, refs, retain, last_used in cluster_rows:
                        count = int(count or 0)
                        floor = 1
                        floors[str(cluster_id)] = max(1, min(max(1, count), floor))
                        counts[str(cluster_id)] = count
                    candidates = self.conn.execute("""SELECT frames.id, frames.screenshot_path, frames.size_bytes, frames.retain_value,
                        'stratified_cluster_guard', frames.state_cluster_id
                        FROM frames LEFT JOIN state_clusters ON state_clusters.cluster_id=frames.state_cluster_id
                        WHERE frames.state_cluster_id IS NOT NULL AND frames.state_cluster_id!=''
                        ORDER BY (frames.retain_value * 0.45 + frames.model_dependency_count * 0.65 + frames.model_refs * 0.55 + frames.asset_ref_count * 0.30 + frames.validation_last_used / 10000000000.0 + CASE WHEN frames.score_status='exact' THEN 0.20 ELSE 0.0 END - CASE WHEN COALESCE(state_clusters.count, frames.state_support_count, 1)>1 THEN 0.60 ELSE 0.0 END) ASC, frames.created ASC
                        LIMIT ?""", (max(8, batch_size * 16),)).fetchall()
                rows = []
                for row in candidates:
                    cluster_id = str(row[5])
                    if counts.get(cluster_id, 0) > floors.get(cluster_id, 1):
                        rows.append(row[:5])
                        counts[cluster_id] = counts.get(cluster_id, 0) - 1
                    if len(rows) >= batch_size:
                        break
                if not rows:
                    break
                removed += self._delete_frame_batch(rows)
                with self.lock:
                    current_assets = self._asset_bytes_locked()
                    remaining_clusters = {str(row[0]) for row in self.conn.execute("SELECT DISTINCT state_cluster_id FROM frames WHERE state_cluster_id IS NOT NULL AND state_cluster_id!=''").fetchall()}
                loss = max(0, len(initial_clusters) - len(remaining_clusters))
                ratio = loss / max(1, len(initial_clusters))
                self.last_prune_coverage_loss = {"before": len(initial_clusters), "after": len(remaining_clusters), "loss": loss, "ratio": ratio, "paused": False, "fallback": "stratified_cluster_guard", "cluster_floor_max": max(floors.values() or [0])}
                span = max(0.0, progress_end - progress_start)
                progress(min(progress_end, progress_start + span * min(1.0, max(0.0, initial_assets - current_assets) / max(1, initial_assets - asset_target))))
            return current_assets <= asset_target

        delete_until(
            "frames.model_dependency_count=0 AND frames.model_refs=0 AND frames.asset_ref_count<=1 AND COALESCE(state_clusters.count, frames.state_support_count, 1)>1 AND frames.validation_last_used<?",
            lambda: (time.time() - 3600.0,),
            56.0,
            70.0,
        )
        if current_assets > asset_target and not cancelled():
            delete_until("frames.model_dependency_count=0 AND frames.model_refs=0 AND COALESCE(state_clusters.count, frames.state_support_count, 1)>1 AND frames.retain_value<1.35", lambda: (), 70.0, 82.0)
        if current_assets > asset_target and not cancelled():
            self._release_model_frame_refs(False)
            delete_until("frames.model_dependency_count=0 AND frames.model_refs=0 AND COALESCE(state_clusters.count, frames.state_support_count, 1)>1", lambda: (), 82.0, 90.0)
        if current_assets > asset_target and not cancelled():
            self._release_model_frame_refs(True)
            delete_until("COALESCE(state_clusters.count, frames.state_support_count, 1)>1", lambda: (), 90.0, 95.0)
        if current_assets > asset_target and not cancelled():
            delete_stratified(95.0, 96.0)
        self._cleanup_pool_files()
        self._prune_metadata_before_assets(cooperative)
        with self.lock:
            current_assets = self._asset_bytes_locked()
        self._compact_database(cooperative)
        metadata = self._database_bytes_now()
        remaining = self.pool_size()
        if remaining > target_total and metadata > int(target_total * 0.45):
            self._hard_compact_database(cooperative)
            metadata = self._database_bytes_now()
            remaining = self.pool_size()
        success = remaining <= target_total
        self.last_experience_prune_result = {"initial": initial, "removed": removed, "target": target_total, "asset_target": max(0, target_total - metadata), "remaining": remaining, "asset_bytes": current_assets, "metadata_bytes": metadata, "dynamic_target": dict(dynamic_target_info), "coverage_loss": dict(self.last_prune_coverage_loss), "learned_deletion_scorer": "state_cluster_coverage+recent_use+exact_score+model_refs+training_gain_proxy+delete_after_coverage_loss", "success": success}
        if success:
            self._set_capacity_status(False, target_total, remaining)
        elif not cancelled():
            self._set_capacity_status(True, target_total, remaining)
            self.add_system_event(None, "pool_prune_incomplete_after_reference_release", dict(self.last_experience_prune_result))
        progress(96.0)
        return removed, remaining


def filetime_value(value):
    return (value.dwHighDateTime << 32) | value.dwLowDateTime

def process_snapshot():
    result = {}
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == INVALID_HANDLE_VALUE:
        return result
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        success = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while success:
            result[int(entry.th32ProcessID)] = (int(entry.th32ParentProcessID), str(entry.szExeFile).lower())
            success = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return result

def process_tree(seed_pids):
    seeds = {int(pid) for pid in (seed_pids or set()) if int(pid) > 0}
    table = process_snapshot()
    children = {}
    for pid, (parent, name) in table.items():
        children.setdefault(parent, set()).add(pid)
    result = set(seeds)
    pending = list(seeds)
    while pending:
        pid = pending.pop()
        for child in children.get(pid, ()):
            if child not in result:
                result.add(child)
                pending.append(child)
    return result

def processes_for_name(name):
    wanted = name.lower()
    return {pid for pid, (_, exe) in process_snapshot().items() if exe == wanted}

def process_full_path(pid):
    if not IS_WINDOWS:
        try:
            return os.readlink("/proc/{}/exe".format(int(pid)))
        except Exception:
            return ""
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return buffer.value
    finally:
        kernel32.CloseHandle(handle)
    return ""

def client_rect(hwnd):
    if not IS_WINDOWS:
        return PLATFORM_BACKEND.client_rect(hwnd)
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    first = POINT(rect.left, rect.top)
    second = POINT(rect.right, rect.bottom)
    if not user32.ClientToScreen(hwnd, ctypes.byref(first)):
        return None
    if not user32.ClientToScreen(hwnd, ctypes.byref(second)):
        return None
    if second.x <= first.x or second.y <= first.y:
        return None
    return (first.x, first.y, second.x, second.y)

def point_inside(point, rect):
    return rect[0] <= point[0] < rect[2] and rect[1] <= point[1] < rect[3]

def cursor_position():
    if not IS_WINDOWS:
        return PLATFORM_BACKEND.cursor_position() if hasattr(PLATFORM_BACKEND, "cursor_position") else (0, 0)
    point = POINT()
    if user32.GetCursorPos(ctypes.byref(point)):
        return (int(point.x), int(point.y))
    return (0, 0)

def send_ai_mouse(x, y, flags, marker, absolute=True):
    if not IS_WINDOWS:
        return PLATFORM_BACKEND.inject_mouse({"action": "move", "x": int(x), "y": int(y), "flags": int(flags), "marker": int(marker or 0), "absolute": bool(absolute)})
    with AI_INPUT_SERIAL_LOCK:
        event = INPUT()
        event.type = INPUT_MOUSE
        if absolute:
            screen = virtual_screen_rect()
            width = max(1, screen[2] - screen[0] - 1)
            height = max(1, screen[3] - screen[1] - 1)
            dx = int(max(0, min(65535, round((x - screen[0]) * 65535 / width))))
            dy = int(max(0, min(65535, round((y - screen[1]) * 65535 / height))))
            flags |= MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        else:
            dx = 0
            dy = 0
        event.mi = MOUSEINPUT(dx, dy, 0, flags, 0, ULONG_PTR(int(marker)))
        return int(user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))) == 1

def ai_move_to(x, y, marker):
    return send_ai_mouse(x, y, MOUSEEVENTF_MOVE, marker, True)

def ai_left_click(down_marker, up_marker):
    if not IS_WINDOWS:
        return PLATFORM_BACKEND.inject_mouse({"action": "left_click", "marker": int(up_marker or down_marker or 0)})
    with AI_INPUT_SERIAL_LOCK:
        down = send_ai_mouse(0, 0, MOUSEEVENTF_LEFTDOWN, down_marker, False)
        up = send_ai_mouse(0, 0, MOUSEEVENTF_LEFTUP, up_marker, False)
        return down and up

def ai_right_click(down_marker, up_marker):
    if not IS_WINDOWS:
        return PLATFORM_BACKEND.inject_mouse({"action": "right_click", "marker": int(up_marker or down_marker or 0)})
    with AI_INPUT_SERIAL_LOCK:
        down = send_ai_mouse(0, 0, MOUSEEVENTF_RIGHTDOWN, down_marker, False)
        up = send_ai_mouse(0, 0, MOUSEEVENTF_RIGHTUP, up_marker, False)
        return down and up

def ai_wheel(delta, marker, horizontal=False):
    if not IS_WINDOWS:
        action = "wheel_horizontal" if horizontal else "wheel"
        return PLATFORM_BACKEND.inject_mouse({"action": action, "delta": int(delta), "marker": int(marker or 0)})
    with AI_INPUT_SERIAL_LOCK:
        event = INPUT()
        event.type = INPUT_MOUSE
        event.mi = MOUSEINPUT(0, 0, int(delta), MOUSEEVENTF_HWHEEL if horizontal else MOUSEEVENTF_WHEEL, 0, ULONG_PTR(int(marker)))
        return int(user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))) == 1

def virtual_screen_rect():
    left = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    top = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    return (left, top, left + user32.GetSystemMetrics(SM_CXVIRTUALSCREEN), top + user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))

def physical_monitor_coverage(rect):
    target = RECT(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    target_area = max(1, (target.right - target.left) * (target.bottom - target.top))
    covered = 0
    monitors = []
    def callback(hmonitor, hdc, monitor_rect, lparam):
        current = monitor_rect.contents
        left = max(target.left, current.left); top = max(target.top, current.top)
        right = min(target.right, current.right); bottom = min(target.bottom, current.bottom)
        if right > left and bottom > top:
            monitors.append((left, top, right, bottom))
        return True
    user32.EnumDisplayMonitors(0, ctypes.byref(target), MonitorEnumProc(callback), 0)
    for left, top, right, bottom in monitors:
        covered += (right - left) * (bottom - top)
    return covered / target_area, monitors

def screen_contains(rect):
    coverage, monitors = physical_monitor_coverage(rect)
    return coverage >= 0.999 and bool(monitors)

def root_window(hwnd):
    if not IS_WINDOWS:
        return int(hwnd or 0)
    return user32.GetAncestor(hwnd, GA_ROOT)

def rectangle_overlap(first, second):
    return first[0] < second[2] and second[0] < first[2] and first[1] < second[3] and second[1] < first[3]

def rectangle_area(rect):
    if rect is None:
        return 0
    return max(0, int(rect[2]) - int(rect[0])) * max(0, int(rect[3]) - int(rect[1]))

def rectangle_overlap_area(first, second):
    if first is None or second is None:
        return 0
    left = max(int(first[0]), int(second[0]))
    top = max(int(first[1]), int(second[1]))
    right = min(int(first[2]), int(second[2]))
    bottom = min(int(first[3]), int(second[3]))
    return max(0, right - left) * max(0, bottom - top)

def substantive_overlap(first, second):
    area = rectangle_overlap_area(first, second)
    return area >= max(64, min(4096, int(rectangle_area(second) * 0.005)))

def inset_rectangle(rect, pixels):
    left, top, right, bottom = (int(value) for value in rect)
    pixels = max(0, min(int(pixels), max(0, right - left - 1) // 2, max(0, bottom - top - 1) // 2))
    return (left + pixels, top + pixels, right - pixels, bottom - pixels)

def rectangle_grid_points(rect, count=5, inset=3):
    inner = inset_rectangle(rect, inset)
    left, top, right, bottom = inner
    count = max(3, min(5, int(count)))
    width = max(1, right - left - 1)
    height = max(1, bottom - top - 1)
    points = []
    for y_index in range(count):
        y = top + int(round(height * y_index / max(1, count - 1)))
        for x_index in range(count):
            x = left + int(round(width * x_index / max(1, count - 1)))
            points.append((x, y))
    return points

def window_rectangle(hwnd):
    if not IS_WINDOWS:
        return PLATFORM_BACKEND.client_rect(hwnd)
    rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    if rect.right <= rect.left or rect.bottom <= rect.top:
        return None
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))

def window_is_transparent(hwnd):
    try:
        return bool(int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE)) & WS_EX_TRANSPARENT)
    except Exception:
        return False

def window_is_toolwindow(hwnd):
    try:
        return bool(int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE)) & WS_EX_TOOLWINDOW)
    except Exception:
        return False

def window_is_cloaked(hwnd):
    if dwmapi is None:
        return False
    value = wintypes.DWORD()
    try:
        return dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(value), ctypes.sizeof(value)) == 0 and bool(value.value)
    except Exception:
        return False

def window_obstruction_kind(hwnd):
    if window_is_transparent(hwnd):
        return "transparent_overlay"
    if window_is_cloaked(hwnd):
        return "cloaked_overlay"
    if window_is_toolwindow(hwnd):
        return "toolwindow_overlay"
    return "window_overlap"

def register_own_overlay_window(hwnd, enabled=True):
    try:
        value = int(hwnd)
    except Exception:
        return False
    with OWN_OVERLAY_LOCK:
        if enabled:
            OWN_TRANSPARENT_OVERLAY_WINDOWS.add(value)
        else:
            OWN_TRANSPARENT_OVERLAY_WINDOWS.discard(value)
    return True

def own_registered_transparent_overlay(hwnd):
    try:
        value = int(hwnd)
        root = int(root_window(hwnd) or 0)
    except Exception:
        return False
    with OWN_OVERLAY_LOCK:
        registered = value in OWN_TRANSPARENT_OVERLAY_WINDOWS or root in OWN_TRANSPARENT_OVERLAY_WINDOWS
    if not registered or not window_is_transparent(hwnd):
        return False
    pid = wintypes.DWORD()
    try:
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value) == os.getpid()
    except Exception:
        return False

def ignored_overlay_window(hwnd):
    return own_registered_transparent_overlay(hwnd)

def client_unobscured(hwnd, rect):
    own_root = root_window(hwnd)
    if not own_root:
        return False
    client_unobscured.last_obstruction = None
    client_unobscured.last_overlay = None
    for x, y in rectangle_grid_points(rect, 5, 3):
        hit = WindowFromPoint(POINT(int(x), int(y)))
        hit_root = root_window(hit) if hit else 0
        if hit and hit_root != own_root:
            target = hit_root or hit
            pid = wintypes.DWORD(); user32.GetWindowThreadProcessId(target, ctypes.byref(pid))
            details = {"kind": window_obstruction_kind(target), "title": window_title(target), "pid": int(pid.value), "point": (int(x), int(y)), "rect": window_rectangle(target)}
            if ignored_overlay_window(target):
                client_unobscured.last_overlay = details
                continue
            client_unobscured.last_obstruction = details
            return False
    above = user32.GetWindow(own_root, GW_HWNDPREV)
    checked = set()
    while above and above not in checked:
        checked.add(above)
        if root_window(above) != own_root and user32.IsWindowVisible(above) and not user32.IsIconic(above):
            candidate = window_rectangle(above)
            if candidate is not None and substantive_overlap(candidate, rect):
                pid = wintypes.DWORD(); user32.GetWindowThreadProcessId(above, ctypes.byref(pid))
                details = {"kind": window_obstruction_kind(above), "title": window_title(above), "pid": int(pid.value), "rect": candidate, "overlap_area": rectangle_overlap_area(candidate, rect)}
                if ignored_overlay_window(above):
                    client_unobscured.last_overlay = details
                else:
                    client_unobscured.last_obstruction = details
                    return False
        above = user32.GetWindow(above, GW_HWNDPREV)
    return True

def valid_client(hwnd, require_cursor=True):
    valid_client.last_reason = ""
    if not IS_WINDOWS:
        if not bool(getattr(PLATFORM_BACKEND, "learning_training_enabled", False)):
            valid_client.last_reason = PLATFORM_BACKEND.mode_unavailable_reason()
            return None
        rect = PLATFORM_BACKEND.client_rect(hwnd)
        if rect is None or rect[2] - rect[0] < 96 or rect[3] - rect[1] < 96:
            valid_client.last_reason = "客户区尺寸异常"
            return None
        if require_cursor and not point_inside(cursor_position(), rect):
            valid_client.last_reason = "鼠标不在客户区内"
            return None
        return rect
    if not hwnd or not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
        valid_client.last_reason = "窗口不可见或最小化"
        return None
    if window_is_cloaked(hwnd):
        valid_client.last_reason = "DWM 标记窗口不可见"
        return None
    rect = client_rect(hwnd)
    if rect is None or rect[2] - rect[0] < 96 or rect[3] - rect[1] < 96:
        valid_client.last_reason = "客户区尺寸异常"
        return None
    coverage, _ = physical_monitor_coverage(rect)
    if coverage < 0.999:
        valid_client.last_reason = "窗口跨屏不完整，物理显示器覆盖率 {:.1%}".format(coverage)
        return None
    if not client_unobscured(hwnd, rect):
        details = getattr(client_unobscured, "last_obstruction", {}) or {}
        valid_client.last_reason = "客户区被{}遮挡".format(details.get("kind", "窗口"))
        return None
    if require_cursor and not point_inside(cursor_position(), rect):
        valid_client.last_reason = "鼠标不在客户区内"
        return None
    return rect

def foreground_root_matches(hwnd):
    if not IS_WINDOWS:
        return bool(hwnd)
    expected = root_window(hwnd)
    foreground = root_window(user32.GetForegroundWindow())
    return bool(expected and foreground and expected == foreground)

def activate_root_window(hwnd, timeout=0.80):
    if not IS_WINDOWS:
        ok = PLATFORM_BACKEND.activate_window(hwnd)
        activate_root_window.last_reason = "" if ok else PLATFORM_BACKEND.mode_unavailable_reason()
        return bool(ok)
    root = root_window(hwnd)
    if not root or not user32.IsWindow(root):
        activate_root_window.last_reason = "窗口句柄无效"
        return False
    try:
        user32.ShowWindow(root, SW_RESTORE)
        user32.SetForegroundWindow(root)
    except Exception as error:
        activate_root_window.last_reason = str(error)
        return False
    deadline = time.monotonic() + max(0.05, float(timeout))
    while time.monotonic() < deadline:
        if foreground_root_matches(root):
            activate_root_window.last_reason = ""
            return True
        time.sleep(0.02)
    activate_root_window.last_reason = "前台窗口不是绑定的目标根窗口"
    return False

def window_title(hwnd):
    if not IS_WINDOWS:
        for item in PLATFORM_BACKEND.list_windows():
            if int(item.get("hwnd", 0) or 0) == int(hwnd or 0):
                return str(item.get("title") or "")
        return ""
    length = max(0, int(user32.GetWindowTextLengthW(hwnd)))
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value

def window_class_name(hwnd):
    buffer = ctypes.create_unicode_buffer(256)
    try:
        user32.GetClassNameW(hwnd, buffer, len(buffer))
        return buffer.value
    except Exception:
        return ""

def is_ld_window_candidate(path, title):
    exe = Path(path).name.lower() if path else ""
    title_key = str(title or "").lower()
    return exe in {"dnplayer.exe", "ldplayer.exe", "ldplayer9.exe"} or "雷电" in str(title or "") or "ldplayer" in title_key

def selectable_top_level_window(hwnd):
    if not hwnd or not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
        return False
    if window_is_toolwindow(hwnd) or window_is_transparent(hwnd) or window_is_cloaked(hwnd):
        return False
    cls = window_class_name(hwnd)
    if cls in {"Shell_TrayWnd", "Shell_SecondaryTrayWnd", "Progman", "WorkerW", "DV2ControlHost", "TaskListThumbnailWnd", "Windows.UI.Core.CoreWindow", "ApplicationFrameInputSinkWindow"}:
        return False
    pid = wintypes.DWORD(); user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if int(pid.value or 0) == os.getpid():
        return False
    title = window_title(hwnd)
    rect = client_rect(hwnd)
    if rect is None or rect[2] - rect[0] < 96 or rect[3] - rect[1] < 96:
        return False
    return bool(title.strip() or Path(process_full_path(int(pid.value))).name)

def normalized_windows_path(value):
    try:
        return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expandvars(os.path.expanduser(str(value))))))
    except Exception:
        return os.path.normcase(str(value))

def find_emulator_window_candidates(configured_path=""):
    if not IS_WINDOWS:
        selected = os.path.realpath(os.path.expanduser(str(configured_path))) if configured_path else ""
        rows = PLATFORM_BACKEND.list_windows()
        if selected:
            matched = [item for item in rows if os.path.realpath(str(item.get("path", ""))) == selected]
            if matched:
                return matched
        return rows
    selected = normalized_windows_path(configured_path) if configured_path else ""
    candidates = []
    def callback(hwnd, _):
        if not selectable_top_level_window(hwnd):
            return True
        pid = wintypes.DWORD(); user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return True
        path = process_full_path(int(pid.value))
        normalized = normalized_windows_path(path)
        title = window_title(hwnd)
        rect = client_rect(hwnd)
        ld_rank = 0 if is_ld_window_candidate(path, title) else 1
        selected_rank = 0 if selected and normalized == selected else 1
        candidates.append({"hwnd": hwnd, "pid": int(pid.value), "title": title, "path": path, "rect": rect, "area": (rect[2] - rect[0]) * (rect[3] - rect[1]), "class": window_class_name(hwnd), "ld_rank": ld_rank, "selected_rank": selected_rank})
        return True
    user32.EnumWindows(EnumWindowsProc(callback), 0)
    return sorted(candidates, key=lambda item: (item["ld_rank"], item["selected_rank"], item["title"].lower(), item["pid"], -item["area"]))

def find_emulator_candidates(executable):
    selected = normalized_windows_path(executable) if executable else ""
    rows = find_emulator_window_candidates(executable)
    if selected:
        matched = [item for item in rows if normalized_windows_path(item.get("path", "")) == selected]
        if matched:
            return matched
    return rows

def find_emulator_window(executable, selected_pid=None, selected_title=None):
    candidates = find_emulator_candidates(executable)
    if selected_pid:
        candidates = [item for item in candidates if int(item["pid"]) == int(selected_pid)]
    if selected_title:
        candidates = [item for item in candidates if item["title"] == selected_title]
    if len(candidates) != 1:
        return None
    return candidates[0]["hwnd"]

def png_chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

def encode_png(width, height, rgb, compression_level=6):
    rows = bytearray()
    row_size = width * 3
    for offset in range(0, len(rgb), row_size):
        rows.append(0)
        rows.extend(rgb[offset:offset + row_size])
    level = max(1, min(9, int(compression_level)))
    compressor = zlib.compressobj(level)
    payload = compressor.compress(bytes(rows)) + compressor.flush()
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + png_chunk(b"IDAT", payload) + png_chunk(b"IEND", b"")

def encode_png_bgra(width, height, bgra, compression_level=6):
    rows = bytearray()
    stride = width * 4
    for offset in range(0, len(bgra), stride):
        rows.append(0)
        row = bgra[offset:offset + stride]
        for index in range(0, len(row), 4):
            rows.extend((row[index + 2], row[index + 1], row[index]))
    level = max(1, min(9, int(compression_level)))
    compressor = zlib.compressobj(level)
    payload = compressor.compress(bytes(rows)) + compressor.flush()
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + png_chunk(b"IDAT", payload) + png_chunk(b"IEND", b"")

def _pb_varint(value):
    value = int(value)
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)

def _pb_key(field, wire):
    return _pb_varint((int(field) << 3) | int(wire))

def _pb_int(field, value):
    return _pb_key(field, 0) + _pb_varint(value)

def _pb_string(field, value):
    data = str(value).encode("utf-8")
    return _pb_key(field, 2) + _pb_varint(len(data)) + data

def _pb_data(field, payload):
    payload = bytes(payload)
    return _pb_key(field, 2) + _pb_varint(len(payload)) + payload

def _pb_message(field, payload):
    payload = bytes(payload)
    return _pb_key(field, 2) + _pb_varint(len(payload)) + payload

def _onnx_shape(dims):
    return b"".join(_pb_message(1, _pb_int(1, dim)) for dim in dims)

def _onnx_value_info(name, dims):
    tensor = _pb_int(1, 1) + _pb_message(2, _onnx_shape(dims))
    return _pb_string(1, name) + _pb_message(2, _pb_message(1, tensor))

def _onnx_tensor(name, dims, values):
    raw = struct.pack("<{}f".format(len(values)), *[float(value) for value in values])
    return b"".join(_pb_int(1, dim) for dim in dims) + _pb_int(2, 1) + _pb_string(8, name) + _pb_data(9, raw)

def _onnx_node(op_type, inputs, outputs, name):
    payload = b"".join(_pb_string(1, item) for item in inputs)
    payload += b"".join(_pb_string(2, item) for item in outputs)
    payload += _pb_string(3, name)
    payload += _pb_string(4, op_type)
    return payload

def _logit(value):
    value = max(0.001, min(0.999, float(value)))
    return math.log(value / (1.0 - value))

def _clamp_float(value, low, high):
    try:
        value = float(value)
    except Exception:
        value = low
    return max(float(low), min(float(high), value))

def _sigmoid_float(value):
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))

def _gray32_values(value):
    data = feature_bytes(value, 32 * 18)
    if not data:
        return []
    return [item / 255.0 for item in data]

def _gray_patch(values, rx, ry):
    if not values:
        return []
    cx = max(1, min(30, int(round(float(rx) * 31))))
    cy = max(1, min(16, int(round(float(ry) * 17))))
    patch = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            patch.append(values[(cy + dy) * 32 + (cx + dx)])
    return patch

def visual_policy_input_vector(features):
    features = features or {}
    gray = feature_bytes(features.get("gray32x18"), POLICY_GRAY_SIZE)
    if not gray:
        gray = bytes([128] * POLICY_GRAY_SIZE)
    vector = [max(0.0, min(1.0, item / 255.0)) for item in gray]
    histogram = histogram_values(features.get("color_histogram"))
    if len(histogram) != 24 or sum(histogram) <= 0:
        vector.extend([0.0] * 24)
    else:
        total = float(sum(histogram))
        vector.extend([max(0.0, min(1.0, float(item) / total)) for item in histogram[:24]])
    vector.append(_clamp_float(features.get("edge_density", 0.0), 0.0, 1.0))
    recent = features.get("recent_action_summary") or {}
    vector.extend([_clamp_float(recent.get("move_rate", 0.0), 0.0, 1.0), _clamp_float(recent.get("button_rate", 0.0), 0.0, 1.0), _clamp_float(recent.get("wheel_rate", 0.0), 0.0, 1.0), _clamp_float(recent.get("mean_score_delta", 0.0), -1.0, 1.0)])
    if len(vector) < POLICY_INPUT_SIZE:
        vector.extend([0.0] * (POLICY_INPUT_SIZE - len(vector)))
    return vector[:POLICY_INPUT_SIZE]

def visual_policy_target_vector(example):
    output = [0.0] * POLICY_OUTPUT_SIZE
    action_type = str((example or {}).get("action_type", "移动"))
    try:
        action_index = POLICY_ACTION_TYPES.index(action_type)
    except ValueError:
        action_index = 0
    confidence = _clamp_float((example or {}).get("confidence", 0.5), 0.0, 1.0)
    for index in range(len(POLICY_ACTION_TYPES)):
        output[index] = 0.10
    output[action_index] = max(output[action_index], confidence)
    rx = _clamp_float((example or {}).get("rx", (example or {}).get("action_rx", 0.5)), 0.0, 1.0)
    ry = _clamp_float((example or {}).get("ry", (example or {}).get("action_ry", 0.5)), 0.0, 1.0)
    gx = max(0, min(POLICY_GRID_WIDTH - 1, int(rx * POLICY_GRID_WIDTH)))
    gy = max(0, min(POLICY_GRID_HEIGHT - 1, int(ry * POLICY_GRID_HEIGHT)))
    output[len(POLICY_ACTION_TYPES) + gy * POLICY_GRID_WIDTH + gx] = max(0.10, confidence)
    wheel_delta = int((example or {}).get("wheel_delta", 0) or 0)
    output[-3] = _clamp_float(wheel_delta / 1200.0, -1.0, 1.0)
    output[-2] = _clamp_float(0.5 + float((example or {}).get("score_delta", 0.0) or 0.0) * 2.5, 0.0, 1.0)
    output[-1] = _clamp_float((example or {}).get("uncertainty", 0.5), 0.02, 0.95)
    return output

def parse_visual_policy_output(values):
    values = [float(item) if math.isfinite(float(item)) else 0.0 for item in list(values)[:POLICY_OUTPUT_SIZE]]
    if len(values) < POLICY_OUTPUT_SIZE:
        values += [0.0] * (POLICY_OUTPUT_SIZE - len(values))
    action_values = values[:len(POLICY_ACTION_TYPES)]
    action_index = max(range(len(action_values)), key=lambda index: action_values[index])
    grid_values = values[len(POLICY_ACTION_TYPES):len(POLICY_ACTION_TYPES) + POLICY_GRID_WIDTH * POLICY_GRID_HEIGHT]
    grid_index = max(range(len(grid_values)), key=lambda index: grid_values[index]) if grid_values else 0
    gx = grid_index % POLICY_GRID_WIDTH
    gy = grid_index // POLICY_GRID_WIDTH
    confidence = _sigmoid_float(max(action_values) * 5.0 - 2.5) if max(action_values) <= 1.0 and min(action_values) >= 0.0 else _sigmoid_float(max(action_values))
    wheel_delta = int(max(-1200, min(1200, round(values[-3] * 1200.0 / 120.0) * 120)))
    return {"action_type": POLICY_ACTION_TYPES[action_index], "x_ratio": (gx + 0.5) / POLICY_GRID_WIDTH, "y_ratio": (gy + 0.5) / POLICY_GRID_HEIGHT, "wheel_delta": wheel_delta, "value": _clamp_float(values[-2], 0.0, 1.0), "uncertainty": _clamp_float(values[-1], 0.02, 0.95), "confidence": _clamp_float(confidence, 0.0, 1.0), "grid_index": grid_index, "action_index": action_index}

def _normalize_kernel(values, fallback=None):
    values = [float(item) for item in values[:9]]
    if len(values) < 9:
        values += [0.0] * (9 - len(values))
    mean = sum(values) / 9.0
    centered = [item - mean for item in values]
    scale = max([abs(item) for item in centered] + [1e-6])
    if scale <= 1e-6:
        centered = [float(item) for item in (fallback or [0.0] * 9)[:9]]
        if len(centered) < 9:
            centered += [0.0] * (9 - len(centered))
        mean = sum(centered) / 9.0
        centered = [item - mean for item in centered]
        scale = max([abs(item) for item in centered] + [1e-6])
    return [max(-1.0, min(1.0, item / scale * 0.42)) for item in centered]

def conv_policy_features(gray_values, conv_weight, conv_bias):
    if len(gray_values) != 32 * 18:
        return []
    features = []
    weights = [float(item) for item in conv_weight]
    biases = [float(item) for item in conv_bias]
    if len(weights) != 4 * 9 or len(biases) != 4:
        return []
    for channel in range(4):
        kernel = weights[channel * 9:(channel + 1) * 9]
        bias = biases[channel]
        for y in range(16):
            row = y * 32
            for x in range(30):
                value = bias
                value += gray_values[row + x] * kernel[0]
                value += gray_values[row + x + 1] * kernel[1]
                value += gray_values[row + x + 2] * kernel[2]
                value += gray_values[row + 32 + x] * kernel[3]
                value += gray_values[row + 32 + x + 1] * kernel[4]
                value += gray_values[row + 32 + x + 2] * kernel[5]
                value += gray_values[row + 64 + x] * kernel[6]
                value += gray_values[row + 64 + x + 1] * kernel[7]
                value += gray_values[row + 64 + x + 2] * kernel[8]
                features.append(value if value > 0.0 else 0.0)
    return features

def policy_onnx_bytes(conv_weight, fc_weight, bias, conv_bias=None):
    conv_weight = [float(item) for item in (conv_weight or [])]
    fc_weight = [float(item) for item in (fc_weight or [])]
    bias = [float(item) for item in (bias or [])]
    conv_bias = [float(item) for item in (conv_bias or [0.0, 0.0, 0.0, 0.0])]
    if len(conv_weight) != 4 * 1 * 3 * 3:
        raise ValueError("conv_weight 必须是 4×1×3×3")
    if len(conv_bias) != 4:
        raise ValueError("conv_bias 必须是 4")
    if len(fc_weight) != 4 * 16 * 30 * 2:
        raise ValueError("fc_weight 必须是 1920×2")
    if len(bias) != 2:
        raise ValueError("bias 必须是 2")
    graph = _pb_message(1, _onnx_node("Conv", ["input", "conv_weight", "conv_bias"], ["conv"], "vision_conv3x3"))
    graph += _pb_message(1, _onnx_node("Relu", ["conv"], ["relu"], "vision_relu"))
    graph += _pb_message(1, _onnx_node("Flatten", ["relu"], ["flat"], "vision_flatten"))
    graph += _pb_message(1, _onnx_node("MatMul", ["flat", "fc_weight"], ["logits"], "vision_matmul"))
    graph += _pb_message(1, _onnx_node("Add", ["logits", "bias"], ["biased"], "vision_bias"))
    graph += _pb_message(1, _onnx_node("Sigmoid", ["biased"], ["output"], "vision_sigmoid"))
    graph += _pb_string(2, "ld_training_visual_policy")
    graph += _pb_message(5, _onnx_tensor("conv_weight", [4, 1, 3, 3], conv_weight))
    graph += _pb_message(5, _onnx_tensor("conv_bias", [4], conv_bias))
    graph += _pb_message(5, _onnx_tensor("fc_weight", [4 * 16 * 30, 2], fc_weight))
    graph += _pb_message(5, _onnx_tensor("bias", [1, 2], bias))
    graph += _pb_message(11, _onnx_value_info("input", [1, 1, 18, 32]))
    graph += _pb_message(12, _onnx_value_info("output", [1, 2]))
    opset = _pb_string(1, "") + _pb_int(2, 13)
    return _pb_int(1, 7) + _pb_string(2, "ld_training_panel") + _pb_message(7, graph) + _pb_message(8, opset)

def policy_multihead_onnx_bytes(fc_weight, bias):
    fc_weight = [float(item) for item in (fc_weight or [])]
    bias = [float(item) for item in (bias or [])]
    if len(fc_weight) != POLICY_INPUT_SIZE * POLICY_OUTPUT_SIZE:
        raise ValueError("multihead fc_weight 必须是 {}×{}".format(POLICY_INPUT_SIZE, POLICY_OUTPUT_SIZE))
    if len(bias) != POLICY_OUTPUT_SIZE:
        raise ValueError("multihead bias 必须是 {}".format(POLICY_OUTPUT_SIZE))
    graph = _pb_message(1, _onnx_node("MatMul", ["input", "fc_weight"], ["logits"], "multihead_matmul"))
    graph += _pb_message(1, _onnx_node("Add", ["logits", "bias"], ["policy_output"], "multihead_bias"))
    graph += _pb_string(2, "ld_training_multihead_visual_policy")
    graph += _pb_message(5, _onnx_tensor("fc_weight", [POLICY_INPUT_SIZE, POLICY_OUTPUT_SIZE], fc_weight))
    graph += _pb_message(5, _onnx_tensor("bias", [1, POLICY_OUTPUT_SIZE], bias))
    graph += _pb_message(11, _onnx_value_info("input", [1, POLICY_INPUT_SIZE]))
    graph += _pb_message(12, _onnx_value_info("policy_output", [1, POLICY_OUTPUT_SIZE]))
    opset = _pb_string(1, "") + _pb_int(2, 13)
    return _pb_int(1, 7) + _pb_string(2, "ld_training_panel") + _pb_message(7, graph) + _pb_message(8, opset)

def _capture_client_gdi(hwnd, max_width=640, max_height=360):
    capture_started_monotonic_ns = time.monotonic_ns()
    capture_started = time.time()
    local = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(local)):
        return None
    source_width = int(local.right - local.left); source_height = int(local.bottom - local.top)
    if source_width <= 0 or source_height <= 0:
        return None
    scale = min(1.0, max_width / source_width, max_height / source_height)
    width = max(1, int(source_width * scale)); height = max(1, int(source_height * scale))
    source_dc = user32.GetDC(hwnd)
    if not source_dc:
        return None
    memory_dc = 0; bitmap = 0; old_object = 0
    try:
        memory_dc = gdi32.CreateCompatibleDC(source_dc); bitmap = gdi32.CreateCompatibleBitmap(source_dc, width, height)
        if not memory_dc or not bitmap:
            return None
        old_object = gdi32.SelectObject(memory_dc, bitmap)
        if not gdi32.StretchBlt(memory_dc, 0, 0, width, height, source_dc, 0, 0, source_width, source_height, SRCCOPY):
            return None
        info = BITMAPINFO(); info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER); info.bmiHeader.biWidth = width; info.bmiHeader.biHeight = -height; info.bmiHeader.biPlanes = 1; info.bmiHeader.biBitCount = 32; info.bmiHeader.biCompression = BI_RGB
        raw = (ctypes.c_ubyte * (width * height * 4))()
        if gdi32.GetDIBits(memory_dc, bitmap, 0, height, ctypes.byref(raw), ctypes.byref(info), DIB_RGB_COLORS) != height:
            return None
        finished_ns = time.monotonic_ns(); finished = time.time()
        return {"width": width, "height": height, "bgra": bytes(raw), "pixel_format": "BGRA", "capture_started_monotonic_ns": capture_started_monotonic_ns, "capture_finished_monotonic_ns": finished_ns, "capture_started": capture_started, "capture_finished": finished, "capture_backend": "gdi", "capture_elapsed_ms": (finished_ns-capture_started_monotonic_ns)/1000000.0, "capture_fallback": 0, "capture_failure_reason": ""}
    finally:
        if old_object and memory_dc: gdi32.SelectObject(memory_dc, old_object)
        if bitmap: gdi32.DeleteObject(bitmap)
        if memory_dc: gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(hwnd, source_dc)


def capture_looks_invalid(image):
    if not image:
        return True
    width = int(image.get("width", 0) or 0)
    height = int(image.get("height", 0) or 0)
    bgra = image.get("bgra")
    rgb = image.get("rgb")
    if width <= 0 or height <= 0:
        return True
    if bgra is not None:
        return len(bgra) < width * height * 4
    if rgb is not None:
        return len(rgb) < width * height * 3
    return True

def capture_validation(hwnd):
    rect = valid_client(hwnd, False)
    if rect is None:
        return None
    root = root_window(hwnd)
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not root or not pid.value:
        return None
    coverage, monitors = physical_monitor_coverage(rect)
    obstruction = getattr(client_unobscured, "last_obstruction", None)
    overlay = getattr(client_unobscured, "last_overlay", None)
    return {"hwnd": int(hwnd), "root": int(root), "pid": int(pid.value), "rect": tuple(int(value) for value in rect), "monitor_coverage": float(coverage), "monitors": [tuple(int(v) for v in item) for item in monitors], "obstruction": obstruction or {}, "ignored_overlay": overlay or {}}

def same_capture_validation(first, second):
    return bool(first and second and int(first.get("hwnd", 0)) == int(second.get("hwnd", 0)) and int(first.get("root", 0)) == int(second.get("root", 0)) and int(first.get("pid", 0)) == int(second.get("pid", 0)) and tuple(first.get("rect", ())) == tuple(second.get("rect", ())))

def _capture_client_desktop(hwnd, max_width=640, max_height=360, failure_reason="gdi_failed", validation_before=None, capture_generation=0):
    capture_started_monotonic_ns = time.monotonic_ns()
    capture_started = time.time()
    before = validation_before or capture_validation(hwnd)
    if before is None:
        return None
    rect = before["rect"]
    source_width = int(rect[2] - rect[0])
    source_height = int(rect[3] - rect[1])
    if source_width <= 0 or source_height <= 0:
        return None
    scale = min(1.0, max_width / source_width, max_height / source_height)
    width = max(1, int(source_width * scale))
    height = max(1, int(source_height * scale))
    source_dc = user32.GetDC(wintypes.HWND(0))
    if not source_dc:
        return None
    memory_dc = 0
    bitmap = 0
    old_object = 0
    try:
        memory_dc = gdi32.CreateCompatibleDC(source_dc)
        bitmap = gdi32.CreateCompatibleBitmap(source_dc, width, height)
        if not memory_dc or not bitmap:
            return None
        old_object = gdi32.SelectObject(memory_dc, bitmap)
        if not gdi32.StretchBlt(memory_dc, 0, 0, width, height, source_dc, rect[0], rect[1], source_width, source_height, SRCCOPY):
            return None
        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB
        raw = (ctypes.c_ubyte * (width * height * 4))()
        if gdi32.GetDIBits(memory_dc, bitmap, 0, height, ctypes.byref(raw), ctypes.byref(info), DIB_RGB_COLORS) != height:
            return None
        validation_after = capture_validation(hwnd)
        if not same_capture_validation(before, validation_after):
            return None
        finished_ns = time.monotonic_ns()
        finished = time.time()
        return {"width": width, "height": height, "bgra": bytes(raw), "pixel_format": "BGRA", "capture_started_monotonic_ns": capture_started_monotonic_ns, "capture_finished_monotonic_ns": finished_ns, "capture_started": capture_started, "capture_finished": finished, "capture_backend": "desktop", "capture_elapsed_ms": (finished_ns-capture_started_monotonic_ns)/1000000.0, "capture_fallback": 1, "capture_failure_reason": failure_reason, "fallback_reason": failure_reason, "monitor_coverage": float((validation_after or before or {}).get("monitor_coverage", 0.0) or 0.0), "obstruction": (validation_after or before or {}).get("obstruction") or {}, "validation_before": before, "validation_after": validation_after, "capture_generation": int(capture_generation)}
    finally:
        if old_object and memory_dc:
            gdi32.SelectObject(memory_dc, old_object)
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if memory_dc:
            gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(wintypes.HWND(0), source_dc)

def capture_client(hwnd, max_width=640, max_height=360, use_fallback=False):
    capture_client.last_failure_reason = ""
    if not IS_WINDOWS:
        rect = valid_client(hwnd, False)
        image = PLATFORM_BACKEND.capture_client(hwnd, rect) if rect else None
        if image is None:
            capture_client.last_failure_reason = PLATFORM_BACKEND.mode_unavailable_reason()
        return image
    capture_client.last_failure_audit = {}
    generation = time.monotonic_ns()
    before = capture_validation(hwnd)
    if before is None:
        capture_client.last_failure_reason = getattr(valid_client, "last_reason", "截图前窗口校验失败")
        capture_client.last_failure_audit = {"validation_before": {}, "validation_after": {}, "obstruction": getattr(client_unobscured, "last_obstruction", None) or {}, "ignored_overlay": getattr(client_unobscured, "last_overlay", None) or {}, "fallback_reason": capture_client.last_failure_reason}
        return None
    cache_key = (int(before.get("root", 0)), int(before.get("pid", 0)), int(before.get("hwnd", 0)))
    with CAPTURE_BACKEND_CACHE_LOCK:
        preferred = CAPTURE_BACKEND_CACHE.get(cache_key, "desktop" if use_fallback else "gdi")
    prefer_desktop = str(preferred) == "desktop" or bool(use_fallback)
    image = None if prefer_desktop else _capture_client_gdi(hwnd, max_width, max_height)
    reason = "preferred_desktop" if prefer_desktop else "gdi_failed"
    if image is not None and capture_looks_invalid(image):
        reason = "gdi_buffer_invalid"
        image = None
    if image is not None:
        after = capture_validation(hwnd)
        if not same_capture_validation(before, after):
            capture_client.last_failure_reason = "GDI 截图期间客户区、绑定对象或遮挡状态变化"
            capture_client.last_failure_audit = {"validation_before": before, "validation_after": after or {}, "fallback_reason": capture_client.last_failure_reason}
            with CAPTURE_BACKEND_CACHE_LOCK:
                CAPTURE_BACKEND_CACHE[cache_key] = "desktop"
            return None
        image.update({"validation_before": before, "validation_after": after, "capture_generation": int(generation), "fallback_reason": "", "monitor_coverage": float((after or before or {}).get("monitor_coverage", 0.0) or 0.0), "obstruction": (after or before or {}).get("obstruction") or {}})
        with CAPTURE_BACKEND_CACHE_LOCK:
            CAPTURE_BACKEND_CACHE[cache_key] = "gdi"
        return image
    image = _capture_client_desktop(hwnd, max_width, max_height, reason, before, generation)
    if image is not None and capture_looks_invalid(image):
        capture_client.last_failure_reason = "桌面回退截图缓冲区无效"
        image["capture_failure_reason"] = capture_client.last_failure_reason
        return None
    if image is None:
        capture_client.last_failure_reason = "桌面回退截图前后校验失败或客户区被遮挡"
        capture_client.last_failure_audit = {"validation_before": before, "validation_after": {}, "fallback_reason": capture_client.last_failure_reason}
    else:
        with CAPTURE_BACKEND_CACHE_LOCK:
            CAPTURE_BACKEND_CACHE[cache_key] = "desktop"
    return image

def _image_bgra_sample(image, limit=4096):
    if not image or image.get("bgra") is None:
        return []
    bgra = image.get("bgra")
    width = max(1, int(image.get("width", 0) or 0))
    height = max(1, int(image.get("height", 0) or 0))
    total = max(0, min(len(bgra) // 4, width * height))
    if total <= 0:
        return []
    step = max(1, total // max(1, int(limit)))
    values = []
    for index in range(0, total, step):
        base = index * 4
        if base + 2 < len(bgra):
            values.append((bgra[base], bgra[base + 1], bgra[base + 2]))
    return values

def image_black_ratio(image):
    values = _image_bgra_sample(image)
    if not values:
        return 1.0
    return sum(1 for b, g, r in values if r <= 3 and g <= 3 and b <= 3) / float(len(values))

def image_white_ratio(image):
    values = _image_bgra_sample(image)
    if not values:
        return 1.0
    return sum(1 for b, g, r in values if r >= 252 and g >= 252 and b >= 252) / float(len(values))

def image_mean_difference(left, right):
    left_values = _image_bgra_sample(left)
    right_values = _image_bgra_sample(right)
    count = min(len(left_values), len(right_values))
    if count <= 0:
        return 1.0
    total = 0.0
    for a, b in zip(left_values[:count], right_values[:count]):
        total += (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])) / 765.0
    return max(0.0, min(1.0, total / count))

def image_luma_variance(image):
    values = _image_bgra_sample(image)
    if not values:
        return 0.0
    lumas = [0.114 * b + 0.587 * g + 0.299 * r for b, g, r in values]
    mean = sum(lumas) / len(lumas)
    return sum((value - mean) ** 2 for value in lumas) / len(lumas)

def entry_capture_pixel_validation(hwnd):
    entry_capture_pixel_validation.last_reason = ""
    if not IS_WINDOWS:
        image = capture_client(hwnd, 320, 180)
        if capture_looks_invalid(image):
            entry_capture_pixel_validation.last_reason = "进入模式失败：客户区画面无法被完整记录"
            return False
        return True
    before = capture_validation(hwnd)
    if before is None:
        entry_capture_pixel_validation.last_reason = "进入模式失败：客户区画面无法被完整记录"
        return False
    rect = tuple(before.get("rect", ()))
    if len(rect) != 4:
        entry_capture_pixel_validation.last_reason = "进入模式失败：客户区画面无法被完整记录"
        return False
    source_width = max(1, int(rect[2]) - int(rect[0]))
    source_height = max(1, int(rect[3]) - int(rect[1]))
    scale = min(1.0, 320.0 / source_width, 180.0 / source_height)
    expected_width = max(1, int(source_width * scale))
    expected_height = max(1, int(source_height * scale))
    samples = []
    for _ in range(3):
        image = _capture_client_gdi(hwnd, 320, 180)
        if capture_looks_invalid(image):
            image = _capture_client_desktop(hwnd, 320, 180, "entry_pixel_validation", before, time.monotonic_ns())
        after = capture_validation(hwnd)
        if not same_capture_validation(before, after):
            entry_capture_pixel_validation.last_reason = "进入模式失败：客户区画面无法被完整记录"
            return False
        if capture_looks_invalid(image):
            entry_capture_pixel_validation.last_reason = "进入模式失败：客户区画面无法被完整记录"
            return False
        if int(image.get("width", 0) or 0) != expected_width or int(image.get("height", 0) or 0) != expected_height:
            entry_capture_pixel_validation.last_reason = "进入模式失败：客户区画面无法被完整记录"
            return False
        samples.append(image)
        time.sleep(0.06)
    black = max(image_black_ratio(item) for item in samples)
    white = max(image_white_ratio(item) for item in samples)
    variance = max(image_luma_variance(item) for item in samples)
    diff = max(image_mean_difference(samples[index], samples[index + 1]) for index in range(len(samples) - 1)) if len(samples) >= 2 else 0.0
    if black >= 0.985 or white >= 0.985 or (diff <= 0.0005 and variance <= 1.5):
        entry_capture_pixel_validation.last_reason = "进入模式失败：客户区画面无法被完整记录"
        return False
    cache_key = (int(before.get("root", 0)), int(before.get("pid", 0)), int(before.get("hwnd", 0)))
    with CAPTURE_BACKEND_CACHE_LOCK:
        CAPTURE_BACKEND_CACHE[cache_key] = str(samples[-1].get("capture_backend") or "gdi")
    return True

def extract_frame_features(image):
    width = int(image["width"])
    height = int(image["height"])
    bgra = image.get("bgra")
    rgb = image.get("rgb")
    channels = 4 if bgra is not None else 3
    raw = memoryview(bgra if bgra is not None else rgb)
    if width <= 0 or height <= 0 or len(raw) < width * height * channels:
        image.update({"phash": "0" * 16, "dhash64": "0" * 16, "capture_complete": 0, "content_valuable": 0, "capture_failure_reason": str(image.get("capture_failure_reason") or "像素缓冲区尺寸不足"), "brightness": 0.0, "variance": 0.0, "black_ratio": 1.0, "gray32x18": bytes(32 * 18), "edge_density": 0.0, "color_histogram": struct.pack("<24I", *([0] * 24))})
        return image
    def luma_at(x, y):
        index = (int(y) * width + int(x)) * channels
        if channels == 4:
            b = raw[index]
            g = raw[index + 1]
            r = raw[index + 2]
        else:
            r = raw[index]
            g = raw[index + 1]
            b = raw[index + 2]
        return (r * 299 + g * 587 + b * 114) // 1000, r, g, b
    low = [0] * 72
    for y in range(8):
        sy = min(height - 1, int((y + 0.5) * height / 8))
        for x in range(9):
            sx = min(width - 1, int((x + 0.5) * width / 9))
            low[y * 9 + x] = luma_at(sx, sy)[0]
    dhash = 0
    for y in range(8):
        base = y * 9
        for x in range(8):
            dhash = (dhash << 1) | (1 if low[base + x] > low[base + x + 1] else 0)
    lowfreq = [low[y * 9 + x] for y in range(8) for x in range(8)]
    low_mean = sum(lowfreq) / max(1, len(lowfreq))
    perceptual = 0
    for item in lowfreq:
        perceptual = (perceptual << 1) | (1 if item >= low_mean else 0)
    sample_gray = bytearray(32 * 18)
    hist = [0] * 24
    total_luma = 0.0
    total_luma2 = 0.0
    sample_count = 0
    black_count = 0
    edge_hits = 0
    grid_points = set()
    for gy in range(18):
        sy = min(height - 1, int((gy + 0.5) * height / 18))
        for gx in range(32):
            sx = min(width - 1, int((gx + 0.5) * width / 32))
            luma, r, g, b = luma_at(sx, sy)
            sample_gray[gy * 32 + gx] = luma
            total_luma += luma
            total_luma2 += luma * luma
            sample_count += 1
            if luma <= 3:
                black_count += 1
            hist[min(7, r // 32)] += 1
            hist[8 + min(7, g // 32)] += 1
            hist[16 + min(7, b // 32)] += 1
            grid_points.add(sy * width + sx)
    for gy in range(18):
        base = gy * 32
        for gx in range(31):
            if abs(sample_gray[base + gx] - sample_gray[base + gx + 1]) > 24:
                edge_hits += 1
    total_pixels = width * height
    step_pixels = max(1, total_pixels // 12288)
    for point in range(0, total_pixels, step_pixels):
        if point in grid_points:
            continue
        y = point // width
        x = point - y * width
        luma, r, g, b = luma_at(x, y)
        total_luma += luma
        total_luma2 += luma * luma
        sample_count += 1
        if luma <= 3:
            black_count += 1
        hist[min(7, r // 32)] += 1
        hist[8 + min(7, g // 32)] += 1
        hist[16 + min(7, b // 32)] += 1
    mean = total_luma / max(1, sample_count)
    variance = max(0.0, total_luma2 / max(1, sample_count) - mean * mean)
    image.update({"phash": f"{perceptual:016x}", "dhash64": f"{dhash:016x}", "capture_complete": 1, "content_valuable": 1 if mean >= 3.0 and variance >= 2.0 else 0, "true_static_candidate": 1 if mean < 3.0 or variance < 2.0 else 0, "brightness": mean, "variance": variance, "black_ratio": black_count / max(1, sample_count), "gray32x18": bytes(sample_gray), "edge_density": edge_hits / (18 * 31), "color_histogram": struct.pack("<24I", *hist)})
    return image

def compress_frame_png(image):
    if image.get("bgra") is not None:
        image["png"] = encode_png_bgra(int(image["width"]), int(image["height"]), image.pop("bgra"))
    else:
        image["png"] = encode_png(int(image["width"]), int(image["height"]), image.pop("rgb"))
    return image

def bit_count(value):
    try:
        return value.bit_count()
    except AttributeError:
        return bin(value).count("1")

def frame_score(dhash, historical, current_features=None):
    details = historical if isinstance(historical, dict) else {"hashes": historical or [], "candidate_count": len(historical or []), "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "legacy", "exact_or_approx": "unknown", "recall_guard": False, "score_valid": False, "provisional": False}
    hashes = details.get("hashes", [])
    similarities = details.get("similarities", [])
    meta = {"candidate_count": int(details.get("candidate_count", 0)), "top_k_distance": float(details.get("top_k_distance", 64.0)), "retrieval_fallback": bool(details.get("retrieval_fallback", False)), "retrieval_mode": str(details.get("retrieval_mode", "unknown")), "exact_or_approx": str(details.get("exact_or_approx", "unknown")), "recall_guard": bool(details.get("recall_guard", False)), "score_valid": bool(details.get("score_valid", False)), "provisional": bool(details.get("provisional", False))}
    if not hashes and meta["retrieval_mode"] == "warmup_no_history" and int(details.get("total_history", 0) or 0) <= 0:
        return 1.0, dict(meta, score_valid=True, provisional=False, recall_guard=True, exact_or_approx="exact")
    if not hashes or (not meta["recall_guard"] and not meta["provisional"]):
        return None, dict(meta, score_valid=False)
    weighted = []
    if similarities and len(similarities) == len(hashes):
        for index, similarity in enumerate(similarities):
            try:
                weighted.append((max(0.0, min(1.0, float(similarity))), 1.0 / (1.0 + index)))
            except (TypeError, ValueError):
                pass
    else:
        try:
            current = int(dhash, 16)
        except Exception:
            return None, dict(meta, score_valid=False)
        for index, previous in enumerate(hashes):
            try:
                distance = bit_count(current ^ int(previous, 16))
                weighted.append((1.0 - distance / 64.0, 1.0 / (1.0 + index)))
            except Exception:
                pass
    if not weighted:
        return None, dict(meta, score_valid=False)
    ordered = sorted((max(0.0, min(1.0, value)) for value, _ in weighted), reverse=True)
    top_count = max(1, min(8, len(ordered)))
    top_similarity = sum(ordered[:top_count]) / top_count
    weighted_similarity = sum(value * weight for value, weight in weighted) / max(1e-9, sum(weight for _, weight in weighted))
    similarity = max(0.0, min(1.0, 0.68 * top_similarity + 0.32 * weighted_similarity))
    meta["score_valid"] = not meta["provisional"]
    return max(0.0, min(1.0, 1.0 - similarity)), meta


class AIInputTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = []
        self.session_marker = 0
        self.sequence = 0

    def begin_session(self):
        with self.lock:
            self.pending = []
            self.sequence = 0
            self.session_marker = uuid.uuid4().int & ((1 << (ctypes.sizeof(ULONG_PTR) * 8)) - 1)
            if not self.session_marker:
                self.session_marker = 1
            return self.session_marker

    def clear(self):
        with self.lock:
            self.pending = []

    def register(self, event_type, button, wheel, x, y, behavior_probability=None):
        with self.lock:
            self.sequence += 1
            marker = uuid.uuid4().int & ((1 << (ctypes.sizeof(ULONG_PTR) * 8)) - 1)
            if not marker:
                marker = (self.sequence << 1) | 1
            self.pending.append({"marker": marker, "sequence": self.sequence, "session_marker": self.session_marker, "event_type": event_type, "button": button, "wheel": wheel, "x": x, "y": y, "behavior_probability": behavior_probability, "deadline_ns": time.monotonic_ns() + 250_000_000})
            return marker

    def consume(self, marker, event_type, button, wheel, x, y, now_ns):
        with self.lock:
            self.pending = [item for item in self.pending if item["deadline_ns"] >= now_ns]
            for index, item in enumerate(self.pending):
                if int(marker or 0) != int(item["marker"]):
                    continue
                same_type = item["event_type"] == event_type
                same_button = item["button"] == button
                same_wheel = item["wheel"] == wheel
                same_position = abs(item["x"] - x) <= 1 and abs(item["y"] - y) <= 1
                if same_type and same_button and same_wheel and same_position:
                    matched = self.pending.pop(index)
                    matched["matched_by"] = "marker"
                    return matched
            best = None
            for index, item in enumerate(self.pending):
                if item["event_type"] != event_type or item["button"] != button:
                    continue
                wheel_gap = abs(int(item.get("wheel", 0) or 0) - int(wheel or 0))
                if wheel_gap > 0 and event_type in ("wheel", "horizontal_wheel"):
                    continue
                time_gap = max(0, int(now_ns) - int(item.get("deadline_ns", now_ns)) + 250_000_000)
                if time_gap > 250_000_000:
                    continue
                distance = math.hypot(float(item.get("x", 0) - x), float(item.get("y", 0) - y))
                score = distance + time_gap / 20_000_000.0 + abs(int(item.get("sequence", 0)) - int(self.sequence)) * 0.01
                if distance <= 6.0 and (best is None or score < best[0]):
                    best = (score, index)
            if best is not None:
                matched = self.pending.pop(best[1])
                matched["matched_by"] = "nearest"
                return matched
        return None


class RawHookRingBuffer:
    def __init__(self, capacity):
        self.capacity = max(64, int(capacity))
        self.items = collections.deque()
        self.lock = threading.Lock()
        self.overflow_count = 0

    def put(self, item):
        with self.lock:
            if len(self.items) >= self.capacity:
                self.overflow_count += 1
                return False
            self.items.append(item)
            return True

    def get(self, timeout=0.0):
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            with self.lock:
                if self.items:
                    return self.items.popleft()
            if timeout <= 0.0 or time.monotonic() >= deadline:
                raise queue.Empty
            time.sleep(min(0.001, max(0.0, deadline - time.monotonic())))

    def empty(self):
        with self.lock:
            return not self.items

    def qsize(self):
        with self.lock:
            return len(self.items)

    def pop_overflow_count(self):
        with self.lock:
            value = self.overflow_count
            self.overflow_count = 0
            return value

class DbWriter:
    def __init__(self):
        self.queue = queue.Queue(maxsize=4096)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="DbWriter", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                func, done, holder = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                holder["result"] = func()
            except Exception as error:
                holder["error"] = error
            finally:
                done.set()

    def is_writer_thread(self):
        return threading.current_thread() is self.thread

    def call(self, func, timeout=None):
        if self.is_writer_thread():
            return func()
        if self.stop_event.is_set():
            raise RuntimeError("DbWriter 已停止")
        done = threading.Event()
        holder = {}
        self.queue.put((func, done, holder), timeout=timeout if timeout is not None else 30.0)
        if not done.wait(timeout if timeout is not None else 60.0):
            raise TimeoutError("DbWriter 写入超时")
        if "error" in holder:
            raise holder["error"]
        return holder.get("result")

    def close(self, timeout=5.0):
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(max(0.1, float(timeout)))

class MouseHook:
    def __init__(self, sink):
        self.sink = sink
        self.thread = None
        self.thread_id = 0
        self.handle = None
        self.callback_ref = None
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error = ""
        self.callback_errors = 0
        self.last_callback_error = ""

    def start(self):
        if not IS_WINDOWS:
            if self.thread and self.thread.is_alive():
                return True
            if not PLATFORM_BACKEND.install_mouse_hook():
                self.error = PLATFORM_BACKEND.mode_unavailable_reason()
                return False
            self.stop_event.clear()
            self.ready.clear()
            self.error = ""
            self.thread = threading.Thread(target=self._run_poll, name="MouseHook", daemon=True)
            self.thread.start()
            if not self.ready.wait(2.0):
                self.error = "鼠标监听启动超时"
                return False
            return True
        if self.thread and self.thread.is_alive() and self.handle:
            return True
        self.stop_event.clear()
        self.ready.clear()
        self.error = ""
        self.thread = threading.Thread(target=self._run, name="MouseHook", daemon=True)
        self.thread.start()
        if not self.ready.wait(2.0):
            self.error = "鼠标钩子启动超时"
            return False
        return bool(self.handle)

    def stop(self):
        self.stop_event.set()
        if self.thread_id:
            user32.PostThreadMessageW(self.thread_id, WM_QUIT, 0, 0)
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(2.0)
        if not (self.thread and self.thread.is_alive()):
            self.thread = None
            self.thread_id = 0

    def _run_poll(self):
        self.thread_id = 0
        last_x, last_y, last_mask = 0, 0, 0
        self.ready.set()
        while not self.stop_event.is_set():
            try:
                if hasattr(PLATFORM_BACKEND, "pointer_state"):
                    x, y, mask = PLATFORM_BACKEND.pointer_state()
                else:
                    x, y = PLATFORM_BACKEND.cursor_position() if hasattr(PLATFORM_BACKEND, "cursor_position") else (0, 0)
                    mask = 0
                now = time.time()
                ns = time.monotonic_ns()
                if x != last_x or y != last_y:
                    self.sink((WM_MOUSEMOVE, 0, int(x), int(y), now, ns, 0, 0))
                for bit, down_msg, up_msg in ((8, WM_LBUTTONDOWN, WM_LBUTTONUP), (10, WM_RBUTTONDOWN, WM_RBUTTONUP)):
                    was = bool(last_mask & (1 << bit))
                    is_down = bool(mask & (1 << bit))
                    if was != is_down:
                        self.sink((down_msg if is_down else up_msg, 0, int(x), int(y), now, ns, 0, 0))
                last_x, last_y, last_mask = int(x), int(y), int(mask)
            except Exception as error:
                self.callback_errors += 1
                self.last_callback_error = str(error)
            time.sleep(0.01)

    def _run(self):
        self.thread_id = int(kernel32.GetCurrentThreadId())
        message = MSG()
        user32.PeekMessageW(ctypes.byref(message), None, 0, 0, PM_NOREMOVE)
        self.callback_ref = LowLevelMouseProc(self._callback)
        self.handle = user32.SetWindowsHookExW(WH_MOUSE_LL, self.callback_ref, kernel32.GetModuleHandleW(None), 0)
        if not self.handle:
            self.error = "鼠标钩子安装失败，错误码 {}".format(ctypes.get_last_error())
        self.ready.set()
        if not self.handle:
            return
        try:
            while not self.stop_event.is_set():
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result <= 0:
                    break
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))
        finally:
            if self.handle:
                user32.UnhookWindowsHookEx(self.handle)
            self.handle = None

    def _callback(self, code, wparam, lparam):
        try:
            if code == HC_ACTION:
                info = ctypes.cast(lparam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                self.sink((int(wparam), int(info.mouseData), int(info.pt.x), int(info.pt.y), time.time(), time.monotonic_ns(), int(info.flags), int(info.dwExtraInfo)))
        except Exception as error:
            self.callback_errors += 1
            self.last_callback_error = str(error)
        return user32.CallNextHookEx(self.handle, code, wparam, lparam)


class KeyboardHook:
    def __init__(self, sink):
        self.sink = sink
        self.thread = None
        self.thread_id = 0
        self.handle = None
        self.callback_ref = None
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error = ""
        self.callback_errors = 0
        self.last_callback_error = ""

    def start(self):
        if not IS_WINDOWS:
            if self.thread and self.thread.is_alive():
                return True
            if not PLATFORM_BACKEND.install_keyboard_hook():
                self.error = PLATFORM_BACKEND.mode_unavailable_reason()
                return False
            self.stop_event.clear()
            self.ready.clear()
            self.error = ""
            self.thread = threading.Thread(target=self._run_poll, name="KeyboardHook", daemon=True)
            self.thread.start()
            if not self.ready.wait(2.0):
                self.error = "键盘监听启动超时"
                return False
            return True
        if self.thread and self.thread.is_alive() and self.handle:
            return True
        self.stop_event.clear()
        self.ready.clear()
        self.error = ""
        self.thread = threading.Thread(target=self._run, name="KeyboardHook", daemon=True)
        self.thread.start()
        if not self.ready.wait(2.0):
            self.error = "键盘钩子启动超时"
            return False
        return bool(self.handle)

    def stop(self):
        self.stop_event.set()
        if self.thread_id:
            user32.PostThreadMessageW(self.thread_id, WM_QUIT, 0, 0)
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(2.0)
        if not (self.thread and self.thread.is_alive()):
            self.thread = None
            self.thread_id = 0

    def _run_poll(self):
        self.thread_id = 0
        self.ready.set()
        was_down = False
        while not self.stop_event.is_set():
            try:
                down = bool(PLATFORM_BACKEND.escape_pressed()) if hasattr(PLATFORM_BACKEND, "escape_pressed") else False
                if down and not was_down:
                    self.sink("esc", "检测到 ESC 键", time.time())
                was_down = down
            except Exception as error:
                self.callback_errors += 1
                self.last_callback_error = str(error)
            time.sleep(0.03)

    def _run(self):
        self.thread_id = int(kernel32.GetCurrentThreadId())
        message = MSG()
        user32.PeekMessageW(ctypes.byref(message), None, 0, 0, PM_NOREMOVE)
        self.callback_ref = LowLevelKeyboardProc(self._callback)
        self.handle = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self.callback_ref, kernel32.GetModuleHandleW(None), 0)
        if not self.handle:
            self.error = "键盘钩子安装失败，错误码 {}".format(ctypes.get_last_error())
        self.ready.set()
        if not self.handle:
            return
        try:
            while not self.stop_event.is_set():
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result <= 0:
                    break
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))
        finally:
            if self.handle:
                user32.UnhookWindowsHookEx(self.handle)
            self.handle = None

    def _callback(self, code, wparam, lparam):
        try:
            if code == HC_ACTION and int(wparam) in (WM_KEYDOWN, WM_SYSKEYDOWN):
                info = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                if int(info.vkCode) == VK_ESCAPE:
                    self.sink("esc", "检测到 ESC 键", time.time())
        except Exception as error:
            self.callback_errors += 1
            self.last_callback_error = str(error)
        return user32.CallNextHookEx(self.handle, code, wparam, lparam)

class WindowEventGuard:
    def __init__(self, event_sink):
        self.event_sink = event_sink
        self.lock = threading.Lock()
        self.target_hwnd = 0
        self.target_root = 0
        self.target_rect = None
        self.hooks = []
        self.last_emit = 0.0
        self.callback = WinEventProc(self._callback)
        self.callback_errors = 0
        self.last_callback_error = ""

    def _callback(self, hook, event, hwnd, object_id, child_id, event_thread, event_time):
        with self.lock:
            target_hwnd = self.target_hwnd
            target_root = self.target_root
            target_rect = self.target_rect
        if not hwnd or not target_root:
            return
        try:
            root = int(root_window(hwnd) or 0)
            should_check = root == int(target_root)
            if not should_check and event in (EVENT_SYSTEM_FOREGROUND, EVENT_OBJECT_LOCATIONCHANGE):
                rect = window_rectangle(hwnd)
                should_check = bool(rect and target_rect and rectangle_overlap(rect, target_rect))
            if not should_check:
                return
            now = time.monotonic()
            with self.lock:
                if now - self.last_emit < 0.05:
                    return
                if target_hwnd and user32.IsWindow(target_hwnd):
                    self.target_rect = client_rect(target_hwnd) or self.target_rect
                self.last_emit = now
            self.event_sink()
        except Exception as error:
            self.callback_errors += 1
            self.last_callback_error = str(error)

    def start(self, hwnd):
        self.stop()
        target_root = root_window(hwnd)
        if not target_root:
            return False
        hooks = []
        for event in (EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_MINIMIZESTART, EVENT_OBJECT_LOCATIONCHANGE):
            handle = user32.SetWinEventHook(event, event, 0, self.callback, 0, 0, WINEVENT_OUTOFCONTEXT)
            if handle:
                hooks.append(handle)
        with self.lock:
            self.target_hwnd = int(hwnd)
            self.target_root = int(target_root)
            self.target_rect = client_rect(hwnd)
            self.hooks = hooks
        return bool(hooks)

    def stop(self):
        with self.lock:
            hooks = list(self.hooks)
            self.hooks = []
            self.target_hwnd = 0
            self.target_root = 0
            self.target_rect = None
        for hook in hooks:
            try:
                user32.UnhookWinEvent(hook)
            except Exception:
                pass

class CapturePipeline:
    def __init__(self, controller):
        self.controller = controller

    def start(self, mode, context_data):
        return self.controller._start_runtime_threads(mode, context_data)

    def request_immediate(self, reason=""):
        return self.controller.request_immediate_capture(reason)

class MouseRecorder:
    def __init__(self, controller):
        self.controller = controller

    def record_synthetic_ai(self, event_type, button, wheel, x, y, behavior_probability=None):
        return self.controller._record_synthetic_ai_mouse(event_type, button, wheel, x, y, behavior_probability)

    def flush(self, timeout=3.0):
        return self.controller.flush_mouse_records(timeout)

class SleepManager:
    def __init__(self, controller):
        self.controller = controller

    def enter_manual(self):
        return self.controller.enter_sleep("manual")

    def enter_auto(self):
        return self.controller.enter_sleep("auto_sleep_worth")

class PolicyRuntime:
    def __init__(self, controller):
        self.controller = controller

    def decide(self, features, rect):
        return self.controller._policy_decision(features, rect)

class PersistenceRecovery:
    def __init__(self, controller):
        self.controller = controller

    def prepare(self, mode, automatic=False):
        return self.controller._prepare_storage_and_recovery(mode, automatic)

    def replay_mouse(self, max_items):
        return self.controller.db_writer.call(lambda: self.controller.store.replay_accepted_journal(kinds={"mouse", "mouse_segment"}, max_items=max(1, int(max_items))), timeout=30.0)

class Controller:
    def __init__(self, settings, event_sink):
        self.settings = settings
        self.event_sink = event_sink
        self.store = DataStore()
        self.platform_backend = PLATFORM_BACKEND
        self.pending_critical_errors = []
        self.resources = ResourceGovernor(self.settings.data["storage_path"])
        self.lock = threading.RLock()
        self.shutdown_lock = threading.Lock()
        self.shutdown_started = False
        self.state = "idle"
        self.epoch = 0
        self.cancel_event = threading.Event()
        self.target_hwnd = None
        self.target_root = None
        self.target_pid = 0
        self.target_process_path = ""
        self.target_rect = None
        self.session_id = None
        self.session_mode = None
        self.session_started = 0.0
        self.hunger_anchor_ns = 0
        self.last_valid_score = None
        self.frame_scores = []
        self.frame_count = 0
        self.mouse_count = 0
        self.session_valid_frames = 0
        self.session_valid_actions = 0
        self.training_auto_sleep_count = 0
        self.last_auto_sleep_at = 0.0
        self.last_auto_sleep_fingerprint = ""
        self.auto_sleep_times = []
        self.last_auto_sleep_pool_bytes = 0
        self.last_sleep_fingerprint_check = 0.0
        self.pending_sleep_decision = None
        self.current_training_fingerprint = ""
        self.gdi_failures = 0
        self.gdi_static_count = 0
        self.last_feature_hash = ""
        self.stable_feature_frames = 0
        self.last_gdi_hash = ""
        self.last_gdi_hash_input_ns = 0
        self.fallback_capture_pending = False
        self.last_mouse_activity_ns = 0
        self.last_mouse_by_source = {"ai": None, "user": None, "external_injected": None}
        self.ai_step = 0
        self.ai_plan = []
        self.latest_frame_features = None
        self.latest_frame_id = None
        self.immediate_capture_event = threading.Event()
        self.last_immediate_capture = 0.0
        self.ai_region_counts = collections.Counter()
        self.exact_state_cache = collections.deque(maxlen=128)
        self.action_limits = {}
        self.action_benefit_fuse = {}
        self.recent_ai_actions = []
        self.last_model_training = 0.0
        self.last_training_attempt = 0.0
        self.last_training_success = 0.0
        self.last_training_attempt_fingerprint = ""
        self.last_successful_training_fingerprint = ""
        self.last_training_failure_reason = ""
        self.training_retry_count = 0
        self.next_training_retry_at = 0.0
        self.last_observation = 0.0
        self.capture_interval_seconds = 1.0
        self.capture_failures = 0
        self.capture_failure_started = 0.0
        self.last_persisted_frame_time = 0.0
        self.last_skipped_frame_notice = 0.0
        self.last_capacity_notice_at = 0.0
        self.last_capacity_notice_tier = 0
        self.mouse_queue = queue.Queue(maxsize=12000)
        self.mouse_segment_queue = queue.Queue(maxsize=4096)
        self.mouse_sqlite_degraded_until = 0.0
        self.frame_replay_pause_until = 0.0
        self.storage_sqlite_lock = threading.RLock()
        self.raw_mouse_queue = queue.Queue(maxsize=16000)
        self.raw_critical_queue = queue.Queue(maxsize=2048)
        self.raw_hook_ring = RawHookRingBuffer(32768)
        self.raw_mouse_stop = threading.Event()
        self.raw_mouse_drops = 0
        self.raw_mouse_losses = {}
        self.raw_loss_lock = threading.Lock()
        self.pipeline_context = None
        self.capture_queue = queue.Queue(maxsize=32)
        self.feature_queue = queue.Queue(maxsize=32)
        self.persist_queue = queue.Queue(maxsize=32)
        self.pipeline_stop = threading.Event()
        self.pipeline_threads = []
        self.pipeline_losses = {}
        self.move_segments = {}
        self.pipeline_last_emit = 0.0
        self.control_queue = queue.Queue(maxsize=64)
        self.stop_requested = threading.Event()
        self.emergency_stop_event = threading.Event()
        self.emergency_stop_reason = ""
        self.emergency_stop_token = None
        self.recovery_pending = False
        self.recovery_reason = ""
        self.last_move_kept = None
        self.writer_stop = threading.Event()
        self.writer_busy = threading.Event()
        self.worker_threads = []
        self.db_writer = DbWriter()
        self.store.attach_db_writer(self.db_writer)
        self.capture_pipeline = CapturePipeline(self)
        self.mouse_recorder = MouseRecorder(self)
        self.sleep_manager = SleepManager(self)
        self.policy_runtime = PolicyRuntime(self)
        self.persistence_recovery = PersistenceRecovery(self)
        self.writer = threading.Thread(target=self._mouse_writer, name="StorageWriter")
        self.writer.start()
        self.raw_mouse_thread = threading.Thread(target=self._raw_mouse_loop, name="MouseParser")
        self.raw_mouse_thread.start()
        self.control_thread = threading.Thread(target=self._control_loop, name="SessionControl")
        self.control_thread.start()
        self.hook = MouseHook(self.raw_hook_ring.put)
        self.keyboard_hook = KeyboardHook(self.on_control_signal)
        self.window_guard = WindowEventGuard(self._on_window_event)
        self.capture_threads = []
        self.loss_lock = threading.Lock()
        self.move_loss = {}
        self.ai_input_tracker = AIInputTracker()
        self.sleep_origin = ""
        self.sleep_task1_done = False
        self.sleep_task2_done = False
        self.sleep_task_terminal_state = {"task1_terminal_state": "not_started", "task2_started": False, "task2_done": False, "files_deleted": 0, "needs_recovery": False}
        self.state_events = set(STATE_TRANSITION_EVENTS)
        self.state_transitions = {key: set(value) for key, value in STATE_TRANSITION_EVENTS.items()}
        self.detached_pipeline_contexts = []

    def emit(self, kind, payload):
        self.event_sink(kind, payload)

    def post_state(self, detail=""):
        with self.lock:
            state = self.state
            sample = self.resources.sample()
        self.emit("state", {"state": state, "detail": detail, "cpu": sample["cpu"], "memory": sample["memory"]})

    def _record_resource_decisions(self, session_id=None):
        try:
            decisions = self.resources.pop_resource_decisions()
        except Exception:
            return
        if not decisions:
            return
        with self.lock:
            sid = session_id or self.session_id
        for item in decisions:
            try:
                self.store.add_system_event(sid, "resource_state_transition", item)
            except Exception:
                pass

    def _stop_for_client_recording_budget(self, token, internal_reason, payload=None):
        with self.lock:
            sid = self.session_id
        detail = {"internal_reason": str(internal_reason or ""), "payload": dict(payload or {}), "resource": self.resources.sample(), "time": time.time()}
        try:
            self.store.add_system_event(sid, "client_recording_impossible_due_to_resource_budget", detail)
        except Exception:
            pass
        return self.request_idle("客户区画面无法在资源安全预算内完整记录", token)

    def _critical_recording_failure(self, session_id, stage, reason, error=None, token=None, payload=None):
        reason = str(reason or "客户区画面无法被完整记录")
        token = self.epoch if token is None else token
        try:
            exc = error if error is not None else RuntimeError(reason)
            self.store.add_critical_exception("CapturePipeline", str(stage), exc, session_id=session_id, token=token, resource_state=self.resources.capture_snapshot().state, payload=dict(payload or {}, reason=reason))
        except Exception:
            pass
        try:
            self.store.mark_session_untrainable(session_id, reason)
        except Exception:
            pass
        return self.request_idle("客户区画面无法被完整记录：" + reason, token)

    def _pipeline_hard_gate_reason(self, context=None):
        context = context or self.pipeline_context
        if context is not None:
            checks = (("capture_queue", context.capture_queue), ("feature_queue", context.feature_queue), ("persist_queue", context.persist_queue))
            for name, item in checks:
                try:
                    if item.full():
                        return name + " 满"
                except Exception:
                    pass
        sample = self.resources.sample()
        if float(sample.get("png_encode_ms_p95", sample.get("png_encode_ms", 0.0)) or 0.0) >= 1500.0:
            return "PNG 编码超时"
        if float(sample.get("sqlite_transaction_ms_p95", sample.get("sqlite_transaction_ms", 0.0)) or 0.0) >= 2000.0 or float(sample.get("sqlite_latency_p95", sample.get("sqlite_latency", 0.0)) or 0.0) >= 2000.0:
            return "SQLite 事务超时"
        if int(sample.get("wal_bytes", 0) or 0) >= 1024 * 1024 * 1024:
            return "WAL 超限"
        if float(sample.get("disk_write_latency_p95", sample.get("disk_write_latency", 0.0)) or 0.0) >= 650.0:
            return "fsync 延迟超限"
        if float(sample.get("ui_heartbeat_jitter_ms_p95", sample.get("ui_heartbeat_jitter_ms", 0.0)) or 0.0) >= 1500.0:
            return "UI heartbeat 抖动超限"
        try:
            backlog = self.store.accepted_journal_backlog()
            if int(backlog.get("total", 0) or 0) >= 256:
                return "accepted journal 未完成数量超限"
        except Exception:
            pass
        return ""

    def _public_stop_reason(self, reason):
        text = str(reason or "")
        if "ESC" in text or "Esc" in text or "esc" in text:
            return "检测到 ESC 键"
        if "鼠标" in text and ("离开" in text or "客户区外" in text):
            return "鼠标已离开目标窗口客户区"
        return "客户区画面无法被完整记录"

    def _event_for_stop_reason(self, reason):
        text = self._public_stop_reason(reason)
        if "ESC" in text:
            return "esc"
        if "鼠标" in text:
            return "mouse_outside"
        return "client_invalid"

    def _transition_state_locked(self, event, source, target, reason="", token=None):
        event = str(event or "")
        source = str(source or self.state)
        target = str(target or "")
        actual = self.state
        if actual != source:
            try:
                self.store.add_system_event(self.session_id, "state_transition_rejected", {"event": event, "declared_source": source, "actual_source": actual, "to": target, "reason": reason, "token": token, "time": time.time()})
            except Exception:
                pass
            return False
        if source == target:
            return True
        try:
            assert_transition(event, source, target)
        except AssertionError as error:
            try:
                self.store.add_system_event(self.session_id, "state_transition_assertion_failed", {"event": event, "from": source, "to": target, "reason": reason, "token": token, "error": str(error), "time": time.time()})
            except Exception:
                pass
            return False
        if event not in self.state_events or (source, target) not in self.state_transitions.get(event, set()):
            try:
                self.store.add_system_event(self.session_id, "state_transition_rejected", {"event": event, "from": source, "to": target, "reason": reason, "token": token, "time": time.time()})
            except Exception:
                pass
            return False
        self.state = target
        now = time.time()
        cursor = cursor_position()
        window_state = "hwnd={} pid={} rect={}".format(int(self.target_hwnd or 0), int(self.target_pid or 0), self.target_rect)
        resource = self.resources.sample()
        payload = {"event": event, "from": source, "to": target, "reason": reason, "trigger": event, "token": token, "time": now, "timestamp": now, "target_hwnd": int(self.target_hwnd or 0), "target_pid": int(self.target_pid or 0), "window_state": window_state, "cursor_position": cursor, "resource_state": resource.get("resource_state", ""), "resource": {"cpu": resource.get("cpu"), "memory": resource.get("memory"), "pause_reason": resource.get("pause_reason", "")}}
        try:
            self.store.add_system_event(self.session_id, "state_transition", payload)
            self.store.record_mode_transition(self.session_id, source, target, reason, event, window_state, cursor, resource.get("resource_state", ""), payload)
        except Exception:
            pass
        return True

    def _transition_state(self, event, source, target, reason="", token=None):
        with self.lock:
            return self._transition_state_locked(event, source, target, reason, token)

    def _cache_exact_state(self, frame_id, image_or_features, score, hunger=None):
        if not frame_id or score is None or not isinstance(image_or_features, dict):
            return
        entry = {"frame_id": str(frame_id), "dhash64": str(image_or_features.get("dhash64") or image_or_features.get("state_hash") or ""), "gray32x18": feature_bytes(image_or_features.get("gray32x18"), 32 * 18), "edge_density": float(image_or_features.get("edge_density", 0.0) or 0.0), "color_histogram": histogram_blob(image_or_features.get("color_histogram")), "score": float(score), "hunger": 1e-9 if hunger is None else float(hunger), "created": time.time()}
        with self.lock:
            self.exact_state_cache.append(entry)

    def _exact_cache_distance(self, image, item):
        try:
            dhash_a = str(image.get("dhash64") or "")
            dhash_b = str(item.get("dhash64") or "")
            hash_distance = bit_count(int(dhash_a, 16) ^ int(dhash_b, 16)) / 64.0 if dhash_a and dhash_b else 1.0
        except Exception:
            hash_distance = 1.0
        gray_a = feature_bytes(image.get("gray32x18"), 32 * 18)
        gray_b = feature_bytes(item.get("gray32x18"), 32 * 18)
        gray_distance = 1.0
        if gray_a and gray_b:
            gray_distance = sum(abs(a - b) for a, b in zip(gray_a, gray_b)) / (255.0 * len(gray_a))
        try:
            edge_distance = min(1.0, abs(float(image.get("edge_density", 0.0) or 0.0) - float(item.get("edge_density", 0.0) or 0.0)) * 2.0)
        except Exception:
            edge_distance = 1.0
        return max(0.0, min(1.0, 0.45 * hash_distance + 0.45 * gray_distance + 0.10 * edge_distance))

    def _inherit_recent_exact_score(self, image, score, meta):
        if not isinstance(image, dict) or not isinstance(meta, dict):
            return score, meta
        with self.lock:
            cache = list(self.exact_state_cache)
        best = None
        for item in cache:
            distance = self._exact_cache_distance(image, item)
            if best is None or distance < best[0]:
                best = (distance, item)
        if best is None or best[0] > 0.030:
            return score, meta
        inherited = dict(meta)
        inherited.update({"score_valid": False, "provisional": True, "recall_guard": True, "exact_or_approx": "provisional_cache", "retrieval_mode": "recent_exact_cache", "score_status": "provisional_cache", "score_valid_for_training": 0, "exact_cache_frame_id": best[1].get("frame_id"), "exact_cache_distance": float(best[0])})
        return float(best[1].get("score", score if score is not None else 0.0)), inherited

    def busy(self):
        with self.lock:
            return self.state != "idle"

    def current_state(self):
        with self.lock:
            return self.state

    def ensure_store(self):
        self.resources.set_storage_path(self.settings.data["storage_path"])
        self.resources.set_emulator_path(self.settings.data["emulator_path"])
        self.store.set_transaction_reserve(self.settings.data.get("transaction_reserve_bytes", 8 * 1024 * 1024))
        self.store.ensure(self.settings.data["storage_path"])
        self._flush_pending_critical_errors()

    def _remember_critical_exception(self, module, function, error, session_id=None, token=None, resource_state=None, payload=None):
        item = {"module": str(module), "function": str(function), "exception_type": type(error).__name__, "message": str(error), "session_id": session_id, "token": token, "resource_state": resource_state, "payload": dict(payload or {}), "created": time.time()}
        with self.lock:
            self.pending_critical_errors = ([item] + list(self.pending_critical_errors))[:20]
        try:
            if self.store.conn is not None:
                self.store.add_critical_exception(module, function, error, session_id=session_id, token=token, resource_state=resource_state, payload=payload)
        except Exception:
            pass

    def _flush_pending_critical_errors(self):
        with self.lock:
            pending = list(reversed(self.pending_critical_errors))
            self.pending_critical_errors = []
        for item in pending:
            try:
                self.store.add_critical_exception(item["module"], item["function"], RuntimeError(item["message"]), session_id=item.get("session_id"), token=item.get("token"), resource_state=item.get("resource_state"), payload=dict(item.get("payload") or {}, original_exception_type=item.get("exception_type"), original_created=item.get("created")))
            except Exception:
                with self.lock:
                    self.pending_critical_errors = ([item] + list(self.pending_critical_errors))[:20]
                break

    def _on_window_event(self):
        with self.lock:
            token = self.epoch
            active = self.state in ("learning", "training")
        if active:
            self.on_control_signal("window_validate", "窗口状态变化", token=token)

    def request_start_session(self, mode):
        self.on_control_signal("start_mode", str(mode))

    def on_control_signal(self, kind, reason, created=None, token=None):
        emergency = kind in ("stop", "esc")
        if emergency:
            with self.lock:
                if self.stop_requested.is_set() and self.emergency_stop_event.is_set():
                    return
                self.stop_requested.set()
                self.emergency_stop_reason = str(reason or "紧急停止")
                self.emergency_stop_token = token
                self.emergency_stop_event.set()
        try:
            self.control_queue.put_nowait({"kind": kind, "reason": reason, "created": created or time.time(), "token": token})
        except queue.Full:
            if emergency:
                return

    def _control_loop(self):
        while True:
            if self.emergency_stop_event.is_set():
                with self.lock:
                    reason = self.emergency_stop_reason or "紧急停止"
                    token = self.emergency_stop_token
                    self.emergency_stop_event.clear()
                    self.emergency_stop_reason = ""
                    self.emergency_stop_token = None
                self._perform_stop(reason, token)
                continue
            try:
                item = self.control_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                return
            kind = item.get("kind")
            if kind in ("stop", "esc"):
                self._perform_stop(item.get("reason") or "停止请求", item.get("token"))
            elif kind == "auto_sleep":
                self._begin_auto_sleep(item.get("token"))
            elif kind == "sleep_finish":
                self._finish_idle(item.get("token"), item.get("reason") or "睡眠任务结束", True)
            elif kind == "start_mode":
                mode = str(item.get("reason") or "")
                if not self.start_session(mode):
                    self.post_state("未进入模式，请检查目标窗口、客户区状态和资源预算。")
            elif kind == "window_validate":
                with self.lock:
                    token = self.epoch
                    active = self.state in ("learning", "training")
                if active and item.get("token") == token:
                    rect, reason = self._validate_bound_target(require_cursor=True, require_foreground=False)
                    if rect is None:
                        self._perform_stop("窗口事件触发客户区校验失败：" + reason, token)

    def _note_raw_mouse_loss(self, queue_name, event_type, button, wheel, created):
        with self.lock:
            session_id = self.session_id if self.state in ("learning", "training") else None
        if not session_id:
            return
        critical = event_type != "move" or bool(button) or bool(wheel)
        key = (session_id, queue_name)
        with self.raw_loss_lock:
            item = self.raw_mouse_losses.get(key)
            if item is None:
                self.raw_mouse_losses[key] = {"session_id": session_id, "queue": queue_name, "event_type": event_type, "started": created, "ended": created, "count": 1, "critical": critical, "recovered": False}
            else:
                item["ended"] = created
                item["count"] += 1
                item["critical"] = bool(item["critical"] or critical)
                if critical:
                    item["event_type"] = event_type

    def _mark_raw_queue_recovered(self, queue_name):
        with self.raw_loss_lock:
            for item in self.raw_mouse_losses.values():
                if item["queue"] == queue_name:
                    item["recovered"] = True

    def _flush_raw_mouse_losses(self, force=False):
        with self.raw_loss_lock:
            keys = [key for key, item in self.raw_mouse_losses.items() if force or item.get("recovered")]
            losses = [self.raw_mouse_losses.pop(key) for key in keys]
        for item in losses:
            rule = "原始鼠标队列丢失；来源队列={};事件类型={};{}".format(item["queue"], item["event_type"], "关键事件丢失" if item["critical"] else "普通移动事件压缩/丢弃")
            try:
                self.store.record_mouse_loss(item["session_id"], item["started"], item["ended"], item["count"], rule)
            except Exception:
                pass

    def _dispatch_raw_hook_mouse(self, raw):
        wparam, mouse_data, x, y, created, created_monotonic_ns, flags, extra_info = raw
        event_map = {WM_MOUSEMOVE: ("move", ""), WM_LBUTTONDOWN: ("button_down", "left"), WM_LBUTTONUP: ("button_up", "left"), WM_RBUTTONDOWN: ("button_down", "right"), WM_RBUTTONUP: ("button_up", "right"), WM_MOUSEWHEEL: ("wheel", "vertical"), WM_MOUSEHWHEEL: ("wheel", "horizontal"), WM_XBUTTONDOWN: ("button_down", "x"), WM_XBUTTONUP: ("button_up", "x")}
        if int(wparam) not in event_map:
            return
        event_type, button = event_map[int(wparam)]
        wheel = ctypes.c_short((int(mouse_data) >> 16) & 0xFFFF).value if int(wparam) in (WM_MOUSEWHEEL, WM_MOUSEHWHEEL) else 0
        self.enqueue_raw_mouse(event_type, button, wheel, int(x), int(y), float(created), int(created_monotonic_ns), int(flags), int(extra_info))

    def enqueue_raw_mouse(self, event_type, button, wheel, x, y, created, created_monotonic_ns, flags, extra_info):
        item = (event_type, button, wheel, x, y, created, created_monotonic_ns, flags, extra_info)
        critical = event_type != "move" or bool(button) or bool(wheel)
        target = self.raw_critical_queue if critical else self.raw_mouse_queue
        try:
            target.put_nowait(item)
            self._mark_raw_queue_recovered("critical" if critical else "move")
        except queue.Full:
            self.raw_mouse_drops += 1
            self._note_raw_mouse_loss("critical" if critical else "move", event_type, button, wheel, created)
            with self.lock:
                active = self.state in ("learning", "training") and bool(self.session_id)
            if active:
                self._flush_raw_mouse_losses(True)
                self.on_control_signal("stop", "鼠标事件无法进入固定容量队列，已安全结束会话")

    def _raw_mouse_loop(self):
        while not self.raw_mouse_stop.is_set() or not self.raw_hook_ring.empty() or not self.raw_critical_queue.empty() or not self.raw_mouse_queue.empty():
            overflow = self.raw_hook_ring.pop_overflow_count()
            if overflow:
                self.raw_mouse_drops += int(overflow)
                with self.lock:
                    active = self.state in ("learning", "training") and bool(self.session_id)
                    sid = self.session_id
                if active:
                    try:
                        self.store.mark_session_untrainable(sid, "低级鼠标 hook ring buffer 溢出 {} 次".format(overflow))
                        self.store.add_system_event(sid, "raw_hook_overflow", {"lost_count": int(overflow), "queue": "hook_ring"})
                    except Exception as error:
                        self._remember_critical_exception("DataStore", "raw_hook_overflow_log", error, session_id=sid, token=self.epoch, resource_state=self.resources.capture_snapshot().state)
                    self.on_control_signal("stop", "低级鼠标 hook ring buffer 溢出，已安全结束会话")
            self._flush_raw_mouse_losses()
            try:
                raw = self.raw_hook_ring.get(timeout=0.002)
                self._dispatch_raw_hook_mouse(raw)
                continue
            except queue.Empty:
                pass
            except Exception as error:
                try:
                    self.store.add_critical_exception("TrainingController", "_dispatch_raw_hook_mouse", error, session_id=self.session_id, token=self.epoch, resource_state=self.resources.capture_snapshot().state)
                except Exception:
                    pass
                self.on_control_signal("stop", "鼠标 hook 原始事件解析失败:" + str(error))
                continue
            item = None
            try:
                item = self.raw_critical_queue.get_nowait()
            except queue.Empty:
                try:
                    item = self.raw_mouse_queue.get(timeout=0.05)
                except queue.Empty:
                    pass
            if item is None:
                continue
            self._mark_raw_queue_recovered("critical" if (item[0] != "move" or bool(item[1]) or bool(item[2])) else "move")
            try:
                self.on_mouse(*item)
            except Exception as error:
                try:
                    self.store.add_critical_exception("TrainingController", "on_mouse", error, session_id=self.session_id, token=self.epoch, resource_state=self.resources.capture_snapshot().state)
                except Exception:
                    pass
                self.on_control_signal("stop", "鼠标事件解析失败:" + str(error))

    def _degraded_mouse_records(self, records):
        raise RuntimeError("鼠标轨迹禁止静默抽样降级")

    def _mouse_writer(self):
        pending = []
        pending_segments = []
        last_write = time.monotonic()
        last_replay = 0.0
        while not self.writer_stop.is_set() or not self.mouse_queue.empty() or not self.mouse_segment_queue.empty() or pending or pending_segments:
            oldest = pending[0].get("created", time.time()) if pending else time.time()
            self.resources.update_queue(self.mouse_queue.qsize() + self.mouse_segment_queue.qsize() + len(pending) + len(pending_segments), max(0.0, time.time() - float(oldest)))
            try:
                pending.append(self.mouse_queue.get(timeout=0.05))
                self.writer_busy.set()
            except queue.Empty:
                pass
            for _ in range(64):
                try:
                    pending_segments.append(self.mouse_segment_queue.get_nowait())
                    self.writer_busy.set()
                except queue.Empty:
                    break
            write_budget = self.resources.acquire("capture")
            due = time.monotonic() - last_write >= max(0.15, write_budget.next_interval) or self.writer_stop.is_set()
            batch_ready = len(pending) >= max(8, write_budget.database_batch_size)
            segment_ready = len(pending_segments) >= 8
            if (pending or pending_segments) and (batch_ready or segment_ready or due):
                self.writer_busy.set()
                try:
                    def db_task():
                        with self.resources.budget_slot("sqlite"):
                            if pending_segments:
                                for segment in list(pending_segments):
                                    self.store.record_mouse_compression(segment)
                            if pending:
                                self.store.save_mouse_batch(pending)
                    self.db_writer.call(db_task, timeout=30.0)
                    pending_segments = []
                    pending = []
                    self.mouse_sqlite_degraded_until = 0.0
                    now = time.monotonic()
                    if now - last_replay >= 1.0:
                        self.persistence_recovery.replay_mouse(write_budget.database_batch_size)
                        last_replay = now
                except Exception as error:
                    sessions = {record.get("session_id") for record in pending if record.get("session_id")}
                    sessions.update(segment.get("session_id") for segment in pending_segments if segment.get("session_id"))
                    self.mouse_sqlite_degraded_until = time.monotonic() + 5.0
                    for sid in sessions:
                        try:
                            self.store.mark_session_untrainable(sid, "鼠标轨迹无法无损落盘")
                            self.store.add_critical_exception("DataStore", "storage_writer_mouse", error, session_id=sid, token=self.epoch, resource_state=self.resources.capture_snapshot().state, payload={"stage": "mouse_sqlite_batch", "mouse_queue_length": self.mouse_queue.qsize(), "mouse_segment_queue_length": self.mouse_segment_queue.qsize(), "pending_records": len(pending), "pending_segments": len(pending_segments)})
                        except Exception:
                            pass
                    self.on_control_signal("stop", "鼠标轨迹无法无损落盘，已停止当前会话并标记不可训练")
                    time.sleep(0.20)
                finally:
                    last_write = time.monotonic()
                    if not pending and not pending_segments:
                        self.writer_busy.clear()
            elif not pending and not pending_segments:
                self.writer_busy.clear()

    def flush_mouse_records(self, timeout=3.0):
        deadline = time.monotonic() + max(0.1, timeout)
        while time.monotonic() < deadline:
            if self.mouse_queue.empty() and self.mouse_segment_queue.empty() and not self.writer_busy.is_set():
                return True
            time.sleep(0.04)
        return self.mouse_queue.empty() and self.mouse_segment_queue.empty() and not self.writer_busy.is_set()

    def classify_mouse_source(self, event_type, button, wheel, x, y, flags, extra_info, created_monotonic_ns):
        injected = bool(flags & (LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED))
        matched = self.ai_input_tracker.consume(extra_info, event_type, button, wheel, x, y, created_monotonic_ns)
        probability = matched.get("behavior_probability") if isinstance(matched, dict) else None
        if matched:
            return "ai", probability
        if injected:
            return "external_injected", None
        return "user", None

    def _append_move_compression(self, session_id, source, last_kept, x, y, created, created_ns, speed, direction, rect):
        key = (session_id, source)
        item = self.move_segments.get(key)
        if item is None:
            base_x, base_y, base_ns, base_created = int(last_kept[0]), int(last_kept[1]), int(last_kept[2]), float(last_kept[3])
            dt, dx, dy = max(0, int(created_ns) - base_ns), int(x) - base_x, int(y) - base_y
            item = {"session_id": session_id, "source": source, "started": base_created, "ended": created, "started_ns": base_ns, "ended_ns": int(created_ns), "start_x": base_x, "start_y": base_y, "end_x": int(x), "end_y": int(y), "count": 1, "max_speed": float(speed), "speed_sum": float(speed), "speed_count": 1, "average_speed": float(speed), "last_direction": float(direction), "direction_change_count": 0, "click_pre_dwell_ms": 0.0, "crossed_client_boundary": not point_inside((int(x), int(y)), rect), "path_length": math.hypot(dx, dy), "points": [(dt, dx, dy)], "client_rect": tuple(int(value) for value in rect), "rule": "varint/zigzag 无损相对时间与坐标轨迹压缩"}
            self.move_segments[key] = item
        else:
            prev_x, prev_y, prev_ns = int(item["end_x"]), int(item["end_y"]), int(item["ended_ns"])
            dt, dx, dy = max(0, int(created_ns) - prev_ns), int(x) - prev_x, int(y) - prev_y
            item["ended"] = created
            item["ended_ns"] = int(created_ns)
            item["path_length"] += math.hypot(dx, dy)
            item["end_x"] = int(x)
            item["end_y"] = int(y)
            item["count"] += 1
            previous_direction = float(item.get("last_direction", direction))
            turn = abs(math.atan2(math.sin(float(direction) - previous_direction), math.cos(float(direction) - previous_direction)))
            if turn >= 0.65:
                item["direction_change_count"] = int(item.get("direction_change_count", 0)) + 1
            item["last_direction"] = float(direction)
            item["max_speed"] = max(float(item["max_speed"]), float(speed))
            item["speed_sum"] = float(item.get("speed_sum", 0.0)) + float(speed)
            item["speed_count"] = int(item.get("speed_count", 0)) + 1
            item["average_speed"] = float(item.get("speed_sum", 0.0)) / max(1, int(item.get("speed_count", 1)))
            item["crossed_client_boundary"] = bool(item.get("crossed_client_boundary")) or not point_inside((int(x), int(y)), rect)
            item["points"].append((dt, dx, dy))

    def _flush_move_compression(self, session_id=None, source=None):
        keys = [key for key in self.move_segments if (session_id is None or key[0] == session_id) and (source is None or key[1] == source)]
        for key in keys:
            segment = self.move_segments.pop(key, None)
            if not segment:
                continue
            try:
                self.store.journal_mouse_segment(segment, durable=True)
            except Exception as error:
                try:
                    self.store.add_critical_exception("DataStore", "journal_mouse_segment", error, session_id=segment.get("session_id"), token=self.epoch, resource_state=self.resources.capture_snapshot().state, payload={"stage": "mouse_segment_journal"})
                except Exception:
                    pass
                return False
            try:
                self.mouse_segment_queue.put_nowait(segment)
            except queue.Full:
                try:
                    self.store.mark_session_untrainable(segment.get("session_id"), "鼠标轨迹段队列已满，无法无损落盘")
                    self.store.add_system_event(segment.get("session_id"), "mouse_segment_queue_full", {"queue": "mouse_segment_queue", "journal_id": segment.get("accepted_journal_id"), "count": int(segment.get("count", 0) or 0)})
                except Exception:
                    pass
                self.on_control_signal("stop", "鼠标轨迹段队列已满，已停止当前会话并标记不可训练")
                return False
        return True

    def on_mouse(self, event_type, button, wheel, x, y, created, created_monotonic_ns, flags, extra_info):
        with self.lock:
            current_state = self.state
            if current_state not in ("learning", "training") or not self.session_id or not self.target_rect:
                return
            session_id = self.session_id
            rect = self.target_rect
            training = current_state == "training"
        if training:
            source, behavior_probability = self.classify_mouse_source(event_type, button, wheel, x, y, flags, extra_info, created_monotonic_ns)
        else:
            source, behavior_probability = "user", None
        if source == "user":
            self.resources.update_user_input()
        if training and source != "ai":
            try:
                self.store.mark_session_untrainable(session_id, "训练模式检测到非 AI 鼠标来源：" + str(source))
                self.store.add_system_event(session_id, "training_mouse_source_interrupted", {"source": source, "event_type": event_type, "x": int(x), "y": int(y), "time": time.time()})
            except Exception:
                pass
        with self.lock:
            if self.state != current_state or self.session_id != session_id:
                return
            outside = not point_inside((x, y), rect)
            previous = self.last_mouse_by_source.get(source)
            critical = event_type != "move" or button or wheel
        if outside:
            self.on_control_signal("stop", "鼠标已离开目标窗口客户区")
            return
        input_budget = self.resources.capture_snapshot()
        if input_budget.must_pause:
            self.mouse_sqlite_degraded_until = max(self.mouse_sqlite_degraded_until, time.monotonic() + 2.0)
        dx = dy = direction = speed = 0.0
        if previous is not None:
            dt = max(0.000001, (created_monotonic_ns - previous[2]) / 1_000_000_000.0)
            dx = float(x - previous[0])
            dy = float(y - previous[1])
            direction = math.atan2(dy, dx) if dx or dy else 0.0
            speed = math.hypot(dx, dy) / dt
        with self.lock:
            kept_map = self.last_move_kept if isinstance(self.last_move_kept, dict) else {}
            last_kept = kept_map.get(source)
            direction_turn = last_kept is not None and previous is not None and abs(math.atan2(math.sin(direction - last_kept[4]), math.cos(direction - last_kept[4]))) >= 0.65
            if not critical and last_kept is not None and (created_monotonic_ns - last_kept[2]) < 10_000_000 and abs(x - last_kept[0]) < 3 and abs(y - last_kept[1]) < 3 and not direction_turn:
                self._append_move_compression(session_id, source, last_kept, x, y, created, created_monotonic_ns, speed, direction, rect)
                self.last_mouse_by_source[source] = (x, y, created_monotonic_ns)
                return
        if critical:
            with self.lock:
                segment = self.move_segments.get((session_id, source))
                if segment is not None:
                    segment["click_pre_dwell_ms"] = max(float(segment.get("click_pre_dwell_ms", 0.0)), max(0.0, (int(created_monotonic_ns) - int(segment.get("ended_ns", created_monotonic_ns))) / 1_000_000.0))
        if not self._flush_move_compression(session_id, source):
            return
        with self.lock:
            if not critical:
                kept_map[source] = (x, y, created_monotonic_ns, created, direction)
                self.last_move_kept = kept_map
            self.last_mouse_by_source[source] = (x, y, created_monotonic_ns)
            self.last_mouse_activity_ns = max(self.last_mouse_activity_ns, int(created_monotonic_ns))
            self.mouse_count += 1
            before_frame_id = self.latest_frame_id
        width = max(1, rect[2] - rect[0])
        height = max(1, rect[3] - rect[1])
        record = {"session_id": session_id, "created": created, "created_monotonic_ns": int(created_monotonic_ns), "source": source, "event_type": event_type, "button": button, "wheel": wheel, "x": x, "y": y, "relative_x": (x - rect[0]) / width, "relative_y": (y - rect[1]) / height, "dx": dx, "dy": dy, "direction": direction, "speed": speed, "behavior_probability": behavior_probability, "before_frame_id": before_frame_id, "after_frame_id": None}
        try:
            self.store.journal_mouse_record(record, durable=critical)
            self.mouse_queue.put_nowait(record)
            if ((not training and source == "user") or (training and source == "ai")) and event_type in ("button_up", "wheel", "move"):
                with self.lock:
                    self.session_valid_actions += 1
            if ((not training and source == "user") or (training and source == "ai")) and event_type in ("button_up", "wheel"):
                self.request_immediate_capture("mouse_action")
        except queue.Full:
            try:
                self.store.mark_session_untrainable(session_id, "鼠标事件队列已满，无法无损落盘")
                self.store.add_system_event(session_id, "mouse_queue_full", {"queue": "mouse_queue", "journal_id": record.get("accepted_journal_id"), "critical": bool(critical), "mouse_queue_length": self.mouse_queue.qsize()})
            except Exception:
                pass
            self.on_control_signal("stop", "鼠标事件队列已满，已停止当前会话并标记不可训练")
            return
        if input_budget.must_pause:
            self.mouse_sqlite_degraded_until = max(self.mouse_sqlite_degraded_until, time.monotonic() + 2.0)

    def _is_current(self, token, states=None):
        with self.lock:
            if token != self.epoch or self.cancel_event.is_set():
                return False
            return states is None or self.state in states

    def _find_valid_target(self, cursor_required=False):
        candidates = find_emulator_candidates(self.settings.data["emulator_path"])
        if not candidates:
            return None, None, "未检测到可选择的目标窗口。"
        selected_pid = int(self.settings.data.get("emulator_pid", 0) or 0)
        selected_title = str(self.settings.data.get("emulator_title", "") or "")
        if len(candidates) > 1 and not selected_pid and not selected_title:
            return None, None, "检测到多个候选目标窗口。请先在控制面板中明确选择窗口标题和 PID。"
        if selected_pid or selected_title:
            candidates = [item for item in candidates if (not selected_pid or item["pid"] == selected_pid) and (not selected_title or item["title"] == selected_title)]
            if len(candidates) != 1:
                return None, None, "已选择的目标窗口不存在、标题已变化或不再唯一。请重新选择窗口。"
        elif len(candidates) == 1:
            pass
        else:
            return None, None, "多个目标窗口未明确选择。"
        chosen = candidates[0]
        rect = valid_client(chosen["hwnd"], cursor_required)
        if rect is None:
            return None, None, "目标窗口客户区异常：" + (getattr(valid_client, "last_reason", "不可见、被遮挡、最小化或未完全位于物理显示器范围内"))
        return chosen["hwnd"], rect, ""

    def _place_cursor_before_entry(self, hwnd, rect):
        current = cursor_position()
        if not point_inside(current, rect):
            if not PLATFORM_BACKEND.move_cursor_into_client(hwnd, rect):
                return False
            time.sleep(0.03)
        return valid_client(hwnd, True) is not None

    def _precheck_client_recording(self, hwnd, rect, budget):
        deadline = time.monotonic() + CLIENT_RECORDING_FAILURE_SECONDS if CLIENT_RECORDING_FAILURE_SECONDS > 0 else time.monotonic()
        attempts = max(1, CLIENT_RECORDING_FAILURE_LIMIT if CLIENT_RECORDING_FAILURE_LIMIT > 1 else 6)
        last_reason = ""
        for _ in range(attempts):
            try:
                image = capture_client(hwnd, budget.max_capture_resolution[0], budget.max_capture_resolution[1], False)
                if image is None:
                    last_reason = str(getattr(capture_client, "last_failure_reason", "capture_client 返回 None") or "capture_client 返回 None")
                elif not image.get("capture_complete", 1):
                    last_reason = str(image.get("capture_failure_reason") or "capture_complete != 1")
                else:
                    featured = extract_frame_features(image)
                    png = encode_png_bgra(int(featured["width"]), int(featured["height"]), featured.get("bgra"), 1) if featured.get("bgra") is not None else encode_png(int(featured["width"]), int(featured["height"]), featured.get("rgb"), 1)
                    if png:
                        return True, ""
                    last_reason = "PNG 缺失"
            except Exception as error:
                last_reason = str(error)
            if time.monotonic() >= deadline and CLIENT_RECORDING_FAILURE_SECONDS > 0:
                break
            time.sleep(0.05)
        return False, "客户区画面无法被完整记录：" + (last_reason or "预检查失败")

    def _prepare_storage_and_recovery(self, mode, automatic=False):
        with self.lock:
            permitted = self.state == "idle" or (automatic and self.state == "sleep")
            if not permitted:
                if not automatic:
                    self.emit("notice", "当前不是空闲状态。")
                return False
            recovery_pending = bool(self.recovery_pending)
        if recovery_pending:
            if not self._wait_detached_pipeline_contexts(10.0):
                self.post_state("空闲；已接受数据仍在排空，暂不允许启动新会话。")
                self.emit("notice", "已接受数据仍在排空，暂不允许启动新会话。")
                return False
            try:
                ok, detail = self.store.recover_after_forced_stop()
            except Exception as error:
                ok, detail = False, str(error)
            if not ok:
                self.post_state("空闲；数据恢复待处理，禁止启动新会话：" + str(detail))
                self.emit("notice", "数据恢复待处理，禁止启动新会话：" + str(detail))
                return False
            with self.lock:
                self.recovery_pending = False
                self.recovery_reason = ""
        try:
            self.ensure_store()
        except Exception as error:
            self._remember_critical_exception("DataStore", "ensure_store", error, resource_state=self.resources.capture_snapshot().state)
            self.emit("notice", "无法创建存储路径：" + str(error))
            return False
        if mode in ("learning", "training") and not bool(getattr(self.platform_backend, "learning_training_enabled", False)):
            self.emit("notice", self.platform_backend.mode_unavailable_reason())
            return False
        if mode in ("learning", "training"):
            try:
                capacity = self.store.capacity_status() if self.store.conn else {"tier": 0, "blocked": False}
                if capacity.get("blocked") or int(capacity.get("tier", 0) or 0) >= 95:
                    self.emit("notice", "经验池已达 95%/100% 水位，禁止进入新学习/训练；请先进入睡眠模式清理。")
                    return False
                self.store.preflight_persistence_check(mode)
            except Exception as error:
                self._remember_critical_exception("DataStore", "preflight_persistence_check", error, resource_state=self.resources.capture_snapshot().state)
                self.emit("notice", str(error))
                return False
        return True

    def _prepare_hooks(self, mode):
        if not self.hook.start():
            try:
                self.store.add_critical_exception("MouseHook", "start", RuntimeError(self.hook.error or "鼠标钩子未启动"), resource_state=self.resources.capture_snapshot().state)
            except Exception:
                pass
            self.emit("notice", self.hook.error or "鼠标钩子未启动，禁止进入模式。")
            return False
        if not self.keyboard_hook.start():
            try:
                self.store.add_critical_exception("KeyboardHook", "start", RuntimeError(self.keyboard_hook.error or "键盘钩子未启动"), resource_state=self.resources.capture_snapshot().state)
            except Exception:
                pass
            self.emit("notice", self.keyboard_hook.error or "键盘钩子未启动，禁止进入模式。")
            self.hook.stop()
            return False
        return True

    def _prepare_target_window(self, mode):
        entry_budget = self.resources.acquire("capture")
        if entry_budget.must_pause:
            self.emit("notice", "资源恢复观察未完成，暂不允许进入模式：" + (entry_budget.pause_reason or "资源红线"))
            return None
        hwnd, rect, reason = self._find_valid_target(False)
        if hwnd is None:
            self.emit("notice", reason)
            return None
        if mode in ("learning", "training") and not activate_root_window(hwnd):
            self.emit("notice", "进入{}模式前无法将目标窗口置为前台：{}".format("学习" if mode == "learning" else "训练", getattr(activate_root_window, "last_reason", "未知")))
            return None
        time.sleep(0.08)
        if mode in ("learning", "training") and not foreground_root_matches(hwnd):
            self.emit("notice", "进入{}模式前前台校验失败：目标窗口不是前台窗口".format("学习" if mode == "learning" else "训练"))
            return None
        if not self._place_cursor_before_entry(hwnd, rect):
            self.emit("notice", "进入模式前无法确认鼠标与目标窗口客户区状态。")
            return None
        if not entry_capture_pixel_validation(hwnd):
            self.emit("notice", "进入模式前截图像素校验失败：" + getattr(entry_capture_pixel_validation, "last_reason", "未知"))
            return None
        rect = valid_client(hwnd, True)
        if rect is None:
            self.emit("notice", "目标窗口客户区状态异常：" + getattr(valid_client, "last_reason", "未知"))
            return None
        ok, precheck_reason = self._precheck_client_recording(hwnd, rect, entry_budget)
        if not ok:
            self.emit("notice", precheck_reason)
            return None
        return {"hwnd": hwnd, "rect": rect}

    def _create_mode_session(self, mode, automatic, target):
        hwnd = target["hwnd"]
        rect = target["rect"]
        try:
            session_id = self.store.create_session(mode)
        except Exception as error:
            self.emit("notice", "无法创建会话记录：" + str(error))
            return None
        target_root = root_window(hwnd)
        if IS_WINDOWS:
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(target_root, ctypes.byref(pid))
            target_pid = int(pid.value)
        else:
            selected = [item for item in PLATFORM_BACKEND.list_windows() if int(item.get("hwnd", 0) or 0) == int(hwnd or 0)]
            target_pid = int((selected[0].get("pid", 0) if selected else 0) or 0)
        target_process_path = normalized_windows_path(process_full_path(target_pid)) if IS_WINDOWS else os.path.realpath(process_full_path(target_pid))
        try:
            self.store.add_system_event(session_id, "mode_enter", {"mode": mode, "automatic": automatic, "time": time.time(), "client_rect": rect, "target_pid": target_pid, "resource": self.resources.sample(), "platform": getattr(self.platform_backend, "name", "unknown")})
        except Exception as error:
            try:
                self.store.close_session(session_id, "进入模式失败：mode_enter 写入失败：" + str(error))
            except Exception:
                pass
            self.emit("notice", "无法写入模式进入事件：" + str(error))
            return None
        return {"session_id": session_id, "target_root": target_root, "target_pid": target_pid, "target_process_path": target_process_path, "hwnd": hwnd, "rect": rect}

    def _start_runtime_threads(self, mode, context_data, token=None):
        session_id = context_data["session_id"]
        hwnd = context_data["hwnd"]
        rect = context_data["rect"]
        automatic = bool(context_data.get("automatic"))
        with self.lock:
            self.epoch += 1
            token = self.epoch
            self.cancel_event = threading.Event()
            self.target_hwnd = hwnd
            self.target_root = context_data["target_root"]
            self.target_pid = context_data["target_pid"]
            self.target_process_path = context_data["target_process_path"]
            self.target_rect = rect
            self.session_id = session_id
            self.session_mode = mode
            self.session_started = time.monotonic()
            self.hunger_anchor_ns = time.monotonic_ns()
            self.last_observation = self.session_started
            self.capture_failures = 0
            self.capture_failure_started = 0.0
            self.last_valid_score = None
            self.frame_scores = []
            self.frame_count = 0
            self.mouse_count = 0
            self.session_valid_frames = 0
            self.session_valid_actions = 0
            self.latest_frame_id = None
            self.last_sleep_fingerprint_check = 0.0
            self.current_training_fingerprint = ""
            self.gdi_failures = 0
            self.gdi_static_count = 0
            self.last_feature_hash = ""
            self.stable_feature_frames = 0
            self.last_gdi_hash = ""
            self.last_gdi_hash_input_ns = 0
            self.fallback_capture_pending = False
            self.last_mouse_activity_ns = 0
            self.raw_mouse_drops = 0
            self.last_mouse_by_source = {"ai": None, "user": None, "external_injected": None}
            self.ai_step = 0
            self.training_auto_sleep_count = 0 if mode == "training" and not automatic else self.training_auto_sleep_count
            model = self.store.best_model() if mode == "training" else None
            can_control_model = bool(isinstance(model, dict) and (model.get("champion") is True or model.get("control_enabled") is True))
            plan = model.get("q_actions", model.get("hotspots", [])) if isinstance(model, dict) and can_control_model else []
            allowed_source = model.get("allowed_action_types") if isinstance(model, dict) else None
            allowed_actions = set(str(item) for item in allowed_source) if isinstance(allowed_source, (list, tuple, set)) else None
            raw_plan = [item for item in plan if isinstance(item, dict)] if isinstance(plan, list) else []
            self.ai_plan = [item for item in raw_plan if allowed_actions is None or str(item.get("action_type", "移动")) in allowed_actions]
            if isinstance(model, dict) and model.get("onnx_path") and model.get("onnx_policy_verified") and self.store.models is not None:
                onnx_path = Path(str(model.get("onnx_path")))
                if not onnx_path.is_absolute():
                    onnx_path = self.store.models / onnx_path
                self.resources.backend.try_enable_gpu_model(onnx_path, self.resources.sample())
            self.ai_input_tracker.begin_session()
            self.action_limits = {}
            self.action_benefit_fuse = {}
            self.last_move_kept = {}
            self.move_segments = {}
            self.pipeline_losses = {}
            self.recent_ai_actions = []
            self.stop_requested.clear()
            available_memory = max(1, int(self.resources.sample().get("avail_memory", 0) or 0))
            byte_capacity = max(1 * 1024 * 1024, min(64 * 1024 * 1024, int(available_memory * 0.02)))
            if self.resources.capture_snapshot().state != "正常":
                byte_capacity = max(1 * 1024 * 1024, byte_capacity // 4)
            pipeline_context = PipelineContext(token=token, session_id=session_id, byte_budget=ByteBudgetSemaphore(byte_capacity))
            self.pipeline_context = pipeline_context
            self.resources.set_emulator_pid(self.target_pid)
            self.capture_queue = pipeline_context.capture_queue
            self.feature_queue = pipeline_context.feature_queue
            self.persist_queue = pipeline_context.persist_queue
            self.pipeline_stop = pipeline_context.stop_event
            event = "idle_click_training" if mode == "training" and not automatic else "auto_sleep_task2_done_resume_training" if mode == "training" and automatic else "idle_click_learning"
            transition_ok = self._transition_state_locked(event, "sleep" if automatic else "idle", mode, "start_session", token)
        if not transition_ok:
            try:
                self.store.close_session(session_id, "进入模式失败：状态切换被拒绝")
            except Exception:
                pass
            return False
        self.window_guard.start(self.target_hwnd)
        detail = "已进入" + ("学习模式" if mode == "learning" else ("训练模式；无已验证模型，安全观察且不执行点击、右键或滚轮" if not self.ai_plan else "训练模式"))
        self.post_state(detail)
        pipeline = [threading.Thread(target=self._pipeline_feature_loop, args=(pipeline_context,), name="FeatureScore"), threading.Thread(target=self._pipeline_encode_loop, args=(pipeline_context,), name="PngEncode"), threading.Thread(target=self._pipeline_persist_loop, args=(pipeline_context,), name="FramePersist"), threading.Thread(target=self._pipeline_exact_score_loop, args=(pipeline_context,), name="ExactScore")]
        threads = pipeline + [threading.Thread(target=self._capture_loop, args=(pipeline_context,), name="CaptureLoop"), threading.Thread(target=self._monitor_loop, args=(token,), name="SessionMonitor")]
        if mode == "training":
            threads.append(threading.Thread(target=self._ai_loop, args=(token,), name="AIControl"))
        pipeline_context.threads = pipeline
        with self.lock:
            self.pipeline_threads = pipeline
            self.capture_threads = threads
            self.worker_threads = [thread for thread in self.worker_threads if thread.is_alive()] + threads
        for thread in threads:
            thread.start()
        return True

    def start_session(self, mode, automatic=False):
        if mode not in ("learning", "training"):
            self.emit("notice", "未知模式：" + str(mode))
            return False
        if not self._prepare_storage_and_recovery(mode, automatic):
            return False
        if not self._prepare_hooks(mode):
            return False
        target = None
        session_context = None
        try:
            target = self._prepare_target_window(mode)
            if target is None:
                return False
            session_context = self._create_mode_session(mode, automatic, target)
            if session_context is None:
                return False
            session_context["automatic"] = bool(automatic)
            return self.capture_pipeline.start(mode, session_context)
        except Exception as error:
            self._remember_critical_exception("Controller", "start_session", error, resource_state=self.resources.capture_snapshot().state)
            self.emit("notice", "进入模式失败：" + str(error))
            return False
        finally:
            with self.lock:
                active = self.state in ("learning", "training") and self.session_id == (session_context or {}).get("session_id")
            if not active:
                self.hook.stop()
                self.keyboard_hook.stop()

    def _drop_pipeline(self, session_id, stage, reason):
        now=time.time(); key=(session_id,stage,reason); item=self.pipeline_losses.get(key)
        if item is None: self.pipeline_losses[key]=[now,now,1]
        else: item[1]=now;item[2]+=1
        if now-self.pipeline_last_emit>=2.0:
            self._flush_pipeline_losses(); self.pipeline_last_emit=now

    def _flush_pipeline_losses(self):
        losses=list(self.pipeline_losses.items()); self.pipeline_losses={}
        for (sid,stage,reason),(started,ended,count) in losses:
            try:self.store.record_pipeline_loss(sid,started,ended,count,stage,reason)
            except Exception as error:self.store.record_exception_event(sid, "pipeline_loss_record_failed", error, {"stage": stage, "reason": reason, "lost_count": count})

    def _pipeline_byte_capacity(self):
        sample = self.resources.sample()
        available = max(1, int(sample.get("avail_memory", 0) or 0))
        capacity = max(1 * 1024 * 1024, min(64 * 1024 * 1024, int(available * 0.02)))
        if float(sample.get("memory_p95", sample.get("memory", 0.0)) or 0.0) >= 88.0 or float(sample.get("pipeline_queue_ratio", 0.0) or 0.0) >= 0.70:
            capacity = max(1 * 1024 * 1024, capacity // 4)
        return capacity

    def _release_packet_budget(self, packet, context=None):
        context = context or self.pipeline_context
        if context is None or not isinstance(packet, dict):
            return
        amount = int(packet.pop("_byte_reservation", 0) or 0)
        if amount:
            context.byte_budget.release(amount)

    def _resize_packet_budget(self, packet, context):
        if context is None or not isinstance(packet, dict):
            return True
        wanted = packet_byte_cost(packet)
        previous = int(packet.get("_byte_reservation", 0) or 0)
        if wanted <= previous:
            context.byte_budget.release(previous - wanted)
            packet["_byte_reservation"] = wanted
            return True
        if context.byte_budget.acquire(wanted - previous, timeout=0.0):
            packet["_byte_reservation"] = wanted
            return True
        return False

    def _put_value_packet(self, target_queue, packet, stage, context=None):
        context = context or self.pipeline_context
        timeout = float(getattr(context, "queue_wait_timeout_seconds", 1.0) if context is not None else 1.0)
        if context is not None:
            context.byte_budget.resize(self._pipeline_byte_capacity())
            if not self._resize_packet_budget(packet, context):
                self._drop_pipeline(packet.get("session_id"), stage, "在途字节预算不足")
                if stage == "capture":
                    return False
                self._release_packet_budget(packet, context)
                if context.accepting:
                    self.on_control_signal("stop", "{} 在途字节预算不足，已安全结束会话".format(stage), token=packet.get("token"))
                return False
        try:
            target_queue.put(packet, timeout=max(0.05, timeout))
            return True
        except queue.Full:
            self._release_packet_budget(packet, context)
            session_id = packet.get("session_id")
            reason = "{} 固定容量队列满超过 {:.1f} 秒".format(stage, timeout)
            self._drop_pipeline(session_id, stage, reason)
            self._flush_pipeline_losses()
            if context is not None and context.accepting:
                self._critical_recording_failure(session_id, stage, reason, token=packet.get("token"), payload={"queue_stage": stage, "queue_wait_timeout_seconds": timeout})
            return False

    def _pipeline_age(self, context=None):
        context = context or self.pipeline_context
        if context is None:
            return 0.0
        ages = []
        for q in (context.capture_queue, context.feature_queue, context.persist_queue):
            try:
                first = q.queue[0]
                ages.append(max(0.0, time.monotonic() - float(first.get("queued_monotonic", time.monotonic()))))
            except Exception:
                pass
        return max(ages or [0.0])

    def _packet_current(self, context, packet):
        if context is None or context.closed or not isinstance(packet, dict):
            return False
        if packet.get("token") != context.token or packet.get("session_id") != context.session_id:
            return False
        return bool(context.accepting or context.draining)

    def request_immediate_capture(self, reason=""):
        with self.lock:
            context = self.pipeline_context
            active = self.state in ("learning", "training") and context is not None and context.accepting and not context.stop_event.is_set()
        if not active:
            return False
        budget = self.resources.capture_snapshot()
        if not budget.allowed or budget.must_pause or float(getattr(budget, "queue_fill_ratio", 0.0) or 0.0) >= 0.85:
            return False
        now = time.monotonic()
        with self.lock:
            if now - self.last_immediate_capture < 0.10:
                return False
            self.last_immediate_capture = now
        self.immediate_capture_event.set()
        return True

    def _capture_loop(self, context):
        token = context.token
        try:
            while context.accepting and self._is_current(token, ("learning", "training")) and not context.stop_event.is_set():
                capacity = self.store.capacity_status() if self.store.conn else {"blocked": False}
                with self.lock:
                    mode = self.session_mode
                    session_id = self.session_id
                if capacity.get("blocked"):
                    self._stop_for_client_recording_budget(token, "capacity_blocked", capacity)
                    return
                budget = self.resources.acquire("capture")
                self._record_resource_decisions(context.session_id)
                hard_gate_reason = self._pipeline_hard_gate_reason(context)
                if hard_gate_reason:
                    self._stop_for_client_recording_budget(token, "pipeline_hard_gate_capture", {"reason": hard_gate_reason})
                    return
                if float(getattr(budget, "queue_fill_ratio", 0.0) or 0.0) >= 0.90 or time.monotonic() < float(self.frame_replay_pause_until or 0.0):
                    time.sleep(max(0.20, float(budget.next_interval)))
                    continue
                capacity_tier = int(capacity.get("tier", 0) or 0)
                if capacity_tier >= 95:
                    if mode == "training":
                        self.on_control_signal("auto_sleep", "经验池 95% 以上，暂停新采集并进入睡眠清理", token=token)
                    else:
                        self.request_idle("经验池 95% 以上，已暂停学习采集；请进入睡眠模式清理", token)
                    return
                if capacity_tier >= 80:
                    budget.next_interval = max(float(budget.next_interval), 0.40 if capacity_tier < 90 else 1.20)
                    budget.max_capture_resolution = (320, 180) if capacity_tier >= 90 else (426, 240)
                    if capacity_tier >= 80:
                        now_notice = time.monotonic()
                        with self.lock:
                            should_notice = capacity_tier > self.last_capacity_notice_tier or now_notice - self.last_capacity_notice_at >= 60.0
                            if should_notice:
                                self.last_capacity_notice_at = now_notice
                                self.last_capacity_notice_tier = capacity_tier
                        if should_notice:
                            self.emit("notice", "经验池已达到 {}% 水位，已降低 PNG 保存率并优先保留 feature；90% 后进一步降采样，95% 暂停新采集。".format(capacity_tier))
                if not budget.allowed:
                    if budget.must_pause:
                        self._stop_for_client_recording_budget(token, "resource_redline_capture", {"pause_reason": budget.pause_reason or "资源红线"})
                        return
                    time.sleep(max(0.05, budget.next_interval))
                    continue
                with self.lock:
                    self.capture_interval_seconds = float(budget.next_interval)
                if int(capacity.get("tier", 0) or 0) >= 95 and mode == "training":
                    self.on_control_signal("auto_sleep", "经验池 95% 以上，优先进入睡眠清理", token=token)
                rect, reason = self._validate_bound_target(require_cursor=True, require_foreground=False)
                if rect is None:
                    self.request_idle("绑定目标窗口校验失败：" + reason, token)
                    return
                with self.lock:
                    hwnd = self.target_hwnd
                    session_id = self.session_id
                    mode = self.session_mode
                    self.target_rect = rect
                now_observation = time.monotonic()
                observation_due = False
                with self.lock:
                    if now_observation - self.last_observation >= 5.0:
                        self.last_observation = now_observation
                        observation_due = True
                if session_id and observation_due:
                    self.store.add_system_event(session_id, "client_observation", {"mode": mode, "client_rect": rect, "cursor": cursor_position(), "resource": self.resources.sample()})
                    overlay = getattr(client_unobscured, "last_overlay", None)
                    if overlay:
                        self.store.add_system_event(session_id, "client_transparent_overlay", overlay)
                with self.lock:
                    use_fallback = self.fallback_capture_pending or self.gdi_failures >= 2
                image = None
                if session_id:
                    try:
                        with self.resources.budget_slot("capture"):
                            image = capture_client(hwnd, budget.max_capture_resolution[0], budget.max_capture_resolution[1], use_fallback)
                    except ResourceBudgetBusy:
                        time.sleep(max(0.02, budget.next_interval))
                        continue
                if image is None:
                    self.resources.update_capture_metrics(None, True)
                    fallback_reason = str(getattr(capture_client, "last_failure_reason", "") or "")
                    if fallback_reason and context.accepting:
                        try:
                            self.store.add_system_event(session_id, "frame_capture_audit_failure", dict(getattr(capture_client, "last_failure_audit", {}) or {}, fallback_reason=fallback_reason, resource=self.resources.sample()))
                        except Exception as error:
                            self._remember_critical_exception("DataStore", "frame_capture_audit_failure", error, payload={"reason": fallback_reason})
                        self.request_idle("客户区画面无法被完整记录：" + fallback_reason, token)
                        return
                    self.request_idle("客户区画面无法被完整记录：capture_client 返回 None", token)
                    return
                else:
                    self.resources.update_capture_metrics(image.get("capture_elapsed_ms"), False)
                    with self.lock:
                        if image.get("capture_backend") == "gdi":
                            self.gdi_failures = 0
                        else:
                            self.gdi_failures += 1
                            self.fallback_capture_pending = False
                    packet = {"token": token, "session_id": session_id, "image": image, "queued_at": time.time(), "queued_monotonic": time.monotonic()}
                    if self._packet_current(context, packet):
                        self._put_value_packet(context.capture_queue, packet, "capture", context)
                        self.resources.update_pipeline_queue(context.capture_queue.qsize() + context.feature_queue.qsize() + context.persist_queue.qsize(), self._pipeline_age(context), context.queue_capacity * 3)
                wait_time = max(0.05, float(budget.next_interval))
                if self.immediate_capture_event.wait(wait_time):
                    self.immediate_capture_event.clear()
        finally:
            context.capture_done.set()

    def _should_persist_featured_frame(self, image, mode):
        if not isinstance(image, dict):
            return True
        if mode in ("learning", "training"):
            with self.lock:
                self.last_persisted_frame_time = time.monotonic()
            return True
        now = time.monotonic()
        try:
            delta = float(image.get("capture_hash_delta", 64.0) or 64.0)
        except Exception:
            delta = 64.0
        with self.lock:
            last_saved = float(self.last_persisted_frame_time or 0.0)
            mouse_recent = self.last_mouse_activity_ns > 0 and time.monotonic_ns() - int(self.last_mouse_activity_ns) <= 1_200_000_000
            stable = int(self.stable_feature_frames or 0)
            must_keep = mouse_recent or delta >= 3.0 or stable <= 2 or now - last_saved >= 3.0 or bool(image.get("score_valid")) or str(image.get("score_status", "")) == "exact"
            if must_keep:
                self.last_persisted_frame_time = now
                return True
            if now - float(self.last_skipped_frame_notice or 0.0) >= 5.0:
                self.last_skipped_frame_notice = now
                try:
                    self.store.add_system_event(self.session_id, "low_change_frame_decimated", {"delta": delta, "stable_frames": stable, "mode": mode, "time": time.time()})
                except Exception:
                    pass
            return False

    def _pipeline_feature_loop(self, context):
        try:
            while not context.stop_event.is_set() and (not context.capture_done.is_set() or not context.capture_queue.empty()):
                try:
                    packet = context.capture_queue.get(timeout=0.15)
                except queue.Empty:
                    continue
                session_id = packet.get("session_id")
                if not self._packet_current(context, packet):
                    self._drop_pipeline(session_id, "feature", "非本会话数据包被隔离")
                    self._release_packet_budget(packet, context)
                    continue
                image = packet.get("image")
                try:
                    image = extract_frame_features(image)
                    if not image.get("capture_complete", 1):
                        reason = "客户区画面无法被完整记录"
                        self._drop_pipeline(session_id, "feature", reason)
                        if context.accepting:
                            self._critical_recording_failure(session_id, "feature", reason, token=context.token, payload={"capture_failure_reason": image.get("capture_failure_reason", "")})
                        self._release_packet_budget(packet, context)
                        continue
                    if "content_valuable" not in image:
                        image["content_valuable"] = 1
                    budget = self.resources.acquire("capture")
                    self._record_resource_decisions(context.session_id)
                    if budget.must_pause and context.accepting:
                        self._stop_for_client_recording_budget(context.token, "resource_redline_feature", {"pause_reason": budget.pause_reason or "资源红线"})
                    deadline = time.monotonic() + float(budget.retrieval_deadline_seconds)
                    lsh_radius = 2 if budget.state == "正常" and float(getattr(budget, "queue_fill_ratio", 0.0) or 0.0) < 0.35 and int(budget.retrieval_candidate_limit) >= 64 else 1
                    historical = self.store.nearest_hashes(
                        image["dhash64"],
                        8,
                        candidate_limit=budget.retrieval_candidate_limit,
                        deadline=deadline,
                        cancelled=lambda: not self._packet_current(context, packet),
                        yield_if_pressure=lambda: self.resources.capture_snapshot().allowed,
                        current_features=image,
                        exclude_frame_id="",
                        before_capture_finished_ns=int(image.get("capture_finished_monotonic_ns", 0)),
                        lsh_radius=lsh_radius,
                    )
                    online_score, meta = frame_score(image["dhash64"], historical, image)
                    online_score, meta = self._inherit_recent_exact_score(image, online_score, meta)
                    image.update({
                        "online_score": online_score,
                        "score_candidate_count": meta["candidate_count"],
                        "score_top_k_distance": meta["top_k_distance"],
                        "score_retrieval_fallback": 1 if meta["retrieval_fallback"] else 0,
                        "score_retrieval_mode": meta["retrieval_mode"],
                        "score_exact_or_approx": meta["exact_or_approx"],
                        "score_recall_guard": meta["recall_guard"],
                        "score_provisional": bool(meta.get("provisional", True)),
                        "score_valid": False,
                        "score_status": str(meta.get("score_status") or ("provisional" if online_score is not None else "invalid")),
                        "score_valid_for_training": 0,
                        "reward_source": "screen_score_only",
                        "screen_score": online_score,
                        "score_generation": "online",
                    })
                    exact_online = None
                    if self.session_mode == "training":
                        try:
                            exact_deadline = time.monotonic() + min(0.18, max(0.05, float(getattr(budget, "retrieval_deadline_seconds", 0.08) or 0.08) * 2.0))
                            exact_online, exact_meta = self.db_writer.call(lambda: self.store.compute_exact_score_for_image(image, cancelled=lambda: not self._packet_current(context, packet), cooperative=lambda: self.resources.capture_snapshot().allowed, deadline=exact_deadline, limit=8), timeout=30.0)
                            if exact_online is not None and exact_meta.get("score_valid"):
                                image.update({"score_status": "exact", "score_valid": True, "score_provisional": False, "score_valid_for_training": 1, "score_generation": "training_fast_path_exact", "score_retrieval_mode": exact_meta.get("retrieval_mode", "training_fast_path_exact"), "score_exact_or_approx": "exact", "score_recall_guard": 1, "score_candidate_count": exact_meta.get("candidate_count", image.get("score_candidate_count", 0)), "score_top_k_distance": exact_meta.get("top_k_distance", image.get("score_top_k_distance", 64.0)), "score_retrieval_fallback": 0, "screen_score": float(exact_online)})
                        except Exception as error:
                            self._remember_critical_exception("DataStore", "training_fast_path_exact", error, session_id=session_id)
                    try:
                        current_cursor = cursor_position()
                        with self.lock:
                            current_rect = self.target_rect
                        self.db_writer.call(lambda: self.store.record_capture_contract_tick(session_id, image, current_cursor, bool(current_rect and point_inside(current_cursor, current_rect)), False, None, "sample_tick"), timeout=30.0)
                    except Exception as error:
                        self._remember_critical_exception("DataStore", "capture_contract_tick", error, session_id=session_id)
                    with self.lock:
                        previous_hash = self.last_gdi_hash
                        if previous_hash:
                            try:
                                image["capture_hash_delta"] = float(bit_count(int(image["dhash64"], 16) ^ int(previous_hash, 16)))
                            except Exception:
                                image["capture_hash_delta"] = 64.0
                        else:
                            image["capture_hash_delta"] = 64.0
                        if image.get("capture_backend") == "gdi":
                            if previous_hash == image["dhash64"] and (self.last_mouse_activity_ns > self.last_gdi_hash_input_ns or self.gdi_static_count > 0):
                                self.gdi_static_count += 1
                            else:
                                self.gdi_static_count = 0
                            self.last_gdi_hash = image["dhash64"]
                            self.last_gdi_hash_input_ns = self.last_mouse_activity_ns
                            if self.gdi_static_count >= 3:
                                self.fallback_capture_pending = True
                        else:
                            self.gdi_static_count = 0
                        if image.get("dhash64") == self.last_feature_hash:
                            self.stable_feature_frames += 1
                        else:
                            self.stable_feature_frames = 1
                        self.last_feature_hash = str(image.get("dhash64") or "")
                        image["state_stable"] = self.stable_feature_frames >= 2 and bool(image.get("capture_complete", 0))
                    packet.update({
                        "image": image,
                        "online_score": online_score,
                        "exact_score": exact_online,
                        "hunger": 0.0 if exact_online is not None else None,
                        "reward": float(exact_online) if exact_online is not None else None,
                        "queued_at": time.time(),
                        "queued_monotonic": time.monotonic(),
                    })
                    if self._packet_current(context, packet):
                        if self._should_persist_featured_frame(image, self.session_mode):
                            self._put_value_packet(context.feature_queue, packet, "feature", context)
                        else:
                            self._drop_pipeline(session_id, "feature", "低变化帧动态降采样未进入 PNG 编码")
                            self._release_packet_budget(packet, context)
                    else:
                        self._drop_pipeline(session_id, "feature", "会话已关闭，禁止进入编码")
                        self._release_packet_budget(packet, context)
                except Exception as error:
                    reason = "特征/评分失败:" + str(error)
                    self._drop_pipeline(session_id, "feature", reason)
                    if context.accepting:
                        self._critical_recording_failure(session_id, "feature_score", reason, error=error, token=context.token)
                    self._release_packet_budget(packet, context)
                finally:
                    self.resources.update_pipeline_queue(context.capture_queue.qsize() + context.feature_queue.qsize() + context.persist_queue.qsize(), self._pipeline_age(context), context.queue_capacity * 3)
        finally:
            context.feature_done.set()

    def _pipeline_encode_loop(self, context):
        try:
            while not context.stop_event.is_set() and (not context.feature_done.is_set() or not context.feature_queue.empty()):
                try:
                    first = context.feature_queue.get(timeout=0.15)
                except queue.Empty:
                    continue
                packets = [first]
                budget = self.resources.acquire("capture")
                while len(packets) < max(1, int(budget.max_batch)):
                    try:
                        packets.append(context.feature_queue.get_nowait())
                    except queue.Empty:
                        break
                active = [packet for packet in packets if self._packet_current(context, packet)]
                for packet in packets:
                    if packet not in active:
                        self._drop_pipeline(packet.get("session_id"), "encode", "非本会话数据包被隔离")
                        self._release_packet_budget(packet, context)
                if not active:
                    continue
                try:
                    hard_gate_reason = self._pipeline_hard_gate_reason(context)
                    if hard_gate_reason:
                        for packet in active:
                            self._drop_pipeline(packet.get("session_id"), "encode", hard_gate_reason)
                            self._release_packet_budget(packet, context)
                        if context.accepting:
                            self._critical_recording_failure(context.session_id, "encode", hard_gate_reason, token=context.token)
                        continue
                    with self.resources.budget_slot("png"):
                        encoded = self.resources.backend.encode_frames([packet["image"] for packet in active], budget)
                    for packet, image in zip(active, encoded):
                        if not self._packet_current(context, packet):
                            self._drop_pipeline(packet.get("session_id"), "encode", "会话已关闭，禁止进入持久化")
                            self._release_packet_budget(packet, context)
                            continue
                        packet["image"] = image
                        png = image.get("png")
                        if isinstance(png, memoryview):
                            png = png.tobytes()
                        if isinstance(png, bytearray):
                            png = bytes(png)
                        if not isinstance(png, bytes) or not png:
                            self._drop_pipeline(packet.get("session_id"), "encode", "客户区画面无法被完整记录：PNG 缺失")
                            self._release_packet_budget(packet, context)
                            if context.accepting:
                                self._critical_recording_failure(packet.get("session_id"), "encode", "客户区画面无法被完整记录：PNG 缺失", token=context.token)
                            continue
                        image["png"] = png
                        jid = self.store.journal_frame_packet(packet)
                        if not jid:
                            self._drop_pipeline(packet.get("session_id"), "encode", "客户区画面无法被完整记录：accepted journal 未写入")
                            self._release_packet_budget(packet, context)
                            if context.accepting:
                                self._critical_recording_failure(packet.get("session_id"), "encode", "客户区画面无法被完整记录：accepted journal 未写入", token=context.token)
                            continue
                        packet["accepted_journal_id"] = jid
                        packet["queued_at"] = time.time()
                        packet["queued_monotonic"] = time.monotonic()
                        self._put_value_packet(context.persist_queue, packet, "encode", context)
                except Exception as error:
                    for packet in active:
                        self._drop_pipeline(packet.get("session_id"), "encode", "PNG 并行压缩失败:" + str(error))
                        self._release_packet_budget(packet, context)
                    if context.accepting:
                        self.request_idle("客户区画面无法被完整记录：PNG 编码失败：" + str(error), context.token)
                finally:
                    self.resources.update_pipeline_queue(context.capture_queue.qsize() + context.feature_queue.qsize() + context.persist_queue.qsize(), self._pipeline_age(context), context.queue_capacity * 3)
        finally:
            context.encode_done.set()

    def _pipeline_persist_loop(self, context):
        try:
            while not context.stop_event.is_set() and (not context.encode_done.is_set() or not context.persist_queue.empty()):
                try:
                    packet = context.persist_queue.get(timeout=0.15)
                except queue.Empty:
                    continue
                if not self._packet_current(context, packet):
                    self._drop_pipeline(packet.get("session_id"), "persist", "非本会话数据包被隔离")
                    self._release_packet_budget(packet, context)
                    continue
                try:
                    hard_gate_reason = self._pipeline_hard_gate_reason(context)
                    if hard_gate_reason:
                        self._drop_pipeline(packet.get("session_id"), "persist", hard_gate_reason)
                        self._release_packet_budget(packet, context)
                        if context.accepting:
                            self._critical_recording_failure(packet.get("session_id"), "persist", hard_gate_reason, token=context.token)
                        continue
                    if not (packet.get("accepted_journal_id") or packet["image"].get("accepted_journal_id")):
                        self._drop_pipeline(packet.get("session_id"), "persist", "客户区画面无法被完整记录：accepted journal 未写入")
                        self._release_packet_budget(packet, context)
                        if context.accepting:
                            self._critical_recording_failure(packet.get("session_id"), "persist", "客户区画面无法被完整记录：accepted journal 未写入", token=context.token)
                        continue
                    started = time.monotonic()
                    def db_task():
                        with self.resources.budget_slot("sqlite"):
                            frame_id_value = self.store.save_frame(
                                packet["session_id"],
                                packet["image"],
                                packet["image"]["phash"],
                                online_score=packet.get("online_score"),
                                exact_score=packet.get("exact_score"),
                                hunger=packet.get("hunger"),
                                reward=packet.get("reward"),
                                experience_limit=self.settings.data.get("experience_limit"),
                            )
                            self.store.bind_mouse_events_after_frame(packet["session_id"], frame_id_value, packet["image"].get("capture_finished_monotonic_ns", 0))
                            self.store.complete_accepted_journal(packet.get("accepted_journal_id") or packet["image"].get("accepted_journal_id"))
                            self.store.mark_capture_contract_persisted(packet["image"].get("contract_tick_id"), frame_id_value)
                            return frame_id_value
                    frame_id = self.db_writer.call(db_task, timeout=60.0)
                    elapsed_ms = (time.monotonic() - started) * 1000.0
                    self.resources.update_sqlite_latency(elapsed_ms)
                    wal = self.store.wal_metrics()
                    self.resources.update_database_metrics(wal.get("wal_bytes"), wal.get("checkpoint_ms"), elapsed_ms)
                    with self.lock:
                        self.frame_count += 1
                        if packet.get("exact_score") is not None:
                            self.session_valid_frames += 1
                            self.frame_scores.append(float(packet.get("exact_score")))
                            self.frame_scores = self.frame_scores[-120:]
                        self.latest_frame_id = frame_id
                        self.latest_frame_features = {
                            "id": frame_id,
                            "seq": self.frame_count,
                            "state_hash": packet["image"].get("dhash64"),
                            "gray32x18": packet["image"].get("gray32x18"),
                            "edge_density": packet["image"].get("edge_density", 0.0),
                            "color_histogram": packet["image"].get("color_histogram"),
                            "aspect": float(packet["image"].get("width", 1)) / max(1.0, float(packet["image"].get("height", 1))),
                            "capture_finished_monotonic_ns": int(packet["image"].get("capture_finished_monotonic_ns", 0)),
                            "online_score": packet.get("online_score"),
                            "score": packet.get("exact_score"),
                            "hunger": packet.get("hunger"),
                            "score_status": "exact" if packet.get("exact_score") is not None else str(packet["image"].get("score_status", "provisional")),
                            "score_valid_for_training": 1 if packet.get("exact_score") is not None else 0,
                            "reward_source": "screen_score_only",
                            "score_generation": packet["image"].get("score_generation", "online"),
                            "history_boundary_frame_id": packet["image"].get("history_boundary_frame_id"),
                            "state_stable": bool(packet["image"].get("state_stable")),
                            "capture_complete": bool(packet["image"].get("capture_complete")),
                        }
                        if packet.get("exact_score") is not None:
                            self._cache_exact_state(frame_id, packet["image"], packet.get("exact_score"), packet.get("hunger"))
                        self.capture_failures = 0
                        self.capture_failure_started = 0.0
                except PoolCapacityBlocked:
                    self._drop_pipeline(packet.get("session_id"), "persist", "经验池硬上限阻止新截图写入")
                    if context.accepting:
                        self._stop_for_client_recording_budget(context.token, "capacity_blocked_sqlite", {"stage": "persist"})
                except Exception as error:
                    self.resources.update_sqlite_latency(1000.0)
                    try:
                        self.store.add_critical_exception("DataStore", "save_frame", error, session_id=packet.get("session_id"), token=context.token, resource_state=self.resources.capture_snapshot().state)
                    except Exception:
                        pass
                    self._drop_pipeline(packet.get("session_id"), "persist", "SQLite/文件写入失败:" + str(error))
                    if context.accepting:
                        self._handle_frame_persist_failure(context, packet, error)
                finally:
                    self._release_packet_budget(packet, context)
                    self.resources.update_pipeline_queue(context.capture_queue.qsize() + context.feature_queue.qsize() + context.persist_queue.qsize(), self._pipeline_age(context), context.queue_capacity * 3)
        finally:
            context.persist_done.set()

    def _handle_frame_persist_failure(self, context, packet, error):
        sid = packet.get("session_id") if isinstance(packet, dict) else None
        try:
            with self.lock:
                self.frame_replay_pause_until = max(self.frame_replay_pause_until, time.monotonic() + 3.0)
            self.store.add_system_event(sid, "frame_pending_replay", {"reason": str(error), "journal_id": (packet.get("accepted_journal_id") or (packet.get("image") or {}).get("accepted_journal_id")), "persist_queue_length": context.persist_queue.qsize(), "mouse_queue_length": self.mouse_queue.qsize(), "mouse_segment_queue_length": self.mouse_segment_queue.qsize()})
        except Exception:
            pass
        self.flush_mouse_records(3.0)
        last_error = error
        for attempt in range(3):
            try:
                restored = self.db_writer.call(lambda: self.store.replay_accepted_journal(experience_limit=self.settings.data.get("experience_limit"), kinds={"frame"}, max_items=1), timeout=60.0)
                if restored > 0:
                    with self.lock:
                        self.frame_replay_pause_until = 0.0
                    try:
                        self.store.add_system_event(sid, "frame_replay_complete", {"attempt": attempt + 1, "restored": int(restored)})
                    except Exception:
                        pass
                    return True
            except Exception as replay_error:
                last_error = replay_error
                try:
                    self.store.add_critical_exception("DataStore", "frame_journal_replay", replay_error, session_id=sid, token=context.token, resource_state=self.resources.capture_snapshot().state, payload={"stage": "frame_journal_replay", "attempt": attempt + 1, "original_error": str(error), "persist_queue_length": context.persist_queue.qsize(), "mouse_queue_length": self.mouse_queue.qsize(), "mouse_segment_queue_length": self.mouse_segment_queue.qsize()})
                except Exception:
                    pass
            time.sleep(0.15 * (attempt + 1))
        self.request_idle("客户区画面无法被完整记录：截图文件/SQLite 无法持久化：" + str(last_error), context.token)
        return False

    def _pipeline_exact_score_loop(self, context):
        try:
            while not context.stop_event.is_set():
                status = self.store.deferred_score_status(context.session_id)
                oldest_age = max(0.0, time.time() - float(status["oldest"])) if status.get("oldest") else 0.0
                self.resources.update_exact_score_metrics(status["pending"], oldest_age)
                if status["pending"] <= 0:
                    if context.persist_done.is_set():
                        break
                    time.sleep(0.02)
                    continue
                budget = self.resources.acquire("maintenance")
                self._record_resource_decisions(context.session_id)
                if not budget.allowed:
                    if budget.must_pause and context.accepting:
                        self._stop_for_client_recording_budget(context.token, "resource_redline_exact", {"pause_reason": budget.pause_reason or "资源红线"})
                    time.sleep(max(0.02, budget.next_interval))
                    continue
                try:
                    slot = self.resources.budget_slot("exact")
                    slot.__enter__()
                except ResourceBudgetBusy:
                    time.sleep(max(0.02, budget.next_interval))
                    continue
                try:
                    resolved = self.db_writer.call(lambda: self.store.process_deferred_exact_scores(
                    cancelled=lambda: context.stop_event.is_set() or not self._packet_current(context, {"token": context.token, "session_id": context.session_id}),
                    cooperative=lambda: self.resources.capture_snapshot().allowed,
                    maximum=max(1, int(getattr(budget, "database_batch_size", 1))),
                        session_id=context.session_id,
                    ), timeout=60.0)
                finally:
                    slot.__exit__(None, None, None)
                if resolved > 0:
                    summary = self.store.session_score_summary(context.session_id)
                    with self.lock:
                        if self.session_id == context.session_id and self.epoch == context.token:
                            self.session_valid_frames = int(summary["valid_frames"])
                            self.frame_scores = list(summary["scores"])[-120:]
                            latest = summary.get("latest")
                            if latest and self.latest_frame_features and int(self.latest_frame_features.get("capture_finished_monotonic_ns", 0) or 0) == int(latest[4] or 0):
                                self.latest_frame_features.update({"score": float(latest[1]), "hunger": float(latest[2]), "score_status": "exact", "score_generation": "deferred_exact"})
                                self._cache_exact_state(latest[0], self.latest_frame_features, float(latest[1]), float(latest[2]))
                    continue
                time.sleep(max(0.01, budget.next_interval * 0.25))
        finally:
            status = self.store.deferred_score_status(context.session_id)
            oldest_age = max(0.0, time.time() - float(status["oldest"])) if status.get("oldest") else 0.0
            self.resources.update_exact_score_metrics(status["pending"], oldest_age)
            context.exact_done.set()
            context.drain_complete.set()

    def keyboard_hook_escape_pressed(self):
        if IS_WINDOWS:
            try:
                return bool(user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)
            except Exception:
                return False
        return bool(PLATFORM_BACKEND.escape_pressed()) if hasattr(PLATFORM_BACKEND, "escape_pressed") else False

    def _monitor_loop(self, token):
        while self._is_current(token, ("learning", "training")):
            if self.keyboard_hook_escape_pressed():
                self.request_idle("检测到 ESC 键", token); return
            with self.lock:
                session_id = self.session_id
            rect, reason = self._validate_bound_target(require_cursor=True, require_foreground=False)
            if rect is None:
                details = getattr(client_unobscured, "last_obstruction", None) or {"kind": "bound_target_invalid", "reason": reason}
                if session_id:
                    try:
                        self.store.add_system_event(session_id, "client_validation_failure", details)
                    except Exception:
                        pass
                self.request_idle("目标窗口客户区或绑定实例异常：" + reason, token)
                return
            with self.lock:self.target_rect=rect
            if self._should_sleep(token):
                self.on_control_signal("auto_sleep", "AI 判断进入睡眠模式", token=token)
                return
            time.sleep(0.08)

    def _current_score_stagnation_pressure(self):
        base_rate = 0.00004
        try:
            state = self.resources.capture_snapshot().state
            if state == "暂停":
                resource_factor = 0.10
            elif state in ("降速", "排空"):
                resource_factor = 0.40
            else:
                resource_factor = 1.0
            with self.lock:
                stable_frames = int(self.stable_feature_frames or 0)
                scores = list(self.frame_scores[-24:])
            stagnation_factor = 1.0 + min(2.0, max(0.0, stable_frames - 4) / 32.0)
            if len(scores) >= 12 and abs(scores[-1] - scores[0]) < 0.01:
                stagnation_factor += 0.50
            return max(0.000004, min(0.00016, base_rate * resource_factor * stagnation_factor))
        except Exception:
            return base_rate

    def sleep_decision_model(self, features):
        history = self.store.sleep_decision_history(128) if self.store.conn else []
        base_gain = max(0.0, float(features.get("action_uncertainty", 1.0)) * 0.25 + (1.0 - float(features.get("sample_coverage", 0.0))) * 0.20 + float(features.get("resource_pressure", 0.0)) * 0.20)
        if len(history) < 8:
            probability = max(0.0, min(1.0, 0.10 + base_gain))
            return {"sleep_probability": probability, "expected_sleep_gain": base_gain, "trained": False, "sample_count": len(history)}
        weighted = []
        for item in history:
            distance = abs(float(item.get("action_uncertainty", 0.0)) - float(features.get("action_uncertainty", 0.0))) + abs(float(item.get("sample_coverage", 0.0)) - float(features.get("sample_coverage", 0.0))) + abs(float(item.get("resource_pressure", 0.0)) - float(features.get("resource_pressure", 0.0))) + abs(float(item.get("queue_pressure", 0.0)) - float(features.get("queue_pressure", 0.0)))
            outcome = float(item.get("actual_quality_delta", 0.0)) + float(item.get("restored_training_gain", 0.0)) + min(0.25, float(item.get("cleanup_bytes", 0)) / max(1.0, float(self.settings.data.get("experience_limit", 1)))) - min(0.25, float(item.get("duration_seconds", 0.0)) / 3600.0)
            weighted.append((1.0 / (0.05 + distance), outcome))
        total = sum(weight for weight, _ in weighted)
        learned_gain = sum(weight * outcome for weight, outcome in weighted) / max(1e-9, total)
        expected = 0.55 * learned_gain + 0.45 * base_gain
        probability = max(0.0, min(1.0, 0.5 + expected * 2.0))
        return {"sleep_probability": probability, "expected_sleep_gain": expected, "trained": True, "sample_count": len(history)}

    def _should_sleep(self, token):
        with self.lock:
            if token != self.epoch or self.state != "training":
                return False
            elapsed = time.monotonic() - self.session_started
            scores = list(self.frame_scores[-40:])
            plan = list(self.ai_plan)
            valid_frames = int(self.session_valid_frames)
            valid_actions = int(self.session_valid_actions)
            auto_sleeps = int(self.training_auto_sleep_count)
            last_auto_sleep_at = float(self.last_auto_sleep_at)
            auto_sleep_times = list(self.auto_sleep_times)
            last_auto_sleep_pool_bytes = int(self.last_auto_sleep_pool_bytes or 0)
            last_fingerprint_check = float(self.last_sleep_fingerprint_check)
            queue_empty = self.mouse_queue.empty() and self.raw_mouse_queue.empty() and self.raw_critical_queue.empty() and self.raw_hook_ring.empty() and self.capture_queue.empty() and self.feature_queue.empty() and self.persist_queue.empty()
            session_id = self.session_id
        pending = self.store.deferred_score_status(session_id)
        pending_exact = int(pending.get("pending", 0) or 0)
        oldest_exact = max(0.0, time.time() - float(pending["oldest"])) if pending.get("oldest") else 0.0
        self.resources.update_exact_score_metrics(pending_exact, oldest_exact)
        pool_now = 0
        try:
            pool_now = int(self.store.pool_size_fast())
        except Exception:
            pool_now = 0
        if elapsed < 60.0:
            return False
        current_time = time.time()
        recent_hour = [item for item in auto_sleep_times if current_time - float(item) < 3600.0]
        if current_time - last_auto_sleep_at < 300.0 or len(recent_hour) >= 4:
            return False
        sample = self.resources.sample()
        if sample.get("resource_state") == "暂停":
            return False
        limit = int(self.settings.data.get("experience_limit", 1) or 1)
        capacity_status = self.store.capacity_status() if self.store.conn else {"tier": 0}
        capacity_tier = int(capacity_status.get("tier", 0) or 0)
        pool_pressure = pool_now >= int(limit * 0.95) or capacity_tier >= 95
        pool_near_cleanup = pool_now >= int(limit * 0.90) or capacity_tier >= 90
        pool_growth = pool_pressure or pool_near_cleanup or (pool_now > 0 and pool_now >= last_auto_sleep_pool_bytes + 64 * 1024 * 1024)
        resource_pressure = 1.0 if sample.get("resource_state") == "降速" else 0.0
        coverage = min(1.0, len(plan) / 64.0)
        effective_frames = valid_frames + min(8, pending_exact)
        if effective_frames <= 0 and pending_exact <= 0:
            return False
        if sample.get("disk_free", 1) < 1024 * 1024 * 1024 or sample.get("avail_memory", 1) < 384 * 1024 * 1024:
            return False
        if not queue_empty and sample.get("resource_state") != "排空" and not pool_pressure and pending_exact < 16:
            return False
        now = time.monotonic()
        if now - last_fingerprint_check < 5.0:
            return False
        fingerprint = self.store.training_data_fingerprint()
        with self.lock:
            if token != self.epoch or self.state != "training":
                return False
            self.last_sleep_fingerprint_check = now
            self.current_training_fingerprint = fingerprint
            unchanged = fingerprint == self.last_successful_training_fingerprint or fingerprint == self.last_auto_sleep_fingerprint
        if unchanged and pending_exact <= 0:
            return False
        if fingerprint == self.last_training_attempt_fingerprint and time.time() < self.next_training_retry_at and pending_exact <= 0:
            return False
        mean = sum(scores) / max(1, len(scores))
        variance = sum((value - mean) ** 2 for value in scores) / max(1, len(scores))
        trend = scores[-1] - scores[0] if len(scores) >= 2 else 0.0
        lcbs = [float(item.get("confidence_lower_bound", 0.0)) for item in plan]
        uncs = [float(item.get("uncertainty", 1.0)) for item in plan]
        features = {
            "score_mean": mean,
            "score_variance": variance,
            "score_trend": trend,
            "score_stagnation_pressure": self._current_score_stagnation_pressure(),
            "action_lcb_mean": sum(lcbs) / max(1, len(lcbs)),
            "action_uncertainty": sum(uncs) / max(1, len(uncs)),
            "sample_coverage": coverage,
            "valid_frames": valid_frames,
            "valid_actions": valid_actions,
            "exact_score_backlog": pending_exact,
            "exact_score_oldest_age": oldest_exact,
            "mouse_queue_length": self.mouse_queue.qsize(),
            "queue_pressure": min(1.0, self._pipeline_age() / 2.0),
            "resource_pressure": resource_pressure,
        }
        decision = self.sleep_decision_model(features)
        estimated_sleep_seconds = max(8.0, 6.0 + pending_exact / max(1.0, float(self.resources.capture_snapshot().database_batch_size)) * 0.40 + min(90.0, effective_frames * 0.03))
        expected_gain_rate = float(decision["expected_sleep_gain"]) / estimated_sleep_seconds
        features["estimated_sleep_seconds"] = estimated_sleep_seconds
        features["expected_gain_rate"] = expected_gain_rate
        baseline_ready = effective_frames >= 4 and valid_actions >= 4
        backlog_ready = pending_exact >= 1 and (oldest_exact >= 1.0 or sample.get("resource_state") == "排空")
        backlog_hard = pending_exact >= 96 or (pending_exact >= 16 and oldest_exact >= 30.0) or (pending_exact > 0 and sample.get("resource_state") == "排空")
        minimum_gain = 0.02 if decision.get("trained") else 0.03
        minimum_gain_rate = 0.0012 if decision.get("trained") else 0.0020
        score_stalled = len(scores) >= 12 and abs(trend) < 0.01 and variance < 0.0025
        expected_task2_positive = pool_near_cleanup or pending_exact >= 1 or int(capacity_status.get("remaining", 0) or 0) > int(capacity_status.get("target", 0) or 0) > 0
        expected_training_positive = baseline_ready and not unchanged and valid_actions >= 4
        expected_positive = expected_task2_positive or expected_training_positive or backlog_hard
        worth_sleeping = pool_growth or score_stalled or decision["expected_sleep_gain"] >= minimum_gain
        if not expected_positive or expected_gain_rate <= minimum_gain_rate or decision["expected_sleep_gain"] <= 0.0:
            return False
        should = backlog_hard or (worth_sleeping and (baseline_ready or backlog_ready or pool_near_cleanup) and (decision["sleep_probability"] >= 0.52 or resource_pressure > 0 or backlog_ready or pool_near_cleanup) and decision["expected_sleep_gain"] >= minimum_gain and expected_gain_rate >= minimum_gain_rate)
        if should:
            with self.lock:
                self.pending_sleep_decision = self.store.record_sleep_decision(features, decision["expected_sleep_gain"])
        return bool(should)

    def _current_cluster(self):
        with self.lock:
            scores = list(self.frame_scores[-8:])
        if not scores:
            return "unknown"
        mean = sum(scores) / len(scores)
        if mean >= 0.66:
            return "high"
        if mean >= 0.33:
            return "mid"
        return "low"

    def _state_distance_details(self, current, item):
        if not current or not item:
            return 1.0, {"dhash": 1.0, "gray": 1.0, "color": 1.0, "edge": 1.0, "aspect": 1.0}, True
        components = {"dhash": 1.0, "gray": 1.0, "color": 1.0, "edge": 1.0, "aspect": 1.0}
        try:
            if current.get("state_hash") and item.get("state_hash"):
                components["dhash"] = bit_count(int(current["state_hash"], 16) ^ int(item["state_hash"], 16)) / 64.0
        except Exception:
            pass
        try:
            a = feature_bytes(current.get("gray32x18"), 32 * 18)
            b = feature_bytes(item.get("gray32x18"), 32 * 18)
            if a and b and len(a) == len(b):
                components["gray"] = sum(abs(x - y) for x, y in zip(a, b)) / (255.0 * len(a))
        except Exception:
            pass
        try:
            left = histogram_values(current.get("color_histogram"))
            right = histogram_values(item.get("color_histogram"))
            if left and right and len(left) == len(right) and sum(left) > 0 and sum(right) > 0:
                left_total = float(sum(left))
                right_total = float(sum(right))
                components["color"] = min(1.0, sum(abs(a / left_total - b / right_total) for a, b in zip(left, right)) / 2.0)
        except Exception:
            pass
        try:
            components["edge"] = min(1.0, abs(float(current.get("edge_density", 0.0)) - float(item.get("edge_density", 0.0))))
        except Exception:
            pass
        try:
            a = float(current.get("aspect", 0.0))
            b = float(item.get("aspect", 0.0))
            if a > 0 and b > 0:
                components["aspect"] = min(1.0, abs(a - b) / max(a, b))
        except Exception:
            pass
        distance = 0.35 * components["dhash"] + 0.30 * components["gray"] + 0.20 * components["color"] + 0.10 * components["edge"] + 0.05 * components["aspect"]
        critical = components["dhash"] > 0.45 or components["gray"] > 0.45 or components["color"] > 0.40 or components["aspect"] > 0.15
        return max(0.0, min(1.0, distance)), components, critical

    def _state_distance(self, current, item):
        return self._state_distance_details(current, item)[0]

    def _action_benefit_fused(self, action_type):
        action_type = str(action_type or "")
        if action_type == "移动":
            return False
        now = time.time()
        cached = self.action_benefit_fuse.get(action_type)
        if cached and now - float(cached.get("checked", 0.0) or 0.0) < 30.0:
            return bool(cached.get("fused"))
        try:
            benefit = self.store.recent_action_benefit(action_type, 8)
        except Exception as error:
            note_strict_exception("action_benefit_fuse", error, {"action_type": action_type})
            return False
        fused = int(benefit.get("count", 0) or 0) >= 6 and (float(benefit.get("mean", 0.0) or 0.0) < 0.0 or bool(benefit.get("unstable")))
        self.action_benefit_fuse[action_type] = dict(benefit, checked=now, fused=fused)
        if fused:
            try:
                self.store.add_system_event(self.session_id, "action_benefit_fuse", {"action_type": action_type, "benefit": benefit, "until": "sleep_training_refresh"})
            except Exception as error:
                note_strict_exception("action_benefit_fuse_event", error, {"action_type": action_type})
        return fused

    def _reset_action_benefit_fuse_after_training(self, training_result=None):
        result = training_result or {}
        if str(result.get("status", "")) in ("trained", "saved") or int(result.get("samples", 0) or 0) > 0:
            self.action_benefit_fuse = {}
            try:
                self.store.add_system_event(self.session_id, "action_benefit_fuse_reset", {"training_status": str(result.get("status", "")), "samples": int(result.get("samples", 0) or 0)})
            except Exception as error:
                note_strict_exception("action_benefit_fuse_reset_event", error, {})

    def _update_sleep_terminal_state(self, **changes):
        with self.lock:
            current = dict(self.sleep_task_terminal_state or {})
            current.update(changes)
            current["updated_at"] = time.time()
            current["sleep_origin"] = self.sleep_origin
            self.sleep_task_terminal_state = current
            sid = self.session_id
        try:
            self.store.add_system_event(sid, "sleep_task_terminal_state", current)
        except Exception as error:
            note_strict_exception("sleep_terminal_state_event", error, current)

    def _recent_action_summary(self):
        with self.lock:
            recent = [item for item in self.recent_ai_actions if time.time() - float(item.get("created", 0.0) or 0.0) < 30.0]
        total = max(1, len(recent))
        move = sum(1 for item in recent if item.get("action_type") == "移动")
        button = sum(1 for item in recent if item.get("action_type") in ("左键", "右键"))
        wheel = sum(1 for item in recent if item.get("action_type") in ("滚轮", "水平滚轮"))
        mean_delta = sum(float(item.get("score_delta", 0.0) or 0.0) for item in recent) / total
        return {"move_rate": move / total, "button_rate": button / total, "wheel_rate": wheel / total, "mean_score_delta": mean_delta}

    def _ai_target(self, rect):
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        with self.lock:
            step = self.ai_step
            self.ai_step += 1
            plan = list(self.ai_plan)
            current = dict(self.latest_frame_features) if self.latest_frame_features else None
            limits = dict(self.action_limits)
            counts = dict(self.ai_region_counts)
        gates = {"左键": (12, 0.60, 0.0, 2.0, 20), "右键": (16, 0.65, 0.02, 3.0, 12), "滚轮": (14, 0.62, 0.0, 1.5, 24), "水平滚轮": (16, 0.65, 0.02, 2.0, 12)}
        total_visits = max(1, sum(int(v) for v in counts.values()))
        viable = []
        sample_metrics = self.resources.sample()
        direct_current = dict(current or {})
        if direct_current:
            direct_current["recent_action_summary"] = self._recent_action_summary()
            direct = self.resources.backend.suggest_visual_action(direct_current, sample_metrics, timeout=0.04)
            if direct is not None:
                x_ratio = min(0.95, max(0.05, float(direct.get("x_ratio", 0.5))))
                y_ratio = min(0.95, max(0.05, float(direct.get("y_ratio", 0.5))))
                action_type = str(direct.get("action_type", "移动"))
                if action_type != "移动" and (str(direct_current.get("score_status", "")) != "exact" or not bool(direct_current.get("state_stable")) or not bool(direct_current.get("capture_complete"))):
                    action_type = "移动"
                if action_type != "移动" and (float(direct.get("confidence", 0.0) or 0.0) < 0.58 or float(direct.get("uncertainty", 1.0) or 1.0) > 0.24):
                    action_type = "移动"
                if action_type != "移动" and self._action_benefit_fused(action_type):
                    action_type = "移动"
                x = rect[0] + int(width * x_ratio)
                y = rect[1] + int(height * y_ratio)
                gx = max(0, min(9, int(x_ratio * 10)))
                gy = max(0, min(5, int(y_ratio * 6)))
                with self.lock:
                    self.ai_region_counts[(gx, gy)] += 1
                return {"x": x, "y": y, "action_type": action_type, "wheel_delta": int(direct.get("wheel_delta", 0) or 0), "confidence_probability": float(direct.get("confidence", 0.0) or 0.0), "confidence": float(direct.get("confidence", 0.0) or 0.0), "uncertainty": float(direct.get("uncertainty", 1.0) or 1.0), "samples": 0, "state_match_distance": 0.0, "model_available": True, "ucb_exploration": 0.0, "region_visit_count": int(counts.get((gx, gy), 0) or 0), "resource_metrics": sample_metrics, "capture_finished_monotonic_ns": int(direct_current.get("capture_finished_monotonic_ns", 0) or 0), "score_status": str(direct_current.get("score_status", "invalid")), "state_stable": bool(direct_current.get("state_stable")), "capture_complete": bool(direct_current.get("capture_complete")), "policy_backend": "GPU ONNX 多头视觉策略"}
        for item in plan:
            if not isinstance(item, dict):
                continue
            try:
                action_type = str(item.get("action_type", "移动"))
                samples = int(item.get("samples", 0))
                lcb = float(item.get("confidence_lower_bound", -1.0))
                probability = max(0.0, min(1.0, float(item.get("confidence_probability", 0.0) or 0.0)))
                uncertainty = max(0.0, min(1.0, float(item.get("uncertainty", 1.0) or 1.0)))
                effective_samples = float(item.get("effective_samples", samples) or 0.0)
                interval_width = float(item.get("confidence_interval_width", 999.0) or 999.0)
                baseline_support = float(item.get("baseline_support", 0.0) or 0.0)
                queue_age = float(sample_metrics.get("pipeline_queue_age", 0.0) or 0.0)
                fresh = int((current or {}).get("capture_finished_monotonic_ns", 0) or 0) > 0 and time.monotonic_ns() - int((current or {}).get("capture_finished_monotonic_ns", 0) or 0) <= 1_000_000_000
                distance, state_parts, critical_mismatch = self._state_distance_details(current, item)
                if distance > float(item.get("state_similarity_threshold", 0.38)):
                    continue
                if critical_mismatch:
                    action_type = "移动"
                x_ratio = min(0.95, max(0.05, float(item.get("x", 0.5))))
                y_ratio = min(0.95, max(0.05, float(item.get("y", 0.5))))
                gx = max(0, min(9, int(x_ratio * 10)))
                gy = max(0, min(5, int(y_ratio * 6)))
                visit_count = int(counts.get((gx, gy), 0) or 0)
                current_local = local_visual_descriptor((current or {}).get("gray32x18"), (current or {}).get("color_histogram"), (current or {}).get("edge_density", 0.0), x_ratio, y_ratio, int((item.get("local_descriptor") or {}).get("radius", 3) or 3))
                local_distance = local_visual_distance(current_local, item.get("local_descriptor") or {})
                local_radius = max(0.0, min(1.0, float(item.get("local_uncertainty_radius", 1.0) or 1.0)))
                if action_type != "移动" and (local_distance > 0.28 or local_radius > 0.14):
                    action_type = "移动"
                if action_type != "移动":
                    min_samples, min_probability, min_lcb, cooldown, per_minute = gates.get(action_type, (999999, 1.0, 1.0, 999.0, 0))
                    stat = limits.get(action_type, {"last": 0.0, "times": []})
                    recent = [t for t in stat.get("times", []) if time.monotonic() - t < 60.0]
                    if samples < min_samples or effective_samples < 8.0 or probability < min_probability or lcb <= min_lcb or float(item.get("validation_lower_bound", -1.0)) <= 0 or int(item.get("validation_samples", 0)) < 8 or float(item.get("validation_false_positive_rate", 1.0)) > 0.10 or interval_width >= 0.24 or baseline_support < 4.0 or uncertainty > 0.12 or distance >= 0.32 or not fresh or not bool((current or {}).get("state_stable")) or not bool((current or {}).get("capture_complete")) or str((current or {}).get("score_status", "")) != "exact" or queue_age > 0.25 or time.monotonic() - stat.get("last", 0.0) < cooldown or len(recent) >= per_minute:
                        action_type = "移动"
                if action_type != "移动" and self._action_benefit_fused(action_type):
                    action_type = "移动"
                edge_risk = max(0.0, 0.12 - min(x_ratio, 1.0 - x_ratio, y_ratio, 1.0 - y_ratio)) * 3.0
                exploration = math.sqrt(math.log(total_visits + 2.0) / (visit_count + 1.0))
                coverage_bonus = 0.18 * exploration
                uncertainty_bonus = 0.30 * uncertainty if action_type == "移动" else -0.55 * uncertainty
                stability_bonus = 0.05 if bool((current or {}).get("state_stable")) else -0.05
                risk = edge_risk + max(0.0, local_distance - 0.20) * 0.35 + distance * 0.35
                score = probability + uncertainty_bonus + coverage_bonus + stability_bonus - risk + min(0.10, samples / 400.0)
                viable.append((score, exploration, samples, action_type, distance, uncertainty, probability, x_ratio, y_ratio, item))
            except (TypeError, ValueError):
                pass
        viable.sort(reverse=True, key=lambda value: (value[0], value[1], value[2]))
        item_tuple = viable[0] if viable else None
        if item_tuple is not None:
            _, exploration, samples, action_type, distance, uncertainty, probability, x_ratio, y_ratio, item = item_tuple
            wheel_delta = max(-1200, min(1200, int(item.get("wheel_delta", 120 if action_type in ("滚轮", "水平滚轮") else 0))))
            model_available = True
        else:
            grid = []
            for gy in range(6):
                for gx in range(10):
                    x_ratio = (gx + 0.5) / 10.0
                    y_ratio = (gy + 0.5) / 6.0
                    visit_count = int(counts.get((gx, gy), 0) or 0)
                    edge_risk = max(0.0, 0.10 - min(x_ratio, 1.0 - x_ratio, y_ratio, 1.0 - y_ratio)) * 2.0
                    score = math.sqrt(math.log(total_visits + 2.0) / (visit_count + 1.0)) - edge_risk + ((step + gx * 17 + gy * 31) % 7) * 0.0001
                    grid.append((score, gx, gy, x_ratio, y_ratio))
            grid.sort(reverse=True)
            _, gx, gy, x_ratio, y_ratio = grid[0]
            action_type = "移动"
            probability = 0.52
            uncertainty = 0.68
            samples = 0
            wheel_delta = 0
            distance = 1.0
            exploration = 1.0
            model_available = False
        x = rect[0] + int(width * x_ratio)
        y = rect[1] + int(height * y_ratio)
        gx = max(0, min(9, int(x_ratio * 10)))
        gy = max(0, min(5, int(y_ratio * 6)))
        with self.lock:
            self.ai_region_counts[(gx, gy)] += 1
        return {"x": x, "y": y, "action_type": action_type, "wheel_delta": wheel_delta, "confidence_probability": probability, "confidence": probability, "uncertainty": uncertainty, "samples": samples, "state_match_distance": distance, "model_available": model_available, "ucb_exploration": exploration, "region_visit_count": int(counts.get((gx, gy), 0) or 0), "resource_metrics": sample_metrics, "capture_finished_monotonic_ns": int((current or {}).get("capture_finished_monotonic_ns", 0) or 0), "score_status": str((current or {}).get("score_status", "invalid")), "state_stable": bool((current or {}).get("state_stable")), "capture_complete": bool((current or {}).get("capture_complete"))}

    def _validate_bound_target(self, require_cursor=True, require_foreground=False):
        with self.lock:
            hwnd = self.target_hwnd
            expected_root = self.target_root
            expected_pid = self.target_pid
            expected_path = self.target_process_path
        if not hwnd or not expected_root:
            return None, "未绑定目标窗口"
        if IS_WINDOWS:
            if not expected_pid:
                return None, "未绑定目标进程"
            if not user32.IsWindow(hwnd) or root_window(hwnd) != expected_root:
                return None, "窗口句柄已失效或根窗口已变化"
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(expected_root, ctypes.byref(pid))
            if int(pid.value) != int(expected_pid):
                return None, "绑定窗口 PID 已变化"
            actual_path = normalized_windows_path(process_full_path(int(pid.value)))
            configured_path = normalized_windows_path(self.settings.data.get("emulator_path", "")) if self.settings.data.get("emulator_path", "") else ""
            if expected_path and (not actual_path or actual_path != expected_path):
                return None, "绑定窗口可执行文件路径已变化"
            if configured_path and actual_path and actual_path != configured_path:
                return None, "绑定窗口可执行文件路径与配置不一致"
        else:
            candidates = [item for item in PLATFORM_BACKEND.list_windows() if int(item.get("hwnd", 0) or 0) == int(hwnd or 0)]
            if not candidates:
                return None, "窗口句柄已失效或根窗口已变化"
            if expected_pid and int(candidates[0].get("pid", 0) or 0) != int(expected_pid):
                return None, "绑定窗口 PID 已变化"
            actual_path = os.path.realpath(str(candidates[0].get("path", "")))
            if expected_path and actual_path and os.path.realpath(expected_path) != actual_path:
                return None, "绑定窗口可执行文件路径已变化"
        rect = valid_client(hwnd, require_cursor)
        if rect is None:
            return None, getattr(valid_client, "last_reason", "客户区无效")
        if require_foreground:
            if IS_WINDOWS:
                foreground = root_window(user32.GetForegroundWindow())
                if not foreground or foreground != expected_root:
                    return None, "目标窗口不是前台窗口"
                fg_pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(foreground, ctypes.byref(fg_pid))
                if int(fg_pid.value) != int(expected_pid):
                    return None, "前台窗口 PID 不匹配"
                foreground_path = normalized_windows_path(process_full_path(int(fg_pid.value)))
                if foreground_path != expected_path:
                    return None, "前台窗口可执行文件路径不匹配"
            elif not foreground_root_matches(hwnd):
                return None, "目标窗口不是前台窗口"
        return rect, ""

    def _foreground_matches_target(self):
        rect, _ = self._validate_bound_target(require_cursor=True, require_foreground=True)
        return rect is not None

    def _record_synthetic_ai_mouse(self, event_type, button, wheel, x, y, behavior_probability=None):
        with self.lock:
            if self.state != "training" or not self.session_id or not self.target_rect:
                return False
            session_id = self.session_id
            rect = self.target_rect
            before_frame_id = self.latest_frame_id
        width = max(1, rect[2] - rect[0])
        height = max(1, rect[3] - rect[1])
        now = time.time()
        mono = time.monotonic_ns()
        record = {"session_id": session_id, "created": now, "created_monotonic_ns": mono, "source": "ai", "event_type": event_type, "button": button, "wheel": int(wheel or 0), "x": int(x), "y": int(y), "relative_x": (int(x) - rect[0]) / width, "relative_y": (int(y) - rect[1]) / height, "dx": 0.0, "dy": 0.0, "direction": 0.0, "speed": 0.0, "behavior_probability": behavior_probability, "before_frame_id": before_frame_id, "after_frame_id": None}
        try:
            self.store.journal_mouse_record(record, durable=True)
            self.mouse_queue.put_nowait(record)
            return True
        except Exception as error:
            try:
                self.store.mark_session_untrainable(session_id, "AI 合成鼠标事件无法无损记录")
                self.store.add_critical_exception("DataStore", "synthetic_ai_mouse", error, session_id=session_id, token=self.epoch, resource_state=self.resources.capture_snapshot().state)
            except Exception:
                pass
            self.on_control_signal("stop", "AI 合成鼠标事件无法无损记录，已停止当前训练会话")
            return False

    def _ai_loop(self, token):
        while self._is_current(token, ("training",)):
            budget = self.resources.acquire("ai_inference")
            if not budget.allowed:
                time.sleep(max(0.05, budget.next_interval))
                continue
            rect, reason = self._validate_bound_target(require_cursor=True, require_foreground=True)
            if rect is None:
                self.request_idle("AI 输入前绑定校验失败：" + reason, token)
                return
            target = self._ai_target(rect)
            frame_finished_ns = int(target.get("capture_finished_monotonic_ns", 0) or 0)
            frame_age = max(0.0, (time.monotonic_ns() - frame_finished_ns) / 1_000_000_000.0) if frame_finished_ns else float("inf")
            with self.lock:
                capture_interval = float(self.capture_interval_seconds)
            freshness_limit = max(0.250, 1.5 * capture_interval)
            queue_age = float(self.resources.sample().get("pipeline_queue_age", 0.0) or 0.0)
            capacity = self.store.capacity_status() if self.store.conn else {"tier": 0}
            if int(capacity.get("tier", 0) or 0) >= 95:
                target["action_type"] = "移动"
                target["capacity_gate"] = "capacity_95_move_only"
            if target.get("action_type") != "移动" and (frame_age > freshness_limit or queue_age > 0.250 or budget.queue_fill_ratio >= 0.90 or target.get("score_status") != "exact" or not target.get("state_stable") or not target.get("capture_complete")):
                target["action_type"] = "移动"
                target["freshness_gate"] = "stale_backlogged_or_provisional"
            policy = self.resources.backend.infer_policy(target, budget)
            target["confidence"] = float(policy.get("confidence", target.get("confidence_probability", 0.0)))
            target["uncertainty"] = float(policy.get("uncertainty", target.get("uncertainty", 1.0)))
            target["policy_backend"] = policy.get("backend")
            band = self.resources.runtime.confidence_band(float(target.get("confidence", 0.0)), budget.state == "暂停")
            if band == "low":
                time.sleep(max(0.05, budget.next_interval))
                continue
            if (budget.state in ("降速", "暂停") or band in ("medium", "pressure")) and target.get("action_type") != "移动":
                target["action_type"] = "移动"
            x, y = target["x"], target["y"]
            rect, reason = self._validate_bound_target(require_cursor=True, require_foreground=True)
            if rect is None or not point_inside((x, y), rect):
                self.request_idle("AI 移动前绑定或客户区校验失败：" + (reason or "目标点越界"), token)
                return
            move_marker = self.ai_input_tracker.register("move", "", 0, x, y, target.get("confidence_probability"))
            self._record_synthetic_ai_mouse("move", "", 0, x, y, target.get("confidence_probability"))
            if not ai_move_to(x, y, move_marker):
                self.request_idle("AI 鼠标移动无法确认位于绑定客户区内", token)
                return
            time.sleep(0.05)
            rect, reason = self._validate_bound_target(require_cursor=True, require_foreground=True)
            if rect is None or not point_inside(cursor_position(), rect):
                self.request_idle("AI 移动后绑定或客户区校验失败：" + (reason or "鼠标已离开客户区"), token)
                return
            action_type = target.get("action_type", "移动")
            ok = True
            if action_type != "移动":
                rect, reason = self._validate_bound_target(require_cursor=True, require_foreground=True)
                if rect is None:
                    self.request_idle("AI 动作前绑定校验失败：" + reason, token)
                    return
            if action_type == "左键":
                cx, cy = cursor_position()
                down_marker = self.ai_input_tracker.register("button_down", "left", 0, cx, cy, target.get("confidence_probability"))
                up_marker = self.ai_input_tracker.register("button_up", "left", 0, cx, cy, target.get("confidence_probability"))
                self._record_synthetic_ai_mouse("button_down", "left", 0, cx, cy, target.get("confidence_probability"))
                ok = ai_left_click(down_marker, up_marker)
                self._record_synthetic_ai_mouse("button_up", "left", 0, cx, cy, target.get("confidence_probability"))
            elif action_type == "右键":
                cx, cy = cursor_position()
                down_marker = self.ai_input_tracker.register("button_down", "right", 0, cx, cy, target.get("confidence_probability"))
                up_marker = self.ai_input_tracker.register("button_up", "right", 0, cx, cy, target.get("confidence_probability"))
                self._record_synthetic_ai_mouse("button_down", "right", 0, cx, cy, target.get("confidence_probability"))
                ok = ai_right_click(down_marker, up_marker)
                self._record_synthetic_ai_mouse("button_up", "right", 0, cx, cy, target.get("confidence_probability"))
            elif action_type == "滚轮":
                cx, cy = cursor_position()
                delta = target.get("wheel_delta", 120)
                wheel_marker = self.ai_input_tracker.register("wheel", "vertical", delta, cx, cy, target.get("confidence_probability"))
                self._record_synthetic_ai_mouse("wheel", "vertical", delta, cx, cy, target.get("confidence_probability"))
                ok = ai_wheel(delta, wheel_marker, False)
            elif action_type == "水平滚轮":
                cx, cy = cursor_position()
                delta = target.get("wheel_delta", 120)
                wheel_marker = self.ai_input_tracker.register("wheel", "horizontal", delta, cx, cy, target.get("confidence_probability"))
                self._record_synthetic_ai_mouse("wheel", "horizontal", delta, cx, cy, target.get("confidence_probability"))
                ok = ai_wheel(delta, wheel_marker, True)
            action_finished_ns = time.monotonic_ns()
            if action_type in ("移动", "左键", "右键", "滚轮", "水平滚轮"):
                with self.lock:
                    self.recent_ai_actions.append({"created": time.time(), "action_type": action_type, "confidence": float(target.get("confidence", 0.0)), "score_delta": 0.0})
                    self.recent_ai_actions = self.recent_ai_actions[-32:]
            if action_type != "移动" and ok:
                with self.lock:
                    stat = self.action_limits.setdefault(action_type, {"last": 0.0, "times": []})
                    now = time.monotonic()
                    stat["last"] = now
                    stat["times"] = [t for t in stat.get("times", []) if now - t < 60.0] + [now]
            if not ok:
                self.request_idle("AI 鼠标动作无法执行" + str(action_type), token)
                return
            if ok:
                self.request_immediate_capture("ai_action_complete")
            if action_type != "移动":
                if not self._wait_for_post_action_frame(token, action_finished_ns):
                    self.request_idle("非移动动作后未获得新鲜后置画面，已安全结束训练会话", token)
                    return
            time.sleep(max(0.05, budget.next_interval))

    def _wait_for_post_action_frame(self, token, action_finished_ns):
        deadline = time.monotonic() + 2.0
        while self._is_current(token, ("training",)) and time.monotonic() < deadline:
            with self.lock:
                current = dict(self.latest_frame_features) if self.latest_frame_features else {}
            fresh = int(current.get("capture_finished_monotonic_ns", 0) or 0) > int(action_finished_ns)
            queue_age = float(self.resources.sample().get("pipeline_queue_age", 0.0) or 0.0)
            if fresh and queue_age <= 0.250:
                return True
            time.sleep(0.02)
        return False

    def _write_barrier(self, session_id, reason, context=None):
        context = context or self.pipeline_context
        forced_detail = ""
        if context is not None:
            with self.lock:
                context.accepting = False
                context.draining = True
            exact_drain_seconds = 8.0
            deadline = time.monotonic() + exact_drain_seconds
            context.capture_done.wait(max(0.1, deadline - time.monotonic()))
            for thread in list(context.threads):
                if thread is threading.current_thread():
                    continue
                remaining = max(0.1, deadline - time.monotonic())
                thread.join(remaining)
            alive = [thread.name for thread in context.threads if thread is not threading.current_thread() and thread.is_alive()]
            queue_sets = (("capture", context.capture_queue), ("feature", context.feature_queue), ("persist", context.persist_queue))
            queues_left = sum(item[1].qsize() for item in queue_sets)
            pending_exact = 0
            try:
                pending_exact = int(self.store.deferred_score_status(session_id).get("pending", 0) or 0)
            except Exception:
                pending_exact = 0
            if alive or queues_left:
                if pending_exact > 0:
                    try:
                        timed_out = self.store.mark_deferred_exact_timeout(session_id, exact_drain_seconds)
                        self.emit("notice", "本次停止排空超时：待补评分 {}，已标记 exact_timeout；本次可训练帧 {}。".format(timed_out, self.session_valid_frames))
                    except Exception:
                        pass
                with self.lock:
                    if context not in self.detached_pipeline_contexts:
                        self.detached_pipeline_contexts.append(context)
                forced_detail = "写入屏障超时：线程 {}，队列剩余 {}；已停止接收新数据，保留已接受数据继续排空或由 journal 恢复".format(",".join(alive) or "无", queues_left)
            else:
                context.closed = True
                context.stop_event.set()
                context.drain_complete.set()
        self._flush_pipeline_losses()
        self._flush_move_compression(session_id)
        self._flush_raw_mouse_losses(True)
        self.flush_mouse_records(10.0)
        with self.loss_lock:
            losses = self.move_loss.pop(session_id, None) if session_id else None
        if losses:
            self.store.record_mouse_loss(session_id, losses[0], losses[1], losses[2], "鼠标事件队列写入失败")
        if forced_detail:
            self.store.add_system_event(session_id, "write_barrier_timeout", {"reason": reason, "time": time.time(), "detail": forced_detail, "pipeline_token": context.token if context is not None else None})
            return False, forced_detail
        ok, detail = self.store.validate_consistency()
        if not ok:
            return False, "写入屏障一致性校验失败：" + detail
        self.store.add_system_event(session_id, "write_barrier", {"reason": reason, "time": time.time(), "consistency": detail, "pipeline_queue_age": self._pipeline_age(context), "pipeline_token": context.token if context is not None else None, "drained": True})
        return True, detail

    def _wait_detached_pipeline_contexts(self, timeout=10.0):
        deadline = time.monotonic() + max(0.1, float(timeout))
        with self.lock:
            contexts = list(self.detached_pipeline_contexts)
        remaining_contexts = []
        for context in contexts:
            for thread in list(context.threads):
                if thread is threading.current_thread():
                    continue
                thread.join(max(0.05, deadline - time.monotonic()))
            alive = [thread for thread in context.threads if thread is not threading.current_thread() and thread.is_alive()]
            queues_left = context.capture_queue.qsize() + context.feature_queue.qsize() + context.persist_queue.qsize()
            if alive or queues_left:
                remaining_contexts.append(context)
            else:
                context.closed = True
                context.stop_event.set()
                context.drain_complete.set()
        with self.lock:
            self.detached_pipeline_contexts = remaining_contexts
        return not remaining_contexts

    def _close_active_session(self, reason, barrier=True):
        with self.lock:
            session_id = self.session_id
            context = self.pipeline_context
        barrier_ok = True
        barrier_detail = ""
        if session_id:
            try:
                if barrier:
                    barrier_ok, barrier_detail = self._write_barrier(session_id, reason, context)
                if not barrier_ok:
                    self.store.mark_session_untrainable(session_id, barrier_detail)
                self.store.add_system_event(session_id, "mode_exit", {"reason": reason, "time": time.time(), "barrier_ok": barrier_ok, "barrier_detail": barrier_detail})
                self.store.close_session(session_id, reason if barrier_ok else reason + "；" + barrier_detail)
            except Exception as error:
                barrier_ok = False
                barrier_detail = str(error)
                try:
                    self.store.mark_session_untrainable(session_id, barrier_detail)
                    self.store.close_session(session_id, reason + "；关闭期间异常：" + barrier_detail)
                except Exception:
                    pass
                self.emit("notice", barrier_detail)
        self.window_guard.stop()
        with self.lock:
            if self.session_id == session_id:
                self.session_id = None
                self.session_mode = None
                self.target_hwnd = None
                self.target_root = None
                self.target_pid = 0
                self.target_process_path = ""
                self.target_rect = None
                self.pipeline_context = None
                self.resources.set_emulator_pid(0)
        return barrier_ok, barrier_detail

    def _perform_stop(self, reason, token=None):
        if threading.current_thread() is not self.control_thread:
            self.on_control_signal("stop", reason, token=token)
            return False
        raw_reason = str(reason or "")
        public_reason = self._public_stop_reason(raw_reason)
        with self.lock:
            if token is not None and token != self.epoch:
                return False
            if self.state == "idle":
                return False
            if self.state == "stopping":
                return True
            previous = self.state
            try:
                self.store.add_system_event(self.session_id, "mode_stop_internal_reason", {"public_reason": public_reason, "internal_reason": raw_reason, "state": previous, "token": token, "time": time.time()})
            except Exception:
                pass
            self.stop_requested.set()
            self.cancel_event.set()
            context = self.pipeline_context
            if previous in ("learning", "training"):
                self._transition_state_locked(self._event_for_stop_reason(raw_reason), previous, "stopping", public_reason, token)
                if context is not None:
                    context.accepting = False
                    context.draining = True
            elif previous == "sleep":
                self._transition_state_locked("esc", "sleep", "stopping", public_reason, token)
            else:
                self._transition_state_locked("client_invalid", previous, "idle", public_reason, token)
        clean = True
        detail = ""
        if previous in ("learning", "training"):
            self.hook.stop()
            self.keyboard_hook.stop()
            self.post_state("正在停止接入并排空有效样本记录队列")
            clean, detail = self._close_active_session(public_reason, barrier=True)
        elif previous == "sleep":
            self.keyboard_hook.stop()
        with self.lock:
            if not clean:
                self.recovery_pending = True
                self.recovery_reason = detail or "停止期间数据恢复待处理"
            self.epoch += 1
            if previous in ("learning", "training", "sleep"):
                self._transition_state_locked(self._event_for_stop_reason(raw_reason) if previous in ("learning", "training") else "esc", "stopping", "idle", public_reason, token)
            else:
                self._transition_state_locked("client_invalid", previous, "idle", public_reason, token)
            self.stop_requested.clear()
        self.ai_input_tracker.clear()
        self.emit("progress", 0.0)
        final_detail = public_reason if previous != "sleep" else "睡眠模式已中止：" + public_reason
        if not clean:
            final_detail += "；已进入空闲/恢复待处理，当前会话已标记不可训练"
        self.post_state(final_detail)
        return clean

    def request_idle(self, reason, token=None):
        self.on_control_signal("stop", reason, token=token)
        return True

    def start_sleep(self):
        with self.lock:
            if self.state != "idle":
                self.emit("notice", "仅空闲状态可手动进入睡眠模式。")
                return False
        try:
            self.ensure_store()
        except Exception as error:
            self._remember_critical_exception("DataStore", "ensure_store", error, resource_state=self.resources.capture_snapshot().state)
            self.emit("notice", "无法创建存储路径：" + str(error))
            return False
        if not self.keyboard_hook.start():
            try:
                self.store.add_critical_exception("KeyboardHook", "start_sleep", RuntimeError(self.keyboard_hook.error or "键盘钩子未启动"), resource_state=self.resources.capture_snapshot().state)
            except Exception:
                pass
            self.emit("notice", self.keyboard_hook.error or "键盘钩子未启动，禁止进入睡眠模式。")
            return False
        with self.lock:
            self.epoch += 1
            token = self.epoch
            self.cancel_event = threading.Event()
            self.sleep_origin = "manual"
            self.sleep_task1_done = False
            self.sleep_task2_done = False
            self._transition_state_locked("manual_sleep", "idle", "sleep", "manual_sleep", token)
        self.post_state("已进入睡眠模式")
        threads = [threading.Thread(target=self._sleep_monitor, args=(token,), name="SleepMonitor"), threading.Thread(target=self._sleep_worker, args=(token, False), name="SleepWorker")]
        with self.lock:
            self.worker_threads = [thread for thread in self.worker_threads if thread.is_alive()] + threads
        for thread in threads:
            thread.start()
        return True

    def _begin_auto_sleep(self, token):
        if threading.current_thread() is not self.control_thread:
            self.on_control_signal("auto_sleep", "AI 判断进入睡眠模式", token=token)
            return
        with self.lock:
            if token != self.epoch or self.state != "training":
                return
            context = self.pipeline_context
            self._transition_state_locked("auto_sleep_worth", "training", "stopping", "auto_sleep_worth", token)
            self.cancel_event.set()
            if context is not None:
                context.accepting = False
                context.draining = True
        self.hook.stop()
        self.post_state("AI 正在停止新截图并排空现有会话数据")
        closed, detail = self._close_active_session("AI 判断进入睡眠模式", barrier=True)
        if not closed:
            with self.lock:
                self.recovery_pending = True
                self.recovery_reason = detail or "自动睡眠前写入屏障未完成"
                self._transition_state_locked("client_invalid", "stopping", "idle", "auto_sleep_cancelled", token)
                self.epoch += 1
            self.post_state("空闲；自动睡眠取消，数据恢复待处理：" + self.recovery_reason)
            return
        with self.lock:
            self.epoch += 1
            sleep_token = self.epoch
            self.cancel_event = threading.Event()
            now_auto_sleep = time.time()
            self.training_auto_sleep_count += 1
            self.last_auto_sleep_at = now_auto_sleep
            self.auto_sleep_times = [item for item in self.auto_sleep_times if now_auto_sleep - float(item) < 3600.0] + [now_auto_sleep]
            self.last_auto_sleep_fingerprint = self.current_training_fingerprint
            self.sleep_origin = "auto"
            self.sleep_task1_done = False
            self.sleep_task2_done = False
            try:
                self.last_auto_sleep_pool_bytes = int(self.store.pool_size_fast())
            except Exception:
                self.last_auto_sleep_pool_bytes = 0
            self._transition_state_locked("auto_sleep_worth", "stopping", "sleep", "auto_sleep_worth", sleep_token)
        self.post_state("AI 判断当前值得进入睡眠模式")
        threads = [threading.Thread(target=self._sleep_monitor, args=(sleep_token,), name="AutoSleepMonitor"), threading.Thread(target=self._sleep_worker, args=(sleep_token, True), name="AutoSleepWorker")]
        with self.lock:
            self.worker_threads = [thread for thread in self.worker_threads if thread.is_alive()] + threads
        for thread in threads:
            thread.start()

    def _sleep_monitor(self, token):
        while self._is_current(token, ("sleep",)):
            if self.keyboard_hook_escape_pressed():
                self.request_idle("检测到 ESC 键", token)
                return
            time.sleep(0.08)

    def _cancelled(self, token):
        return not self._is_current(token, ("sleep",))

    def _wait_resource(self, token, purpose="training"):
        task = "maintenance" if purpose == "maintenance" else "sleep_training"
        while not self._cancelled(token):
            budget = self.resources.acquire(task)
            if budget.allowed:
                return True
            if budget.must_pause:
                if task == "maintenance":
                    sample = self.resources.sample()
                    self.emit("state", {"state": "sleep", "detail": budget.pause_reason or "维护预算暂停，等待内存/磁盘/SQLite 恢复，按 ESC 才能空闲", "cpu": sample["cpu"], "memory": sample["memory"]})
                    time.sleep(max(1.0, budget.next_interval))
                    continue
                try:
                    self.store.add_system_event(None, "sleep_task1_budget_blocked", {"internal_reason": "resource_redline_training", "pause_reason": budget.pause_reason or "资源红线", "resource": self.resources.sample(), "time": time.time()})
                except Exception:
                    pass
                sample = self.resources.sample()
                self.emit("state", {"state": "sleep", "detail": "任务1训练预算暂停，本轮无新模型并进入任务2清理", "cpu": sample["cpu"], "memory": sample["memory"]})
                return False
            sample = self.resources.sample()
            self.emit("state", {"state": "sleep", "detail": budget.pause_reason or "资源预算要求暂停睡眠任务", "cpu": sample["cpu"], "memory": sample["memory"]})
            time.sleep(max(0.05, budget.next_interval))
        return False

    def _trajectory_features(self, history, now_ns):
        recent = [row for row in history if int(now_ns) - int(row[2]) <= 1_000_000_000]
        if not recent:
            return {"speed_mean": 0.0, "speed_max": 0.0, "acceleration_mean": 0.0, "dwell_ms": 1000.0, "turns": 0, "path_length": 0.0, "cursor_stability": 1.0}
        speeds = [max(0.0, float(row[8] or 0.0)) for row in recent]
        accelerations = []
        turns = 0
        path_length = 0.0
        for before, after in zip(recent, recent[1:]):
            dt = max(1e-9, (int(after[2]) - int(before[2])) / 1_000_000_000.0)
            accelerations.append(abs(float(after[8] or 0.0) - float(before[8] or 0.0)) / dt)
            path_length += math.hypot(float(after[9] or 0.0), float(after[10] or 0.0))
            turn = abs(math.atan2(math.sin(float(after[11] or 0.0) - float(before[11] or 0.0)), math.cos(float(after[11] or 0.0) - float(before[11] or 0.0))))
            if turn >= 0.65:
                turns += 1
        dwell_ms = max(0.0, (int(now_ns) - int(recent[-1][2])) / 1_000_000.0)
        stability = max(0.0, min(1.0, 1.0 - min(1.0, (sum(speeds) / max(1, len(speeds))) / 2000.0)))
        return {"speed_mean": sum(speeds) / max(1, len(speeds)), "speed_max": max(speeds or [0.0]), "acceleration_mean": sum(accelerations) / max(1, len(accelerations)), "dwell_ms": dwell_ms, "turns": turns, "path_length": path_length, "cursor_stability": stability}

    def _semantic_actions(self, mouse_rows):
        actions = []
        down = {}
        wheel_bucket = None
        last_move = None
        history = []
        for row in mouse_rows:
            row = tuple(row)
            if len(row) == 14:
                row = row + (None,)
            if len(row) < 15:
                continue
            if len(row) > 15:
                row = row[:15]
            mid, sid, created_ns, created, event_type, source, rx, ry, speed, dx, dy, direction, button, wheel, behavior_probability = row
            if source not in ("user", "ai", "用户", "AI") or rx is None or ry is None:
                continue
            ns = int(created_ns)
            if event_type == "move":
                history.append(row)
                if len(history) > 96:
                    history = history[-96:]
            trajectory = self._trajectory_features(history, ns)
            if event_type == "button_down" and button in ("left", "right"):
                down[button] = row
            elif event_type == "button_up" and button in down:
                d = down.pop(button)
                duration = ns - int(d[2])
                distance = math.hypot(float(rx) - float(d[6]), float(ry) - float(d[7]))
                if 0 <= duration <= 1_500_000_000 and distance <= 0.08:
                    action_id = "click:{}:{}:{}".format(sid, d[0], mid)
                    actions.append({"action_id": action_id, "mouse_event_id": mid, "session_id": sid, "action_time": ns, "source": source, "rx": float(rx), "ry": float(ry), "action_type": "左键" if button == "left" else "右键", "trajectory": trajectory, "behavior_probability": behavior_probability})
            elif event_type == "wheel" and wheel:
                axis = "horizontal" if button == "horizontal" else "vertical"
                if wheel_bucket and ns - wheel_bucket["last_ns"] <= 180_000_000 and sid == wheel_bucket["session_id"] and axis == wheel_bucket["wheel_axis"]:
                    wheel_bucket["last_ns"] = ns
                    wheel_bucket["signed_delta"] += int(wheel)
                    wheel_bucket["step_count"] += 1
                    wheel_bucket["duration_ms"] = max(0.0, (ns - wheel_bucket["action_time"]) / 1_000_000.0)
                    wheel_bucket["mouse_event_id"] = mid
                    wheel_bucket["rx"] = float(rx)
                    wheel_bucket["ry"] = float(ry)
                    wheel_bucket["trajectory"] = trajectory
                    wheel_bucket["behavior_probability"] = behavior_probability if behavior_probability is not None else wheel_bucket.get("behavior_probability")
                else:
                    if wheel_bucket:
                        wheel_bucket["action_type"] = "水平滚轮" if wheel_bucket["wheel_axis"] == "horizontal" else "滚轮"
                        wheel_bucket["wheel_delta"] = max(-1200, min(1200, int(wheel_bucket["signed_delta"])))
                        actions.append(wheel_bucket)
                    wheel_bucket = {"action_id": "wheel:{}:{}".format(sid, mid), "mouse_event_id": mid, "session_id": sid, "action_time": ns, "last_ns": ns, "source": source, "rx": float(rx), "ry": float(ry), "wheel_axis": axis, "signed_delta": int(wheel), "step_count": 1, "duration_ms": 0.0, "trajectory": trajectory, "behavior_probability": behavior_probability}
            elif event_type == "move":
                keep = last_move is None or ns - int(last_move[2]) >= 350_000_000 or abs(float(direction) - float(last_move[11])) >= 0.85 or math.hypot(float(rx) - float(last_move[6]), float(ry) - float(last_move[7])) >= 0.10
                if keep:
                    actions.append({"action_id": "move:{}:{}".format(sid, mid), "mouse_event_id": mid, "session_id": sid, "action_time": ns, "source": source, "rx": float(rx), "ry": float(ry), "action_type": "移动", "trajectory": trajectory, "behavior_probability": behavior_probability})
                    last_move = row
        if wheel_bucket:
            wheel_bucket["action_type"] = "水平滚轮" if wheel_bucket["wheel_axis"] == "horizontal" else "滚轮"
            wheel_bucket["wheel_delta"] = max(-1200, min(1200, int(wheel_bucket["signed_delta"])))
            actions.append(wheel_bucket)
        return actions

    def _record_training_failure(self, fingerprint, reason):
        self.last_training_attempt_fingerprint = fingerprint
        self.last_training_failure_reason = str(reason)
        self.training_retry_count = min(3, int(self.training_retry_count) + 1)
        delay = (300.0, 900.0, 3600.0)[self.training_retry_count - 1]
        self.next_training_retry_at = time.time() + delay
        self.store.add_system_event(None, "model_training_failed", {"reason": str(reason), "fingerprint": fingerprint, "retry_seconds": delay, "time": time.time()})

    def _save_layered_model_payload(self, payload, outcomes, validation_outcomes, layer):
        if not isinstance(payload, dict) or self.store.models is None:
            return None
        payload["model_layer"] = str(layer)
        payload["champion"] = False
        payload["saved_at"] = time.time()
        payload.setdefault("id", uuid.uuid4().hex)
        name = "model_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + str(layer) + "_" + str(payload["id"])[:8] + ".json"
        final_path = self.store._assert_storage_path(self.store.models / name, "model_json_layered")
        temp_path = self.store._assert_storage_path(final_path.with_suffix(".tmp"), "model_json_layered_tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(final_path)
        self.store._fsync_directory(final_path.parent)
        self.store.register_model_metadata(payload["id"], final_path, payload, outcomes, validation_outcomes)
        return final_path

    def _training_result(self, status, reason="", model=None, training_samples=0, validation_samples=0, quality_delta=0.0, champion_persisted=False):
        return {
            "status": str(status),
            "reason": str(reason or ""),
            "model": model if isinstance(model, dict) else None,
            "training_samples": int(training_samples or 0),
            "validation_samples": int(validation_samples or 0),
            "quality_delta": float(quality_delta or 0.0),
            "champion_persisted": bool(champion_persisted),
        }

    def _sleep_task1_terminal_result(self, result):
        result = result if isinstance(result, dict) else self._training_result("failed", "训练返回值无效")
        status = str(result.get("status", ""))
        if status == "trained":
            result["terminal_status"] = "trained"
            result["new_model_created"] = True
            return result
        champion = self.store.best_model()
        champion_layer = str(champion.get("model_layer", "")) if isinstance(champion, dict) else ""
        champion_verified = bool(isinstance(champion, dict) and (champion.get("onnx_policy_verified") is True or champion.get("champion") is True or (champion.get("control_enabled") is True and champion_layer in ("move", "click_wheel"))))
        if champion_verified:
            terminal = self._training_result("reused_best_model", "本轮无新模型；复用当前已验证冠军模型。原始状态：{}；{}".format(status or "unknown", result.get("reason", "")), champion, result.get("training_samples", 0), result.get("validation_samples", 0), 0.0, False)
            terminal["terminal_status"] = "reused_best_model"
            terminal["original_status"] = status
            terminal["new_model_created"] = False
            return terminal
        terminal = self._training_result("no_trainable_data", "没有可训练数据且没有可复用冠军模型。原始状态：{}；{}".format(status or "unknown", result.get("reason", "")), None, result.get("training_samples", 0), result.get("validation_samples", 0), 0.0, False)
        terminal["terminal_status"] = "no_trainable_data"
        terminal["original_status"] = status
        terminal["new_model_created"] = False
        return terminal

    def _visual_policy_target(self, example):
        score_delta = float(example.get("score_delta", 0.0) or 0.0)
        baseline_delta = float(example.get("baseline_score_delta", 0.0) or 0.0)
        advantage = float(example.get("action_advantage", score_delta - baseline_delta) or 0.0)
        stability = _clamp_float(example.get("stability", 0.0), 0.0, 1.0)
        baseline = max(0.0, float(example.get("baseline_count", 0.0) or 0.0))
        confidence = _clamp_float(0.5 + 0.44 * math.tanh(advantage / 0.08) + 0.08 * math.tanh(score_delta / 0.08), 0.02, 0.98)
        uncertainty = _clamp_float(0.08 + 0.62 * (1.0 - stability) + 0.20 / math.sqrt(1.0 + baseline) + 0.10 * (1.0 - min(1.0, abs(advantage) / 0.08)), 0.03, 0.95)
        weight = _clamp_float(0.25 + 0.75 * stability + min(1.0, baseline / 8.0) * 0.5 + min(1.0, abs(advantage) / 0.12) * 0.5, 0.05, 2.0)
        return confidence, uncertainty, weight

    def _visual_policy_examples(self, items):
        result = []
        for item in items or []:
            example = item[-1] if isinstance(item, tuple) else item
            if not isinstance(example, dict) or not example.get("outcome_valid"):
                continue
            gray = feature_bytes(example.get("gray32x18"), 32 * 18)
            if not gray:
                continue
            try:
                rx = _clamp_float(example.get("action_rx", example.get("rx", 0.5)), 0.0, 1.0)
                ry = _clamp_float(example.get("action_ry", example.get("ry", 0.5)), 0.0, 1.0)
                confidence, uncertainty, weight = self._visual_policy_target(example)
                result.append({"gray": gray.hex(), "color_histogram": histogram_blob(example.get("color_histogram")), "edge_density": float(example.get("edge_density", 0.0) or 0.0), "rx": rx, "ry": ry, "action_rx": rx, "action_ry": ry, "action_type": str(example.get("action_type", "移动")), "wheel_delta": int(example.get("wheel_delta", 0) or 0), "confidence": confidence, "uncertainty": uncertainty, "weight": weight, "advantage": float(example.get("action_advantage", 0.0) or 0.0), "score_delta": float(example.get("score_delta", 0.0) or 0.0), "screen_score_delta": float(example.get("score_delta", 0.0) or 0.0), "split_role": str(example.get("split_role", "")), "action_id": str(example.get("action_id", example.get("mouse_event_id", "")))})
            except Exception:
                continue
        return result

    def _train_visual_policy_weights(self, train_items, validation_items, token):
        train_examples = self._visual_policy_examples(train_items)
        validation_examples = self._visual_policy_examples([item[1] if isinstance(item, tuple) and len(item) == 2 else item for item in validation_items or []])
        if len(train_examples) < 16 or len(validation_examples) < 4:
            return None, "视觉策略样本不足：训练 {}，验证 {}".format(len(train_examples), len(validation_examples))
        train_examples = sorted(train_examples, key=lambda item: item.get("action_id", ""))[-512:]
        prepared = []
        output_sum = [0.0] * POLICY_OUTPUT_SIZE
        weight_sum = 0.0
        signal = 0.0
        for item in train_examples:
            vector = visual_policy_input_vector(item)
            target = visual_policy_target_vector(item)
            weight = float(item.get("weight", 1.0))
            if len(vector) != POLICY_INPUT_SIZE:
                continue
            prepared.append((vector, target, weight, item))
            weight_sum += weight
            signal += abs(float(item.get("advantage", 0.0) or 0.0)) * weight + abs(float(item.get("score_delta", 0.0) or 0.0)) * weight + float(item.get("confidence", 0.0) or 0.0) * 0.05
            for index, value in enumerate(target):
                output_sum[index] += value * weight
        if len(prepared) < 16:
            return None, "视觉策略特征不足"
        if signal <= 1e-6:
            return None, "视觉样本没有可学习的 screen_score_delta 信号"
        bias = [output_sum[index] / max(1e-9, weight_sum) for index in range(POLICY_OUTPUT_SIZE)]
        fc_weight = [0.0] * (POLICY_INPUT_SIZE * POLICY_OUTPUT_SIZE)
        for row_index, (vector, target, weight, item) in enumerate(prepared):
            if self._cancelled(token):
                return None, "训练被取消"
            if row_index % 16 == 0:
                budget = self.resources.acquire("sleep_training")
                if not budget.allowed:
                    return None, "视觉策略训练资源预算不足"
            try:
                action_index = POLICY_ACTION_TYPES.index(str(item.get("action_type", "移动")))
            except ValueError:
                action_index = 0
            gx = max(0, min(POLICY_GRID_WIDTH - 1, int(float(item.get("rx", 0.5)) * POLICY_GRID_WIDTH)))
            gy = max(0, min(POLICY_GRID_HEIGHT - 1, int(float(item.get("ry", 0.5)) * POLICY_GRID_HEIGHT)))
            active_outputs = list(range(len(POLICY_ACTION_TYPES))) + [len(POLICY_ACTION_TYPES) + gy * POLICY_GRID_WIDTH + gx, POLICY_OUTPUT_SIZE - 3, POLICY_OUTPUT_SIZE - 2, POLICY_OUTPUT_SIZE - 1]
            for output_index in active_outputs:
                error = (target[output_index] - bias[output_index]) * weight
                if abs(error) <= 1e-9:
                    continue
                for input_index, value in enumerate(vector):
                    if value:
                        weight_index = input_index * POLICY_OUTPUT_SIZE + output_index
                        fc_weight[weight_index] += error * value
        scale = max([abs(value) for value in fc_weight] + [1e-9])
        fc_weight = [_clamp_float(value / scale * 0.85, -0.85, 0.85) for value in fc_weight]
        validation_rows = []
        action_hits = 0
        position_hits = 0
        mae_unc = 0.0
        for item in validation_examples:
            vector = visual_policy_input_vector(item)
            logits = list(bias)
            for input_index, value in enumerate(vector):
                if not value:
                    continue
                base = input_index * POLICY_OUTPUT_SIZE
                for output_index in range(POLICY_OUTPUT_SIZE):
                    logits[output_index] += value * fc_weight[base + output_index]
            parsed = parse_visual_policy_output(logits)
            try:
                expected_action = POLICY_ACTION_TYPES.index(str(item.get("action_type", "移动")))
            except ValueError:
                expected_action = 0
            expected_gx = max(0, min(POLICY_GRID_WIDTH - 1, int(float(item.get("rx", 0.5)) * POLICY_GRID_WIDTH)))
            expected_gy = max(0, min(POLICY_GRID_HEIGHT - 1, int(float(item.get("ry", 0.5)) * POLICY_GRID_HEIGHT)))
            predicted_gx = int(parsed["grid_index"]) % POLICY_GRID_WIDTH
            predicted_gy = int(parsed["grid_index"]) // POLICY_GRID_WIDTH
            action_hits += int(int(parsed["action_index"]) == expected_action)
            position_hits += int(abs(predicted_gx - expected_gx) <= 1 and abs(predicted_gy - expected_gy) <= 1)
            mae_unc += abs(float(parsed["uncertainty"]) - float(item.get("uncertainty", 0.5)))
            validation_rows.append(parsed)
        if len(validation_rows) < 4:
            return None, "视觉策略验证样本不足"
        action_hit_rate = action_hits / max(1, len(validation_rows))
        position_hit_rate = position_hits / max(1, len(validation_rows))
        mae_unc /= len(validation_rows)
        if action_hit_rate < 0.15 and position_hit_rate < 0.15:
            return None, "视觉策略验证未通过：动作命中 {:.1%}，位置邻域命中 {:.1%}".format(action_hit_rate, position_hit_rate)
        weights = {"input_size": POLICY_INPUT_SIZE, "output_size": POLICY_OUTPUT_SIZE, "fc_weight": [round(float(v), 7) for v in fc_weight], "bias": [round(float(v), 7) for v in bias], "action_types": list(POLICY_ACTION_TYPES), "grid": [POLICY_GRID_WIDTH, POLICY_GRID_HEIGHT]}
        metrics = {"train_examples": len(prepared), "validation_examples": len(validation_rows), "validation_action_hit_rate": action_hit_rate, "validation_position_hit_rate": position_hit_rate, "validation_uncertainty_mae": mae_unc, "input": "gray32x18+color_histogram+edge_density+recent_action_summary", "target": "action_type_logits+10x6_position_logits+wheel_delta+value+uncertainty", "reward_target": "screen_score_delta_only", "split": "session_or_time_block"}
        return {"weights": weights, "metrics": metrics}, ""

    def _export_policy_onnx(self, payload, path):
        try:
            weights = (payload or {}).get("visual_policy_weights") or {}
            fc_weight = weights.get("fc_weight")
            bias = weights.get("bias")
            path = self.store._assert_storage_path(path, "onnx_export")
            path.parent.mkdir(parents=True, exist_ok=True)
            if int(weights.get("output_size", 0) or 0) >= POLICY_OUTPUT_SIZE and int(weights.get("input_size", 0) or 0) == POLICY_INPUT_SIZE:
                temporary_bytes = policy_multihead_onnx_bytes(fc_weight, bias)
                payload["onnx_input_schema"] = ["gray32x18_normalized+color_histogram_normalized+edge_density+recent_action_summary", "NC", 1, POLICY_INPUT_SIZE]
                payload["onnx_output_schema"] = {"action_type_logits": [0, len(POLICY_ACTION_TYPES)], "position_logits_10x6": [len(POLICY_ACTION_TYPES), len(POLICY_ACTION_TYPES) + POLICY_GRID_WIDTH * POLICY_GRID_HEIGHT], "wheel_delta_head": POLICY_OUTPUT_SIZE - 3, "value_head": POLICY_OUTPUT_SIZE - 2, "uncertainty_head": POLICY_OUTPUT_SIZE - 1}
                payload["onnx_model_kind"] = "trained_multihead_visual_mouse_policy"
            else:
                conv_weight = weights.get("conv_weight")
                conv_bias = weights.get("conv_bias")
                temporary_bytes = policy_onnx_bytes(conv_weight, fc_weight, bias, conv_bias)
                payload["onnx_input_schema"] = ["gray32x18_normalized", "NCHW", 1, 18, 32]
                payload["onnx_output_schema"] = ["confidence", "uncertainty"]
                payload["onnx_model_kind"] = "trained_visual_confidence_fallback_policy"
            with path.open("wb") as handle:
                handle.write(temporary_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            payload["onnx_exported"] = True
            return True
        except Exception as error:
            payload["onnx_exported"] = False
            payload["onnx_export_error"] = str(error)
            try:
                self.store.add_system_event(None, "onnx_export_failed", {"reason": str(error), "time": time.time()})
            except Exception:
                pass
            return False

    def _verify_policy_onnx(self, path, payload):
        started = time.perf_counter()
        try:
            import onnxruntime as ort
            import numpy as np
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if len(inputs) != 1 or len(outputs) < 1:
                raise RuntimeError("ONNX 输入输出数量不符合策略 schema")
            sample = inputs[0]
            shape = [1 if not isinstance(dim, int) or dim <= 0 else int(dim) for dim in sample.shape]
            if len(shape) == 2 and shape[1] == POLICY_INPUT_SIZE:
                result = session.run(None, {sample.name: np.full(shape, 0.5, dtype=np.float32)})
                values = np.asarray(result[0], dtype=np.float32).reshape(-1)
                if values.size < POLICY_OUTPUT_SIZE:
                    raise RuntimeError("ONNX 多头输出数量不足")
                parsed = parse_visual_policy_output(values)
                confidence = float(parsed["confidence"])
                uncertainty = float(parsed["uncertainty"])
            elif len(shape) == 4 and shape[1:] == [1, 18, 32]:
                result = session.run(None, {sample.name: np.full(shape, 0.5, dtype=np.float32)})
                values = np.asarray(result[0], dtype=np.float32).reshape(-1)
                if values.size < 2:
                    raise RuntimeError("ONNX 输出数量不足")
                confidence = float(values[0])
                uncertainty = float(values[1])
            else:
                raise RuntimeError("ONNX 输入 shape 不符合视觉策略 schema")
            if not (math.isfinite(confidence) and math.isfinite(uncertainty) and 0.0 <= confidence <= 1.0 and 0.0 <= uncertainty <= 1.0):
                raise RuntimeError("ONNX 输出范围异常")
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if elapsed_ms > 200.0:
                raise RuntimeError("ONNX 自测耗时过高")
            payload["onnx_policy_verified"] = True
            payload["onnx_self_test"] = {"confidence": confidence, "uncertainty": uncertainty, "elapsed_ms": elapsed_ms, "fallback": "CPU 表格策略", "model_kind": payload.get("onnx_model_kind", "")}
            return True
        except Exception as error:
            payload["onnx_policy_verified"] = False
            payload["onnx_self_test_error"] = str(error)
            try:
                self.store.add_system_event(None, "onnx_self_test_failed", {"reason": str(error), "time": time.time()})
            except Exception:
                pass
            return False

    def _train_model(self, token):
        self.flush_mouse_records()
        if not self._wait_resource(token):
            return self._training_result("cancelled" if self._cancelled(token) else "skipped", "资源预算不允许训练")
        frames_by_session, mouse_by_session = self.store.collect_training_data()
        fingerprint=self.store.training_data_fingerprint()
        now_attempt=time.time()
        if fingerprint == self.last_successful_training_fingerprint:
            self.store.add_system_event(None, "model_skipped", {"reason": "训练数据自上次成功后未变化", "fingerprint": fingerprint, "time": now_attempt})
            return self._training_result("skipped", "训练数据自上次成功后未变化", self.store.best_model())
        if now_attempt < self.next_training_retry_at and fingerprint == self.last_training_attempt_fingerprint:
            self.store.add_system_event(None, "model_skipped", {"reason": "训练失败退避中", "fingerprint": fingerprint, "retry_at": self.next_training_retry_at, "time": now_attempt})
            return self._training_result("skipped", "训练失败退避中", self.store.best_model())
        self.last_training_attempt = now_attempt
        self.last_training_attempt_fingerprint = fingerprint
        ordered_sessions=sorted(frames_by_session, key=lambda sid: frames_by_session[sid][0][3] if frames_by_session.get(sid) else 0)
        single_session_split = len(ordered_sessions) == 1
        validation_sessions=set(ordered_sessions[-max(1,math.ceil(len(ordered_sessions)*0.3)):]) if len(ordered_sessions)>=2 else set()
        outcomes=[];validation_outcomes=[];persisted=[];states={};all_actions_count=0;validation_blocks=[]
        for session_id,mouse_rows in mouse_by_session.items():
            frame_rows=frames_by_session.get(session_id,[])
            if len(frame_rows)<4: continue
            semantic_actions=self._semantic_actions(mouse_rows);all_actions_count+=len(semantic_actions)
            frame_finish=[int(row[15] or row[2]) for row in frame_rows];frame_start=[int(row[14] or row[2]) for row in frame_rows]
            critical=sorted(int(row[2]) for row in mouse_rows if row[4] in ("button_down","button_up","wheel"))
            start_ns,end_ns=frame_start[0],frame_finish[-1];gap_ns=min(3_000_000_000,max(250_000_000,int((end_ns-start_ns)*0.03)));cut=start_ns+int((end_ns-start_ns)*0.7)
            def split_role_at(ns):
                if session_id in validation_sessions:
                    return "validation"
                if single_session_split:
                    if ns <= cut - gap_ns:
                        return "train"
                    if ns >= cut + gap_ns:
                        return "validation"
                    return "excluded_gap"
                if ns <= cut - gap_ns:
                    return "train"
                return "excluded_gap"
            baselines={}
            for i,before in enumerate(frame_rows[:-1]):
                before_time=frame_finish[i]
                j=bisect_right(frame_start,before_time+250_000_000-1)
                if j>=len(frame_rows) or frame_start[j]>before_time+3_000_000_000: continue
                transition_end=frame_start[j]
                baseline_role=split_role_at(transition_end)
                if baseline_role=="excluded_gap": continue
                if any(before_time<t<transition_end for t in critical): continue
                if not before[16] or not frame_rows[j][16]: continue
                state_key=self.store.assign_state_cluster(before[4] or before[5],before[9],before[10],before[11],before[12])
                baselines.setdefault(state_key,[]).append(((transition_end-before_time)/1_000_000.0, float(frame_rows[j][6])-float(before[6]), float(frame_rows[j][7])-float(before[7]), transition_end, baseline_role))
            def stability(index):
                origin=frame_rows[index][4] or frame_rows[index][5]
                if not origin:return 0.0
                values=[]
                for nearby in (index-2,index-1,index+1,index+2):
                    if 0<=nearby<len(frame_rows):
                        other=frame_rows[nearby][4] or frame_rows[nearby][5]
                        try:values.append(bit_count(int(origin,16)^int(other,16))/64.0)
                        except Exception as error:self.store.record_exception_event(session_id, "training_stability_hash_failed", error, {"frame_index": index, "nearby": nearby})
                return max(0.0,min(1.0,1.0-(sum(values)/max(1,len(values))) if values else 0.0))
            for index,action in enumerate(semantic_actions):
                training_budget=self.resources.acquire("sleep_training")
                if index % max(8, training_budget.training_block_size) == 0 and not self._wait_resource(token):
                    return self._training_result("cancelled" if self._cancelled(token) else "skipped", "训练块资源预算不足", training_samples=len(outcomes), validation_samples=len(validation_outcomes))
                action_ns=int(action["action_time"]);before_i=bisect_right(frame_finish,action_ns)-1;after_i=bisect_right(frame_start,action_ns+250_000_000-1)
                if before_i<0 or after_i>=len(frame_rows) or frame_start[after_i]>action_ns+3_000_000_000: continue
                if any(action_ns<t<frame_start[after_i] for t in critical): continue
                before,after=frame_rows[before_i],frame_rows[after_i]
                if not before[16] or not after[16]: continue
                post_ms=(frame_start[after_i]-action_ns)/1_000_000.0
                state_key=self.store.assign_state_cluster(before[4] or before[5],before[9],before[10],before[11],before[12])
                role=split_role_at(action_ns)
                candidates=[item for item in baselines.get(state_key,[]) if item[3] < action_ns and item[4] == role]
                nearest=sorted(candidates,key=lambda item:abs(item[0]-post_ms))[:12]
                baseline_score_delta=sum(item[1] for item in nearest)/max(1,len(nearest)) if nearest else 0.0
                stable=min(stability(before_i),stability(after_i))
                stable_required=0.45 if action["action_type"]=="移动" else 0.65
                score_delta=float(after[6])-float(before[6]);advantage=score_delta-baseline_score_delta
                gx=min(15,max(0,int(float(action["rx"])*16)));gy=min(8,max(0,int(float(action["ry"])*9)))
                local_descriptor=local_visual_descriptor(before[11],before[13],before[12],action["rx"],action["ry"],3)
                local_key=local_descriptor.get("gray","")[:48]
                source_clear=action.get("source") in ("user","ai","用户","AI")
                post_delay_ok=80.0<=post_ms<=2000.0
                post_fresh=0<frame_start[after_i]-action_ns<=2_000_000_000
                propensity_known=action.get("behavior_probability") is not None
                non_move_propensity_ok=action["action_type"]=="移动" or (action.get("source") in ("ai","AI") and propensity_known)
                causal_policy_eligible=role=="train" and stable>=stable_required and source_clear and post_delay_ok and post_fresh and len(nearest)>=2 and non_move_propensity_ok
                observation_only=not causal_policy_eligible
                example={"action_id":action["action_id"],"session_id":session_id,"before_frame_id":before[0],"after_frame_id":after[0],"mouse_event_id":action["mouse_event_id"],"action_time":action_ns,"post_action_delay_ms":post_ms,"score_delta":score_delta,"baseline_score_delta":baseline_score_delta,"action_advantage":advantage,"stability":stable,"baseline_count":len(nearest),"trajectory":dict(action.get("trajectory") or {}),"source":action.get("source"),"behavior_probability":action.get("behavior_probability"),"causal_policy_eligible":causal_policy_eligible,"observation_only":observation_only,"causal_claim":"not_claimed_without_propensity","local_descriptor":local_descriptor,"gray32x18":before[11],"edge_density":before[12],"color_histogram":before[13],"action_rx":float(action["rx"]),"action_ry":float(action["ry"]),"outcome_valid":role!="excluded_gap" and stable>=stable_required and source_clear and post_delay_ok and post_fresh,"split_role":role}
                persisted.append(example)
                wheel_axis=action.get("wheel_axis","");signed=int(action.get("wheel_delta",action.get("signed_delta",0)) or 0);wheel_direction=1 if signed>0 else -1 if signed<0 else 0;wheel_bucket=min(10,abs(signed)//120)
                key=(state_key,gx,gy,action["action_type"],wheel_axis,wheel_direction,wheel_bucket,local_key)
                if role=="train" and stable>=stable_required and (action["action_type"]=="移动" or causal_policy_eligible):
                    item=states.setdefault(key,{"samples":0,"human":0,"ai":0,"sum":0.0,"sum2":0.0,"score_sum":0.0,"baseline_support":0,"stability_sum":0.0,"trajectory_speed_sum":0.0,"trajectory_acceleration_sum":0.0,"trajectory_dwell_sum":0.0,"trajectory_turn_sum":0.0,"trajectory_path_sum":0.0,"trajectory_stability_sum":0.0,"examples":[],"state_hash":before[4] or before[5],"gray32x18":before[11],"edge_density":before[12],"color_histogram":before[13],"local_descriptor":local_descriptor,"aspect":before[9]/max(1,before[10])})
                    item["samples"]+=1;item["human" if action["source"] in ("user","用户") else "ai"]+=1;item["sum"]+=advantage;item["sum2"]+=advantage*advantage;item["score_sum"]+=score_delta;item["baseline_support"]+=len(nearest);item["stability_sum"]+=stable;trajectory=action.get("trajectory") or {};item["trajectory_speed_sum"]+=float(trajectory.get("speed_mean",0.0));item["trajectory_acceleration_sum"]+=float(trajectory.get("acceleration_mean",0.0));item["trajectory_dwell_sum"]+=float(trajectory.get("dwell_ms",0.0));item["trajectory_turn_sum"]+=float(trajectory.get("turns",0.0));item["trajectory_path_sum"]+=float(trajectory.get("path_length",0.0));item["trajectory_stability_sum"]+=float(trajectory.get("cursor_stability",0.0));item["examples"].append(dict(example,wheel_delta=signed,wheel_axis=wheel_axis));outcomes.append(example)
                elif role=="validation" and stable>=stable_required:
                    validation_outcomes.append((key,example));validation_blocks.append(session_id)
        for start in range(0,len(persisted),500):
            if self._cancelled(token):
                return self._training_result("cancelled", "睡眠任务被取消", training_samples=len(outcomes), validation_samples=len(validation_outcomes))
            self.store.save_action_outcomes(persisted[start:start+500])
        if not outcomes:
            reason = "样本不足：没有满足稳定性与时间隔离条件的训练动作"
            self._record_training_failure(fingerprint, reason)
            self.store.add_system_event(None, "model_skipped", {"reason": reason, "time": time.time(), "semantic_actions": all_actions_count})
            return self._training_result("skipped", reason, self.store.best_model(), 0, 0)
        actions=[];policy={}
        for key,item in states.items():
            n=item["samples"];raw_mean=item["sum"]/max(1,n);raw_var=max(0.0,item["sum2"]/max(1,n)-raw_mean*raw_mean);prior_strength=8.0;mean=raw_mean*n/(n+prior_strength);var=max(0.0025,raw_var+(prior_strength/(n+prior_strength))*0.0025);standard_error=math.sqrt(var/max(1.0,n));t_critical=2.776 if n<=5 else 2.447 if n<=7 else 2.262 if n<=10 else 2.131 if n<=20 else 2.045 if n<=30 else 1.96;lcb=mean-t_critical*standard_error;ci_width=2.0*t_critical*standard_error;confidence_probability=0.5*(1.0+math.erf(mean/max(1e-9,standard_error*math.sqrt(2.0))));baseline_avg=item["baseline_support"]/max(1,n);stable_avg=item["stability_sum"]/max(1,n)
            if item["human"]<=0 and key[3]!="移动":lcb=min(lcb,-0.01)
            if baseline_avg<2 or stable_avg<0.65:lcb=min(lcb,-0.01)
            wheel_examples=[e for e in item["examples"] if "wheel_delta" in e]
            payload={"state_key":key[0],"state_hash":item["state_hash"],"gray32x18":item["gray32x18"],"edge_density":item["edge_density"],"color_histogram":item["color_histogram"],"local_descriptor":item["local_descriptor"],"local_uncertainty_radius":min(0.25,0.03+standard_error*2.0),"aspect":item["aspect"],"x":(key[1]+0.5)/16.0,"y":(key[2]+0.5)/9.0,"action_type":key[3],"wheel_axis":key[4],"wheel_direction":key[5],"wheel_magnitude_bucket":key[6],"wheel_delta":int(round(sum(e.get("wheel_delta",0) for e in wheel_examples)/max(1,len(wheel_examples)))) if key[3] in ("滚轮","水平滚轮") else 0,"samples":n,"effective_samples":n*n/(n+prior_strength),"human_samples":item["human"],"ai_samples":item["ai"],"average_action_advantage":mean,"advantage_variance":var,"confidence_interval_width":ci_width,"average_score_delta":item["score_sum"]/max(1,n),"baseline_support":baseline_avg,"stability":stable_avg,"trajectory_profile":{"speed_mean":item["trajectory_speed_sum"]/max(1,n),"acceleration_mean":item["trajectory_acceleration_sum"]/max(1,n),"dwell_ms":item["trajectory_dwell_sum"]/max(1,n),"turns":item["trajectory_turn_sum"]/max(1,n),"path_length":item["trajectory_path_sum"]/max(1,n),"cursor_stability":item["trajectory_stability_sum"]/max(1,n)},"confidence_lower_bound":lcb,"confidence_probability":confidence_probability,"uncertainty":standard_error,"state_similarity_threshold":0.32,"causal_claim":"not_claimed_without_propensity","causal_policy_samples":sum(1 for e in item["examples"] if e.get("causal_policy_eligible")),"observation_samples":sum(1 for e in item["examples"] if e.get("observation_only"))}
            actions.append(payload);policy[key]=payload
        values=[];hits=0;failures=0;absolute_errors=[];sign_hits=0;lcb_coverages=0;metrics_by_key={};action_validation={}
        for key,example in validation_outcomes:
            chosen=policy.get(key)
            if chosen is None:
                failures+=1
                continue
            actual=float(example["action_advantage"])
            prediction=float(chosen.get("average_action_advantage", 0.0))
            predicted_lcb=float(chosen.get("confidence_lower_bound", -1.0))
            action_type=str(chosen.get("action_type", "移动"))
            entry=metrics_by_key.setdefault(key, {"actual":[],"errors":[],"sign_hits":0,"lcb_coverage":0,"false_positives":0,"sessions":set(),"states":set(),"action_type":action_type})
            entry["actual"].append(actual)
            entry["errors"].append(abs(prediction-actual))
            entry["sign_hits"] += int((prediction > 0) == (actual > 0))
            entry["lcb_coverage"] += int(actual >= predicted_lcb)
            entry["false_positives"] += int(action_type != "移动" and predicted_lcb > 0 and actual <= 0)
            entry["sessions"].add(str(example.get("session_id", "")))
            entry["states"].add(str(chosen.get("state_key", "")))
            hits+=1
            values.append(actual)
            absolute_errors.append(abs(prediction-actual))
            sign_hits += int((prediction > 0) == (actual > 0))
            lcb_coverages += int(actual >= predicted_lcb)
            if actual<=0 or example["stability"]<0.65:
                failures+=1
        for key,chosen in policy.items():
            entry=metrics_by_key.get(key, {})
            actuals=list(entry.get("actual", []))
            n=len(actuals)
            mean_actual=sum(actuals)/max(1,n)
            variance=max(0.0025,sum((value-mean_actual)**2 for value in actuals)/max(1,n))
            critical=2.776 if n<=5 else 2.262 if n<=10 else 2.045 if n<=30 else 1.96
            validation_lcb=mean_actual-critical*math.sqrt(variance/max(1,n))
            metrics={"validation_samples":n,"validation_mean_actual":mean_actual,"validation_mean_absolute_error":sum(entry.get("errors", []))/max(1,n),"validation_sign_hit_rate":entry.get("sign_hits",0)/max(1,n),"validation_lcb_coverage_rate":entry.get("lcb_coverage",0)/max(1,n),"validation_false_positive_rate":entry.get("false_positives",0)/max(1,n),"validation_lower_bound":validation_lcb,"validation_sessions":sorted(entry.get("sessions",set())),"validation_state_coverage":len(entry.get("states",set()))}
            chosen.update(metrics)
            action_validation[key]=metrics
        validation_n=len(values)
        val_mean=sum(values)/max(1,validation_n)
        val_var=max(0.0025,sum((v-val_mean)**2 for v in values)/max(1,validation_n))
        val_t=2.776 if validation_n<=5 else 2.262 if validation_n<=10 else 2.045 if validation_n<=30 else 1.96
        val_ci=val_t*math.sqrt(val_var/max(1,validation_n))
        quality=val_mean-val_ci
        nonmoving=[item for item in actions if item["action_type"]!="移动"]
        move_actions=[item for item in actions if item["action_type"]=="移动"]
        validated_move=[item for item in move_actions if item.get("samples",0)>=12 and item.get("effective_samples",0)>=8 and item.get("confidence_lower_bound",-1.0)>0 and item.get("baseline_support",0.0)>=4 and item.get("stability",0.0)>=0.65 and item.get("confidence_interval_width",999.0)<0.24]
        validated_nonmoving=[] if single_session_split else [item for item in nonmoving if item.get("validation_samples",0)>=8 and item.get("validation_lower_bound",-1.0)>0 and item.get("validation_state_coverage",0)>=1 and item.get("validation_false_positive_rate",1.0)<=0.10]
        click_actions=[item for item in validated_nonmoving if item["action_type"] in ("左键","右键")]
        wheel_actions=[item for item in validated_nonmoving if item["action_type"] in ("滚轮","水平滚轮")]
        move_validation_values=[float(example["action_advantage"]) for key,example in validation_outcomes if key in policy and policy[key].get("action_type")=="移动"]
        move_n=len(move_validation_values)
        move_mean=sum(move_validation_values)/max(1,move_n)
        move_var=max(0.0025,sum((value-move_mean)**2 for value in move_validation_values)/max(1,move_n))
        move_t=2.776 if move_n<=5 else 2.262 if move_n<=10 else 2.045 if move_n<=30 else 1.96
        move_ci=move_t*math.sqrt(move_var/max(1,move_n))
        move_quality=move_mean-move_ci if move_n else sum(item.get("average_action_advantage",0.0) for item in validated_move)/max(1,len(validated_move))
        candidate_quality=move_quality if validated_move else quality
        propensity_values=[example.get("behavior_probability") for _,example in validation_outcomes if example.get("behavior_probability") is not None]
        offline_policy={"recorded_propensities":len(propensity_values),"total_validation_actions":len(validation_outcomes),"causal_claim":"not_claimed_without_propensity" if not propensity_values else "propensity_recorded_observational_only"}
        safe_actions=validated_move+validated_nonmoving
        policy_layers={"state_encoder":{"verified":True,"source":"gray32x18+dhash64+color_histogram","uncertainty":"student_t_lcb_and_local_radius"},"action_value_model":{"verified":bool(validated_move),"target":"action_advantage","cross_session":not single_session_split},"move_policy":{"verified":bool(validated_move),"actions":len(validated_move),"quality":move_quality,"validation_samples":move_n},"click_policy":{"verified":bool(click_actions),"actions":len(click_actions)},"sleep_policy":{"verified":True,"source":"sleep_decision_samples","actions":0},"delete_policy":{"verified":True,"source":"retain_value_pruning","actions":0},"wheel_policy":{"verified":bool(wheel_actions),"actions":len(wheel_actions)}}
        model_grade="single_session_move_only" if single_session_split else "cross_session_validated"
        payload={
            "id":uuid.uuid4().hex,
            "trained_at":time.time(),
            "quality":candidate_quality,
            "train_quality":sum(a["average_action_advantage"] for a in actions)/max(1,len(actions)),
            "frame_count":sum(len(v) for v in frames_by_session.values()),
            "mouse_count":sum(len(v) for v in mouse_by_session.values()),
            "training_samples":len(outcomes),
            "semantic_actions":all_actions_count,
            "validation_samples":len(validation_outcomes),
            "validation_hits":hits,
            "validation_mean_action_advantage":val_mean,
            "validation_confidence_interval":val_ci,
            "validation_failure_rate":failures/max(1,len(validation_outcomes)),
            "validation_mean_absolute_error":sum(absolute_errors)/max(1,len(absolute_errors)),
            "validation_sign_hit_rate":sign_hits/max(1,hits),
            "validation_lcb_coverage_rate":lcb_coverages/max(1,hits),
            "validation_state_coverage":len({key[0] for key,_ in validation_outcomes}),
            "validation_sessions":sorted(set(validation_blocks)),
            "coverage_states":len({a["state_key"] for a in actions}),
            "failure_rate":len([a for a in actions if a["confidence_lower_bound"]<=0])/max(1,len(actions)),
            "nonmoving_candidates":len(nonmoving),
            "nonmoving_validated":len(validated_nonmoving),
            "move_candidates":len(move_actions),
            "move_validated":len(validated_move),
            "policy_layers":policy_layers,
            "offline_policy_evaluation":offline_policy,
            "model_version":12,
            "model_grade":model_grade,
            "model_layer":"candidate",
            "allowed_action_types":["移动",
            "左键",
            "右键",
            "滚轮",
            "水平滚轮"],
            "control_enabled":True,
            "champion":True,
            "last_used":time.time(),
            "action_quality":candidate_quality,
            "validation_quality":candidate_quality,
            "global_validation_quality":quality,
            "move_validation_quality":move_quality,
            "policy":{"min_samples":12,
            "min_effective_samples":8,
            "min_validation_samples":8,
            "uncertainty_threshold":0.12,
            "max_confidence_interval_width":0.24,
            "min_confidence_lower_bound":0.0,
            "min_baseline_support":4,
            "similarity_threshold":0.78,
            "max_nonmoving_false_positive_rate":0.10,
            "low_confidence_action":"move_only",
            "blacklist_regions":[],
            "target":"action_advantage"},
            "q_actions":sorted(safe_actions,key=lambda a:(a.get("validation_lower_bound",-1.0),a["confidence_lower_bound"],a["samples"]),reverse=True)[:256],
            "outcome_examples":outcomes[-256:]
        }
        visual_result, visual_reason = self._train_visual_policy_weights(outcomes, validation_outcomes, token)
        if visual_result is None:
            payload["visual_policy_error"] = visual_reason
            payload["model_layer"] = "observation"
            payload["allowed_action_types"] = []
            payload["control_enabled"] = False
            saved_path = self._save_layered_model_payload(payload, outcomes, validation_outcomes, "observation")
            self.store.add_system_event(None, "visual_policy_training_saved_observation", {"reason": visual_reason, "path": str(saved_path or ""), "training_samples": len(outcomes), "validation_samples": len(validation_outcomes), "fingerprint": fingerprint, "time": time.time()})
            return self._training_result("saved_observation", visual_reason, payload, len(outcomes), len(validation_outcomes), 0.0, False)
        payload["visual_policy_weights"] = visual_result["weights"]
        payload["visual_policy_metrics"] = visual_result["metrics"]
        payload["policy_layers"]["vision_policy"] = {"verified": True, "source": "gray32x18+local_visual_features+action_result+score_delta", "validation": visual_result["metrics"]}
        champion=self.store.best_model();champion_layers=champion.get("policy_layers",{}) if isinstance(champion,dict) else {};champion_quality=float(champion_layers.get("move_policy",{}).get("quality",champion.get("validation_quality",-999999.0))) if isinstance(champion,dict) else -999999.0
        enough=len(outcomes)>=24 and bool(validated_move) and len({a["state_key"] for a in move_actions})>=1 and len(set(validation_blocks))>=1 and (move_n>=4 or hits>=8)
        if not enough or candidate_quality <= champion_quality:
            reason = "移动策略样本不足或验证未通过" if not enough else "移动策略未超过当前冠军"
            payload["champion"] = False
            payload["model_layer"] = "move" if validated_move else "observation"
            payload["allowed_action_types"] = ["移动"] if validated_move else []
            payload["control_enabled"] = bool(validated_move)
            saved_path = self._save_layered_model_payload(payload, outcomes, validation_outcomes, payload["model_layer"])
            self.store.add_system_event(None, "model_candidate_layer_saved", {"reason": reason, "layer": payload["model_layer"], "path": str(saved_path or ""), "validation_quality": candidate_quality, "champion_quality": champion_quality, "validation_samples": len(validation_outcomes), "validation_hits": hits, "training_samples": len(outcomes), "move_validated": len(validated_move), "nonmoving_validated": len(validated_nonmoving), "fingerprint": fingerprint, "time": time.time()})
            return self._training_result("saved_" + payload["model_layer"], reason, champion if isinstance(champion, dict) and champion.get("champion") else payload, len(outcomes), len(validation_outcomes), candidate_quality - champion_quality, False)
        payload["model_layer"] = "click_wheel" if validated_nonmoving else "move"
        payload["allowed_action_types"] = ["移动", "左键", "右键", "滚轮", "水平滚轮"] if validated_nonmoving else ["移动"]
        payload["control_enabled"] = True
        name="model_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")+"_"+payload["id"][:8]+".json"
        final_path=self.store._assert_storage_path(self.store.models/name, "model_json")
        onnx_path=self.store._assert_storage_path(final_path.with_suffix(".onnx"), "model_onnx")
        temp_onnx=self.store._assert_storage_path(onnx_path.with_name(onnx_path.name+".tmp"), "model_onnx_tmp")
        export_ok = self._export_policy_onnx(payload, temp_onnx)
        verify_ok = self._verify_policy_onnx(temp_onnx, payload) if export_ok else False
        if not (export_ok and verify_ok and payload.get("onnx_exported") is True and payload.get("onnx_policy_verified") is True):
            self.store._safe_unlink_storage(temp_onnx, "onnx_export_failed_cleanup")
            payload["champion"] = False
            reason = "ONNX 导出或验证失败，拒绝保存冠军模型：{}".format(payload.get("onnx_export_error") or payload.get("onnx_self_test_error") or "未通过")
            self._record_training_failure(fingerprint, reason)
            try:
                self.store.add_system_event(None, "model_candidate_rejected", {"reason": reason, "onnx_exported": bool(payload.get("onnx_exported")), "onnx_policy_verified": bool(payload.get("onnx_policy_verified")), "fingerprint": fingerprint, "time": time.time()})
            except Exception:
                pass
            return self._training_result("failed", reason, payload, len(outcomes), len(validation_outcomes), candidate_quality - champion_quality, False)
        temp_onnx.replace(onnx_path)
        self.store._fsync_directory(onnx_path.parent)
        payload["onnx_path"] = onnx_path.name
        temp_path=self.store._assert_storage_path(final_path.with_suffix(".tmp"), "model_json_tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload,ensure_ascii=False,indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        if self._cancelled(token):
            self.store._safe_unlink_storage(temp_path, "model_json_cancelled_cleanup")
            self.store._safe_unlink_storage(onnx_path, "model_onnx_cancelled_cleanup")
            return self._training_result("cancelled", "模型落盘前被取消", training_samples=len(outcomes), validation_samples=len(validation_outcomes))
        temp_path.replace(final_path)
        self.store._fsync_directory(final_path.parent)
        self.store.register_model_metadata(payload["id"], final_path, payload, outcomes, validation_outcomes)
        self.last_model_training = time.time()
        self.last_training_success = self.last_model_training
        self.last_successful_training_fingerprint = fingerprint
        self.training_retry_count = 0
        self.next_training_retry_at = 0.0
        self.last_training_failure_reason = ""
        return self._training_result("trained", "新冠军视觉策略已落盘并通过 ONNX 自测", payload, len(outcomes), len(validation_outcomes), candidate_quality - champion_quality, True)

    def _run_sleep_task2_until_done(self, token, experience_limit, before_pool, pool_was_over_limit):
        retry = 0
        while not self._cancelled(token):
            self.store.recover_deletions()
            if not self._wait_resource(token, "maintenance"):
                continue
            budget = self.resources.acquire("maintenance")
            if not budget.allowed:
                time.sleep(max(1.0, budget.next_interval))
                continue
            batch_size = max(1, min(8, int(budget.max_batch)))
            self.flush_mouse_records()
            model_removed = self.store.prune_models(max(1, int(self.settings.data["model_limit"])), lambda: self._cancelled(token), lambda: self._wait_resource(token, "maintenance"), batch_size=batch_size)
            if not self._wait_resource(token, "maintenance"):
                continue
            budget = self.resources.acquire("maintenance")
            if not budget.allowed:
                time.sleep(max(1.0, budget.next_interval))
                continue
            batch_size = max(1, min(8, int(budget.max_batch)))
            experience_removed, remaining = self.store.prune_experience(max(1, experience_limit), lambda: self._cancelled(token), lambda value: self.emit("progress", value), lambda: self._wait_resource(token, "maintenance"), batch_size=batch_size)
            if self._cancelled(token):
                return None
            model_status = getattr(self.store, "last_model_prune_result", {"success": True, "remaining": len(self.store.model_files()), "target": len(self.store.model_files())})
            pool_status = getattr(self.store, "last_experience_prune_result", {"success": remaining <= experience_limit, "remaining": remaining, "target": int(experience_limit * 0.5)})
            actual_breakdown = self.store.pool_breakdown(True)
            remaining = int(actual_breakdown.get("experience_total_bytes", 0))
            target = int(experience_limit * 0.5)
            model_ok = bool(model_status.get("success"))
            pool_ok = bool(pool_status.get("success")) and (not pool_was_over_limit or remaining <= target)
            if model_ok and pool_ok:
                return model_removed, experience_removed, remaining
            retry += 1
            detail = "任务2未完成：模型 {} / 目标 {}；经验池 {} / 目标 {}。保持睡眠模式低频重试清理；按 ESC 才能空闲。".format(model_status.get("remaining"), model_status.get("target"), remaining, target)
            try:
                self.store.add_system_event(None, "sleep_task2_retry", {"reason": detail, "retry": retry, "model_status": model_status, "pool_status": pool_status, "time": time.time()})
            except Exception:
                pass
            sample = self.resources.sample()
            self.emit("state", {"state": "sleep", "detail": detail, "cpu": sample["cpu"], "memory": sample["memory"]})
            self.emit("progress", 56.0)
            time.sleep(min(30.0, 3.0 + retry))
        return None

    def _wait_auto_sleep_training_resume(self, token, detail):
        self._update_sleep_terminal_state(task2_done=True, resume_state="task2_done_resume_pending", resume_block_reason="等待目标窗口校验")
        while not self._cancelled(token):
            hwnd, rect, reason = self._find_valid_target(False)
            resume_reason = reason or ""
            ok = False
            if hwnd is not None and rect is not None and activate_root_window(hwnd):
                time.sleep(0.08)
                if foreground_root_matches(hwnd) and self._place_cursor_before_entry(hwnd, rect) and not self._cancelled(token) and valid_client(hwnd, True) is not None and entry_capture_pixel_validation(hwnd):
                    ok = True
                else:
                    resume_reason = "前台、鼠标入区或客户区像素校验失败：" + getattr(entry_capture_pixel_validation, "last_reason", "校验失败")
            else:
                resume_reason = resume_reason or getattr(activate_root_window, "last_reason", "客户区状态异常")
            if ok:
                self.emit("progress", 0.0)
                with self.lock:
                    self.sleep_origin = "auto"
                with self.lock:
                    self.sleep_task2_done = True
                self._update_sleep_terminal_state(task2_done=True, resume_state="auto_sleep_task2_done_resume_pending", resume_block_reason="")
                if self.start_session("training", automatic=True):
                    self._update_sleep_terminal_state(task2_done=True, resume_state="auto_sleep_task2_done_resume_training", resume_block_reason="")
                    return True
                resume_reason = "自动睡眠完成，但恢复训练模式启动失败"
                self._update_sleep_terminal_state(task2_done=True, resume_state="task2_done_resume_failed", resume_block_reason=resume_reason)
            else:
                self._update_sleep_terminal_state(task2_done=True, resume_state="task2_done_resume_pending", resume_block_reason=str(resume_reason))
            sample = self.resources.sample()
            self.emit("state", {"state": "sleep", "detail": detail + "；任务2已完成，恢复训练被目标窗口校验阻塞：" + str(resume_reason), "cpu": sample["cpu"], "memory": sample["memory"]})
            time.sleep(2.0)
        return False

    def _sleep_worker(self, token, resume_training):
        started = time.monotonic()
        before_model = self.store.best_model() or {}
        before_quality = float(before_model.get("validation_quality", before_model.get("quality", 0.0)) or 0.0) if isinstance(before_model, dict) else 0.0
        before_pool = int(self.store.pool_breakdown(False).get("experience_total_bytes", 0))
        experience_limit = int(self.settings.data["experience_limit"])
        pool_was_over_limit = before_pool > experience_limit
        decision_id = self.pending_sleep_decision
        training_result = self._training_result("cancelled", "睡眠任务尚未开始")
        try:
            self._update_sleep_terminal_state(task1_terminal_state="started", task2_started=False, task2_done=False, files_deleted=0, needs_recovery=False)
            self.emit("progress", 4.0)
            self.emit("state", {"state": "sleep", "detail": "低优先级精确评分队列与任务1：训练 AI 模型", "cpu": self.resources.sample()["cpu"], "memory": self.resources.sample()["memory"]})
            while not self._cancelled(token):
                pending = self.store.deferred_score_status()
                oldest_wait = max(0.0, time.time() - pending["oldest"]) if pending.get("oldest") else 0.0
                self.resources.update_exact_score_metrics(pending["pending"], oldest_wait)
                if pending["failed"] > 0:
                    sessions = self.store.exclude_failed_deferred_sessions()
                    training_result = self._sleep_task1_terminal_result(self._training_result("no_trainable_data", "{} 个会话存在三次失败的精确评分，已排除不可训练会话".format(len(sessions))))
                    try:
                        self.store.add_system_event(None, "sleep_task1_exact_failed_sessions_excluded", {"sessions": sessions, "terminal_status": training_result.get("terminal_status"), "time": time.time()})
                    except Exception:
                        pass
                    break
                if pending["pending"] <= 0:
                    break
                budget = self.resources.acquire("maintenance")
                if not budget.allowed:
                    if budget.must_pause:
                        training_result = self._training_result("cancelled", "维护预算暂停精确评分")
                        self.store.finalize_sleep_decision(decision_id, training_result, 0, time.monotonic() - started, 0.0)
                        sample = self.resources.sample()
                        self.emit("state", {"state": "sleep", "detail": "维护暂停精确评分，保持睡眠；按 ESC 才能空闲：" + (budget.pause_reason or "资源红线"), "cpu": sample["cpu"], "memory": sample["memory"]})
                        time.sleep(max(1.0, budget.next_interval))
                        continue
                    time.sleep(max(0.02, budget.next_interval))
                    continue
                resolved = self.store.process_deferred_exact_scores(lambda: self._cancelled(token), lambda: self.resources.capture_snapshot().allowed, maximum=max(1, int(getattr(budget, "database_batch_size", 1))))
                status = self.store.deferred_score_status()
                oldest_wait = max(0.0, time.time() - status["oldest"]) if status.get("oldest") else 0.0
                self.resources.update_exact_score_metrics(status["pending"], oldest_wait)
                self.emit("state", {"state": "sleep", "detail": "精确评分待处理 {} 帧，最老等待 {:.1f} 秒".format(status["pending"], oldest_wait), "cpu": self.resources.sample()["cpu"], "memory": self.resources.sample()["memory"]})
                if resolved <= 0 and status["pending"] > 0:
                    time.sleep(max(0.02, budget.next_interval * 0.25))
            if self._cancelled(token):
                return
            if str(training_result.get("terminal_status", "")) not in ("reused_best_model", "no_trainable_data"):
                try:
                    with self.resources.budget_slot("training"):
                        training_result = self._train_model(token)
                except ResourceBudgetBusy:
                    training_result = self._training_result("skipped", "训练 in-flight 预算已满")
                training_result = self._sleep_task1_terminal_result(training_result)
            if self._cancelled(token):
                return
            with self.lock:
                self.sleep_task1_done = True
            model_payload = training_result.get("model") or {}
            onnx_value = model_payload.get("onnx_path") if isinstance(model_payload, dict) else None
            new_model_path = None
            if onnx_value and self.store.models is not None:
                new_model_path = Path(str(onnx_value))
                if not new_model_path.is_absolute():
                    new_model_path = self.store.models / new_model_path
            if new_model_path is not None:
                gpu_ready = self.resources.backend.try_enable_gpu_model(new_model_path, self.resources.sample())
                try:
                    self.store.add_system_event(None, "sleep_task1_gpu_prewarm", {"model_path": str(new_model_path), "enabled": bool(gpu_ready), "backend": self.resources.backend.snapshot(), "time": time.time()})
                except Exception:
                    pass
            terminal_status = str(training_result.get("terminal_status", training_result.get("status", "")))
            self._update_sleep_terminal_state(task1_terminal_state=terminal_status, task2_started=False, task2_done=False, needs_recovery=False, training_reason=str(training_result.get("reason", "")))
            self._reset_action_benefit_fuse_after_training(training_result)
            if terminal_status == "trained":
                task1_detail = "任务1完成：已产出并落盘新冠军视觉模型"
            elif terminal_status == "reused_best_model":
                task1_detail = "任务1完成：本轮无新模型，复用当前已验证冠军模型"
            else:
                task1_detail = "任务1完成：没有可训练数据，本轮无新模型"
            try:
                self.store.add_system_event(None, "sleep_task1_terminal", {"terminal_status": terminal_status, "reason": training_result.get("reason", ""), "training_samples": training_result.get("training_samples", 0), "validation_samples": training_result.get("validation_samples", 0), "time": time.time()})
            except Exception:
                pass
            self._update_sleep_terminal_state(task1_terminal_state=terminal_status, task2_started=True, task2_done=False, needs_recovery=False)
            self.emit("progress", 56.0)
            self.emit("state", {"state": "sleep", "detail": task1_detail + "；任务2：检查 AI 模型与经验池", "cpu": self.resources.sample()["cpu"], "memory": self.resources.sample()["memory"]})
            task2_result = self._run_sleep_task2_until_done(token, experience_limit, before_pool, pool_was_over_limit)
            if task2_result is None:
                return
            model_removed, experience_removed, remaining = task2_result
            after_model = self.store.best_model() or {}
            after_quality = float(after_model.get("validation_quality", after_model.get("quality", 0.0)) or 0.0) if isinstance(after_model, dict) else 0.0
            restored_gain = after_quality - before_quality if resume_training and training_result.get("terminal_status", training_result.get("status")) == "trained" else 0.0
            self.store.finalize_sleep_decision(decision_id, training_result, max(0, before_pool - remaining), time.monotonic() - started, restored_gain)
            self.pending_sleep_decision = None
            detail = "{}；任务2完成：删除 AI 模型 {} 个，删除经验 {} 条，经验池 {:.2f} MB".format(task1_detail, model_removed, experience_removed, remaining / 1024 / 1024)
            if resume_training:
                with self.lock:
                    self.sleep_task2_done = True
                self._update_sleep_terminal_state(task1_terminal_state=terminal_status, task2_started=True, task2_done=True, resume_state="task2_done_resume_pending", files_deleted=int(model_removed or 0) + int(experience_removed or 0), remaining_bytes=int(remaining or 0), needs_recovery=False)
                self.emit("progress", 100.0)
                self._wait_auto_sleep_training_resume(token, detail)
            else:
                with self.lock:
                    self.sleep_task2_done = True
                self._update_sleep_terminal_state(task1_terminal_state=terminal_status, task2_started=True, task2_done=True, resume_state="manual_task2_done_idle", files_deleted=int(model_removed or 0) + int(experience_removed or 0), remaining_bytes=int(remaining or 0), needs_recovery=False)
                self.emit("progress", 100.0)
                self._finish_idle(token, detail, True)
        except Exception as error:
            training_result = self._training_result("failed", str(error))
            self._update_sleep_terminal_state(needs_recovery=True, task2_done=False, error=str(error))
            try:
                self.store.finalize_sleep_decision(decision_id, training_result, 0, time.monotonic() - started, 0.0)
            except Exception:
                note_strict_exception("sleep_finalize_failed", error, {"decision_id": decision_id or ""})
            sample = self.resources.sample()
            self.emit("state", {"state": "sleep", "detail": "睡眠模式发生错误；任务2未确认完成，按 ESC 可空闲：" + str(error), "cpu": sample["cpu"], "memory": sample["memory"]})

    def _finish_idle(self, token, detail, release_keyboard=True):
        if threading.current_thread() is not self.control_thread:
            self.on_control_signal("sleep_finish", detail, token=token)
            return
        with self.lock:
            if token != self.epoch:
                return
            origin = self.sleep_origin
            task2_done = bool(self.sleep_task2_done)
            event = "manual_sleep_task2_done" if origin == "manual" and task2_done else "esc"
            self._transition_state_locked(event, "sleep", "idle" if event == "manual_sleep_task2_done" else "idle", detail, token)
            self.cancel_event.set()
            self.epoch += 1
        if release_keyboard:
            self.keyboard_hook.stop()
        self.emit("progress", 0.0)
        self.post_state(detail)

    def information_snapshot(self):
        sample = self.resources.capture_snapshot()
        metrics = self.resources.sample()
        with self.lock:
            state = self.state
            frames = self.frame_count
            mouse = self.mouse_count
            session = self.session_id
            queue_snapshot = {"capture_queue": self.capture_queue.qsize(), "feature_queue": self.feature_queue.qsize(), "persist_queue": self.persist_queue.qsize(), "mouse_queue": self.mouse_queue.qsize(), "mouse_segment_queue": self.mouse_segment_queue.qsize(), "raw_mouse_queue": self.raw_mouse_queue.qsize(), "raw_critical_queue": self.raw_critical_queue.qsize(), "raw_hook_ring": self.raw_hook_ring.qsize() if hasattr(self.raw_hook_ring, "qsize") else 0, "mouse_sqlite_degraded": time.monotonic() < float(self.mouse_sqlite_degraded_until or 0.0), "frame_replay_paused": time.monotonic() < float(self.frame_replay_pause_until or 0.0)}
        backend_status = self.resources.backend.snapshot()
        return {
            "state": state,
            "mode": state,
            "mode_reason": "状态机当前值",
            "sleep_exit_condition": "快照；慢字段后台刷新",
            "recording": state in ("learning",
            "training"),
            "mouse_source_enabled": state == "training",
            "client_valid": False,
            "client_validity": "快照未校验，后台刷新",
            "frames": frames,
            "mouse": mouse,
            "session": session or "无",
            "recovery_pending": bool(self.recovery_pending),
            "recovery_reason": self.recovery_reason,
            "cpu": metrics.get("cpu", 0.0),
            "memory": metrics.get("memory", 0.0),
            "pool_size": 0,
            "pool_size_fast": 0,
            "pool_breakdown": {},
            "model_count": 0,
            "current_model_onnx_verified": False,
            "current_model_id": "后台刷新中",
            "current_model_layer": "后台刷新中",
            "current_model_allowed_actions": [],
            "capacity": {"blocked": False},
            "journal_backlog": {"total": 0,
            "frame": 0,
            "mouse": 0,
            "mouse_segment": 0},
            "queue_snapshot": queue_snapshot,
            "resource": dict(metrics),
            "gpu_name": self.resources.backend.name(),
            "backend": self.resources.backend.name(),
            "gpu_backend_status": backend_status,
            "gpu_runtime_provider": backend_status.get("runtime_provider",
            "不可用"),
            "gpu_real_ready": bool(backend_status.get("runtime_ready")),
            "gpu_failure_reason": backend_status.get("gpu_failure_reason",
            ""),
            "recent_errors": [],
            "training_readiness": {"ready": False,
            "sessions": 0,
            "actions": 0,
            "frames": 0,
            "compressed_trajectory_points": 0},
            "training_gap": "后台刷新中",
            "gpu": metrics.get("gpu"),
            "gpu_total": metrics.get("gpu_dedicated_total"),
            "gpu_used": metrics.get("gpu_dedicated_used"),
            "gpu_free": metrics.get("gpu_dedicated_free"),
            "gpu_batch_size": 0,
            "cpu_workers": sample.cpu_workers,
            "capture_fps": 1.0 / max(0.001, sample.next_interval),
            "capture_resolution": "{}×{}".format(*sample.max_capture_resolution),
            "queue_age": metrics.get("queue_age", 0.0),
            "pipeline_queue_age": metrics.get("pipeline_queue_age", 0.0),
            "ui_heartbeat_jitter_ms": metrics.get("ui_heartbeat_jitter_ms", 0.0),
            "resource_state": sample.state,
            "pause_reason": sample.pause_reason or "无",
            "metric_sources": metrics.get("metric_sources", {}),
            "ldplayer_cpu": metrics.get("ldplayer_cpu", 0.0),
            "program_cpu": metrics.get("process_cpu", 0.0),
            "program_gpu": metrics.get("program_gpu"),
            "ldplayer_gpu": metrics.get("ldplayer_gpu"),
            "gpu_sampling_source": metrics.get("gpu_sampling_source",
            "不可用"),
            "disk_write_latency": metrics.get("disk_write_latency"),
            "sqlite_latency": metrics.get("sqlite_latency", 0.0),
            "reward_definition_version": REWARD_DEFINITION_VERSION,
            "platform_backend": self.platform_backend.resource_snapshot()
        }

    def information(self, reconcile=False):
        sample = self.resources.sample()
        with self.lock:
            state = self.state
            frames = self.frame_count
            mouse = self.mouse_count
            session = self.session_id
            queue_snapshot = {"capture_queue": self.capture_queue.qsize(), "feature_queue": self.feature_queue.qsize(), "persist_queue": self.persist_queue.qsize(), "mouse_queue": self.mouse_queue.qsize(), "mouse_segment_queue": self.mouse_segment_queue.qsize(), "raw_mouse_queue": self.raw_mouse_queue.qsize(), "raw_critical_queue": self.raw_critical_queue.qsize(), "raw_hook_ring": self.raw_hook_ring.qsize() if hasattr(self.raw_hook_ring, "qsize") else 0, "mouse_sqlite_degraded": time.monotonic() < float(self.mouse_sqlite_degraded_until or 0.0), "frame_replay_paused": time.monotonic() < float(self.frame_replay_pause_until or 0.0)}
        try:
            breakdown = self.store.pool_breakdown(bool(reconcile)) if self.store.pool else {}
            pool_size = int(breakdown.get("experience_total_bytes", 0))
            model_count = len(self.store.model_files()) if self.store.models else 0
            capacity = self.store.capacity_status() if self.store.conn else {"blocked": False}
            journal_backlog = self.store.accepted_journal_backlog() if self.store.conn else {"total": 0, "frame": 0, "mouse": 0, "mouse_segment": 0}
            recent_errors = self.store.recent_critical_errors(5) if self.store.conn else []
            with self.lock:
                pending_errors = list(self.pending_critical_errors)[:5]
            recent_errors = ([{"created": float(item.get("created", time.time())), "kind": "critical_exception_pending", "payload": dict(item)} for item in pending_errors] + recent_errors)[:5]
        except Exception:
            breakdown = {}
            pool_size = 0
            model_count = 0
            capacity = {"blocked": False}
            journal_backlog = {"total": 0, "frame": 0, "mouse": 0, "mouse_segment": 0}
            recent_errors = []
        capture_budget = self.resources.capture_snapshot()
        backend_status = self.resources.backend.snapshot()
        best_model = self.store.best_model() if self.store.conn else {}
        client_rect, client_reason = self._validate_bound_target(require_cursor=False, require_foreground=False)
        sleep_exit_condition = "任务1总是输出 trained / reused_best_model / no_trainable_data；随后进入任务2；手动睡眠任务2完成后空闲，自动睡眠任务2完成后回训练"
        recording = state in ("learning", "training")
        mouse_source_enabled = state == "training"
        onnx_verified = bool(isinstance(best_model, dict) and best_model.get("onnx_policy_verified") is True)
        model_layer = str(best_model.get("model_layer", "无") if isinstance(best_model, dict) else "无")
        allowed_actions = best_model.get("allowed_action_types", []) if isinstance(best_model, dict) else []
        readiness = self.store.training_readiness() if self.store.conn else {"ready": False, "sessions": 0, "actions": 0, "frames": 0, "compressed_trajectory_points": 0}
        gap_frames = max(0, 24 - int(readiness.get("frames", 0) or 0))
        gap_actions = max(0, 30 - int(readiness.get("actions", 0) or 0))
        gap_sessions = max(0, 1 - int(readiness.get("sessions", 0) or 0))
        training_gap = "已满足训练样本下限" if readiness.get("ready") else "有效帧 {}，有效动作 {}，可训练会话 {}；还差 {} 帧 / {} 个有效点击、滚轮或移动样本 / {} 个会话".format(readiness.get("frames", 0), readiness.get("actions", 0), readiness.get("sessions", 0), gap_frames, gap_actions, gap_sessions)
        info = {
            "state": state,
            "mode": state,
            "mode_reason": "状态机当前值",
            "sleep_exit_condition": sleep_exit_condition,
            "recording": recording,
            "mouse_source_enabled": mouse_source_enabled,
            "client_valid": client_rect is not None,
            "client_validity": "合法" if client_rect is not None else client_reason,
            "frames": frames,
            "mouse": mouse,
            "session": session or "无",
            "recovery_pending": bool(self.recovery_pending),
            "recovery_reason": self.recovery_reason,
            "cpu": sample["cpu"],
            "memory": sample["memory"],
            "pool_size": pool_size,
            "pool_size_fast": pool_size,
            "pool_breakdown": breakdown,
            "model_count": model_count,
            "current_model_onnx_verified": onnx_verified,
            "current_model_id": best_model.get("id",
            "") if isinstance(best_model, dict) else "",
            "current_model_layer": model_layer,
            "current_model_allowed_actions": allowed_actions,
            "capacity": capacity,
            "journal_backlog": journal_backlog,
            "queue_snapshot": queue_snapshot,
            "resource": dict(sample),
            "gpu_name": self.resources.backend.name(),
            "backend": self.resources.backend.name(),
            "gpu_backend_status": backend_status,
            "gpu_runtime_provider": backend_status.get("runtime_provider",
            "不可用"),
            "gpu_real_ready": bool(backend_status.get("runtime_ready")),
            "gpu_failure_reason": backend_status.get("gpu_failure_reason",
            ""),
            "recent_errors": recent_errors,
            "training_readiness": readiness,
            "training_gap": training_gap,
            "gpu": sample.get("gpu"),
            "gpu_total": sample.get("gpu_dedicated_total"),
            "gpu_used": sample.get("gpu_dedicated_used"),
            "gpu_free": sample.get("gpu_dedicated_free"),
            "gpu_batch_size": 0,
            "cpu_workers": capture_budget.cpu_workers,
            "capture_fps": 1.0 / max(0.001, capture_budget.next_interval),
            "capture_resolution": "{}×{}".format(*capture_budget.max_capture_resolution),
            "queue_age": sample.get("queue_age", 0.0),
            "pipeline_queue_age": sample.get("pipeline_queue_age", 0.0),
            "ui_heartbeat_jitter_ms": sample.get("ui_heartbeat_jitter_ms", 0.0),
            "resource_state": capture_budget.state,
            "pause_reason": capture_budget.pause_reason or "无",
            "metric_sources": sample.get("metric_sources", {}),
            "ldplayer_cpu": sample.get("ldplayer_cpu", 0.0),
            "program_cpu": sample.get("process_cpu", 0.0),
            "program_gpu": sample.get("program_gpu"),
            "ldplayer_gpu": sample.get("ldplayer_gpu"),
            "gpu_sampling_source": sample.get("gpu_sampling_source",
            "不可用"),
            "disk_write_latency": sample.get("disk_write_latency"),
            "sqlite_latency": sample.get("sqlite_latency", 0.0),
            "reward_definition_version": REWARD_DEFINITION_VERSION
        }
        return info

    def shutdown(self):
        with self.shutdown_lock:
            if self.shutdown_started:
                return
            self.shutdown_started = True
        self.request_idle("程序关闭")
        self.cancel_event.set()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with self.lock:
                active = [thread for thread in self.worker_threads if thread.is_alive() and thread is not threading.current_thread()]
            if not active:
                break
            for thread in active:
                thread.join(min(0.5, max(0.05, deadline - time.monotonic())))
        flushed = self.flush_mouse_records(5.0)
        self.raw_mouse_stop.set()
        self.raw_mouse_thread.join(5.0)
        self.writer_stop.set()
        self.writer.join(5.0)
        try:
            self.db_writer.close(5.0)
        except Exception:
            pass
        self.control_queue.put(None)
        self.control_thread.join(3.0)
        self.hook.stop()
        self.keyboard_hook.stop()
        self.window_guard.stop()
        with self.lock:
            active = [thread for thread in self.worker_threads if thread.is_alive()]
        if active or not flushed:
            try:
                self.store.add_system_event(None, "incomplete_shutdown", {"active_threads": [thread.name for thread in active], "mouse_flushed": flushed, "time": time.time()})
            except Exception:
                pass
        try:
            with self.store.lock:
                store_open = self.store.conn is not None
            if store_open:
                self.store.recover_deletions()
                self.store._compact_database()
                self.store.validate_consistency()
        except Exception as error:
            note_strict_exception("controller_shutdown_store_maintenance", error, {})
        finally:
            self.store.close()
            self.resources.shutdown()

def work_area_for_window(window=None):
    try:
        if window is not None and window.winfo_exists():
            x = window.winfo_rootx(); y = window.winfo_rooty(); w = max(1, window.winfo_width()); h = max(1, window.winfo_height())
        else:
            x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN); y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN); w = max(1, user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)); h = max(1, user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
        rect = RECT(int(x), int(y), int(x + w), int(y + h))
        monitor = user32.MonitorFromRect(ctypes.byref(rect), 2)
        info = MONITORINFO(); info.cbSize = ctypes.sizeof(MONITORINFO)
        if monitor and user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return (info.rcWork.left, info.rcWork.top, info.rcWork.right, info.rcWork.bottom)
    except Exception:
        pass
    width = max(800, user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
    height = max(600, user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
    return (0, 0, width, height)

def storage_status_text(storage_path):
    try:
        root = Path(storage_path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        checked_storage_path(root / "config" / "settings.json", root)
        test = checked_storage_path(root / ".write_test", root)
        writable = False
        try:
            with test.open("wb") as handle:
                handle.write(b"1")
                handle.flush()
                os.fsync(handle.fileno())
            test.unlink(missing_ok=True)
            writable = True
        except Exception:
            writable = False
        free = shutil.disk_usage(root).free
        return "存储根目录已锁定：{}{}路径校验：通过{}可写性：{}{}剩余空间：{:.2f} GB".format(root, chr(10), chr(10), "通过" if writable else "失败", chr(10), free / 1024 / 1024 / 1024)
    except Exception as error:
        return "存储根目录：{}{}路径校验：失败{}可写性：失败{}原因：{}".format(storage_path, chr(10), chr(10), chr(10), error)

def emulator_window_text(settings):
    pid = int(settings.data.get("emulator_pid", 0) or 0)
    title = str(settings.data.get("emulator_title", "") or "")
    path = str(settings.data.get("emulator_path", "") or "")
    if pid:
        return "PID {} · {}{}{}".format(pid, title or "无标题", chr(10), path)
    return "未绑定；请点击按钮选择已打开的目标窗口"

class Panel:
    def __init__(self, root):
        self.root = root
        self.settings = Settings()
        self.events = queue.Queue(maxsize=512)
        self.pending_state_event = None
        self.pending_progress_event = None
        self.notice_drop_seen = {}
        self.controller = Controller(self.settings, self.enqueue)
        self.notice_last_seen = {}
        self.notice_critical_seen = set()
        self.startup_self_check_failed = False
        self.path_var = StringVar(value=self.settings.data["emulator_path"])
        self.emulator_window_var = StringVar(value=emulator_window_text(self.settings))
        self.storage_var = StringVar(value=storage_status_text(self.settings.data["storage_path"]))
        self.instance_var = self.emulator_window_var
        self.experience_var = StringVar(value=self.format_bytes(self.settings.data["experience_limit"]))
        self.model_var = StringVar(value=str(self.settings.data["model_limit"]) + " 个")
        self.mode_var = StringVar(value="空闲")
        self.status_var = StringVar(value=("配置读取错误：" + "；".join(self.settings.config_errors)) if self.settings.config_errors else "控制面板已就绪。")
        self.performance_var = StringVar(value="CPU 0.0% · 内存 0.0%")
        self.layout_after = None
        self.layout_signature = None
        self.panel_restore_geometry = None
        self.panel_hidden_for_mode = False
        self.footer_status_label = None
        self.footer_perf_label = None
        self.progress_var = DoubleVar(value=0.0)
        self.mode_buttons = []
        self.configuration_buttons = []
        self.control_buttons = []
        self.scroll_canvas = None
        self.build()
        check_ok, check_detail = self.startup_self_check()
        if not check_ok:
            self.startup_self_check_failed = True
            self.status_var.set("启动自检失败：" + check_detail)
            self.mode_var.set("自检失败")
            try:
                self.controller.shutdown()
            except Exception:
                pass
            for button in self.control_buttons:
                try:
                    button.configure(state="disabled")
                except Exception:
                    pass
        try:
            self.root.update_idletasks()
            register_own_overlay_window(self.root.winfo_id(), True)
        except Exception:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        if not self.startup_self_check_failed:
            self.root.after(90, self.drain)
            self.root.after(1200, self.refresh_performance)
            self.heartbeat_expected = time.monotonic() + 0.25
            self.root.after(250, self.ui_heartbeat)

    def startup_self_check(self):
        checks = []
        checks.append((self.controller.current_state() == "idle", "初始状态不是 idle"))
        checks.append((len(self.control_buttons) == 8 and all(button.winfo_exists() for button in self.control_buttons), "8 个按钮不存在或数量不对"))
        checks.append((getattr(self, "progress_widget", None) is not None and self.progress_widget.winfo_exists(), "进度条不存在"))
        try:
            checks.append((tuple(bool(item) for item in self.root.resizable()) == (True, True), "窗口不是 resizable(True, True)"))
        except Exception:
            checks.append((False, "窗口 resizable 状态无法验证"))
        root_path = Path(self.settings.data["storage_path"]).expanduser().resolve()
        checks.append((root_path.exists(), "storage root 不存在"))
        try:
            checked_storage_path(root_path / "config" / "settings.json", root_path)
            storage_ok = True
        except Exception:
            storage_ok = False
        checks.append((storage_ok, "生成路径未通过 checked_storage_path"))
        allowed = {key: set(value) for key, value in STATE_TRANSITION_EVENTS.items()}
        checks.append((self.controller.state_transitions == allowed, "状态转移表含未允许转移"))
        try:
            assert_transition("manual_sleep_task2_done", "sleep", "idle")
            assert_transition("auto_sleep_task2_done_resume_training", "sleep", "training")
            transition_assertions = True
        except AssertionError:
            transition_assertions = False
        checks.append((transition_assertions, "状态转移断言自检失败"))
        checks.append((REWARD_DEFINITION_VERSION == "screen_score_only", "reward 定义版本不是 screen_score_only"))
        try:
            source_path = Path(__file__).resolve()
            source_text = source_path.read_text(encoding="utf-8")
            comment_tokens = [token for token in tokenize.generate_tokens(io.StringIO(source_text).readline) if token.type == tokenize.COMMENT]
            tree = compile(source_text, str(source_path), "exec", ast.PyCF_ONLY_AST)
            docstrings = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node) is not None:
                    docstrings.append(node)
            checks.append((not comment_tokens and not docstrings, "当前文件存在 Python 注释或 docstring"))
        except Exception as error:
            checks.append((False, "源码注释/docstring 自检失败：" + str(error)))
        for ok, message in checks:
            if not ok:
                return False, message
        return True, "通过"

    def ui_heartbeat(self):
        try:
            now = time.monotonic()
            jitter_ms = max(0.0, (now - float(self.heartbeat_expected)) * 1000.0)
            self.controller.resources.update_ui_heartbeat(jitter_ms)
            self.heartbeat_expected = now + 0.25
        except Exception:
            pass
        try:
            self.root.after(250, self.ui_heartbeat)
        except Exception:
            pass

    def format_bytes(self, value):
        return "{:.2f} GB".format(float(value) / 1024 / 1024 / 1024)

    def enqueue(self, kind, payload):
        if kind == "state":
            self.pending_state_event = (kind, payload)
        elif kind == "progress":
            self.pending_progress_event = (kind, payload)
        elif kind == "notice":
            text = str(payload.get("message", payload)) if isinstance(payload, dict) else str(payload)
            category = str(payload.get("category", "runtime")) if isinstance(payload, dict) else "runtime"
            if category != "critical":
                key = category + chr(31) + text
                now = time.monotonic()
                if now - float(self.notice_drop_seen.get(key, 0.0) or 0.0) < 30.0:
                    return
                self.notice_drop_seen[key] = now
            try:
                self.events.put_nowait((kind, payload))
            except queue.Full:
                return
            return
        else:
            try:
                self.events.put_nowait((kind, payload))
            except queue.Full:
                return
            return
        if self.events.full():
            return
        try:
            self.events.put_nowait((kind, payload))
        except queue.Full:
            pass

    def button(self, parent, text, command, color, row=None, column=None, **grid):
        item = Button(parent, text=text, command=command, bg=color, fg="white", activebackground=color, activeforeground="white", relief="flat", bd=0, font=("Microsoft YaHei UI", 10, "bold"), cursor="hand2", padx=14, pady=10, takefocus=True)
        if row is not None:
            item.grid(row=row, column=column, **grid)
        return item

    def build(self):
        self.root.title("目标窗口智能学习与训练控制面板")
        wx1, wy1, wx2, wy2 = work_area_for_window(self.root)
        width = int(min(960, (wx2 - wx1) * 0.72))
        height = int(min(660, (wy2 - wy1) * 0.78))
        self.root.geometry("{}x{}+{}+{}".format(width, height, wx1 + max(0, ((wx2 - wx1) - width) // 2), wy1 + max(0, ((wy2 - wy1) - height) // 2)))
        self.root.minsize(360, 420)
        self.root.resizable(True, True)
        self.root.configure(bg="#101826")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        host = Frame(self.root, bg="#101826")
        host.grid(row=0, column=0, sticky="nsew")
        host.grid_columnconfigure(0, weight=1)
        host.grid_rowconfigure(0, weight=1)
        canvas = Canvas(host, bg="#101826", highlightthickness=0, bd=0, yscrollincrement=16)
        vertical = ttk.Scrollbar(host, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vertical.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        outer = Frame(canvas, bg="#101826", padx=18, pady=16)
        outer_id = canvas.create_window((0, 0), window=outer, anchor="nw")
        self.scroll_canvas = canvas

        def sync_region(event=None):
            bounds = canvas.bbox("all")
            if bounds is not None:
                canvas.configure(scrollregion=bounds)

        def sync_width(event):
            canvas.itemconfigure(outer_id, width=max(320, event.width))
            sync_region()

        outer.bind("<Configure>", sync_region)
        canvas.bind("<Configure>", sync_width)
        self.root.bind_all("<MouseWheel>", self.scroll_wheel, add="+")
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(2, weight=1)
        header = Frame(outer, bg="#101826")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        Label(header, text="目标窗口智能学习与训练控制面板", bg="#101826", fg="white", font=("Microsoft YaHei UI", 20, "bold")).grid(row=0, column=0, sticky="w")
        Label(header, textvariable=self.mode_var, bg="#1e293b", fg="#f8fafc", font=("Microsoft YaHei UI", 10, "bold"), padx=12, pady=6).grid(row=0, column=1, sticky="e")
        rainbow = Canvas(outer, height=10, bg="#101826", highlightthickness=0, bd=0)
        rainbow.grid(row=1, column=0, sticky="ew", pady=(12, 16))

        def draw_rainbow(event=None):
            rainbow.delete("all")
            colors = ("#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#3b82f6", "#a855f7")
            width = max(1, rainbow.winfo_width())
            section = width / len(colors)
            for index, color in enumerate(colors):
                rainbow.create_rectangle(index * section, 0, (index + 1) * section + 1, 10, fill=color, outline=color)

        rainbow.bind("<Configure>", draw_rainbow)
        body = Frame(outer, bg="#f8fafc", padx=18, pady=18)
        body.grid(row=2, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        labels = (("目标窗口", self.emulator_window_var), ("存储路径", self.storage_var), ("经验池上限", self.experience_var), ("AI 模型数量上限", self.model_var))
        colors = ("#ef4444", "#eab308", "#22c55e", "#06b6d4")
        commands = (self.choose_emulator_window, self.choose_storage, self.change_experience, self.change_models)
        texts = ("选择窗口", "选择存储路径", "修改经验池上限", "修改AI模型数量上限")
        config_panel = Frame(body, bg="#f8fafc")
        config_panel.grid(row=0, column=0, sticky="nsew")
        config_cards = []
        value_labels = []
        for index, ((title, variable), color, command, text) in enumerate(zip(labels, colors, commands, texts)):
            card = Frame(config_panel, bg="#ffffff", highlightthickness=1, highlightbackground="#dbe4f0", padx=12, pady=11)
            card.grid_columnconfigure(0, weight=1)
            Label(card, text=title, bg="#ffffff", fg="#334155", font=("Microsoft YaHei UI", 10, "bold"), anchor="w").grid(row=0, column=0, sticky="w")
            value = Label(card, textvariable=variable, bg="#f1f5f9", fg="#0f172a", font=("Consolas", 9), anchor="w", padx=10, pady=8, justify="left", wraplength=360)
            value.grid(row=1, column=0, sticky="ew", pady=(8, 8))
            action = self.button(card, text, command, color, row=2, column=0, sticky="ew")
            self.configuration_buttons.append(action)
            config_cards.append(card)
            value_labels.append(value)
        def layout_config(event=None):
            width = max(1, config_panel.winfo_width())
            minimum = max([button.winfo_reqwidth() for button in self.configuration_buttons] + [300]) + 42
            columns = max(1, min(2, width // max(1, minimum)))
            for column in range(2):
                config_panel.grid_columnconfigure(column, weight=1 if column < columns else 0, uniform="config" if columns == 2 else "")
            for index, card in enumerate(config_cards):
                card.grid_configure(row=index // columns, column=index % columns, sticky="nsew", padx=6, pady=6)
            wrap = max(160, width // columns - 62)
            for label in value_labels:
                label.configure(wraplength=wrap)
            sync_region()
        self.layout_config = layout_config
        divider = Frame(body, bg="#cbd5e1", height=1)
        divider.grid(row=1, column=0, sticky="ew", pady=(12, 12))
        actions = Frame(body, bg="#f8fafc")
        actions.grid(row=2, column=0, sticky="ew")
        for index in range(4):
            actions.grid_columnconfigure(index, weight=1)
        info_button = self.button(actions, "更多信息", self.more_info, "#06b6d4", row=0, column=0, sticky="ew", padx=(0, 7))
        learn = self.button(actions, "学习模式", lambda: self.start_mode("learning"), "#3b82f6", row=0, column=1, sticky="ew", padx=7)
        train = self.button(actions, "训练模式", lambda: self.start_mode("training"), "#a855f7", row=0, column=2, sticky="ew", padx=7)
        sleep = self.button(actions, "睡眠模式", self.controller.start_sleep, "#ef4444", row=0, column=3, sticky="ew", padx=(7, 0))
        action_buttons = [info_button, learn, train, sleep]
        self.control_buttons = self.configuration_buttons + action_buttons
        def layout_actions(event=None):
            width = max(1, actions.winfo_width())
            minimum = max([button.winfo_reqwidth() for button in action_buttons] + [130]) + 20
            columns = max(1, min(4, width // max(1, minimum)))
            if columns == 3:
                columns = 2
            for i in range(4):
                actions.grid_columnconfigure(i, weight=1 if i < columns else 0)
            for index, button in enumerate(action_buttons):
                button.grid_configure(row=index // columns, column=index % columns, padx=5, pady=5, sticky="ew")
        self.layout_actions = layout_actions
        self.mode_buttons = [learn, train, sleep]
        Label(body, text="任务进度", bg="#f8fafc", fg="#334155", font=("Microsoft YaHei UI", 10, "bold"), anchor="w").grid(row=3, column=0, sticky="w", pady=(17, 6))
        progress = ttk.Progressbar(body, orient="horizontal", maximum=100.0, variable=self.progress_var, mode="determinate")
        self.progress_widget = progress
        progress.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        footer = Frame(body, bg="#eef2ff", padx=12, pady=10)
        footer.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        footer.grid_columnconfigure(0, weight=1)
        self.footer_status_label = Label(footer, textvariable=self.status_var, bg="#eef2ff", fg="#1e3a8a", font=("Microsoft YaHei UI", 9), anchor="w", justify="left")
        self.footer_perf_label = Label(footer, textvariable=self.performance_var, bg="#eef2ff", fg="#475569", font=("Microsoft YaHei UI", 9), anchor="e")
        self.footer_status_label.grid(row=0, column=0, sticky="ew")
        self.footer_perf_label.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.footer = footer
        def schedule_layout(event=None):
            if self.layout_after is not None:
                self.root.after_cancel(self.layout_after)
            self.layout_after = self.root.after(40, self.apply_adaptive_layout)
        self.root.bind("<Configure>", schedule_layout, add="+")
        config_panel.bind("<Configure>", schedule_layout)
        actions.bind("<Configure>", schedule_layout)
        self.root.after_idle(self.apply_adaptive_layout)
        self.root.after_idle(sync_region)

    def apply_adaptive_layout(self):
        self.layout_after = None
        try:
            if hasattr(self, "layout_config"):
                self.layout_config()
            if hasattr(self, "layout_actions"):
                self.layout_actions()
            if self.footer_status_label is not None and self.footer_perf_label is not None:
                footer_width = max(1, self.footer.winfo_width())
                perf_width = self.footer_perf_label.winfo_reqwidth()
                inline = footer_width >= perf_width + 360
                self.footer_status_label.grid_configure(row=0, column=0, columnspan=1 if inline else 2, sticky="ew")
                self.footer_perf_label.grid_configure(row=0 if inline else 1, column=1 if inline else 0, columnspan=1 if inline else 2, sticky="e" if inline else "w", padx=(12, 0) if inline else (0, 0), pady=(0, 0) if inline else (6, 0))
                wrap = max(160, footer_width - (perf_width + 38 if inline else 24))
                self.footer_status_label.configure(wraplength=wrap)
            wx1, wy1, wx2, wy2 = work_area_for_window(self.root)
            dpi = 96
            try:
                dpi = int(self.root.winfo_fpixels("1i"))
            except Exception:
                pass
            signature = (wx1, wy1, wx2, wy2, dpi)
            if signature != self.layout_signature:
                self.layout_signature = signature
                min_width = max(360, max([button.winfo_reqwidth() for button in self.configuration_buttons + self.mode_buttons] + [320]) + 80)
                self.root.minsize(min_width, 420)
        except Exception:
            pass

    def scroll_wheel(self, event):
        canvas = self.scroll_canvas
        if canvas is None:
            return None
        try:
            x = self.root.winfo_pointerx()
            y = self.root.winfo_pointery()
            inside = canvas.winfo_rootx() <= x < canvas.winfo_rootx() + canvas.winfo_width() and canvas.winfo_rooty() <= y < canvas.winfo_rooty() + canvas.winfo_height()
            if not inside or not event.delta:
                return None
            steps = -1 if event.delta > 0 else 1
            canvas.yview_scroll(steps * 3, "units")
            return "break"
        except Exception:
            return None

    def restore_panel(self):
        try:
            if self.panel_restore_geometry:
                self.root.geometry(self.panel_restore_geometry)
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.panel_hidden_for_mode = False
        except Exception:
            pass

    def hide_panel_for_mode(self):
        self.panel_restore_geometry = self.root.winfo_geometry()
        try:
            self.root.attributes("-topmost", False)
        except Exception:
            pass
        try:
            self.root.iconify()
            self.root.update_idletasks()
            self.panel_hidden_for_mode = True
            return True
        except Exception:
            pass
        try:
            wx1, wy1, wx2, wy2 = work_area_for_window(self.root)
            self.root.geometry("{}x{}+{}+{}".format(max(1, self.root.winfo_width()), max(1, self.root.winfo_height()), wx2 + 200, wy2 + 200))
            self.root.update_idletasks()
            self.panel_hidden_for_mode = True
            return True
        except Exception:
            return False

    def start_mode(self, mode):
        if self.controller.busy():
            self.controller.emit("notice", "当前模式：" + self.controller.current_state() + "，拒绝重复进入。")
            return False
        if not self.hide_panel_for_mode():
            self.controller.emit("notice", "控制面板无法临时隐藏，禁止进入模式以避免遮挡客户区。")
            self.restore_panel()
            return False
        def continue_start():
            self.controller.request_start_session(mode)
            self.status_var.set("已投递进入模式请求，等待控制线程预检。")
        self.root.after(80, continue_start)
        return True

    def protect_configuration(self):
        if self.controller.busy():
            messagebox.showwarning("当前状态", "运行中的模式不会修改配置。请先返回空闲状态。", parent=self.root)
            return False
        return True

    def choose_emulator_window(self):
        if not self.protect_configuration():
            return
        candidates = find_emulator_window_candidates(self.settings.data.get("emulator_path", ""))
        if not candidates:
            messagebox.showwarning("未发现窗口", "未发现可见的目标窗口。请先启动目标窗口后再选择。", parent=self.root)
            return
        dialog = Toplevel(self.root)
        dialog.title("选择目标窗口")
        dialog.geometry("900x420")
        dialog.minsize(720, 320)
        dialog.transient(self.root)
        dialog.grab_set()
        container = Frame(dialog, bg="#f8fafc", padx=12, pady=12)
        container.grid(row=0, column=0, sticky="nsew")
        dialog.grid_rowconfigure(0, weight=1)
        dialog.grid_columnconfigure(0, weight=1)
        Label(container, text="双击窗口或选中后确认", bg="#f8fafc", fg="#0f172a", font=("Microsoft YaHei UI", 11, "bold"), anchor="w").grid(row=0, column=0, sticky="ew", pady=(0, 8))
        columns = ("title", "pid", "path", "size", "backend")
        tree = ttk.Treeview(container, columns=columns, show="headings", selectmode="browse")
        headings = {"title": "标题", "pid": "PID", "path": "路径", "size": "尺寸", "backend": "平台后端"}
        widths = {"title": 260, "pid": 80, "path": 330, "size": 110, "backend": 110}
        for key in columns:
            tree.heading(key, text=headings[key])
            tree.column(key, width=widths[key], anchor="w", stretch=key in ("title", "path"))
        scroll_y = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        scroll_x = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        tree.grid(row=1, column=0, sticky="nsew")
        scroll_y.grid(row=1, column=1, sticky="ns")
        scroll_x.grid(row=2, column=0, sticky="ew")
        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)
        for index, item in enumerate(candidates):
            rect = item.get("rect") or item.get("client_rect") or (0, 0, 0, 0)
            try:
                size = "{}×{}".format(max(0, int(rect[2]) - int(rect[0])), max(0, int(rect[3]) - int(rect[1])))
            except Exception:
                size = "未知"
            backend = str(item.get("class") or getattr(PLATFORM_BACKEND, "name", "windows"))
            tree.insert("", "end", iid=str(index), values=(str(item.get("title") or "无标题"), str(item.get("pid", "")), str(item.get("path", "")), size, backend))
        buttons = Frame(container, bg="#f8fafc")
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(10, 0))
        chosen_holder = {"item": None}
        def confirm(event=None):
            selection = tree.selection()
            if not selection:
                messagebox.showwarning("请选择窗口", "请先在列表中选择一个目标窗口。", parent=dialog)
                return
            chosen_holder["item"] = candidates[int(selection[0])]
            dialog.destroy()
        def cancel():
            dialog.destroy()
        Button(buttons, text="取消", command=cancel, bg="#e2e8f0", fg="#0f172a", relief="flat", padx=18, pady=8).grid(row=0, column=0, padx=(0, 8))
        Button(buttons, text="确认选择", command=confirm, bg="#3b82f6", fg="white", relief="flat", padx=18, pady=8).grid(row=0, column=1)
        tree.bind("<Double-1>", confirm)
        if len(candidates) == 1:
            tree.selection_set("0")
            tree.focus("0")
        self.root.wait_window(dialog)
        chosen = chosen_holder.get("item")
        if chosen is None:
            return
        self.settings.data["emulator_path"] = str(chosen.get("path", ""))
        self.settings.data["emulator_pid"] = int(chosen["pid"])
        self.settings.data["emulator_title"] = str(chosen["title"])
        self.settings.save()
        self.path_var.set(self.settings.data["emulator_path"])
        self.emulator_window_var.set(emulator_window_text(self.settings))
        self.status_var.set("已选择目标窗口；会话期间将固定校验窗口、PID 和路径。")

    def choose_emulator(self):
        if not self.protect_configuration(): return
        selected=filedialog.askopenfilename(parent=self.root,title="选择雷电模拟器路径",initialfile=Path(self.settings.data["emulator_path"]).name,filetypes=[("可执行文件","*.exe"),("所有文件","*.*")])
        if selected:
            self.settings.data["emulator_path"]=selected;self.settings.data["emulator_pid"]=0;self.settings.data["emulator_title"]="";self.settings.save();self.path_var.set(selected);self.emulator_window_var.set(emulator_window_text(self.settings));self.status_var.set("已更新雷电模拟器路径。")

    def choose_emulator_instance(self):
        if not self.protect_configuration(): return
        candidates=find_emulator_candidates(self.settings.data["emulator_path"])
        if not candidates:
            messagebox.showwarning("未发现实例","当前路径没有对应的可见雷电实例。",parent=self.root);return
        choices=["{}: PID {} · {}".format(index+1,item["pid"],item["title"] or "无标题") for index,item in enumerate(candidates)]
        answer=simpledialog.askstring("选择雷电实例","请输入实例编号：" + chr(10) + chr(10).join(choices),initialvalue="1" if len(candidates)==1 else "",parent=self.root)
        if answer is None:return
        try:
            index=int(answer.strip())-1
            chosen=candidates[index]
        except Exception:
            messagebox.showerror("输入错误","请输入列表中的有效实例编号。",parent=self.root);return
        self.settings.data["emulator_pid"]=int(chosen["pid"]);self.settings.data["emulator_title"]=str(chosen["title"]);self.settings.save();self.emulator_window_var.set(emulator_window_text(self.settings));self.status_var.set("已绑定雷电实例；会话期间将固定校验 HWND 和 PID。")

    def choose_storage(self):
        if not self.protect_configuration():
            return
        selected = filedialog.askdirectory(parent=self.root, title="选择存储路径", initialdir=self.settings.data["storage_path"])
        if selected:
            try:
                self.settings.migrate_storage_path(selected)
            except Exception as error:
                messagebox.showerror("迁移失败", "配置迁移到新存储路径失败：" + str(error), parent=self.root)
                return
            self.storage_var.set(storage_status_text(self.settings.data["storage_path"]))
            self.status_var.set("已更新存储路径；settings.json 已迁移到该路径的 config 目录。")

    def change_experience(self):
        if not self.protect_configuration():
            return
        current = self.settings.data["experience_limit"] / 1024 / 1024 / 1024
        answer = simpledialog.askstring("修改经验池上限", "请输入经验池上限，单位为 GB：", initialvalue="{:.2f}".format(current), parent=self.root)
        if answer is None:
            return
        try:
            value = float(answer.strip().lower().replace("gb", "").strip())
            if not 0.1 <= value <= 4096:
                raise ValueError
            self.settings.data["experience_limit"] = int(value * 1024 * 1024 * 1024)
            self.settings.save()
            self.experience_var.set(self.format_bytes(self.settings.data["experience_limit"]))
            self.status_var.set("已更新经验池上限。")
        except ValueError:
            messagebox.showerror("输入错误", "请输入 0.1 到 4096 之间的有效 GB 数值。", parent=self.root)

    def change_models(self):
        if not self.protect_configuration():
            return
        answer = simpledialog.askstring("修改AI模型数量上限", "请输入 AI 模型数量上限：", initialvalue=str(self.settings.data["model_limit"]), parent=self.root)
        if answer is None:
            return
        try:
            value = int(answer.strip())
            if not 1 <= value <= 100000:
                raise ValueError
            self.settings.data["model_limit"] = value
            self.settings.save()
            self.model_var.set(str(value) + " 个")
            self.status_var.set("已更新 AI 模型数量上限。")
        except ValueError:
            messagebox.showerror("输入错误", "请输入 1 到 100000 之间的整数。", parent=self.root)

    def more_info(self):
        info = self.controller.information_snapshot()
        window = Toplevel(self.root); window.title("更多信息")
        wx1, wy1, wx2, wy2 = work_area_for_window(self.root); width = max(520, min(980, int((wx2 - wx1) * 0.55))); height = max(420, min(820, int((wy2 - wy1) * 0.65)))
        window.geometry("{}x{}+{}+{}".format(width, height, wx1 + max(0, ((wx2 - wx1) - width) // 2), wy1 + max(0, ((wy2 - wy1) - height) // 2))); window.resizable(True, True); window.configure(bg="#0f172a"); window.grid_columnconfigure(0, weight=1); window.grid_rowconfigure(1, weight=1)
        Label(window, text="运行信息", bg="#0f172a", fg="white", font=("Microsoft YaHei UI", 18, "bold"), padx=20, pady=18).grid(row=0, column=0, sticky="w")
        canvas = Canvas(window, bg="#f8fafc", highlightthickness=0, bd=0); scroll = ttk.Scrollbar(window, orient="vertical", command=canvas.yview); canvas.configure(yscrollcommand=scroll.set); canvas.grid(row=1, column=0, sticky="nsew", padx=(16, 0), pady=(0, 16)); scroll.grid(row=1, column=1, sticky="ns", padx=(0, 16), pady=(0, 16))
        content = Frame(canvas, bg="#f8fafc", padx=20, pady=18); content_id = canvas.create_window((0, 0), window=content, anchor="nw"); content.grid_columnconfigure(1, weight=1); content.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all"))); canvas.bind("<Configure>", lambda event: canvas.itemconfigure(content_id, width=event.width))
        def number(value, unit=""):
            return "不可用" if value is None else "{:.1f}{}".format(float(value), unit)
        capacity = info.get("capacity", {})
        backend_status = info.get("gpu_backend_status", {})
        provider = backend_status.get("runtime_provider", "不可用")
        warmup_text = "已过 30 秒 warm-up" if backend_status.get("warmup_complete") else "warm-up {:.1f} 秒".format(float(backend_status.get("warmup_age", 0.0) or 0.0))
        if backend_status.get("runtime_ready") and provider != "不可用":
            gpu_state = "GPU ONNX 已验证策略；provider={}；{}".format(provider, warmup_text)
        elif provider != "不可用":
            reason = backend_status.get("gpu_failure_reason") or "等待模型加载或 warm-up 完成"
            gpu_state = "GPU ONNX 未就绪；provider={}；{}；{}".format(provider, warmup_text, reason)
        else:
            gpu_state = "GPU ONNX 不可用；" + (backend_status.get("gpu_failure_reason") or "未发现 CUDA/DML provider")
        recent_error_lines = []
        for item in info.get("recent_errors", []):
            payload = item.get("payload", {}) if isinstance(item, dict) else {}
            moment = time.strftime("%m-%d %H:%M:%S", time.localtime(float(item.get("created", 0.0) or 0.0))) if isinstance(item, dict) else "未知时间"
            message = payload.get("message") or payload.get("error") or payload.get("reason") or str(payload)
            details = []
            for key in ("exception_type", "errno", "winerror", "stage", "path", "filename", "disk_free", "sqlite_busy_count", "session_id"):
                value = payload.get(key)
                if value not in (None, "", {}):
                    details.append("{}={}".format(key, value))
            for key in ("wal", "capacity", "journal_backlog"):
                value = payload.get(key)
                if isinstance(value, dict):
                    details.append("{}={}".format(key, json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:240]))
            recent_error_lines.append("{} · {} · {}{}".format(moment, item.get("kind", "事件"), message, " · " + "；".join(details) if details else ""))
        queue_snapshot = info.get("queue_snapshot", {})
        journal_backlog = info.get("journal_backlog", {})
        queue_line = "capture={capture_queue} feature={feature_queue} persist={persist_queue} mouse={mouse_queue} segment={mouse_segment_queue} raw={raw_mouse_queue}/{raw_critical_queue} hook={raw_hook_ring} degraded={mouse_sqlite_degraded} replay_pause={frame_replay_paused}".format(**dict({"capture_queue": 0, "feature_queue": 0, "persist_queue": 0, "mouse_queue": 0, "mouse_segment_queue": 0, "raw_mouse_queue": 0, "raw_critical_queue": 0, "raw_hook_ring": 0, "mouse_sqlite_degraded": False, "frame_replay_paused": False}, **queue_snapshot))
        journal_line = "total={total} frame={frame} mouse={mouse} segment={mouse_segment}".format(**dict({"total": 0, "frame": 0, "mouse": 0, "mouse_segment": 0}, **journal_backlog))
        rows = [
            ("当前状态", info["state"]),
            ("当前模式", info.get("mode", info["state"])),
            ("进入原因", info.get("mode_reason", "状态机当前值")),
            ("退出条件", info.get("sleep_exit_condition", "按状态机规则")),
            ("客户区合法性", info.get("client_validity", "未知")),
            ("正在记录", "是" if info.get("recording") else "否"),
            ("鼠标来源区分", "启用" if info.get("mouse_source_enabled") else "未启用"),
            ("当前模型 ONNX 验证", "通过" if info.get("current_model_onnx_verified") else "未通过或未加载"),
            ("当前模型 ID", str(info.get("current_model_id") or "无")),
            ("当前模型层级", str(info.get("current_model_layer", "无")) + "；允许动作=" + ",".join(str(item) for item in info.get("current_model_allowed_actions", []))),
            ("训练样本缺口", info.get("training_gap", "未知")),
            ("UI 心跳抖动", "{:.1f} ms".format(float(info.get("ui_heartbeat_jitter_ms", 0.0) or 0.0))),
            ("GPU 真实参与", "是" if info.get("gpu_real_ready") else "否"),
            ("本次会话", info["session"]),
            ("本次记录画面", str(info["frames"])),
            ("本次记录鼠标事件", str(info["mouse"])),
            ("本程序 CPU", number(info.get("program_cpu"), "%") + " · " + info.get("metric_sources", {}).get("本程序 CPU", "未知来源")),
            ("目标进程 CPU", number(info.get("ldplayer_cpu"), "%") + " · " + info.get("metric_sources", {}).get("目标进程 CPU", "未知来源")),
            ("策略执行后端", info.get("backend", "CPU 表格策略")),
            ("GPU 状态", gpu_state),
            ("ONNX Provider", str(provider)),
            ("ONNX Runtime Ready", "是" if backend_status.get("runtime_ready") else "否"),
            ("GPU 模型路径", str(backend_status.get("gpu_model_path") or "未加载")),
            ("GPU 失败原因", str(backend_status.get("gpu_failure_reason") or "无")),
            ("最近策略后端", str(backend_status.get("last_backend", info.get("backend", "CPU 表格策略")))),
            ("本程序 GPU 引擎", number(info.get("program_gpu"), "%") + " · " + info.get("gpu_sampling_source", "不可用")),
            ("目标进程 GPU 引擎", number(info.get("ldplayer_gpu"), "%") + " · " + info.get("gpu_sampling_source", "不可用")),
            ("可用显存", "不可用" if info.get("gpu_free") is None else self.format_bytes(info.get("gpu_free", 0))),
            ("磁盘写入延迟", number(info.get("disk_write_latency"), " ms") + " · fsync 探针"),
            ("SQLite 写入延迟", number(info.get("sqlite_latency"), " ms") + " · 实际事务计时"),
            ("当前截图频率", "约 {:.2f} FPS".format(info.get("capture_fps", 0.0))),
            ("当前截图分辨率", info.get("capture_resolution", "未知")),
            ("鼠标队列年龄", "{:.2f} 秒".format(info.get("queue_age", 0.0))),
            ("流水线队列年龄", "{:.2f} 秒".format(info.get("pipeline_queue_age", 0.0))),
            ("Pipeline/鼠标队列", queue_line),
            ("Journal backlog", journal_line),
            ("当前资源状态", info.get("resource_state", "正常")),
            ("限速原因", info.get("pause_reason", "无")),
            ("经验池大小", self.format_bytes(info["pool_size"]) + "（容量账本；更多信息时已校验）"),
            ("容量阶段", "{}% 预警；事务预留 {}".format(capacity.get("tier", 0), self.format_bytes(capacity.get("transaction_reserve", 0)))),
            ("经验池硬状态", "已停止采集；{} / 目标 {}".format(self.format_bytes(capacity.get("remaining", 0)), self.format_bytes(capacity.get("target", 0))) if capacity.get("blocked") else "正常"),
            ("AI 模型数量", str(info["model_count"])),
            ("奖励定义", "reward = exact_screen_score"),
            ("目标窗口路径", self.settings.data.get("emulator_path", "")),
            ("检索保障", "两阶段评分：LSH/哈希召回 → 灰度结构/颜色/局部变化精确距离 → 历史聚合相似度"),
            ("资源保护", "预算合同、in-flight 上限、字节预算、队列限速、降分辨率、SQLite/磁盘延迟监测、硬容量阻断"),
            ("最近 5 条关键错误", "无" if not recent_error_lines else "\n".join(recent_error_lines[:5]))
        ]
        value_labels = {}
        for index, (name, value) in enumerate(rows):
            Label(content, text=name, bg="#f8fafc", fg="#475569", font=("Microsoft YaHei UI", 10, "bold"), anchor="w").grid(row=index, column=0, sticky="w", pady=5)
            label = Label(content, text=value, bg="#f8fafc", fg="#0f172a", font=("Microsoft YaHei UI", 10), anchor="w", wraplength=520, justify="left")
            label.grid(row=index, column=1, sticky="ew", padx=(18, 0), pady=5)
            value_labels[name] = label
        def refresh_runtime_info():
            try:
                refreshed = self.controller.information(reconcile=False)
                q = refreshed.get("queue_snapshot", {})
                j = refreshed.get("journal_backlog", {})
                qline = "capture={capture_queue} feature={feature_queue} persist={persist_queue} mouse={mouse_queue} segment={mouse_segment_queue} raw={raw_mouse_queue}/{raw_critical_queue} hook={raw_hook_ring} degraded={mouse_sqlite_degraded} replay_pause={frame_replay_paused}".format(**dict({"capture_queue": 0, "feature_queue": 0, "persist_queue": 0, "mouse_queue": 0, "mouse_segment_queue": 0, "raw_mouse_queue": 0, "raw_critical_queue": 0, "raw_hook_ring": 0, "mouse_sqlite_degraded": False, "frame_replay_paused": False}, **q))
                jline = "total={total} frame={frame} mouse={mouse} segment={mouse_segment}".format(**dict({"total": 0, "frame": 0, "mouse": 0, "mouse_segment": 0}, **j))
                errors = []
                for item in refreshed.get("recent_errors", []):
                    payload = item.get("payload", {}) if isinstance(item, dict) else {}
                    moment = time.strftime("%m-%d %H:%M:%S", time.localtime(float(item.get("created", 0.0) or 0.0))) if isinstance(item, dict) else "未知时间"
                    message = payload.get("message") or payload.get("error") or payload.get("reason") or str(payload)
                    errors.append("{} · {} · {}".format(moment, item.get("kind", "事件"), message))
                updates = {
                    "当前状态": refreshed.get("state", "未知"),
                    "当前模式": refreshed.get("mode", "未知"),
                    "客户区合法性": refreshed.get("client_validity", "未知"),
                    "正在记录": "是" if refreshed.get("recording") else "否",
                    "鼠标来源区分": "启用" if refreshed.get("mouse_source_enabled") else "未启用",
                    "当前模型 ONNX 验证": "通过" if refreshed.get("current_model_onnx_verified") else "未通过或未加载",
                    "当前模型 ID": str(refreshed.get("current_model_id") or "无"),
                    "当前模型层级": str(refreshed.get("current_model_layer", "无")) + "；允许动作=" + ",".join(str(item) for item in refreshed.get("current_model_allowed_actions", [])),
                    "训练样本缺口": refreshed.get("training_gap", "未知"),
                    "本次会话": refreshed.get("session", "无"),
                    "本次记录画面": str(refreshed.get("frames", 0)),
                    "本次记录鼠标事件": str(refreshed.get("mouse", 0)),
                    "Pipeline/鼠标队列": qline,
                    "Journal backlog": jline,
                    "经验池大小": self.format_bytes(refreshed.get("pool_size", 0)) + "（快照；精确大小继续后台校验）",
                    "AI 模型数量": str(refreshed.get("model_count", 0)),
                    "最近 5 条关键错误": "无" if not errors else "\n".join(errors[:5]),
                    "磁盘写入延迟": number(refreshed.get("disk_write_latency"), " ms") + " · fsync 探针",
                    "SQLite 写入延迟": number(refreshed.get("sqlite_latency"), " ms") + " · 实际事务计时",
                }
                def apply_runtime_refresh():
                    try:
                        if not window.winfo_exists():
                            return
                        for key, value in updates.items():
                            if key in value_labels:
                                value_labels[key].configure(text=value)
                    except Exception:
                        pass
                self.root.after(0, apply_runtime_refresh)
            except Exception:
                pass
        threading.Thread(target=refresh_runtime_info, name="MoreInfoFastRefresh", daemon=True).start()

        def refresh_reconciled_info():
            try:
                refreshed = self.controller.information(reconcile=True)
                value = self.format_bytes(refreshed.get("pool_size", 0)) + "（后台账本重算已完成）"
                def apply_refresh():
                    try:
                        if window.winfo_exists() and "经验池大小" in value_labels:
                            value_labels["经验池大小"].configure(text=value)
                    except Exception:
                        pass
                self.root.after(0, apply_refresh)
            except Exception:
                pass
        threading.Thread(target=refresh_reconciled_info, name="MoreInfoReconcile", daemon=True).start()

    def drain(self):
        try:
            for pending_name in ("pending_state_event", "pending_progress_event"):
                pending = getattr(self, pending_name, None)
                if pending is not None:
                    setattr(self, pending_name, None)
                    try:
                        self.events.put_nowait(pending)
                    except queue.Full:
                        pass
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "state":
                    state = payload.get("state", "idle")
                    names = {"idle": "空闲", "learning": "学习模式", "training": "训练模式", "sleep": "睡眠模式", "stopping": "正在停止"}
                    self.mode_var.set(names.get(state, state))
                    detail = payload.get("detail", "")
                    self.status_var.set(detail or "控制面板已就绪。")
                    if state == "idle": self.restore_panel()
                    self.performance_var.set("CPU {:.1f}% · 内存 {:.1f}%".format(payload.get("cpu", 0.0), payload.get("memory", 0.0)))
                    normal = "normal" if state == "idle" else "disabled"
                    for button in self.mode_buttons:
                        button.configure(state=normal)
                    for button in self.configuration_buttons:
                        button.configure(state=normal)
                elif kind == "progress":
                    self.progress_var.set(max(0.0, min(100.0, float(payload))))
                elif kind == "notice":
                    text = str(payload.get("message", payload)) if isinstance(payload, dict) else str(payload)
                    category = str(payload.get("category", "runtime")) if isinstance(payload, dict) else "runtime"
                    now = time.monotonic()
                    key = category + chr(31) + text
                    last = float(self.notice_last_seen.get(key, 0.0) or 0.0)
                    if now - last < 60.0:
                        continue
                    self.notice_last_seen[key] = now
                    self.status_var.set(text)
                    serious = category == "critical" or "严重" in text or "发生错误" in text
                    if category == "config":
                        messagebox.showwarning("提示", text, parent=self.root)
                    elif serious and text not in self.notice_critical_seen:
                        self.notice_critical_seen.add(text)
                        messagebox.showerror("严重错误", text, parent=self.root)
        except queue.Empty:
            pass
        try:
            self.root.after(90, self.drain)
        except Exception:
            pass

    def refresh_performance(self):
        try:
            info=self.controller.information();gpu=info.get("gpu")
            gpu_text="GPU 不可用" if gpu is None else "GPU {:.1f}%".format(gpu)
            self.performance_var.set("CPU {:.1f}% · 内存 {:.1f}% · {} · {}".format(info["cpu"],info["memory"],gpu_text,info.get("resource_state","正常")))
            self.root.after(1200,self.refresh_performance)
        except Exception:
            pass

    def close(self):
        try:
            self.controller.shutdown()
        finally:
            self.root.destroy()

def _controller_state_contract_selftest():
    transitions = {
        "idle_click_learning": {("idle", "learning")},
        "idle_click_training": {("idle", "training")},
        "manual_sleep": {("idle", "sleep")},
        "auto_sleep_worth": {("training", "stopping"), ("stopping", "sleep")},
        "esc": {("learning", "stopping"), ("training", "stopping"), ("sleep", "stopping"), ("stopping", "idle")},
        "mouse_outside": {("learning", "stopping"), ("training", "stopping"), ("stopping", "idle")},
        "client_invalid": {("learning", "stopping"), ("training", "stopping"), ("stopping", "idle")},
        "manual_sleep_task2_done": {("sleep", "idle")},
        "auto_sleep_task2_done_resume_training": {("sleep", "training")},
    }
    class FakeStore:
        def __init__(self):
            self.events = []
        def add(self, name, payload=None):
            self.events.append((name, dict(payload or {})))
    class FakeWin32:
        def __init__(self):
            self.capture_calls = 0
            self.mouse_calls = 0
            self.ai_calls = 0
        def capture(self, state):
            if state in ("idle", "sleep"):
                raise AssertionError("forbidden_capture:" + state)
            self.capture_calls += 1
        def record_mouse(self, state, source):
            if state in ("idle", "sleep"):
                raise AssertionError("forbidden_mouse:" + state)
            if state == "learning" and source != "user":
                raise AssertionError("learning_source:" + str(source))
            if state == "training" and source not in ("ai", "user", "external_injected"):
                raise AssertionError("training_source:" + str(source))
            self.mouse_calls += 1
        def ai_input(self, state):
            if state != "training":
                raise AssertionError("forbidden_ai:" + state)
            self.ai_calls += 1
    class FakeController:
        def __init__(self):
            self.state = "idle"
            self.store = FakeStore()
            self.win32 = FakeWin32()
            self.sleep_order = []
        def transition(self, event):
            for left, right in transitions.get(event, set()):
                if left == self.state:
                    self.state = right
                    self.store.add("transition", {"event": event, "to": right})
                    return True
            return False
        def sleep_task1(self):
            if self.state != "sleep":
                raise AssertionError("task1_requires_sleep")
            self.sleep_order.append("task1")
        def sleep_task2(self, origin):
            if self.sleep_order != ["task1"]:
                raise AssertionError("task2_requires_task1")
            self.sleep_order.append("task2")
            return self.transition("auto_sleep_task2_done_resume_training" if origin == "auto" else "manual_sleep_task2_done")
    fake = FakeController()
    try:
        fake.win32.capture("idle")
        raise AssertionError("idle_capture_not_blocked")
    except AssertionError as error:
        assert str(error).startswith("forbidden_capture")
    try:
        fake.win32.record_mouse("idle", "user")
        raise AssertionError("idle_mouse_not_blocked")
    except AssertionError as error:
        assert str(error).startswith("forbidden_mouse")
    try:
        fake.win32.ai_input("idle")
        raise AssertionError("idle_ai_not_blocked")
    except AssertionError as error:
        assert str(error).startswith("forbidden_ai")
    fake.transition("idle_click_learning")
    fake.win32.capture(fake.state)
    fake.win32.record_mouse(fake.state, "user")
    try:
        fake.win32.record_mouse(fake.state, "ai")
        raise AssertionError("learning_ai_source_not_blocked")
    except AssertionError as error:
        assert str(error).startswith("learning_source")
    fake.transition("esc"); fake.transition("esc")
    fake.transition("idle_click_training")
    for source in ("ai", "user", "external_injected"):
        fake.win32.record_mouse(fake.state, source)
    fake.win32.ai_input(fake.state)
    fake.transition("auto_sleep_worth"); fake.transition("auto_sleep_worth")
    try:
        fake.win32.capture(fake.state)
        raise AssertionError("sleep_capture_not_blocked")
    except AssertionError as error:
        assert str(error).startswith("forbidden_capture")
    try:
        fake.win32.record_mouse(fake.state, "user")
        raise AssertionError("sleep_mouse_not_blocked")
    except AssertionError as error:
        assert str(error).startswith("forbidden_mouse")
    fake.sleep_task1(); assert fake.sleep_task2("auto") and fake.state == "training"
    fake = FakeController(); assert fake.transition("manual_sleep"); fake.sleep_task1(); assert fake.sleep_task2("manual") and fake.state == "idle"
    flattened = {edge for edges in transitions.values() for edge in edges}
    assert ("learning", "training") not in flattened and ("idle", "sleep") in transitions["manual_sleep"]
    return True


def main():
    _controller_state_contract_selftest()
    root = Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("Horizontal.TProgressbar", troughcolor="#dbeafe", background="#3b82f6", bordercolor="#dbeafe", lightcolor="#3b82f6", darkcolor="#3b82f6")
    Panel(root)
    root.mainloop()

if __name__ == "__main__":
    main()
