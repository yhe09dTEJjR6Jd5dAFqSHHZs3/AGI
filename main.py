import ctypes
import datetime
from bisect import bisect_right
import json
import math
import os
import queue
import sqlite3
import struct
import sys
import threading
import time
import uuid
import zlib
from ctypes import wintypes
from pathlib import Path
from tkinter import Tk, Toplevel, StringVar, DoubleVar, Canvas, Frame, Label, Button, Entry, filedialog, messagebox, simpledialog
from tkinter import ttk

if os.name != "nt":
    raise SystemExit("本程序仅支持 Windows。")

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)

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

LowLevelMouseProc = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
LowLevelKeyboardProc = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

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
user32.GetWindowFromPoint.argtypes = [POINT]
user32.GetWindowFromPoint.restype = wintypes.HWND
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
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, ctypes.c_void_p, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
user32.UnhookWindowsHookEx.restype = wintypes.BOOL
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
user32.MonitorFromRect.argtypes = [ctypes.POINTER(RECT), wintypes.DWORD]
user32.MonitorFromRect.restype = wintypes.HMONITOR
user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]
user32.GetMonitorInfoW.restype = wintypes.BOOL
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

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

class ResourceMeter:
    def __init__(self):
        self.lock = threading.Lock()
        self.storage_path = Path.cwd()
        self.previous = None
        self.last_sample = 0.0
        self.value = {"cpu": 0.0, "memory": 0.0, "avail_memory": 0, "disk_free": 0, "queue": 0, "queue_age": 0.0, "io_latency": 0.0, "process_cpu": 0.0, "process_memory": 0, "capture_latency": 0.0, "sqlite_latency": 0.0, "capture_failure_rate": 0.0, "fsync_latency": 0.0, "wal_bytes": 0}
        self.process_previous = None
        self.process_last_sample = 0.0

    def sample(self):
        now = time.monotonic()
        with self.lock:
            if now - self.last_sample < 0.8:
                return dict(self.value)
            idle = FILETIME()
            kernel = FILETIME()
            user = FILETIME()
            cpu = self.value["cpu"]
            if kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                current = tuple((item.dwHighDateTime << 32) | item.dwLowDateTime for item in (idle, kernel, user))
                if self.previous is not None:
                    idle_delta = current[0] - self.previous[0]
                    total_delta = current[1] + current[2] - self.previous[1] - self.previous[2]
                    if total_delta > 0:
                        cpu = max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))
                self.previous = current
            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            memory = self.value["memory"]
            avail_memory = self.value.get("avail_memory", 0)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                memory = float(status.dwMemoryLoad)
                avail_memory = int(status.ullAvailPhys)
            disk_free = self.value.get("disk_free", 0)
            try:
                disk_free = int(__import__("shutil").disk_usage(self.storage_path).free)
            except Exception:
                pass
            io_latency = max(float(self.value.get("capture_latency", 0.0)) / 1000.0, float(self.value.get("sqlite_latency", 0.0)) / 1000.0, float(self.value.get("fsync_latency", 0.0)) / 1000.0, float(self.value.get("queue_age", 0.0)) / 10.0)
            process_cpu = self.value.get("process_cpu", 0.0)
            process_memory = self.value.get("process_memory", 0)
            try:
                creation = FILETIME(); exit_time = FILETIME(); kernel_time = FILETIME(); user_time = FILETIME()
                if kernel32.GetProcessTimes(kernel32.GetCurrentProcess(), ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel_time), ctypes.byref(user_time)):
                    proc = filetime_value(kernel_time) + filetime_value(user_time)
                    if self.process_previous is not None:
                        dt = max(0.001, now - self.process_last_sample)
                        process_cpu = max(0.0, min(100.0, (proc - self.process_previous) / 10000000.0 / dt * 100.0 / max(1, os.cpu_count() or 1)))
                    self.process_previous = proc
                    self.process_last_sample = now
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                    process_memory = int(counters.WorkingSetSize)
            except Exception:
                pass
            self.value = {"cpu": cpu, "memory": memory, "avail_memory": avail_memory, "disk_free": disk_free, "queue": self.value.get("queue", 0), "queue_age": self.value.get("queue_age", 0.0), "io_latency": io_latency, "process_cpu": process_cpu, "process_memory": process_memory, "capture_latency": self.value.get("capture_latency", 0.0), "sqlite_latency": self.value.get("sqlite_latency", 0.0), "capture_failure_rate": self.value.get("capture_failure_rate", 0.0), "fsync_latency": self.value.get("fsync_latency", 0.0), "wal_bytes": self.value.get("wal_bytes", 0)}
            self.last_sample = now
            return dict(self.value)

    def interval(self):
        sample = self.sample()
        if sample["cpu"] >= 90 or sample["memory"] >= 94:
            return 12.0
        if sample["cpu"] >= 80 or sample["memory"] >= 88:
            return 5.0
        if sample["cpu"] >= 70 or sample["memory"] >= 82:
            return 2.0
        return 1.0

    def set_storage_path(self, path):
        with self.lock:
            self.storage_path = Path(path)

    def update_queue(self, length, age=0.0):
        with self.lock:
            self.value["queue"] = int(length)
            self.value["queue_age"] = float(age)

    def update_capture_metrics(self, elapsed_ms=None, failed=False):
        with self.lock:
            if elapsed_ms is not None:
                old = float(self.value.get("capture_latency", 0.0))
                self.value["capture_latency"] = elapsed_ms if old <= 0 else old * 0.85 + float(elapsed_ms) * 0.15
            old_rate = float(self.value.get("capture_failure_rate", 0.0))
            self.value["capture_failure_rate"] = old_rate * 0.95 + (0.05 if failed else 0.0)

    def update_sqlite_latency(self, elapsed_ms):
        with self.lock:
            old = float(self.value.get("sqlite_latency", 0.0))
            self.value["sqlite_latency"] = elapsed_ms if old <= 0 else old * 0.85 + float(elapsed_ms) * 0.15

    def hard_stop(self):
        sample = self.sample()
        return sample.get("avail_memory", 1) < 256 * 1024 * 1024 or sample.get("sqlite_latency", 0.0) > 800.0 or sample.get("capture_latency", 0.0) > 800.0 or sample.get("queue_age", 0.0) > 12.0

    def allow_capture(self):
        sample = self.sample()
        return sample.get("disk_free", 0) >= 1024 * 1024 * 1024 and sample.get("queue", 0) < 9000 and sample.get("queue_age", 0.0) < 8.0 and sample.get("capture_failure_rate", 0.0) < 0.35 and sample["cpu"] < 76 and sample["memory"] < 84

    def allow_training(self):
        sample = self.sample()
        return sample.get("avail_memory", 0) >= 768 * 1024 * 1024 and sample.get("disk_free", 0) >= 1024 * 1024 * 1024 and sample["cpu"] < 60 and sample["memory"] < 76 and sample.get("sqlite_latency", 0.0) < 250.0

    def allow_maintenance(self):
        sample = self.sample()
        return sample.get("avail_memory", 0) >= 256 * 1024 * 1024 and sample.get("sqlite_latency", 0.0) < 1000.0 and sample.get("queue_age", 0.0) < 12.0

    def allow_compute(self):
        return self.allow_training()

    def critical(self):
        sample = self.sample()
        return self.hard_stop() or sample["cpu"] >= 90 or sample["memory"] >= 94

class Settings:
    def __init__(self):
        appdata = Path(os.environ.get("APPDATA", str(Path.home())))
        self.path = appdata / "LDTrainingPanel" / "settings.json"
        self.data = {
            "emulator_path": r"D:\LDPlayer9\dnplayer.exe",
            "storage_path": r"C:\Users\Administrator\Desktop\AAA",
            "experience_limit": 10 * 1024 * 1024 * 1024,
            "model_limit": 100
        }
        self.config_errors = []
        self.load()

    def load(self):
        self.config_errors = []
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as error:
            if self.path.exists():
                self.config_errors.append("配置读取失败:" + str(error))
            return
        if not isinstance(loaded, dict):
            self.config_errors.append("配置根对象不是字典")
            return
        if "emulator_path" in loaded:
            if isinstance(loaded["emulator_path"], str):
                self.data["emulator_path"] = loaded["emulator_path"]
            else:
                self.config_errors.append("emulator_path 类型无效")
        if "storage_path" in loaded:
            if isinstance(loaded["storage_path"], str):
                self.data["storage_path"] = loaded["storage_path"]
            else:
                self.config_errors.append("storage_path 类型无效")
        if "experience_limit" in loaded:
            value = loaded["experience_limit"]
            if isinstance(value, int) and int(0.1 * 1024 * 1024 * 1024) <= value <= 4096 * 1024 * 1024 * 1024:
                self.data["experience_limit"] = value
            else:
                self.config_errors.append("experience_limit 超出范围或类型无效")
        if "model_limit" in loaded:
            value = loaded["model_limit"]
            if isinstance(value, int) and 1 <= value <= 100000:
                self.data["model_limit"] = value
            else:
                self.config_errors.append("model_limit 超出范围或类型无效")

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

class DataStore:
    def __init__(self):
        self.root = None
        self.pool = None
        self.models = None
        self.screens = None
        self.conn = None
        self.lock = threading.RLock()

    def ensure(self, location):
        root = Path(location).expanduser().resolve()
        pool = root / "experience_pool"
        models = root / "models"
        screens = pool / "screens"
        with self.lock:
            if self.root == root and self.conn is not None:
                return
            self.close()
            root.mkdir(parents=True, exist_ok=True)
            pool.mkdir(parents=True, exist_ok=True)
            models.mkdir(parents=True, exist_ok=True)
            screens.mkdir(parents=True, exist_ok=True)
            self.root = root
            self.pool = pool
            self.models = models
            self.screens = screens
            self.conn = sqlite3.connect(str(pool / "records.sqlite3"), check_same_thread=False, timeout=30)
            self.conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                started REAL NOT NULL,
                ended REAL,
                frame_count INTEGER NOT NULL DEFAULT 0,
                mouse_count INTEGER NOT NULL DEFAULT 0,
                reason TEXT
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
                gray32x18 TEXT,
                edge_density REAL NOT NULL DEFAULT 0,
                color_histogram TEXT,
                dhash64 TEXT,
                score REAL NOT NULL,
                hunger REAL NOT NULL,
                reward REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                novelty REAL NOT NULL DEFAULT 0,
                action_result REAL NOT NULL DEFAULT 0,
                coverage REAL NOT NULL DEFAULT 0,
                model_refs INTEGER NOT NULL DEFAULT 0,
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
                outcome_valid INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_mouse_session ON mouse_events(session_id, created);
            CREATE INDEX IF NOT EXISTS idx_mouse_session_mono ON mouse_events(session_id, created_monotonic_ns);
            CREATE INDEX IF NOT EXISTS idx_frames_session_mono ON frames(session_id, created_monotonic_ns);
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
            """)
            self._migrate()
            self.conn.commit()
            self.recover_ingestions()
            self.recover_deletions()

    def _migrate(self):
        frame_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(frames)").fetchall()}
        additions = {
            "size_bytes": "INTEGER NOT NULL DEFAULT 0",
            "novelty": "REAL NOT NULL DEFAULT 0",
            "action_result": "REAL NOT NULL DEFAULT 0",
            "coverage": "REAL NOT NULL DEFAULT 0",
            "model_refs": "INTEGER NOT NULL DEFAULT 0",
            "last_used": "REAL NOT NULL DEFAULT 0",
            "dhash64": "TEXT",
            "bucket0": "INTEGER NOT NULL DEFAULT 0",
            "bucket1": "INTEGER NOT NULL DEFAULT 0",
            "bucket2": "INTEGER NOT NULL DEFAULT 0",
            "bucket3": "INTEGER NOT NULL DEFAULT 0",
            "state_cluster_id": "TEXT",
            "state_support_count": "INTEGER NOT NULL DEFAULT 1",
            "action_outcome_information": "REAL NOT NULL DEFAULT 0",
            "model_dependency_count": "INTEGER NOT NULL DEFAULT 0",
            "validation_last_used": "REAL NOT NULL DEFAULT 0",
            "created_monotonic_ns": "INTEGER NOT NULL DEFAULT 0",
            "capture_backend": "TEXT NOT NULL DEFAULT 'gdi'",
            "capture_elapsed_ms": "REAL NOT NULL DEFAULT 0",
            "capture_complete": "INTEGER NOT NULL DEFAULT 1",
            "brightness": "REAL NOT NULL DEFAULT 0",
            "variance": "REAL NOT NULL DEFAULT 0",
            "gray32x18": "TEXT",
            "edge_density": "REAL NOT NULL DEFAULT 0",
            "color_histogram": "TEXT",
            "asset_ref_count": "INTEGER NOT NULL DEFAULT 1",
            "score_candidate_count": "INTEGER NOT NULL DEFAULT 0",
            "score_top_k_distance": "REAL NOT NULL DEFAULT 64",
            "score_retrieval_fallback": "INTEGER NOT NULL DEFAULT 0",
            "capture_started_monotonic_ns": "INTEGER NOT NULL DEFAULT 0",
            "capture_finished_monotonic_ns": "INTEGER NOT NULL DEFAULT 0",
            "capture_started": "REAL NOT NULL DEFAULT 0",
            "capture_finished": "REAL NOT NULL DEFAULT 0"
        }
        for name, definition in additions.items():
            if name not in frame_columns:
                self.conn.execute(f"ALTER TABLE frames ADD COLUMN {name} {definition}")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_buckets ON frames(bucket0, bucket1, bucket2, bucket3)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS mouse_loss_events (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, created REAL NOT NULL, started REAL NOT NULL, ended REAL NOT NULL, lost_count INTEGER NOT NULL, rule TEXT NOT NULL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS deletion_journal (id TEXT PRIMARY KEY, object_type TEXT NOT NULL, object_id TEXT NOT NULL, path TEXT, stage TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, error TEXT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS pool_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
        self.conn.execute("INSERT OR IGNORE INTO pool_meta(key, value) VALUES ('total_asset_bytes', COALESCE((SELECT SUM(size_bytes) FROM frames), 0))")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_cluster ON frames(state_cluster_id, state_support_count)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_session_mono ON frames(session_id, created_monotonic_ns)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mouse_session_mono ON mouse_events(session_id, created_monotonic_ns)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS state_clusters (cluster_id TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS frame_lsh (key INTEGER NOT NULL, frame_id TEXT NOT NULL, PRIMARY KEY(key, frame_id))")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frame_lsh_key ON frame_lsh(key)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS ingestion_journal (id TEXT PRIMARY KEY, object_type TEXT NOT NULL, object_id TEXT NOT NULL, path TEXT, stage TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, error TEXT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS action_outcomes (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), mouse_event_id TEXT NOT NULL, before_frame_id TEXT NOT NULL, after_frame_id TEXT NOT NULL, action_time INTEGER NOT NULL, post_action_delay_ms REAL NOT NULL, score_delta REAL NOT NULL, reward_delta REAL NOT NULL, outcome_valid INTEGER NOT NULL)")
        action_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(action_outcomes)").fetchall()}
        if "action_id" not in action_columns:
            self.conn.execute("ALTER TABLE action_outcomes ADD COLUMN action_id TEXT")
        if "split_role" not in action_columns:
            self.conn.execute("ALTER TABLE action_outcomes ADD COLUMN split_role TEXT NOT NULL DEFAULT 'unknown'")
        self.conn.execute("UPDATE action_outcomes SET action_id=COALESCE(action_id, mouse_event_id) WHERE action_id IS NULL OR action_id=''")
        self.conn.execute("DELETE FROM action_outcomes WHERE rowid NOT IN (SELECT MAX(rowid) FROM action_outcomes GROUP BY action_id)")
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_action_outcomes_action_id ON action_outcomes(action_id)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS model_frame_refs (model_id TEXT NOT NULL, frame_id TEXT NOT NULL REFERENCES frames(id), role TEXT NOT NULL, PRIMARY KEY(model_id, frame_id, role))")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_model_frame_refs_frame ON model_frame_refs(frame_id)")
        missing_lsh = self.conn.execute("SELECT COUNT(*) FROM frames WHERE (dhash64 IS NOT NULL OR phash IS NOT NULL) AND id NOT IN (SELECT frame_id FROM frame_lsh)").fetchone()[0]
        if missing_lsh:
            for fid, dhash, phash in self.conn.execute("SELECT id, dhash64, phash FROM frames WHERE (dhash64 IS NOT NULL OR phash IS NOT NULL) AND id NOT IN (SELECT frame_id FROM frame_lsh)").fetchall():
                value = dhash or phash
                self.conn.executemany("INSERT OR IGNORE INTO frame_lsh(key, frame_id) VALUES (?, ?)", [(key, fid) for key in self._hash_buckets(value)])

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
            self.root = None
            self.pool = None
            self.models = None
            self.screens = None

    def create_session(self, mode):
        identifier = uuid.uuid4().hex
        now = time.time()
        with self.lock:
            self.conn.execute("INSERT INTO sessions(id, mode, started) VALUES (?, ?, ?)", (identifier, mode, now))
            self.conn.commit()
        return identifier

    def close_session(self, session_id, reason):
        with self.lock:
            if self.conn is None or not session_id:
                return
            self.conn.execute("UPDATE sessions SET ended=?, reason=? WHERE id=?", (time.time(), reason, session_id))
            self.conn.commit()

    def add_system_event(self, session_id, kind, payload):
        with self.lock:
            if self.conn is None:
                return
            self.conn.execute("INSERT INTO system_events(id, session_id, created, kind, payload) VALUES (?, ?, ?, ?, ?)", (uuid.uuid4().hex, session_id, time.time(), kind, json.dumps(payload, ensure_ascii=False)))
            self.conn.commit()

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

    def nearest_hashes(self, dhash, limit=8, strict=True):
        current = int(dhash, 16)
        parts = tuple((current >> shift) & 0xFFFF for shift in (48, 32, 16, 0))
        candidate = {}
        fallback = False
        max_candidates = 4096
        with self.lock:
            if self.conn is None:
                return {"hashes": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False}
            for radius in (0, 1):
                keys = []
                seen = set()
                for index, part in enumerate(parts):
                    for variant in set(self._bucket_variants(part, radius)):
                        key = (index << 16) | variant
                        if key not in seen:
                            seen.add(key)
                            keys.append(key)
                for start in range(0, len(keys), 400):
                    chunk = keys[start:start + 400]
                    if not chunk:
                        continue
                    placeholders = ",".join("?" for _ in chunk)
                    rows = self.conn.execute(f"""
                        SELECT frames.id, frames.dhash64, frames.phash
                        FROM frame_lsh JOIN frames ON frames.id=frame_lsh.frame_id
                        WHERE frame_lsh.key IN ({placeholders})
                        ORDER BY frames.created DESC
                        LIMIT {max_candidates}
                    """, chunk).fetchall()
                    for row in rows:
                        candidate[row[0]] = row
                        if len(candidate) >= max_candidates:
                            break
                    if len(candidate) >= max_candidates:
                        break
                if len(candidate) >= max_candidates:
                    break
            total = self.conn.execute("SELECT COUNT(*) FROM frames WHERE dhash64 IS NOT NULL OR phash IS NOT NULL").fetchone()[0]
            if strict and len(candidate) < limit and total <= 20000:
                fallback = True
                rows = self.conn.execute("SELECT id, dhash64, phash FROM frames WHERE dhash64 IS NOT NULL OR phash IS NOT NULL LIMIT 20000").fetchall()
                for row in rows:
                    candidate[row[0]] = row
        ranked = []
        for row in candidate.values():
            try:
                stored = row[1] or row[2]
                ranked.append((bit_count(current ^ int(stored, 16)), stored))
            except Exception:
                pass
        ranked.sort(key=lambda item: item[0])
        top = ranked[:limit]
        return {"hashes": [item[1] for item in top], "candidate_count": len(ranked), "top_k_distance": float(top[-1][0] if top else 64), "retrieval_fallback": fallback}

    def record_mouse_loss(self, session_id, started, ended, count, rule):
        with self.lock:
            if self.conn is None or not session_id or count <= 0:
                return
            self.conn.execute("INSERT INTO mouse_loss_events(id, session_id, created, started, ended, lost_count, rule) VALUES (?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, session_id, time.time(), started, ended, int(count), rule))
            self.conn.commit()

    def save_frame(self, session_id, image, phash, score, hunger, reward):
        identifier = uuid.uuid4().hex
        moment = float(image.get("capture_finished", time.time()))
        mono = int(image.get("capture_finished_monotonic_ns", time.monotonic_ns()))
        capture_started_mono = int(image.get("capture_started_monotonic_ns", mono))
        capture_started_wall = float(image.get("capture_started", moment))
        folder = self.screens / session_id
        folder.mkdir(parents=True, exist_ok=True)
        relative = Path("screens") / session_id / (identifier + ".png")
        final_path = self.pool / relative
        temporary = final_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            handle.write(image["png"])
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(final_path)
        size_bytes = final_path.stat().st_size
        dhash = image.get("dhash64") or phash
        buckets = self._hash_buckets(dhash)
        state_cluster_id = self.assign_state_cluster(dhash, image.get("width"), image.get("height"), image.get("gray32x18"), image.get("edge_density", 0.0))
        with self.lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self.conn.execute("INSERT INTO ingestion_journal(id, object_type, object_id, path, stage, created, updated) VALUES (?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, "frame", identifier, str(relative), "prepared", moment, moment))
                support = self.conn.execute("SELECT COUNT(*) + 1 FROM frames WHERE state_cluster_id=?", (state_cluster_id,)).fetchone()[0]
                self.conn.execute("INSERT INTO frames(id, session_id, created, created_monotonic_ns, capture_started_monotonic_ns, capture_finished_monotonic_ns, capture_started, capture_finished, screenshot_path, phash, dhash64, score, hunger, reward, width, height, size_bytes, novelty, action_result, coverage, model_refs, last_used, bucket0, bucket1, bucket2, bucket3, state_cluster_id, state_support_count, action_outcome_information, model_dependency_count, validation_last_used, asset_ref_count, capture_backend, capture_elapsed_ms, capture_complete, brightness, variance, gray32x18, edge_density, color_histogram, score_candidate_count, score_top_k_distance, score_retrieval_fallback) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (identifier, session_id, moment, mono, capture_started_mono, mono, capture_started_wall, moment, str(relative), phash, dhash, score, hunger, reward, image["width"], image["height"], size_bytes, score, reward, score, 0, moment, *buckets, state_cluster_id, support, abs(reward), 0, moment, 1, image.get("capture_backend", "gdi"), image.get("capture_elapsed_ms", 0.0), image.get("capture_complete", 1), image.get("brightness", 0.0), image.get("variance", 0.0), image.get("gray32x18"), image.get("edge_density", 0.0), image.get("color_histogram"), int(image.get("score_candidate_count", 0)), float(image.get("score_top_k_distance", 64.0)), int(image.get("score_retrieval_fallback", 0))))
                self.conn.execute("UPDATE ingestion_journal SET stage=?, updated=? WHERE object_id=?", ("complete", time.time(), identifier))
                self.conn.execute("INSERT OR IGNORE INTO state_clusters(cluster_id, count, updated_at) VALUES (?, 0, ?)", (state_cluster_id, moment))
                self.conn.execute("UPDATE state_clusters SET count=count+1, updated_at=? WHERE cluster_id=?", (moment, state_cluster_id))
                self.conn.executemany("INSERT OR IGNORE INTO frame_lsh(key, frame_id) VALUES (?, ?)", [(key, identifier) for key in self._hash_buckets(dhash)])
                self.conn.execute("UPDATE sessions SET frame_count=frame_count+1 WHERE id=?", (session_id,))
                self.conn.execute("INSERT OR REPLACE INTO pool_meta(key, value) VALUES ('total_asset_bytes', COALESCE((SELECT value FROM pool_meta WHERE key='total_asset_bytes'), 0) + ?)", (int(size_bytes),))
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise


    def save_mouse_batch(self, records):
        if not records:
            return
        values = []
        counts = {}
        for record in records:
            values.append((uuid.uuid4().hex, record["session_id"], record["created"], record.get("created_monotonic_ns", 0), record["source"], record["event_type"], record["button"], record["wheel"], record["x"], record["y"], record["relative_x"], record["relative_y"], record["dx"], record["dy"], record["direction"], record["speed"]))
            counts[record["session_id"]] = counts.get(record["session_id"], 0) + 1
        with self.lock:
            if self.conn is None:
                return
            self.conn.executemany("INSERT INTO mouse_events(id, session_id, created, created_monotonic_ns, source, event_type, button, wheel, x, y, relative_x, relative_y, dx, dy, direction, speed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
            self.conn.executemany("UPDATE sessions SET mouse_count=mouse_count+? WHERE id=?", [(value, key) for key, value in counts.items()])
            self.conn.commit()

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
        best = None
        for path, quality, trained_at in self.model_summaries():
            key = (quality, trained_at)
            if best is None or key > best[0]:
                try:
                    best = (key, json.loads(path.read_text(encoding="utf-8")))
                except Exception:
                    pass
        return best[1] if best else None

    def collect_training_data(self):
        with self.lock:
            sessions = [row[0] for row in self.conn.execute("SELECT id FROM sessions ORDER BY started DESC LIMIT 16").fetchall()]
            frames = {}
            mouse = {}
            for session_id in sessions:
                frame_rows = self.conn.execute("""
                    SELECT id, session_id, created_monotonic_ns, created, dhash64, phash, score, reward, width, height, gray32x18, edge_density, color_histogram, capture_started_monotonic_ns, capture_finished_monotonic_ns
                    FROM frames
                    WHERE session_id=? AND capture_complete=1 AND created_monotonic_ns>0
                    ORDER BY created_monotonic_ns ASC
                    LIMIT 6000
                """, (session_id,)).fetchall()
                mouse_rows = self.conn.execute("""
                    SELECT id, session_id, created_monotonic_ns, created, event_type, source, relative_x, relative_y, speed, dx, dy, direction, button, wheel
                    FROM mouse_events
                    WHERE session_id=? AND created_monotonic_ns>0
                    ORDER BY created_monotonic_ns ASC
                    LIMIT 18000
                """, (session_id,)).fetchall()
                if frame_rows:
                    frames[session_id] = frame_rows
                if mouse_rows:
                    mouse[session_id] = mouse_rows
            sampled = [row[0] for rows in frames.values() for row in rows[-256:]]
            if sampled:
                self.conn.executemany("UPDATE frames SET validation_last_used=?, model_dependency_count=MAX(model_dependency_count, 0) WHERE id=?", [(time.time(), item) for item in sampled])
            self.conn.commit()
        return frames, mouse


    def save_action_outcomes(self, outcomes):
        if not outcomes:
            return
        rows = [(item.get("action_id") or item["mouse_event_id"], uuid.uuid4().hex, item["session_id"], item["mouse_event_id"], item["before_frame_id"], item["after_frame_id"], item["action_time"], item["post_action_delay_ms"], item["score_delta"], item["reward_delta"], 1 if item.get("outcome_valid") else 0, item.get("split_role", "unknown")) for item in outcomes]
        with self.lock:
            self.conn.executemany("INSERT INTO action_outcomes(action_id, id, session_id, mouse_event_id, before_frame_id, after_frame_id, action_time, post_action_delay_ms, score_delta, reward_delta, outcome_valid, split_role) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(action_id) DO UPDATE SET session_id=excluded.session_id, mouse_event_id=excluded.mouse_event_id, before_frame_id=excluded.before_frame_id, after_frame_id=excluded.after_frame_id, action_time=excluded.action_time, post_action_delay_ms=excluded.post_action_delay_ms, score_delta=excluded.score_delta, reward_delta=excluded.reward_delta, outcome_valid=excluded.outcome_valid, split_role=excluded.split_role", rows)
            self.conn.executemany("UPDATE mouse_events SET before_frame_id=?, after_frame_id=?, action_time=?, post_action_delay_ms=?, score_delta=?, reward_delta=?, outcome_valid=? WHERE id=?", [(item["before_frame_id"], item["after_frame_id"], item["action_time"], item["post_action_delay_ms"], item["score_delta"], item["reward_delta"], 1 if item.get("outcome_valid") else 0, item["mouse_event_id"]) for item in outcomes])
            self.conn.commit()

    def save_model_frame_refs(self, model_id, outcomes, validation_outcomes):
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
            self.conn.executemany("INSERT OR IGNORE INTO model_frame_refs(model_id, frame_id, role) VALUES (?, ?, ?)", refs)
            self.conn.execute("UPDATE frames SET model_dependency_count=(SELECT COUNT(DISTINCT model_id) FROM model_frame_refs WHERE model_frame_refs.frame_id=frames.id), model_refs=(SELECT COUNT(DISTINCT model_id) FROM model_frame_refs WHERE model_frame_refs.frame_id=frames.id) WHERE id IN (SELECT frame_id FROM model_frame_refs WHERE model_id=?)", (model_id,))
            self.conn.commit()

    def pool_breakdown(self):
        result = {"frame_asset_bytes": 0, "database_bytes": 0, "transient_bytes": 0, "experience_total_bytes": 0}
        if self.pool is None:
            return result
        with self.lock:
            try:
                result["frame_asset_bytes"] = int(self.conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM frames").fetchone()[0] or 0)
            except Exception:
                pass
        try:
            for item in self.pool.rglob("*"):
                try:
                    if not item.is_file():
                        continue
                    size = item.stat().st_size
                    name = item.name.lower()
                    parts = {part.lower() for part in item.parts}
                    if name.startswith("records.sqlite3"):
                        result["database_bytes"] += size
                    elif "trash" in parts or name.endswith(".tmp"):
                        result["transient_bytes"] += size
                    result["experience_total_bytes"] += size
                except OSError:
                    pass
        except OSError:
            pass
        return result

    def pool_size(self):
        return int(self.pool_breakdown().get("experience_total_bytes", 0))

    def _reconcile_asset_bytes(self):
        with self.lock:
            total = self.conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM frames").fetchone()[0]
            self.conn.execute("INSERT OR REPLACE INTO pool_meta(key, value) VALUES ('total_asset_bytes', ?)", (int(total or 0),))
            self.conn.commit()

    def prune_models(self, maximum, cancelled=None, cooperative=None):
        summaries = self.model_summaries()
        if len(summaries) <= maximum:
            return 0
        target = max(0, int(math.floor(maximum * 0.5)))
        removed = 0
        for index, (path, quality, trained_at) in enumerate(sorted(summaries, key=lambda value: (value[1], value[2]))):
            if len(summaries) - removed <= target:
                break
            if cancelled is not None and cancelled():
                break
            if index % 12 == 0 and cooperative is not None and not cooperative():
                break
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
        return removed

    def _safe_screen_path(self, stored):
        candidate = (self.pool / stored).resolve()
        base = self.pool.resolve()
        if candidate == base or base not in candidate.parents:
            return None
        return candidate

    def _trash_path(self, journal_id):
        path = self.pool / "trash" / journal_id
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
            for identifier, stored, size_bytes in rows:
                journal_id = uuid.uuid4().hex
                self.conn.execute("INSERT INTO deletion_journal(id, object_type, object_id, path, stage, created, updated, error) VALUES (?, 'frame', ?, ?, 'pending', ?, ?, '')", (journal_id, identifier, stored, now, now))
                moved.append((journal_id, identifier, stored, int(size_bytes or 0)))
            self.conn.commit()
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
                self.conn.execute("INSERT OR REPLACE INTO pool_meta(key, value) VALUES ('total_asset_bytes', MAX(0, COALESCE((SELECT value FROM pool_meta WHERE key='total_asset_bytes'), 0) - ?))", (size_bytes,))
                self.conn.commit()
            trash.unlink(missing_ok=True)
            with self.lock:
                self.conn.execute("UPDATE deletion_journal SET stage='complete', updated=? WHERE id=?", (time.time(), journal_id))
                self.conn.commit()
        return len(moved)


    def recover_ingestions(self):
        if self.conn is None or self.pool is None:
            return
        if self.pool.exists():
            for item in self.pool.rglob("*.tmp"):
                try:
                    item.unlink(missing_ok=True)
                except OSError:
                    pass
        with self.lock:
            journals = self.conn.execute("SELECT object_id, path, stage FROM ingestion_journal WHERE object_type='frame' AND stage!='complete'").fetchall()
            referenced = {row[0] for row in self.conn.execute("SELECT screenshot_path FROM frames").fetchall()}
        for object_id, stored, stage in journals:
            path = self._safe_screen_path(stored) if stored else None
            with self.lock:
                exists = self.conn.execute("SELECT 1 FROM frames WHERE id=?", (object_id,)).fetchone() is not None
            if path is not None and path.exists() and not exists:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            with self.lock:
                self.conn.execute("UPDATE ingestion_journal SET stage='complete', updated=?, error='' WHERE object_id=?", (time.time(), object_id))
                self.conn.commit()
        if self.screens and self.screens.exists():
            for item in self.screens.rglob("*.png"):
                try:
                    rel = str(item.relative_to(self.pool))
                    if rel not in referenced:
                        item.unlink(missing_ok=True)
                except OSError:
                    pass
        self._reconcile_asset_bytes()

    def recover_deletions(self):
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
                        self.conn.commit()
                elif stage == "db_deleted":
                    self._trash_path(journal_id).unlink(missing_ok=True)
                    with self.lock:
                        self.conn.execute("UPDATE deletion_journal SET stage='complete', updated=?, error='' WHERE id=?", (time.time(), journal_id))
                        self.conn.commit()
            except OSError as error:
                with self.lock:
                    self.conn.execute("UPDATE deletion_journal SET updated=?, error=? WHERE id=?", (time.time(), str(error), journal_id))
                    self.conn.commit()
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
            for item in self.screens.rglob("*.png"):
                try:
                    if item.resolve() not in referenced:
                        item.unlink(missing_ok=True)
                except OSError:
                    pass
        with self.lock:
            self.conn.execute("DELETE FROM system_events WHERE session_id IS NOT NULL AND session_id NOT IN (SELECT id FROM sessions)")
            self.conn.commit()
            self._reconcile_asset_bytes()

    def validate_consistency(self):
        if self.conn is None or self.pool is None:
            return True, "存储未打开"
        missing = []
        referenced = set()
        with self.lock:
            rows = self.conn.execute("SELECT id, screenshot_path FROM frames").fetchall()
            bad_mouse = self.conn.execute("SELECT COUNT(*) FROM mouse_events WHERE session_id NOT IN (SELECT id FROM sessions)").fetchone()[0]
            bad_system = self.conn.execute("SELECT COUNT(*) FROM system_events WHERE session_id IS NOT NULL AND session_id NOT IN (SELECT id FROM sessions)").fetchone()[0]
            bad_lsh = self.conn.execute("SELECT COUNT(*) FROM frame_lsh WHERE frame_id NOT IN (SELECT id FROM frames)").fetchone()[0]
            bad_outcomes = self.conn.execute("SELECT COUNT(*) FROM action_outcomes WHERE before_frame_id NOT IN (SELECT id FROM frames) OR after_frame_id NOT IN (SELECT id FROM frames)").fetchone()[0]
            bad_clusters = self.conn.execute("SELECT COUNT(*) FROM state_clusters WHERE count!=(SELECT COUNT(*) FROM frames WHERE frames.state_cluster_id=state_clusters.cluster_id)").fetchone()[0]
            bad_model_refs = self.conn.execute("SELECT COUNT(*) FROM model_frame_refs WHERE frame_id NOT IN (SELECT id FROM frames)").fetchone()[0]
            meta_bytes = self.conn.execute("SELECT value FROM pool_meta WHERE key='total_asset_bytes'").fetchone()
            sum_bytes = self.conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM frames").fetchone()[0]
        for identifier, stored in rows:
            path = self._safe_screen_path(stored)
            if path is not None:
                referenced.add(path.resolve())
            if path is None or not path.exists():
                missing.append(identifier)
                if len(missing) >= 20:
                    break
        if missing:
            return False, "数据库引用了缺失截图 {} 条".format(len(missing))
        orphan = 0
        if self.screens and self.screens.exists():
            for item in self.screens.rglob("*.png"):
                try:
                    if item.resolve() not in referenced:
                        orphan += 1
                        if orphan >= 20:
                            break
                except OSError:
                    pass
        if orphan:
            return False, "经验池存在无引用截图 {} 条".format(orphan)
        if bad_lsh or bad_outcomes or bad_clusters or bad_model_refs:
            return False, "存在孤儿或失真索引：LSH {} 条，动作结果 {} 条，聚类 {} 条，模型引用 {} 条".format(bad_lsh, bad_outcomes, bad_clusters, bad_model_refs)
        if meta_bytes and int(meta_bytes[0]) != int(sum_bytes or 0):
            return False, "资产字节数统计不一致"
        if bad_mouse or bad_system:
            return False, "存在引用缺失会话的记录：鼠标 {} 条，系统 {} 条".format(bad_mouse, bad_system)
        return True, "一致"

    def _compact_database(self, cooperative=None):
        with self.lock:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                freelist = int(self.conn.execute("PRAGMA freelist_count").fetchone()[0] or 0)
                while freelist > 0:
                    if cooperative is not None and not cooperative():
                        break
                    pages = min(4096, freelist)
                    self.conn.execute("PRAGMA incremental_vacuum({})".format(pages))
                    freelist = int(self.conn.execute("PRAGMA freelist_count").fetchone()[0] or 0)
                self.conn.commit()
            except sqlite3.Error:
                pass

    def prune_experience(self, maximum, cancelled, progress, cooperative=None):
        maximum = max(1, int(maximum))
        self.recover_deletions()
        self._compact_database(cooperative)
        current = self.pool_size()
        if current <= maximum:
            return 0, current
        target = int(math.floor(maximum * 0.5))
        removed = 0
        def build_candidates(force):
            with self.lock:
                self.conn.execute("DROP TABLE IF EXISTS prune_candidates")
                self.conn.execute("CREATE TEMP TABLE prune_candidates(id TEXT PRIMARY KEY, screenshot_path TEXT NOT NULL, size_bytes INTEGER NOT NULL, rank INTEGER NOT NULL)")
                guard = "AND frames.model_dependency_count=0 AND frames.model_refs=0" if force else "AND frames.model_dependency_count=0 AND frames.model_refs=0 AND frames.asset_ref_count<=1 AND COALESCE(state_clusters.count, frames.state_support_count, 1)>1 AND frames.validation_last_used<?"
                params = [time.time()]
                if not force:
                    params.append(time.time() - 3600.0)
                query = """
                    SELECT frames.id, frames.screenshot_path, frames.size_bytes,
                           ((MAX(1, COALESCE(state_clusters.count, frames.state_support_count, 1)) - 1.0) / MAX(1, COALESCE(state_clusters.count, frames.state_support_count, 1))) AS redundancy,
                           MIN(1.0, MAX(0.0, (? - frames.created) / 2592000.0)) AS age,
                           MIN(1.0, MAX(0.0, (1.0 - frames.reward))) AS low_value,
                           MIN(1.0, MAX(0.0, (1.0 - frames.action_outcome_information))) AS low_information,
                           MIN(1.0, MAX(0.0, frames.size_bytes / 10485760.0)) AS size_gain,
                           ROW_NUMBER() OVER (PARTITION BY frames.state_cluster_id ORDER BY frames.reward DESC, frames.action_outcome_information DESC, frames.created DESC) AS cluster_rank
                    FROM frames LEFT JOIN state_clusters ON state_clusters.cluster_id=frames.state_cluster_id
                    WHERE 1=1 {guard}
                    ORDER BY ((redundancy * 0.30) + (low_value * 0.25) + (low_information * 0.20) + (age * 0.15) + (size_gain * 0.10) + CASE WHEN cluster_rank>1 THEN 0.25 ELSE 0 END) DESC
                    LIMIT 200000
                """.format(guard=guard)
                rows = self.conn.execute(query, params).fetchall()
                rank = 0
                for row in rows:
                    if force or row[3] > 0 or row[8] > 1:
                        rank += 1
                        self.conn.execute("INSERT OR IGNORE INTO prune_candidates(id, screenshot_path, size_bytes, rank) VALUES (?, ?, ?, ?)", (row[0], row[1], row[2], rank))
                self.conn.commit()
                return rank
        for force in (False, True):
            build_candidates(force)
            offset_rank = 0
            while current > target and not cancelled():
                if cooperative is not None and not cooperative():
                    time.sleep(0.5)
                    continue
                with self.lock:
                    rows = self.conn.execute("SELECT id, screenshot_path, size_bytes, rank FROM prune_candidates WHERE rank>? ORDER BY rank LIMIT 512", (offset_rank,)).fetchall()
                if not rows:
                    break
                offset_rank = int(rows[-1][3])
                before = current
                removed += self._delete_frame_batch([(row[0], row[1], row[2]) for row in rows])
                self._compact_database(cooperative)
                current = self.pool_size()
                if current <= target:
                    break
                if before <= current and force:
                    break
                progress(min(95.0, 56.0 + 39.0 * min(1.0, (maximum - current) / max(1, maximum - target))))
            if current <= target or cancelled():
                break
        self._compact_database(cooperative)
        remaining = self.pool_size()
        if remaining > target and not cancelled():
            self.add_system_event(None, "pool_prune_incomplete", {"remaining": remaining, "target": target, "breakdown": self.pool_breakdown()})
        progress(96.0)
        return removed, remaining


def filetime_value(value):
    return (value.dwHighDateTime << 32) | value.dwLowDateTime

def processes_for_name(name):
    wanted = name.lower()
    result = set()
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == INVALID_HANDLE_VALUE:
        return result
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        success = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while success:
            if entry.szExeFile.lower() == wanted:
                result.add(int(entry.th32ProcessID))
            success = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return result

def process_full_path(pid):
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
    point = POINT()
    if user32.GetCursorPos(ctypes.byref(point)):
        return (int(point.x), int(point.y))
    return (0, 0)

def send_ai_mouse(x, y, flags, absolute=True):
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
    event.mi = MOUSEINPUT(dx, dy, 0, flags, 0, AI_MOUSE_MARKER)
    return int(user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))) == 1

def ai_move_to(x, y):
    return send_ai_mouse(x, y, MOUSEEVENTF_MOVE, True)

def ai_left_click():
    down = send_ai_mouse(0, 0, MOUSEEVENTF_LEFTDOWN, False)
    up = send_ai_mouse(0, 0, MOUSEEVENTF_LEFTUP, False)
    return down and up

def ai_right_click():
    down = send_ai_mouse(0, 0, MOUSEEVENTF_RIGHTDOWN, False)
    up = send_ai_mouse(0, 0, MOUSEEVENTF_RIGHTUP, False)
    return down and up

def ai_wheel(delta, horizontal=False):
    event = INPUT()
    event.type = INPUT_MOUSE
    event.mi = MOUSEINPUT(0, 0, int(delta), MOUSEEVENTF_HWHEEL if horizontal else MOUSEEVENTF_WHEEL, 0, AI_MOUSE_MARKER)
    return int(user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))) == 1

def virtual_screen_rect():
    left = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    top = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    return (left, top, left + user32.GetSystemMetrics(SM_CXVIRTUALSCREEN), top + user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))

def screen_contains(rect):
    virtual = virtual_screen_rect()
    return rect[0] >= virtual[0] and rect[1] >= virtual[1] and rect[2] <= virtual[2] and rect[3] <= virtual[3]

def root_window(hwnd):
    return user32.GetAncestor(hwnd, GA_ROOT)

def rectangle_overlap(first, second):
    return first[0] < second[2] and second[0] < first[2] and first[1] < second[3] and second[1] < first[3]

def window_rectangle(hwnd):
    rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    if rect.right <= rect.left or rect.bottom <= rect.top:
        return None
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))

def client_unobscured(hwnd, rect):
    own_root = root_window(hwnd)
    if not own_root:
        return False
    points = ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), (rect[0] + 2, rect[1] + 2), (rect[2] - 2, rect[1] + 2), (rect[0] + 2, rect[3] - 2), (rect[2] - 2, rect[3] - 2)
    for x, y in points:
        hit = user32.GetWindowFromPoint(POINT(int(x), int(y)))
        if hit and root_window(hit) != own_root:
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hit, ctypes.byref(pid))
            client_unobscured.last_obstruction = {"title": window_title(hit), "pid": int(pid.value), "point": (int(x), int(y)), "rect": window_rectangle(hit)}
            return False
    above = user32.GetWindow(own_root, GW_HWNDPREV)
    checked = set()
    while above and above not in checked:
        checked.add(above)
        if root_window(above) == own_root:
            above = user32.GetWindow(above, GW_HWNDPREV)
            continue
        style = int(user32.GetWindowLongW(above, GWL_EXSTYLE))
        title = window_title(above)
        if "tooltip" in title.lower() or "工具提示" in title:
            above = user32.GetWindow(above, GW_HWNDPREV)
            continue
        if user32.IsWindowVisible(above) and not user32.IsIconic(above):
            candidate = window_rectangle(above)
            if candidate is not None and rectangle_overlap(candidate, rect):
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(above, ctypes.byref(pid))
                client_unobscured.last_obstruction = {"title": title, "pid": int(pid.value), "rect": candidate}
                return False
        above = user32.GetWindow(above, GW_HWNDPREV)
    return True

def valid_client(hwnd, require_cursor=True):
    if not hwnd or not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
        return None
    rect = client_rect(hwnd)
    if rect is None or rect[2] - rect[0] < 96 or rect[3] - rect[1] < 96:
        return None
    if not screen_contains(rect):
        return None
    if not client_unobscured(hwnd, rect):
        return None
    if require_cursor and not point_inside(cursor_position(), rect):
        return None
    return rect

def window_title(hwnd):
    length = max(0, int(user32.GetWindowTextLengthW(hwnd)))
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value

def normalized_windows_path(value):
    try:
        return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expandvars(os.path.expanduser(str(value))))))
    except Exception:
        return os.path.normcase(str(value))

def find_emulator_window(executable):
    selected = normalized_windows_path(executable)
    candidates = []

    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        full = normalized_windows_path(process_full_path(int(pid.value)))
        if full != selected:
            return True
        rect = client_rect(hwnd)
        if rect is None:
            return True
        area = (rect[2] - rect[0]) * (rect[3] - rect[1])
        candidates.append((area, int(pid.value), hwnd))
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][2]

def png_chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

def encode_png(width, height, rgb):
    rows = bytearray()
    row_size = width * 3
    for offset in range(0, len(rgb), row_size):
        rows.append(0)
        rows.extend(rgb[offset:offset + row_size])
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + png_chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + png_chunk(b"IEND", b"")

def capture_client(hwnd, max_width=640, max_height=360):
    capture_started_monotonic_ns = time.monotonic_ns()
    capture_started = time.time()
    started_capture = time.monotonic()
    local = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(local)):
        return None
    source_width = int(local.right - local.left)
    source_height = int(local.bottom - local.top)
    if source_width <= 0 or source_height <= 0:
        return None
    scale = min(1.0, max_width / source_width, max_height / source_height)
    width = max(1, int(source_width * scale))
    height = max(1, int(source_height * scale))
    source_dc = user32.GetDC(hwnd)
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
        if not gdi32.StretchBlt(memory_dc, 0, 0, width, height, source_dc, 0, 0, source_width, source_height, SRCCOPY):
            return None
        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB
        raw_size = width * height * 4
        raw = (ctypes.c_ubyte * raw_size)()
        received = gdi32.GetDIBits(memory_dc, bitmap, 0, height, ctypes.byref(raw), ctypes.byref(info), DIB_RGB_COLORS)
        if received != height:
            return None
        rgb = bytearray(width * height * 3)
        grayscale = [[0] * 9 for _ in range(8)]
        for y in range(height):
            raw_y = height - 1 - y
            source_row = raw_y * width * 4
            output_row = y * width * 3
            for x in range(width):
                source_index = source_row + x * 4
                output_index = output_row + x * 3
                blue = raw[source_index]
                green = raw[source_index + 1]
                red = raw[source_index + 2]
                rgb[output_index] = red
                rgb[output_index + 1] = green
                rgb[output_index + 2] = blue
        for y in range(8):
            sy = min(height - 1, int((y + 0.5) * height / 8))
            for x in range(9):
                sx = min(width - 1, int((x + 0.5) * width / 9))
                index = (sy * width + sx) * 3
                grayscale[y][x] = (rgb[index] * 299 + rgb[index + 1] * 587 + rgb[index + 2] * 114) // 1000
        value = 0
        for y in range(8):
            for x in range(8):
                value <<= 1
                if grayscale[y][x] > grayscale[y][x + 1]:
                    value |= 1
        lowfreq = [grayscale[y][x] for y in range(8) for x in range(8)]
        low_mean = sum(lowfreq) / max(1, len(lowfreq))
        perceptual = 0
        for item in lowfreq:
            perceptual <<= 1
            if item >= low_mean:
                perceptual |= 1
        mean = sum(rgb) / max(1, len(rgb))
        variance = sum((component - mean) ** 2 for component in rgb[::max(1, len(rgb)//4096)]) / max(1, len(rgb[::max(1, len(rgb)//4096)]))
        sample_gray = []
        for gy in range(18):
            sy = min(height - 1, int((gy + 0.5) * height / 18))
            for gx in range(32):
                sx = min(width - 1, int((gx + 0.5) * width / 32))
                index = (sy * width + sx) * 3
                sample_gray.append((rgb[index] * 299 + rgb[index + 1] * 587 + rgb[index + 2] * 114) // 1000)
        edge_total = 0
        edge_hits = 0
        for gy in range(18):
            for gx in range(31):
                delta = abs(sample_gray[gy * 32 + gx] - sample_gray[gy * 32 + gx + 1])
                edge_total += delta
                if delta > 24:
                    edge_hits += 1
        hist = [0] * 24
        step = max(1, len(rgb) // 12288)
        for index in range(0, len(rgb), 3 * step):
            hist[min(7, rgb[index] // 32)] += 1
            hist[8 + min(7, rgb[index + 1] // 32)] += 1
            hist[16 + min(7, rgb[index + 2] // 32)] += 1
        complete = 1 if mean >= 3.0 and variance >= 2.0 else 0
        capture_finished_monotonic_ns = time.monotonic_ns()
        capture_finished = time.time()
        return {"width": width, "height": height, "png": encode_png(width, height, rgb), "phash": f"{perceptual:016x}", "dhash64": f"{value:016x}", "capture_started_monotonic_ns": capture_started_monotonic_ns, "capture_finished_monotonic_ns": capture_finished_monotonic_ns, "capture_started": capture_started, "capture_finished": capture_finished, "capture_backend": "gdi", "capture_elapsed_ms": (capture_finished_monotonic_ns - capture_started_monotonic_ns) / 1000000.0, "capture_complete": complete, "brightness": mean, "variance": variance, "gray32x18": bytes(sample_gray).hex(), "edge_density": edge_hits / (18 * 31), "color_histogram": json.dumps(hist, separators=(",", ":"))}
    finally:
        if old_object and memory_dc:
            gdi32.SelectObject(memory_dc, old_object)
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if memory_dc:
            gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(hwnd, source_dc)

def bit_count(value):
    try:
        return value.bit_count()
    except AttributeError:
        return bin(value).count("1")

def frame_score(dhash, historical):
    details = historical if isinstance(historical, dict) else {"hashes": historical or [], "candidate_count": len(historical or []), "top_k_distance": 64.0, "retrieval_fallback": False}
    hashes = details.get("hashes", [])
    if not hashes:
        return 1.0, {"candidate_count": int(details.get("candidate_count", 0)), "top_k_distance": float(details.get("top_k_distance", 64.0)), "retrieval_fallback": bool(details.get("retrieval_fallback", False))}
    current = int(dhash, 16)
    weighted = []
    for index, previous in enumerate(hashes):
        try:
            distance = bit_count(current ^ int(previous, 16))
            similarity = 1.0 - distance / 64.0
            weight = 1.0 / (1.0 + index)
            weighted.append((similarity, weight, distance))
        except Exception:
            pass
    if not weighted:
        return 1.0, {"candidate_count": int(details.get("candidate_count", 0)), "top_k_distance": 64.0, "retrieval_fallback": bool(details.get("retrieval_fallback", False))}
    top_k_similarity = sum(value * weight for value, weight, _ in weighted) / max(1e-9, sum(weight for _, weight, _ in weighted))
    meta = {"candidate_count": int(details.get("candidate_count", len(weighted))), "top_k_distance": float(max(distance for _, _, distance in weighted)), "retrieval_fallback": bool(details.get("retrieval_fallback", False))}
    return max(0.0, min(1.0, 1.0 - top_k_similarity)), meta

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

    def start(self):
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
        if self.thread:
            self.thread.join(2.0)
        if not (self.thread and self.thread.is_alive()):
            self.thread = None
            self.thread_id = 0

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
                event_map = {
                    WM_MOUSEMOVE: ("move", ""),
                    WM_LBUTTONDOWN: ("button_down", "left"),
                    WM_LBUTTONUP: ("button_up", "left"),
                    WM_RBUTTONDOWN: ("button_down", "right"),
                    WM_RBUTTONUP: ("button_up", "right"),
                    WM_MOUSEWHEEL: ("wheel", "vertical"),
                    WM_MOUSEHWHEEL: ("wheel", "horizontal"),
                    WM_XBUTTONDOWN: ("button_down", "x"),
                    WM_XBUTTONUP: ("button_up", "x")
                }
                if int(wparam) in event_map:
                    event_type, button = event_map[int(wparam)]
                    wheel = 0
                    if int(wparam) in (WM_MOUSEWHEEL, WM_MOUSEHWHEEL):
                        wheel = ctypes.c_short((int(info.mouseData) >> 16) & 0xFFFF).value
                    source = "AI" if int(info.dwExtraInfo) == AI_MOUSE_MARKER else "用户"
                    self.sink(event_type, button, wheel, int(info.pt.x), int(info.pt.y), time.time(), time.monotonic_ns(), source)
        except Exception:
            pass
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

    def start(self):
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
        if self.thread:
            self.thread.join(2.0)
        if not (self.thread and self.thread.is_alive()):
            self.thread = None
            self.thread_id = 0

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
        except Exception:
            pass
        return user32.CallNextHookEx(self.handle, code, wparam, lparam)

class Controller:
    def __init__(self, settings, event_sink):
        self.settings = settings
        self.event_sink = event_sink
        self.store = DataStore()
        self.resources = ResourceMeter()
        self.lock = threading.RLock()
        self.state = "idle"
        self.epoch = 0
        self.cancel_event = threading.Event()
        self.target_hwnd = None
        self.target_rect = None
        self.session_id = None
        self.session_mode = None
        self.session_started = 0.0
        self.hunger_anchor = 0.0
        self.last_score = None
        self.frame_scores = []
        self.frame_count = 0
        self.mouse_count = 0
        self.last_mouse = None
        self.ai_step = 0
        self.ai_plan = []
        self.latest_frame_features = None
        self.action_limits = {}
        self.last_model_training = 0.0
        self.last_training_attempt = 0.0
        self.last_training_success = 0.0
        self.last_training_data_fingerprint = ""
        self.last_observation = 0.0
        self.capture_failures = 0
        self.mouse_queue = queue.Queue(maxsize=12000)
        self.raw_mouse_queue = queue.Queue(maxsize=16000)
        self.raw_critical_queue = queue.Queue(maxsize=2048)
        self.raw_mouse_stop = threading.Event()
        self.raw_mouse_drops = 0
        self.control_queue = queue.Queue(maxsize=64)
        self.stop_requested = threading.Event()
        self.last_move_kept = None
        self.writer_stop = threading.Event()
        self.writer_busy = threading.Event()
        self.worker_threads = []
        self.writer = threading.Thread(target=self._mouse_writer, name="MouseWriter")
        self.writer.start()
        self.raw_mouse_thread = threading.Thread(target=self._raw_mouse_loop, name="MouseParser")
        self.raw_mouse_thread.start()
        self.control_thread = threading.Thread(target=self._control_loop, name="SessionControl")
        self.control_thread.start()
        self.hook = MouseHook(self.enqueue_raw_mouse)
        self.keyboard_hook = KeyboardHook(self.on_control_signal)
        self.capture_threads = []
        self.loss_lock = threading.Lock()
        self.move_loss = {}

    def emit(self, kind, payload):
        self.event_sink(kind, payload)

    def post_state(self, detail=""):
        with self.lock:
            state = self.state
            sample = self.resources.sample()
        self.emit("state", {"state": state, "detail": detail, "cpu": sample["cpu"], "memory": sample["memory"]})

    def busy(self):
        with self.lock:
            return self.state != "idle"

    def current_state(self):
        with self.lock:
            return self.state

    def ensure_store(self):
        self.resources.set_storage_path(self.settings.data["storage_path"])
        self.store.ensure(self.settings.data["storage_path"])

    def on_control_signal(self, kind, reason, created=None, token=None):
        if kind in ("stop", "esc"):
            with self.lock:
                if self.stop_requested.is_set():
                    return
                self.stop_requested.set()
        try:
            self.control_queue.put_nowait({"kind": kind, "reason": reason, "created": created or time.time(), "token": token})
        except queue.Full:
            pass

    def _control_loop(self):
        while True:
            item = self.control_queue.get()
            if item is None:
                return
            if item.get("kind") == "stop":
                self._perform_stop(item.get("reason") or "停止请求", item.get("token"))
            elif item.get("kind") == "esc":
                self._perform_stop(item.get("reason") or "检测到 ESC 键", item.get("token"))

    def enqueue_raw_mouse(self, event_type, button, wheel, x, y, created, created_monotonic_ns, source):
        item = (event_type, button, wheel, x, y, created, created_monotonic_ns, source)
        target = self.raw_critical_queue if event_type != "move" or button or wheel else self.raw_mouse_queue
        try:
            target.put_nowait(item)
        except queue.Full:
            self.raw_mouse_drops += 1
            if target is self.raw_critical_queue or self.raw_mouse_drops >= 64:
                self.on_control_signal("stop", "鼠标原始事件队列过载")

    def _raw_mouse_loop(self):
        while not self.raw_mouse_stop.is_set() or not self.raw_critical_queue.empty() or not self.raw_mouse_queue.empty():
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
            try:
                self.on_mouse(*item)
            except Exception as error:
                self.on_control_signal("stop", "鼠标事件解析失败:" + str(error))

    def _mouse_writer(self):
        pending = []
        last_write = time.monotonic()
        while not self.writer_stop.is_set() or not self.mouse_queue.empty() or pending:
            self.resources.update_queue(self.mouse_queue.qsize(), max(0.0, time.time() - float(pending[0].get("created", time.time()))) if pending else 0.0)
            try:
                pending.append(self.mouse_queue.get(timeout=0.25))
                self.writer_busy.set()
            except queue.Empty:
                pass
            if pending and (len(pending) >= 80 or time.monotonic() - last_write >= 0.6 or self.writer_stop.is_set()):
                self.writer_busy.set()
                try:
                    self.store.save_mouse_batch(pending)
                except Exception as error:
                    sessions = {}
                    for record in pending:
                        item = sessions.setdefault(record["session_id"], [record["created"], record["created"], 0])
                        item[0] = min(item[0], record["created"])
                        item[1] = max(item[1], record["created"])
                        item[2] += 1
                    for sid, values in sessions.items():
                        try:
                            self.store.record_mouse_loss(sid, values[0], values[1], values[2], "批量写入失败:" + str(error))
                        except Exception:
                            pass
                finally:
                    pending = []
                    last_write = time.monotonic()
                    self.writer_busy.clear()
            elif not pending:
                self.writer_busy.clear()

    def flush_mouse_records(self, timeout=3.0):
        deadline = time.monotonic() + max(0.1, timeout)
        while time.monotonic() < deadline:
            if self.mouse_queue.empty() and not self.writer_busy.is_set():
                return True
            time.sleep(0.04)
        return self.mouse_queue.empty() and not self.writer_busy.is_set()

    def on_mouse(self, event_type, button, wheel, x, y, created, created_monotonic_ns, source):
        with self.lock:
            if self.state not in ("learning", "training") or not self.session_id or not self.target_rect:
                return
            session_id = self.session_id
            rect = self.target_rect
            outside = not point_inside((x, y), rect)
            previous = self.last_mouse
            critical = event_type != "move" or button or wheel
            if self.state == "training" and source == "用户" and (critical or (previous is not None and math.hypot(x - previous[0], y - previous[1]) >= 12)):
                self.on_control_signal("stop", "训练模式检测到真实用户鼠标操作，AI 已停止")
            if not critical:
                last_kept = self.last_move_kept
                if last_kept is not None and (created_monotonic_ns - last_kept[2]) < 10_000_000 and abs(x - last_kept[0]) < 3 and abs(y - last_kept[1]) < 3:
                    self.last_mouse = (x, y, created_monotonic_ns)
                    return
                self.last_move_kept = (x, y, created_monotonic_ns)
            self.last_mouse = (x, y, created_monotonic_ns)
            self.mouse_count += 1
        dx = 0.0
        dy = 0.0
        direction = 0.0
        speed = 0.0
        if previous is not None:
            dt = max(0.000001, (created_monotonic_ns - previous[2]) / 1_000_000_000.0)
            dx = float(x - previous[0])
            dy = float(y - previous[1])
            direction = math.atan2(dy, dx) if dx or dy else 0.0
            speed = math.hypot(dx, dy) / dt
        width = max(1, rect[2] - rect[0])
        height = max(1, rect[3] - rect[1])
        record = {
            "session_id": session_id,
            "created": created,
            "created_monotonic_ns": int(created_monotonic_ns),
            "source": source,
            "event_type": event_type,
            "button": button,
            "wheel": wheel,
            "x": x,
            "y": y,
            "relative_x": (x - rect[0]) / width,
            "relative_y": (y - rect[1]) / height,
            "dx": dx,
            "dy": dy,
            "direction": direction,
            "speed": speed
        }
        try:
            self.mouse_queue.put_nowait(record)
        except queue.Full:
            if critical:
                self.on_control_signal("stop", "鼠标高优先级事件队列过载")
            with self.loss_lock:
                item = self.move_loss.get(session_id)
                if item is None:
                    self.move_loss[session_id] = [created, created, 1]
                else:
                    item[1] = created
                    item[2] += 1
        if outside:
            self.on_control_signal("stop", "鼠标已离开雷电模拟器客户区")

    def _is_current(self, token, states=None):
        with self.lock:
            if token != self.epoch or self.cancel_event.is_set():
                return False
            return states is None or self.state in states

    def _find_valid_target(self, cursor_required=False):
        hwnd = find_emulator_window(self.settings.data["emulator_path"])
        if not hwnd:
            return None, None, "未检测到已启动的雷电模拟器窗口。"
        rect = valid_client(hwnd, cursor_required)
        if rect is None:
            return None, None, "雷电模拟器客户区不可见、被遮挡、最小化或未完全位于屏幕范围内。"
        return hwnd, rect, ""

    def _place_cursor_before_entry(self, hwnd, rect):
        current = cursor_position()
        if not point_inside(current, rect):
            x = (rect[0] + rect[2]) // 2
            y = (rect[1] + rect[3]) // 2
            if not user32.SetCursorPos(x, y):
                return False
            time.sleep(0.03)
        return valid_client(hwnd, True) is not None

    def start_session(self, mode, automatic=False):
        with self.lock:
            permitted = self.state == "idle" or (automatic and self.state == "sleep")
            if not permitted:
                if not automatic:
                    self.emit("notice", "当前不是空闲状态。")
                return False
        if not self.hook.start():
            self.emit("notice", self.hook.error or "鼠标钩子未启动，禁止进入模式。")
            return False
        if not self.keyboard_hook.start():
            self.emit("notice", self.keyboard_hook.error or "键盘钩子未启动，禁止进入模式。")
            self.hook.stop()
            return False
        try:
            self.ensure_store()
        except Exception as error:
            self.emit("notice", "无法创建存储路径：" + str(error))
            self.hook.stop()
            self.keyboard_hook.stop()
            return False
        hwnd, rect, reason = self._find_valid_target(False)
        if hwnd is None:
            self.emit("notice", reason)
            self.hook.stop()
            self.keyboard_hook.stop()
            return False
        if not self._place_cursor_before_entry(hwnd, rect):
            self.emit("notice", "进入模式前无法确认鼠标与雷电模拟器客户区状态。")
            self.hook.stop()
            self.keyboard_hook.stop()
            return False
        rect = valid_client(hwnd, True)
        if rect is None:
            self.emit("notice", "雷电模拟器客户区状态异常。")
            self.hook.stop()
            self.keyboard_hook.stop()
            return False
        try:
            session_id = self.store.create_session(mode)
        except Exception as error:
            self.emit("notice", "无法创建会话记录：" + str(error))
            self.hook.stop()
            self.keyboard_hook.stop()
            return False
        with self.lock:
            self.epoch += 1
            token = self.epoch
            self.cancel_event = threading.Event()
            self.state = mode
            self.target_hwnd = hwnd
            self.target_rect = rect
            self.session_id = session_id
            self.session_mode = mode
            self.session_started = time.monotonic()
            self.hunger_anchor = self.session_started
            self.last_observation = self.session_started
            self.capture_failures = 0
            self.last_score = None
            self.frame_scores = []
            self.frame_count = 0
            self.mouse_count = 0
            self.last_mouse = None
            self.ai_step = 0
            model = self.store.best_model() if mode == "training" else None
            plan = model.get("q_actions", model.get("hotspots", [])) if isinstance(model, dict) else []
            self.ai_plan = [item for item in plan if isinstance(item, dict)] if isinstance(plan, list) else []
        with self.lock:
            self.action_limits = {}
            self.last_move_kept = None
            self.stop_requested.clear()
        self.store.add_system_event(session_id, "mode_enter", {"mode": mode, "automatic": automatic, "time": time.time(), "client_rect": rect, "resource": self.resources.sample()})
        self.post_state("已进入" + ("学习模式" if mode == "learning" else "训练模式"))
        threads = [threading.Thread(target=self._capture_loop, args=(token,), name="CaptureLoop"), threading.Thread(target=self._monitor_loop, args=(token,), name="SessionMonitor")]
        if mode == "training":
            threads.append(threading.Thread(target=self._ai_loop, args=(token,), name="AIControl"))
        with self.lock:
            self.capture_threads = threads
            self.worker_threads = [thread for thread in self.worker_threads if thread.is_alive()] + threads
        for thread in threads:
            thread.start()
        return True

    def _capture_loop(self, token):
        while self._is_current(token, ("learning", "training")):
            if not self.resources.allow_capture():
                sample = self.resources.sample()
                self.emit("state", {"state": self.current_state(), "detail": "系统资源繁忙，已自动限速记录", "cpu": sample["cpu"], "memory": sample["memory"]})
                time.sleep(max(2.0, self.resources.interval()))
                continue
            interval = self.resources.interval()
            with self.lock:
                hwnd = self.target_hwnd
                session_id = self.session_id
                mode = self.session_mode
            rect = valid_client(hwnd, True) if hwnd else None
            if rect is None:
                obstruction = getattr(client_unobscured, "last_obstruction", None)
                if obstruction and self.session_id:
                    self.store.add_system_event(self.session_id, "client_obstruction", obstruction)
                self.request_idle("雷电模拟器客户区异常或鼠标已离开客户区", token)
                return
            observation_due = False
            now_observation = time.monotonic()
            with self.lock:
                self.target_rect = rect
                if now_observation - self.last_observation >= 5.0:
                    self.last_observation = now_observation
                    observation_due = True
            if session_id and observation_due:
                try:
                    self.store.add_system_event(session_id, "client_observation", {"mode": mode, "client_rect": rect, "cursor": cursor_position(), "resource": self.resources.sample()})
                except Exception:
                    pass
            if session_id:
                image = capture_client(hwnd)
                if image is not None:
                    self.resources.update_capture_metrics(image.get("capture_elapsed_ms"), False)
                if not self._is_current(token, ("learning", "training")):
                    return
                if image is None:
                    self.resources.update_capture_metrics(None, True)
                    with self.lock:
                        self.capture_failures += 1
                        failures = self.capture_failures
                    if failures >= 12:
                        self.request_idle("连续无法记录雷电模拟器画面", token)
                        return
                    sample = self.resources.sample()
                    self.emit("state", {"state": self.current_state(), "detail": "暂时无法记录画面，正在重试", "cpu": sample["cpu"], "memory": sample["memory"]})
                else:
                    try:
                        if not image.get("capture_complete", 1):
                            with self.lock:
                                self.capture_failures += 1
                            time.sleep(max(interval, 1.5))
                            continue
                        historical = self.store.nearest_hashes(image["dhash64"], 8)
                        if not self._is_current(token, ("learning", "training")):
                            return
                        score, score_meta = frame_score(image["dhash64"], historical)
                        image["score_candidate_count"] = score_meta["candidate_count"]
                        image["score_top_k_distance"] = score_meta["top_k_distance"]
                        image["score_retrieval_fallback"] = 1 if score_meta["retrieval_fallback"] else 0
                        now = time.monotonic()
                        with self.lock:
                            hunger = 1e-9 + max(0.0, now - self.hunger_anchor) * 0.00004
                            reset_hunger = self.last_score is not None and score > self.last_score
                            if reset_hunger:
                                hunger = 1e-9
                            reward = score - hunger
                        if not self._is_current(token, ("learning", "training")):
                            return
                        sqlite_start = time.monotonic()
                        self.store.save_frame(session_id, image, image["phash"], score, hunger, reward)
                        self.resources.update_sqlite_latency((time.monotonic() - sqlite_start) * 1000.0)
                        with self.lock:
                            if reset_hunger:
                                self.hunger_anchor = now
                            self.last_score = score
                            self.frame_scores.append(score)
                            self.frame_scores = self.frame_scores[-120:]
                            self.frame_count += 1
                            self.latest_frame_features = {"seq": self.frame_count, "state_hash": image.get("dhash64"), "gray32x18": image.get("gray32x18"), "edge_density": image.get("edge_density", 0.0), "color_histogram": image.get("color_histogram"), "score": score, "hunger": hunger}
                            self.capture_failures = 0
                    except Exception as error:
                        with self.lock:
                            self.capture_failures += 1
                            failures = self.capture_failures
                        if failures >= 12:
                            self.request_idle("连续无法写入经验池：" + str(error), token)
                            return
                        sample = self.resources.sample()
                        self.emit("state", {"state": self.current_state(), "detail": "记录已限速：" + str(error), "cpu": sample["cpu"], "memory": sample["memory"]})
            time.sleep(interval)

    def _monitor_loop(self, token):
        while self._is_current(token, ("learning", "training")):
            if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                self.request_idle("检测到 ESC 键", token)
                return
            with self.lock:
                hwnd = self.target_hwnd
            rect = valid_client(hwnd, True) if hwnd else None
            if rect is None:
                self.request_idle("雷电模拟器客户区异常或鼠标已离开客户区", token)
                return
            with self.lock:
                self.target_rect = rect
            if self._should_sleep(token):
                self._begin_auto_sleep(token)
                return
            time.sleep(0.08)

    def sleep_decision_model(self, features):
        scores_mean = features.get("score_mean", 0.0)
        scores_var = features.get("score_variance", 0.0)
        trend = features.get("score_trend", 0.0)
        action_lcb = features.get("action_lcb_mean", 0.0)
        uncertainty = features.get("action_uncertainty", 1.0)
        coverage = features.get("sample_coverage", 0.0)
        pressure = features.get("queue_pressure", 0.0) + features.get("resource_pressure", 0.0)
        gain = max(0.0, uncertainty * 0.35 + (1.0 - coverage) * 0.25 - max(0.0, trend) * 0.6 - action_lcb * 0.2)
        probability = max(0.0, min(1.0, 0.15 + gain + pressure * 0.4 + (0.15 if scores_var < 0.0005 and trend <= 0 else 0.0)))
        return {"sleep_probability": probability, "expected_sleep_gain": gain}

    def _should_sleep(self, token):
        with self.lock:
            if token != self.epoch or self.state != "training":
                return False
            elapsed = time.monotonic() - self.session_started
            scores = list(self.frame_scores[-40:])
            mouse_count = self.mouse_count
            plan = list(self.ai_plan)
            queue_len = self.mouse_queue.qsize()
            hunger_speed = 0.00004
        sample = self.resources.sample()
        try:
            if self.store.pool_size() >= int(self.settings.data["experience_limit"] * 0.95):
                return True
        except Exception:
            pass
        if sample.get("disk_free", 1) < 1024 * 1024 * 1024 or sample.get("avail_memory", 1) < 384 * 1024 * 1024 or queue_len > 10000:
            return True
        if not plan or mouse_count < 30 or time.time() - self.last_model_training < 180.0:
            return False
        if elapsed < 60.0 or len(scores) < 12:
            return False
        mean = sum(scores) / len(scores)
        variance = sum((value - mean) ** 2 for value in scores) / len(scores)
        trend = scores[-1] - scores[0]
        lcbs = [float(a.get("confidence_lower_bound", 0.0)) for a in plan if isinstance(a, dict)]
        uncs = [float(a.get("uncertainty", 1.0)) for a in plan if isinstance(a, dict)]
        features = {"score_mean": mean, "score_variance": variance, "score_trend": trend, "hunger_growth_speed": hunger_speed, "action_lcb_mean": sum(lcbs) / max(1, len(lcbs)), "action_uncertainty": sum(uncs) / max(1, len(uncs)), "sample_coverage": min(1.0, len(plan) / 128.0), "mouse_queue_length": queue_len, "queue_pressure": min(1.0, queue_len / 12000.0), "resource_pressure": 1.0 if self.resources.critical() else 0.0, "cpu_margin": max(0.0, 100.0 - sample.get("cpu", 0.0)), "memory_margin": max(0.0, 100.0 - sample.get("memory", 0.0)), "disk_free": sample.get("disk_free", 0)}
        decision = self.sleep_decision_model(features)
        return decision["sleep_probability"] >= 0.62 and decision["expected_sleep_gain"] > 0.0

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

    def _state_distance(self, current, item):
        if not current:
            return 1.0
        distances = []
        try:
            if current.get("state_hash") and item.get("state_hash"):
                distances.append(bit_count(int(current["state_hash"], 16) ^ int(item["state_hash"], 16)) / 64.0)
        except Exception:
            pass
        try:
            a = bytes.fromhex(current.get("gray32x18") or "")
            b = bytes.fromhex(item.get("gray32x18") or "")
            if a and b and len(a) == len(b):
                distances.append(sum(abs(x - y) for x, y in zip(a, b)) / (255.0 * len(a)))
        except Exception:
            pass
        try:
            distances.append(abs(float(current.get("edge_density", 0.0)) - float(item.get("edge_density", 0.0))))
        except Exception:
            pass
        return sum(distances) / len(distances) if distances else 1.0

    def _ai_target(self, rect):
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        with self.lock:
            step = self.ai_step
            self.ai_step += 1
            plan = list(self.ai_plan)
            current = dict(self.latest_frame_features) if self.latest_frame_features else None
            limits = dict(self.action_limits)
        gates = {"左键": (5, 0.0, 2.0, 20), "右键": (8, 0.02, 3.0, 12), "滚轮": (6, 0.0, 1.5, 24), "水平滚轮": (8, 0.02, 2.0, 12)}
        viable = []
        for item in plan:
            if not isinstance(item, dict):
                continue
            try:
                action_type = str(item.get("action_type", "移动"))
                samples = int(item.get("samples", 0))
                lcb = float(item.get("confidence_lower_bound", item.get("confidence", -1.0)))
                uncertainty = float(item.get("uncertainty", 1.0))
                distance = self._state_distance(current, item)
                if distance > float(item.get("state_similarity_threshold", 0.32)):
                    continue
                if action_type != "移动":
                    min_samples, min_lcb, cooldown, per_minute = gates.get(action_type, (999999, 1.0, 999.0, 0))
                    stat = limits.get(action_type, {"last": 0.0, "times": []})
                    recent = [t for t in stat.get("times", []) if time.monotonic() - t < 60.0]
                    if samples < min_samples or lcb < min_lcb or uncertainty > 0.25 or time.monotonic() - stat.get("last", 0.0) < cooldown or len(recent) >= per_minute:
                        action_type = "移动"
                viable.append((lcb - distance, samples, action_type, distance, item))
            except (TypeError, ValueError):
                pass
        viable.sort(reverse=True, key=lambda value: (value[0], value[1]))
        item_tuple = viable[step % len(viable)] if viable else None
        if item_tuple is not None:
            _, _, action_type, distance, item = item_tuple
            x_ratio = min(0.95, max(0.05, float(item.get("x", 0.5))))
            y_ratio = min(0.95, max(0.05, float(item.get("y", 0.5))))
            confidence = float(item.get("confidence_lower_bound", 0.0))
            wheel_delta = max(-1200, min(1200, int(item.get("wheel_delta", 120 if action_type in ("滚轮", "水平滚轮") else 0))))
        else:
            x_ratio = 0.08 + 0.84 * ((step * 0.618033988749895) % 1.0)
            y_ratio = 0.08 + 0.84 * ((step * 0.414213562373095) % 1.0)
            action_type = "移动"
            confidence = 0.0
            wheel_delta = 0
            distance = 1.0
        x = rect[0] + int(width * x_ratio)
        y = rect[1] + int(height * y_ratio)
        return {"x": x, "y": y, "action_type": action_type, "wheel_delta": wheel_delta, "confidence": confidence, "state_match_distance": distance}

    def _ai_loop(self, token):
        while self._is_current(token, ("training",)):
            if not self.resources.allow_compute():
                time.sleep(max(1.5, self.resources.interval()))
                continue
            with self.lock:
                hwnd = self.target_hwnd
            rect = valid_client(hwnd, True) if hwnd else None
            if rect is None:
                self.request_idle("雷电模拟器客户区异常或鼠标已离开客户区", token)
                return
            target = self._ai_target(rect)
            x, y = target["x"], target["y"]
            if not point_inside((x, y), rect) or not ai_move_to(x, y):
                self.request_idle("AI 鼠标操作无法确认位于雷电模拟器客户区内", token)
                return
            time.sleep(0.05)
            rect = valid_client(hwnd, True) if hwnd else None
            if rect is None or not point_inside(cursor_position(), rect):
                self.request_idle("雷电模拟器客户区异常或鼠标已离开客户区", token)
                return
            action_type = target.get("action_type", "移动")
            ok = True
            if action_type == "左键" and self.resources.allow_compute():
                ok = ai_left_click()
            elif action_type == "右键" and self.resources.allow_compute():
                ok = ai_right_click()
            elif action_type == "滚轮" and self.resources.allow_compute():
                ok = ai_wheel(target.get("wheel_delta", 120), False)
            elif action_type == "水平滚轮" and self.resources.allow_compute():
                ok = ai_wheel(target.get("wheel_delta", 120), True)
            if action_type != "移动" and ok:
                with self.lock:
                    stat = self.action_limits.setdefault(action_type, {"last": 0.0, "times": []})
                    now_limit = time.monotonic()
                    stat["last"] = now_limit
                    stat["times"] = [t for t in stat.get("times", []) if now_limit - t < 60.0] + [now_limit]
            if not ok:
                self.request_idle("AI 鼠标动作无法执行" + str(action_type), token)
                return
            time.sleep(max(0.7, self.resources.interval()))

    def _write_barrier(self, session_id, reason):
        deadline = time.monotonic() + 5.0
        for thread in list(self.capture_threads):
            if thread is threading.current_thread():
                continue
            remaining = max(0.05, deadline - time.monotonic())
            thread.join(min(1.0, remaining))
        self.flush_mouse_records(5.0)
        with self.loss_lock:
            losses = self.move_loss.pop(session_id, None) if session_id else None
        if losses:
            self.store.record_mouse_loss(session_id, losses[0], losses[1], losses[2], "模式切换写入屏障刷新普通移动降采样统计")
        ok, detail = self.store.validate_consistency()
        if not ok:
            raise RuntimeError("写入屏障失败：" + detail)
        self.store.add_system_event(session_id, "write_barrier", {"reason": reason, "time": time.time(), "consistency": detail})

    def _close_active_session(self, reason, barrier=True):
        with self.lock:
            session_id = self.session_id
            self.session_id = None
            self.session_mode = None
            self.target_hwnd = None
            self.target_rect = None
        if session_id:
            try:
                if barrier:
                    self._write_barrier(session_id, reason)
                self.store.add_system_event(session_id, "mode_exit", {"reason": reason, "time": time.time()})
                self.store.close_session(session_id, reason)
            except Exception as error:
                self.emit("notice", str(error))

    def _perform_stop(self, reason, token=None):
        with self.lock:
            if token is not None and token != self.epoch:
                return False
            if self.state == "idle":
                return False
            previous = self.state
            self.stop_requested.set()
            self.cancel_event.set()
            self.epoch += 1
            if previous in ("learning", "training"):
                self.state = "stopping"
            else:
                self.state = "idle"
        if previous in ("learning", "training"):
            self.hook.stop()
            self.keyboard_hook.stop()
            self.post_state("正在停止并执行统一写入屏障")
            self._close_active_session(reason, barrier=True)
        elif previous == "sleep":
            self.keyboard_hook.stop()
        with self.lock:
            if self.state == "stopping":
                self.state = "idle"
            self.stop_requested.clear()
        self.emit("progress", 0.0)
        self.post_state(reason if previous != "sleep" else "睡眠模式已中止：" + reason)
        return True

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
            self.emit("notice", "无法创建存储路径：" + str(error))
            return False
        if not self.keyboard_hook.start():
            self.emit("notice", self.keyboard_hook.error or "键盘钩子未启动，禁止进入睡眠模式。")
            return False
        with self.lock:
            self.epoch += 1
            token = self.epoch
            self.cancel_event = threading.Event()
            self.state = "sleep"
        self.post_state("已进入睡眠模式")
        threads = [threading.Thread(target=self._sleep_monitor, args=(token,), name="SleepMonitor"), threading.Thread(target=self._sleep_worker, args=(token, False), name="SleepWorker")]
        with self.lock:
            self.worker_threads = [thread for thread in self.worker_threads if thread.is_alive()] + threads
        for thread in threads:
            thread.start()
        return True

    def _begin_auto_sleep(self, token):
        with self.lock:
            if token != self.epoch or self.state != "training":
                return
            self.epoch += 1
            sleep_token = self.epoch
            self.cancel_event = threading.Event()
            self.state = "sleep"
        self.hook.stop()
        self._close_active_session("AI 判断进入睡眠模式", barrier=True)
        self.post_state("AI 判断当前值得进入睡眠模式")
        threads = [threading.Thread(target=self._sleep_monitor, args=(sleep_token,), name="AutoSleepMonitor"), threading.Thread(target=self._sleep_worker, args=(sleep_token, True), name="AutoSleepWorker")]
        with self.lock:
            self.worker_threads = [thread for thread in self.worker_threads if thread.is_alive()] + threads
        for thread in threads:
            thread.start()

    def _sleep_monitor(self, token):
        while self._is_current(token, ("sleep",)):
            if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                self.request_idle("检测到 ESC 键", token)
                return
            time.sleep(0.08)

    def _cancelled(self, token):
        return not self._is_current(token, ("sleep",))

    def _wait_resource(self, token, purpose="training"):
        allowed = self.resources.allow_maintenance if purpose == "maintenance" else self.resources.allow_training
        while not self._cancelled(token) and not allowed():
            sample = self.resources.sample()
            self.emit("state", {"state": "sleep", "detail": "系统资源繁忙，睡眠任务已暂缓", "cpu": sample["cpu"], "memory": sample["memory"]})
            time.sleep(1.2)
        return not self._cancelled(token)

    def _semantic_actions(self, mouse_rows):
        actions = []
        down = {}
        wheel_bucket = None
        last_move = None
        for row in mouse_rows:
            mid, sid, created_ns, created, event_type, source, rx, ry, speed, dx, dy, direction, button, wheel = row
            if source not in ("用户", "AI") or rx is None or ry is None:
                continue
            ns = int(created_ns)
            if event_type == "button_down" and button in ("left", "right"):
                down[button] = row
            elif event_type == "button_up" and button in down:
                d = down.pop(button)
                duration = ns - int(d[2])
                distance = math.hypot(float(rx) - float(d[6]), float(ry) - float(d[7]))
                if 0 <= duration <= 1_500_000_000 and distance <= 0.08:
                    action_id = "click:{}:{}:{}".format(sid, d[0], mid)
                    actions.append({"action_id": action_id, "mouse_event_id": mid, "session_id": sid, "action_time": ns, "source": source, "rx": float(rx), "ry": float(ry), "action_type": "左键" if button == "left" else "右键"})
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
                else:
                    if wheel_bucket:
                        wheel_bucket["action_type"] = "水平滚轮" if wheel_bucket["wheel_axis"] == "horizontal" else "滚轮"
                        wheel_bucket["wheel_delta"] = max(-1200, min(1200, int(wheel_bucket["signed_delta"])))
                        actions.append(wheel_bucket)
                    wheel_bucket = {"action_id": "wheel:{}:{}".format(sid, mid), "mouse_event_id": mid, "session_id": sid, "action_time": ns, "last_ns": ns, "source": source, "rx": float(rx), "ry": float(ry), "wheel_axis": axis, "signed_delta": int(wheel), "step_count": 1, "duration_ms": 0.0}
            elif event_type == "move":
                keep = last_move is None or ns - int(last_move[2]) >= 350_000_000 or abs(float(direction) - float(last_move[11])) >= 0.85 or math.hypot(float(rx) - float(last_move[6]), float(ry) - float(last_move[7])) >= 0.10
                if keep:
                    actions.append({"action_id": "move:{}:{}".format(sid, mid), "mouse_event_id": mid, "session_id": sid, "action_time": ns, "source": source, "rx": float(rx), "ry": float(ry), "action_type": "移动"})
                    last_move = row
        if wheel_bucket:
            wheel_bucket["action_type"] = "水平滚轮" if wheel_bucket["wheel_axis"] == "horizontal" else "滚轮"
            wheel_bucket["wheel_delta"] = max(-1200, min(1200, int(wheel_bucket["signed_delta"])))
            actions.append(wheel_bucket)
        return actions

    def _train_model(self, token):
        self.flush_mouse_records()
        if not self._wait_resource(token):
            return None
        frames_by_session, mouse_by_session = self.store.collect_training_data()
        fingerprint = str((sum(len(v) for v in frames_by_session.values()), sum(len(v) for v in mouse_by_session.values()), max([row[3] for rows in frames_by_session.values() for row in rows] or [0])))
        now_attempt = time.time()
        if fingerprint == self.last_training_data_fingerprint and now_attempt - self.last_training_attempt < 900.0:
            self.store.add_system_event(None, "model_skipped", {"reason": "训练数据指纹未变化", "fingerprint": fingerprint, "time": now_attempt})
            return self.store.best_model()
        self.last_training_attempt = now_attempt
        self.last_training_data_fingerprint = fingerprint
        if self._cancelled(token):
            return None
        timeline = []
        for sid, rows in frames_by_session.items():
            if rows:
                timeline.append((int(rows[0][2]), int(rows[-1][2]), sid))
        timeline.sort()
        all_start = min([a for a, _, _ in timeline] or [0])
        all_end = max([b for _, b, _ in timeline] or [0])
        split_ns = all_start + int((all_end - all_start) * 0.7)
        gap_ns = min(3_000_000_000, max(0, int((all_end - all_start) * 0.03)))
        outcomes = []
        validation_outcomes = []
        states = {}
        all_actions_count = 0
        for session_id, mouse_rows in mouse_by_session.items():
            frame_rows = frames_by_session.get(session_id, [])
            if len(frame_rows) < 2:
                continue
            semantic_actions = self._semantic_actions(mouse_rows)
            all_actions_count += len(semantic_actions)
            frame_finish_times = [int(row[14] or row[2]) for row in frame_rows]
            frame_start_times = [int(row[13] or row[2]) for row in frame_rows]
            critical_times = [int(row[2]) for row in mouse_rows if row[4] in ("button_down", "button_up", "wheel") and (row[12] or row[13])]
            for index, action in enumerate(semantic_actions):
                if index % 500 == 0 and not self._wait_resource(token):
                    return None
                action_ns = int(action["action_time"])
                before_index = bisect_right(frame_finish_times, action_ns) - 1
                after_start = action_ns + 250_000_000
                after_index = bisect_right(frame_start_times, after_start - 1)
                if before_index < 0 or after_index >= len(frame_rows):
                    continue
                if frame_start_times[after_index] < after_start or frame_start_times[after_index] > action_ns + 3_000_000_000:
                    continue
                if any(action_ns < t < frame_start_times[after_index] for t in critical_times):
                    continue
                before = frame_rows[before_index]
                after = frame_rows[after_index]
                score_delta = float(after[6]) - float(before[6])
                reward_delta = float(after[7]) - float(before[7])
                post_ms = (frame_start_times[after_index] - action_ns) / 1_000_000.0
                state_key = self.store.assign_state_cluster(before[4] or before[5], before[8], before[9], before[10], before[11])
                gx = min(15, max(0, int(action["rx"] * 16)))
                gy = min(8, max(0, int(action["ry"] * 9)))
                role = "train" if action_ns <= split_ns - gap_ns else "validation" if action_ns >= split_ns + gap_ns else "excluded_gap"
                example = {"action_id": action["action_id"], "session_id": session_id, "before_frame_id": before[0], "after_frame_id": after[0], "mouse_event_id": action["mouse_event_id"], "action_time": action_ns, "post_action_delay_ms": post_ms, "score_delta": score_delta, "reward_delta": reward_delta, "outcome_valid": True, "split_role": role}
                if role == "train":
                    wheel_axis = action.get("wheel_axis", "")
                    wheel_direction = 1 if int(action.get("wheel_delta", action.get("signed_delta", 0)) or 0) > 0 else -1 if int(action.get("wheel_delta", action.get("signed_delta", 0)) or 0) < 0 else 0
                    wheel_bucket = min(10, abs(int(action.get("wheel_delta", 0) or action.get("signed_delta", 0))) // 120)
                    key = (state_key, gx, gy, action["action_type"], wheel_axis, wheel_direction, wheel_bucket)
                    item = states.setdefault(key, {"samples": 0, "human_samples": 0, "ai_samples": 0, "sum": 0.0, "sum2": 0.0, "recent": 0.0, "examples": [], "state_hash": before[4] or before[5], "gray32x18": before[10], "edge_density": before[11], "color_histogram": before[12], "aspect": before[8] / max(1, before[9])})
                    item["samples"] += 1
                    item["human_samples" if action["source"] == "用户" else "ai_samples"] += 1
                    item["sum"] += reward_delta
                    item["sum2"] += reward_delta * reward_delta
                    item["recent"] = reward_delta
                    if "wheel_delta" in action:
                        example["wheel_delta"] = int(action.get("wheel_delta", 0))
                        example["wheel_axis"] = action.get("wheel_axis", "")
                    item["examples"].append(example)
                    outcomes.append(example)
                elif role == "validation":
                    self.store.save_action_outcomes([dict(example, split_role="validation")])
                    wheel_axis = action.get("wheel_axis", "")
                    wheel_direction = 1 if int(action.get("wheel_delta", action.get("signed_delta", 0)) or 0) > 0 else -1 if int(action.get("wheel_delta", action.get("signed_delta", 0)) or 0) < 0 else 0
                    wheel_bucket = min(10, abs(int(action.get("wheel_delta", 0) or action.get("signed_delta", 0))) // 120)
                    validation_outcomes.append((state_key, gx, gy, action["action_type"], wheel_axis, wheel_direction, wheel_bucket, example))
                else:
                    self.store.save_action_outcomes([dict(example, outcome_valid=False, split_role="excluded_gap")])
                    self.store.add_system_event(session_id, "training_sample_excluded", {"action_id": action["action_id"], "reason": "时间间隔隔离", "time": time.time()})
        if outcomes:
            for start in range(0, len(outcomes), 750):
                if self._cancelled(token):
                    return None
                self.store.save_action_outcomes(outcomes[start:start + 750])
        if not outcomes:
            self.store.add_system_event(None, "model_skipped", {"reason": "没有新增有效语义动作数据", "time": time.time(), "semantic_actions": all_actions_count})
            return self.store.best_model()
        actions = []
        policy_lookup = {}
        for (state_key, gx, gy, action_type, wheel_axis, wheel_direction, wheel_bucket), item in states.items():
            n = item["samples"]
            mean = item["sum"] / max(1, n)
            var = max(0.0, item["sum2"] / max(1, n) - mean * mean)
            lcb = mean - 1.96 * math.sqrt(var / max(1, n))
            if item.get("human_samples", 0) <= 0 and action_type != "移动":
                lcb = min(lcb, -0.01)
            action_payload = {"state_key": state_key, "state_hash": item["state_hash"], "gray32x18": item["gray32x18"], "edge_density": item["edge_density"], "color_histogram": item["color_histogram"], "aspect": item["aspect"], "x": (gx + 0.5) / 16.0, "y": (gy + 0.5) / 9.0, "action_type": action_type, "wheel_axis": wheel_axis, "wheel_direction": wheel_direction, "wheel_magnitude_bucket": wheel_bucket, "wheel_delta": int(round(sum(ex.get("wheel_delta", 0) for ex in item["examples"]) / max(1, len([ex for ex in item["examples"] if "wheel_delta" in ex])))) if action_type in ("滚轮", "水平滚轮") else 0, "samples": n, "human_samples": item.get("human_samples", 0), "ai_samples": item.get("ai_samples", 0), "average_reward_delta": mean, "reward_variance": var, "recent_validation": item["recent"], "confidence_lower_bound": lcb, "uncertainty": math.sqrt(var / max(1, n))}
            actions.append(action_payload)
            policy_lookup[(state_key, gx, gy, action_type, wheel_axis, wheel_direction, wheel_bucket)] = action_payload
        validation_rewards = []
        validation_hits = 0
        validation_failures = 0
        for state_key, gx, gy, action_type, wheel_axis, wheel_direction, wheel_bucket, example in validation_outcomes:
            chosen = policy_lookup.get((state_key, gx, gy, action_type, wheel_axis, wheel_direction, wheel_bucket))
            if chosen is None:
                validation_failures += 1
                continue
            validation_hits += 1
            validation_rewards.append(float(example["reward_delta"]))
            if example["reward_delta"] <= 0:
                validation_failures += 1
        validation_mean = sum(validation_rewards) / max(1, len(validation_rewards))
        validation_var = sum((v - validation_mean) ** 2 for v in validation_rewards) / max(1, len(validation_rewards))
        validation_ci = 1.96 * math.sqrt(validation_var / max(1, len(validation_rewards)))
        validation_quality = validation_mean - validation_ci
        validation_sample_count = len(validation_outcomes)
        payload = {"id": uuid.uuid4().hex, "trained_at": time.time(), "quality": validation_quality, "train_quality": sum(a["average_reward_delta"] for a in actions) / max(1, len(actions)), "frame_count": sum(len(v) for v in frames_by_session.values()), "mouse_count": sum(len(v) for v in mouse_by_session.values()), "training_samples": len(outcomes), "semantic_actions": all_actions_count, "validation_samples": validation_sample_count, "validation_hits": validation_hits, "validation_mean_reward": validation_mean, "validation_confidence_interval": validation_ci, "validation_failure_rate": validation_failures / max(1, validation_sample_count), "validation_state_coverage": len({item[0] for item in validation_outcomes}), "coverage_states": len({a["state_key"] for a in actions}), "failure_rate": len([a for a in actions if a["average_reward_delta"] <= 0]) / max(1, len(actions)), "model_version": 4, "champion": True, "last_used": time.time(), "mean_reward": sum(o["reward_delta"] for o in outcomes) / max(1, len(outcomes)), "action_quality": validation_quality, "validation_quality": validation_quality, "policy": {"min_samples": 5, "uncertainty_threshold": 0.25, "min_confidence_lower_bound": 0.0, "similarity_threshold": 0.78, "low_confidence_action": "move_only", "blacklist_regions": []}, "q_actions": sorted(actions, key=lambda a: (a["confidence_lower_bound"], a["samples"]), reverse=True)[:256], "outcome_examples": outcomes[-256:]}
        champion = self.store.best_model()
        champion_quality = float(champion.get("validation_quality", -999999.0)) if isinstance(champion, dict) else -999999.0
        enough = len(outcomes) >= 30 and len({a["state_key"] for a in actions}) >= 4 and validation_sample_count >= 12 and validation_hits >= 6
        if not enough or validation_quality <= champion_quality:
            payload["champion"] = False
            self.store.add_system_event(None, "model_candidate_rejected", {"validation_quality": validation_quality, "champion_quality": champion_quality, "validation_samples": validation_sample_count, "validation_hits": validation_hits, "training_samples": len(outcomes), "fingerprint": fingerprint, "time": time.time()})
            return champion if isinstance(champion, dict) else payload
        name = "model_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + payload["id"][:8] + ".json"
        final_path = self.store.models / name
        temp_path = final_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if self._cancelled(token):
            temp_path.unlink(missing_ok=True)
            return None
        temp_path.replace(final_path)
        self.store.save_model_frame_refs(payload["id"], outcomes, validation_outcomes)
        self.last_model_training = time.time()
        self.last_training_success = self.last_model_training
        return payload

    def _sleep_worker(self, token, resume_training):
        try:
            self.emit("progress", 4.0)
            self.emit("state", {"state": "sleep", "detail": "任务1：训练 AI 模型", "cpu": self.resources.sample()["cpu"], "memory": self.resources.sample()["memory"]})
            model = self._train_model(token)
            if self._cancelled(token):
                return
            self.emit("progress", 56.0)
            self.emit("state", {"state": "sleep", "detail": "任务1完成；任务2：检查 AI 模型与经验池", "cpu": self.resources.sample()["cpu"], "memory": self.resources.sample()["memory"]})
            self.store.recover_deletions()
            if not self._wait_resource(token, "maintenance"):
                return
            self.flush_mouse_records()
            model_removed = self.store.prune_models(max(1, int(self.settings.data["model_limit"])), lambda: self._cancelled(token), lambda: self._wait_resource(token, "maintenance"))
            def update(value):
                self.emit("progress", value)
            experience_removed, remaining = self.store.prune_experience(max(1, int(self.settings.data["experience_limit"])), lambda: self._cancelled(token), update, lambda: self._wait_resource(token, "maintenance"))
            if self._cancelled(token):
                return
            self.emit("progress", 100.0)
            detail = "任务2完成：删除 AI 模型 {} 个，删除经验 {} 条，经验池 {:.2f} MB".format(model_removed, experience_removed, remaining / 1024 / 1024)
            if resume_training:
                hwnd, rect, reason = self._find_valid_target(False)
                if hwnd is None or rect is None or not self._place_cursor_before_entry(hwnd, rect):
                    self._finish_idle(token, "自动睡眠完成，但无法恢复训练：" + (reason or "客户区状态异常"), True)
                    return
                if self._cancelled(token) or valid_client(hwnd, True) is None:
                    self._finish_idle(token, "自动睡眠完成，但雷电模拟器客户区状态异常", True)
                    return
                self.emit("progress", 0.0)
                if not self.start_session("training", automatic=True):
                    self._finish_idle(token, "自动睡眠完成，但无法恢复训练模式", True)
            else:
                self._finish_idle(token, detail, True)
        except Exception as error:
            self._finish_idle(token, "睡眠模式发生错误：" + str(error), True)

    def _finish_idle(self, token, detail, release_keyboard=True):
        with self.lock:
            if token != self.epoch:
                return
            self.state = "idle"
            self.cancel_event.set()
            self.epoch += 1
        if release_keyboard:
            self.keyboard_hook.stop()
        self.emit("progress", 0.0)
        self.post_state(detail)

    def information(self):
        sample = self.resources.sample()
        with self.lock:
            state = self.state
            frames = self.frame_count
            mouse = self.mouse_count
            session = self.session_id
        try:
            pool_size = self.store.pool_size() if self.store.pool else 0
            model_count = len(self.store.model_files()) if self.store.models else 0
        except Exception:
            pool_size = 0
            model_count = 0
        resource = dict(sample)
        return {
            "state": state,
            "frames": frames,
            "mouse": mouse,
            "session": session or "无",
            "cpu": sample["cpu"],
            "memory": sample["memory"],
            "pool_size": pool_size,
            "model_count": model_count,
            "resource": resource
        }

    def shutdown(self):
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
        self.control_queue.put(None)
        self.control_thread.join(3.0)
        self.hook.stop()
        self.keyboard_hook.stop()
        with self.lock:
            active = [thread for thread in self.worker_threads if thread.is_alive()]
        if active or not flushed:
            try:
                self.store.add_system_event(None, "incomplete_shutdown", {"active_threads": [thread.name for thread in active], "mouse_flushed": flushed, "time": time.time()})
            except Exception:
                pass
        try:
            if self.store.conn is not None:
                self.store.conn.interrupt()
        except Exception:
            pass
        try:
            self.store.recover_deletions()
            self.store._compact_database()
            self.store.validate_consistency()
        finally:
            self.store.close()

class Panel:
    def __init__(self, root):
        self.root = root
        self.settings = Settings()
        self.events = queue.Queue()
        self.controller = Controller(self.settings, self.enqueue)
        self.path_var = StringVar(value=self.settings.data["emulator_path"])
        self.storage_var = StringVar(value=self.settings.data["storage_path"])
        self.experience_var = StringVar(value=self.format_bytes(self.settings.data["experience_limit"]))
        self.model_var = StringVar(value=str(self.settings.data["model_limit"]) + " 个")
        self.mode_var = StringVar(value="空闲")
        self.status_var = StringVar(value=("配置读取错误：" + "；".join(self.settings.config_errors)) if self.settings.config_errors else "控制面板已就绪。")
        self.performance_var = StringVar(value="CPU 0.0% · 内存 0.0%")
        self.progress_var = DoubleVar(value=0.0)
        self.mode_buttons = []
        self.configuration_buttons = []
        self.scroll_canvas = None
        self.panel_hidden_for_mode = False
        self.build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(90, self.drain)
        self.root.after(1200, self.refresh_performance)

    def format_bytes(self, value):
        return "{:.2f} GB".format(float(value) / 1024 / 1024 / 1024)

    def enqueue(self, kind, payload):
        self.events.put((kind, payload))

    def button(self, parent, text, command, color, row=None, column=None, **grid):
        item = Button(parent, text=text, command=command, bg=color, fg="white", activebackground=color, activeforeground="white", relief="flat", bd=0, font=("Microsoft YaHei UI", 10, "bold"), cursor="hand2", padx=14, pady=10, takefocus=True)
        if row is not None:
            item.grid(row=row, column=column, **grid)
        return item

    def build(self):
        self.root.title("雷电智能学习与训练控制面板")
        self.root.geometry("960x660")
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
        horizontal = ttk.Scrollbar(host, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
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
        Label(header, text="雷电智能学习与训练控制面板", bg="#101826", fg="white", font=("Microsoft YaHei UI", 20, "bold")).grid(row=0, column=0, sticky="w")
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
        body.grid_columnconfigure(1, weight=1)
        labels = (("雷电模拟器", self.path_var), ("存储路径", self.storage_var), ("经验池上限", self.experience_var), ("AI 模型数量上限", self.model_var))
        colors = ("#ef4444", "#f97316", "#eab308", "#22c55e")
        commands = (self.choose_emulator, self.choose_storage, self.change_experience, self.change_models)
        texts = ("选择雷电模拟器路径", "选择存储路径", "修改经验池上限", "修改AI模型数量上限")
        for row, ((title, variable), color, command, text) in enumerate(zip(labels, colors, commands, texts)):
            base_row = row * 3
            Label(body, text=title, bg="#f8fafc", fg="#334155", font=("Microsoft YaHei UI", 10, "bold"), anchor="w").grid(row=base_row, column=0, columnspan=3, sticky="w", pady=(8, 2))
            value = Label(body, textvariable=variable, bg="#e2e8f0", fg="#0f172a", font=("Consolas", 9), anchor="w", padx=10, pady=9, justify="left", wraplength=720)
            value.grid(row=base_row + 1, column=0, columnspan=3, sticky="ew", pady=3)
            action = self.button(body, text, command, color, row=base_row + 2, column=0, columnspan=3, sticky="ew", pady=(3, 8))
            self.configuration_buttons.append(action)
            def update_wrap(event=None, label=value):
                label.configure(wraplength=max(140, body.winfo_width() - 72))
            body.bind("<Configure>", update_wrap, add="+")
        divider = Frame(body, bg="#cbd5e1", height=1)
        divider.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(12, 12))
        actions = Frame(body, bg="#f8fafc")
        actions.grid(row=13, column=0, columnspan=3, sticky="ew")
        for index in range(4):
            actions.grid_columnconfigure(index, weight=1)
        info_button = self.button(actions, "更多信息", self.more_info, "#06b6d4", row=0, column=0, sticky="ew", padx=(0, 7))
        learn = self.button(actions, "学习模式", lambda: self.start_mode("learning"), "#3b82f6", row=0, column=1, sticky="ew", padx=7)
        train = self.button(actions, "训练模式", lambda: self.start_mode("training"), "#a855f7", row=0, column=2, sticky="ew", padx=7)
        sleep = self.button(actions, "睡眠模式", self.controller.start_sleep, "#ef4444", row=0, column=3, sticky="ew", padx=(7, 0))
        action_buttons = [info_button, learn, train, sleep]
        def layout_actions(event=None):
            width = actions.winfo_width()
            columns = 4 if width >= 760 else (2 if width >= 480 else 1)
            for i in range(4):
                actions.grid_columnconfigure(i, weight=1 if i < columns else 0)
            for index, button in enumerate(action_buttons):
                button.grid_configure(row=index // columns, column=index % columns, padx=5, pady=5, sticky="ew")
        actions.bind("<Configure>", layout_actions)
        self.root.after_idle(layout_actions)
        self.mode_buttons = [learn, train, sleep]
        Label(body, text="任务进度", bg="#f8fafc", fg="#334155", font=("Microsoft YaHei UI", 10, "bold"), anchor="w").grid(row=14, column=0, sticky="w", pady=(17, 6))
        progress = ttk.Progressbar(body, orient="horizontal", maximum=100.0, variable=self.progress_var, mode="determinate")
        progress.grid(row=14, column=1, columnspan=2, sticky="ew", pady=(17, 6))
        footer = Frame(body, bg="#eef2ff", padx=12, pady=10)
        footer.grid(row=15, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        footer.grid_columnconfigure(0, weight=1)
        Label(footer, textvariable=self.status_var, bg="#eef2ff", fg="#1e3a8a", font=("Microsoft YaHei UI", 9), anchor="w", justify="left", wraplength=550).grid(row=0, column=0, sticky="ew")
        Label(footer, textvariable=self.performance_var, bg="#eef2ff", fg="#475569", font=("Microsoft YaHei UI", 9), anchor="e").grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.root.after_idle(sync_region)

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
            if event.state & 0x0001:
                canvas.xview_scroll(steps * 3, "units")
            else:
                canvas.yview_scroll(steps * 3, "units")
            return "break"
        except Exception:
            return None

    def restore_panel(self):
        if not self.panel_hidden_for_mode:
            return
        self.panel_hidden_for_mode = False
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.update_idletasks()
        except Exception:
            pass

    def start_mode(self, mode):
        if self.controller.busy():
            self.controller.emit("notice", "当前模式：" + self.controller.current_state() + "，拒绝重复进入。")
            return False
        self.panel_hidden_for_mode = True
        try:
            self.root.withdraw()
            self.root.update_idletasks()
        except Exception:
            self.restore_panel()
            self.controller.emit("notice", "控制面板无法临时隐藏，未进入模式。")
            return False
        def continue_start():
            started = self.controller.start_session(mode)
            if not started:
                self.restore_panel()
        self.root.after(80, continue_start)
        return True

    def protect_configuration(self):
        if self.controller.busy():
            messagebox.showwarning("当前状态", "运行中的模式不会修改配置。请先返回空闲状态。", parent=self.root)
            return False
        return True

    def choose_emulator(self):
        if not self.protect_configuration():
            return
        selected = filedialog.askopenfilename(parent=self.root, title="选择雷电模拟器路径", initialfile=Path(self.settings.data["emulator_path"]).name, filetypes=[("可执行文件", "*.exe"), ("所有文件", "*.*")])
        if selected:
            self.settings.data["emulator_path"] = selected
            self.settings.save()
            self.path_var.set(selected)
            self.status_var.set("已更新雷电模拟器路径。")

    def choose_storage(self):
        if not self.protect_configuration():
            return
        selected = filedialog.askdirectory(parent=self.root, title="选择存储路径", initialdir=self.settings.data["storage_path"])
        if selected:
            self.settings.data["storage_path"] = selected
            self.settings.save()
            self.storage_var.set(selected)
            self.status_var.set("已更新存储路径。")

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
        info = self.controller.information()
        window = Toplevel(self.root)
        window.title("更多信息")
        window.geometry("670x500")
        window.resizable(True, True)
        window.configure(bg="#0f172a")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(1, weight=1)
        Label(window, text="运行信息", bg="#0f172a", fg="white", font=("Microsoft YaHei UI", 18, "bold"), padx=20, pady=18).grid(row=0, column=0, sticky="w")
        content = Frame(window, bg="#f8fafc", padx=20, pady=18)
        content.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        content.grid_columnconfigure(1, weight=1)
        rows = [
            ("当前状态", info["state"]),
            ("本次会话", info["session"]),
            ("本次记录画面", str(info["frames"])),
            ("本次记录鼠标事件", str(info["mouse"])),
            ("CPU 使用率", "{:.1f}%".format(info["cpu"])),
            ("内存使用率", "{:.1f}%".format(info["memory"])),
            ("经验池大小", self.format_bytes(info["pool_size"])),
            ("AI 模型数量", str(info["model_count"])),
            ("奖励定义", "画面评分 − 饥饿值"),
            ("画面评分", "与最相似的一批历史画面越相似，评分越低"),
            ("饥饿重置", "当前有效画面评分高于上一条时重置为正极小值"),
            ("资源保护", "自动限速、批量写入、训练暂停、可取消清理")
        ]
        for index, (name, value) in enumerate(rows):
            Label(content, text=name, bg="#f8fafc", fg="#475569", font=("Microsoft YaHei UI", 10, "bold"), anchor="w").grid(row=index, column=0, sticky="w", pady=5)
            Label(content, text=value, bg="#f8fafc", fg="#0f172a", font=("Microsoft YaHei UI", 10), anchor="w", wraplength=410, justify="left").grid(row=index, column=1, sticky="ew", padx=(18, 0), pady=5)

    def drain(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "state":
                    state = payload.get("state", "idle")
                    names = {"idle": "空闲", "learning": "学习模式", "training": "训练模式", "sleep": "睡眠模式", "stopping": "正在停止"}
                    self.mode_var.set(names.get(state, state))
                    detail = payload.get("detail", "")
                    self.status_var.set(detail or "控制面板已就绪。")
                    self.performance_var.set("CPU {:.1f}% · 内存 {:.1f}%".format(payload.get("cpu", 0.0), payload.get("memory", 0.0)))
                    if state == "idle":
                        self.restore_panel()
                    normal = "normal" if state == "idle" else "disabled"
                    for button in self.mode_buttons:
                        button.configure(state=normal)
                    for button in self.configuration_buttons:
                        button.configure(state=normal)
                elif kind == "progress":
                    self.progress_var.set(max(0.0, min(100.0, float(payload))))
                elif kind == "notice":
                    self.status_var.set(str(payload))
                    messagebox.showwarning("提示", str(payload), parent=self.root)
        except queue.Empty:
            pass
        try:
            self.root.after(90, self.drain)
        except Exception:
            pass

    def refresh_performance(self):
        try:
            info = self.controller.information()
            self.performance_var.set("CPU {:.1f}% · 内存 {:.1f}%".format(info["cpu"], info["memory"]))
            self.root.after(1200, self.refresh_performance)
        except Exception:
            pass

    def close(self):
        try:
            self.controller.shutdown()
        finally:
            self.root.destroy()

def main():
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
