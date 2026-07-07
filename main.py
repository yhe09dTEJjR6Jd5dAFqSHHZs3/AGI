import ctypes
import concurrent.futures
import heapq
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
    dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
except OSError:
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
    for key in ("rgb", "png", "gray32x18", "color_histogram"):
        value = image.get(key) if isinstance(image, dict) else None
        if isinstance(value, memoryview):
            total += len(value)
        elif isinstance(value, (bytes, bytearray, str)):
            total += len(value)
    return max(4096, total + 8192)

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

class HardwareProbe:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_probe = 0.0
        self.last_runtime_probe = 0.0
        self.gpus = []
        self.runtime = {"available": False, "source": "GPU 指标尚未初始化", "program_gpu": 0.0, "ldplayer_gpu": 0.0, "gpu_engine": 0.0, "dedicated_used": 0, "dedicated_total": None, "dedicated_free": None}
        self.backend = "CPU 表格策略"
        self._pdh_query = None
        self._pdh_engine = []
        self._pdh_memory = []
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
        if not self._pdh_ready:
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
        result["source"] = str(result.get("source") or "GPU 指标不可用") + "；未加载经 warmup 验证的 ONNX 模型，当前仅 CPU 表格策略"
        with self.lock:
            self.runtime = result
            self.last_runtime_probe = now
            return dict(result)

    def choose_gpu(self, metrics):
        return "CPU"

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

    def run_gpu_features(self, vector, metrics):
        if not self.refresh_gpu_stability(metrics):
            return None
        try:
            import numpy as np
            with self.lock:
                session = self.model_session
            inputs = session.get_inputs()
            sample = inputs[0]
            array = np.asarray(vector, dtype=np.float32)
            return session.run(None, {sample.name: array})
        except Exception as error:
            self.disable_gpu("GPU 推理失败：" + str(error), 300.0)
            return None

    def disable_gpu(self, reason, retry_seconds=300.0):
        with self.lock:
            self.runtime_ready = False
            self.model_session = None
            self.gpu_failure_reason = str(reason)
            self.gpu_retry_after = time.monotonic() + max(30.0, float(retry_seconds))
            self.last_backend = "CPU 表格策略；GPU 回退"

    def _ensure_executor(self, workers):
        workers = max(1, min(4, int(workers)))
        with self.lock:
            if self.executor is None:
                self.executor_workers = 4
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="PngEncode")
            return self.executor

    def _encode_parallelism(self, workers):
        return max(1, min(4, int(workers)))

    def set_encode_metrics_sink(self, sink):
        with self.lock:
            self.encode_metrics_sink = sink

    def name(self):
        with self.lock:
            return self.last_backend

    def shutdown(self):
        with self.lock:
            executor = self.executor
            self.executor = None
            self.executor_workers = 0
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _encode_png_one(self, frame, queued_at=None):
        started = time.monotonic()
        with self._png_lock:
            self._png_active += 1
            active = self._png_active
        try:
            image = dict(frame)
            image["png"] = encode_png(int(image["width"]), int(image["height"]), image.pop("rgb"))
            image["compute_backend"] = "CPU PNG 并行编码"
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
        encoded = []
        for start in range(0, len(items), slots):
            batch = items[start:start + slots]
            futures = [executor.submit(self._encode_png_one, item, queued_at) for item in batch]
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
        with self.lock:
            self.last_backend = "CPU 表格策略"
        return {"backend": "CPU 表格策略", "confidence": max(0.0, min(1.0, confidence)), "uncertainty": uncertainty, "executed": True}

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
    def __init__(self):
        self.lock = threading.Lock()
        self.storage_path = Path.cwd()
        self.emulator_path = ""
        self.emulator_pid = 0
        self.previous = None
        self.process_previous = {}
        self.process_last_sample = {}
        self.last_sample = 0.0
        self.last_disk_probe = 0.0
        self.window = []
        self.metrics = {"cpu": 0.0, "process_cpu": 0.0, "process_memory": 0, "ldplayer_cpu": 0.0, "ldplayer_memory": 0, "memory": 0.0, "avail_memory": 0, "commit_free": 0, "disk_free": 0, "disk_write_latency": None, "sqlite_latency": 0.0, "capture_latency": 0.0, "queue": 0, "queue_age": 0.0, "pipeline_queue": 0, "pipeline_queue_age": 0.0, "pipeline_queue_capacity": 96, "pipeline_queue_ratio": 0.0, "gpu": None, "gpu_dedicated_total": None, "gpu_dedicated_used": None, "gpu_dedicated_free": None, "gpu_engine": None, "ldplayer_gpu": None, "program_gpu": None, "gpu_metrics_available": False, "gpu_sampling_source": "Windows GPU 性能计数器不可用", "last_user_input": time.time(), "capture_failure_rate": 0.0, "png_encode_ms": 0.0, "png_active": 0, "png_queue_age": 0.0, "exact_score_backlog": 0, "exact_score_oldest_age": 0.0, "wal_bytes": 0, "wal_checkpoint_ms": 0.0, "sqlite_transaction_ms": 0.0, "metric_sources": {"本程序 CPU": "GetProcessTimes", "雷电 CPU": "GetProcessTimes（绑定雷电进程树）", "本程序 GPU 引擎": "Windows GPU 性能计数器", "雷电 GPU 引擎": "Windows GPU 性能计数器", "可用显存": "Windows GPU Adapter Memory 性能计数器", "磁盘写入延迟": "fsync 探针", "SQLite 写入延迟": "实际 SQLite 事务计时", "队列年龄": "队列记录时间戳"}}
        self.probe = HardwareProbe()
        self.backend = ComputeBackend(self.probe)
        self.backend.set_encode_metrics_sink(self.update_encode_metrics)
        self.gpu_scheduler = GpuScheduler(self.probe)
        self.resource_state = "正常"
        self.resource_state_since = time.monotonic()
        self.resource_decisions = []
        self.runtime = ModelRuntime(self.backend)
        self.levels = {}
        self.last_pressure = 0.0
        self.healthy_since = time.monotonic()
        self.red_latched = False
        self.red_recovery_since = None
        self.pressure_reasons = []
        self._slow_stop = threading.Event()
        self._slow_thread = None
        self.sample()
        self._slow_thread = threading.Thread(target=self._slow_sample_loop, name="ResourceSlowProbe", daemon=True)
        self._slow_thread.start()
        self._fast_stop = threading.Event()
        self._fast_budget = ResourceBudget(True, 1.0, 1, 1, "CPU", 0, (640, 360), False, "", "正常")
        self._fast_thread = threading.Thread(target=self._fast_budget_loop, name="ResourceBudgetSnapshot", daemon=True)
        self._fast_thread.start()

    def set_storage_path(self, path):
        with self.lock:
            self.storage_path = Path(path)

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
                previous = self.process_previous.get(int(pid))
                previous_time = self.process_last_sample.get(int(pid), now)
                if previous is not None:
                    elapsed = max(0.001, now - previous_time)
                    cpu = max(0.0, min(100.0, (total - previous) / 10000000.0 / elapsed * 100.0 / max(1, os.cpu_count() or 1)))
                self.process_previous[int(pid)] = total
                self.process_last_sample[int(pid)] = now
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
        probe_path = Path(self.storage_path) / "experience_pool" / ".write_latency_probe.tmp"
        try:
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            with probe_path.open("wb") as handle:
                handle.write(b"0" * 4096)
                handle.flush()
                os.fsync(handle.fileno())
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
            cpu = self.metrics.get("cpu", 0.0)
            idle = FILETIME(); kernel = FILETIME(); user = FILETIME()
            if kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                current = tuple((item.dwHighDateTime << 32) | item.dwLowDateTime for item in (idle, kernel, user))
                if self.previous is not None:
                    idle_delta = current[0] - self.previous[0]
                    total_delta = current[1] + current[2] - self.previous[1] - self.previous[2]
                    if total_delta > 0:
                        cpu = max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))
                self.previous = current
            status = MEMORYSTATUSEX(); status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            memory = self.metrics.get("memory", 0.0)
            avail_memory = self.metrics.get("avail_memory", 0)
            commit_free = self.metrics.get("commit_free", 0)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                memory = float(status.dwMemoryLoad)
                avail_memory = int(status.ullAvailPhys)
                commit_free = max(0, int(status.ullAvailPageFile))
            try:
                disk_free = int(shutil.disk_usage(self.storage_path).free)
            except Exception:
                disk_free = 0
            process_cpu, process_memory, ld_cpu, ld_memory = self._process_metrics(now)
            self.metrics.update({"cpu": cpu, "memory": memory, "avail_memory": avail_memory, "commit_free": commit_free, "disk_free": disk_free, "process_cpu": process_cpu, "process_memory": process_memory, "ldplayer_cpu": ld_cpu, "ldplayer_memory": ld_memory})
            point = dict(self.metrics, t=now)
            self.window.append(point)
            self.window = [p for p in self.window if now - p["t"] <= 10.0]
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
        return result

    def acquire(self, task):
        sample = self.sample()
        now = time.monotonic()
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
        red_now = False
        if cpu_p95 >= 95.0:
            red_now = True
            reasons.append("系统 CPU P95 ≥ 95%")
        if int(sample.get("avail_memory", 0) or 0) < 384 * 1024 * 1024:
            red_now = True
            reasons.append("可用内存不足 384 MB")
        if queue_age > 2.0:
            red_now = True
            reasons.append("高优先级队列等待超过 2 秒")
        if disk_p95 > 150.0:
            red_now = True
            reasons.append("磁盘写入 P95 超过 150 ms")
        if int(sample.get("disk_free", 0) or 0) < 1024 * 1024 * 1024:
            red_now = True
            reasons.append("磁盘剩余空间不足 1 GB")
        if capture_failure_rate >= 0.35:
            red_now = True
            reasons.append("截图连续失败率过高")
        if checkpoint_p95 > 2000.0:
            red_now = True
            reasons.append("SQLite WAL checkpoint P95 超过 2 秒")
        if cpu_p95 >= 85.0:
            yellow.append("系统 CPU P95 ≥ 85%")
        if float(sample.get("process_cpu", 0.0) or 0.0) >= 60.0:
            yellow.append("本程序 CPU ≥ 60%")
        if float(sample.get("ldplayer_cpu", 0.0) or 0.0) >= 70.0:
            yellow.append("雷电 CPU ≥ 70%")
        if float(sample.get("memory", 0.0) or 0.0) >= 88.0:
            yellow.append("内存占用 ≥ 88%")
        if sample.get("gpu_metrics_available") and float(sample.get("gpu_engine_p95", sample.get("gpu_engine", 0.0)) or 0.0) >= 85.0:
            yellow.append("GPU 引擎 P95 ≥ 85%")
            if self.backend.can_use_gpu():
                self.backend.disable_gpu("GPU 引擎 P95 过高", 300.0)
        if float(sample.get("capture_latency_p95", sample.get("capture_latency", 0.0)) or 0.0) > 150.0:
            yellow.append("截图 P95 延迟过高")
        if float(sample.get("sqlite_latency_p95", sample.get("sqlite_latency", 0.0)) or 0.0) > 100.0:
            yellow.append("SQLite 写入 P95 延迟过高")
        if disk_p95 > 100.0:
            yellow.append("磁盘写入延迟升高")
        if queue_age > 1.0:
            yellow.append("队列等待超过 1 秒")
        if queue_ratio >= 0.70:
            yellow.append("流水线队列达到 70%")
        if capture_failure_rate >= 0.08:
            yellow.append("截图失败率升高")
        if png_p95 > 120.0:
            yellow.append("PNG 编码 P95 延迟过高")
        if checkpoint_p95 > 500.0:
            yellow.append("SQLite WAL checkpoint P95 延迟过高")
        if wal_bytes > 512 * 1024 * 1024:
            yellow.append("SQLite WAL 文件增长过大")
        if exact_backlog >= 96:
            yellow.append("精确评分积压过多")
        with self.lock:
            if red_now:
                self.red_latched = True
                self.red_recovery_since = None
            elif self.red_latched:
                if self.red_recovery_since is None:
                    self.red_recovery_since = now
                elif now - self.red_recovery_since >= 20.0 and queue_age <= 0.25:
                    self.red_latched = False
                    self.red_recovery_since = None
            red_pause = bool(self.red_latched)
            pressure = red_pause or bool(yellow)
            if pressure:
                self.last_pressure = now
                self.healthy_since = now
                for key in ("capture", "ai_inference", "sleep_training", "maintenance"):
                    self.levels[key] = max(1, int(self.levels.get(key, 4) / 2))
            elif now - self.healthy_since >= 20.0 and now - self.last_pressure >= 20.0:
                self.levels[task] = min(16, int(self.levels.get(task, 4)) + 1)
                self.healthy_since = now
            level = int(self.levels.get(task, 4))
            recovery_note = "红色条件已消失，恢复观察中" if red_pause and not red_now else ""
            self.pressure_reasons = reasons + yellow + ([recovery_note] if recovery_note else [])
        cores = max(1, os.cpu_count() or 1)
        workers = 1 if task in ("maintenance", "ai_inference") else max(1, min(max(1, cores - max(1, math.ceil(cores * 0.25))), level))
        interval_base = {"capture": 1.0, "ai_inference": 0.8, "sleep_training": 0.05, "maintenance": 1.0}.get(task, 1.0)
        interval = max(0.05, interval_base * max(1.0, 4.0 / max(1, level)))
        if png_p95 > 120.0:
            workers = 1
        resolution = (640, 360) if level >= 4 else (426, 240) if level >= 2 else (320, 180)
        if png_p95 > 250.0:
            resolution = (426, 240)
        if png_p95 > 500.0:
            interval *= 2.0
        if queue_ratio >= 0.70:
            interval *= 2.0
        draining = queue_ratio >= 0.85 or exact_backlog >= 128
        if draining:
            workers = 1
            resolution = (320, 180)
            interval *= 3.0
        max_batch = 1 if draining else max(1, min(64, level * (2 if task == "sleep_training" else 1)))
        pause = red_pause or (task in ("sleep_training", "maintenance") and bool(yellow))
        state = "暂停" if pause else "排空" if draining else "降速" if pressure else "正常"
        with self.lock:
            self._set_resource_state_locked(state, self.pressure_reasons)
        deadline = max(0.03, min(0.25, interval * 0.35))
        gpu_assignment = self.gpu_scheduler.assign(sample, self.backend)
        gpu_batch = max_batch if gpu_assignment == "GPU" else 0
        return ResourceBudget(not pause, interval, max_batch, workers, gpu_assignment, gpu_batch, resolution, pause, "；".join(self.pressure_reasons), state, max(8, min(512, level * 16)), max(8, min(128, level * 8)), max(8, min(256, level * 16)), max(1, min(workers, 4)), deadline, queue_ratio)

class Settings:
    def __init__(self):
        appdata = Path(os.environ.get("APPDATA", str(Path.home())))
        self.path = appdata / "LDTrainingPanel" / "settings.json"
        self.data = {"emulator_path": r"D:\LDPlayer9\dnplayer.exe", "storage_path": r"C:\Users\Administrator\Desktop\AAA", "experience_limit": 10 * 1024 * 1024 * 1024, "model_limit": 100, "transaction_reserve_bytes": 8 * 1024 * 1024, "emulator_pid": 0, "emulator_title": ""}
        self.config_errors = []
        self.load()

    def load(self):
        self.config_errors = []
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as error:
            if self.path.exists(): self.config_errors.append("配置读取失败:" + str(error))
            return
        if not isinstance(loaded, dict):
            self.config_errors.append("配置根对象不是字典")
            return
        for key in ("emulator_path", "storage_path", "emulator_title"):
            if key in loaded:
                if isinstance(loaded[key], str): self.data[key] = loaded[key]
                else: self.config_errors.append(key + " 类型无效")
        for key, minimum, maximum in (("experience_limit", int(0.1*1024*1024*1024), 4096*1024*1024*1024), ("transaction_reserve_bytes", 1*1024*1024, 512*1024*1024), ("model_limit", 1, 100000), ("emulator_pid", 0, 2**31-1)):
            if key in loaded:
                value=loaded[key]
                if isinstance(value,int) and minimum <= value <= maximum: self.data[key]=value
                else: self.config_errors.append(key + " 超出范围或类型无效")

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

class PoolCapacityBlocked(RuntimeError):
    pass

class DataStore:
    def __init__(self):
        self.root = None
        self.pool = None
        self.models = None
        self.screens = None
        self.conn = None
        self.lock = threading.RLock()
        self._database_bytes_cached = 0
        self._database_bytes_checked = 0.0
        self.transaction_reserve_bytes = 8 * 1024 * 1024
        self._recent_png_sizes = []
        self.faults = {}
        self.exact_score_lock = threading.Lock()
        self.last_wal_metrics = {"wal_bytes": 0, "checkpoint_ms": 0.0, "transaction_ms": 0.0}
        self.last_prune_coverage_loss = {"before": 0, "after": 0, "loss": 0, "ratio": 0.0, "paused": False}

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

    def set_transaction_reserve(self, value):
        with self.lock:
            self.transaction_reserve_bytes = max(1 * 1024 * 1024, min(512 * 1024 * 1024, int(value)))

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
            self._database_bytes_cached = 0
            self._database_bytes_checked = 0.0
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
                score_status TEXT NOT NULL DEFAULT 'unknown',
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
                rule TEXT NOT NULL
            );
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
            """)
            self._migrate()
            self.conn.commit()
            self.recover_ingestions()
            self.recover_deletions()

    def _migrate(self):
        frame_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(frames)").fetchall()}
        additions = {
            "size_bytes": "INTEGER NOT NULL DEFAULT 0", "online_score": "REAL", "raw_score": "REAL", "raw_reward": "REAL", "score_status": "TEXT NOT NULL DEFAULT 'unknown'", "novelty": "REAL NOT NULL DEFAULT 0", "action_result": "REAL NOT NULL DEFAULT 0", "coverage": "REAL NOT NULL DEFAULT 0", "model_refs": "INTEGER NOT NULL DEFAULT 0", "retain_value": "REAL NOT NULL DEFAULT 0", "retain_version": "INTEGER NOT NULL DEFAULT 1", "last_used": "REAL NOT NULL DEFAULT 0", "dhash64": "TEXT", "bucket0": "INTEGER NOT NULL DEFAULT 0", "bucket1": "INTEGER NOT NULL DEFAULT 0", "bucket2": "INTEGER NOT NULL DEFAULT 0", "bucket3": "INTEGER NOT NULL DEFAULT 0", "state_cluster_id": "TEXT", "state_support_count": "INTEGER NOT NULL DEFAULT 1", "action_outcome_information": "REAL NOT NULL DEFAULT 0", "model_dependency_count": "INTEGER NOT NULL DEFAULT 0", "validation_last_used": "REAL NOT NULL DEFAULT 0", "created_monotonic_ns": "INTEGER NOT NULL DEFAULT 0", "capture_backend": "TEXT NOT NULL DEFAULT 'gdi'", "capture_elapsed_ms": "REAL NOT NULL DEFAULT 0", "capture_complete": "INTEGER NOT NULL DEFAULT 1", "brightness": "REAL NOT NULL DEFAULT 0", "variance": "REAL NOT NULL DEFAULT 0", "gray32x18": "BLOB", "edge_density": "REAL NOT NULL DEFAULT 0", "color_histogram": "BLOB", "asset_ref_count": "INTEGER NOT NULL DEFAULT 1", "score_candidate_count": "INTEGER NOT NULL DEFAULT 0", "score_top_k_distance": "REAL NOT NULL DEFAULT 64", "score_retrieval_fallback": "INTEGER NOT NULL DEFAULT 0", "score_retrieval_mode": "TEXT NOT NULL DEFAULT 'warmup'", "score_exact_or_approx": "TEXT NOT NULL DEFAULT 'exact'", "score_recall_guard": "INTEGER NOT NULL DEFAULT 0", "score_valid": "INTEGER NOT NULL DEFAULT 0", "capture_started_monotonic_ns": "INTEGER NOT NULL DEFAULT 0", "capture_finished_monotonic_ns": "INTEGER NOT NULL DEFAULT 0", "capture_started": "REAL NOT NULL DEFAULT 0", "capture_finished": "REAL NOT NULL DEFAULT 0", "capture_failure_reason": "TEXT NOT NULL DEFAULT ''", "capture_hash_delta": "REAL NOT NULL DEFAULT 64", "capture_fallback": "INTEGER NOT NULL DEFAULT 0"
        }
        for name, definition in additions.items():
            if name not in frame_columns:
                self.conn.execute(f"ALTER TABLE frames ADD COLUMN {name} {definition}")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_buckets ON frames(bucket0, bucket1, bucket2, bucket3)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_cluster ON frames(state_cluster_id, state_support_count)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_session_mono ON frames(session_id, created_monotonic_ns)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_session_finished_id ON frames(session_id, capture_finished_monotonic_ns, id)")
        session_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if "trainable" not in session_columns:
            self.conn.execute("ALTER TABLE sessions ADD COLUMN trainable INTEGER NOT NULL DEFAULT 1")
        if "training_exclusion_reason" not in session_columns:
            self.conn.execute("ALTER TABLE sessions ADD COLUMN training_exclusion_reason TEXT NOT NULL DEFAULT ''")
        mouse_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(mouse_events)").fetchall()}
        if "behavior_probability" not in mouse_columns:
            self.conn.execute("ALTER TABLE mouse_events ADD COLUMN behavior_probability REAL")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mouse_session_mono ON mouse_events(session_id, created_monotonic_ns)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS mouse_loss_events (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, created REAL NOT NULL, started REAL NOT NULL, ended REAL NOT NULL, lost_count INTEGER NOT NULL, rule TEXT NOT NULL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS mouse_compression_segments (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source TEXT NOT NULL, started REAL NOT NULL, ended REAL NOT NULL, started_monotonic_ns INTEGER NOT NULL, ended_monotonic_ns INTEGER NOT NULL, start_x INTEGER NOT NULL, start_y INTEGER NOT NULL, end_x INTEGER NOT NULL, end_y INTEGER NOT NULL, original_count INTEGER NOT NULL, max_speed REAL NOT NULL, path_length REAL NOT NULL, trajectory_blob BLOB NOT NULL DEFAULT X'', trajectory_codec TEXT NOT NULL DEFAULT 'varint-zigzag-dtxy-v1', rule TEXT NOT NULL)")
        segment_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(mouse_compression_segments)").fetchall()}
        if "trajectory_blob" not in segment_columns:
            self.conn.execute("ALTER TABLE mouse_compression_segments ADD COLUMN trajectory_blob BLOB")
        if "trajectory_codec" not in segment_columns:
            self.conn.execute("ALTER TABLE mouse_compression_segments ADD COLUMN trajectory_codec TEXT NOT NULL DEFAULT 'varint-zigzag-dtxy-v1'")
        for name, definition in (("client_left", "INTEGER"), ("client_top", "INTEGER"), ("client_right", "INTEGER"), ("client_bottom", "INTEGER")):
            if name not in segment_columns:
                self.conn.execute("ALTER TABLE mouse_compression_segments ADD COLUMN {} {}".format(name, definition))
        self.conn.execute("CREATE TABLE IF NOT EXISTS deferred_exact_scores (frame_id TEXT PRIMARY KEY REFERENCES frames(id), dhash64 TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'pending', last_error TEXT NOT NULL DEFAULT '')")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_deferred_exact_scores_state ON deferred_exact_scores(state, created)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS sleep_decision_samples (id TEXT PRIMARY KEY, created REAL NOT NULL, features TEXT NOT NULL, expected_gain REAL NOT NULL DEFAULT 0, actual_quality_delta REAL, cleanup_bytes INTEGER, duration_seconds REAL, restored_training_gain REAL, training_status TEXT NOT NULL DEFAULT '', training_reason TEXT NOT NULL DEFAULT '', training_samples INTEGER NOT NULL DEFAULT 0, validation_samples INTEGER NOT NULL DEFAULT 0, outcome_ready INTEGER NOT NULL DEFAULT 0)")
        sleep_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(sleep_decision_samples)").fetchall()}
        for name, definition in (("training_status", "TEXT NOT NULL DEFAULT ''"), ("training_reason", "TEXT NOT NULL DEFAULT ''"), ("training_samples", "INTEGER NOT NULL DEFAULT 0"), ("validation_samples", "INTEGER NOT NULL DEFAULT 0")):
            if name not in sleep_columns:
                self.conn.execute("ALTER TABLE sleep_decision_samples ADD COLUMN {} {}".format(name, definition))
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
        self._sync_model_metadata_locked()
        self._recalculate_model_refs_locked([row[0] for row in self.conn.execute("SELECT id FROM frames WHERE model_dependency_count!=0 OR model_refs!=0").fetchall()])

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
        identifier = uuid.uuid4().hex
        now = time.time()
        with self.lock:
            self.conn.execute("INSERT INTO sessions(id, mode, started) VALUES (?, ?, ?)", (identifier, mode, now))
            self.conn.commit()
        return identifier

    def mark_session_untrainable(self, session_id, reason):
        with self.lock:
            if self.conn is None or not session_id:
                return
            self.conn.execute("UPDATE sessions SET trainable=0, training_exclusion_reason=? WHERE id=?", (str(reason), str(session_id)))
            self.conn.execute("INSERT INTO system_events(id, session_id, created, kind, payload) VALUES (?, ?, ?, ?, ?)", (uuid.uuid4().hex, str(session_id), time.time(), "session_forced_untrainable", json.dumps({"reason": str(reason)}, ensure_ascii=False)))
            self.conn.commit()

    def recover_after_forced_stop(self):
        try:
            self.recover_ingestions()
            self.recover_deletions()
            self._cleanup_pool_files()
            self.reconcile_pool_ledger()
            return self.validate_consistency()
        except Exception as error:
            return False, str(error)

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

    def _online_candidate_rows(self, connection, current, candidate_cap, exclude_frame_id, before_capture_finished_ns):
        candidate_cap = max(16, min(2048, int(candidate_cap)))
        parts = tuple((current >> shift) & 0xFFFF for shift in (48, 32, 16, 0))
        keys = [(index << 16) | part for index, part in enumerate(parts)]
        lsh_rows, truncated = self._lsh_candidate_rows(connection, keys, max(8, candidate_cap // 2), exclude_frame_id, before_capture_finished_ns)
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
        return rows, bool(truncated), {"lsh": len(lsh_rows), "recent": len(recent_rows), "high_value": len(value_rows)}

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

    def nearest_hashes(self, dhash, limit=8, strict=True, candidate_limit=None, deadline=None, cancelled=None, yield_if_pressure=None, force_exact=False, current_features=None, exclude_frame_id=None, before_capture_finished_ns=None):
        try:
            current = int(dhash, 16)
        except Exception:
            return {"hashes": [], "frame_ids": [], "similarities": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "invalid_hash", "exact_or_approx": "unknown", "recall_guard": False, "total_history": 0, "score_valid": False, "provisional": False}
        if before_capture_finished_ns is None:
            return {"hashes": [], "frame_ids": [], "similarities": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "missing_history_boundary", "exact_or_approx": "unknown", "recall_guard": False, "total_history": 0, "score_valid": False, "provisional": False}
        limit = max(1, int(limit))
        candidate_cap = max(limit * 8, min(2048, max(64, int(candidate_limit or 256))))
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
                rows, truncated, sources = self._online_candidate_rows(connection, current, candidate_cap, exclude_frame_id, before_capture_finished_ns)
                entries = self._frame_similarity_entries(dhash, current_features, rows, limit)
                if not entries:
                    return {"hashes": [], "frame_ids": [], "similarities": [], "candidate_count": len(rows), "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "online_candidate_empty", "exact_or_approx": "approximate", "recall_guard": False, "total_history": total, "score_valid": False, "provisional": True, "candidate_sources": sources}
                return response(entries, len(rows), "online_lsh_recent_high_value" + ("_capped" if truncated else ""), "approximate", False, True, sources)
            entries, scanned, status = self._full_exact_hashes(connection, dhash, current_features, limit, candidate_limit or 512, exclude_frame_id, before_capture_finished_ns, deadline, cancelled, yield_if_pressure)
            complete = status == "complete" and len(entries) >= min(limit, total)
            if not complete:
                return {"hashes": [item["hash"] for item in entries], "frame_ids": [item["frame_id"] for item in entries], "similarities": [float(item["similarity"]) for item in entries], "candidate_count": scanned, "top_k_distance": float(max((item["distance"] for item in entries), default=64.0)), "retrieval_fallback": True, "retrieval_mode": "sleep_exact_deferred_{}".format(status), "exact_or_approx": "unknown", "recall_guard": False, "total_history": total, "score_valid": False, "provisional": False, "candidate_sources": {"exact": scanned}}
            return response(entries, scanned, "sleep_full_exact_composite", "exact", True, False, {"exact": scanned})
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
        with self.lock:
            if self.conn is None or not session_id or count <= 0:
                return
            self.conn.execute("INSERT INTO mouse_loss_events(id, session_id, created, started, ended, lost_count, rule) VALUES (?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, session_id, time.time(), started, ended, int(count), rule))
            self.conn.commit()

    def record_mouse_compression(self, segment):
        if not segment or not segment.get("session_id"):
            raise RuntimeError("无效的无损轨迹段")
        payload = self._encode_trajectory(segment.get("points", []))
        if not payload and int(segment.get("count", 0)) > 0:
            raise RuntimeError("无损轨迹编码为空")
        rect = segment.get("client_rect") or (None, None, None, None)
        with self.lock:
            if self.conn is None:
                raise RuntimeError("存储未打开")
            self.conn.execute("INSERT INTO mouse_compression_segments(id, session_id, source, started, ended, started_monotonic_ns, ended_monotonic_ns, start_x, start_y, end_x, end_y, original_count, max_speed, path_length, trajectory_blob, trajectory_codec, rule, client_left, client_top, client_right, client_bottom) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, segment["session_id"], segment.get("source", "user"), float(segment["started"]), float(segment["ended"]), int(segment["started_ns"]), int(segment["ended_ns"]), int(segment["start_x"]), int(segment["start_y"]), int(segment["end_x"]), int(segment["end_y"]), int(segment["count"]), float(segment.get("max_speed", 0.0)), float(segment.get("path_length", 0.0)), sqlite3.Binary(payload), "varint-zigzag-dtxy-v1", str(segment.get("rule", "无损移动轨迹压缩")), *[None if value is None else int(value) for value in rect]))
            self.conn.commit()
        return True

    def record_pipeline_loss(self, session_id, started, ended, count, stage, reason):
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
        return max(0, int(total))

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
                    transient += sum(int(child.stat().st_size) for child in item.rglob("*") if child.is_file()) if item.is_dir() else int(item.stat().st_size)
                elif item.is_file():
                    other += int(item.stat().st_size)
                elif item.is_dir():
                    other += sum(int(child.stat().st_size) for child in item.rglob("*") if child.is_file())
            except OSError:
                pass
        return transient, other

    def _capacity_snapshot_locked(self):
        asset = int(self.conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM frames").fetchone()[0] or 0) if self.conn is not None else 0
        database = self._database_bytes_precise_locked()
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

    def save_frame(self, session_id, image, phash, online_score=None, exact_score=None, hunger=None, reward=None, experience_limit=None):
        if not session_id or not isinstance(image, dict) or not image.get("png"):
            raise RuntimeError("无效截图写入请求")
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
                snapshot = self._capacity_snapshot_locked()
                reservation = self._png_reservation_locked(declared_size)
                projected = snapshot["total"] + snapshot["reserved_asset_bytes"] + reservation
                usage = 0.0 if limit <= 0 else projected / float(limit)
                tier = 100 if usage >= 1.0 else 95 if usage >= 0.95 else 90 if usage >= 0.90 else 85 if usage >= 0.85 else 0
                if limit > 0 and projected > limit:
                    target = int(math.floor(limit * 0.5))
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
            folder = screens / session_id
            folder.mkdir(parents=True, exist_ok=True)
            final_path = pool / relative
            temporary = final_path.with_suffix(".tmp")
            self._inject_fault("png_write")
            with temporary.open("wb") as handle:
                handle.write(png)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(final_path)
            actual_size = int(final_path.stat().st_size)
            with self.lock:
                self._inject_fault("sqlite_write")
                self.conn.execute("BEGIN IMMEDIATE")
                self.conn.execute("UPDATE ingestion_journal SET stage='file_ready', updated=?, error='' WHERE id=?", (time.time(), journal_id))
                support = self.conn.execute("SELECT COUNT(*) + 1 FROM frames WHERE state_cluster_id=?", (state_cluster_id,)).fetchone()[0]
                score_valid = exact_score is not None and bool(image.get("score_valid")) and not bool(image.get("score_provisional"))
                online_value = float(online_score) if online_score is not None else None
                score_value = float(exact_score) if score_valid else None
                hunger_value = float(hunger) if score_valid and hunger is not None else 0.0
                reward_value = float(reward) if score_valid and reward is not None else None
                score_status = "valid" if score_valid else "pending_exact" if online_value is not None or dhash else "unknown"
                frame_values = (identifier, session_id, moment, mono, capture_started_mono, mono, capture_started_wall, moment, str(relative), phash, dhash, score_value, online_value, hunger_value, reward_value, score_value, reward_value, score_status, image["width"], image["height"], actual_size, 0.0, 0.0, 0.0, 0, moment, *buckets, state_cluster_id, support, 0.0, 0, moment, 1, image.get("capture_backend", "gdi"), image.get("capture_elapsed_ms", 0.0), image.get("capture_complete", 1), image.get("brightness", 0.0), image.get("variance", 0.0), sqlite3.Binary(feature_bytes(image.get("gray32x18"), 32 * 18)), image.get("edge_density", 0.0), sqlite3.Binary(histogram_blob(image.get("color_histogram"))), str(image.get("capture_failure_reason", "")), float(image.get("capture_hash_delta", 64.0)), 1 if image.get("capture_fallback") else 0, int(image.get("score_candidate_count", 0)), float(image.get("score_top_k_distance", 64.0)), int(image.get("score_retrieval_fallback", 0)))
                self.conn.execute("INSERT INTO frames(id, session_id, created, created_monotonic_ns, capture_started_monotonic_ns, capture_finished_monotonic_ns, capture_started, capture_finished, screenshot_path, phash, dhash64, score, online_score, hunger, reward, raw_score, raw_reward, score_status, width, height, size_bytes, novelty, action_result, coverage, model_refs, last_used, bucket0, bucket1, bucket2, bucket3, state_cluster_id, state_support_count, action_outcome_information, model_dependency_count, validation_last_used, asset_ref_count, capture_backend, capture_elapsed_ms, capture_complete, brightness, variance, gray32x18, edge_density, color_histogram, capture_failure_reason, capture_hash_delta, capture_fallback, score_candidate_count, score_top_k_distance, score_retrieval_fallback) VALUES ({})".format(",".join("?" for _ in frame_values)), frame_values)
                self.conn.execute("UPDATE frames SET score_retrieval_mode=?, score_exact_or_approx=?, score_recall_guard=?, score_valid=? WHERE id=?", (str(image.get("score_retrieval_mode", "online_pending")), "exact" if score_valid else "approximate", 1 if score_valid else 0, 1 if score_valid else 0, identifier))
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
                snapshot_after = self._capacity_snapshot_locked()
                remaining_reserved = max(0, snapshot_after["reserved_asset_bytes"] - reservation)
                final_projected = snapshot_after["total"]
                if limit > 0 and final_projected > limit:
                    raise PoolCapacityBlocked("提交前容量校验失败：实际目录与 SQLite/WAL 为 {} 字节，上限 {} 字节".format(final_projected, limit))
                self._write_ledger_locked(snapshot_after, remaining_reserved)
                usage_after = 0.0 if limit <= 0 else final_projected / float(limit)
                tier_after = 100 if usage_after >= 1.0 else 95 if usage_after >= 0.95 else 90 if usage_after >= 0.90 else 85 if usage_after >= 0.85 else 0
                self.conn.executemany("INSERT OR REPLACE INTO pool_meta(key, value) VALUES (?, ?)", (
                    ("pool_capacity_blocked", 0), ("pool_capacity_target", int(limit or 0)),
                    ("pool_capacity_remaining", int(final_projected)), ("pool_capacity_updated", int(time.time())),
                    ("pool_capacity_tier", tier_after), ("pool_capacity_transaction_reserve", int(self.transaction_reserve_bytes))))
                self.conn.execute("UPDATE ingestion_journal SET stage='complete', updated=?, error='' WHERE id=?", (time.time(), journal_id))
                self.conn.commit()
                self._database_bytes_cached = self._database_bytes_precise_locked()
                self._database_bytes_checked = time.monotonic()
                self.conn.execute("INSERT OR REPLACE INTO pool_meta(key, value) VALUES ('database_bytes', ?)", (int(self._database_bytes_cached),))
                self.conn.commit()
                reserved = False
            return identifier
        except Exception as error:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            if final_path is not None:
                try:
                    final_path.unlink(missing_ok=True)
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
        if not records:
            return
        values = []
        counts = {}
        for record in records:
            values.append((uuid.uuid4().hex, record["session_id"], record["created"], record.get("created_monotonic_ns", 0), record["source"], record["event_type"], record["button"], record["wheel"], record["x"], record["y"], record["relative_x"], record["relative_y"], record["dx"], record["dy"], record["direction"], record["speed"], record.get("behavior_probability")))
            counts[record["session_id"]] = counts.get(record["session_id"], 0) + 1
        with self.lock:
            if self.conn is None:
                return
            self.conn.executemany("INSERT INTO mouse_events(id, session_id, created, created_monotonic_ns, source, event_type, button, wheel, x, y, relative_x, relative_y, dx, dy, direction, speed, behavior_probability) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
            self.conn.executemany("UPDATE sessions SET mouse_count=mouse_count+? WHERE id=?", [(value, key) for key, value in counts.items()])
            self.conn.commit()

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
            frames = int(self.conn.execute("SELECT COUNT(*) FROM frames WHERE session_id IN (" + eligible + ") AND capture_complete=1 AND score_valid=1").fetchone()[0] or 0)
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
            row = self.conn.execute("SELECT file_name FROM model_metadata WHERE champion=1 ORDER BY validation_quality DESC, updated DESC LIMIT 1").fetchone()
            if row is None:
                row = self.conn.execute("SELECT file_name FROM model_metadata ORDER BY validation_quality DESC, quality DESC, updated DESC LIMIT 1").fetchone()
            path = self.models / str(row[0]) if row else None
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["champion"] = True
            return payload
        except Exception:
            return None

    def collect_training_data(self):
        with self.lock:
            sessions = [row[0] for row in self.conn.execute("""
                SELECT s.id FROM sessions s
                WHERE COALESCE(s.trainable, 1)=1
                  AND NOT EXISTS (SELECT 1 FROM pipeline_loss_events p WHERE p.session_id=s.id)
                  AND NOT EXISTS (SELECT 1 FROM mouse_loss_events m WHERE m.session_id=s.id AND (m.lost_count>0 OR instr(m.rule, '关键事件丢失')>0))
                  AND NOT EXISTS (SELECT 1 FROM deferred_exact_scores d JOIN frames f ON f.id=d.frame_id WHERE f.session_id=s.id AND d.state!='complete')
                ORDER BY s.started DESC LIMIT 32
            """).fetchall()]
            frames = {}
            mouse = {}
            for session_id in sessions:
                frame_rows = self.conn.execute("""
                    SELECT id, session_id, created_monotonic_ns, created, dhash64, phash, score, reward, hunger, width, height, gray32x18, edge_density, color_histogram, capture_started_monotonic_ns, capture_finished_monotonic_ns, score_valid
                    FROM frames
                    WHERE session_id=? AND capture_complete=1 AND score_valid=1 AND created_monotonic_ns>0
                    ORDER BY capture_finished_monotonic_ns ASC, id ASC
                    LIMIT 9000
                """, (session_id,)).fetchall()
                frame_rows = [tuple(list(row[:11]) + [feature_hex(row[11])] + list(row[12:13]) + [histogram_text(row[13])] + list(row[14:])) for row in frame_rows]
                mouse_rows = list(self.conn.execute("""
                    SELECT id, session_id, created_monotonic_ns, created, event_type, source, relative_x, relative_y, speed, dx, dy, direction, button, wheel, behavior_probability
                    FROM mouse_events
                    WHERE session_id=? AND created_monotonic_ns>0
                    ORDER BY created_monotonic_ns ASC, id ASC
                    LIMIT 24000
                """, (session_id,)).fetchall())
                segments = self.conn.execute("""
                    SELECT id, source, started, started_monotonic_ns, start_x, start_y, trajectory_blob, client_left, client_top, client_right, client_bottom
                    FROM mouse_compression_segments
                    WHERE session_id=? AND ended_monotonic_ns>0
                    ORDER BY ended_monotonic_ns ASC, id ASC
                    LIMIT 4096
                """, (session_id,)).fetchall()
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
                            decoded.append(("segment:{}:{}".format(segment_id, index), session_id, ns, float(started) + (ns - int(start_ns)) / 1_000_000_000.0, "move", source, rx, ry, speed, float(dx), float(dy), direction, "", 0))
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
        if not outcomes:
            return
        rows = []
        for item in outcomes:
            rows.append((item.get("action_id") or item["mouse_event_id"], uuid.uuid4().hex, item["session_id"], item["mouse_event_id"], item["before_frame_id"], item["after_frame_id"], item["action_time"], item["post_action_delay_ms"], item["score_delta"], item["reward_delta"], float(item.get("hunger_delta_expected", 0.0)), float(item.get("baseline_score_delta", 0.0)), float(item.get("expected_no_action_reward_delta", 0.0)), float(item.get("action_advantage", 0.0)), float(item.get("stability", 0.0)), int(item.get("baseline_count", 0)), 1 if item.get("outcome_valid") else 0, item.get("split_role", "unknown")))
        with self.lock:
            self.conn.executemany("INSERT INTO action_outcomes(action_id, id, session_id, mouse_event_id, before_frame_id, after_frame_id, action_time, post_action_delay_ms, score_delta, reward_delta, hunger_delta_expected, baseline_score_delta, expected_no_action_reward_delta, action_advantage, stability, baseline_count, outcome_valid, split_role) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(action_id) DO UPDATE SET session_id=excluded.session_id, mouse_event_id=excluded.mouse_event_id, before_frame_id=excluded.before_frame_id, after_frame_id=excluded.after_frame_id, action_time=excluded.action_time, post_action_delay_ms=excluded.post_action_delay_ms, score_delta=excluded.score_delta, reward_delta=excluded.reward_delta, hunger_delta_expected=excluded.hunger_delta_expected, baseline_score_delta=excluded.baseline_score_delta, expected_no_action_reward_delta=excluded.expected_no_action_reward_delta, action_advantage=excluded.action_advantage, stability=excluded.stability, baseline_count=excluded.baseline_count, outcome_valid=excluded.outcome_valid, split_role=excluded.split_role", rows)
            self.conn.executemany("UPDATE mouse_events SET before_frame_id=?, after_frame_id=?, action_time=?, post_action_delay_ms=?, score_delta=?, reward_delta=?, outcome_valid=? WHERE id=?", [(item["before_frame_id"], item["after_frame_id"], item["action_time"], item["post_action_delay_ms"], item["score_delta"], item["reward_delta"], 1 if item.get("outcome_valid") else 0, item["mouse_event_id"]) for item in outcomes])
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
        rows = self.conn.execute("SELECT id FROM model_metadata ORDER BY champion DESC, validation_quality DESC, quality DESC, updated DESC, id DESC").fetchall()
        if rows:
            winner = str(rows[0][0])
            self.conn.execute("UPDATE model_metadata SET champion=CASE WHEN id=? THEN 1 ELSE 0 END", (winner,))

    def sync_model_metadata(self):
        with self.lock:
            if self.conn is None:
                return
            self._sync_model_metadata_locked()
            self.conn.commit()

    def register_model_metadata(self, model_id, path, payload, outcomes=None, validation_outcomes=None):
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
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.executemany("INSERT OR IGNORE INTO model_frame_refs(model_id, frame_id, role) VALUES (?, ?, ?)", refs)
            self._recalculate_model_refs_locked([item[1] for item in refs])
            self.conn.commit()

    def record_sleep_decision(self, features, expected_gain):
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
            self.conn.commit()

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

    def _recalculate_session_scores_locked(self, session_id):
        rows = self.conn.execute("""
            SELECT id, capture_finished_monotonic_ns, score, score_valid, hunger
            FROM frames WHERE session_id=?
            ORDER BY capture_finished_monotonic_ns ASC, id ASC
        """, (session_id,)).fetchall()
        last_score = None
        anchor_ns = None
        updates = []
        for frame_id, finished_ns, score, score_valid, prior_hunger in rows:
            finished_ns = int(finished_ns or 0)
            if not score_valid or score is None:
                continue
            if anchor_ns is None:
                inferred = finished_ns - int(max(0.0, float(prior_hunger or 0.0) - 1e-9) * 1_000_000_000.0 / 0.00004)
                anchor_ns = max(0, inferred)
            hunger = 1e-9 + max(0, finished_ns - anchor_ns) * 0.00004 / 1_000_000_000.0
            if last_score is not None and float(score) > last_score:
                hunger = 1e-9
                anchor_ns = finished_ns
            reward = float(score) - hunger
            updates.append((hunger, reward, float(score), reward, frame_id))
            last_score = float(score)
        if updates:
            self.conn.executemany("UPDATE frames SET hunger=?, reward=?, raw_score=?, raw_reward=? WHERE id=?", updates)
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
            SELECT f.id, f.state_support_count, f.reward, f.capture_complete, f.variance,
                   f.validation_last_used, f.model_dependency_count, f.model_refs,
                   f.created, f.session_id,
                   COALESCE((SELECT MAX(ABS(a.action_advantage)) FROM action_outcomes a WHERE a.before_frame_id=f.id OR a.after_frame_id=f.id), 0),
                   COALESCE((SELECT COUNT(*) FROM action_outcomes a WHERE a.before_frame_id=f.id OR a.after_frame_id=f.id), 0),
                   COALESCE((SELECT COUNT(*) FROM frames later WHERE later.session_id=f.session_id AND later.dhash64=f.dhash64 AND later.created>f.created), 0)
            FROM frames f
        """ + where, params).fetchall()
        now = time.time()
        updates = []
        for frame_id, support, reward, complete, variance, validation_used, dependency_count, model_refs, created, session_id, action_info, action_refs, duplicates in rows:
            rarity = 1.0 / max(1.0, float(support or 1))
            action_value = min(0.85, abs(float(action_info or 0.0)) * 2.0 + min(0.25, float(action_refs or 0) * 0.08))
            model_value = min(0.90, float(dependency_count or 0) * 0.20 + float(model_refs or 0) * 0.10)
            validation_value = 0.35 if float(validation_used or 0.0) > 0 else 0.0
            coverage_value = 0.20 if float(created or 0.0) <= now - 60.0 else 0.10
            reward_value = min(0.45, abs(float(reward or 0.0)))
            quality_value = (0.20 if int(complete or 0) else 0.0) + min(0.15, max(0.0, float(variance or 0.0) / 2550.0))
            duplicate_penalty = min(0.70, float(duplicates or 0) * 0.08)
            retain = max(0.0, rarity + action_value + model_value + validation_value + coverage_value + reward_value + quality_value - duplicate_penalty)
            updates.append((retain, frame_id))
        if updates:
            self.conn.executemany("UPDATE frames SET retain_value=?, retain_version=2 WHERE id=?", updates)

    def session_score_summary(self, session_id):
        with self.lock:
            if self.conn is None or not session_id:
                return {"valid_frames": 0, "scores": [], "latest": None}
            count = int(self.conn.execute("SELECT COUNT(*) FROM frames WHERE session_id=? AND score_valid=1 AND score IS NOT NULL", (session_id,)).fetchone()[0] or 0)
            scores = [float(row[0]) for row in self.conn.execute("SELECT score FROM frames WHERE session_id=? AND score_valid=1 AND score IS NOT NULL ORDER BY capture_finished_monotonic_ns DESC, id DESC LIMIT 120", (session_id,)).fetchall()]
            latest = self.conn.execute("SELECT id, score, hunger, reward, capture_finished_monotonic_ns FROM frames WHERE session_id=? AND score_valid=1 AND score IS NOT NULL ORDER BY capture_finished_monotonic_ns DESC, id DESC LIMIT 1", (session_id,)).fetchone()
        scores.reverse()
        return {"valid_frames": count, "scores": scores, "latest": latest}

    def process_deferred_exact_scores(self, cancelled=None, cooperative=None, maximum=1, session_id=None):
        if not self.exact_score_lock.acquire(blocking=False):
            return 0
        try:
            with self.lock:
                if self.conn is None:
                    return 0
                predicate = ""
                params = []
                if session_id:
                    predicate = " AND f.session_id=?"
                    params.append(str(session_id))
                rows = self.conn.execute(
                    "SELECT d.frame_id, d.dhash64 FROM deferred_exact_scores d JOIN frames f ON f.id=d.frame_id "
                    "WHERE d.state='pending'{} ORDER BY f.capture_finished_monotonic_ns ASC, f.id ASC LIMIT ?".format(predicate),
                    tuple(params + [max(1, int(maximum))]),
                ).fetchall()
            resolved = 0
            for frame_id, dhash in rows:
                if cancelled is not None and cancelled():
                    break
                with self.lock:
                    feature_row = self.conn.execute("SELECT session_id, phash, gray32x18, edge_density, color_histogram, capture_finished_monotonic_ns FROM frames WHERE id=?", (frame_id,)).fetchone()
                if feature_row is None:
                    with self.lock:
                        self.conn.execute("DELETE FROM deferred_exact_scores WHERE frame_id=?", (frame_id,))
                        self.conn.commit()
                    continue
                session_key, phash, gray, edge, histogram, finished_ns = feature_row
                current_features = {"phash": phash, "gray32x18": gray, "edge_density": edge, "color_histogram": histogram}
                result = self.nearest_hashes(dhash, 8, candidate_limit=512, deadline=None, cancelled=cancelled, yield_if_pressure=cooperative, force_exact=True, current_features=current_features, exclude_frame_id=frame_id, before_capture_finished_ns=int(finished_ns or 0))
                score, meta = frame_score(dhash, result, current_features)
                if not meta.get("score_valid") or score is None:
                    with self.lock:
                        if str(meta.get("retrieval_mode", "")) == "warmup_no_history":
                            self.conn.execute("UPDATE frames SET score_status='warmup_exact', score_valid=0, score_retrieval_mode='warmup_no_history', score_exact_or_approx='exact', score_recall_guard=1 WHERE id=?", (frame_id,))
                            self.conn.execute("UPDATE deferred_exact_scores SET state='complete', updated=?, attempts=attempts+1, last_error='warmup_no_history' WHERE frame_id=?", (time.time(), frame_id))
                        else:
                            self.conn.execute("UPDATE deferred_exact_scores SET attempts=attempts+1, updated=?, state=CASE WHEN attempts+1>=3 THEN 'failed' ELSE 'pending' END, last_error=? WHERE frame_id=?", (time.time(), str(meta.get("retrieval_mode", "deferred")), frame_id))
                        self.conn.commit()
                    continue
                with self.lock:
                    row = self.conn.execute("SELECT id FROM frames WHERE id=?", (frame_id,)).fetchone()
                    if row is None:
                        self.conn.execute("DELETE FROM deferred_exact_scores WHERE frame_id=?", (frame_id,))
                    else:
                        self.conn.execute("BEGIN IMMEDIATE")
                        self.conn.execute("UPDATE frames SET score=?, raw_score=?, score_status='valid', score_valid=1, score_retrieval_mode=?, score_exact_or_approx='exact', score_recall_guard=1 WHERE id=?", (float(score), float(score), str(meta.get("retrieval_mode", "sleep_full_exact_composite")), frame_id))
                        self._recalculate_session_scores_locked(session_key)
                        self.conn.execute("UPDATE deferred_exact_scores SET state='complete', updated=?, attempts=attempts+1, last_error='' WHERE frame_id=?", (time.time(), frame_id))
                        self.conn.commit()
                        resolved += 1
            return resolved
        finally:
            self.exact_score_lock.release()

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
            self.conn.commit()
        return sessions

    def _ledger_values_locked(self):
        keys = ("asset_bytes", "reserved_asset_bytes", "database_bytes", "transient_bytes", "other_bytes", "last_reconciled_at")
        rows = dict(self.conn.execute("SELECT key, value FROM pool_meta WHERE key IN ({})".format(",".join("?" for _ in keys)), keys).fetchall())
        return {key: max(0, int(rows.get(key, 0) or 0)) for key in keys}

    def reconcile_pool_ledger(self):
        result = {"frame_asset_bytes": 0, "reserved_asset_bytes": 0, "database_bytes": 0, "transient_bytes": 0, "other_bytes": 0, "experience_total_bytes": 0, "last_reconciled_at": int(time.time())}
        if self.pool is None:
            return result
        try:
            for item in self.pool.rglob("*"):
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



    def _cleanup_pool_files(self):
        if self.pool is None:
            return
        referenced = set()
        with self.lock:
            if self.conn is not None:
                referenced = {str(row[0]) for row in self.conn.execute("SELECT screenshot_path FROM frames").fetchall()}
        for item in list(self.pool.rglob("*.tmp")):
            try:
                item.unlink(missing_ok=True)
            except OSError:
                pass
        if self.screens and self.screens.exists():
            for item in self.screens.rglob("*.png"):
                try:
                    if str(item.relative_to(self.pool)) not in referenced:
                        item.unlink(missing_ok=True)
                except OSError:
                    pass
        trash = self.pool / "trash"
        if trash.exists():
            for item in trash.rglob("*"):
                try:
                    if item.is_file():
                        item.unlink(missing_ok=True)
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

    def prune_models(self, maximum, cancelled=None, cooperative=None):
        maximum = max(1, int(maximum))
        self.sync_model_metadata()
        with self.lock:
            rows = self.conn.execute("""
                SELECT m.id, m.file_name, m.validation_quality, m.quality, m.champion, m.updated,
                       COALESCE((SELECT COUNT(DISTINCT frame_id) FROM model_frame_refs r WHERE r.model_id=m.id), 0)
                FROM model_metadata m
                ORDER BY m.champion ASC, m.validation_quality ASC, m.quality ASC, 7 ASC, m.updated ASC, m.id ASC
            """).fetchall()
        initial = len(rows)
        if initial <= maximum:
            self.last_model_prune_result = {"initial": initial, "removed": 0, "target": initial, "remaining": initial, "success": True}
            return 0
        target = max(1, int(math.floor(maximum * 0.5)))
        removed = 0
        trash_root = self.models / ".trash"
        trash_root.mkdir(parents=True, exist_ok=True)
        for index, (model_id, file_name, validation_quality, quality, champion, updated, ref_count) in enumerate(rows):
            if initial - removed <= target:
                break
            if int(champion):
                continue
            if cancelled is not None and cancelled():
                break
            if index % 8 == 0 and cooperative is not None and not cooperative():
                break
            path = self.models / str(file_name)
            trash = trash_root / (uuid.uuid4().hex + ".json")
            try:
                path.replace(trash)
            except OSError:
                continue
            try:
                with self.lock:
                    self.conn.execute("BEGIN IMMEDIATE")
                    frame_ids = [row[0] for row in self.conn.execute("SELECT DISTINCT frame_id FROM model_frame_refs WHERE model_id=?", (model_id,)).fetchall()]
                    self.conn.execute("DELETE FROM model_frame_refs WHERE model_id=?", (model_id,))
                    self.conn.execute("DELETE FROM model_metadata WHERE id=? AND champion=0", (model_id,))
                    self._recalculate_model_refs_locked(frame_ids)
                    self.conn.commit()
                trash.unlink(missing_ok=True)
                removed += 1
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
            remaining = int(self.conn.execute("SELECT COUNT(*) FROM model_metadata").fetchone()[0] or 0)
            champions = int(self.conn.execute("SELECT COUNT(*) FROM model_metadata WHERE champion=1").fetchone()[0] or 0)
        success = remaining <= target and champions == 1
        self.last_model_prune_result = {"initial": initial, "removed": removed, "target": target, "remaining": remaining, "success": success}
        if not success and not (cancelled is not None and cancelled()):
            self.add_system_event(None, "model_prune_incomplete", dict(self.last_model_prune_result))
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
            for row in rows:
                identifier, stored, size_bytes = row[:3]
                retain_value = float(row[3] or 0.0) if len(row) > 3 else 0.0
                deletion_reason = str(row[4]) if len(row) > 4 else "retention"
                journal_id = uuid.uuid4().hex
                self.conn.execute("INSERT INTO deletion_journal(id, object_type, object_id, path, stage, created, updated, error) VALUES (?, 'frame', ?, ?, 'pending', ?, ?, '')", (journal_id, identifier, stored, now, now))
                self.conn.execute("INSERT INTO system_events(id, session_id, created, kind, payload) SELECT ?, session_id, ?, 'frame_pruned', ? FROM frames WHERE id=?", (uuid.uuid4().hex, now, json.dumps({"reason": deletion_reason, "retain_value": retain_value, "retain_version": 1}, ensure_ascii=False), identifier))
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
                self.conn.execute("INSERT OR REPLACE INTO pool_meta(key, value) VALUES ('asset_bytes', MAX(0, COALESCE((SELECT value FROM pool_meta WHERE key='asset_bytes'), 0) - ?))", (size_bytes,))
                self.conn.execute("INSERT OR REPLACE INTO pool_meta(key, value) VALUES ('total_asset_bytes', COALESCE((SELECT value FROM pool_meta WHERE key='asset_bytes'), 0))")
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
                    path.unlink(missing_ok=True)
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
            for item in self.screens.rglob("*.png"):
                try:
                    rel = str(item.relative_to(self.pool))
                    if rel not in referenced:
                        item.unlink(missing_ok=True)
                except OSError:
                    pass
        self.reconcile_pool_ledger()

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
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                freelist = int(self.conn.execute("PRAGMA freelist_count").fetchone()[0] or 0)
                while freelist > 0:
                    if cooperative is not None and not cooperative():
                        break
                    self.conn.execute("PRAGMA incremental_vacuum({})".format(min(4096, freelist)))
                    freelist = int(self.conn.execute("PRAGMA freelist_count").fetchone()[0] or 0)
                self.conn.commit()
            except sqlite3.Error:
                pass
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

    def prune_experience(self, maximum, cancelled, progress, cooperative=None):
        maximum = max(1, int(maximum))
        target_total = int(math.floor(maximum * 0.5))
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
            self._set_capacity_status(True, target_total, initial)
            self.last_experience_prune_result = {"initial": initial, "removed": 0, "target": target_total, "asset_target": asset_target, "remaining": initial, "metadata_bytes": initial_metadata, "success": False, "reason": "SQLite/WAL 文件超过经验池目标；已 checkpoint 与增量 vacuum，仍无法缩小"}
            self.add_system_event(None, "pool_prune_blocked_database", dict(self.last_experience_prune_result))
            return 0, initial
        if initial <= maximum:
            self._set_capacity_status(False, target_total, initial)
            self.last_experience_prune_result = {"initial": initial, "removed": 0, "target": target_total, "asset_target": asset_target, "remaining": initial, "metadata_bytes": initial_metadata, "success": initial <= maximum}
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
                        (CASE WHEN COALESCE(state_clusters.count, frames.state_support_count, 1)>1 THEN 1 ELSE 0 END) DESC,
                        frames.retain_value ASC, frames.created ASC
                        LIMIT 256""".format(predicate), params()).fetchall()
                if not rows:
                    break
                removed += self._delete_frame_batch(rows)
                with self.lock:
                    current_assets = self._asset_bytes_locked()
                    remaining_clusters = {str(row[0]) for row in self.conn.execute("SELECT DISTINCT state_cluster_id FROM frames WHERE state_cluster_id IS NOT NULL AND state_cluster_id!=''").fetchall()}
                loss = max(0, len(initial_clusters) - len(remaining_clusters))
                ratio = loss / max(1, len(initial_clusters))
                self.last_prune_coverage_loss = {"before": len(initial_clusters), "after": len(remaining_clusters), "loss": loss, "ratio": ratio, "paused": ratio > 0.10}
                if ratio > 0.10:
                    self.add_system_event(None, "pool_prune_paused_coverage_loss", dict(self.last_prune_coverage_loss))
                    break
                span = max(0.0, progress_end - progress_start)
                progress(min(progress_end, progress_start + span * min(1.0, max(0.0, initial_assets - current_assets) / max(1, initial_assets - asset_target))))
            return current_assets <= asset_target

        delete_until(
            "frames.model_dependency_count=0 AND frames.model_refs=0 AND frames.asset_ref_count<=1 AND COALESCE(state_clusters.count, frames.state_support_count, 1)>1 AND frames.validation_last_used<?",
            lambda: (time.time() - 3600.0,),
            56.0,
            70.0,
        )
        if current_assets > asset_target and not cancelled() and not self.last_prune_coverage_loss.get("paused"):
            delete_until("frames.model_dependency_count=0 AND frames.model_refs=0 AND COALESCE(state_clusters.count, frames.state_support_count, 1)>1 AND frames.retain_value<1.35", lambda: (), 70.0, 82.0)
        if current_assets > asset_target and not cancelled() and not self.last_prune_coverage_loss.get("paused"):
            self._release_model_frame_refs(False)
            delete_until("frames.model_dependency_count=0 AND frames.model_refs=0 AND COALESCE(state_clusters.count, frames.state_support_count, 1)>1", lambda: (), 82.0, 90.0)
        if current_assets > asset_target and not cancelled() and not self.last_prune_coverage_loss.get("paused"):
            self._release_model_frame_refs(True)
            delete_until("COALESCE(state_clusters.count, frames.state_support_count, 1)>1", lambda: (), 90.0, 95.0)
        self._cleanup_pool_files()
        self._prune_metadata_before_assets(cooperative)
        with self.lock:
            current_assets = self._asset_bytes_locked()
        self._compact_database(cooperative)
        metadata = self._database_bytes_now()
        remaining = self.pool_size()
        success = remaining <= target_total
        self.last_experience_prune_result = {"initial": initial, "removed": removed, "target": target_total, "asset_target": max(0, target_total - metadata), "remaining": remaining, "asset_bytes": current_assets, "metadata_bytes": metadata, "coverage_loss": dict(self.last_prune_coverage_loss), "success": success and not self.last_prune_coverage_loss.get("paused")}
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

def send_ai_mouse(x, y, flags, marker, absolute=True):
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
    with AI_INPUT_SERIAL_LOCK:
        down = send_ai_mouse(0, 0, MOUSEEVENTF_LEFTDOWN, down_marker, False)
        up = send_ai_mouse(0, 0, MOUSEEVENTF_LEFTUP, up_marker, False)
        return down and up

def ai_right_click(down_marker, up_marker):
    with AI_INPUT_SERIAL_LOCK:
        down = send_ai_mouse(0, 0, MOUSEEVENTF_RIGHTDOWN, down_marker, False)
        up = send_ai_mouse(0, 0, MOUSEEVENTF_RIGHTUP, up_marker, False)
        return down and up

def ai_wheel(delta, marker, horizontal=False):
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

def window_is_transparent(hwnd):
    try:
        return bool(int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE)) & WS_EX_TRANSPARENT)
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
    return "window_overlap"

def client_unobscured(hwnd, rect):
    own_root = root_window(hwnd)
    if not own_root:
        return False
    client_unobscured.last_obstruction = None
    client_unobscured.last_overlay = None
    points = ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), (rect[0] + 2, rect[1] + 2), (rect[2] - 2, rect[1] + 2), (rect[0] + 2, rect[3] - 2), (rect[2] - 2, rect[3] - 2)
    for x, y in points:
        hit = WindowFromPoint(POINT(int(x), int(y)))
        if hit and root_window(hit) != own_root:
            pid = wintypes.DWORD(); user32.GetWindowThreadProcessId(hit, ctypes.byref(pid))
            details = {"kind": window_obstruction_kind(hit), "title": window_title(hit), "pid": int(pid.value), "point": (int(x), int(y)), "rect": window_rectangle(hit)}
            client_unobscured.last_obstruction = details
            return False
    above = user32.GetWindow(own_root, GW_HWNDPREV)
    checked = set()
    while above and above not in checked:
        checked.add(above)
        if root_window(above) != own_root and user32.IsWindowVisible(above) and not user32.IsIconic(above):
            candidate = window_rectangle(above)
            if candidate is not None and rectangle_overlap(candidate, rect):
                pid = wintypes.DWORD(); user32.GetWindowThreadProcessId(above, ctypes.byref(pid))
                details = {"kind": window_obstruction_kind(above), "title": window_title(above), "pid": int(pid.value), "rect": candidate}
                client_unobscured.last_obstruction = details
                return False
        above = user32.GetWindow(above, GW_HWNDPREV)
    return True

def valid_client(hwnd, require_cursor=True):
    valid_client.last_reason = ""
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

def find_emulator_candidates(executable):
    selected = normalized_windows_path(executable)
    candidates = []
    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        pid = wintypes.DWORD(); user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if normalized_windows_path(process_full_path(int(pid.value))) != selected:
            return True
        rect = client_rect(hwnd)
        if rect is None:
            return True
        candidates.append({"hwnd": hwnd, "pid": int(pid.value), "title": window_title(hwnd), "rect": rect, "area": (rect[2] - rect[0]) * (rect[3] - rect[1])})
        return True
    user32.EnumWindows(EnumWindowsProc(callback), 0)
    return sorted(candidates, key=lambda item: (item["title"].lower(), item["pid"], -item["area"]))

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

def encode_png(width, height, rgb):
    rows = bytearray()
    row_size = width * 3
    for offset in range(0, len(rgb), row_size):
        rows.append(0)
        rows.extend(rgb[offset:offset + row_size])
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + png_chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + png_chunk(b"IEND", b"")

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
        info = BITMAPINFO(); info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER); info.bmiHeader.biWidth = width; info.bmiHeader.biHeight = height; info.bmiHeader.biPlanes = 1; info.bmiHeader.biBitCount = 32; info.bmiHeader.biCompression = BI_RGB
        raw = (ctypes.c_ubyte * (width * height * 4))()
        if gdi32.GetDIBits(memory_dc, bitmap, 0, height, ctypes.byref(raw), ctypes.byref(info), DIB_RGB_COLORS) != height:
            return None
        rgb = bytearray(width * height * 3)
        for y in range(height):
            source_row = (height - 1 - y) * width * 4; output_row = y * width * 3
            for x in range(width):
                src = source_row + x * 4; dst = output_row + x * 3
                rgb[dst] = raw[src + 2]; rgb[dst + 1] = raw[src + 1]; rgb[dst + 2] = raw[src]
        finished_ns = time.monotonic_ns(); finished = time.time()
        return {"width": width, "height": height, "rgb": bytes(rgb), "capture_started_monotonic_ns": capture_started_monotonic_ns, "capture_finished_monotonic_ns": finished_ns, "capture_started": capture_started, "capture_finished": finished, "capture_backend": "gdi", "capture_elapsed_ms": (finished_ns-capture_started_monotonic_ns)/1000000.0, "capture_fallback": 0, "capture_failure_reason": ""}
    finally:
        if old_object and memory_dc: gdi32.SelectObject(memory_dc, old_object)
        if bitmap: gdi32.DeleteObject(bitmap)
        if memory_dc: gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(hwnd, source_dc)


def capture_looks_invalid(image):
    if not image or not image.get("rgb"):
        return True
    rgb = image["rgb"]
    sample = rgb[::max(1, len(rgb) // 4096)]
    if not sample:
        return True
    mean = sum(sample) / len(sample)
    variance = sum((value - mean) ** 2 for value in sample) / len(sample)
    return mean < 3.0 or variance < 2.0

def capture_validation(hwnd):
    rect = valid_client(hwnd, False)
    if rect is None:
        return None
    root = root_window(hwnd)
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not root or not pid.value:
        return None
    return {"hwnd": int(hwnd), "root": int(root), "pid": int(pid.value), "rect": tuple(int(value) for value in rect)}

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
        info.bmiHeader.biHeight = height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB
        raw = (ctypes.c_ubyte * (width * height * 4))()
        if gdi32.GetDIBits(memory_dc, bitmap, 0, height, ctypes.byref(raw), ctypes.byref(info), DIB_RGB_COLORS) != height:
            return None
        rgb = bytearray(width * height * 3)
        for y in range(height):
            source_row = (height - 1 - y) * width * 4
            output_row = y * width * 3
            for x in range(width):
                src = source_row + x * 4
                dst = output_row + x * 3
                rgb[dst] = raw[src + 2]
                rgb[dst + 1] = raw[src + 1]
                rgb[dst + 2] = raw[src]
        validation_after = capture_validation(hwnd)
        if not same_capture_validation(before, validation_after):
            return None
        finished_ns = time.monotonic_ns()
        finished = time.time()
        return {"width": width, "height": height, "rgb": bytes(rgb), "capture_started_monotonic_ns": capture_started_monotonic_ns, "capture_finished_monotonic_ns": finished_ns, "capture_started": capture_started, "capture_finished": finished, "capture_backend": "desktop", "capture_elapsed_ms": (finished_ns-capture_started_monotonic_ns)/1000000.0, "capture_fallback": 1, "capture_failure_reason": failure_reason, "validation_before": before, "validation_after": validation_after, "capture_generation": int(capture_generation)}
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
    generation = time.monotonic_ns()
    before = capture_validation(hwnd)
    if before is None:
        capture_client.last_failure_reason = getattr(valid_client, "last_reason", "截图前窗口校验失败")
        return None
    image = None if use_fallback else _capture_client_gdi(hwnd, max_width, max_height)
    reason = "fallback_requested" if use_fallback else "gdi_failed"
    if image is not None and capture_looks_invalid(image):
        reason = "gdi_black_or_low_variance"
        image = None
    if image is not None:
        after = capture_validation(hwnd)
        if not same_capture_validation(before, after):
            capture_client.last_failure_reason = "GDI 截图期间客户区、绑定对象或遮挡状态变化"
            return None
        image.update({"validation_before": before, "validation_after": after, "capture_generation": int(generation)})
        return image
    image = _capture_client_desktop(hwnd, max_width, max_height, reason, before, generation)
    if image is None:
        capture_client.last_failure_reason = "桌面回退截图前后校验失败或客户区被遮挡"
    return image

def extract_frame_features(image):
    width = int(image["width"]); height = int(image["height"]); rgb = image["rgb"]
    grayscale = [[0] * 9 for _ in range(8)]
    for y in range(8):
        sy = min(height - 1, int((y + 0.5) * height / 8))
        for x in range(9):
            sx = min(width - 1, int((x + 0.5) * width / 9)); index = (sy * width + sx) * 3
            grayscale[y][x] = (rgb[index] * 299 + rgb[index + 1] * 587 + rgb[index + 2] * 114) // 1000
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (1 if grayscale[y][x] > grayscale[y][x + 1] else 0)
    lowfreq = [grayscale[y][x] for y in range(8) for x in range(8)]; low_mean = sum(lowfreq) / max(1, len(lowfreq)); perceptual = 0
    for item in lowfreq: perceptual = (perceptual << 1) | (1 if item >= low_mean else 0)
    mean = sum(rgb) / max(1, len(rgb)); sample = rgb[::max(1, len(rgb)//4096)]; variance = sum((v-mean)**2 for v in sample) / max(1, len(sample))
    sample_gray=[]
    for gy in range(18):
        sy=min(height-1,int((gy+0.5)*height/18))
        for gx in range(32):
            sx=min(width-1,int((gx+0.5)*width/32)); index=(sy*width+sx)*3; sample_gray.append((rgb[index]*299+rgb[index+1]*587+rgb[index+2]*114)//1000)
    edge_hits=0
    for gy in range(18):
        for gx in range(31):
            if abs(sample_gray[gy*32+gx]-sample_gray[gy*32+gx+1]) > 24: edge_hits += 1
    hist=[0]*24; step=max(1,len(rgb)//12288)
    for index in range(0,len(rgb),3*step):
        hist[min(7,rgb[index]//32)]+=1; hist[8+min(7,rgb[index+1]//32)]+=1; hist[16+min(7,rgb[index+2]//32)]+=1
    image.update({"phash": f"{perceptual:016x}", "dhash64": f"{value:016x}", "capture_complete": 1 if mean >= 3.0 and variance >= 2.0 else 0, "brightness": mean, "variance": variance, "gray32x18": bytes(sample_gray), "edge_density": edge_hits/(18*31), "color_histogram": struct.pack("<24I", *hist)})
    return image

def compress_frame_png(image):
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
    similarity = sum(value * weight for value, weight in weighted) / max(1e-9, sum(weight for _, weight in weighted))
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
                    return self.pending.pop(index)
        return None

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
        if self.thread and self.thread is not threading.current_thread():
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
                    self.sink(event_type, button, wheel, int(info.pt.x), int(info.pt.y), time.time(), time.monotonic_ns(), int(info.flags), int(info.dwExtraInfo))
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
        if self.thread and self.thread is not threading.current_thread():
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

class WindowEventGuard:
    def __init__(self, event_sink):
        self.event_sink = event_sink
        self.lock = threading.Lock()
        self.target_root = 0
        self.hooks = []
        self.last_emit = 0.0
        self.callback = WinEventProc(self._callback)

    def _callback(self, hook, event, hwnd, object_id, child_id, event_thread, event_time):
        with self.lock:
            target_root = self.target_root
        if not hwnd or not target_root:
            return
        try:
            now = time.monotonic()
            with self.lock:
                if now - self.last_emit < 0.05:
                    return
                if int(root_window(hwnd) or 0) != int(target_root):
                    return
                self.last_emit = now
            self.event_sink()
        except Exception:
            pass

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
            self.target_root = int(target_root)
            self.hooks = hooks
        return bool(hooks)

    def stop(self):
        with self.lock:
            hooks = list(self.hooks)
            self.hooks = []
            self.target_root = 0
        for hook in hooks:
            try:
                user32.UnhookWinEvent(hook)
            except Exception:
                pass

class Controller:
    def __init__(self, settings, event_sink):
        self.settings = settings
        self.event_sink = event_sink
        self.store = DataStore()
        self.resources = ResourceGovernor()
        self.lock = threading.RLock()
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
        self.action_limits = {}
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
        self.mouse_queue = queue.Queue(maxsize=12000)
        self.raw_mouse_queue = queue.Queue(maxsize=16000)
        self.raw_critical_queue = queue.Queue(maxsize=2048)
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
        self.writer = threading.Thread(target=self._mouse_writer, name="MouseWriter")
        self.writer.start()
        self.raw_mouse_thread = threading.Thread(target=self._raw_mouse_loop, name="MouseParser")
        self.raw_mouse_thread.start()
        self.control_thread = threading.Thread(target=self._control_loop, name="SessionControl")
        self.control_thread.start()
        self.hook = MouseHook(self.enqueue_raw_mouse)
        self.keyboard_hook = KeyboardHook(self.on_control_signal)
        self.window_guard = WindowEventGuard(self._on_window_event)
        self.capture_threads = []
        self.loss_lock = threading.Lock()
        self.move_loss = {}
        self.ai_input_tracker = AIInputTracker()

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

    def _on_window_event(self):
        with self.lock:
            token = self.epoch
            active = self.state in ("learning", "training")
        if active:
            self.on_control_signal("window_validate", "窗口状态变化", token=token)

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
        while not self.raw_mouse_stop.is_set() or not self.raw_critical_queue.empty() or not self.raw_mouse_queue.empty():
            self._flush_raw_mouse_losses()
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
            write_budget = self.resources.acquire("capture")
            if pending and (len(pending) >= max(8, write_budget.database_batch_size) or time.monotonic() - last_write >= max(0.15, write_budget.next_interval) or self.writer_stop.is_set()):
                self.writer_busy.set()
                try:
                    self.store.save_mouse_batch(pending)
                except Exception as error:
                    sessions = {record.get("session_id") for record in pending if record.get("session_id")}
                    for sid in sessions:
                        self.on_control_signal("stop", "鼠标轨迹持久化失败，已安全结束会话:" + str(error))
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

    def classify_mouse_source(self, event_type, button, wheel, x, y, flags, extra_info, created_monotonic_ns):
        injected = bool(flags & (LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED))
        matched = self.ai_input_tracker.consume(extra_info, event_type, button, wheel, x, y, created_monotonic_ns)
        probability = matched.get("behavior_probability") if isinstance(matched, dict) else None
        if injected and matched:
            return "ai", probability
        if injected:
            return "external_injected", None
        return "user", None

    def _append_move_compression(self, session_id, source, last_kept, x, y, created, created_ns, speed, rect):
        key = (session_id, source)
        item = self.move_segments.get(key)
        if item is None:
            base_x, base_y, base_ns, base_created = int(last_kept[0]), int(last_kept[1]), int(last_kept[2]), float(last_kept[3])
            dt, dx, dy = max(0, int(created_ns) - base_ns), int(x) - base_x, int(y) - base_y
            item = {"session_id": session_id, "source": source, "started": base_created, "ended": created, "started_ns": base_ns, "ended_ns": int(created_ns), "start_x": base_x, "start_y": base_y, "end_x": int(x), "end_y": int(y), "count": 1, "max_speed": float(speed), "path_length": math.hypot(dx, dy), "points": [(dt, dx, dy)], "client_rect": tuple(int(value) for value in rect), "rule": "varint/zigzag 无损相对时间与坐标轨迹压缩"}
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
            item["max_speed"] = max(float(item["max_speed"]), float(speed))
            item["points"].append((dt, dx, dy))

    def _flush_move_compression(self, session_id=None, source=None):
        keys = [key for key in self.move_segments if (session_id is None or key[0] == session_id) and (source is None or key[1] == source)]
        ok = True
        for key in keys:
            segment = self.move_segments.pop(key, None)
            if segment:
                try:
                    self.store.record_mouse_compression(segment)
                except Exception as error:
                    ok = False
                    self.on_control_signal("stop", "无损鼠标轨迹无法持久化，已安全结束会话:" + str(error))
        return ok

    def on_mouse(self, event_type, button, wheel, x, y, created, created_monotonic_ns, flags, extra_info):
        source, behavior_probability = self.classify_mouse_source(event_type, button, wheel, x, y, flags, extra_info, created_monotonic_ns)
        if source == "user":
            self.resources.update_user_input()
        with self.lock:
            if self.state not in ("learning", "training") or not self.session_id or not self.target_rect:
                return
            session_id = self.session_id
            rect = self.target_rect
            outside = not point_inside((x, y), rect)
            previous = self.last_mouse_by_source.get(source)
            critical = event_type != "move" or button or wheel
            training = self.state == "training"
        if outside:
            self.on_control_signal("stop", "鼠标已离开雷电模拟器客户区")
            return
        if training and source == "user":
            self.cancel_event.set()
            self.on_control_signal("stop", "训练模式检测到真实用户鼠标事件，AI 已立即停止")
            return
        if training and source != "ai":
            self.cancel_event.set()
            self.on_control_signal("stop", "训练模式检测到未匹配的外部鼠标注入，AI 已立即停止")
            return
        input_budget = self.resources.capture_snapshot()
        if input_budget.must_pause and not critical:
            self.on_control_signal("stop", "资源红色条件触发，停止会话以避免遗漏鼠标事件：" + (input_budget.pause_reason or "资源红线"))
            return
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
                self._append_move_compression(session_id, source, last_kept, x, y, created, created_monotonic_ns, speed, rect)
                self.last_mouse_by_source[source] = (x, y, created_monotonic_ns)
                return
        if not self._flush_move_compression(session_id, source):
            return
        with self.lock:
            if not critical:
                kept_map[source] = (x, y, created_monotonic_ns, created, direction)
                self.last_move_kept = kept_map
            self.last_mouse_by_source[source] = (x, y, created_monotonic_ns)
            self.last_mouse_activity_ns = max(self.last_mouse_activity_ns, int(created_monotonic_ns))
            self.mouse_count += 1
        width = max(1, rect[2] - rect[0])
        height = max(1, rect[3] - rect[1])
        record = {"session_id": session_id, "created": created, "created_monotonic_ns": int(created_monotonic_ns), "source": source, "event_type": event_type, "button": button, "wheel": wheel, "x": x, "y": y, "relative_x": (x - rect[0]) / width, "relative_y": (y - rect[1]) / height, "dx": dx, "dy": dy, "direction": direction, "speed": speed, "behavior_probability": behavior_probability}
        try:
            self.mouse_queue.put_nowait(record)
            if source in ("user", "ai") and event_type in ("button_up", "wheel", "move"):
                with self.lock:
                    self.session_valid_actions += 1
        except queue.Full:
            self.on_control_signal("stop", "鼠标轨迹事件无法进入数据库写入队列，已安全结束会话")
            return
        if input_budget.must_pause:
            self.on_control_signal("stop", "资源红色条件触发，关键鼠标事件已入队后停止会话：" + (input_budget.pause_reason or "资源红线"))

    def _is_current(self, token, states=None):
        with self.lock:
            if token != self.epoch or self.cancel_event.is_set():
                return False
            return states is None or self.state in states

    def _find_valid_target(self, cursor_required=False):
        candidates = find_emulator_candidates(self.settings.data["emulator_path"])
        if not candidates:
            return None, None, "未检测到已启动的雷电模拟器窗口。"
        selected_pid = int(self.settings.data.get("emulator_pid", 0) or 0)
        selected_title = str(self.settings.data.get("emulator_title", "") or "")
        if len(candidates) > 1 and not selected_pid and not selected_title:
            return None, None, "检测到多个雷电实例。请先在控制面板中明确选择窗口标题、PID 或实例编号。"
        if selected_pid or selected_title:
            candidates = [item for item in candidates if (not selected_pid or item["pid"] == selected_pid) and (not selected_title or item["title"] == selected_title)]
            if len(candidates) != 1:
                return None, None, "已选择的雷电实例不存在、标题已变化或不再唯一。请重新选择实例。"
        elif len(candidates) == 1:
            pass
        else:
            return None, None, "多个雷电实例未明确选择。"
        chosen = candidates[0]
        rect = valid_client(chosen["hwnd"], cursor_required)
        if rect is None:
            return None, None, "雷电模拟器客户区异常：" + (getattr(valid_client, "last_reason", "不可见、被遮挡、最小化或未完全位于物理显示器范围内"))
        return chosen["hwnd"], rect, ""

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
                if not automatic: self.emit("notice", "当前不是空闲状态。")
                return False
        with self.lock:
            recovery_pending = bool(self.recovery_pending)
        if recovery_pending:
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
        if not self.hook.start(): self.emit("notice", self.hook.error or "鼠标钩子未启动，禁止进入模式。"); return False
        if not self.keyboard_hook.start(): self.emit("notice", self.keyboard_hook.error or "键盘钩子未启动，禁止进入模式。"); self.hook.stop(); return False
        try: self.ensure_store()
        except Exception as error:
            self.emit("notice", "无法创建存储路径：" + str(error)); self.hook.stop(); self.keyboard_hook.stop(); return False
        entry_budget = self.resources.acquire("capture")
        if entry_budget.must_pause:
            self.emit("notice", "资源恢复观察未完成，暂不允许进入模式：" + (entry_budget.pause_reason or "资源红线")); self.hook.stop(); self.keyboard_hook.stop(); return False
        hwnd, rect, reason = self._find_valid_target(False)
        if hwnd is None:
            self.emit("notice", reason); self.hook.stop(); self.keyboard_hook.stop(); return False
        if not self._place_cursor_before_entry(hwnd, rect):
            self.emit("notice", "进入模式前无法确认鼠标与雷电模拟器客户区状态。"); self.hook.stop(); self.keyboard_hook.stop(); return False
        rect = valid_client(hwnd, True)
        if rect is None:
            self.emit("notice", "雷电模拟器客户区状态异常：" + getattr(valid_client, "last_reason", "未知")); self.hook.stop(); self.keyboard_hook.stop(); return False
        try: session_id = self.store.create_session(mode)
        except Exception as error:
            self.emit("notice", "无法创建会话记录：" + str(error)); self.hook.stop(); self.keyboard_hook.stop(); return False
        pid = wintypes.DWORD(); user32.GetWindowThreadProcessId(root_window(hwnd), ctypes.byref(pid))
        with self.lock:
            self.epoch += 1; token = self.epoch; self.cancel_event = threading.Event(); self.state = mode
            self.target_hwnd = hwnd; self.target_root = root_window(hwnd); self.target_pid = int(pid.value); self.target_process_path = normalized_windows_path(process_full_path(int(pid.value))); self.target_rect = rect; self.session_id = session_id; self.session_mode = mode; self.session_started = time.monotonic(); self.hunger_anchor_ns = time.monotonic_ns(); self.last_observation = self.session_started; self.capture_failures = 0; self.last_valid_score = None; self.frame_scores = []; self.frame_count = 0; self.mouse_count = 0; self.session_valid_frames = 0; self.session_valid_actions = 0; self.last_sleep_fingerprint_check = 0.0; self.current_training_fingerprint = ""; self.gdi_failures = 0; self.gdi_static_count = 0; self.last_feature_hash = ""; self.stable_feature_frames = 0; self.last_gdi_hash = ""; self.last_gdi_hash_input_ns = 0; self.fallback_capture_pending = False; self.last_mouse_activity_ns = 0; self.raw_mouse_drops = 0; self.last_mouse_by_source = {"ai": None, "user": None, "external_injected": None}; self.ai_step = 0; self.training_auto_sleep_count = 0 if mode == "training" and not automatic else self.training_auto_sleep_count
            model = self.store.best_model() if mode == "training" else None
            plan = model.get("q_actions", model.get("hotspots", [])) if isinstance(model, dict) and model.get("champion", True) else []
            self.ai_plan = [item for item in plan if isinstance(item, dict)] if isinstance(plan, list) else []
            if isinstance(model, dict) and model.get("id") and self.store.models is not None:
                self.resources.backend.try_enable_gpu_model(self.store.models / (str(model.get("id")) + ".onnx"), self.resources.sample())
            self.ai_input_tracker.begin_session()
            self.action_limits = {}; self.last_move_kept = {}; self.move_segments = {}; self.pipeline_losses = {}; self.stop_requested.clear()
            available_memory = max(1, int(self.resources.sample().get("avail_memory", 0) or 0))
            byte_capacity = max(1 * 1024 * 1024, min(64 * 1024 * 1024, int(available_memory * 0.02)))
            if self.resources.capture_snapshot().state != "正常":
                byte_capacity = max(1 * 1024 * 1024, byte_capacity // 4)
            context = PipelineContext(token=token, session_id=session_id, byte_budget=ByteBudgetSemaphore(byte_capacity))
            self.pipeline_context = context
            self.resources.set_emulator_pid(self.target_pid)
            self.capture_queue = context.capture_queue
            self.feature_queue = context.feature_queue
            self.persist_queue = context.persist_queue
            self.pipeline_stop = context.stop_event
        self.store.add_system_event(session_id, "mode_enter", {"mode": mode, "automatic": automatic, "time": time.time(), "client_rect": rect, "target_pid": self.target_pid, "resource": self.resources.sample()})
        self.window_guard.start(self.target_hwnd)
        detail = "已进入" + ("学习模式" if mode == "learning" else ("训练模式；无已验证模型，安全观察且不执行点击、右键或滚轮" if not self.ai_plan else "训练模式"))
        self.post_state(detail)
        pipeline = [threading.Thread(target=self._pipeline_feature_loop,args=(context,),name="FeatureScore"), threading.Thread(target=self._pipeline_encode_loop,args=(context,),name="PngEncode"), threading.Thread(target=self._pipeline_persist_loop,args=(context,),name="FramePersist"), threading.Thread(target=self._pipeline_exact_score_loop,args=(context,),name="ExactScore")]
        threads = pipeline + [threading.Thread(target=self._capture_loop,args=(context,),name="CaptureLoop"), threading.Thread(target=self._monitor_loop,args=(token,),name="SessionMonitor")]
        if mode == "training": threads.append(threading.Thread(target=self._ai_loop,args=(token,),name="AIControl"))
        context.threads = pipeline
        with self.lock:
            self.pipeline_threads = pipeline; self.capture_threads = threads; self.worker_threads = [thread for thread in self.worker_threads if thread.is_alive()] + threads
        for thread in threads: thread.start()
        return True

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
            except Exception:pass

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
            self._drop_pipeline(session_id, stage, "固定容量队列满超过 {:.1f} 秒".format(timeout))
            self._flush_pipeline_losses()
            if stage != "capture":
                self.on_control_signal("stop", "{} 队列满超过 {:.1f} 秒，已安全结束会话并执行写入屏障".format(stage, timeout), token=packet.get("token"))
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

    def _capture_loop(self, context):
        token = context.token
        try:
            while context.accepting and self._is_current(token, ("learning", "training")) and not context.stop_event.is_set():
                capacity = self.store.capacity_status() if self.store.conn else {"blocked": False}
                with self.lock:
                    mode = self.session_mode
                    session_id = self.session_id
                if capacity.get("blocked"):
                    self.request_idle("经验池容量清理未完成，停止会话以避免截图记录断裂", token)
                    return
                budget = self.resources.acquire("capture")
                self._record_resource_decisions(context.session_id)
                capacity_tier = int(capacity.get("tier", 0) or 0)
                if capacity_tier >= 85:
                    budget.next_interval = max(float(budget.next_interval), 0.40 if capacity_tier < 95 else 1.20)
                    budget.max_capture_resolution = (320, 180) if capacity_tier >= 90 else (426, 240)
                    if capacity_tier >= 95:
                        self.emit("notice", "经验池已达到 95% 预警，已降低采样；建议进入睡眠模式执行精确评分与清理。")
                if not budget.allowed:
                    if budget.must_pause:
                        self.request_idle("资源红色条件触发，停止截图与会话：" + (budget.pause_reason or "资源红线"), token)
                        return
                    time.sleep(max(0.05, budget.next_interval))
                    continue
                with self.lock:
                    self.capture_interval_seconds = float(budget.next_interval)
                rect, reason = self._validate_bound_target(require_cursor=True, require_foreground=False)
                if rect is None:
                    self.request_idle("绑定雷电实例校验失败：" + reason, token)
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
                image = capture_client(hwnd, budget.max_capture_resolution[0], budget.max_capture_resolution[1], use_fallback) if session_id else None
                if image is None:
                    self.resources.update_capture_metrics(None, True)
                    fallback_reason = str(getattr(capture_client, "last_failure_reason", "") or "")
                    if fallback_reason and context.accepting:
                        self.request_idle("桌面回退截图校验失败：" + fallback_reason, token)
                        return
                    with self.lock:
                        self.capture_failures += 1
                        failures = self.capture_failures
                    if failures >= 12:
                        self.request_idle("连续无法记录雷电模拟器画面", token)
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
                time.sleep(budget.next_interval)
        finally:
            context.capture_done.set()

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
                        self._drop_pipeline(session_id, "feature", "截图内容不完整")
                        continue
                    budget = self.resources.acquire("capture")
                    self._record_resource_decisions(context.session_id)
                    if budget.must_pause and context.accepting:
                        self.request_idle("资源红色条件触发，停止新截图并排空写入：" + (budget.pause_reason or "资源红线"), context.token)
                    deadline = time.monotonic() + float(budget.retrieval_deadline_seconds)
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
                    )
                    online_score, meta = frame_score(image["dhash64"], historical, image)
                    image.update({
                        "online_score": online_score,
                        "score_candidate_count": meta["candidate_count"],
                        "score_top_k_distance": meta["top_k_distance"],
                        "score_retrieval_fallback": 1 if meta["retrieval_fallback"] else 0,
                        "score_retrieval_mode": meta["retrieval_mode"],
                        "score_exact_or_approx": meta["exact_or_approx"],
                        "score_recall_guard": meta["recall_guard"],
                        "score_provisional": True,
                        "score_valid": False,
                    })
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
                        "exact_score": None,
                        "hunger": None,
                        "reward": None,
                        "queued_at": time.time(),
                        "queued_monotonic": time.monotonic(),
                    })
                    if self._packet_current(context, packet):
                        self._put_value_packet(context.feature_queue, packet, "feature", context)
                except Exception as error:
                    self._drop_pipeline(session_id, "feature", "特征评分失败:" + str(error))
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
                    encoded = self.resources.backend.encode_frames([packet["image"] for packet in active], budget)
                    for packet, image in zip(active, encoded):
                        if not self._packet_current(context, packet):
                            self._drop_pipeline(packet.get("session_id"), "encode", "会话已关闭，禁止进入持久化")
                            self._release_packet_budget(packet, context)
                            continue
                        packet["image"] = image
                        packet["queued_at"] = time.time()
                        packet["queued_monotonic"] = time.monotonic()
                        self._put_value_packet(context.persist_queue, packet, "encode", context)
                except Exception as error:
                    for packet in active:
                        self._drop_pipeline(packet.get("session_id"), "encode", "PNG 并行压缩失败:" + str(error))
                        self._release_packet_budget(packet, context)
                    if context.accepting:
                        self.request_idle("PNG 编码失败，已停止新截图并排空会话：" + str(error), context.token)
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
                    started = time.monotonic()
                    self.store.save_frame(
                        packet["session_id"],
                        packet["image"],
                        packet["image"]["phash"],
                        online_score=packet.get("online_score"),
                        exact_score=None,
                        hunger=None,
                        reward=None,
                        experience_limit=self.settings.data.get("experience_limit"),
                    )
                    elapsed_ms = (time.monotonic() - started) * 1000.0
                    self.resources.update_sqlite_latency(elapsed_ms)
                    wal = self.store.wal_metrics()
                    self.resources.update_database_metrics(wal.get("wal_bytes"), wal.get("checkpoint_ms"), elapsed_ms)
                    with self.lock:
                        self.frame_count += 1
                        self.latest_frame_features = {
                            "seq": self.frame_count,
                            "state_hash": packet["image"].get("dhash64"),
                            "gray32x18": packet["image"].get("gray32x18"),
                            "edge_density": packet["image"].get("edge_density", 0.0),
                            "color_histogram": packet["image"].get("color_histogram"),
                            "aspect": float(packet["image"].get("width", 1)) / max(1.0, float(packet["image"].get("height", 1))),
                            "capture_finished_monotonic_ns": int(packet["image"].get("capture_finished_monotonic_ns", 0)),
                            "online_score": packet.get("online_score"),
                            "score": None,
                            "hunger": None,
                            "score_status": "pending_exact",
                            "state_stable": bool(packet["image"].get("state_stable")),
                            "capture_complete": bool(packet["image"].get("capture_complete")),
                        }
                        self.capture_failures = 0
                except PoolCapacityBlocked:
                    self._drop_pipeline(packet.get("session_id"), "persist", "经验池硬上限阻止新截图写入")
                    if context.accepting:
                        self.request_idle("经验池硬上限拒绝截图，已停止新截图并排空已有数据", context.token)
                except Exception as error:
                    self.resources.update_sqlite_latency(1000.0)
                    self._drop_pipeline(packet.get("session_id"), "persist", "SQLite/文件写入失败:" + str(error))
                    if context.accepting:
                        self.request_idle("截图持久化失败，已停止新截图并排空会话：" + str(error), context.token)
                finally:
                    self._release_packet_budget(packet, context)
                    self.resources.update_pipeline_queue(context.capture_queue.qsize() + context.feature_queue.qsize() + context.persist_queue.qsize(), self._pipeline_age(context), context.queue_capacity * 3)
        finally:
            context.persist_done.set()

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
                        self.request_idle("资源红色条件触发，停止新截图并排空精确评分：" + (budget.pause_reason or "资源红线"), context.token)
                    time.sleep(max(0.02, budget.next_interval))
                    continue
                resolved = self.store.process_deferred_exact_scores(
                    cancelled=lambda: context.stop_event.is_set() or not self._packet_current(context, {"token": context.token, "session_id": context.session_id}),
                    cooperative=lambda: self.resources.capture_snapshot().allowed,
                    maximum=1,
                    session_id=context.session_id,
                )
                if resolved > 0:
                    summary = self.store.session_score_summary(context.session_id)
                    with self.lock:
                        if self.session_id == context.session_id and self.epoch == context.token:
                            self.session_valid_frames = int(summary["valid_frames"])
                            self.frame_scores = list(summary["scores"])[-120:]
                            latest = summary.get("latest")
                            if latest and self.latest_frame_features and int(self.latest_frame_features.get("capture_finished_monotonic_ns", 0) or 0) == int(latest[4] or 0):
                                self.latest_frame_features.update({"score": float(latest[1]), "hunger": float(latest[2]), "score_status": "valid"})
                    continue
                time.sleep(max(0.01, budget.next_interval * 0.25))
        finally:
            status = self.store.deferred_score_status(context.session_id)
            oldest_age = max(0.0, time.time() - float(status["oldest"])) if status.get("oldest") else 0.0
            self.resources.update_exact_score_metrics(status["pending"], oldest_age)
            context.exact_done.set()
            context.drain_complete.set()

    def _monitor_loop(self, token):
        while self._is_current(token, ("learning", "training")):
            if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
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
                self.request_idle("雷电模拟器客户区或绑定实例异常：" + reason, token)
                return
            with self.lock:self.target_rect=rect
            if self._should_sleep(token):
                self.on_control_signal("auto_sleep", "AI 判断进入睡眠模式", token=token)
                return
            time.sleep(0.08)

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
            last_fingerprint_check = float(self.last_sleep_fingerprint_check)
            queue_empty = self.mouse_queue.empty() and self.raw_mouse_queue.empty() and self.raw_critical_queue.empty() and self.capture_queue.empty() and self.feature_queue.empty() and self.persist_queue.empty()
            session_id = self.session_id
        pending = self.store.deferred_score_status(session_id)
        pending_exact = int(pending.get("pending", 0) or 0)
        oldest_exact = max(0.0, time.time() - float(pending["oldest"])) if pending.get("oldest") else 0.0
        self.resources.update_exact_score_metrics(pending_exact, oldest_exact)
        try:
            if self.store.pool_size_fast() >= int(self.settings.data["experience_limit"] * 0.95):
                return True
        except Exception:
            pass
        if elapsed < 60.0:
            return False
        if auto_sleeps >= 1 or time.time() - last_auto_sleep_at < 300.0:
            return False
        sample = self.resources.sample()
        resource_pressure = 1.0 if sample.get("resource_state") in ("降速", "排空", "暂停") else 0.0
        coverage = min(1.0, len(plan) / 64.0)
        effective_frames = valid_frames + min(8, pending_exact)
        if effective_frames <= 0 and pending_exact <= 0:
            return False
        if sample.get("disk_free", 1) < 1024 * 1024 * 1024 or sample.get("avail_memory", 1) < 384 * 1024 * 1024:
            return True
        if not queue_empty and sample.get("resource_state") not in ("排空", "暂停"):
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
            "hunger_growth_speed": 0.00004,
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
        baseline_ready = effective_frames >= 4 and valid_actions >= 4
        backlog_ready = pending_exact >= 1 and (oldest_exact >= 1.0 or sample.get("resource_state") == "排空")
        minimum_gain = 0.02 if decision.get("trained") else 0.03
        should = (baseline_ready or backlog_ready) and (decision["sleep_probability"] >= 0.52 or resource_pressure > 0 or backlog_ready) and decision["expected_sleep_gain"] >= -0.05
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

    def _ai_target(self, rect):
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        with self.lock:
            step = self.ai_step
            self.ai_step += 1
            plan = list(self.ai_plan)
            current = dict(self.latest_frame_features) if self.latest_frame_features else None
            limits = dict(self.action_limits)
        gates = {"左键": (12, 0.60, 0.0, 2.0, 20), "右键": (16, 0.65, 0.02, 3.0, 12), "滚轮": (14, 0.62, 0.0, 1.5, 24), "水平滚轮": (16, 0.65, 0.02, 2.0, 12)}
        viable = []
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
                queue_age = float(self.resources.sample().get("pipeline_queue_age", 0.0) or 0.0)
                fresh = int((current or {}).get("capture_finished_monotonic_ns", 0) or 0) > 0 and time.monotonic_ns() - int((current or {}).get("capture_finished_monotonic_ns", 0) or 0) <= 1_000_000_000
                distance, state_parts, critical_mismatch = self._state_distance_details(current, item)
                if distance > float(item.get("state_similarity_threshold", 0.32)):
                    continue
                if critical_mismatch:
                    action_type = "移动"
                x_ratio = min(0.95, max(0.05, float(item.get("x", 0.5))))
                y_ratio = min(0.95, max(0.05, float(item.get("y", 0.5))))
                current_local = local_visual_descriptor((current or {}).get("gray32x18"), (current or {}).get("color_histogram"), (current or {}).get("edge_density", 0.0), x_ratio, y_ratio, int((item.get("local_descriptor") or {}).get("radius", 3) or 3))
                local_distance = local_visual_distance(current_local, item.get("local_descriptor") or {})
                local_radius = max(0.0, min(1.0, float(item.get("local_uncertainty_radius", 1.0) or 1.0)))
                if action_type != "移动" and (local_distance > 0.28 or local_radius > 0.14):
                    action_type = "移动"
                if action_type != "移动":
                    min_samples, min_probability, min_lcb, cooldown, per_minute = gates.get(action_type, (999999, 1.0, 1.0, 999.0, 0))
                    stat = limits.get(action_type, {"last": 0.0, "times": []})
                    recent = [t for t in stat.get("times", []) if time.monotonic() - t < 60.0]
                    if samples < min_samples or effective_samples < 8.0 or probability < min_probability or lcb <= min_lcb or float(item.get("validation_lower_bound", -1.0)) <= 0 or int(item.get("validation_samples", 0)) < 8 or float(item.get("validation_false_positive_rate", 1.0)) > 0.10 or interval_width >= 0.24 or baseline_support < 4.0 or uncertainty > 0.12 or distance >= 0.32 or not fresh or not bool((current or {}).get("state_stable")) or not bool((current or {}).get("capture_complete")) or str((current or {}).get("score_status", "")) != "valid" or queue_age > 0.25 or time.monotonic() - stat.get("last", 0.0) < cooldown or len(recent) >= per_minute:
                        action_type = "移动"
                viable.append((probability - distance * 0.25, samples, action_type, distance, uncertainty, probability, item))
            except (TypeError, ValueError):
                pass
        viable.sort(reverse=True, key=lambda value: (value[0], value[1]))
        item_tuple = viable[step % len(viable)] if viable else None
        if item_tuple is not None:
            _, samples, action_type, distance, uncertainty, probability, item = item_tuple
            x_ratio = min(0.95, max(0.05, float(item.get("x", 0.5))))
            y_ratio = min(0.95, max(0.05, float(item.get("y", 0.5))))
            wheel_delta = max(-1200, min(1200, int(item.get("wheel_delta", 120 if action_type in ("滚轮", "水平滚轮") else 0))))
            model_available = True
        else:
            x_ratio = 0.08 + 0.84 * ((step * 0.618033988749895) % 1.0)
            y_ratio = 0.08 + 0.84 * ((step * 0.414213562373095) % 1.0)
            action_type = "移动"
            probability = 0.58
            uncertainty = 0.55
            samples = 0
            wheel_delta = 0
            distance = 1.0
            model_available = False
        x = rect[0] + int(width * x_ratio)
        y = rect[1] + int(height * y_ratio)
        return {"x": x, "y": y, "action_type": action_type, "wheel_delta": wheel_delta, "confidence_probability": probability, "confidence": probability, "uncertainty": uncertainty, "samples": samples, "state_match_distance": distance, "model_available": model_available, "capture_finished_monotonic_ns": int((current or {}).get("capture_finished_monotonic_ns", 0) or 0), "score_status": str((current or {}).get("score_status", "unknown")), "state_stable": bool((current or {}).get("state_stable")), "capture_complete": bool((current or {}).get("capture_complete"))}

    def _validate_bound_target(self, require_cursor=True, require_foreground=False):
        with self.lock:
            hwnd = self.target_hwnd
            expected_root = self.target_root
            expected_pid = self.target_pid
            expected_path = self.target_process_path
        if not hwnd or not expected_root or not expected_pid:
            return None, "未绑定雷电实例"
        if not user32.IsWindow(hwnd) or root_window(hwnd) != expected_root:
            return None, "窗口句柄已失效或根窗口已变化"
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(expected_root, ctypes.byref(pid))
        if int(pid.value) != int(expected_pid):
            return None, "绑定窗口 PID 已变化"
        actual_path = normalized_windows_path(process_full_path(int(pid.value)))
        configured_path = normalized_windows_path(self.settings.data.get("emulator_path", ""))
        if not actual_path or actual_path != expected_path or actual_path != configured_path:
            return None, "绑定窗口可执行文件路径已变化"
        rect = valid_client(hwnd, require_cursor)
        if rect is None:
            return None, getattr(valid_client, "last_reason", "客户区无效")
        if require_foreground:
            foreground = root_window(user32.GetForegroundWindow())
            if not foreground or foreground != expected_root:
                return None, "模拟器不是前台窗口"
            fg_pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(foreground, ctypes.byref(fg_pid))
            if int(fg_pid.value) != int(expected_pid):
                return None, "前台窗口 PID 不匹配"
            foreground_path = normalized_windows_path(process_full_path(int(fg_pid.value)))
            if foreground_path != expected_path:
                return None, "前台窗口可执行文件路径不匹配"
        return rect, ""

    def _foreground_matches_target(self):
        rect, _ = self._validate_bound_target(require_cursor=True, require_foreground=True)
        return rect is not None

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
            if target.get("action_type") != "移动" and (frame_age > freshness_limit or queue_age > 0.250 or budget.queue_fill_ratio >= 0.90 or target.get("score_status") != "valid" or not target.get("state_stable") or not target.get("capture_complete")):
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
                ok = ai_left_click(down_marker, up_marker)
            elif action_type == "右键":
                cx, cy = cursor_position()
                down_marker = self.ai_input_tracker.register("button_down", "right", 0, cx, cy, target.get("confidence_probability"))
                up_marker = self.ai_input_tracker.register("button_up", "right", 0, cx, cy, target.get("confidence_probability"))
                ok = ai_right_click(down_marker, up_marker)
            elif action_type == "滚轮":
                cx, cy = cursor_position()
                delta = target.get("wheel_delta", 120)
                wheel_marker = self.ai_input_tracker.register("wheel", "vertical", delta, cx, cy, target.get("confidence_probability"))
                ok = ai_wheel(delta, wheel_marker, False)
            elif action_type == "水平滚轮":
                cx, cy = cursor_position()
                delta = target.get("wheel_delta", 120)
                wheel_marker = self.ai_input_tracker.register("wheel", "horizontal", delta, cx, cy, target.get("confidence_probability"))
                ok = ai_wheel(delta, wheel_marker, True)
            action_finished_ns = time.monotonic_ns()
            if action_type != "移动" and ok:
                with self.lock:
                    stat = self.action_limits.setdefault(action_type, {"last": 0.0, "times": []})
                    now = time.monotonic()
                    stat["last"] = now
                    stat["times"] = [t for t in stat.get("times", []) if now - t < 60.0] + [now]
            if not ok:
                self.request_idle("AI 鼠标动作无法执行" + str(action_type), token)
                return
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
            deadline = time.monotonic() + 30.0
            context.capture_done.wait(max(0.1, deadline - time.monotonic()))
            for thread in list(context.threads):
                if thread is threading.current_thread():
                    continue
                remaining = max(0.1, deadline - time.monotonic())
                thread.join(remaining)
            alive = [thread.name for thread in context.threads if thread is not threading.current_thread() and thread.is_alive()]
            queue_sets = (("capture", context.capture_queue), ("feature", context.feature_queue), ("persist", context.persist_queue))
            queues_left = sum(item[1].qsize() for item in queue_sets)
            if alive or queues_left:
                context.stop_event.set()
                lost = 0
                for stage, pending in queue_sets:
                    stage_lost = 0
                    while True:
                        try:
                            packet = pending.get_nowait()
                        except queue.Empty:
                            break
                        stage_lost += 1
                        self._release_packet_budget(packet, context)
                    if stage_lost:
                        lost += stage_lost
                        self.store.record_pipeline_loss(session_id, time.time(), time.time(), stage_lost, stage, "写入屏障超时强制丢弃")
                context.closed = True
                context.drain_complete.set()
                forced_detail = "写入屏障超时：线程 {}，强制隔离并丢弃数据包 {}".format(",".join(alive) or "无", lost)
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
        with self.lock:
            if token is not None and token != self.epoch:
                return False
            if self.state == "idle":
                return False
            if self.state == "stopping":
                return True
            previous = self.state
            self.stop_requested.set()
            self.cancel_event.set()
            context = self.pipeline_context
            if previous in ("learning", "training"):
                self.state = "stopping"
                if context is not None:
                    context.accepting = False
                    context.draining = True
            else:
                self.state = "idle"
        clean = True
        detail = ""
        if previous in ("learning", "training"):
            self.hook.stop()
            self.keyboard_hook.stop()
            self.post_state("正在停止接入并排空截图、特征、编码和持久化队列")
            clean, detail = self._close_active_session(reason, barrier=True)
        elif previous == "sleep":
            self.keyboard_hook.stop()
        with self.lock:
            if not clean:
                self.recovery_pending = True
                self.recovery_reason = detail or "停止期间数据恢复待处理"
            self.epoch += 1
            self.state = "idle"
            self.stop_requested.clear()
        self.ai_input_tracker.clear()
        self.emit("progress", 0.0)
        final_detail = reason if previous != "sleep" else "睡眠模式已中止：" + reason
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
        if threading.current_thread() is not self.control_thread:
            self.on_control_signal("auto_sleep", "AI 判断进入睡眠模式", token=token)
            return
        with self.lock:
            if token != self.epoch or self.state != "training":
                return
            context = self.pipeline_context
            self.state = "stopping"
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
                self.state = "idle"
                self.epoch += 1
            self.post_state("空闲；自动睡眠取消，数据恢复待处理：" + self.recovery_reason)
            return
        with self.lock:
            self.epoch += 1
            sleep_token = self.epoch
            self.cancel_event = threading.Event()
            self.training_auto_sleep_count += 1
            self.last_auto_sleep_at = time.time()
            self.last_auto_sleep_fingerprint = self.current_training_fingerprint
            self.state = "sleep"
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
        task = "maintenance" if purpose == "maintenance" else "sleep_training"
        while not self._cancelled(token):
            budget = self.resources.acquire(task)
            if budget.allowed:
                return True
            if budget.must_pause:
                self.request_idle("资源红色条件触发，停止训练和维护：" + (budget.pause_reason or "资源红线"), token)
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
                        except Exception:pass
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
                expected_no_action_reward_delta=sum(item[2] for item in nearest)/max(1,len(nearest)) if nearest else 0.0
                stable=min(stability(before_i),stability(after_i))
                score_delta=float(after[6])-float(before[6]);reward_delta=float(after[7])-float(before[7]);expected_hunger=max(0.0,post_ms/1000.0*0.00004);advantage=reward_delta-expected_no_action_reward_delta
                gx=min(15,max(0,int(float(action["rx"])*16)));gy=min(8,max(0,int(float(action["ry"])*9)))
                local_descriptor=local_visual_descriptor(before[11],before[13],before[12],action["rx"],action["ry"],3)
                local_key=local_descriptor.get("gray","")[:48]
                example={"action_id":action["action_id"],"session_id":session_id,"before_frame_id":before[0],"after_frame_id":after[0],"mouse_event_id":action["mouse_event_id"],"action_time":action_ns,"post_action_delay_ms":post_ms,"score_delta":score_delta,"reward_delta":reward_delta,"hunger_delta_expected":expected_hunger,"baseline_score_delta":baseline_score_delta,"expected_no_action_reward_delta":expected_no_action_reward_delta,"action_advantage":advantage,"stability":stable,"baseline_count":len(nearest),"trajectory":dict(action.get("trajectory") or {}),"behavior_probability":action.get("behavior_probability"),"local_descriptor":local_descriptor,"outcome_valid":role!="excluded_gap" and stable>=0.45,"split_role":role}
                persisted.append(example)
                wheel_axis=action.get("wheel_axis","");signed=int(action.get("wheel_delta",action.get("signed_delta",0)) or 0);wheel_direction=1 if signed>0 else -1 if signed<0 else 0;wheel_bucket=min(10,abs(signed)//120)
                key=(state_key,gx,gy,action["action_type"],wheel_axis,wheel_direction,wheel_bucket,local_key)
                if role=="train" and stable>=0.45:
                    item=states.setdefault(key,{"samples":0,"human":0,"ai":0,"sum":0.0,"sum2":0.0,"score_sum":0.0,"reward_sum":0.0,"hunger_sum":0.0,"baseline_support":0,"stability_sum":0.0,"trajectory_speed_sum":0.0,"trajectory_acceleration_sum":0.0,"trajectory_dwell_sum":0.0,"trajectory_turn_sum":0.0,"trajectory_path_sum":0.0,"trajectory_stability_sum":0.0,"examples":[],"state_hash":before[4] or before[5],"gray32x18":before[11],"edge_density":before[12],"color_histogram":before[13],"local_descriptor":local_descriptor,"aspect":before[9]/max(1,before[10])})
                    item["samples"]+=1;item["human" if action["source"] in ("user","用户") else "ai"]+=1;item["sum"]+=advantage;item["sum2"]+=advantage*advantage;item["score_sum"]+=score_delta;item["reward_sum"]+=reward_delta;item["hunger_sum"]+=expected_hunger;item["baseline_support"]+=len(nearest);item["stability_sum"]+=stable;trajectory=action.get("trajectory") or {};item["trajectory_speed_sum"]+=float(trajectory.get("speed_mean",0.0));item["trajectory_acceleration_sum"]+=float(trajectory.get("acceleration_mean",0.0));item["trajectory_dwell_sum"]+=float(trajectory.get("dwell_ms",0.0));item["trajectory_turn_sum"]+=float(trajectory.get("turns",0.0));item["trajectory_path_sum"]+=float(trajectory.get("path_length",0.0));item["trajectory_stability_sum"]+=float(trajectory.get("cursor_stability",0.0));item["examples"].append(dict(example,wheel_delta=signed,wheel_axis=wheel_axis));outcomes.append(example)
                elif role=="validation" and stable>=0.45:
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
            payload={"state_key":key[0],"state_hash":item["state_hash"],"gray32x18":item["gray32x18"],"edge_density":item["edge_density"],"color_histogram":item["color_histogram"],"local_descriptor":item["local_descriptor"],"local_uncertainty_radius":min(0.25,0.03+standard_error*2.0),"aspect":item["aspect"],"x":(key[1]+0.5)/16.0,"y":(key[2]+0.5)/9.0,"action_type":key[3],"wheel_axis":key[4],"wheel_direction":key[5],"wheel_magnitude_bucket":key[6],"wheel_delta":int(round(sum(e.get("wheel_delta",0) for e in wheel_examples)/max(1,len(wheel_examples)))) if key[3] in ("滚轮","水平滚轮") else 0,"samples":n,"effective_samples":n*n/(n+prior_strength),"human_samples":item["human"],"ai_samples":item["ai"],"average_action_advantage":mean,"advantage_variance":var,"confidence_interval_width":ci_width,"average_score_delta":item["score_sum"]/max(1,n),"average_reward_delta":item["reward_sum"]/max(1,n),"average_hunger_delta_expected":item["hunger_sum"]/max(1,n),"baseline_support":baseline_avg,"stability":stable_avg,"trajectory_profile":{"speed_mean":item["trajectory_speed_sum"]/max(1,n),"acceleration_mean":item["trajectory_acceleration_sum"]/max(1,n),"dwell_ms":item["trajectory_dwell_sum"]/max(1,n),"turns":item["trajectory_turn_sum"]/max(1,n),"path_length":item["trajectory_path_sum"]/max(1,n),"cursor_stability":item["trajectory_stability_sum"]/max(1,n)},"confidence_lower_bound":lcb,"confidence_probability":confidence_probability,"uncertainty":standard_error,"state_similarity_threshold":0.32}
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
        validated_nonmoving=[item for item in nonmoving if item.get("validation_samples",0)>=8 and item.get("validation_lower_bound",-1.0)>0 and item.get("validation_state_coverage",0)>=1 and item.get("validation_false_positive_rate",1.0)<=0.10]
        propensity_values=[example.get("behavior_probability") for _,example in validation_outcomes if example.get("behavior_probability") is not None]
        offline_policy={"recorded_propensities":len(propensity_values),"total_validation_actions":len(validation_outcomes),"causal_claim":"not_claimed_without_propensity" if not propensity_values else "propensity_recorded_observational_only"}
        safe_actions=[item for item in actions if item["action_type"]=="移动" or item in validated_nonmoving]
        payload={"id":uuid.uuid4().hex,"trained_at":time.time(),"quality":quality,"train_quality":sum(a["average_action_advantage"] for a in actions)/max(1,len(actions)),"frame_count":sum(len(v) for v in frames_by_session.values()),"mouse_count":sum(len(v) for v in mouse_by_session.values()),"training_samples":len(outcomes),"semantic_actions":all_actions_count,"validation_samples":len(validation_outcomes),"validation_hits":hits,"validation_mean_action_advantage":val_mean,"validation_confidence_interval":val_ci,"validation_failure_rate":failures/max(1,len(validation_outcomes)),"validation_mean_absolute_error":sum(absolute_errors)/max(1,len(absolute_errors)),"validation_sign_hit_rate":sign_hits/max(1,hits),"validation_lcb_coverage_rate":lcb_coverages/max(1,hits),"validation_state_coverage":len({key[0] for key,_ in validation_outcomes}),"validation_sessions":sorted(set(validation_blocks)),"coverage_states":len({a["state_key"] for a in actions}),"failure_rate":len([a for a in actions if a["confidence_lower_bound"]<=0])/max(1,len(actions)),"nonmoving_candidates":len(nonmoving),"nonmoving_validated":len(validated_nonmoving),"offline_policy_evaluation":offline_policy,"model_version":9,"champion":True,"last_used":time.time(),"action_quality":quality,"validation_quality":quality,"policy":{"min_samples":12,"min_effective_samples":8,"min_validation_samples":8,"uncertainty_threshold":0.12,"max_confidence_interval_width":0.24,"min_confidence_lower_bound":0.0,"min_baseline_support":4,"similarity_threshold":0.78,"max_nonmoving_false_positive_rate":0.10,"low_confidence_action":"move_only","blacklist_regions":[],"target":"action_advantage"},"q_actions":sorted(safe_actions,key=lambda a:(a.get("validation_lower_bound",-1.0),a["confidence_lower_bound"],a["samples"]),reverse=True)[:256],"outcome_examples":outcomes[-256:]}
        champion=self.store.best_model();champion_quality=float(champion.get("validation_quality",-999999.0)) if isinstance(champion,dict) else -999999.0
        enough=len(outcomes)>=48 and len({a["state_key"] for a in actions})>=4 and len(validation_sessions)>=1 and len(validation_outcomes)>=20 and hits>=12 and val_ci<0.24 and all(item in validated_nonmoving for item in nonmoving)
        if not enough or quality <= champion_quality:
            reason = "样本不足或验证未通过"
            payload["champion"] = False
            self._record_training_failure(fingerprint, reason)
            self.store.add_system_event(None, "model_candidate_rejected", {"validation_quality": quality, "champion_quality": champion_quality, "validation_samples": len(validation_outcomes), "validation_hits": hits, "training_samples": len(outcomes), "fingerprint": fingerprint, "time": time.time()})
            return self._training_result("rejected", reason, champion if isinstance(champion, dict) else payload, len(outcomes), len(validation_outcomes), quality - champion_quality, False)
        name="model_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")+"_"+payload["id"][:8]+".json";final_path=self.store.models/name;temp_path=final_path.with_suffix(".tmp");temp_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
        if self._cancelled(token):
            temp_path.unlink(missing_ok=True)
            return self._training_result("cancelled", "模型落盘前被取消", training_samples=len(outcomes), validation_samples=len(validation_outcomes))
        temp_path.replace(final_path)
        self.store.register_model_metadata(payload["id"], final_path, payload, outcomes, validation_outcomes)
        self.last_model_training = time.time()
        self.last_training_success = self.last_model_training
        self.last_successful_training_fingerprint = fingerprint
        self.training_retry_count = 0
        self.next_training_retry_at = 0.0
        self.last_training_failure_reason = ""
        return self._training_result("trained", "新冠军模型已落盘", payload, len(outcomes), len(validation_outcomes), quality - champion_quality, True)

    def _sleep_worker(self, token, resume_training):
        started = time.monotonic()
        before_model = self.store.best_model() or {}
        before_quality = float(before_model.get("validation_quality", before_model.get("quality", 0.0)) or 0.0) if isinstance(before_model, dict) else 0.0
        before_pool = int(self.store.pool_breakdown(False).get("experience_total_bytes", 0))
        decision_id = self.pending_sleep_decision
        training_result = self._training_result("cancelled", "睡眠任务尚未开始")
        try:
            self.emit("progress", 4.0)
            self.emit("state", {"state": "sleep", "detail": "低优先级精确评分队列与任务1：训练 AI 模型", "cpu": self.resources.sample()["cpu"], "memory": self.resources.sample()["memory"]})
            while not self._cancelled(token):
                pending = self.store.deferred_score_status()
                oldest_wait = max(0.0, time.time() - pending["oldest"]) if pending.get("oldest") else 0.0
                self.resources.update_exact_score_metrics(pending["pending"], oldest_wait)
                if pending["failed"] > 0:
                    sessions = self.store.exclude_failed_deferred_sessions()
                    training_result = self._training_result("failed", "{} 个会话存在三次失败的精确评分".format(len(sessions)))
                    self.store.finalize_sleep_decision(decision_id, training_result, 0, time.monotonic() - started, 0.0)
                    self._finish_idle(token, "任务1未产出新模型：{}".format(training_result["reason"]), True)
                    return
                if pending["pending"] <= 0:
                    break
                budget = self.resources.acquire("maintenance")
                if not budget.allowed:
                    if budget.must_pause:
                        training_result = self._training_result("cancelled", "资源红线阻止精确评分")
                        self.store.finalize_sleep_decision(decision_id, training_result, 0, time.monotonic() - started, 0.0)
                        self._finish_idle(token, "任务1未产出新模型：资源红线", True)
                        return
                    time.sleep(max(0.02, budget.next_interval))
                    continue
                resolved = self.store.process_deferred_exact_scores(lambda: self._cancelled(token), lambda: self.resources.capture_snapshot().allowed, maximum=1)
                status = self.store.deferred_score_status()
                oldest_wait = max(0.0, time.time() - status["oldest"]) if status.get("oldest") else 0.0
                self.resources.update_exact_score_metrics(status["pending"], oldest_wait)
                self.emit("state", {"state": "sleep", "detail": "精确评分待处理 {} 帧，最老等待 {:.1f} 秒".format(status["pending"], oldest_wait), "cpu": self.resources.sample()["cpu"], "memory": self.resources.sample()["memory"]})
                if resolved <= 0 and status["pending"] > 0:
                    time.sleep(max(0.02, budget.next_interval * 0.25))
            if self._cancelled(token):
                return
            training_result = self._train_model(token)
            if self._cancelled(token):
                return
            if training_result["status"] == "trained":
                task1_detail = "任务1完成：已产出并落盘新冠军模型"
            else:
                task1_detail = "任务1未产出新模型：{}".format(training_result.get("reason") or training_result.get("status"))
            self.emit("progress", 56.0)
            self.emit("state", {"state": "sleep", "detail": task1_detail + "；任务2：检查 AI 模型与经验池", "cpu": self.resources.sample()["cpu"], "memory": self.resources.sample()["memory"]})
            self.store.recover_deletions()
            if not self._wait_resource(token, "maintenance"):
                return
            self.flush_mouse_records()
            model_removed = self.store.prune_models(max(1, int(self.settings.data["model_limit"])), lambda: self._cancelled(token), lambda: self._wait_resource(token, "maintenance"))
            experience_removed, remaining = self.store.prune_experience(max(1, int(self.settings.data["experience_limit"])), lambda: self._cancelled(token), lambda value: self.emit("progress", value), lambda: self._wait_resource(token, "maintenance"))
            if self._cancelled(token):
                return
            model_status = getattr(self.store, "last_model_prune_result", {"success": True, "remaining": len(self.store.model_files()), "target": len(self.store.model_files())})
            pool_status = getattr(self.store, "last_experience_prune_result", {"success": remaining <= int(self.settings.data["experience_limit"]), "remaining": remaining, "target": int(self.settings.data["experience_limit"] * 0.5)})
            if not model_status.get("success") or not pool_status.get("success"):
                detail = "任务2未完成：模型 {} / 目标 {}；经验池 {} / 目标 {}。已停止采集新截图，等待清理完成。".format(model_status.get("remaining"), model_status.get("target"), pool_status.get("remaining"), pool_status.get("target"))
                self.store.finalize_sleep_decision(decision_id, training_result, max(0, before_pool - remaining), time.monotonic() - started, 0.0)
                self._finish_idle(token, detail, True)
                return
            actual_breakdown = self.store.pool_breakdown(True)
            remaining = int(actual_breakdown.get("experience_total_bytes", 0))
            if remaining > int(self.settings.data["experience_limit"] * 0.5):
                detail = "任务2实际目录大小未降至经验池上限的 50%，禁止恢复训练"
                self.store.finalize_sleep_decision(decision_id, training_result, max(0, before_pool - remaining), time.monotonic() - started, 0.0)
                self._finish_idle(token, detail, True)
                return
            after_model = self.store.best_model() or {}
            after_quality = float(after_model.get("validation_quality", after_model.get("quality", 0.0)) or 0.0) if isinstance(after_model, dict) else 0.0
            restored_gain = after_quality - before_quality if resume_training and training_result["status"] == "trained" else 0.0
            self.store.finalize_sleep_decision(decision_id, training_result, max(0, before_pool - remaining), time.monotonic() - started, restored_gain)
            self.pending_sleep_decision = None
            self.emit("progress", 100.0)
            detail = "{}；任务2完成：删除 AI 模型 {} 个，删除经验 {} 条，经验池 {:.2f} MB".format(task1_detail, model_removed, experience_removed, remaining / 1024 / 1024)
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
            training_result = self._training_result("failed", str(error))
            try:
                self.store.finalize_sleep_decision(decision_id, training_result, 0, time.monotonic() - started, 0.0)
            except Exception:
                pass
            self._finish_idle(token, "睡眠模式发生错误：" + str(error), True)

    def _finish_idle(self, token, detail, release_keyboard=True):
        if threading.current_thread() is not self.control_thread:
            self.on_control_signal("sleep_finish", detail, token=token)
            return
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

    def information(self, reconcile=False):
        sample = self.resources.sample()
        with self.lock:
            state = self.state
            frames = self.frame_count
            mouse = self.mouse_count
            session = self.session_id
        try:
            breakdown = self.store.pool_breakdown(bool(reconcile)) if self.store.pool else {}
            pool_size = int(breakdown.get("experience_total_bytes", 0))
            model_count = len(self.store.model_files()) if self.store.models else 0
            capacity = self.store.capacity_status() if self.store.conn else {"blocked": False}
        except Exception:
            breakdown = {}
            pool_size = 0
            model_count = 0
            capacity = {"blocked": False}
        capture_budget = self.resources.capture_snapshot()
        return {"state": state, "frames": frames, "mouse": mouse, "session": session or "无", "recovery_pending": bool(self.recovery_pending), "recovery_reason": self.recovery_reason, "cpu": sample["cpu"], "memory": sample["memory"], "pool_size": pool_size, "pool_size_fast": pool_size, "pool_breakdown": breakdown, "model_count": model_count, "capacity": capacity, "resource": dict(sample), "gpu_name": self.resources.backend.name(), "backend": self.resources.backend.name(), "gpu": sample.get("gpu"), "gpu_total": sample.get("gpu_dedicated_total"), "gpu_used": sample.get("gpu_dedicated_used"), "gpu_batch_size": 0, "cpu_workers": capture_budget.cpu_workers, "capture_fps": 1.0 / max(0.001, capture_budget.next_interval), "capture_resolution": "{}×{}".format(*capture_budget.max_capture_resolution), "queue_age": sample.get("queue_age", 0.0), "pipeline_queue_age": sample.get("pipeline_queue_age", 0.0), "resource_state": capture_budget.state, "pause_reason": capture_budget.pause_reason or "无", "metric_sources": sample.get("metric_sources", {}), "ldplayer_cpu": sample.get("ldplayer_cpu", 0.0), "program_cpu": sample.get("process_cpu", 0.0), "program_gpu": sample.get("program_gpu"), "ldplayer_gpu": sample.get("ldplayer_gpu"), "gpu_sampling_source": sample.get("gpu_sampling_source", "不可用"), "disk_write_latency": sample.get("disk_write_latency"), "sqlite_latency": sample.get("sqlite_latency", 0.0)}

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
        self.window_guard.stop()
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

class Panel:
    def __init__(self, root):
        self.root = root
        self.settings = Settings()
        self.events = queue.Queue()
        self.controller = Controller(self.settings, self.enqueue)
        self.path_var = StringVar(value=self.settings.data["emulator_path"])
        self.storage_var = StringVar(value=self.settings.data["storage_path"])
        self.instance_var = StringVar(value=("PID {} · {}".format(self.settings.data.get("emulator_pid", 0), self.settings.data.get("emulator_title", "")) if self.settings.data.get("emulator_pid", 0) else "未绑定；多开时必须明确选择"))
        self.experience_var = StringVar(value=self.format_bytes(self.settings.data["experience_limit"]))
        self.model_var = StringVar(value=str(self.settings.data["model_limit"]) + " 个")
        self.mode_var = StringVar(value="空闲")
        self.status_var = StringVar(value=("配置读取错误：" + "；".join(self.settings.config_errors)) if self.settings.config_errors else "控制面板已就绪。")
        self.performance_var = StringVar(value="CPU 0.0% · 内存 0.0%")
        self.layout_after = None
        self.layout_signature = None
        self.footer_status_label = None
        self.footer_perf_label = None
        self.progress_var = DoubleVar(value=0.0)
        self.mode_buttons = []
        self.configuration_buttons = []
        self.scroll_canvas = None
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
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        labels = (("雷电模拟器", self.path_var), ("绑定雷电实例", self.instance_var), ("存储路径", self.storage_var), ("经验池上限", self.experience_var), ("AI 模型数量上限", self.model_var))
        colors = ("#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4")
        commands = (self.choose_emulator, self.choose_emulator_instance, self.choose_storage, self.change_experience, self.change_models)
        texts = ("选择雷电模拟器路径", "选择雷电实例", "选择存储路径", "修改经验池上限", "修改AI模型数量上限")
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
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass

    def start_mode(self, mode):
        if self.controller.busy():
            self.controller.emit("notice", "当前模式：" + self.controller.current_state() + "，拒绝重复进入。")
            return False
        self.root.attributes("-topmost", False)
        self.root.lower()
        self.root.update_idletasks()
        def continue_start():
            started = self.controller.start_session(mode)
            if not started:
                self.status_var.set("未进入模式，请检查雷电实例选择和客户区状态。")
                self.restore_panel()
        self.root.after_idle(continue_start)
        return True

    def protect_configuration(self):
        if self.controller.busy():
            messagebox.showwarning("当前状态", "运行中的模式不会修改配置。请先返回空闲状态。", parent=self.root)
            return False
        return True

    def choose_emulator(self):
        if not self.protect_configuration(): return
        selected=filedialog.askopenfilename(parent=self.root,title="选择雷电模拟器路径",initialfile=Path(self.settings.data["emulator_path"]).name,filetypes=[("可执行文件","*.exe"),("所有文件","*.*")])
        if selected:
            self.settings.data["emulator_path"]=selected;self.settings.data["emulator_pid"]=0;self.settings.data["emulator_title"]="";self.settings.save();self.path_var.set(selected);self.instance_var.set("未绑定；多开时必须明确选择");self.status_var.set("已更新雷电模拟器路径。")

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
        self.settings.data["emulator_pid"]=int(chosen["pid"]);self.settings.data["emulator_title"]=str(chosen["title"]);self.settings.save();self.instance_var.set("PID {} · {}".format(chosen["pid"],chosen["title"] or "无标题"));self.status_var.set("已绑定雷电实例；会话期间将固定校验 HWND 和 PID。")

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
        info=self.controller.information(reconcile=True)
        window=Toplevel(self.root);window.title("更多信息")
        wx1,wy1,wx2,wy2=work_area_for_window(self.root);width=max(520,min(980,int((wx2-wx1)*0.55)));height=max(420,min(820,int((wy2-wy1)*0.65)))
        window.geometry("{}x{}+{}+{}".format(width,height,wx1+max(0,((wx2-wx1)-width)//2),wy1+max(0,((wy2-wy1)-height)//2)));window.resizable(True,True);window.configure(bg="#0f172a");window.grid_columnconfigure(0,weight=1);window.grid_rowconfigure(1,weight=1)
        Label(window,text="运行信息",bg="#0f172a",fg="white",font=("Microsoft YaHei UI",18,"bold"),padx=20,pady=18).grid(row=0,column=0,sticky="w")
        canvas=Canvas(window,bg="#f8fafc",highlightthickness=0,bd=0);scroll=ttk.Scrollbar(window,orient="vertical",command=canvas.yview);canvas.configure(yscrollcommand=scroll.set);canvas.grid(row=1,column=0,sticky="nsew",padx=(16,0),pady=(0,16));scroll.grid(row=1,column=1,sticky="ns",padx=(0,16),pady=(0,16))
        content=Frame(canvas,bg="#f8fafc",padx=20,pady=18);content_id=canvas.create_window((0,0),window=content,anchor="nw");content.grid_columnconfigure(1,weight=1);content.bind("<Configure>",lambda event:canvas.configure(scrollregion=canvas.bbox("all")));canvas.bind("<Configure>",lambda event:canvas.itemconfigure(content_id,width=event.width))
        def number(value,unit=""):
            return "不可用" if value is None else "{:.1f}{}".format(float(value),unit)
        capacity=info.get("capacity",{})
        rows=[("当前状态",info["state"]),("本次会话",info["session"]),("本次记录画面",str(info["frames"])),("本次记录鼠标事件",str(info["mouse"])),("本程序 CPU",number(info.get("program_cpu"),"%")+" · "+info.get("metric_sources",{}).get("本程序 CPU","未知来源")),("雷电 CPU",number(info.get("ldplayer_cpu"),"%")+" · "+info.get("metric_sources",{}).get("雷电 CPU","未知来源")),("策略执行后端",info.get("backend","CPU 表格策略")),("GPU 状态","仅监测 GPU 指标；未加载经真实推理验证的 GPU 模型"),("本程序 GPU 引擎",number(info.get("program_gpu"),"%")+" · "+info.get("gpu_sampling_source","不可用")),("雷电 GPU 引擎",number(info.get("ldplayer_gpu"),"%")+" · "+info.get("gpu_sampling_source","不可用")),("可用显存", "不可用" if info.get("gpu_total") is None else self.format_bytes(max(0,info.get("gpu_total",0)-info.get("gpu_used",0)))),("磁盘写入延迟",number(info.get("disk_write_latency")," ms")+" · fsync 探针"),("SQLite 写入延迟",number(info.get("sqlite_latency")," ms")+" · 实际事务计时"),("当前截图频率","约 {:.2f} FPS".format(info.get("capture_fps",0.0))),("当前截图分辨率",info.get("capture_resolution","未知")),("鼠标队列年龄","{:.2f} 秒".format(info.get("queue_age",0.0))),("流水线队列年龄","{:.2f} 秒".format(info.get("pipeline_queue_age",0.0))),("当前资源状态",info.get("resource_state","正常")),("限速原因",info.get("pause_reason","无")),("经验池大小",self.format_bytes(info["pool_size"])+"（容量账本；更多信息时已校验）"),("容量阶段","{}% 预警；事务预留 {}".format(capacity.get("tier",0),self.format_bytes(capacity.get("transaction_reserve",0)))),("经验池硬状态", "已停止采集；{} / 目标 {}".format(self.format_bytes(capacity.get("remaining",0)),self.format_bytes(capacity.get("target",0))) if capacity.get("blocked") else "正常"),("AI 模型数量",str(info["model_count"])),("奖励定义","精确画面评分 − 饥饿值；暂定评分不参与重置或训练"),("检索保障","在线仅 LSH、近期与高价值候选；精确评分仅在睡眠执行"),("资源保护","字节预算、队列限速、降分辨率、SQLite/磁盘延迟监测、硬容量阻断")]
        for index,(name,value) in enumerate(rows):
            Label(content,text=name,bg="#f8fafc",fg="#475569",font=("Microsoft YaHei UI",10,"bold"),anchor="w").grid(row=index,column=0,sticky="w",pady=5)
            Label(content,text=value,bg="#f8fafc",fg="#0f172a",font=("Microsoft YaHei UI",10),anchor="w",wraplength=410,justify="left").grid(row=index,column=1,sticky="ew",padx=(18,0),pady=5)

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
