import ctypes
import datetime
import shutil
import subprocess
from bisect import bisect_right
from dataclasses import dataclass
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
MonitorEnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(RECT), wintypes.LPARAM)

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

@dataclass
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

class HardwareProbe:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_probe = 0.0
        self.last_runtime_probe = 0.0
        self.gpus = []
        self.runtime = {"available": False, "source": "Windows GPU 性能计数器不可用", "program_gpu": None, "ldplayer_gpu": None, "gpu_engine": None, "dedicated_used": None, "dedicated_total": None, "dedicated_free": None}
        self.backend = "CPU 规则策略"

    def probe(self):
        now = time.monotonic()
        with self.lock:
            if now - self.last_probe < 5.0 and self.gpus:
                return [dict(item) for item in self.gpus]
            gpus = self._probe_windows_gpus()
            self.gpus = gpus or [{"name": "未检测到可用 GPU 清单", "adapter_type": "未知", "hardware": False, "software": False, "dedicated_total": None, "dedicated_used": None, "dedicated_free": None, "utilization": None, "engine_utilization": None, "ldplayer": None, "program": None, "sampling_source": "Windows GPU 性能计数器"}]
            self.backend = "CPU 规则策略"
            self.last_probe = now
            return [dict(item) for item in self.gpus]

    def _probe_windows_gpus(self):
        if os.name != "nt":
            return []
        items = []
        try:
            raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,VideoProcessor,PNPDeviceID | ConvertTo-Json -Compress"], timeout=3.0, creationflags=0x08000000).decode("utf-8", "ignore").strip()
            data = json.loads(raw) if raw else []
            if isinstance(data, dict):
                data = [data]
            for index, item in enumerate(data):
                name = str(item.get("Name") or "GPU {}".format(index + 1))
                dedicated = int(item.get("AdapterRAM") or 0) or None
                lower = name.lower()
                software = any(x in lower for x in ("basic render", "software", "warp", "remote"))
                discrete = any(x in lower for x in ("nvidia", "radeon", "arc"))
                items.append({"name": name, "adapter_type": "独立 GPU" if discrete else "集成/通用 GPU", "hardware": not software, "software": software, "dedicated_total": dedicated, "dedicated_used": None, "dedicated_free": None, "utilization": None, "engine_utilization": None, "ldplayer": None, "program": None, "sampling_source": "Win32_VideoController 清单"})
        except Exception:
            pass
        return items

    def runtime_metrics(self, program_pid, ldplayer_pids):
        now = time.monotonic()
        with self.lock:
            if now - self.last_runtime_probe < 2.0:
                return dict(self.runtime)
        result = {"available": False, "source": "Windows GPU 性能计数器不可用", "program_gpu": None, "ldplayer_gpu": None, "gpu_engine": None, "dedicated_used": None, "dedicated_total": None, "dedicated_free": None}
        try:
            command = "Get-Counter -ErrorAction Stop -Counter '\\GPU Engine(*)\\Utilization Percentage','\\GPU Adapter Memory(*)\\Dedicated Usage','\\GPU Adapter Memory(*)\\Dedicated Limit' | Select-Object -ExpandProperty CounterSamples | Select-Object Path,CookedValue | ConvertTo-Json -Compress"
            raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", command], timeout=4.0, creationflags=0x08000000).decode("utf-8", "ignore").strip()
            rows = json.loads(raw) if raw else []
            if isinstance(rows, dict):
                rows = [rows]
            engine_total = 0.0
            program_total = 0.0
            ld_total = 0.0
            dedicated_used = 0.0
            dedicated_limit = 0.0
            for row in rows:
                metric_path = str(row.get("Path") or "").lower()
                value = max(0.0, float(row.get("CookedValue") or 0.0))
                if "gpu engine" in metric_path:
                    engine_total += value
                    if "pid_{}".format(int(program_pid)) in metric_path:
                        program_total += value
                    if any("pid_{}".format(int(pid)) in metric_path for pid in ldplayer_pids):
                        ld_total += value
                elif "dedicated usage" in metric_path:
                    dedicated_used += value
                elif "dedicated limit" in metric_path:
                    dedicated_limit += value
            result = {"available": True, "source": "Windows Get-Counter GPU Engine/GPU Adapter Memory", "program_gpu": program_total, "ldplayer_gpu": ld_total, "gpu_engine": engine_total, "dedicated_used": int(dedicated_used), "dedicated_total": int(dedicated_limit) if dedicated_limit else None, "dedicated_free": max(0, int(dedicated_limit - dedicated_used)) if dedicated_limit else None}
        except Exception:
            pass
        with self.lock:
            self.runtime = result
            self.last_runtime_probe = now
            return dict(result)

    def choose_gpu(self, metrics):
        return "CPU"

class ComputeBackend:
    def __init__(self, probe):
        self.probe = probe

    def name(self):
        return "CPU 规则策略"

    def encode_frames(self, frames, budget):
        return list(frames[:max(1, min(len(frames), budget.max_batch))])

    def infer_policy(self, features, budget):
        score = float(features.get("score", 0.0)) if features else 0.0
        return {"backend": "CPU 规则策略", "confidence": max(0.0, min(1.0, score)), "uncertainty": 1.0 - max(0.0, min(1.0, score)), "executed": True}

class GpuScheduler:
    def __init__(self, probe):
        self.probe = probe

    def assign(self, metrics):
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
        self.previous = None
        self.process_previous = {}
        self.process_last_sample = {}
        self.last_sample = 0.0
        self.last_disk_probe = 0.0
        self.window = []
        self.metrics = {"cpu": 0.0, "process_cpu": 0.0, "process_memory": 0, "ldplayer_cpu": 0.0, "ldplayer_memory": 0, "memory": 0.0, "avail_memory": 0, "commit_free": 0, "disk_free": 0, "disk_write_latency": None, "sqlite_latency": 0.0, "capture_latency": 0.0, "queue": 0, "queue_age": 0.0, "pipeline_queue": 0, "pipeline_queue_age": 0.0, "gpu": None, "gpu_dedicated_total": None, "gpu_dedicated_used": None, "gpu_dedicated_free": None, "gpu_engine": None, "ldplayer_gpu": None, "program_gpu": None, "gpu_metrics_available": False, "gpu_sampling_source": "Windows GPU 性能计数器不可用", "last_user_input": time.time(), "capture_failure_rate": 0.0, "metric_sources": {"本程序 CPU": "GetProcessTimes", "雷电 CPU": "GetProcessTimes（雷电进程）", "本程序 GPU 引擎": "Windows GPU 性能计数器", "雷电 GPU 引擎": "Windows GPU 性能计数器", "可用显存": "Windows GPU Adapter Memory 性能计数器", "磁盘写入延迟": "fsync 探针", "SQLite 写入延迟": "实际 SQLite 事务计时", "队列年龄": "队列记录时间戳"}}
        self.probe = HardwareProbe()
        self.backend = ComputeBackend(self.probe)
        self.gpu_scheduler = GpuScheduler(self.probe)
        self.runtime = ModelRuntime(self.backend)
        self.levels = {}
        self.last_pressure = 0.0
        self.healthy_since = time.monotonic()
        self.pressure_reasons = []
        self.sample()

    def set_storage_path(self, path):
        with self.lock:
            self.storage_path = Path(path)

    def set_emulator_path(self, path):
        with self.lock:
            self.emulator_path = str(path or "")

    def update_queue(self, length, age=0.0):
        with self.lock:
            self.metrics["queue"] = int(length)
            self.metrics["queue_age"] = float(age)

    def update_pipeline_queue(self, length, age=0.0):
        with self.lock:
            self.metrics["pipeline_queue"] = int(length)
            self.metrics["pipeline_queue_age"] = float(age)
            self.metrics["queue_age"] = max(float(self.metrics.get("queue_age", 0.0)), float(age))

    def update_capture_metrics(self, elapsed_ms=None, failed=False):
        with self.lock:
            if elapsed_ms is not None:
                self.metrics["capture_latency"] = float(elapsed_ms)
            old = float(self.metrics.get("capture_failure_rate", 0.0))
            self.metrics["capture_failure_rate"] = old * 0.95 + (0.05 if failed else 0.0)

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
        name = Path(self.emulator_path).name.lower() if self.emulator_path else "dnplayer.exe"
        ld_cpu = 0.0
        ld_memory = 0
        for pid in processes_for_name(name):
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

    def sample(self):
        now = time.monotonic()
        with self.lock:
            if now - self.last_sample < 0.5 and self.window:
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
            memory = self.metrics.get("memory", 0.0); avail_memory = self.metrics.get("avail_memory", 0); commit_free = self.metrics.get("commit_free", 0)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                memory = float(status.dwMemoryLoad)
                avail_memory = int(status.ullAvailPhys)
                commit_free = max(0, int(status.ullAvailPageFile))
            try:
                disk_free = int(shutil.disk_usage(self.storage_path).free)
            except Exception:
                disk_free = 0
            process_cpu, process_memory, ld_cpu, ld_memory = self._process_metrics(now)
            ld_name = Path(self.emulator_path).name.lower() if self.emulator_path else "dnplayer.exe"
            runtime = self.probe.runtime_metrics(os.getpid(), processes_for_name(ld_name))
            self.probe.probe()
            disk_latency = self._disk_probe_latency(now)
            self.metrics.update({"cpu": cpu, "memory": memory, "avail_memory": avail_memory, "commit_free": commit_free, "disk_free": disk_free, "process_cpu": process_cpu, "process_memory": process_memory, "ldplayer_cpu": ld_cpu, "ldplayer_memory": ld_memory, "gpu": runtime.get("gpu_engine"), "gpu_dedicated_total": runtime.get("dedicated_total"), "gpu_dedicated_used": runtime.get("dedicated_used"), "gpu_dedicated_free": runtime.get("dedicated_free"), "gpu_engine": runtime.get("gpu_engine"), "program_gpu": runtime.get("program_gpu"), "ldplayer_gpu": runtime.get("ldplayer_gpu"), "gpu_metrics_available": bool(runtime.get("available")), "gpu_sampling_source": runtime.get("source"), "disk_write_latency": disk_latency})
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
        result["backend"] = "CPU 规则策略"
        result["gpus"] = self.probe.probe()
        result["resource_state"] = "暂停" if self.pressure_reasons else "正常"
        result["pause_reason"] = "；".join(self.pressure_reasons)
        return result

    def acquire(self, task):
        sample = self.sample()
        now = time.monotonic()
        reasons = []
        hard = False
        if sample.get("avail_memory", 0) < 384 * 1024 * 1024:
            hard = True; reasons.append("可用内存不足")
        if sample.get("disk_free", 0) < 1024 * 1024 * 1024:
            hard = True; reasons.append("磁盘剩余空间不足 1 GB")
        if sample.get("queue_age", 0.0) > 2.0:
            hard = True; reasons.append("高优先级队列等待超过 2 秒")
        yellow = []
        if float(sample.get("capture_latency_p95", sample.get("capture_latency", 0.0)) or 0.0) > 150.0:
            yellow.append("截图 P95 延迟过高")
        if float(sample.get("sqlite_latency_p95", sample.get("sqlite_latency", 0.0)) or 0.0) > 100.0:
            yellow.append("SQLite 写入 P95 延迟过高")
        if float(sample.get("disk_write_latency", 0.0) or 0.0) > 150.0:
            yellow.append("磁盘 fsync 延迟过高")
        if sample.get("queue_age", 0.0) > 1.0:
            yellow.append("队列等待超过 1 秒")
        pressure = hard or bool(yellow)
        with self.lock:
            if pressure:
                self.last_pressure = now
                self.healthy_since = now
                for key in ("capture", "ai_inference", "sleep_training", "maintenance"):
                    self.levels[key] = max(1, int(self.levels.get(key, 4) / 2))
            elif now - self.healthy_since >= 20.0 and now - self.last_pressure >= 5.0:
                self.levels[task] = min(16, int(self.levels.get(task, 4)) + 1)
                self.healthy_since = now
            level = int(self.levels.get(task, 4))
            self.pressure_reasons = reasons + yellow
        cores = max(1, os.cpu_count() or 1)
        workers = 1 if task in ("maintenance", "ai_inference") else max(1, min(max(1, cores - max(1, math.ceil(cores * 0.25))), level))
        interval_base = {"capture": 1.0, "ai_inference": 0.8, "sleep_training": 0.05, "maintenance": 1.0}.get(task, 1.0)
        interval = max(0.05, interval_base * max(1.0, 4.0 / max(1, level)))
        max_batch = max(1, min(64, level * (2 if task == "sleep_training" else 1)))
        resolution = (640, 360) if level >= 4 else (426, 240) if level >= 2 else (320, 180)
        pause = hard or (task == "sleep_training" and bool(yellow)) or (task == "maintenance" and bool(yellow))
        state = "暂停" if pause else "降速" if yellow else "正常"
        return ResourceBudget(not pause, interval, max_batch, workers, "CPU", 0, resolution, pause, "；".join(reasons + yellow), state, max(8, min(512, level * 16)), max(8, min(128, level * 8)), max(8, min(256, level * 16)), 1)

class Settings:
    def __init__(self):
        appdata = Path(os.environ.get("APPDATA", str(Path.home())))
        self.path = appdata / "LDTrainingPanel" / "settings.json"
        self.data = {"emulator_path": r"D:\LDPlayer9\dnplayer.exe", "storage_path": r"C:\Users\Administrator\Desktop\AAA", "experience_limit": 10 * 1024 * 1024 * 1024, "model_limit": 100, "emulator_pid": 0, "emulator_title": ""}
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
        for key, minimum, maximum in (("experience_limit", int(0.1*1024*1024*1024), 4096*1024*1024*1024), ("model_limit", 1, 100000), ("emulator_pid", 0, 2**31-1)):
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
                rule TEXT NOT NULL
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
            "size_bytes": "INTEGER NOT NULL DEFAULT 0", "novelty": "REAL NOT NULL DEFAULT 0", "action_result": "REAL NOT NULL DEFAULT 0", "coverage": "REAL NOT NULL DEFAULT 0", "model_refs": "INTEGER NOT NULL DEFAULT 0", "last_used": "REAL NOT NULL DEFAULT 0", "dhash64": "TEXT", "bucket0": "INTEGER NOT NULL DEFAULT 0", "bucket1": "INTEGER NOT NULL DEFAULT 0", "bucket2": "INTEGER NOT NULL DEFAULT 0", "bucket3": "INTEGER NOT NULL DEFAULT 0", "state_cluster_id": "TEXT", "state_support_count": "INTEGER NOT NULL DEFAULT 1", "action_outcome_information": "REAL NOT NULL DEFAULT 0", "model_dependency_count": "INTEGER NOT NULL DEFAULT 0", "validation_last_used": "REAL NOT NULL DEFAULT 0", "created_monotonic_ns": "INTEGER NOT NULL DEFAULT 0", "capture_backend": "TEXT NOT NULL DEFAULT 'gdi'", "capture_elapsed_ms": "REAL NOT NULL DEFAULT 0", "capture_complete": "INTEGER NOT NULL DEFAULT 1", "brightness": "REAL NOT NULL DEFAULT 0", "variance": "REAL NOT NULL DEFAULT 0", "gray32x18": "TEXT", "edge_density": "REAL NOT NULL DEFAULT 0", "color_histogram": "TEXT", "asset_ref_count": "INTEGER NOT NULL DEFAULT 1", "score_candidate_count": "INTEGER NOT NULL DEFAULT 0", "score_top_k_distance": "REAL NOT NULL DEFAULT 64", "score_retrieval_fallback": "INTEGER NOT NULL DEFAULT 0", "score_retrieval_mode": "TEXT NOT NULL DEFAULT 'warmup'", "score_exact_or_approx": "TEXT NOT NULL DEFAULT 'exact'", "score_recall_guard": "INTEGER NOT NULL DEFAULT 0", "score_valid": "INTEGER NOT NULL DEFAULT 0", "capture_started_monotonic_ns": "INTEGER NOT NULL DEFAULT 0", "capture_finished_monotonic_ns": "INTEGER NOT NULL DEFAULT 0", "capture_started": "REAL NOT NULL DEFAULT 0", "capture_finished": "REAL NOT NULL DEFAULT 0"
        }
        for name, definition in additions.items():
            if name not in frame_columns:
                self.conn.execute(f"ALTER TABLE frames ADD COLUMN {name} {definition}")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_buckets ON frames(bucket0, bucket1, bucket2, bucket3)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_cluster ON frames(state_cluster_id, state_support_count)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_session_mono ON frames(session_id, created_monotonic_ns)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mouse_session_mono ON mouse_events(session_id, created_monotonic_ns)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS mouse_loss_events (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, created REAL NOT NULL, started REAL NOT NULL, ended REAL NOT NULL, lost_count INTEGER NOT NULL, rule TEXT NOT NULL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS mouse_compression_segments (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source TEXT NOT NULL, started REAL NOT NULL, ended REAL NOT NULL, started_monotonic_ns INTEGER NOT NULL, ended_monotonic_ns INTEGER NOT NULL, start_x INTEGER NOT NULL, start_y INTEGER NOT NULL, end_x INTEGER NOT NULL, end_y INTEGER NOT NULL, original_count INTEGER NOT NULL, max_speed REAL NOT NULL, path_length REAL NOT NULL, rule TEXT NOT NULL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS pipeline_loss_events (id TEXT PRIMARY KEY, session_id TEXT, created REAL NOT NULL, started REAL NOT NULL, ended REAL NOT NULL, lost_count INTEGER NOT NULL, stage TEXT NOT NULL, reason TEXT NOT NULL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS deletion_journal (id TEXT PRIMARY KEY, object_type TEXT NOT NULL, object_id TEXT NOT NULL, path TEXT, stage TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, error TEXT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS pool_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
        self.conn.execute("INSERT OR IGNORE INTO pool_meta(key, value) VALUES ('total_asset_bytes', COALESCE((SELECT SUM(size_bytes) FROM frames), 0))")
        for key in ("pool_capacity_blocked", "pool_capacity_target", "pool_capacity_remaining", "pool_capacity_updated"):
            self.conn.execute("INSERT OR IGNORE INTO pool_meta(key, value) VALUES (?, 0)", (key,))
        self.conn.execute("CREATE TABLE IF NOT EXISTS state_clusters (cluster_id TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS frame_lsh (key INTEGER NOT NULL, frame_id TEXT NOT NULL, PRIMARY KEY(key, frame_id))")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frame_lsh_key ON frame_lsh(key)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS ingestion_journal (id TEXT PRIMARY KEY, object_type TEXT NOT NULL, object_id TEXT NOT NULL, path TEXT, stage TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, error TEXT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS action_outcomes (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), mouse_event_id TEXT NOT NULL, before_frame_id TEXT NOT NULL, after_frame_id TEXT NOT NULL, action_time INTEGER NOT NULL, post_action_delay_ms REAL NOT NULL, score_delta REAL NOT NULL, reward_delta REAL NOT NULL, outcome_valid INTEGER NOT NULL)")
        action_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(action_outcomes)").fetchall()}
        action_additions = {"action_id": "TEXT", "split_role": "TEXT NOT NULL DEFAULT 'unknown'", "hunger_delta_expected": "REAL NOT NULL DEFAULT 0", "baseline_score_delta": "REAL NOT NULL DEFAULT 0", "action_advantage": "REAL NOT NULL DEFAULT 0", "stability": "REAL NOT NULL DEFAULT 0", "baseline_count": "INTEGER NOT NULL DEFAULT 0"}
        for name, definition in action_additions.items():
            if name not in action_columns:
                self.conn.execute(f"ALTER TABLE action_outcomes ADD COLUMN {name} {definition}")
        self.conn.execute("UPDATE action_outcomes SET action_id=COALESCE(action_id, mouse_event_id) WHERE action_id IS NULL OR action_id='' ")
        self.conn.execute("DELETE FROM action_outcomes WHERE rowid NOT IN (SELECT MAX(rowid) FROM action_outcomes GROUP BY action_id)")
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_action_outcomes_action_id ON action_outcomes(action_id)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS model_frame_refs (model_id TEXT NOT NULL, frame_id TEXT NOT NULL REFERENCES frames(id), role TEXT NOT NULL, PRIMARY KEY(model_id, frame_id, role))")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_model_frame_refs_frame ON model_frame_refs(frame_id)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS model_metadata (id TEXT PRIMARY KEY, file_name TEXT NOT NULL UNIQUE, created REAL NOT NULL, quality REAL NOT NULL DEFAULT 0, validation_quality REAL NOT NULL DEFAULT 0, champion INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL)")
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

    def nearest_hashes(self, dhash, limit=8, strict=True, candidate_limit=None):
        try:
            current = int(dhash, 16)
        except Exception:
            return {"hashes": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "invalid_hash", "exact_or_approx": "exact", "recall_guard": False, "total_history": 0, "score_valid": False}
        limit = max(1, int(limit))
        ranked = []
        with self.lock:
            if self.conn is None:
                return {"hashes": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "store_closed", "exact_or_approx": "exact", "recall_guard": False, "total_history": 0, "score_valid": False}
            total = int(self.conn.execute("SELECT COUNT(*) FROM frames WHERE dhash64 IS NOT NULL OR phash IS NOT NULL").fetchone()[0] or 0)
            if total <= 0:
                return {"hashes": [], "candidate_count": 0, "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "warmup_no_history", "exact_or_approx": "exact", "recall_guard": True, "total_history": 0, "score_valid": False}
            cursor = self.conn.execute("SELECT id, dhash64, phash FROM frames WHERE dhash64 IS NOT NULL OR phash IS NOT NULL")
            scanned = 0
            while True:
                rows = cursor.fetchmany(max(512, min(8192, int(candidate_limit or 4096))))
                if not rows:
                    break
                for _, stored_dhash, stored_phash in rows:
                    try:
                        stored = stored_dhash or stored_phash
                        ranked.append((bit_count(current ^ int(stored, 16)), stored))
                        scanned += 1
                    except Exception:
                        pass
        ranked.sort(key=lambda item: item[0])
        top = ranked[:min(limit, len(ranked))]
        complete = len(top) >= min(limit, total)
        return {"hashes": [item[1] for item in top], "candidate_count": scanned, "top_k_distance": float(top[-1][0] if top else 64), "retrieval_fallback": True, "retrieval_mode": "full_exact_guard", "exact_or_approx": "exact", "recall_guard": complete, "total_history": total, "score_valid": bool(top) and complete}

    def record_mouse_loss(self, session_id, started, ended, count, rule):
        with self.lock:
            if self.conn is None or not session_id or count <= 0:
                return
            self.conn.execute("INSERT INTO mouse_loss_events(id, session_id, created, started, ended, lost_count, rule) VALUES (?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, session_id, time.time(), started, ended, int(count), rule))
            self.conn.commit()

    def record_mouse_compression(self, segment):
        if not segment or not segment.get("session_id"):
            return
        with self.lock:
            if self.conn is None:
                return
            self.conn.execute("INSERT INTO mouse_compression_segments(id, session_id, source, started, ended, started_monotonic_ns, ended_monotonic_ns, start_x, start_y, end_x, end_y, original_count, max_speed, path_length, rule) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, segment["session_id"], segment.get("source", "user"), float(segment["started"]), float(segment["ended"]), int(segment["started_ns"]), int(segment["ended_ns"]), int(segment["start_x"]), int(segment["start_y"]), int(segment["end_x"]), int(segment["end_y"]), int(segment["count"]), float(segment.get("max_speed", 0.0)), float(segment.get("path_length", 0.0)), str(segment.get("rule", "移动事件降采样"))))
            self.conn.commit()

    def record_pipeline_loss(self, session_id, started, ended, count, stage, reason):
        if count <= 0:
            return
        with self.lock:
            if self.conn is None:
                return
            self.conn.execute("INSERT INTO pipeline_loss_events(id, session_id, created, started, ended, lost_count, stage, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, session_id, time.time(), float(started), float(ended), int(count), str(stage), str(reason)))
            self.conn.commit()

    def save_frame(self, session_id, image, phash, score, hunger, reward):
        with self.lock:
            if self.capacity_status().get("blocked"):
                raise PoolCapacityBlocked("经验池仍高于回收目标，已停止写入新截图并等待清理完成")
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
                self.conn.execute("UPDATE frames SET score_retrieval_mode=?, score_exact_or_approx=?, score_recall_guard=?, score_valid=? WHERE id=?", (str(image.get("score_retrieval_mode", "warmup")), str(image.get("score_exact_or_approx", "exact")), 1 if image.get("score_recall_guard") else 0, 1 if image.get("score_valid") else 0, identifier))
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
        return identifier


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

    def training_readiness(self, minimum_actions=30, minimum_sessions=1):
        with self.lock:
            sessions = int(self.conn.execute("SELECT COUNT(DISTINCT session_id) FROM frames WHERE capture_complete=1").fetchone()[0] or 0)
            actions = int(self.conn.execute("SELECT COUNT(*) FROM mouse_events WHERE event_type IN ('button_up','wheel','move')").fetchone()[0] or 0)
            frames = int(self.conn.execute("SELECT COUNT(*) FROM frames WHERE capture_complete=1").fetchone()[0] or 0)
        return {"ready": sessions >= minimum_sessions and frames >= 24 and actions >= minimum_actions, "sessions": sessions, "actions": actions, "frames": frames}

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
            sessions = [row[0] for row in self.conn.execute("SELECT id FROM sessions ORDER BY started DESC LIMIT 32").fetchall()]
            frames = {}
            mouse = {}
            for session_id in sessions:
                frame_rows = self.conn.execute("""
                    SELECT id, session_id, created_monotonic_ns, created, dhash64, phash, score, reward, hunger, width, height, gray32x18, edge_density, color_histogram, capture_started_monotonic_ns, capture_finished_monotonic_ns, score_valid
                    FROM frames
                    WHERE session_id=? AND capture_complete=1 AND created_monotonic_ns>0
                    ORDER BY created_monotonic_ns ASC
                    LIMIT 9000
                """, (session_id,)).fetchall()
                mouse_rows = self.conn.execute("""
                    SELECT id, session_id, created_monotonic_ns, created, event_type, source, relative_x, relative_y, speed, dx, dy, direction, button, wheel
                    FROM mouse_events
                    WHERE session_id=? AND created_monotonic_ns>0
                    ORDER BY created_monotonic_ns ASC
                    LIMIT 24000
                """, (session_id,)).fetchall()
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
            rows.append((item.get("action_id") or item["mouse_event_id"], uuid.uuid4().hex, item["session_id"], item["mouse_event_id"], item["before_frame_id"], item["after_frame_id"], item["action_time"], item["post_action_delay_ms"], item["score_delta"], item["reward_delta"], float(item.get("hunger_delta_expected", 0.0)), float(item.get("baseline_score_delta", 0.0)), float(item.get("action_advantage", 0.0)), float(item.get("stability", 0.0)), int(item.get("baseline_count", 0)), 1 if item.get("outcome_valid") else 0, item.get("split_role", "unknown")))
        with self.lock:
            self.conn.executemany("INSERT INTO action_outcomes(action_id, id, session_id, mouse_event_id, before_frame_id, after_frame_id, action_time, post_action_delay_ms, score_delta, reward_delta, hunger_delta_expected, baseline_score_delta, action_advantage, stability, baseline_count, outcome_valid, split_role) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(action_id) DO UPDATE SET session_id=excluded.session_id, mouse_event_id=excluded.mouse_event_id, before_frame_id=excluded.before_frame_id, after_frame_id=excluded.after_frame_id, action_time=excluded.action_time, post_action_delay_ms=excluded.post_action_delay_ms, score_delta=excluded.score_delta, reward_delta=excluded.reward_delta, hunger_delta_expected=excluded.hunger_delta_expected, baseline_score_delta=excluded.baseline_score_delta, action_advantage=excluded.action_advantage, stability=excluded.stability, baseline_count=excluded.baseline_count, outcome_valid=excluded.outcome_valid, split_role=excluded.split_role", rows)
            self.conn.executemany("UPDATE mouse_events SET before_frame_id=?, after_frame_id=?, action_time=?, post_action_delay_ms=?, score_delta=?, reward_delta=?, outcome_valid=? WHERE id=?", [(item["before_frame_id"], item["after_frame_id"], item["action_time"], item["post_action_delay_ms"], item["score_delta"], item["reward_delta"], 1 if item.get("outcome_valid") else 0, item["mouse_event_id"]) for item in outcomes])
            self.conn.commit()

    def _recalculate_model_refs_locked(self, frame_ids):
        for frame_id in set(frame_ids or []):
            self.conn.execute("UPDATE frames SET model_dependency_count=(SELECT COUNT(DISTINCT model_id) FROM model_frame_refs WHERE model_frame_refs.frame_id=frames.id), model_refs=(SELECT COUNT(DISTINCT model_id) FROM model_frame_refs WHERE model_frame_refs.frame_id=frames.id) WHERE id=?", (frame_id,))

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
        seen = set()
        for path in self.model_files():
            identifier, payload = self._read_model_payload(path)
            seen.add(identifier)
            self.conn.execute("INSERT INTO model_metadata(id, file_name, created, quality, validation_quality, champion, updated) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET file_name=excluded.file_name, created=excluded.created, quality=excluded.quality, validation_quality=excluded.validation_quality, champion=excluded.champion, updated=excluded.updated", (identifier, path.name, float(payload.get("trained_at", payload.get("validated_at", 0.0)) or 0.0), float(payload.get("quality", 0.0) or 0.0), float(payload.get("validation_quality", payload.get("quality", 0.0)) or 0.0), 1 if payload.get("champion", False) else 0, time.time()))
        stale = [row[0] for row in self.conn.execute("SELECT id FROM model_metadata").fetchall() if row[0] not in seen]
        if stale:
            marks = ",".join("?" for _ in stale)
            frame_ids = [row[0] for row in self.conn.execute("SELECT DISTINCT frame_id FROM model_frame_refs WHERE model_id IN (" + marks + ")", stale).fetchall()]
            self.conn.execute("DELETE FROM model_frame_refs WHERE model_id IN (" + marks + ")", stale)
            self.conn.execute("DELETE FROM model_metadata WHERE id IN (" + marks + ")", stale)
            self._recalculate_model_refs_locked(frame_ids)

    def sync_model_metadata(self):
        with self.lock:
            if self.conn is None:
                return
            self._sync_model_metadata_locked()
            self.conn.commit()

    def register_model_metadata(self, model_id, path, payload):
        with self.lock:
            self.conn.execute("INSERT INTO model_metadata(id, file_name, created, quality, validation_quality, champion, updated) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET file_name=excluded.file_name, created=excluded.created, quality=excluded.quality, validation_quality=excluded.validation_quality, champion=excluded.champion, updated=excluded.updated", (str(model_id), Path(path).name, float(payload.get("trained_at", time.time())), float(payload.get("quality", 0.0)), float(payload.get("validation_quality", payload.get("quality", 0.0))), 1 if payload.get("champion", False) else 0, time.time()))
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

    def pool_breakdown(self):
        result = {"frame_asset_bytes": 0, "database_bytes": 0, "transient_bytes": 0, "experience_total_bytes": 0}
        if self.pool is None:
            return result
        try:
            for item in self.pool.rglob("*"):
                try:
                    if not item.is_file():
                        continue
                    size = int(item.stat().st_size)
                    name = item.name.lower()
                    relative = item.relative_to(self.pool)
                    if relative.parts and relative.parts[0].lower() == "screens":
                        result["frame_asset_bytes"] += size
                    elif name.startswith("records.sqlite3") or name in ("records.sqlite3-wal", "records.sqlite3-shm", "records.sqlite3-journal"):
                        result["database_bytes"] += size
                    elif "trash" in {part.lower() for part in relative.parts} or name.endswith(".tmp") or name.startswith(".write_latency_probe"):
                        result["transient_bytes"] += size
                    result["experience_total_bytes"] += size
                except OSError:
                    pass
        except OSError:
            pass
        return result



    def capacity_status(self):
        with self.lock:
            if self.conn is None:
                return {"blocked": False, "target": 0, "remaining": 0}
            rows = dict(self.conn.execute("SELECT key, value FROM pool_meta WHERE key IN ('pool_capacity_blocked','pool_capacity_target','pool_capacity_remaining','pool_capacity_updated')").fetchall())
        return {"blocked": bool(rows.get("pool_capacity_blocked", 0)), "target": int(rows.get("pool_capacity_target", 0)), "remaining": int(rows.get("pool_capacity_remaining", 0)), "updated": int(rows.get("pool_capacity_updated", 0))}

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

    def pool_size(self):
        return int(self.pool_breakdown().get("experience_total_bytes", 0))

    def _reconcile_asset_bytes(self):
        with self.lock:
            total = self.conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM frames").fetchone()[0]
            self.conn.execute("INSERT OR REPLACE INTO pool_meta(key, value) VALUES ('total_asset_bytes', ?)", (int(total or 0),))
            self.conn.commit()

    def prune_models(self, maximum, cancelled=None, cooperative=None):
        maximum = max(1, int(maximum))
        self.sync_model_metadata()
        summaries = self.model_summaries()
        initial = len(summaries)
        if initial <= maximum:
            self.last_model_prune_result = {"initial": initial, "removed": 0, "target": initial, "remaining": initial, "success": True}
            return 0
        target = max(0, int(math.floor(maximum * 0.5)))
        removed = 0
        trash_root = self.models / ".trash"
        trash_root.mkdir(parents=True, exist_ok=True)
        for index, (path, quality, trained_at) in enumerate(sorted(summaries, key=lambda value: (value[1], value[2], value[0].name))):
            if initial - removed <= target:
                break
            if cancelled is not None and cancelled():
                break
            if index % 8 == 0 and cooperative is not None and not cooperative():
                break
            model_id, _ = self._read_model_payload(path)
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
                    self.conn.execute("DELETE FROM model_metadata WHERE id=?", (model_id,))
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
        remaining = len(self.model_files())
        success = remaining <= target
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

    def prune_experience(self, maximum, cancelled, progress, cooperative=None):
        maximum = max(1, int(maximum))
        self.recover_deletions()
        current = self.pool_size()
        target = int(math.floor(maximum * 0.5))
        if current <= maximum:
            if current <= target:
                self._set_capacity_status(False, target, current)
            self.last_experience_prune_result = {"initial": current, "removed": 0, "target": target, "remaining": current, "success": current <= maximum}
            return 0, current
        removed = 0
        for force in (False, True):
            while current > target and not cancelled():
                if cooperative is not None and not cooperative():
                    time.sleep(0.25)
                    continue
                with self.lock:
                    guard = "frames.model_dependency_count=0 AND frames.model_refs=0"
                    if not force:
                        guard += " AND frames.asset_ref_count<=1 AND COALESCE(state_clusters.count, frames.state_support_count, 1)>1 AND frames.validation_last_used<?"
                        params = (time.time() - 3600.0,)
                    else:
                        params = ()
                    rows = self.conn.execute("""SELECT frames.id, frames.screenshot_path, frames.size_bytes
                        FROM frames LEFT JOIN state_clusters ON state_clusters.cluster_id=frames.state_cluster_id
                        WHERE {} ORDER BY
                        (CASE WHEN COALESCE(state_clusters.count, frames.state_support_count, 1)>1 THEN 1 ELSE 0 END) DESC,
                        frames.model_dependency_count ASC, frames.model_refs ASC, frames.reward ASC, frames.action_outcome_information ASC, frames.created ASC
                        LIMIT 256""".format(guard), params).fetchall()
                if not rows:
                    break
                removed += self._delete_frame_batch(rows)
                self._cleanup_pool_files()
                self._compact_database(cooperative)
                current = self.pool_size()
                progress(min(95.0, 56.0 + 39.0 * min(1.0, max(0, maximum - current) / max(1, maximum - target))))
            if current <= target or cancelled():
                break
        self._cleanup_pool_files()
        self._compact_database(cooperative)
        remaining = self.pool_size()
        success = remaining <= target
        self.last_experience_prune_result = {"initial": current + removed, "removed": removed, "target": target, "remaining": remaining, "success": success, "breakdown": self.pool_breakdown()}
        if success:
            self._set_capacity_status(False, target, remaining)
        elif not cancelled():
            self._set_capacity_status(True, target, remaining)
            self.add_system_event(None, "pool_prune_hard_block", dict(self.last_experience_prune_result))
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
    title = window_title(hwnd)
    if window_is_transparent(hwnd):
        return "transparent_overlay"
    if window_is_cloaked(hwnd):
        return "cloaked_overlay"
    lower = title.lower()
    if any(item in lower for item in ("tooltip", "工具提示", "notification", "通知", "windows security", "安全中心")):
        return "system_prompt"
    return "real_obstruction"

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
            kind = window_obstruction_kind(hit)
            pid = wintypes.DWORD(); user32.GetWindowThreadProcessId(hit, ctypes.byref(pid))
            details = {"kind": kind, "title": window_title(hit), "pid": int(pid.value), "point": (int(x), int(y)), "rect": window_rectangle(hit)}
            if kind in ("transparent_overlay", "cloaked_overlay"):
                client_unobscured.last_overlay = details
                continue
            client_unobscured.last_obstruction = details
            return False
    above = user32.GetWindow(own_root, GW_HWNDPREV)
    checked = set()
    while above and above not in checked:
        checked.add(above)
        if root_window(above) == own_root or not user32.IsWindowVisible(above) or user32.IsIconic(above):
            above = user32.GetWindow(above, GW_HWNDPREV); continue
        candidate = window_rectangle(above)
        if candidate is not None and rectangle_overlap(candidate, rect):
            kind = window_obstruction_kind(above)
            pid = wintypes.DWORD(); user32.GetWindowThreadProcessId(above, ctypes.byref(pid))
            details = {"kind": kind, "title": window_title(above), "pid": int(pid.value), "rect": candidate}
            if kind in ("transparent_overlay", "cloaked_overlay"):
                client_unobscured.last_overlay = details
            else:
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

def capture_client(hwnd, max_width=640, max_height=360):
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
        return {"width": width, "height": height, "rgb": bytes(rgb), "capture_started_monotonic_ns": capture_started_monotonic_ns, "capture_finished_monotonic_ns": finished_ns, "capture_started": capture_started, "capture_finished": finished, "capture_backend": "gdi", "capture_elapsed_ms": (finished_ns-capture_started_monotonic_ns)/1000000.0}
    finally:
        if old_object and memory_dc: gdi32.SelectObject(memory_dc, old_object)
        if bitmap: gdi32.DeleteObject(bitmap)
        if memory_dc: gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(hwnd, source_dc)

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
    image.update({"phash": f"{perceptual:016x}", "dhash64": f"{value:016x}", "capture_complete": 1 if mean >= 3.0 and variance >= 2.0 else 0, "brightness": mean, "variance": variance, "gray32x18": bytes(sample_gray).hex(), "edge_density": edge_hits/(18*31), "color_histogram": json.dumps(hist,separators=(",",":"))})
    return image

def compress_frame_png(image):
    image["png"] = encode_png(int(image["width"]), int(image["height"]), image.pop("rgb"))
    return image

def bit_count(value):
    try:
        return value.bit_count()
    except AttributeError:
        return bin(value).count("1")

def frame_score(dhash, historical):
    details = historical if isinstance(historical, dict) else {"hashes": historical or [], "candidate_count": len(historical or []), "top_k_distance": 64.0, "retrieval_fallback": False, "retrieval_mode": "legacy", "exact_or_approx": "unknown", "recall_guard": False, "score_valid": False}
    hashes = details.get("hashes", [])
    meta = {"candidate_count": int(details.get("candidate_count", 0)), "top_k_distance": float(details.get("top_k_distance", 64.0)), "retrieval_fallback": bool(details.get("retrieval_fallback", False)), "retrieval_mode": str(details.get("retrieval_mode", "unknown")), "exact_or_approx": str(details.get("exact_or_approx", "unknown")), "recall_guard": bool(details.get("recall_guard", False)), "score_valid": bool(details.get("score_valid", False))}
    if not hashes or not meta["recall_guard"]:
        return 0.0, meta
    try:
        current = int(dhash, 16)
    except Exception:
        return 0.0, dict(meta, score_valid=False)
    weighted=[]
    for index, previous in enumerate(hashes):
        try:
            distance=bit_count(current ^ int(previous,16)); weighted.append((1.0-distance/64.0,1.0/(1.0+index),distance))
        except Exception:
            pass
    if not weighted:
        return 0.0, dict(meta, score_valid=False)
    similarity=sum(v*w for v,w,_ in weighted)/max(1e-9,sum(w for _,w,_ in weighted))
    meta["top_k_distance"]=float(max(d for _,_,d in weighted)); meta["score_valid"]=True
    return max(0.0,min(1.0,1.0-similarity)),meta

class AIInputTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = []

    def register(self, event_type, button, wheel, x, y):
        with self.lock:
            self.pending.append({"event_type": event_type, "button": button, "wheel": wheel, "x": x, "y": y, "deadline_ns": time.monotonic_ns() + 500_000_000})

    def consume(self, event_type, button, wheel, x, y, now_ns):
        with self.lock:
            self.pending = [item for item in self.pending if item["deadline_ns"] >= now_ns]
            for index, item in enumerate(self.pending):
                same_type = item["event_type"] == event_type
                same_button = item["button"] == button
                same_wheel = item["wheel"] == wheel
                same_position = abs(item["x"] - x) <= 3 and abs(item["y"] - y) <= 3
                if same_type and same_button and same_wheel and same_position:
                    self.pending.pop(index)
                    return True
        return False

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
        self.resources = ResourceGovernor()
        self.lock = threading.RLock()
        self.state = "idle"
        self.epoch = 0
        self.cancel_event = threading.Event()
        self.target_hwnd = None
        self.target_root = None
        self.target_pid = 0
        self.target_rect = None
        self.session_id = None
        self.session_mode = None
        self.session_started = 0.0
        self.hunger_anchor = 0.0
        self.last_score = None
        self.frame_scores = []
        self.frame_count = 0
        self.mouse_count = 0
        self.last_mouse_by_source = {"ai": None, "user": None, "external_injected": None}
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
        self.ai_input_tracker = AIInputTracker()

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
        self.resources.set_emulator_path(self.settings.data["emulator_path"])
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

    def enqueue_raw_mouse(self, event_type, button, wheel, x, y, created, created_monotonic_ns, flags, extra_info):
        item = (event_type, button, wheel, x, y, created, created_monotonic_ns, flags, extra_info)
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
            write_budget = self.resources.acquire("capture")
            if pending and (len(pending) >= max(8, write_budget.database_batch_size) or time.monotonic() - last_write >= max(0.15, write_budget.next_interval) or self.writer_stop.is_set()):
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

    def classify_mouse_source(self, event_type, button, wheel, x, y, flags, extra_info, created_monotonic_ns):
        injected = bool(flags & (LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED))
        if extra_info == AI_MOUSE_MARKER:
            return "ai"
        if injected and self.ai_input_tracker.consume(event_type, button, wheel, x, y, created_monotonic_ns):
            return "ai"
        if injected:
            return "external_injected"
        return "user"

    def _append_move_compression(self, session_id, source, last_kept, x, y, created, created_ns, speed):
        key = (session_id, source)
        item = self.move_segments.get(key)
        if item is None:
            item = {"session_id": session_id, "source": source, "started": last_kept[3], "ended": created, "started_ns": last_kept[2], "ended_ns": created_ns, "start_x": last_kept[0], "start_y": last_kept[1], "end_x": x, "end_y": y, "count": 1, "max_speed": float(speed), "path_length": math.hypot(x-last_kept[0], y-last_kept[1]), "rule": "10ms 内且位移不足 3px 的同向普通移动压缩"}
            self.move_segments[key] = item
        else:
            item["ended"] = created; item["ended_ns"] = created_ns; item["path_length"] += math.hypot(x-item["end_x"], y-item["end_y"]); item["end_x"] = x; item["end_y"] = y; item["count"] += 1; item["max_speed"] = max(float(item["max_speed"]), float(speed))

    def _flush_move_compression(self, session_id=None, source=None):
        keys = [key for key in self.move_segments if (session_id is None or key[0] == session_id) and (source is None or key[1] == source)]
        for key in keys:
            segment = self.move_segments.pop(key, None)
            if segment:
                try: self.store.record_mouse_compression(segment)
                except Exception: pass

    def on_mouse(self, event_type, button, wheel, x, y, created, created_monotonic_ns, flags, extra_info):
        source = self.classify_mouse_source(event_type, button, wheel, x, y, flags, extra_info, created_monotonic_ns)
        if source == "user": self.resources.update_user_input()
        with self.lock:
            if self.state not in ("learning", "training") or not self.session_id or not self.target_rect: return
            session_id = self.session_id; rect = self.target_rect; outside = not point_inside((x, y), rect)
            previous = self.last_mouse_by_source.get(source)
            critical = event_type != "move" or button or wheel
            if self.state == "training":
                if source == "user" and (critical or (previous is not None and math.hypot(x-previous[0], y-previous[1]) >= 12)): self.on_control_signal("stop", "训练模式检测到真实用户鼠标操作，AI 已停止")
                elif source == "external_injected": self.on_control_signal("stop", "训练模式检测到非本程序注入鼠标操作，AI 已安全停止")
            dx=dy=direction=speed=0.0
            if previous is not None:
                dt=max(0.000001,(created_monotonic_ns-previous[2])/1_000_000_000.0); dx=float(x-previous[0]); dy=float(y-previous[1]); direction=math.atan2(dy,dx) if dx or dy else 0.0; speed=math.hypot(dx,dy)/dt
            kept_map = self.last_move_kept if isinstance(self.last_move_kept, dict) else {}
            last_kept = kept_map.get(source)
            direction_turn = last_kept is not None and previous is not None and abs(math.atan2(math.sin(direction-last_kept[4]), math.cos(direction-last_kept[4]))) >= 0.65
            if not critical and last_kept is not None and (created_monotonic_ns-last_kept[2]) < 10_000_000 and abs(x-last_kept[0]) < 3 and abs(y-last_kept[1]) < 3 and not direction_turn:
                self._append_move_compression(session_id, source, last_kept, x, y, created, created_monotonic_ns, speed)
                self.last_mouse_by_source[source]=(x,y,created_monotonic_ns)
                return
            self._flush_move_compression(session_id, source)
            if not critical: kept_map[source]=(x,y,created_monotonic_ns,created,direction); self.last_move_kept=kept_map
            self.last_mouse_by_source[source]=(x,y,created_monotonic_ns); self.mouse_count += 1
        width=max(1,rect[2]-rect[0]); height=max(1,rect[3]-rect[1])
        record={"session_id":session_id,"created":created,"created_monotonic_ns":int(created_monotonic_ns),"source":source,"event_type":event_type,"button":button,"wheel":wheel,"x":x,"y":y,"relative_x":(x-rect[0])/width,"relative_y":(y-rect[1])/height,"dx":dx,"dy":dy,"direction":direction,"speed":speed}
        try: self.mouse_queue.put_nowait(record)
        except queue.Full:
            if critical: self.on_control_signal("stop", "鼠标高优先级事件队列过载")
            with self.loss_lock:
                item=self.move_loss.get(session_id)
                if item is None:self.move_loss[session_id]=[created,created,1]
                else:item[1]=created;item[2]+=1
        if outside:self.on_control_signal("stop", "鼠标已离开雷电模拟器客户区")

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
        if not self.hook.start(): self.emit("notice", self.hook.error or "鼠标钩子未启动，禁止进入模式。"); return False
        if not self.keyboard_hook.start(): self.emit("notice", self.keyboard_hook.error or "键盘钩子未启动，禁止进入模式。"); self.hook.stop(); return False
        try: self.ensure_store()
        except Exception as error:
            self.emit("notice", "无法创建存储路径：" + str(error)); self.hook.stop(); self.keyboard_hook.stop(); return False
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
            self.target_hwnd = hwnd; self.target_root = root_window(hwnd); self.target_pid = int(pid.value); self.target_rect = rect; self.session_id = session_id; self.session_mode = mode; self.session_started = time.monotonic(); self.hunger_anchor = self.session_started; self.last_observation = self.session_started; self.capture_failures = 0; self.last_score = None; self.frame_scores = []; self.frame_count = 0; self.mouse_count = 0; self.last_mouse_by_source = {"ai": None, "user": None, "external_injected": None}; self.ai_step = 0
            model = self.store.best_model() if mode == "training" else None
            plan = model.get("q_actions", model.get("hotspots", [])) if isinstance(model, dict) and model.get("champion", True) else []
            self.ai_plan = [item for item in plan if isinstance(item, dict)] if isinstance(plan, list) else []
            self.action_limits = {}; self.last_move_kept = {}; self.move_segments = {}; self.pipeline_losses = {}; self.pipeline_stop.clear(); self.stop_requested.clear()
        self.store.add_system_event(session_id, "mode_enter", {"mode": mode, "automatic": automatic, "time": time.time(), "client_rect": rect, "target_pid": self.target_pid, "resource": self.resources.sample()})
        detail = "已进入" + ("学习模式" if mode == "learning" else ("训练模式；无已验证模型，安全观察且不执行点击、右键或滚轮" if not self.ai_plan else "训练模式"))
        self.post_state(detail)
        pipeline = [threading.Thread(target=self._pipeline_feature_loop,args=(token,),name="FeatureScore"), threading.Thread(target=self._pipeline_encode_loop,args=(token,),name="PngEncode"), threading.Thread(target=self._pipeline_persist_loop,args=(token,),name="FramePersist")]
        threads = pipeline + [threading.Thread(target=self._capture_loop,args=(token,),name="CaptureLoop"), threading.Thread(target=self._monitor_loop,args=(token,),name="SessionMonitor")]
        if mode == "training": threads.append(threading.Thread(target=self._ai_loop,args=(token,),name="AIControl"))
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

    def _put_value_packet(self, target_queue, packet, stage):
        try:
            target_queue.put_nowait(packet)
            return True
        except queue.Full:
            incoming=float(packet.get("score",0.0))
            displaced=None
            with target_queue.mutex:
                choices=[(index,item) for index,item in enumerate(target_queue.queue) if isinstance(item,dict)]
                if choices:
                    index,lowest=min(choices,key=lambda value:float(value[1].get("score",0.0)))
                    if float(lowest.get("score",0.0)) < incoming:
                        displaced=target_queue.queue[index]
                        del target_queue.queue[index]
                        target_queue.queue.append(packet)
                        target_queue.not_empty.notify()
            if displaced is not None:
                self._drop_pipeline(displaced.get("session_id"),stage,"队列满时优先淘汰低价值重复帧")
                return True
            self._drop_pipeline(packet.get("session_id"),stage,"队列满且没有可淘汰的低价值帧")
            return False

    def _pipeline_age(self):
        ages=[]
        for q in (self.capture_queue,self.feature_queue,self.persist_queue):
            try:
                first=q.queue[0]
                ages.append(max(0.0,time.time()-float(first.get("queued_at",time.time()))))
            except Exception:pass
        return max(ages or [0.0])

    def _capture_loop(self, token):
        while self._is_current(token, ("learning", "training")):
            capacity=self.store.capacity_status() if self.store.conn else {"blocked":False}
            if capacity.get("blocked"):
                sample=self.resources.sample(); self.emit("state", {"state":self.current_state(),"detail":"经验池硬错误：已停止采集新截图，等待清理完成；当前 {}，目标 {}".format(capacity.get("remaining",0),capacity.get("target",0)),"cpu":sample["cpu"],"memory":sample["memory"]}); time.sleep(1.0); continue
            budget=self.resources.acquire("capture")
            if not budget.allowed: time.sleep(max(0.05,budget.next_interval)); continue
            with self.lock: hwnd=self.target_hwnd; session_id=self.session_id; mode=self.session_mode
            rect=valid_client(hwnd,True) if hwnd else None
            if rect is None:
                obstruction=getattr(client_unobscured,"last_obstruction",None) or {"kind":"client_invalid","reason":getattr(valid_client,"last_reason","未知")}
                if session_id: self.store.add_system_event(session_id,"client_obstruction",obstruction)
                self.request_idle("雷电模拟器客户区异常："+getattr(valid_client,"last_reason","未知"),token); return
            now_observation=time.monotonic(); observation_due=False
            with self.lock:
                self.target_rect=rect
                if now_observation-self.last_observation>=5.0:self.last_observation=now_observation;observation_due=True
            if session_id and observation_due:
                self.store.add_system_event(session_id,"client_observation",{"mode":mode,"client_rect":rect,"cursor":cursor_position(),"resource":self.resources.sample()})
                overlay=getattr(client_unobscured,"last_overlay",None)
                if overlay: self.store.add_system_event(session_id,"client_transparent_overlay",overlay)
            image=capture_client(hwnd,budget.max_capture_resolution[0],budget.max_capture_resolution[1]) if session_id else None
            if image is None:
                self.resources.update_capture_metrics(None,True)
                with self.lock:self.capture_failures+=1;failures=self.capture_failures
                if failures>=12:self.request_idle("连续无法记录雷电模拟器画面",token);return
            else:
                self.resources.update_capture_metrics(image.get("capture_elapsed_ms"),False)
                packet={"session_id":session_id,"image":image,"queued_at":time.time()}
                try:self.capture_queue.put_nowait(packet)
                except queue.Full:self._drop_pipeline(session_id,"capture","截图队列已满，未评分帧因背压丢弃")
                self.resources.update_pipeline_queue(self.capture_queue.qsize()+self.feature_queue.qsize()+self.persist_queue.qsize(),self._pipeline_age())
            time.sleep(budget.next_interval)

    def _pipeline_feature_loop(self, token):
        while not self.pipeline_stop.is_set() or not self.capture_queue.empty():
            try:packet=self.capture_queue.get(timeout=0.15)
            except queue.Empty:continue
            session_id=packet.get("session_id"); image=packet.get("image")
            try:
                image=extract_frame_features(image)
                if not image.get("capture_complete",1): self._drop_pipeline(session_id,"feature","截图内容不完整"); continue
                budget=self.resources.acquire("capture")
                historical=self.store.nearest_hashes(image["dhash64"],8,candidate_limit=budget.retrieval_candidate_limit)
                score,meta=frame_score(image["dhash64"],historical)
                image.update({"score_candidate_count":meta["candidate_count"],"score_top_k_distance":meta["top_k_distance"],"score_retrieval_fallback":1 if meta["retrieval_fallback"] else 0,"score_retrieval_mode":meta["retrieval_mode"],"score_exact_or_approx":meta["exact_or_approx"],"score_recall_guard":meta["recall_guard"],"score_valid":meta["score_valid"]})
                now=time.monotonic()
                with self.lock:
                    hunger=1e-9+max(0.0,now-self.hunger_anchor)*0.00004
                    reset=bool(meta["score_valid"]) and self.last_score is not None and score>self.last_score
                    if reset:hunger=1e-9
                    reward=score-hunger
                packet.update({"image":image,"score":score,"hunger":hunger,"reward":reward,"queued_at":time.time(),"reset_hunger":reset})
                self._put_value_packet(self.feature_queue,packet,"feature")
            except Exception as error:
                self._drop_pipeline(session_id,"feature","特征评分失败:"+str(error))
            finally:self.resources.update_pipeline_queue(self.capture_queue.qsize()+self.feature_queue.qsize()+self.persist_queue.qsize(),self._pipeline_age())

    def _pipeline_encode_loop(self, token):
        while not self.pipeline_stop.is_set() or not self.feature_queue.empty():
            try:packet=self.feature_queue.get(timeout=0.15)
            except queue.Empty:continue
            try:
                packet["image"]=compress_frame_png(packet["image"]);packet["queued_at"]=time.time();self._put_value_packet(self.persist_queue,packet,"encode")
            except queue.Full:self._drop_pipeline(packet.get("session_id"),"encode","PNG/落盘队列已满，丢弃低价值帧")
            except Exception as error:self._drop_pipeline(packet.get("session_id"),"encode","PNG 压缩失败:"+str(error))
            finally:self.resources.update_pipeline_queue(self.capture_queue.qsize()+self.feature_queue.qsize()+self.persist_queue.qsize(),self._pipeline_age())

    def _pipeline_persist_loop(self, token):
        while not self.pipeline_stop.is_set() or not self.persist_queue.empty():
            try:packet=self.persist_queue.get(timeout=0.15)
            except queue.Empty:continue
            try:
                started=time.monotonic(); self.store.save_frame(packet["session_id"],packet["image"],packet["image"]["phash"],packet["score"],packet["hunger"],packet["reward"]); self.resources.update_sqlite_latency((time.monotonic()-started)*1000.0)
                with self.lock:
                    if packet.get("reset_hunger"):self.hunger_anchor=time.monotonic()
                    if packet["image"].get("score_valid"):
                        self.last_score=packet["score"];self.frame_scores=(self.frame_scores+[packet["score"]])[-120:]
                    self.frame_count+=1;self.latest_frame_features={"seq":self.frame_count,"state_hash":packet["image"].get("dhash64"),"gray32x18":packet["image"].get("gray32x18"),"edge_density":packet["image"].get("edge_density",0.0),"color_histogram":packet["image"].get("color_histogram"),"score":packet["score"],"hunger":packet["hunger"]};self.capture_failures=0
            except PoolCapacityBlocked:self._drop_pipeline(packet.get("session_id"),"persist","经验池硬上限阻止新截图写入")
            except Exception as error:
                self.resources.update_sqlite_latency(1000.0);self._drop_pipeline(packet.get("session_id"),"persist","SQLite/文件写入失败:"+str(error))
            finally:self.resources.update_pipeline_queue(self.capture_queue.qsize()+self.feature_queue.qsize()+self.persist_queue.qsize(),self._pipeline_age())

    def _monitor_loop(self, token):
        while self._is_current(token, ("learning", "training")):
            if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                self.request_idle("检测到 ESC 键", token); return
            with self.lock: hwnd=self.target_hwnd; session_id=self.session_id
            rect=valid_client(hwnd,True) if hwnd else None
            if rect is None:
                details=getattr(client_unobscured,"last_obstruction",None) or {"kind":"client_invalid","reason":getattr(valid_client,"last_reason","未知")}
                if session_id:
                    try:self.store.add_system_event(session_id,"client_validation_failure",details)
                    except Exception:pass
                self.request_idle("雷电模拟器客户区异常："+getattr(valid_client,"last_reason","未知"),token);return
            with self.lock:self.target_rect=rect
            if self._should_sleep(token): self._begin_auto_sleep(token);return
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
            if token != self.epoch or self.state != "training": return False
            elapsed=time.monotonic()-self.session_started; scores=list(self.frame_scores[-40:]); mouse_count=self.mouse_count; plan=list(self.ai_plan); queue_len=self.mouse_queue.qsize()
        try:
            if self.store.pool_size() >= int(self.settings.data["experience_limit"]*0.95): return True
        except Exception: pass
        readiness=self.store.training_readiness()
        if not plan:
            return bool(readiness.get("ready")) and elapsed >= 3.0
        sample=self.resources.sample()
        if sample.get("disk_free",1)<1024*1024*1024 or sample.get("avail_memory",1)<384*1024*1024 or queue_len>10000:return True
        if mouse_count<30 or time.time()-self.last_model_training<180.0:return False
        if elapsed<60.0 or len(scores)<12:return False
        mean=sum(scores)/len(scores);variance=sum((v-mean)**2 for v in scores)/len(scores);trend=scores[-1]-scores[0]
        lcbs=[float(a.get("confidence_lower_bound",0.0)) for a in plan];uncs=[float(a.get("uncertainty",1.0)) for a in plan]
        decision=self.sleep_decision_model({"score_mean":mean,"score_variance":variance,"score_trend":trend,"hunger_growth_speed":0.00004,"action_lcb_mean":sum(lcbs)/max(1,len(lcbs)),"action_uncertainty":sum(uncs)/max(1,len(uncs)),"sample_coverage":min(1.0,len(plan)/128.0),"mouse_queue_length":queue_len,"queue_pressure":min(1.0,queue_len/12000.0),"resource_pressure":1.0 if self.resources.acquire("sleep_training").must_pause else 0.0})
        return decision["sleep_probability"]>=0.62 and decision["expected_sleep_gain"]>0.0

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

    def _foreground_matches_target(self):
        foreground=root_window(user32.GetForegroundWindow())
        with self.lock: expected=self.target_root; expected_pid=self.target_pid
        if not foreground or foreground != expected:return False
        pid=wintypes.DWORD();user32.GetWindowThreadProcessId(foreground,ctypes.byref(pid))
        return int(pid.value)==int(expected_pid)

    def _ai_loop(self, token):
        while self._is_current(token,("training",)):
            budget=self.resources.acquire("ai_inference")
            if not budget.allowed:time.sleep(max(0.05,budget.next_interval));continue
            with self.lock:hwnd=self.target_hwnd
            rect=valid_client(hwnd,True) if hwnd else None
            if rect is None:self.request_idle("雷电模拟器客户区异常："+getattr(valid_client,"last_reason","未知"),token);return
            target=self._ai_target(rect);band=self.resources.runtime.confidence_band(float(target.get("confidence",0.0)),budget.state=="暂停")
            if band=="low":time.sleep(max(0.05,budget.next_interval));continue
            if band in ("medium","pressure") and target.get("action_type")!="移动":target["action_type"]="移动"
            x,y=target["x"],target["y"]
            self.ai_input_tracker.register("move","",0,x,y)
            if not point_inside((x,y),rect) or not ai_move_to(x,y):self.request_idle("AI 鼠标移动无法确认位于绑定客户区内",token);return
            time.sleep(0.05)
            rect=valid_client(hwnd,True) if hwnd else None
            if rect is None or not point_inside(cursor_position(),rect):self.request_idle("雷电模拟器客户区异常或鼠标已离开客户区",token);return
            action_type=target.get("action_type","移动");ok=True
            if action_type!="移动" and not self._foreground_matches_target():
                self.request_idle("前台窗口与绑定雷电实例不一致，已禁止点击、右键和滚轮",token);return
            if action_type=="左键":
                cx,cy=cursor_position();self.ai_input_tracker.register("button_down","left",0,cx,cy);self.ai_input_tracker.register("button_up","left",0,cx,cy);ok=ai_left_click()
            elif action_type=="右键":
                cx,cy=cursor_position();self.ai_input_tracker.register("button_down","right",0,cx,cy);self.ai_input_tracker.register("button_up","right",0,cx,cy);ok=ai_right_click()
            elif action_type=="滚轮":
                cx,cy=cursor_position();delta=target.get("wheel_delta",120);self.ai_input_tracker.register("wheel","vertical",delta,cx,cy);ok=ai_wheel(delta,False)
            elif action_type=="水平滚轮":
                cx,cy=cursor_position();delta=target.get("wheel_delta",120);self.ai_input_tracker.register("wheel","horizontal",delta,cx,cy);ok=ai_wheel(delta,True)
            if action_type!="移动" and ok:
                with self.lock:
                    stat=self.action_limits.setdefault(action_type,{"last":0.0,"times":[]});now=time.monotonic();stat["last"]=now;stat["times"]=[t for t in stat.get("times",[]) if now-t<60.0]+[now]
            if not ok:self.request_idle("AI 鼠标动作无法执行"+str(action_type),token);return
            time.sleep(max(0.05,budget.next_interval))

    def _write_barrier(self, session_id, reason):
        self.pipeline_stop.set()
        deadline=time.monotonic()+8.0
        for thread in list(self.capture_threads):
            if thread is threading.current_thread():continue
            thread.join(min(1.0,max(0.05,deadline-time.monotonic())))
        self._flush_pipeline_losses();self._flush_move_compression(session_id)
        self.flush_mouse_records(5.0)
        with self.loss_lock:losses=self.move_loss.pop(session_id,None) if session_id else None
        if losses:self.store.record_mouse_loss(session_id,losses[0],losses[1],losses[2],"鼠标事件队列写入失败")
        ok,detail=self.store.validate_consistency()
        if not ok:raise RuntimeError("写入屏障失败："+detail)
        self.store.add_system_event(session_id,"write_barrier",{"reason":reason,"time":time.time(),"consistency":detail,"pipeline_queue_age":self._pipeline_age()})

    def _close_active_session(self, reason, barrier=True):
        with self.lock:
            session_id = self.session_id
            self.session_id = None
            self.session_mode = None
            self.target_hwnd = None
            self.target_root = None
            self.target_pid = 0
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
        task = "maintenance" if purpose == "maintenance" else "sleep_training"
        while not self._cancelled(token):
            budget = self.resources.acquire(task)
            if budget.allowed:
                return True
            sample = self.resources.sample()
            self.emit("state", {"state": "sleep", "detail": budget.pause_reason or "资源预算要求暂停睡眠任务", "cpu": sample["cpu"], "memory": sample["memory"]})
            time.sleep(max(0.05, budget.next_interval))
        return False

    def _semantic_actions(self, mouse_rows):
        actions = []
        down = {}
        wheel_bucket = None
        last_move = None
        for row in mouse_rows:
            mid, sid, created_ns, created, event_type, source, rx, ry, speed, dx, dy, direction, button, wheel = row
            if source not in ("user", "ai", "用户", "AI") or rx is None or ry is None:
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
        if not self._wait_resource(token): return None
        frames_by_session, mouse_by_session = self.store.collect_training_data()
        fingerprint=str((sum(len(v) for v in frames_by_session.values()),sum(len(v) for v in mouse_by_session.values()),max([row[3] for rows in frames_by_session.values() for row in rows] or [0])))
        now_attempt=time.time()
        if fingerprint==self.last_training_data_fingerprint and now_attempt-self.last_training_attempt<900.0:
            self.store.add_system_event(None,"model_skipped",{"reason":"训练数据指纹未变化","fingerprint":fingerprint,"time":now_attempt});return self.store.best_model()
        self.last_training_attempt=now_attempt;self.last_training_data_fingerprint=fingerprint
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
            baselines={}
            for i,before in enumerate(frame_rows[:-1]):
                before_time=frame_finish[i]
                j=bisect_right(frame_start,before_time+250_000_000-1)
                if j>=len(frame_rows) or frame_start[j]>before_time+3_000_000_000: continue
                if any(before_time<t<frame_start[j] for t in critical): continue
                if not before[16] or not frame_rows[j][16]: continue
                state_key=self.store.assign_state_cluster(before[4] or before[5],before[9],before[10],before[11],before[12])
                baselines.setdefault(state_key,[]).append(((frame_start[j]-before_time)/1_000_000.0,float(frame_rows[j][6])-float(before[6])))
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
                if index%max(8, training_budget.training_block_size)==0 and not self._wait_resource(token): return None
                action_ns=int(action["action_time"]);before_i=bisect_right(frame_finish,action_ns)-1;after_i=bisect_right(frame_start,action_ns+250_000_000-1)
                if before_i<0 or after_i>=len(frame_rows) or frame_start[after_i]>action_ns+3_000_000_000: continue
                if any(action_ns<t<frame_start[after_i] for t in critical): continue
                before,after=frame_rows[before_i],frame_rows[after_i]
                if not before[16] or not after[16]: continue
                post_ms=(frame_start[after_i]-action_ns)/1_000_000.0
                state_key=self.store.assign_state_cluster(before[4] or before[5],before[9],before[10],before[11],before[12])
                candidates=baselines.get(state_key,[])
                nearest=sorted(candidates,key=lambda item:abs(item[0]-post_ms))[:12]
                baseline=sum(value for _,value in nearest)/max(1,len(nearest)) if nearest else 0.0
                stable=min(stability(before_i),stability(after_i))
                score_delta=float(after[6])-float(before[6]);reward_delta=float(after[7])-float(before[7]);expected_hunger=max(0.0,post_ms/1000.0*0.00004);advantage=score_delta-baseline
                gx=min(15,max(0,int(float(action["rx"])*16)));gy=min(8,max(0,int(float(action["ry"])*9)))
                if session_id in validation_sessions:
                    role="validation"
                elif action_ns<=cut-gap_ns:
                    role="train"
                elif not validation_sessions and action_ns>=cut+gap_ns:
                    role="validation"
                else:
                    role="excluded_gap"
                example={"action_id":action["action_id"],"session_id":session_id,"before_frame_id":before[0],"after_frame_id":after[0],"mouse_event_id":action["mouse_event_id"],"action_time":action_ns,"post_action_delay_ms":post_ms,"score_delta":score_delta,"reward_delta":reward_delta,"hunger_delta_expected":expected_hunger,"baseline_score_delta":baseline,"action_advantage":advantage,"stability":stable,"baseline_count":len(nearest),"outcome_valid":role!="excluded_gap" and stable>=0.45,"split_role":role}
                persisted.append(example)
                wheel_axis=action.get("wheel_axis","");signed=int(action.get("wheel_delta",action.get("signed_delta",0)) or 0);wheel_direction=1 if signed>0 else -1 if signed<0 else 0;wheel_bucket=min(10,abs(signed)//120)
                key=(state_key,gx,gy,action["action_type"],wheel_axis,wheel_direction,wheel_bucket)
                if role=="train" and stable>=0.45:
                    item=states.setdefault(key,{"samples":0,"human":0,"ai":0,"sum":0.0,"sum2":0.0,"score_sum":0.0,"reward_sum":0.0,"hunger_sum":0.0,"baseline_support":0,"stability_sum":0.0,"examples":[],"state_hash":before[4] or before[5],"gray32x18":before[11],"edge_density":before[12],"color_histogram":before[13],"aspect":before[9]/max(1,before[10])})
                    item["samples"]+=1;item["human" if action["source"] in ("user","用户") else "ai"]+=1;item["sum"]+=advantage;item["sum2"]+=advantage*advantage;item["score_sum"]+=score_delta;item["reward_sum"]+=reward_delta;item["hunger_sum"]+=expected_hunger;item["baseline_support"]+=len(nearest);item["stability_sum"]+=stable;item["examples"].append(dict(example,wheel_delta=signed,wheel_axis=wheel_axis));outcomes.append(example)
                elif role=="validation" and stable>=0.45:
                    validation_outcomes.append((key,example));validation_blocks.append(session_id)
        for start in range(0,len(persisted),500):
            if self._cancelled(token):return None
            self.store.save_action_outcomes(persisted[start:start+500])
        if not outcomes:
            self.store.add_system_event(None,"model_skipped",{"reason":"没有满足稳定性与时间隔离条件的训练动作","time":time.time(),"semantic_actions":all_actions_count});return self.store.best_model()
        actions=[];policy={}
        for key,item in states.items():
            n=item["samples"];mean=item["sum"]/max(1,n);var=max(0.0,item["sum2"]/max(1,n)-mean*mean);lcb=mean-1.96*math.sqrt(var/max(1,n));baseline_avg=item["baseline_support"]/max(1,n);stable_avg=item["stability_sum"]/max(1,n)
            if item["human"]<=0 and key[3]!="移动":lcb=min(lcb,-0.01)
            if baseline_avg<2 or stable_avg<0.65:lcb=min(lcb,-0.01)
            wheel_examples=[e for e in item["examples"] if "wheel_delta" in e]
            payload={"state_key":key[0],"state_hash":item["state_hash"],"gray32x18":item["gray32x18"],"edge_density":item["edge_density"],"color_histogram":item["color_histogram"],"aspect":item["aspect"],"x":(key[1]+0.5)/16.0,"y":(key[2]+0.5)/9.0,"action_type":key[3],"wheel_axis":key[4],"wheel_direction":key[5],"wheel_magnitude_bucket":key[6],"wheel_delta":int(round(sum(e.get("wheel_delta",0) for e in wheel_examples)/max(1,len(wheel_examples)))) if key[3] in ("滚轮","水平滚轮") else 0,"samples":n,"human_samples":item["human"],"ai_samples":item["ai"],"average_action_advantage":mean,"advantage_variance":var,"average_score_delta":item["score_sum"]/max(1,n),"average_reward_delta":item["reward_sum"]/max(1,n),"average_hunger_delta_expected":item["hunger_sum"]/max(1,n),"baseline_support":baseline_avg,"stability":stable_avg,"confidence_lower_bound":lcb,"uncertainty":math.sqrt(var/max(1,n)),"state_similarity_threshold":0.32}
            actions.append(payload);policy[key]=payload
        values=[];hits=0;failures=0
        for key,example in validation_outcomes:
            chosen=policy.get(key)
            if chosen is None:failures+=1;continue
            hits+=1;values.append(float(example["action_advantage"]))
            if example["action_advantage"]<=0 or example["stability"]<0.65:failures+=1
        val_mean=sum(values)/max(1,len(values));val_var=sum((v-val_mean)**2 for v in values)/max(1,len(values));val_ci=1.96*math.sqrt(val_var/max(1,len(values)));quality=val_mean-val_ci
        payload={"id":uuid.uuid4().hex,"trained_at":time.time(),"quality":quality,"train_quality":sum(a["average_action_advantage"] for a in actions)/max(1,len(actions)),"frame_count":sum(len(v) for v in frames_by_session.values()),"mouse_count":sum(len(v) for v in mouse_by_session.values()),"training_samples":len(outcomes),"semantic_actions":all_actions_count,"validation_samples":len(validation_outcomes),"validation_hits":hits,"validation_mean_action_advantage":val_mean,"validation_confidence_interval":val_ci,"validation_failure_rate":failures/max(1,len(validation_outcomes)),"validation_state_coverage":len({key[0] for key,_ in validation_outcomes}),"validation_sessions":sorted(set(validation_blocks)),"coverage_states":len({a["state_key"] for a in actions}),"failure_rate":len([a for a in actions if a["confidence_lower_bound"]<=0])/max(1,len(actions)),"model_version":6,"champion":True,"last_used":time.time(),"action_quality":quality,"validation_quality":quality,"policy":{"min_samples":5,"uncertainty_threshold":0.25,"min_confidence_lower_bound":0.0,"similarity_threshold":0.78,"low_confidence_action":"move_only","blacklist_regions":[],"target":"action_advantage"},"q_actions":sorted(actions,key=lambda a:(a["confidence_lower_bound"],a["samples"]),reverse=True)[:256],"outcome_examples":outcomes[-256:]}
        champion=self.store.best_model();champion_quality=float(champion.get("validation_quality",-999999.0)) if isinstance(champion,dict) else -999999.0
        enough=len(outcomes)>=30 and len({a["state_key"] for a in actions})>=4 and len(validation_outcomes)>=12 and hits>=6
        if not enough or quality<=champion_quality:
            payload["champion"]=False;self.store.add_system_event(None,"model_candidate_rejected",{"validation_quality":quality,"champion_quality":champion_quality,"validation_samples":len(validation_outcomes),"validation_hits":hits,"training_samples":len(outcomes),"fingerprint":fingerprint,"time":time.time()});return champion if isinstance(champion,dict) else payload
        name="model_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")+"_"+payload["id"][:8]+".json";final_path=self.store.models/name;temp_path=final_path.with_suffix(".tmp");temp_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
        if self._cancelled(token):temp_path.unlink(missing_ok=True);return None
        temp_path.replace(final_path);self.store.register_model_metadata(payload["id"],final_path,payload);self.store.save_model_frame_refs(payload["id"],outcomes,validation_outcomes);self.last_model_training=time.time();self.last_training_success=self.last_model_training
        return payload

    def _sleep_worker(self, token, resume_training):
        try:
            self.emit("progress",4.0);self.emit("state",{"state":"sleep","detail":"任务1：训练 AI 模型","cpu":self.resources.sample()["cpu"],"memory":self.resources.sample()["memory"]})
            self._train_model(token)
            if self._cancelled(token):return
            self.emit("progress",56.0);self.emit("state",{"state":"sleep","detail":"任务1完成；任务2：检查 AI 模型与经验池","cpu":self.resources.sample()["cpu"],"memory":self.resources.sample()["memory"]})
            self.store.recover_deletions()
            if not self._wait_resource(token,"maintenance"):return
            self.flush_mouse_records()
            model_removed=self.store.prune_models(max(1,int(self.settings.data["model_limit"])),lambda:self._cancelled(token),lambda:self._wait_resource(token,"maintenance"))
            experience_removed,remaining=self.store.prune_experience(max(1,int(self.settings.data["experience_limit"])),lambda:self._cancelled(token),lambda value:self.emit("progress",value),lambda:self._wait_resource(token,"maintenance"))
            if self._cancelled(token):return
            model_status=getattr(self.store,"last_model_prune_result",{"success":True,"remaining":len(self.store.model_files()),"target":len(self.store.model_files())})
            pool_status=getattr(self.store,"last_experience_prune_result",{"success":remaining<=int(self.settings.data["experience_limit"]),"remaining":remaining,"target":int(self.settings.data["experience_limit"]*0.5)})
            if not model_status.get("success") or not pool_status.get("success"):
                detail="任务2未完成：模型 {} / 目标 {}；经验池 {} / 目标 {}。已停止采集新截图，等待清理完成。".format(model_status.get("remaining"),model_status.get("target"),pool_status.get("remaining"),pool_status.get("target"))
                self.emit("state",{"state":"sleep","detail":detail,"cpu":self.resources.sample()["cpu"],"memory":self.resources.sample()["memory"]})
                self._finish_idle(token,detail,True);return
            self.emit("progress",100.0);detail="任务2完成：删除 AI 模型 {} 个，删除经验 {} 条，经验池 {:.2f} MB".format(model_removed,experience_removed,remaining/1024/1024)
            if resume_training:
                hwnd,rect,reason=self._find_valid_target(False)
                if hwnd is None or rect is None or not self._place_cursor_before_entry(hwnd,rect):self._finish_idle(token,"自动睡眠完成，但无法恢复训练："+(reason or "客户区状态异常"),True);return
                if self._cancelled(token) or valid_client(hwnd,True) is None:self._finish_idle(token,"自动睡眠完成，但雷电模拟器客户区状态异常",True);return
                self.emit("progress",0.0)
                if not self.start_session("training",automatic=True):self._finish_idle(token,"自动睡眠完成，但无法恢复训练模式",True)
            else:self._finish_idle(token,detail,True)
        except Exception as error:self._finish_idle(token,"睡眠模式发生错误："+str(error),True)

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
        sample=self.resources.sample()
        with self.lock:state=self.state;frames=self.frame_count;mouse=self.mouse_count;session=self.session_id
        try:
            pool_size=self.store.pool_size() if self.store.pool else 0;model_count=len(self.store.model_files()) if self.store.models else 0;capacity=self.store.capacity_status() if self.store.conn else {"blocked":False}
        except Exception:pool_size=0;model_count=0;capacity={"blocked":False}
        capture_budget=self.resources.acquire("capture");training_budget=self.resources.acquire("sleep_training")
        return {"state":state,"frames":frames,"mouse":mouse,"session":session or "无","cpu":sample["cpu"],"memory":sample["memory"],"pool_size":pool_size,"model_count":model_count,"capacity":capacity,"resource":dict(sample),"gpu_name":"未启用 GPU 推理","backend":"CPU 规则策略","gpu":sample.get("gpu"),"gpu_total":sample.get("gpu_dedicated_total"),"gpu_used":sample.get("gpu_dedicated_used"),"gpu_batch_size":0,"cpu_workers":max(capture_budget.cpu_workers,training_budget.cpu_workers),"capture_fps":1.0/max(0.001,capture_budget.next_interval),"capture_resolution":"{}×{}".format(*capture_budget.max_capture_resolution),"queue_age":sample.get("queue_age",0.0),"pipeline_queue_age":sample.get("pipeline_queue_age",0.0),"resource_state":capture_budget.state if capture_budget.state!="正常" else training_budget.state,"pause_reason":capture_budget.pause_reason or training_budget.pause_reason or "无","metric_sources":sample.get("metric_sources",{}),"ldplayer_cpu":sample.get("ldplayer_cpu",0.0),"program_cpu":sample.get("process_cpu",0.0),"program_gpu":sample.get("program_gpu"),"ldplayer_gpu":sample.get("ldplayer_gpu"),"gpu_sampling_source":sample.get("gpu_sampling_source","不可用"),"disk_write_latency":sample.get("disk_write_latency"),"sqlite_latency":sample.get("sqlite_latency",0.0)}

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
        info=self.controller.information()
        window=Toplevel(self.root);window.title("更多信息")
        wx1,wy1,wx2,wy2=work_area_for_window(self.root);width=max(520,min(980,int((wx2-wx1)*0.55)));height=max(420,min(820,int((wy2-wy1)*0.65)))
        window.geometry("{}x{}+{}+{}".format(width,height,wx1+max(0,((wx2-wx1)-width)//2),wy1+max(0,((wy2-wy1)-height)//2)));window.resizable(True,True);window.configure(bg="#0f172a");window.grid_columnconfigure(0,weight=1);window.grid_rowconfigure(1,weight=1)
        Label(window,text="运行信息",bg="#0f172a",fg="white",font=("Microsoft YaHei UI",18,"bold"),padx=20,pady=18).grid(row=0,column=0,sticky="w")
        canvas=Canvas(window,bg="#f8fafc",highlightthickness=0,bd=0);scroll=ttk.Scrollbar(window,orient="vertical",command=canvas.yview);canvas.configure(yscrollcommand=scroll.set);canvas.grid(row=1,column=0,sticky="nsew",padx=(16,0),pady=(0,16));scroll.grid(row=1,column=1,sticky="ns",padx=(0,16),pady=(0,16))
        content=Frame(canvas,bg="#f8fafc",padx=20,pady=18);content_id=canvas.create_window((0,0),window=content,anchor="nw");content.grid_columnconfigure(1,weight=1);content.bind("<Configure>",lambda event:canvas.configure(scrollregion=canvas.bbox("all")));canvas.bind("<Configure>",lambda event:canvas.itemconfigure(content_id,width=event.width))
        def number(value,unit=""):
            return "不可用" if value is None else "{:.1f}{}".format(float(value),unit)
        capacity=info.get("capacity",{})
        rows=[("当前状态",info["state"]),("本次会话",info["session"]),("本次记录画面",str(info["frames"])),("本次记录鼠标事件",str(info["mouse"])),("本程序 CPU",number(info.get("program_cpu"),"%")+" · "+info.get("metric_sources",{}).get("本程序 CPU","未知来源")),("雷电 CPU",number(info.get("ldplayer_cpu"),"%")+" · "+info.get("metric_sources",{}).get("雷电 CPU","未知来源")),("计算后端",info.get("backend","CPU 规则策略")),("GPU 推理", "未启用；当前为 CPU 规则策略"),("本程序 GPU 引擎",number(info.get("program_gpu"),"%")+" · "+info.get("gpu_sampling_source","不可用")),("雷电 GPU 引擎",number(info.get("ldplayer_gpu"),"%")+" · "+info.get("gpu_sampling_source","不可用")),("可用显存", "不可用" if info.get("gpu_total") is None else self.format_bytes(max(0,info.get("gpu_total",0)-info.get("gpu_used",0)))),("磁盘写入延迟",number(info.get("disk_write_latency")," ms")+" · fsync 探针"),("SQLite 写入延迟",number(info.get("sqlite_latency")," ms")+" · 实际事务计时"),("当前截图频率","约 {:.2f} FPS".format(info.get("capture_fps",0.0))),("当前截图分辨率",info.get("capture_resolution","未知")),("鼠标队列年龄","{:.2f} 秒".format(info.get("queue_age",0.0))),("流水线队列年龄","{:.2f} 秒".format(info.get("pipeline_queue_age",0.0))),("当前资源状态",info.get("resource_state","正常")),("限速原因",info.get("pause_reason","无")),("经验池大小",self.format_bytes(info["pool_size"])),("经验池硬状态", "已停止采集；{} / 目标 {}".format(self.format_bytes(capacity.get("remaining",0)),self.format_bytes(capacity.get("target",0))) if capacity.get("blocked") else "正常"),("AI 模型数量",str(info["model_count"])),("奖励定义","画面评分 − 饥饿值；训练主目标为 action_advantage"),("检索保障","评分基于明确 Top-K 历史帧；未检索到不作为高新颖度"),("资源保护","队列限速、降分辨率、SQLite/磁盘延迟监测、硬容量阻断")]
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
