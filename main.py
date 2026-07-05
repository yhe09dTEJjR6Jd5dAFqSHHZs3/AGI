import ctypes
import datetime
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

WH_MOUSE_LL = 14
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
PM_NOREMOVE = 0
GA_ROOT = 2
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

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]

class RGBQUAD(ctypes.Structure):
    _fields_ = [("rgbBlue", ctypes.c_byte), ("rgbGreen", ctypes.c_byte), ("rgbRed", ctypes.c_byte), ("rgbReserved", ctypes.c_byte)]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", RGBQUAD)]

class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("pt", POINT), ("mouseData", wintypes.DWORD), ("flags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ULONG_PTR), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

LowLevelMouseProc = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
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
user32.EnumWindows.argtypes = [EnumWindowsProc, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, LowLevelMouseProc, wintypes.HINSTANCE, wintypes.DWORD]
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
gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.POINTER(BITMAPINFO), wintypes.UINT]
gdi32.GetDIBits.restype = ctypes.c_int

class ResourceMeter:
    def __init__(self):
        self.lock = threading.Lock()
        self.previous = None
        self.last_sample = 0.0
        self.value = {"cpu": 0.0, "memory": 0.0}

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
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                memory = float(status.dwMemoryLoad)
            self.value = {"cpu": cpu, "memory": memory}
            self.last_sample = now
            return dict(self.value)

    def interval(self):
        sample = self.sample()
        if sample["cpu"] >= 92 or sample["memory"] >= 94:
            return 6.0
        if sample["cpu"] >= 82 or sample["memory"] >= 88:
            return 3.0
        if sample["cpu"] >= 70 or sample["memory"] >= 80:
            return 1.8
        return 1.0

    def allow_compute(self):
        sample = self.sample()
        return sample["cpu"] < 85 and sample["memory"] < 90

    def critical(self):
        sample = self.sample()
        return sample["cpu"] >= 95 or sample["memory"] >= 97

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
        self.load()

    def load(self):
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            for key in self.data:
                if key in loaded:
                    self.data[key] = loaded[key]
        except Exception:
            pass

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
                session_id TEXT NOT NULL,
                created REAL NOT NULL,
                screenshot_path TEXT NOT NULL,
                phash TEXT NOT NULL,
                score REAL NOT NULL,
                hunger REAL NOT NULL,
                reward REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_frames_created ON frames(created DESC);
            CREATE INDEX IF NOT EXISTS idx_frames_reward ON frames(reward ASC, created ASC);
            CREATE TABLE IF NOT EXISTS mouse_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                created REAL NOT NULL,
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
                speed REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mouse_session ON mouse_events(session_id, created);
            CREATE TABLE IF NOT EXISTS system_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                created REAL NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            """)
            self.conn.commit()

    def close(self):
        with self.lock:
            if self.conn is not None:
                try:
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

    def recent_hashes(self, count=384):
        with self.lock:
            if self.conn is None:
                return []
            rows = self.conn.execute("SELECT phash FROM frames ORDER BY created DESC LIMIT ?", (count,)).fetchall()
        return [row[0] for row in rows]

    def save_frame(self, session_id, image, phash, score, hunger, reward):
        identifier = uuid.uuid4().hex
        moment = time.time()
        folder = self.screens / session_id
        folder.mkdir(parents=True, exist_ok=True)
        relative = Path("screens") / session_id / (identifier + ".png")
        final_path = self.pool / relative
        temporary = final_path.with_suffix(".tmp")
        temporary.write_bytes(image["png"])
        temporary.replace(final_path)
        with self.lock:
            self.conn.execute("INSERT INTO frames(id, session_id, created, screenshot_path, phash, score, hunger, reward, width, height) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (identifier, session_id, moment, str(relative), phash, score, hunger, reward, image["width"], image["height"]))
            self.conn.execute("UPDATE sessions SET frame_count=frame_count+1 WHERE id=?", (session_id,))
            self.conn.commit()

    def save_mouse_batch(self, records):
        if not records:
            return
        values = []
        counts = {}
        for record in records:
            values.append((uuid.uuid4().hex, record["session_id"], record["created"], record["source"], record["event_type"], record["button"], record["wheel"], record["x"], record["y"], record["relative_x"], record["relative_y"], record["dx"], record["dy"], record["direction"], record["speed"]))
            counts[record["session_id"]] = counts.get(record["session_id"], 0) + 1
        with self.lock:
            if self.conn is None:
                return
            self.conn.executemany("INSERT INTO mouse_events(id, session_id, created, source, event_type, button, wheel, x, y, relative_x, relative_y, dx, dy, direction, speed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
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
                result.append((path, float(content.get("quality", 0.0)), float(content.get("trained_at", 0.0))))
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
            frame_rows = self.conn.execute("SELECT phash, score, reward, created FROM frames ORDER BY created DESC LIMIT 4000").fetchall()
            mouse_rows = self.conn.execute("SELECT event_type, source, relative_x, relative_y, speed, created FROM mouse_events ORDER BY created DESC LIMIT 12000").fetchall()
        return frame_rows, mouse_rows

    def pool_size(self):
        if self.pool is None or not self.pool.exists():
            return 0
        total = 0
        for item in self.pool.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                pass
        return total

    def prune_models(self, maximum):
        summaries = self.model_summaries()
        if len(summaries) <= maximum:
            return 0
        target = max(0, int(maximum * 0.5))
        removed = 0
        for path, quality, trained_at in sorted(summaries, key=lambda value: (value[1], value[2])):
            if len(summaries) - removed <= target:
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

    def prune_experience(self, maximum, cancelled, progress):
        current = self.pool_size()
        if current <= maximum:
            return 0, current
        target = max(0, int(maximum * 0.5))
        removed = 0
        with self.lock:
            rows = self.conn.execute("SELECT id, screenshot_path FROM frames ORDER BY reward ASC, created ASC").fetchall()
        for index, row in enumerate(rows):
            if cancelled():
                return removed, self.pool_size()
            with self.lock:
                self.conn.execute("DELETE FROM frames WHERE id=?", (row[0],))
                self.conn.commit()
            candidate = self._safe_screen_path(row[1])
            if candidate is not None:
                try:
                    candidate.unlink(missing_ok=True)
                except OSError:
                    pass
            removed += 1
            if index % 24 == 0:
                current = self.pool_size()
                progress(min(92.0, 60.0 + 30.0 * min(1.0, removed / max(1, len(rows)))))
                if current <= target:
                    break
        current = self.pool_size()
        if current > target:
            with self.lock:
                sessions = self.conn.execute("SELECT id FROM sessions ORDER BY started ASC").fetchall()
            for row in sessions:
                if cancelled():
                    return removed, self.pool_size()
                session_id = row[0]
                with self.lock:
                    paths = self.conn.execute("SELECT screenshot_path FROM frames WHERE session_id=?", (session_id,)).fetchall()
                    self.conn.execute("DELETE FROM frames WHERE session_id=?", (session_id,))
                    self.conn.execute("DELETE FROM mouse_events WHERE session_id=?", (session_id,))
                    self.conn.execute("DELETE FROM system_events WHERE session_id=?", (session_id,))
                    self.conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
                    self.conn.commit()
                for stored in paths:
                    candidate = self._safe_screen_path(stored[0])
                    if candidate is not None:
                        try:
                            candidate.unlink(missing_ok=True)
                        except OSError:
                            pass
                current = self.pool_size()
                if current <= target:
                    break
        with self.lock:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self.conn.execute("VACUUM")
                self.conn.commit()
            except sqlite3.Error:
                pass
        return removed, self.pool_size()

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

def virtual_screen_rect():
    left = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    top = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    return (left, top, left + user32.GetSystemMetrics(SM_CXVIRTUALSCREEN), top + user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))

def screen_contains(rect):
    screen = virtual_screen_rect()
    return rect[0] >= screen[0] and rect[1] >= screen[1] and rect[2] <= screen[2] and rect[3] <= screen[3]

def root_window(hwnd):
    return user32.GetAncestor(hwnd, GA_ROOT)

def client_unobscured(hwnd, rect):
    own_root = root_window(hwnd)
    if not own_root:
        return False
    x0, y0, x1, y1 = rect
    padding_x = max(3, min(20, (x1 - x0) // 10))
    padding_y = max(3, min(20, (y1 - y0) // 10))
    xs = (x0 + padding_x, (x0 + x1) // 2, x1 - padding_x - 1)
    ys = (y0 + padding_y, (y0 + y1) // 2, y1 - padding_y - 1)
    for x in xs:
        for y in ys:
            found = user32.GetWindowFromPoint(POINT(x, y))
            if not found or root_window(found) != own_root:
                return False
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

def find_emulator_window(executable):
    filename = Path(executable).name.lower()
    pids = processes_for_name(filename)
    candidates = []
    if pids:
        def callback(hwnd, _):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if int(pid.value) in pids and user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
                rect = client_rect(hwnd)
                if rect is not None:
                    area = (rect[2] - rect[0]) * (rect[3] - rect[1])
                    candidates.append((area, hwnd))
            return True
        procedure = EnumWindowsProc(callback)
        user32.EnumWindows(procedure, 0)
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]

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
    local = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(local)):
        return None
    width = int(local.right - local.left)
    height = int(local.bottom - local.top)
    if width <= 0 or height <= 0:
        return None
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
        if not gdi32.BitBlt(memory_dc, 0, 0, width, height, source_dc, 0, 0, SRCCOPY):
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
        scale = min(1.0, max_width / width, max_height / height)
        target_width = max(1, int(width * scale))
        target_height = max(1, int(height * scale))
        rgb = bytearray(target_width * target_height * 3)
        grayscale = [[0] * 9 for _ in range(8)]
        for y in range(target_height):
            source_y = min(height - 1, int(y * height / target_height))
            raw_y = height - 1 - source_y
            source_row = raw_y * width * 4
            output_row = y * target_width * 3
            for x in range(target_width):
                source_x = min(width - 1, int(x * width / target_width))
                source_index = source_row + source_x * 4
                output_index = output_row + x * 3
                blue = raw[source_index]
                green = raw[source_index + 1]
                red = raw[source_index + 2]
                rgb[output_index] = red
                rgb[output_index + 1] = green
                rgb[output_index + 2] = blue
        for y in range(8):
            sy = min(target_height - 1, int((y + 0.5) * target_height / 8))
            for x in range(9):
                sx = min(target_width - 1, int((x + 0.5) * target_width / 9))
                index = (sy * target_width + sx) * 3
                grayscale[y][x] = (rgb[index] * 299 + rgb[index + 1] * 587 + rgb[index + 2] * 114) // 1000
        value = 0
        for y in range(8):
            for x in range(8):
                value <<= 1
                if grayscale[y][x] > grayscale[y][x + 1]:
                    value |= 1
        return {"width": target_width, "height": target_height, "png": encode_png(target_width, target_height, rgb), "phash": f"{value:016x}"}
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

def frame_score(phash, historical):
    if not historical:
        return 1.0
    current = int(phash, 16)
    similarity = []
    for previous in historical:
        try:
            distance = bit_count(current ^ int(previous, 16))
            similarity.append(1.0 - distance / 64.0)
        except Exception:
            pass
    if not similarity:
        return 1.0
    similarity.sort(reverse=True)
    nearest = similarity[:min(8, len(similarity))]
    return max(0.0, min(1.0, 1.0 - sum(nearest) / len(nearest)))

class MouseHook:
    def __init__(self, sink):
        self.sink = sink
        self.thread = None
        self.thread_id = 0
        self.handle = None
        self.callback_ref = None
        self.stop_event = threading.Event()
        self.ready = threading.Event()

    def start(self):
        self.thread = threading.Thread(target=self._run, name="MouseHook", daemon=True)
        self.thread.start()
        self.ready.wait(2.0)

    def stop(self):
        self.stop_event.set()
        if self.thread_id:
            user32.PostThreadMessageW(self.thread_id, WM_QUIT, 0, 0)
        if self.thread:
            self.thread.join(2.0)

    def _run(self):
        self.thread_id = int(kernel32.GetCurrentThreadId())
        message = MSG()
        user32.PeekMessageW(ctypes.byref(message), None, 0, 0, PM_NOREMOVE)
        self.callback_ref = LowLevelMouseProc(self._callback)
        self.handle = user32.SetWindowsHookExW(WH_MOUSE_LL, self.callback_ref, kernel32.GetModuleHandleW(None), 0)
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
                    self.sink(event_type, button, wheel, int(info.pt.x), int(info.pt.y), time.time())
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
        self.ai_until = 0.0
        self.ai_step = 0
        self.ai_plan = []
        self.last_model_training = 0.0
        self.mouse_queue = queue.Queue(maxsize=12000)
        self.writer_stop = threading.Event()
        self.writer = threading.Thread(target=self._mouse_writer, name="MouseWriter", daemon=True)
        self.writer.start()
        self.hook = MouseHook(self.on_mouse)
        self.hook.start()

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
        self.store.ensure(self.settings.data["storage_path"])

    def _mouse_writer(self):
        pending = []
        last_write = time.monotonic()
        while not self.writer_stop.is_set() or not self.mouse_queue.empty():
            try:
                pending.append(self.mouse_queue.get(timeout=0.25))
            except queue.Empty:
                pass
            if pending and (len(pending) >= 80 or time.monotonic() - last_write >= 0.6 or self.writer_stop.is_set()):
                try:
                    self.store.save_mouse_batch(pending)
                except Exception:
                    pass
                pending = []
                last_write = time.monotonic()

    def on_mouse(self, event_type, button, wheel, x, y, created):
        with self.lock:
            if self.state not in ("learning", "training") or not self.session_id or not self.target_rect:
                return
            session_id = self.session_id
            rect = self.target_rect
            source = "AI" if self.state == "training" and time.monotonic() <= self.ai_until else "用户"
            previous = self.last_mouse
            self.last_mouse = (x, y, created)
            self.mouse_count += 1
        dx = 0.0
        dy = 0.0
        direction = 0.0
        speed = 0.0
        if previous is not None:
            dt = max(0.000001, created - previous[2])
            dx = float(x - previous[0])
            dy = float(y - previous[1])
            direction = math.atan2(dy, dx) if dx or dy else 0.0
            speed = math.hypot(dx, dy) / dt
        width = max(1, rect[2] - rect[0])
        height = max(1, rect[3] - rect[1])
        record = {
            "session_id": session_id,
            "created": created,
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
            pass

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
            if self.state != "idle":
                if not automatic:
                    self.emit("notice", "当前不是空闲状态。")
                return False
        try:
            self.ensure_store()
        except Exception as error:
            self.emit("notice", "无法创建存储路径：" + str(error))
            return False
        hwnd, rect, reason = self._find_valid_target(False)
        if hwnd is None:
            self.emit("notice", reason)
            return False
        if not self._place_cursor_before_entry(hwnd, rect):
            self.emit("notice", "进入模式前无法确认鼠标与雷电模拟器客户区状态。")
            return False
        rect = valid_client(hwnd, True)
        if rect is None:
            self.emit("notice", "雷电模拟器客户区状态异常。")
            return False
        session_id = self.store.create_session(mode)
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
            self.last_score = None
            self.frame_scores = []
            self.frame_count = 0
            self.mouse_count = 0
            self.last_mouse = None
            self.ai_until = 0.0
            self.ai_step = 0
            model = self.store.best_model() if mode == "training" else None
            self.ai_plan = model.get("hotspots", []) if isinstance(model, dict) else []
        self.store.add_system_event(session_id, "mode_enter", {"mode": mode, "automatic": automatic, "time": time.time()})
        self.post_state("已进入" + ("学习模式" if mode == "learning" else "训练模式"))
        threading.Thread(target=self._capture_loop, args=(token,), name="CaptureLoop", daemon=True).start()
        threading.Thread(target=self._monitor_loop, args=(token,), name="SessionMonitor", daemon=True).start()
        if mode == "training":
            threading.Thread(target=self._ai_loop, args=(token,), name="AIControl", daemon=True).start()
        return True

    def _capture_loop(self, token):
        while self._is_current(token, ("learning", "training")):
            if self.resources.critical():
                self.emit("state", {"state": self.current_state(), "detail": "系统资源繁忙，已自动限速记录", "cpu": self.resources.sample()["cpu"], "memory": self.resources.sample()["memory"]})
                time.sleep(4.0)
                continue
            interval = self.resources.interval()
            with self.lock:
                hwnd = self.target_hwnd
                session_id = self.session_id
            if hwnd and session_id:
                image = capture_client(hwnd)
                if image is not None:
                    try:
                        historical = self.store.recent_hashes()
                        score = frame_score(image["phash"], historical)
                        now = time.monotonic()
                        with self.lock:
                            hunger = 1e-9 + max(0.0, now - self.hunger_anchor) * 0.00004
                            if self.last_score is not None and score > self.last_score:
                                self.hunger_anchor = now
                                hunger = 1e-9
                            reward = score - hunger
                            self.last_score = score
                            self.frame_scores.append(score)
                            self.frame_scores = self.frame_scores[-120:]
                            self.frame_count += 1
                        self.store.save_frame(session_id, image, image["phash"], score, hunger, reward)
                    except Exception as error:
                        self.emit("state", {"state": self.current_state(), "detail": "记录已限速：" + str(error), "cpu": self.resources.sample()["cpu"], "memory": self.resources.sample()["memory"]})
            time.sleep(interval)

    def _monitor_loop(self, token):
        while self._is_current(token, ("learning", "training")):
            if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                self.request_idle("检测到 ESC 键")
                return
            with self.lock:
                hwnd = self.target_hwnd
            rect = valid_client(hwnd, True) if hwnd else None
            if rect is None:
                self.request_idle("雷电模拟器客户区异常或鼠标已离开客户区")
                return
            with self.lock:
                self.target_rect = rect
            if self._should_sleep(token):
                self._begin_auto_sleep(token)
                return
            time.sleep(0.08)

    def _should_sleep(self, token):
        with self.lock:
            if self.state != "training" or token != self.epoch:
                return False
            elapsed = time.monotonic() - self.session_started
            count = self.frame_count
            scores = list(self.frame_scores)
        if count < 24:
            return False
        if elapsed >= 240.0:
            return True
        if len(scores) >= 24:
            latest = scores[-12:]
            mean = sum(latest) / len(latest)
            variance = sum((value - mean) ** 2 for value in latest) / len(latest)
            return elapsed >= 120.0 and variance < 0.0005
        return False

    def _ai_target(self, rect):
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        with self.lock:
            step = self.ai_step
            self.ai_step += 1
            hotspots = list(self.ai_plan)
        if hotspots and step % 3 != 2:
            item = hotspots[step % len(hotspots)]
            x_ratio = min(0.95, max(0.05, float(item.get("x", 0.5))))
            y_ratio = min(0.95, max(0.05, float(item.get("y", 0.5))))
        else:
            x_ratio = 0.08 + 0.84 * ((step * 0.618033988749895) % 1.0)
            y_ratio = 0.08 + 0.84 * ((step * 0.414213562373095) % 1.0)
        x = rect[0] + int(width * x_ratio)
        y = rect[1] + int(height * y_ratio)
        return x, y

    def _ai_loop(self, token):
        while self._is_current(token, ("training",)):
            if not self.resources.allow_compute():
                time.sleep(1.5)
                continue
            with self.lock:
                rect = self.target_rect
            if rect:
                x, y = self._ai_target(rect)
                with self.lock:
                    self.ai_until = time.monotonic() + 0.18
                user32.SetCursorPos(x, y)
            time.sleep(max(0.55, self.resources.interval() * 0.9))

    def _close_active_session(self, reason):
        with self.lock:
            session_id = self.session_id
            self.session_id = None
            self.session_mode = None
            self.target_hwnd = None
            self.target_rect = None
        if session_id:
            try:
                self.store.add_system_event(session_id, "mode_exit", {"reason": reason, "time": time.time()})
                self.store.close_session(session_id, reason)
            except Exception:
                pass

    def request_idle(self, reason):
        with self.lock:
            if self.state == "idle":
                return
            self.cancel_event.set()
            self.epoch += 1
            previous = self.state
            self.state = "idle"
        self._close_active_session(reason)
        self.emit("progress", 0.0)
        self.post_state(reason if previous != "sleep" else "睡眠模式已中止：" + reason)

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
        with self.lock:
            self.epoch += 1
            token = self.epoch
            self.cancel_event = threading.Event()
            self.state = "sleep"
        self.post_state("已进入睡眠模式")
        threading.Thread(target=self._sleep_monitor, args=(token,), name="SleepMonitor", daemon=True).start()
        threading.Thread(target=self._sleep_worker, args=(token, False), name="SleepWorker", daemon=True).start()
        return True

    def _begin_auto_sleep(self, token):
        with self.lock:
            if token != self.epoch or self.state != "training":
                return
            self.state = "sleep"
        self._close_active_session("AI 判断进入睡眠模式")
        self.post_state("AI 判断当前值得进入睡眠模式")
        threading.Thread(target=self._sleep_monitor, args=(token,), name="AutoSleepMonitor", daemon=True).start()
        threading.Thread(target=self._sleep_worker, args=(token, True), name="AutoSleepWorker", daemon=True).start()

    def _sleep_monitor(self, token):
        while self._is_current(token, ("sleep",)):
            if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                self.request_idle("检测到 ESC 键")
                return
            time.sleep(0.08)

    def _cancelled(self, token):
        return not self._is_current(token, ("sleep",))

    def _wait_resource(self, token):
        while not self._cancelled(token) and not self.resources.allow_compute():
            sample = self.resources.sample()
            self.emit("state", {"state": "sleep", "detail": "系统资源繁忙，睡眠任务已暂缓", "cpu": sample["cpu"], "memory": sample["memory"]})
            time.sleep(1.2)
        return not self._cancelled(token)

    def _train_model(self, token):
        if not self._wait_resource(token):
            return None
        frames, mouse = self.store.collect_training_data()
        if self._cancelled(token):
            return None
        rewards = [float(row[2]) for row in frames]
        scores = [float(row[1]) for row in frames]
        quality = (sum(rewards) / len(rewards)) if rewards else 0.0
        diversity = len(set(row[0] for row in frames)) / max(1, len(frames))
        quality += diversity * 0.15
        grid = {}
        total = max(1, len(mouse))
        for index, row in enumerate(mouse):
            if index % 700 == 0 and not self._wait_resource(token):
                return None
            event_type, source, relative_x, relative_y, speed, created = row
            if source == "用户" and event_type in ("button_down", "button_up", "move") and relative_x is not None and relative_y is not None and 0.0 <= relative_x <= 1.0 and 0.0 <= relative_y <= 1.0:
                gx = min(7, max(0, int(float(relative_x) * 8)))
                gy = min(7, max(0, int(float(relative_y) * 8)))
                key = (gx, gy)
                weight = 4.0 if event_type == "button_down" else 1.0
                grid[key] = grid.get(key, 0.0) + weight
        hotspots = []
        for key, weight in sorted(grid.items(), key=lambda item: item[1], reverse=True)[:12]:
            hotspots.append({"x": (key[0] + 0.5) / 8.0, "y": (key[1] + 0.5) / 8.0, "weight": weight})
        payload = {
            "id": uuid.uuid4().hex,
            "trained_at": time.time(),
            "quality": quality,
            "frame_count": len(frames),
            "mouse_count": len(mouse),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "diversity": diversity,
            "hotspots": hotspots,
            "hash_samples": [row[0] for row in frames[:128]]
        }
        name = "model_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + payload["id"][:8] + ".json"
        final_path = self.store.models / name
        temp_path = final_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if self._cancelled(token):
            temp_path.unlink(missing_ok=True)
            return None
        temp_path.replace(final_path)
        self.last_model_training = time.time()
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
            if not self._wait_resource(token):
                return
            model_removed = self.store.prune_models(max(1, int(self.settings.data["model_limit"])))
            def update(value):
                self.emit("progress", value)
            experience_removed, remaining = self.store.prune_experience(max(1, int(self.settings.data["experience_limit"])), lambda: self._cancelled(token), update)
            if self._cancelled(token):
                return
            self.emit("progress", 100.0)
            detail = "任务2完成：删除 AI 模型 {} 个，删除经验 {} 条，经验池 {:.2f} MB".format(model_removed, experience_removed, remaining / 1024 / 1024)
            if resume_training:
                hwnd, rect, reason = self._find_valid_target(False)
                if hwnd is None or not self._place_cursor_before_entry(hwnd, rect):
                    self._finish_idle(token, "自动睡眠完成，但无法恢复训练：" + (reason or "客户区状态异常"))
                    return
                self._finish_idle(token, detail + "；准备恢复训练模式")
                self.start_session("training", automatic=True)
            else:
                self._finish_idle(token, detail)
        except Exception as error:
            self._finish_idle(token, "睡眠模式发生错误：" + str(error))

    def _finish_idle(self, token, detail):
        with self.lock:
            if token != self.epoch:
                return
            self.state = "idle"
            self.cancel_event.set()
            self.epoch += 1
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
        return {
            "state": state,
            "frames": frames,
            "mouse": mouse,
            "session": session or "无",
            "cpu": sample["cpu"],
            "memory": sample["memory"],
            "pool_size": pool_size,
            "model_count": model_count
        }

    def shutdown(self):
        self.request_idle("程序关闭")
        self.writer_stop.set()
        self.hook.stop()
        self.writer.join(2.0)
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
        self.status_var = StringVar(value="控制面板已就绪。")
        self.performance_var = StringVar(value="CPU 0.0% · 内存 0.0%")
        self.progress_var = DoubleVar(value=0.0)
        self.mode_buttons = []
        self.build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(90, self.drain)
        self.root.after(1200, self.refresh_performance)

    def format_bytes(self, value):
        return "{:.2f} GB".format(float(value) / 1024 / 1024 / 1024)

    def enqueue(self, kind, payload):
        self.events.put((kind, payload))

    def button(self, parent, text, command, color, row=None, column=None, **grid):
        item = Button(parent, text=text, command=command, bg=color, fg="white", activebackground=color, activeforeground="white", relief="flat", bd=0, font=("Microsoft YaHei UI", 10, "bold"), cursor="hand2", padx=14, pady=10)
        if row is not None:
            item.grid(row=row, column=column, **grid)
        return item

    def build(self):
        self.root.title("雷电智能学习与训练控制面板")
        self.root.geometry("940x620")
        self.root.minsize(760, 520)
        self.root.configure(bg="#101826")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        outer = Frame(self.root, bg="#101826", padx=18, pady=16)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(2, weight=1)
        header = Frame(outer, bg="#101826")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        Label(header, text="雷电智能学习与训练控制面板", bg="#101826", fg="white", font=("Microsoft YaHei UI", 20, "bold")).grid(row=0, column=0, sticky="w")
        Label(header, textvariable=self.mode_var, bg="#1e293b", fg="#f8fafc", font=("Microsoft YaHei UI", 10, "bold"), padx=12, pady=6).grid(row=0, column=1, sticky="e")
        rainbow = Canvas(outer, height=10, bg="#101826", highlightthickness=0)
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
        for index in range(8):
            body.grid_rowconfigure(index, weight=0)
        labels = (("雷电模拟器", self.path_var), ("存储路径", self.storage_var), ("经验池上限", self.experience_var), ("AI 模型数量上限", self.model_var))
        colors = ("#ef4444", "#f97316", "#eab308", "#22c55e")
        commands = (self.choose_emulator, self.choose_storage, self.change_experience, self.change_models)
        texts = ("选择雷电模拟器路径", "选择存储路径", "修改经验池上限", "修改AI模型数量上限")
        for row, ((title, variable), color, command, text) in enumerate(zip(labels, colors, commands, texts)):
            Label(body, text=title, bg="#f8fafc", fg="#334155", font=("Microsoft YaHei UI", 10, "bold"), width=15, anchor="w").grid(row=row, column=0, sticky="w", pady=6)
            value = Label(body, textvariable=variable, bg="#e2e8f0", fg="#0f172a", font=("Consolas", 9), anchor="w", padx=10, pady=9)
            value.grid(row=row, column=1, sticky="ew", padx=(8, 10), pady=6)
            self.button(body, text, command, color, row=row, column=2, sticky="e", pady=6)
        divider = Frame(body, bg="#cbd5e1", height=1)
        divider.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 12))
        actions = Frame(body, bg="#f8fafc")
        actions.grid(row=5, column=0, columnspan=3, sticky="ew")
        for index in range(4):
            actions.grid_columnconfigure(index, weight=1)
        self.button(actions, "更多信息", self.more_info, "#06b6d4", row=0, column=0, sticky="ew", padx=(0, 7))
        learn = self.button(actions, "学习模式", lambda: self.controller.start_session("learning"), "#3b82f6", row=0, column=1, sticky="ew", padx=7)
        train = self.button(actions, "训练模式", lambda: self.controller.start_session("training"), "#a855f7", row=0, column=2, sticky="ew", padx=7)
        sleep = self.button(actions, "睡眠模式", self.controller.start_sleep, "#ef4444", row=0, column=3, sticky="ew", padx=(7, 0))
        self.mode_buttons = [learn, train, sleep]
        Label(body, text="任务进度", bg="#f8fafc", fg="#334155", font=("Microsoft YaHei UI", 10, "bold"), anchor="w").grid(row=6, column=0, sticky="w", pady=(17, 6))
        progress = ttk.Progressbar(body, orient="horizontal", maximum=100.0, variable=self.progress_var, mode="determinate")
        progress.grid(row=6, column=1, columnspan=2, sticky="ew", pady=(17, 6))
        footer = Frame(body, bg="#eef2ff", padx=12, pady=10)
        footer.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        footer.grid_columnconfigure(0, weight=1)
        Label(footer, textvariable=self.status_var, bg="#eef2ff", fg="#1e3a8a", font=("Microsoft YaHei UI", 9), anchor="w", justify="left").grid(row=0, column=0, sticky="ew")
        Label(footer, textvariable=self.performance_var, bg="#eef2ff", fg="#475569", font=("Microsoft YaHei UI", 9), anchor="e").grid(row=0, column=1, sticky="e", padx=(12, 0))

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
        window.geometry("650x460")
        window.minsize(520, 360)
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
            ("资源保护", "自动限速、分批写入、低频训练与清理")
        ]
        for index, (name, value) in enumerate(rows):
            Label(content, text=name, bg="#f8fafc", fg="#475569", font=("Microsoft YaHei UI", 10, "bold"), anchor="w").grid(row=index, column=0, sticky="w", pady=5)
            Label(content, text=value, bg="#f8fafc", fg="#0f172a", font=("Microsoft YaHei UI", 10), anchor="w", wraplength=390, justify="left").grid(row=index, column=1, sticky="ew", padx=(18, 0), pady=5)

    def drain(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "state":
                    state = payload.get("state", "idle")
                    names = {"idle": "空闲", "learning": "学习模式", "training": "训练模式", "sleep": "睡眠模式"}
                    self.mode_var.set(names.get(state, state))
                    detail = payload.get("detail", "")
                    self.status_var.set(detail or "控制面板已就绪。")
                    self.performance_var.set("CPU {:.1f}% · 内存 {:.1f}%".format(payload.get("cpu", 0.0), payload.get("memory", 0.0)))
                    normal = "normal" if state == "idle" else "disabled"
                    for button in self.mode_buttons:
                        button.configure(state=normal)
                elif kind == "progress":
                    self.progress_var.set(float(payload))
                elif kind == "notice":
                    self.status_var.set(str(payload))
                    messagebox.showwarning("提示", str(payload), parent=self.root)
        except queue.Empty:
            pass
        self.root.after(90, self.drain)

    def refresh_performance(self):
        info = self.controller.information()
        self.performance_var.set("CPU {:.1f}% · 内存 {:.1f}%".format(info["cpu"], info["memory"]))
        self.root.after(1200, self.refresh_performance)

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
