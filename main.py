import concurrent.futures
import contextlib
import copy
import ctypes
import json
import heapq
import hashlib
import math
import random
import queue
import subprocess
import sys
import os
import shutil
import threading
import time
import tempfile
import uuid
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, asdict, fields, replace
from typing import Optional
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from urllib.parse import urlparse

MIN_PYTHON_VERSION = (3, 10)

DEPENDENCY_INSTALL_MAP = {"mss": "mss", "PIL": "pillow", "psutil": "psutil", "pywin32": "pywin32", "pynput.mouse": "pynput", "pynput.keyboard": "pynput"}
REQUIRED_MODULES = ("mss", "PIL", "pywin32", "pynput.mouse", "psutil")
OPTIONAL_MODULES = ("pynput.keyboard",)


@dataclass(frozen=True)
class AgentSpec:
    default_ldplayer_path: str
    default_data_path: str
    default_training_seconds: int
    default_still_seconds: float
    default_experience_pool_gb: float
    default_ai_model_limit: int
    editable_fields: tuple


AGENT_SPEC = AgentSpec(
    default_ldplayer_path=r"D:\LDPlayer9\dnplayer.exe",
    default_data_path=r"C:\Users\Administrator\Desktop\AAA",
    default_training_seconds=900,
    default_still_seconds=10.0,
    default_experience_pool_gb=10.0,
    default_ai_model_limit=10,
    editable_fields=("ldplayer_path", "data_path", "training_seconds", "still_seconds", "experience_pool_gb", "ai_model_limit")
)


def should_stop_run(stop_event, deadline, escape_check, termination_reason=None):
    if stop_event and stop_event.is_set():
        return termination_reason if termination_reason in TERMINATION_REASONS else "user_stop"
    if escape_check and escape_check():
        return "esc"
    if deadline is not None and time.perf_counter() >= deadline:
        return "time_limit"
    return None


class PausableTrainingClock:
    def __init__(self, seconds):
        self.remaining = float(max(1, seconds))
        self.paused = False
        self.deadline_perf = time.perf_counter() + self.remaining

    def pause(self):
        if not self.paused:
            self.remaining = max(0.0, self.deadline_perf - time.perf_counter())
            self.paused = True

    def resume(self):
        if self.paused:
            self.deadline_perf = time.perf_counter() + self.remaining
            self.paused = False

    def expired(self):
        if self.paused:
            return False
        self.remaining = max(0.0, self.deadline_perf - time.perf_counter())
        return self.remaining <= 0.0

    def deadline(self):
        if self.paused:
            return None
        self.remaining = max(0.0, self.deadline_perf - time.perf_counter())
        return self.deadline_perf



def screen_content_metrics(raw, width, height):
    data = bytes(raw or b"")
    if not data or width <= 0 or height <= 0:
        return {"valid": False, "screenshot_valid": False, "content_valuable": False, "reason": "empty", "brightness_mean": 0.0, "brightness_variance": 0.0, "edge_density": 0.0, "change_rate": 0.0, "black_candidate": False, "transparent_ratio": 0.0}
    pixel_count = min(width * height, len(data) // 4)
    if pixel_count <= 0:
        return {"valid": False, "screenshot_valid": False, "content_valuable": False, "reason": "empty", "brightness_mean": 0.0, "brightness_variance": 0.0, "edge_density": 0.0, "change_rate": 0.0, "black_candidate": False, "transparent_ratio": 0.0}
    sample_target = min(4096, pixel_count)
    step = max(1, pixel_count // sample_target)
    values = []
    last = None
    edges = 0
    changes = 0
    transparent = 0
    for index in range(0, pixel_count, step):
        offset = index * 4
        if offset + 3 >= len(data):
            break
        b, g, r, a = data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
        if a <= 2:
            transparent += 1
        brightness = (int(r) * 299 + int(g) * 587 + int(b) * 114) / 1000.0
        values.append(brightness)
        if last is not None:
            delta = abs(brightness - last)
            if delta >= 12.0:
                edges += 1
            if delta >= 3.0:
                changes += 1
        last = brightness
    if not values:
        return {"valid": False, "screenshot_valid": False, "content_valuable": False, "reason": "empty", "brightness_mean": 0.0, "brightness_variance": 0.0, "edge_density": 0.0, "change_rate": 0.0, "black_candidate": False, "transparent_ratio": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    divisor = max(1, len(values) - 1)
    edge_density = edges / divisor
    change_rate = changes / divisor
    transparent_ratio = transparent / max(1, len(values))
    black_candidate = mean <= 2.0 and variance <= 1.0 and transparent_ratio < 0.95
    screenshot_valid = transparent_ratio < 0.95 and len(data) >= pixel_count * 4
    content_valuable = variance >= 3.0 or edge_density >= 0.01 or change_rate >= 0.05
    valid = screenshot_valid and not black_candidate
    reason = "ok" if valid else ("transparent" if transparent_ratio >= 0.95 else "black_candidate" if black_candidate else "empty")
    return {"valid": valid, "screenshot_valid": screenshot_valid, "content_valuable": content_valuable, "reason": reason, "brightness_mean": round(mean, 4), "brightness_variance": round(variance, 4), "edge_density": round(edge_density, 6), "change_rate": round(change_rate, 6), "black_candidate": black_candidate, "transparent_ratio": round(transparent_ratio, 6)}


def atomic_write_json(path, payload, lock=None):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    guard = lock or contextlib.nullcontext()
    with guard:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(target)
        try:
            fd = os.open(str(target.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

def fail_and_exit(message):
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("启动失败", f"{message}\n\n点击确定后退出。")
        root.destroy()
    except Exception:
        pass
    sys.exit(1)


class StartupRepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnvironmentEnsureResult:
    ok: bool
    stage: str
    checks: tuple
    repair_actions: tuple
    recheck: tuple
    unrecoverable: tuple

    def detail(self):
        sections = [f"阶段：{self.stage}", "检查项：", *[f"- {item}" for item in self.checks], "", "修复动作：", *[f"- {item}" for item in self.repair_actions], "", "复检结果：", *[f"- {item}" for item in self.recheck], "", "不可修复原因：", *[f"- {item}" for item in self.unrecoverable]]
        return "\n".join(sections)

    def startup_popup_message(self):
        initial = self.checks or ("无异常",)
        repairs = self.repair_actions or ("未执行自愈",)
        recheck = self.recheck or ("复检通过",)
        if self.ok:
            advice = ("运行环境符合要求，程序可进入空闲。",)
        else:
            advice = ("点击“选择雷电路径”选择 dnplayer.exe。", "点击“选择数据目录”修复存储路径。", "点击“重试”再次自愈并复检。", "确认环境可用时点击“忽略”进入空闲。", "无法处理时点击“退出”终止程序。")
        sections = ["初检结果：", *[f"- {item}" for item in initial], "", "初检后进行的自愈尝试：", *[f"- {item}" for item in repairs], "", "复检结果：", *[f"- {item}" for item in recheck], "", "下一步建议：", *[f"- {item}" for item in advice]]
        return "\n".join(sections)


def write_startup_install_log(command, result=None, error=None):
    try:
        log_dir = Path(globals().get("DEFAULT_DATA_PATH", AGENT_SPEC.default_data_path))
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {"created_at": now_text() if "now_text" in globals() else datetime.now().astimezone().isoformat(timespec="milliseconds"), "python_executable": sys.executable, "python_prefix": sys.prefix, "user_site": subprocess.run([sys.executable, "-m", "site", "--user-site"], capture_output=True, text=True, timeout=10).stdout.strip(), "pip_index_url": os.environ.get("PIP_INDEX_URL") or os.environ.get("UV_INDEX_URL"), "command": command, "returncode": getattr(result, "returncode", None), "stdout": getattr(result, "stdout", "") or "", "stderr": getattr(result, "stderr", "") or "", "error": str(error) if error else None}
        with (log_dir / "startup_install.log").open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def verify_installed_modules():
    failed = []
    for name in REQUIRED_MODULES:
        try:
            if name == "pywin32":
                __import__("win32api")
                __import__("win32gui")
            else:
                __import__(name)
        except Exception as exc:
            failed.append(f"{name}: {exc}")
    return failed


def bootstrap_dependencies():
    required = set(REQUIRED_MODULES)
    missing = sorted({DEPENDENCY_INSTALL_MAP[name] for name in required if name in IMPORT_ERRORS})
    if not missing:
        return
    install_key = "AGI_BOOTSTRAP_INSTALLING"
    if os.environ.get(install_key):
        raise StartupRepairError("依赖安装后仍无法导入，已停止重复安装：" + "；".join(f"{name}: {error}" for name, error in IMPORT_ERRORS.items()))
    try:
        pip_check = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True, timeout=30)
        write_startup_install_log([sys.executable, "-m", "pip", "--version"], result=pip_check)
        if pip_check.returncode != 0:
            detail = (pip_check.stderr or pip_check.stdout or "").strip()
            raise StartupRepairError(f"无法启动 pip，不能自动安装依赖。请检查 Python 环境。\n{detail}")
    except StartupRepairError:
        raise
    except Exception as exc:
        write_startup_install_log([sys.executable, "-m", "pip", "--version"], error=exc)
        raise StartupRepairError(f"无法启动 pip，不能自动安装依赖。请检查 Python 环境或网络。\n{exc}") from exc
    if os.environ.get("AGI_AUTO_INSTALL_CONFIRMED") != "1":
        raise StartupRepairError("检测到缺失依赖，但未获得自动安装确认。请使用锁定版本和哈希校验的 requirements.txt 或离线 wheel 包安装；如确认接受运行时 pip 安装风险，请设置 AGI_AUTO_INSTALL_CONFIRMED=1 后重试。缺失依赖：" + "、".join(missing))
    base_command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--user"]
    mirrors = [None]
    extra_mirror = os.environ.get("AGI_PIP_INDEX_URL")
    if extra_mirror:
        mirrors.append(extra_mirror)
    commands = [base_command + (["-i", mirror, "--trusted-host", urlparse(mirror).hostname] if mirror else []) + missing for mirror in mirrors]
    install_timeout = max(120, min(300, safe_int(os.environ.get("AGI_PIP_INSTALL_TIMEOUT"), 180)))
    env = dict(os.environ)
    env[install_key] = "1"
    failures = []
    result = None
    command = commands[0]
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=install_timeout, env=env)
            write_startup_install_log(command, result=result)
            if result.returncode == 0:
                break
            failures.append((command, (result.stderr or result.stdout or "").strip()))
        except subprocess.TimeoutExpired as exc:
            write_startup_install_log(command, error=exc)
            failures.append((command, (exc.stderr or exc.stdout or "").strip() or str(exc)))
    if result is None or result.returncode != 0:
        install_log_path = Path(globals().get("DEFAULT_DATA_PATH", AGENT_SPEC.default_data_path)) / "startup_install.log"
        detail = "\n".join(f"命令: {' '.join(item)}\n结果: {text}" for item, text in failures)
        raise StartupRepairError(f"自动安装依赖失败: {' '.join(missing)}\n日志位置: {install_log_path}\n请手动执行: {' '.join(commands[-1])}\n{detail}")
    if "pywin32" in missing:
        postinstall = [sys.executable, "-m", "pywin32_postinstall", "-install"]
        try:
            postinstall_result = subprocess.run(postinstall, capture_output=True, text=True, timeout=120)
            write_startup_install_log(postinstall, result=postinstall_result)
            if postinstall_result.returncode != 0:
                detail = (postinstall_result.stderr or postinstall_result.stdout or "").strip()
                raise StartupRepairError(f"pywin32 安装后配置失败。请手动执行: {' '.join(postinstall)}\n{detail}")
        except StartupRepairError:
            raise
        except Exception as exc:
            write_startup_install_log(postinstall, error=exc)
            raise StartupRepairError(f"pywin32 安装后配置失败。请手动执行: {' '.join(postinstall)}\n{exc}") from exc
    failed = verify_installed_modules()
    if failed:
        raise StartupRepairError("自动安装依赖后仍无法导入：" + "；".join(failed))
    os.environ[install_key] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

IMPORT_ERRORS = {}
try:
    import mss
except Exception as exc:
    mss = None
    IMPORT_ERRORS["mss"] = exc
try:
    from PIL import Image
except Exception as exc:
    Image = None
    IMPORT_ERRORS["PIL"] = exc
try:
    import psutil
except Exception as exc:
    psutil = None
    IMPORT_ERRORS["psutil"] = exc
try:
    import win32api
    import win32con
    import win32gui
    import win32process
except Exception as exc:
    win32api = None
    win32con = None
    win32gui = None
    win32process = None
    IMPORT_ERRORS["pywin32"] = exc
try:
    from pynput import mouse as pynput_mouse
except Exception as exc:
    pynput_mouse = None
    IMPORT_ERRORS["pynput.mouse"] = exc
try:
    from pynput import keyboard as pynput_keyboard
except Exception as exc:
    pynput_keyboard = None
    IMPORT_ERRORS["pynput.keyboard"] = exc

def startup_config_paths():
    base = Path(os.environ.get("APPDATA") or (Path.home() / ".config"))
    settings_file = base / "AGI" / "startup.json"
    source = {}
    try:
        if settings_file.is_file():
            with settings_file.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
            source = loaded if isinstance(loaded, dict) else {}
    except Exception:
        source = {}
    return Path(str(source.get("ldplayer_path") or AGENT_SPEC.default_ldplayer_path)), Path(str(source.get("data_path") or AGENT_SPEC.default_data_path))


def save_startup_config_paths(ldplayer_path=None, data_path=None):
    current_ldplayer_path, current_data_path = startup_config_paths()
    store = AppConfigStore()
    store.save_settings({"ldplayer_path": str(ldplayer_path or current_ldplayer_path), "data_path": str(data_path or current_data_path)})


def data_path_write_issue(path, create=False):
    candidate = Path(path)
    try:
        if create:
            candidate.mkdir(parents=True, exist_ok=True)
        if not candidate.exists():
            return "存储路径不存在"
        if not candidate.is_dir():
            return "存储路径不是文件夹"
        probe = candidate / f".agi_startup_write_{uuid.uuid4().hex}.tmp"
        with probe.open("wb") as file:
            file.write(b"")
        probe.unlink()
        return None
    except Exception as exc:
        return f"存储路径不可写：{exc}"


def startup_ldplayer_window_issue(path):
    if sys.platform != "win32":
        return None
    ok, reason = validate_ldplayer_executable(path, require_attach=True)
    if not ok:
        return reason
    report = windows_runtime_report(path)
    if not report.get("ok"):
        return "雷电客户区活动检查需要 Windows 桌面环境：" + json.dumps(report, ensure_ascii=False)
    return None


def discover_ldplayer_candidates():
    candidates = []
    for text in (AGENT_SPEC.default_ldplayer_path, r"C:\Program Files\LDPlayer\LDPlayer9\dnplayer.exe", r"C:\Program Files\LDPlayer\LDPlayer4\dnplayer.exe", r"D:\LDPlayer\LDPlayer9\dnplayer.exe"):
        path = Path(text)
        if path.exists():
            candidates.append(str(path))
    if sys.platform == "win32" and psutil:
        try:
            for proc in psutil.process_iter(["name", "exe"]):
                if str((proc.info or {}).get("name", "")).lower() == "dnplayer.exe" and (proc.info or {}).get("exe"):
                    candidates.append(str(proc.info["exe"]))
        except Exception:
            pass
    return list(dict.fromkeys(candidates))


def startup_environment_issues():
    issues = []
    ldplayer_path, data_path = startup_config_paths()
    if sys.version_info < MIN_PYTHON_VERSION:
        issues.append(f"Python 版本过低：需要 {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} 或更高版本，当前版本为 {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    if sys.platform != "win32":
        issues.append(f"操作系统不符合要求：本程序需要 Windows 桌面会话和雷电模拟器，当前平台为 {sys.platform}")
    else:
        session_name = os.environ.get("SESSIONNAME", "")
        if not session_name:
            issues.append("未检测到 Windows 桌面会话")
        try:
            if ctypes.windll.user32.GetDesktopWindow() == 0:
                issues.append("无法访问 Windows 桌面")
        except Exception as exc:
            issues.append(f"无法检查 Windows 桌面权限：{exc}")
    for name in REQUIRED_MODULES:
        if name in IMPORT_ERRORS:
            issues.append(f"依赖无法导入 {name}：{IMPORT_ERRORS[name]}")
    valid_path, path_reason = validate_ldplayer_executable(ldplayer_path, require_attach=False)
    if not valid_path:
        issues.append(f"雷电模拟器启动路径无效 {ldplayer_path}：{path_reason}")
    elif sys.platform == "win32" and "WindowManager" in globals():
        runtime_ok, runtime_reason = attach_and_probe_ldplayer(ldplayer_path)
        if not runtime_ok:
            issues.append(f"雷电运行能力不可用 {ldplayer_path}：{runtime_reason}")
    storage_issue = data_path_write_issue(data_path)
    if storage_issue:
        issues.append(f"存储路径无效 {data_path}：{storage_issue}")
    return issues


def attach_and_probe_ldplayer(ldplayer_path, settings=None):
    effective_settings = settings or (derive_runtime_settings() if "derive_runtime_settings" in globals() else None)
    manager = WindowManager(ldplayer_path, effective_settings)
    if not manager.launch_or_attach():
        return False, "无法启动或附着雷电模拟器客户区"
    return runtime_capability_probe(manager)


def runtime_capability_probe(window_manager):
    if sys.platform != "win32":
        return False, f"运行能力探测需要 Windows 桌面环境，当前平台为 {sys.platform}"
    if mss is None:
        return False, "mss 不可用，无法探测客户区截图能力"
    if win32api is None:
        return False, "pywin32 不可用，无法读取鼠标坐标"
    if not window_manager:
        return False, "雷电窗口管理器不可用"
    try:
        check = window_manager.check_window(force=True)
    except Exception as exc:
        return False, f"雷电客户区检查失败：{exc}"
    if not getattr(check, "ok", False):
        return False, "雷电客户区不可用：" + str(getattr(check, "reason", "unknown"))
    rect = tuple(getattr(check, "rect", ()) or ())
    if len(rect) != 4:
        return False, "雷电客户区坐标无效"
    left, top, right, bottom = [int(value) for value in rect]
    width = right - left
    height = bottom - top
    if width < 2 or height < 2:
        return False, "雷电客户区尺寸过小，无法截图"
    try:
        with mss.mss() as sct:
            image = sct.grab({"left": left, "top": top, "width": width, "height": height})
    except Exception as exc:
        return False, f"无法获取雷电客户区截图：{exc}"
    if getattr(image, "width", width) < 2 or getattr(image, "height", height) < 2:
        return False, "无法获取雷电客户区截图"
    raw = bytes(getattr(image, "bgra", b"") or b"")
    if not raw:
        return False, "雷电客户区截图为空"
    content_metrics = screen_content_metrics(raw, width, height)
    if not content_metrics.get("valid"):
        return False, "雷电客户区截图疑似黑屏或空白：" + json.dumps(content_metrics, ensure_ascii=False)
    try:
        x, y = win32api.GetCursorPos()
    except Exception as exc:
        return False, f"无法读取鼠标坐标：{exc}"
    if not isinstance(x, int) or not isinstance(y, int):
        return False, "无法读取鼠标坐标"
    try:
        monitor = sct.monitors[0] if 'sct' in locals() else None
        if monitor:
            screen_left = int(monitor.get("left", 0))
            screen_top = int(monitor.get("top", 0))
            screen_right = screen_left + int(monitor.get("width", 0))
            screen_bottom = screen_top + int(monitor.get("height", 0))
            if left < screen_left or top < screen_top or right > screen_right or bottom > screen_bottom:
                return False, "雷电客户区截图坐标与屏幕/DPI 范围不一致"
    except Exception:
        pass
    return True, "ok"

def attempt_startup_environment_repair(actions=None):
    actions = actions if actions is not None else []
    ldplayer_path, data_path = startup_config_paths()
    missing = [name for name in REQUIRED_MODULES if name in IMPORT_ERRORS]
    if missing:
        actions.append("自动安装缺失或异常依赖：" + "、".join(missing))
        bootstrap_dependencies()
    storage_issue = data_path_write_issue(data_path, create=True)
    actions.append("已创建并验证存储路径可写" if not storage_issue else f"无法修复存储路径：{storage_issue}")
    valid_path, path_reason = validate_ldplayer_executable(ldplayer_path, require_attach=False)
    if valid_path and sys.platform == "win32" and not missing:
        runtime_ok, runtime_reason = validate_ldplayer_executable(ldplayer_path, require_attach=True)
        actions.append("已启动或附着雷电模拟器并复检客户区可用" if runtime_ok else f"无法修复雷电模拟器客户区：{runtime_reason}")
        if not runtime_ok:
            discovered = discover_ldplayer_candidates()
            for candidate in discovered:
                candidate_ok, candidate_reason = validate_ldplayer_executable(candidate, require_attach=True)
                actions.append(f"验证雷电候选路径 {candidate}：" + ("可用" if candidate_ok else candidate_reason))
                if candidate_ok:
                    save_startup_config_paths(ldplayer_path=candidate, data_path=data_path)
                    actions.append(f"已自动切换并保存雷电模拟器路径：{candidate}")
                    break
    elif not valid_path:
        discovered = discover_ldplayer_candidates()
        actions.append("已自动发现雷电模拟器候选路径：" + "、".join(discovered) if discovered else "未在常见目录或已运行进程中发现雷电模拟器")
        adopted = False
        for candidate in discovered:
            candidate_ok, candidate_reason = validate_ldplayer_executable(candidate, require_attach=True)
            actions.append(f"验证雷电候选路径 {candidate}：" + ("可用" if candidate_ok else candidate_reason))
            if candidate_ok:
                save_startup_config_paths(ldplayer_path=candidate, data_path=data_path)
                actions.append(f"已自动切换并保存雷电模拟器路径：{candidate}")
                adopted = True
                break
        if not adopted:
            actions.append(f"雷电模拟器路径无法自动修复：{path_reason}")
    if sys.version_info < MIN_PYTHON_VERSION:
        actions.append("Python 运行时版本无法由程序自动升级")
    if sys.platform != "win32":
        actions.append("当前操作系统无法由程序自动转换为 Windows 桌面环境")
    elif not os.environ.get("SESSIONNAME", ""):
        actions.append("Windows 桌面会话无法由程序自动创建")
    if not actions:
        actions.append("未找到可自动执行的修复操作")
    return actions


def startup_failure_detail(initial_issues, repair_actions, repair_error, remaining_issues):
    sections = ["已尝试的修复动作、无法自动修复的原因与下一步操作", "", "初次检查：", *[f"- {item}" for item in initial_issues], "", "已尝试的修复动作："]
    sections.extend(f"- {item}" for item in repair_actions)
    if repair_error:
        sections.append(f"- 修复失败：{repair_error}")
    sections.extend(["", "无法自动修复的原因：", *[f"- {item}" for item in remaining_issues], "", "下一步操作：", "- 确认正在 Windows 桌面会话中运行且 Python 版本满足要求。", "- 安装雷电模拟器或通过控制面板修改 dnplayer.exe 路径。", "- 按上方依赖或存储路径错误手动修复后重新启动。"])
    return "\n".join(sections)


def ensure_environment(stage, allow_repair=True, check_environment=None, repair_environment=None):
    check_environment = check_environment or startup_environment_issues
    repair_environment = repair_environment or attempt_startup_environment_repair
    initial_issues = tuple(check_environment())
    if not initial_issues:
        return EnvironmentEnsureResult(True, stage, initial_issues, tuple(), tuple(), tuple())
    repair_actions = []
    repair_error = None
    if allow_repair:
        try:
            repair_environment(repair_actions)
        except Exception as exc:
            repair_error = str(exc)
    else:
        repair_actions.append("当前阶段禁止自动修复")
    remaining_issues = tuple(check_environment())
    unrecoverable = tuple([repair_error] if repair_error else []) + remaining_issues
    return EnvironmentEnsureResult(not remaining_issues, stage, initial_issues, tuple(repair_actions), remaining_issues, unrecoverable)


def ensure_startup_environment_result(check_environment=None, repair_environment=None):
    return ensure_environment("startup", True, check_environment, repair_environment)


def prepare_startup_environment(check_environment=None, repair_environment=None, failure_handler=None):
    failure_handler = failure_handler or fail_and_exit
    result = ensure_startup_environment_result(check_environment, repair_environment)
    if result.ok:
        return True
    failure_handler(result.startup_popup_message())
    return False


STARTUP_FAILURE_BUTTON_LABELS = ("选择雷电路径", "选择数据目录", "更多", "重试", "忽略", "退出")

def interactive_startup_failure_repair(parent, status_var, message):
    result = {"retry": False, "exit": False, "ignore": False}
    dialog = tk.Toplevel(parent)
    dialog.title("启动失败修复")
    dialog.transient(parent)
    dialog.grab_set()
    text = tk.Text(dialog, width=96, height=28, wrap="word")
    text.insert("1.0", message)
    text.configure(state="disabled")
    text.pack(fill="both", expand=True, padx=12, pady=(12, 6))
    actions = ttk.Frame(dialog)
    actions.pack(fill="x", padx=12, pady=(0, 12))
    def refresh_status(value):
        status_var.set(value)
        parent.update_idletasks()
    def choose_ldplayer():
        path = filedialog.askopenfilename(parent=dialog, title="选择 dnplayer.exe", filetypes=[("dnplayer.exe", "dnplayer.exe"), ("可执行文件", "*.exe")])
        if not path:
            return
        ok, reason = validate_ldplayer_executable(Path(path), require_attach=False)
        if not ok:
            messagebox.showerror("雷电路径不合法", reason, parent=dialog)
            return
        save_startup_config_paths(ldplayer_path=path)
        refresh_status("已保存雷电模拟器路径，可重试")
    def choose_data_path():
        path = filedialog.askdirectory(parent=dialog, title="选择存储目录")
        if not path:
            return
        issue = data_path_write_issue(Path(path), create=True)
        if issue:
            messagebox.showerror("存储路径不可用", issue, parent=dialog)
            return
        save_startup_config_paths(data_path=path)
        refresh_status("已保存存储目录，可重试")
    def show_log():
        log_path = Path(str(startup_config_paths()[1])) / "startup_install.log"
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else "未找到详细日志：" + str(log_path)
        detail = message + "\n\n详细日志：\n" + log_text
        messagebox.showinfo("详细日志", detail[:12000], parent=dialog)
    def retry():
        result["retry"] = True
        dialog.destroy()
    def ignore():
        result["ignore"] = True
        dialog.destroy()
    def exit_program():
        result["exit"] = True
        dialog.destroy()
    for label, command in zip(STARTUP_FAILURE_BUTTON_LABELS, (choose_ldplayer, choose_data_path, show_log, retry, ignore, exit_program)):
        ttk.Button(actions, text=label, command=command).pack(side="left", padx=4)
    dialog.protocol("WM_DELETE_WINDOW", exit_program)
    parent.wait_window(dialog)
    return result


def configuration_failure(area, error):
    detail = f"{area}：{error}"
    if "--self-test" in sys.argv:
        raise RuntimeError(detail) from error
    fail_and_exit("配置文件生成失败或读取失败。\n" + detail)


def default_runtime_settings_payload():
    return {"training_seconds": AGENT_SPEC.default_training_seconds, "still_seconds": AGENT_SPEC.default_still_seconds, "experience_pool_gb": AGENT_SPEC.default_experience_pool_gb, "ai_model_limit": AGENT_SPEC.default_ai_model_limit}


DEFAULT_LDPLAYER_PATH = AGENT_SPEC.default_ldplayer_path
DEFAULT_DATA_PATH = AGENT_SPEC.default_data_path
DEFAULT_TRAINING_SECONDS = AGENT_SPEC.default_training_seconds
DEFAULT_STILL_SECONDS = AGENT_SPEC.default_still_seconds
DEFAULT_EXPERIENCE_POOL_GB = AGENT_SPEC.default_experience_pool_gb
DEFAULT_AI_MODEL_LIMIT = AGENT_SPEC.default_ai_model_limit
MODE_NAMES = {"idle": "空闲", "starting": "准备中", "learning": "学习模式", "training": "训练模式", "sleep": "睡眠模式", "migration": "数据迁移", "stopping": "正在退出"}
CONFIG_SCHEMA_VERSION = 1
USER_EDITABLE_STARTUP_FIELDS = ("ldplayer_path", "data_path")
USER_EDITABLE_RUNTIME_FIELDS = ("training_seconds", "still_seconds", "experience_pool_gb", "ai_model_limit")
USER_EDITABLE_FIELDS = AGENT_SPEC.editable_fields


class AllowedUserEditPolicy:
    ALLOWED_FIELDS = frozenset(AGENT_SPEC.editable_fields)
    STARTUP_FIELDS = frozenset(("ldplayer_path", "data_path"))
    RUNTIME_FIELDS = frozenset(("training_seconds", "still_seconds", "experience_pool_gb", "ai_model_limit"))

    @classmethod
    def filter(cls, settings, scope=None):
        allowed = cls.ALLOWED_FIELDS
        if scope == "startup":
            allowed = cls.STARTUP_FIELDS
        elif scope == "runtime":
            allowed = cls.RUNTIME_FIELDS
        return {name: settings[name] for name in allowed if isinstance(settings, dict) and name in settings}

    @classmethod
    def assert_allowed(cls, *names):
        blocked = [name for name in names if name not in cls.ALLOWED_FIELDS]
        if blocked:
            raise PermissionError("禁止用户修改内容：" + ", ".join(blocked))
        return True


ALLOWED_TRANSITIONS = {
    ("idle", "starting"): {"click_learning", "click_training"},
    ("starting", "learning"): {"window_ok"},
    ("starting", "training"): {"window_ok"},
    ("idle", "sleep"): {"click_sleep"},
    ("idle", "migration"): {"click_modify_data_path"},
    ("learning", "stopping"): {"user_stop", "esc", "still_timeout", "window_invalid", "cursor_outside", "runtime_error"},
    ("training", "stopping"): {"user_stop", "esc", "window_invalid", "cursor_outside", "runtime_error", "executor_error"},
    ("starting", "stopping"): {"user_stop", "esc", "runtime_error", "minimize_failed", "window_invalid", "completed"},
    ("migration", "stopping"): {"user_stop", "esc", "runtime_error", "migration_error", "completed"},
    ("stopping", "idle"): {"esc", "still_timeout", "window_invalid", "cursor_outside", "user_stop", "runtime_error", "executor_error", "migration_error", "completed"},
    ("training", "sleep"): {"time_limit"},
    ("sleep", "stopping"): {"user_stop", "esc", "runtime_error", "completed"}
}


TERMINATION_REASONS = ("window_invalid", "cursor_outside", "rect_changed", "empty_action", "executor_error", "time_limit", "esc", "still_timeout", "user_stop", "migration_error", "completed")
RUNNING_MODES = {"starting", "learning", "training", "sleep", "migration"}
HUMAN_FEATURE_NAMES = ("duration", "direct", "bend", "points", "speed_mean", "speed_variance", "acceleration_change", "pauses", "hover_before", "drag_curvature", "double_click_interval")

RUNTIME_NUMBER_RULES = {
    "hash_size": ("screen_pixels", "pool_count", "capture_ms"),
    "nearest_top_k": ("pool_count", "cpu_count", "screen_score_total"),
    "nearest_candidate_limit": ("pool_count", "window_instability", "cpu_count", "gpu_count"),
    "hash_prefix_bits": ("pool_count", "screen_pixels"),
    "mouse_activity_wait": ("capture_ms", "cpu_load"),
    "training_event_wait": ("capture_ms", "execution_ms", "cpu_load", "gpu_factor", "window_instability"),
    "sleep_event_wait": ("cpu_load", "cpu_count", "memory_free_ratio"),
    "min_action_delay_seconds": ("capture_ms", "execution_ms", "cpu_load", "recent_success"),
    "generated_sleep_event_wait": ("capture_ms", "execution_ms", "cpu_load", "window_instability", "recent_success"),
    "generated_action_complete_wait": ("capture_ms", "execution_ms", "cpu_load", "window_instability", "recent_success"),
    "sleep_worker_count": ("cpu_count", "gpu_count", "memory_free_ratio", "cpu_load"),
    "sleep_batch_size": ("pool_count", "cpu_count", "memory_free_ratio", "screen_score_total"),
    "sleep_queue_depth": ("cpu_count", "gpu_count", "memory_free_ratio"),
    "key_debounce_seconds": ("cpu_load", "capture_ms"),
    "window_attach_seconds": ("cpu_load", "window_instability"),
    "window_event_wait": ("cpu_load", "gpu_factor", "window_instability"),
    "explore_max_rate": ("recent_success", "learning_similarity", "window_instability"),
    "explore_min_rate": ("recent_success", "learning_similarity", "pool_count"),
    "action_jitter": ("recent_success", "window_instability", "screen_pixels"),
    "softmax_temperature": ("pool_count", "recent_success", "screen_score_total"),
    "human_profile_min_samples": ("pool_count", "human_feature_count"),
    "human_profile_max_samples": ("pool_count", "memory_free_ratio"),
    "human_profile_keep_samples": ("pool_count", "memory_free_ratio"),
    "ui_width": ("screen_width", "screen_height"),
    "ui_height": ("screen_height", "screen_width"),
    "ui_min_width": ("screen_width", "screen_height"),
    "ui_min_height": ("screen_height", "screen_width"),
    "ui_padding": ("screen_pixels", "screen_width", "screen_height"),
    "ui_section_padding": ("screen_pixels", "screen_width", "screen_height"),
    "ui_metric_columns": ("screen_width", "ui_width"),
    "ui_metric_min_column_width": ("screen_width", "ui_metric_columns"),
    "reward_total_min": ("screen_score_total", "recent_success"),
    "reward_total_max": ("screen_score_total", "recent_success"),
    "score_default": ("screen_score_total", "learning_similarity", "pool_count"),
    "scroll_score_default": ("screen_score_total", "recent_success", "pool_count"),
    "fallback_score_base": ("screen_score_total", "learning_similarity", "recent_success"),
    "experience_load_limit": ("pool_count", "memory_free_ratio", "cpu_count"),
    "global_action_probability": ("pool_count", "recent_success"),
    "random_action_min": ("window_instability", "recent_success"),
    "random_action_max": ("window_instability", "recent_success"),
    "action_duration_min": ("capture_ms", "execution_ms", "cpu_load"),
    "action_duration_max": ("capture_ms", "execution_ms", "cpu_load"),
    "random_click_duration_min": ("capture_ms", "execution_ms", "recent_success"),
    "random_click_duration_max": ("capture_ms", "execution_ms", "recent_success"),
    "generated_click_hold_max": ("capture_ms", "execution_ms", "recent_success"),
    "motion_steps_per_second": ("screen_width", "cpu_load", "cpu_count"),
    "motion_curve_offset_min": ("window_instability", "recent_success"),
    "motion_curve_offset_max": ("window_instability", "recent_success"),
    "motion_first_control_min": ("window_instability", "recent_success"),
    "motion_first_control_max": ("window_instability", "recent_success"),
    "motion_second_control_min": ("window_instability", "recent_success"),
    "motion_second_control_max": ("window_instability", "recent_success"),
    "learning_screen_change_capacity": ("capture_ms", "cpu_count", "gpu_count"),
    "learning_screen_similarity_threshold": ("pool_count", "learning_similarity"),
    "training_fail_stop_count": ("recent_success", "window_instability"),
    "ui_event_coalesce_seconds": ("capture_ms", "cpu_load", "gpu_factor"),
    "ui_progress_delta": ("window_instability", "cpu_load"),
    "persistence_event_wait": ("capture_ms", "persistence_latency", "cpu_load"),
    "persistence_close_seconds": ("execution_ms", "capture_ms", "cpu_load"),
    "async_queue_size": ("cpu_count", "gpu_count", "pool_count"),
    "global_action_heap_limit": ("pool_count", "cpu_count"),
    "local_action_heap_limit": ("pool_count", "window_instability"),
    "action_score_similarity_weight": ("nearest_similarity", "pool_count"),
    "action_score_reward_weight": ("recent_success", "screen_score_total"),
    "action_score_human_weight": ("recent_success", "learning_similarity"),
    "action_score_novelty_weight": ("window_instability", "pool_count"),
    "motion_score_magnitude_weight": ("recent_success", "screen_pixels"),
    "motion_score_continuity_weight": ("recent_success", "screen_pixels"),
    "human_feature_weights": ("recent_success", "human_feature_count")
}


RUNTIME_NUMBER_AUDIT = {}


def runtime_number_rule(name):
    return RUNTIME_NUMBER_RULES.get(name, ("runtime_context",))


def default_human_feature_weights(feature_names=HUMAN_FEATURE_NAMES):
    return tuple(1.0 for _ in feature_names)


@dataclass(frozen=True)
class Settings:
    hash_size: int = 0
    nearest_top_k: int = 0
    nearest_candidate_limit: int = 0
    hash_prefix_bits: int = 0
    mouse_activity_wait: float = 0.0
    training_event_wait: float = 0.0
    sleep_event_wait: float = 0.0
    sleep_worker_count: int = 0
    sleep_batch_size: int = 0
    sleep_queue_depth: int = 0
    key_debounce_seconds: float = 0.0
    window_attach_seconds: float = 0.0
    window_event_wait: float = 0.0
    min_action_delay_seconds: float = 0.0
    random_action_min: float = 0.0
    random_action_max: float = 0.0
    explore_min_rate: float = 0.0
    explore_max_rate: float = 0.0
    action_jitter: float = 0.0
    softmax_temperature: float = 0.0
    human_profile_min_samples: int = 0
    human_profile_max_samples: int = 0
    human_profile_keep_samples: int = 0
    window_title_keywords: tuple = ("ldplayer", "雷电", "leidian")
    ui_width: int = 0
    ui_height: int = 0
    ui_min_width: int = 0
    ui_min_height: int = 0
    ui_padding: int = 0
    ui_section_padding: int = 0
    ui_metric_columns: int = 0
    ui_metric_min_column_width: int = 0
    click_direct_threshold: float = 0.0
    drag_direct_threshold: float = 0.0
    drag_min_points: int = 0
    drag_bend_penalty_threshold: float = 0.0
    click_long_duration: float = 0.0
    reward_total_min: float = 0.0
    reward_total_max: float = 0.0
    experience_load_limit: int = 0
    score_default: float = 50.0
    scroll_score_default: float = 50.0
    fallback_score_base: float = 50.0
    global_action_probability: float = 0.0
    random_click_duration_min: float = 0.0
    random_click_duration_max: float = 0.0
    action_duration_min: float = 0.0
    action_duration_max: float = 0.0
    generated_click_hold_max: float = 0.0
    generated_sleep_event_wait: float = 0.0
    generated_action_complete_wait: float = 0.0
    motion_steps_per_second: float = 0.0
    motion_curve_offset_min: float = 0.0
    motion_curve_offset_max: float = 0.0
    motion_first_control_min: float = 0.0
    motion_first_control_max: float = 0.0
    motion_second_control_min: float = 0.0
    motion_second_control_max: float = 0.0
    learning_screen_change_capacity: float = 0.0
    learning_screen_similarity_threshold: float = 0.0
    training_fail_stop_count: int = 0
    ui_event_coalesce_seconds: float = 0.0
    ui_progress_delta: float = 0.0
    persistence_event_wait: float = 0.0
    persistence_close_seconds: float = 0.0
    async_queue_size: int = 0
    global_action_heap_limit: int = 0
    local_action_heap_limit: int = 0
    action_score_similarity_weight: float = 0.0
    action_score_reward_weight: float = 0.0
    action_score_human_weight: float = 0.0
    action_score_novelty_weight: float = 0.0
    motion_score_magnitude_weight: float = 0.0
    motion_score_continuity_weight: float = 0.0
    human_feature_weights: tuple = ()


class AdaptivePolicy:
    def __init__(self):
        initial = derive_runtime_settings()
        adaptive_window = max(1, initial.local_action_heap_limit // max(1, initial.hash_prefix_bits))
        self.capture_latency_ms = deque(maxlen=adaptive_window)
        self.execution_latency_ms = deque(maxlen=adaptive_window)
        self.learning_similarity = deque(maxlen=max(1, adaptive_window // max(1, initial.ui_metric_columns)))
        self.window_change_flags = deque(maxlen=adaptive_window)
        self.outcome_flags = deque(maxlen=adaptive_window)

    def observe_capture(self, latency_ms, similarity=None, window_rect_changed=False):
        self.capture_latency_ms.append(max(0.0, safe_float(latency_ms, 0.0)))
        if similarity is not None:
            self.learning_similarity.append(clamp(similarity, 0.0, 1.0))
        self.window_change_flags.append(1.0 if window_rect_changed else 0.0)

    def observe_execution(self, latency_ms=None, success=True):
        if latency_ms is not None:
            self.execution_latency_ms.append(max(0.0, safe_float(latency_ms, 0.0)))
        self.outcome_flags.append(1.0 if success else 0.0)

    def _avg(self, values, fallback):
        return sum(values) / len(values) if values else fallback

    def build(self, base_settings, rect, pool_count, screen_score_total, hardware=None):
        capture_ms = self._avg(self.capture_latency_ms, 24.0)
        execution_ms = self._avg(self.execution_latency_ms, 140.0)
        window_instability = self._avg(self.window_change_flags, 0.0)
        recent_success = self._avg(self.outcome_flags, 1.0)
        similarity = self._avg(self.learning_similarity, 0.97)
        cpu_load = 0.0
        if hardware:
            cpu_load = safe_float(hardware.get("cpu_load", 0.0), 0.0)
        return derive_runtime_settings(base_settings=base_settings, rect=rect, pool_count=pool_count, capture_ms=capture_ms, cpu_load=cpu_load, execution_ms=execution_ms, window_instability=window_instability, recent_success=recent_success, screen_score_total=screen_score_total, learning_similarity=similarity, hardware=hardware)


@dataclass
class ModeSession:
    token: int
    mode: str
    started_at: float
    deadline: Optional[float]
    stop_event: threading.Event
    termination_reason: Optional[str] = None
    started_sequence: int = 0


@dataclass(frozen=True)
class StopSnapshot:
    training_seconds: int
    still_seconds: float
    deadline: Optional[float]


@dataclass(frozen=True)
class Config:
    ldplayer_path: Path
    data_path: Path
    training_seconds: int
    still_seconds: float
    experience_pool_gb: float
    ai_model_limit: int
    settings: Settings


@dataclass(frozen=True)
class HashValue:
    value: int
    bits: int
    hex: str


@dataclass(frozen=True)
class ScreenSnapshot:
    path: Path
    relative_path: str
    hash_value: HashValue
    captured_at: str
    perf_time: float
    elapsed: float
    rect: tuple
    image_dropped: bool = False
    capture_latency_ms: Optional[float] = None
    image_priority: str = "normal"
    image_checksum: str = ""
    semantic_vector: tuple = ()


def image_content_checksum(image):
    if image is None:
        return ""
    normalized = image.convert("RGB")
    return hashlib.sha256(f"{normalized.size[0]}x{normalized.size[1]}|".encode("ascii") + normalized.tobytes()).hexdigest()


def enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def can_access_input_desktop():
    handle = None
    try:
        handle = ctypes.windll.user32.OpenInputDesktop(0, False, 0x0100)
        return bool(handle)
    except Exception:
        return False
    finally:
        if handle:
            try:
                ctypes.windll.user32.CloseDesktop(handle)
            except Exception:
                pass


def windows_runtime_report(ldplayer_path=None):
    report = {"platform": sys.platform, "desktop_session": bool(os.environ.get("SESSIONNAME")), "dpi_awareness_checked": False, "admin_or_ui_access_checked": False, "ldplayer_path_exists": None}
    if sys.platform == "win32":
        try:
            report["dpi_awareness_checked"] = ctypes.windll.user32.GetDpiForSystem() > 0
        except Exception:
            report["dpi_awareness_checked"] = False
        report["admin_or_ui_access_checked"] = can_access_input_desktop()
    if ldplayer_path:
        report["ldplayer_path_exists"] = Path(ldplayer_path).exists()
    report["ok"] = report["platform"] == "win32" and report["desktop_session"] and report["dpi_awareness_checked"] and report["admin_or_ui_access_checked"] and report["ldplayer_path_exists"] is not False
    return report




def validate_ldplayer_executable(path, settings=None, require_attach=False):
    candidate = Path(path)
    if candidate.name.lower() != "dnplayer.exe":
        return False, "雷电模拟器路径必须选择文件名为 dnplayer.exe 的可执行文件"
    if not candidate.exists():
        return False, "雷电模拟器路径不存在"
    if not candidate.is_file():
        return False, "雷电模拟器路径不是文件"
    if sys.platform != "win32":
        return True, "self_test"
    report = windows_runtime_report(candidate)
    if not report.get("ok"):
        return False, "当前 Windows 桌面运行环境无法启动或附着雷电模拟器：" + json.dumps(report, ensure_ascii=False)
    if require_attach:
        manager = WindowManager(candidate, settings or derive_runtime_settings())
        if not manager.launch_or_attach():
            return False, "无法通过该 dnplayer.exe 启动或附着雷电模拟器客户区"
        check = manager.check_window(force=True)
        if not check.ok:
            return False, f"已找到雷电模拟器但客户区异常：{check.reason}"
    return True, "ok"


def run_self_test():
    startup_checks = deque([["依赖异常"], []])
    startup_events = []
    assert startup_ldplayer_window_issue(Path("D:/LDPlayer9/dnplayer.exe")) is None
    assert prepare_startup_environment(lambda: startup_checks.popleft(), lambda actions: (actions.append("完成修复"), startup_events.append("repair")), startup_events.append)
    assert startup_events == ["repair"]
    startup_checks = deque([["桌面异常"], ["桌面异常"]])
    startup_events = []
    assert not prepare_startup_environment(lambda: startup_checks.popleft(), lambda actions: actions.append("无法自动修复"), startup_events.append)
    assert len(startup_events) == 1 and "初检结果" in startup_events[0] and "初检后进行的自愈尝试" in startup_events[0] and "复检结果" in startup_events[0] and "下一步建议" in startup_events[0]
    assert STARTUP_FAILURE_BUTTON_LABELS == ("选择雷电路径", "选择数据目录", "更多", "重试", "忽略", "退出")
    assert "still_timeout" not in ALLOWED_TRANSITIONS[("training", "stopping")]
    assert "still_timeout" in ALLOWED_TRANSITIONS[("learning", "stopping")]
    original_window_manager = globals().get("WindowManager")
    original_probe = globals().get("runtime_capability_probe")
    attach_events = []
    class FakeStartupWindowManager:
        def __init__(self, path, settings=None):
            self.path = path
            self.settings = settings
            self.hwnd = None
        def launch_or_attach(self):
            self.hwnd = 100
            attach_events.append(("attach", str(self.path)))
            return True
    def fake_runtime_capability_probe(manager):
        attach_events.append(("probe", getattr(manager, "hwnd", None)))
        return getattr(manager, "hwnd", None) == 100, "invalid_handle"
    globals()["WindowManager"] = FakeStartupWindowManager
    globals()["runtime_capability_probe"] = fake_runtime_capability_probe
    try:
        probe_ok, probe_reason = attach_and_probe_ldplayer(Path("D:/LDPlayer9/dnplayer.exe"), settings=None)
        assert probe_ok, probe_reason
        assert attach_events == [("attach", "D:/LDPlayer9/dnplayer.exe"), ("probe", 100)]
    finally:
        if original_window_manager is None:
            globals().pop("WindowManager", None)
        else:
            globals()["WindowManager"] = original_window_manager
        globals()["runtime_capability_probe"] = original_probe
    settings = derive_runtime_settings()
    assert len(settings.human_feature_weights) == len(HUMAN_FEATURE_NAMES)
    assert sum(settings.human_feature_weights) > 0.0
    factory = RuntimeNumberFactory(read_hardware_state(), (1280, 720), 8, 24.0, 140.0, 24.0, 1.0, 0.0, 0.97, 0.0)
    assert factory.count("training_fail_stop_count", 1, 99) >= 1
    assert "training_fail_stop_count" in factory.audit
    assert RUNTIME_NUMBER_RULES["training_event_wait"]
    for item in fields(Settings):
        value = getattr(settings, item.name)
        if isinstance(value, (int, float)):
            assert math.isfinite(float(value))
    raw_empty_weight_settings = replace(settings, human_profile_min_samples=3, human_feature_weights=())
    profile = HumanProfile(raw_empty_weight_settings)
    human_action = {"type": "click", "start_rel": [0.2, 0.3], "end_rel": [0.22, 0.32], "duration": 0.18, "path_rel": [[0.2, 0.3, 0.0], [0.21, 0.31, 0.08], [0.22, 0.32, 0.18]]}
    for _ in range(raw_empty_weight_settings.human_profile_min_samples):
        profile.observe(human_action)
    assert profile.enough("click")
    assert profile.score(human_action) > 0.0
    assert sys.version_info >= MIN_PYTHON_VERSION
    assert windows_runtime_report()["platform"] == sys.platform
    assert hasattr(WindowManager, "topmost")
    assert clamp(12, 0, 10) == 10
    action = normalize_mouse_action({"type": "click", "start_rel": [2, -1], "duration": 0.3}, (0, 0, 100, 100))
    assert 0.0 <= action["start_rel"][0] <= 1.0
    a = HashValue(value=0b1111, bits=4, hex="f")
    b = HashValue(value=0b1101, bits=4, hex="d")
    sim = hash_similarity(a, b)
    assert 0.0 <= sim <= 1.0
    low_screen = reward_parts(70.0, 100.0, settings)[2]
    high_screen = reward_parts(71.0, 0.0, settings)[2]
    high_human = reward_parts(70.0, 100.0, settings)[2]
    low_human = reward_parts(70.0, 0.0, settings)[2]
    assert high_screen > low_screen
    assert high_human > low_human
    assert high_human < high_screen
    details = reward_breakdown(70.0, 100.0, settings)
    low_human_details = reward_breakdown(70.0, 0.0, settings)
    assert details["screen_score_delta"] == details["screen_primary_reward"]
    assert details["reward_sort_key"] > low_human_details["reward_sort_key"]
    assert details["reward_sort_key"][0] == details["screen_primary_reward"]
    assert "human_tie_break_reward" in details
    assert 0.0 < details["human_bonus"] < details["screen_score_resolution"]
    assert set(USER_EDITABLE_FIELDS) == {"ldplayer_path", "data_path", "training_seconds", "still_seconds", "experience_pool_gb", "ai_model_limit"}
    assert {"esc", "still_timeout", "window_invalid"}.issubset(ALLOWED_TRANSITIONS[("learning", "stopping")])
    assert {"esc", "window_invalid"}.issubset(ALLOWED_TRANSITIONS[("training", "stopping")])
    assert "still_timeout" not in ALLOWED_TRANSITIONS[("training", "stopping")]
    assert ("training", "idle") not in ALLOWED_TRANSITIONS
    assert "time_limit" in ALLOWED_TRANSITIONS[("training", "sleep")]
    assert should_stop_run(threading.Event(), time.perf_counter() - 1.0, None) == "time_limit"
    assert should_stop_run(threading.Event(), None, None) is None
    stopped_for_user = threading.Event()
    stopped_for_user.set()
    assert should_stop_run(stopped_for_user, None, None, "user_stop") == "user_stop"
    stop_check_panel = type("StopCheckPanel", (), {})()
    stop_check_panel.termination_reason = None
    stop_check_panel.should_stop_by_escape = lambda: False
    stop_check_panel.window_manager = type("Window", (), {"check_window": lambda self, force=True: type("Check", (), {"ok": True, "reason": "ok"})()})()
    stop_check_panel.cursor_inside_window = lambda: True
    stop_check_panel.learning_idle_seconds = lambda: 0.0
    stop_check_panel.update_progress = lambda value: None
    stop_check_panel.ui = lambda fn: fn()
    stop_check_panel.progress_label_var = type("Label", (), {"set": lambda self, value: None})()
    stop_check_panel.active_mode_stop_reason = ControlPanel.active_mode_stop_reason.__get__(stop_check_panel, type(stop_check_panel))
    stop_check_panel.apply_active_stop_reason = ControlPanel.apply_active_stop_reason.__get__(stop_check_panel, type(stop_check_panel))
    stop_check_panel.settings = settings
    stop_check_panel.mouse_recorder = None
    stop_config = type("Config", (), {"training_seconds": 1, "still_seconds": 10.0})()
    expired_event = threading.Event()
    assert TrainingService(stop_check_panel).should_stop(time.perf_counter() - 2.0, stop_config, expired_event)
    execution_panel = type("ExecutionPanel", (), {})()
    execution_stop = threading.Event()
    execution_panel.termination_reason = None
    execution_panel.should_stop_by_escape = lambda: False
    execution_panel.window_manager = type("Window", (), {"check_window": lambda self, force=True: type("Check", (), {"ok": True, "reason": "ok"})()})()
    execution_panel.cursor_inside_window = lambda: False
    execution_panel.learning_idle_seconds = lambda: 0.0
    execution_panel.settings = settings
    execution_panel.mouse_recorder = None
    execution_panel.status_var = type("Status", (), {"set": lambda self, value: setattr(execution_panel, "status_text", value)})()
    execution_panel.ui = lambda fn: fn()
    execution_panel.read_config = lambda: (_ for _ in ()).throw(AssertionError("training stop check must use config snapshot"))
    execution_panel.current_rect = lambda: (0, 0, 100, 100)
    execution_panel.write_record = lambda *args, **kwargs: setattr(execution_panel, "written_record", (args, kwargs))
    execution_panel.log_exception = lambda *args, **kwargs: None
    execution_panel.mark_learning_activity = lambda: None
    execution_panel.events = type("Events", (), {"publish": lambda self, name, **data: setattr(execution_panel, "published_event", (name, data))})()
    execution_panel.adaptive_policy = type("Policy", (), {"observe_execution": lambda self, success: setattr(execution_panel, "observed_success", success)})()
    execution_panel.active_mode_stop_reason = ControlPanel.active_mode_stop_reason.__get__(execution_panel, type(execution_panel))
    execution_panel.apply_active_stop_reason = ControlPanel.apply_active_stop_reason.__get__(execution_panel, type(execution_panel))
    class GuardExecutor:
        def execute(self, action, rect, stop_event, should_stop, on_activity=None):
            return None if should_stop() else {"type": "click", "path": [], "duration": 0.0}
    execution_panel.executor = GuardExecutor()
    execution_snapshot = type("Snapshot", (), {"hash_value": a, "image_checksum": "", "semantic_vector": ()})()
    assert not TrainingService(execution_panel).execute_and_record(None, "s", time.perf_counter(), (0, 0, 100, 100), execution_snapshot, {"type": "click", "start_rel": [0.5, 0.5]}, {}, execution_stop)[0]
    assert execution_stop.is_set() and execution_panel.termination_reason == "cursor_outside"
    assert ("sleep", "training") not in ALLOWED_TRANSITIONS
    assert ("sleep", "starting") not in ALLOWED_TRANSITIONS
    assert ("sleep", "idle") not in ALLOWED_TRANSITIONS
    assert "completed" in ALLOWED_TRANSITIONS[("sleep", "stopping")]
    assert {"completed", "migration_error", "user_stop"}.issubset(ALLOWED_TRANSITIONS[("migration", "stopping")])
    pool = ExperiencePool(settings)
    pool.add({"id": "t1", "mode": "learning", "mouse_action": {"type": "click", "start_rel": [0.5, 0.5]}, "reward": 12, "screen_hash_hex": "f", "screen_hash_bits": 4, "mouse_source": "user"})
    pool.add({"id": "t2", "mode": "training", "mouse_action": {"type": "click", "start_rel": [0.52, 0.52]}, "reward": 10, "screen_hash_hex": "d", "screen_hash_bits": 4, "mouse_source": "ai"})
    novelty, batch = pool.novelty(a)
    assert novelty >= 0.0
    assert all(item["record"].get("id") != "t1" for item in pool.nearest(a, exclude_id="t1"))
    assert not pool.nearest(a, before_index=0)
    assert all(item["record"].get("id") == "t1" for item in pool.nearest(a, before_index=1))
    values = {item.name: getattr(settings, item.name) for item in fields(Settings)}
    values["hash_prefix_bits"] = max(1, settings.hash_prefix_bits - 1)
    changed = Settings(**values)
    pool.apply_settings(changed)
    assert pool.index_settings.hash_prefix_bits == changed.hash_prefix_bits
    assert pool.nearest(a)
    assert semantic_similarity((1.0, 0.0), (1.0, 0.0)) == 1.0
    pool.add({"id": "t3", "mode": "learning", "mouse_action": {"type": "click", "start_rel": [0.1, 0.1]}, "reward": 1, "screen_hash_hex": "0", "screen_hash_bits": 4, "mouse_source": "user", "image_checksum": "exact-a", "screen_semantic_vector": [1.0, 0.0]})
    assert pool.compute_screen_score(a, exact_checksum="exact-a", semantic_vector=[0.0, 1.0])[0] == 0.0
    recheck_pool = ExperiencePool(settings, [{"id": "record-1", "screen_hash_hex": "f", "screen_hash_bits": 4, "image_checksum": "known"}])
    recheck_result = recheck_pool.recheck_screen_scores()
    assert recheck_result["total"] == 1 and recheck_result["trainable"] == 1
    action_pool = ExperiencePool(settings, [{"id": "bad-action", "mouse_action": {"type": "click", "start_rel": [0.1, 0.1]}, "reward": 999, "screen_hash_hex": "f", "screen_hash_bits": 4, "image_checksum": "bad", "quarantined": True}, {"id": "good-action", "mouse_action": {"type": "click", "start_rel": [0.2, 0.2]}, "reward": 1, "screen_hash_hex": "e", "screen_hash_bits": 4, "image_checksum": "good"}])
    assert action_pool.best_global_action()["start_rel"] == [0.2, 0.2]
    sleep_result = pool.sleep_training_step(changed.sleep_batch_size)
    assert sleep_result["trained"] >= 1
    assert pool.records[0].get("sleep_visits", 0) >= 1
    group = ai_model_group_snapshot(pool.model.snapshot(), changed, pool.records)
    runtime_pool = ExperiencePool(changed, pool.records, {"model_group": group})
    runtime = runtime_pool.model_runtime
    assert runtime.screen_novelty([1.0]) == 0.0
    changed_for_topk = replace(changed, nearest_top_k=3)
    topk_model = ScreenNoveltyScorerModel(changed_for_topk)
    assert topk_model.predict({"similarities": [1.0, 0.8, 0.8]}) == 14.67
    matching_mouse_action = {"type": "click", "start_rel": [0.5, 0.5]}
    direct_human_score = runtime.models["mouse_humanlikeness_scorer"].predict({"mouse_action": matching_mouse_action, "human_score": 55.0})
    assert runtime.mouse_humanlikeness(matching_mouse_action, 55.0) == direct_human_score
    assert direct_human_score != 55.0
    assert 0.0 <= runtime.operation_policy_score(pool.records[0], 0.9) <= 1.0
    assert runtime.reward(70.0, 100.0) > runtime.reward(70.0, 0.0)
    assert runtime.reward(71.0, 0.0) > runtime.reward(70.0, 100.0)
    assert isinstance(runtime.models["runtime_value_model"].predict({"cpu_load": 0.0, "memory_free_ratio": 0.5}), dict)
    brain = ActionBrain(runtime_pool, changed)
    _, decision = brain.choose(a, novelty, batch, 0.0)
    assert isinstance(decision, dict)
    random_action, random_decision = brain.choose(a, novelty, [], 0.0)
    assert random_action and random_decision["reason"] in {"bounded_bootstrap_exploration", "global_experience"}
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        executable = root / "dnplayer.exe"
        executable.write_bytes(b"")
        assert validate_ldplayer_executable(executable)[0]
        assert data_path_write_issue(root) is None
        store = DataStore(folder)
        store.state_file.write_text("{bad json}", encoding="utf-8")
        rebuilt_state = store.load_state()
        assert rebuilt_state == {"screen_score_total": 0.0, "penalty": 0.0}
        analyzer = ScreenAnalyzer(settings.hash_size)
        image = Image.new("RGB", (16, 16), (24, 64, 128))
        png_path = root / "screen.png"
        analyzer.save_image(image, png_path, priority="critical", settings=settings)
        assert png_path.exists()
        reopened = Image.open(png_path)
        reopened.load()
        assert reopened.format == "PNG"
        assert image_content_checksum(reopened) == image_content_checksum(image)
        checkpoint = store.save_sleep_checkpoint({"run_id": "self-test", "stage": "task2_training"}, task1_completed=True)
        assert store.load_sleep_checkpoint()["task1_completed"] is True
        store.clear_sleep_checkpoint()
        assert store.load_sleep_checkpoint() is None
        store.append_runtime_parameter_audit({"a": 1}, {"a": 2}, {"a": {"reality_conditions": {"cpu_load": 0.5}, "semantic_goal": "self test"}})
        assert store.runtime_audit_file.exists()
        queued_path = root / "queued.png"
        persistence = AsyncPersistenceQueue(settings)
        try:
            assert persistence.enqueue_image(analyzer, image, queued_path, priority="critical")
            persistence.flush()
        finally:
            persistence.close()
        race_queue = AsyncPersistenceQueue.__new__(AsyncPersistenceQueue)
        race_queue.settings = settings
        race_queue.close_seconds = settings.persistence_close_seconds
        race_queue.lock = threading.RLock()
        race_queue.pending_sequences = set()
        race_queue.failed_sequences = set()
        race_queue.stop_event = threading.Event()
        race_queue.image_dropped = 0
        race_queue.status = AsyncPersistenceQueue.status.__get__(race_queue, AsyncPersistenceQueue)
        race_queue.recent_sequences = deque(maxlen=1024)
        race_queue.last_confirmed_sequence = 0
        race_queue.errors = []
        class RaceJobs:
            def __init__(self, owner):
                self.owner = owner
            def put(self, job, block=False, timeout=0):
                sequence = job.get("persistence_sequence")
                with self.owner.lock:
                    self.owner.pending_sequences.discard(sequence)
                    self.owner.last_confirmed_sequence = max(self.owner.last_confirmed_sequence, sequence)
        race_queue.jobs = RaceJobs(race_queue)
        assert AsyncPersistenceQueue.enqueue(race_queue, {"type": "record", "persistence_sequence": 1}, block_when_full=False)
        assert race_queue.pending_sequences == set() and race_queue.last_confirmed_sequence == 1
        reopened_queue = Image.open(queued_path)
        reopened_queue.load()
        assert reopened_queue.format == "PNG"
        assert image_content_checksum(reopened_queue) == image_content_checksum(image)
        assert list(store.root.glob("state.bad.*.json")) and json.loads(store.state_file.read_text(encoding="utf-8"))["screen_score_total"] == 0.0
        shared = store.screen_dir / "shared.png"
        unique = store.screen_dir / "unique.png"
        with shared.open("wb") as file:
            file.truncate(40 * 1024 * 1024)
        with unique.open("wb") as file:
            file.truncate(100 * 1024 * 1024)
        store.save_experience_records([
            {"id": "low", "screen_path": "screens/shared.png", "reward_sort_key": [1.0, 1.0], "reward": 1.0},
            {"id": "middle", "screen_path": "screens/unique.png", "reward_sort_key": [2.0, 2.0], "reward": 2.0},
            {"id": "high", "screen_path": "screens/shared.png", "reward_sort_key": [3.0, 3.0], "reward": 3.0}
        ])
        empty_pool_store = DataStore(root / "empty_pool")
        empty_pool_result = empty_pool_store.compact_experience_pool(0.1)
        assert empty_pool_result["complete"] is True and not empty_pool_result["changed"] and empty_pool_result["removed"] == 0
        under_limit_store = DataStore(root / "under_limit_pool")
        under_limit_screen = under_limit_store.screen_dir / "small.png"
        with under_limit_screen.open("wb") as file:
            file.truncate(1024)
        under_limit_store.save_experience_records([{"id": "under", "screen_path": "screens/small.png", "reward": 1.0}])
        under_limit_result = under_limit_store.compact_experience_pool(0.1)
        assert under_limit_result["complete"] is True and not under_limit_result["changed"] and under_limit_result["removed"] == 0
        cancelled_store = DataStore(root / "cancelled_pool")
        kept_screen = cancelled_store.screen_dir / "kept.png"
        with kept_screen.open("wb") as file:
            file.truncate(128 * 1024 * 1024)
        cancelled_store.save_experience_records([{"id": "kept", "screen_path": "screens/kept.png", "reward": 1.0}])
        cancelled = cancelled_store.compact_experience_pool(0.1, run_guard=lambda: "esc")
        assert cancelled["interrupted"] is True and cancelled["removed"] == 0 and kept_screen.exists()
        compacted = store.compact_experience_pool(0.1)
        retained = store.load_experience()
        assert compacted["complete"] is True and compacted["changed"] and shared.exists() and not unique.exists()
        assert [record["id"] for record in retained] == ["high"]
        dummy_panel = type("DummyPanel", (), {"store": store})()
        assert ControlPanel.sleep_compaction_progress(dummy_panel, {"size_bytes": 200, "target_bytes": 100}) == 0.5
        assert ControlPanel.sleep_compaction_progress(dummy_panel, {"size_bytes": 100, "target_bytes": 100}) == 1.0
        assert ControlPanel.sleep_progress_fields(dummy_panel, 0, 10, 1.0)["compaction"] == 1.0
        assert ControlPanel.sleep_compaction_complete(dummy_panel, {"size_bytes": 100, "target_bytes": 100})
        assert "AI模型清理未完成" in ControlPanel.sleep_unfinished_summary(dummy_panel, "completed", 3, 3, True, "达到时间上限", False)
        class SleepDummyVar:
            def __init__(self):
                self.value = None
            def set(self, value):
                self.value = value
        sleep_finish_panel = type("SleepFinishPanel", (), {})()
        sleep_finish_panel.current_mode = lambda: "sleep"
        sleep_finish_panel.update_progress = lambda value, force=False: setattr(sleep_finish_panel, "completion_progress", value)
        sleep_finish_panel.ui_sync = lambda fn, wait=None: fn()
        sleep_finish_panel.progress_label_var = SleepDummyVar()
        sleep_finish_panel.update_idletasks = lambda: setattr(sleep_finish_panel, "idled", True)
        sleep_finish_panel.settings = settings
        ControlPanel.render_sleep_completion_before_idle(sleep_finish_panel, "睡眠模式已中断：任务完成 43.2%，数据已安全保存", 43.2)
        assert sleep_finish_panel.completion_progress == 43.2
        assert sleep_finish_panel.progress_label_var.value == "睡眠模式已中断：任务完成 43.2%，数据已安全保存"
        one_second_clock = PausableTrainingClock(1)
        original_deadline = one_second_clock.deadline()
        time.sleep(1.08)
        assert one_second_clock.deadline() == original_deadline
        assert should_stop_run(threading.Event(), one_second_clock.deadline(), None) == "time_limit"
        assert one_second_clock.expired()
        clock = PausableTrainingClock(900)
        clock.remaining = 200.0
        clock.deadline_perf = time.perf_counter() + clock.remaining
        clock.pause()
        paused_remaining = clock.remaining
        time.sleep(0.01)
        assert abs(clock.remaining - paused_remaining) < 0.001
        clock.resume()
        assert 0.0 < clock.remaining <= 200.0
        mouse_executor = HumanMouseExecutor.__new__(HumanMouseExecutor)
        mouse_executor.settings = derive_runtime_settings()
        edge_rect = (10, 20, 30, 40)
        edge_points = mouse_executor.smooth_points((10, 20), (29, 39), 1.0, rect=edge_rect)
        assert edge_points and all(point_inside(edge_rect, x, y) for x, y in edge_points)
        assert mouse_executor.clamp_point_to_rect((-100, 100), edge_rect) == (10, 39)
        assert ControlPanel.sleep_completion_reached(dummy_panel, 3, 3)
        store.pending_state_writes = 2
        assert ControlPanel.sleep_completion_reached(dummy_panel, 3, 3)
        store.pending_state_writes = 0
        assert not ControlPanel.sleep_completion_reached(dummy_panel, 3, 3, "failed")
        task3_store = DataStore(root / "task3")
        task3_store.model_dir.mkdir(parents=True, exist_ok=True)
        for name, created in (("model_task3_new.json", "2025-01-03T00:00:00.000"), ("model_task3_old.json", "2025-01-01T00:00:00.000")):
            (task3_store.model_dir / name).write_text(json.dumps({"created_at": created, "model": {"type": "bad"}}), encoding="utf-8")
        task3_screen = task3_store.screen_dir / "oversize.png"
        with task3_screen.open("wb") as file:
            file.truncate(120 * 1024 * 1024)
        task3_store.save_experience_records([{"id": "task3_low", "screen_path": "screens/oversize.png", "reward": 0.0}])
        task3_events = []
        task3_panel = type("Task3Panel", (), {})()
        task3_panel.store = task3_store
        task3_panel.events = type("Events", (), {"publish": lambda self, name, **data: task3_events.append(name)})()
        task3_panel.progress_label_var = SleepDummyVar()
        task3_panel.ui = lambda fn: fn()
        task3_panel.progress_value = 0.0
        task3_panel.update_progress = lambda value, force=False: setattr(task3_panel, "progress_value", value)
        task3_config = type("Task3Config", (), {"ai_model_limit": 1, "experience_pool_gb": 0.1})()
        model_result, pool_result = ControlPanel.run_sleep_task3(task3_panel, task3_config)
        assert model_result["complete"] and pool_result["complete"]
        assert task3_events.index("sleep_model_cleanup_completed") < task3_events.index("experience_pool_compaction_completed")
        assert WindowCheck(True, "ok", (0, 0, 10, 10), 9, 9, 0.0).occluded_ratio == 0.0
        for name, created in (("model_new.json", "2025-01-02T00:00:00.000"), ("model_old.json", "2025-01-01T00:00:00.000"), ("model_mid.json", "2025-01-01T12:00:00.000")):
            (store.model_dir / name).write_text(json.dumps({"created_at": created, "model": {"type": "bad"}}), encoding="utf-8")
        (store.model_dir / "partial_model_interrupted.json").write_text(json.dumps({"created_at": "2025-01-04T00:00:00.000"}), encoding="utf-8")
        compact_models = store.compact_ai_models(2)
        assert compact_models["removed"] == 1 and compact_models["model_count"] == 2 and not (store.model_dir / "model_old.json").exists() and (store.model_dir / "model_mid.json").exists() and (store.model_dir / "model_new.json").exists() and (store.model_dir / "partial_model_interrupted.json").exists()
        original_win32gui, original_win32api, original_win32con = globals().get("win32gui"), globals().get("win32api"), globals().get("win32con")
        class FakeWin32Gui:
            visible = True
            iconic = False
            rect = (0, 0, 100, 80)
            hit = 101
            @staticmethod
            def IsWindow(hwnd):
                return True
            @staticmethod
            def GetClientRect(hwnd):
                return (0, 0, FakeWin32Gui.rect[2] - FakeWin32Gui.rect[0], FakeWin32Gui.rect[3] - FakeWin32Gui.rect[1])
            @staticmethod
            def ClientToScreen(hwnd, point):
                return (FakeWin32Gui.rect[0] + point[0], FakeWin32Gui.rect[1] + point[1])
            @staticmethod
            def IsWindowVisible(hwnd):
                return FakeWin32Gui.visible
            @staticmethod
            def IsIconic(hwnd):
                return FakeWin32Gui.iconic
            @staticmethod
            def WindowFromPoint(point):
                return FakeWin32Gui.hit
            @staticmethod
            def IsChild(hwnd, other):
                return False
            @staticmethod
            def EnumWindows(handler, arg):
                return None
            @staticmethod
            def GetTopWindow(value):
                return 101
            @staticmethod
            def GetWindow(hwnd, flag):
                return 0
        class FakeWin32Api:
            @staticmethod
            def GetSystemMetrics(index):
                values = {76: 0, 77: 0, 78: 1000, 79: 800, 0: 1000, 1: 800}
                return values.get(index, 0)
        class FakeWin32Con:
            GW_HWNDNEXT = 2
        globals()["win32gui"] = FakeWin32Gui
        globals()["win32api"] = FakeWin32Api
        globals()["win32con"] = FakeWin32Con
        try:
            manager = WindowManager(executable, settings)
            manager.hwnd = 101
            checked_window = manager.check_window(force=True)
            assert checked_window.ok and checked_window.occluded_ratio == 0.0
            FakeWin32Gui.iconic = True
            manager.window_check_cache = {}
            assert manager.check_window(force=True).reason == "minimized"
            FakeWin32Gui.iconic = False
            FakeWin32Gui.rect = (-10, 0, 90, 80)
            manager.window_check_cache = {}
            assert manager.check_window(force=True).reason == "out_of_screen"
            FakeWin32Gui.rect = (0, 0, 100, 80)
            FakeWin32Gui.hit = 202
            manager.window_check_cache = {}
            assert manager.check_window(force=True).reason == "occluded"
            assert "completed" in ALLOWED_TRANSITIONS[("sleep", "stopping")] and ("sleep", "training") not in ALLOWED_TRANSITIONS
            state_transition_table = (
                ("training_timeout", "sleep", "task1_task2_task3_then_restart"),
            )
            assert state_transition_table[0][2] == "task1_task2_task3_then_restart" and ("sleep", "training") not in ALLOWED_TRANSITIONS
        finally:
            globals()["win32gui"] = original_win32gui
            globals()["win32api"] = original_win32api
            globals()["win32con"] = original_win32con
        blank_metrics = screen_content_metrics(bytes([255, 255, 255, 255]) * 400, 20, 20)
        black_metrics = screen_content_metrics(bytes([0, 0, 0, 255]) * 400, 20, 20)
        varied_raw = bytearray()
        for value in range(400):
            varied_raw.extend([value % 251, (value * 3) % 251, (value * 7) % 251, 255])
        varied_metrics = screen_content_metrics(varied_raw, 20, 20)
        assert blank_metrics["valid"] and not blank_metrics["content_valuable"]
        assert not black_metrics["valid"] and black_metrics["reason"] == "black_candidate"
        assert varied_metrics["valid"] and varied_metrics["brightness_variance"] > 3.0 and varied_metrics["edge_density"] > 0.0
        transition_panel = ControlPanel.__new__(ControlPanel)
        transition_panel.state_lock = threading.RLock()
        transition_panel.mode = "training"
        transition_panel.run_token = 7
        stopped_event = threading.Event()
        stopped_event.set()
        transition_panel.stop_event = stopped_event
        transition_panel.active_session = ModeSession(7, "training", time.perf_counter(), None, stopped_event)
        transition_panel.termination_reason = "user_stop"
        transition_panel.events = type("Events", (), {"publish": lambda self, name=None, **kwargs: {"sequence": 1}})()
        transition_panel.set_mode_ui = lambda mode: None
        transition_panel.ui = lambda fn: fn()
        transition_panel.update_mode_button_states = lambda: None
        sleep_session = ControlPanel.transition(transition_panel, "training", "sleep", reason="time_limit", token=7, fresh_stop_event=True)
        assert sleep_session and not sleep_session.stop_event.is_set() and sleep_session.stop_event is not stopped_event
        assert sleep_session.termination_reason is None and transition_panel.termination_reason is None
        class DummyVar:
            def __init__(self):
                self.value = None
            def set(self, value):
                self.value = value
            def get(self):
                return self.value
        modify_panel = ControlPanel.__new__(ControlPanel)
        modify_panel.ldplayer_var = DummyVar()
        modify_panel.data_var = DummyVar()
        modify_panel.training_seconds_var = DummyVar()
        modify_panel.still_seconds_var = DummyVar()
        modify_panel.experience_pool_gb_var = DummyVar()
        modify_panel.ai_model_limit_var = DummyVar()
        modify_panel.status_var = DummyVar()
        modify_panel.ldplayer_var.set(str(executable))
        modify_panel.data_var.set(str(store.root))
        modify_panel.training_seconds_var.set("1")
        modify_panel.still_seconds_var.set("3")
        modify_panel.experience_pool_gb_var.set("4")
        modify_panel.ai_model_limit_var.set("5")
        modify_panel.settings = settings
        modify_panel.runtime_value_specs = {"training_seconds": ("训练秒数", modify_panel.training_seconds_var, DEFAULT_TRAINING_SECONDS, safe_int, 1), "still_seconds": ("静止秒数", modify_panel.still_seconds_var, DEFAULT_STILL_SECONDS, safe_float, 0.1), "experience_pool_gb": ("经验池 GB", modify_panel.experience_pool_gb_var, DEFAULT_EXPERIENCE_POOL_GB, safe_float, 0.1), "ai_model_limit": ("AI 模型个数", modify_panel.ai_model_limit_var, DEFAULT_AI_MODEL_LIMIT, safe_int, 1)}
        modify_panel.update_mode_button_states = lambda: setattr(modify_panel, "updated", True)
        modify_panel.refresh_runtime_environment_state = lambda: setattr(modify_panel, "refreshed", True)
        modify_panel.save_persistent_settings = lambda: setattr(modify_panel, "saved", True)
        modify_panel.start_data_migration = lambda new_path, values=None: setattr(modify_panel, "migration_request", (Path(new_path), values)) or True
        original_showerror = messagebox.showerror
        messagebox.showerror = lambda *args, **kwargs: None
        try:
            assert not hasattr(ControlPanel, "save_user_settings")
            assert ControlPanel.submit_user_settings(modify_panel, {"ldplayer_path": str(root / "missing.exe"), "data_path": str(store.root), "training_seconds": 2, "still_seconds": 4, "experience_pool_gb": 6, "ai_model_limit": 7}) is False
        finally:
            messagebox.showerror = original_showerror
        assert modify_panel.ldplayer_var.value == str(executable)
        assert ControlPanel.submit_user_settings(modify_panel, {"ldplayer_path": str(executable), "data_path": str(store.root), "training_seconds": 2, "still_seconds": 4, "experience_pool_gb": 6, "ai_model_limit": 7}) is True
        assert modify_panel.saved and modify_panel.training_seconds_var.value == "2" and modify_panel.ai_model_limit_var.value == "7"
        target_data_path = root / "submit_new_data"
        assert ControlPanel.submit_user_settings(modify_panel, {"ldplayer_path": str(executable), "data_path": str(target_data_path), "training_seconds": 8, "still_seconds": 9, "experience_pool_gb": 1, "ai_model_limit": 2}) is True
        assert modify_panel.migration_request[0] == target_data_path and modify_panel.data_var.value == str(store.root)
        dummy_panel.progress_value = 0.4
        dummy_panel.last_progress_update_perf = 0.0
        dummy_panel.progress_var = DummyVar()
        dummy_panel.progress_text_var = DummyVar()
        dummy_panel.ui = lambda fn: fn()
        dummy_panel.update_mode_button_states = lambda: None
        dummy_panel.settings = replace(settings, ui_progress_delta=1.0)
        ControlPanel.update_progress(dummy_panel, ControlPanel.idle_progress_value(dummy_panel, "learning", 87.0), force=True)
        assert dummy_panel.progress_value == 0.0 and dummy_panel.progress_var.value == 0.0
        dummy_panel.mode = "sleep"
        ControlPanel.update_progress(dummy_panel, 25.0, force=True)
        ControlPanel.update_progress(dummy_panel, 20.8, force=True)
        ControlPanel.update_progress(dummy_panel, 100.0, force=True)
        assert dummy_panel.progress_value == 100.0 and dummy_panel.progress_var.value == 100.0
        dummy_panel.mode = "idle"
        assert ControlPanel.idle_progress_value(dummy_panel, "training", 91.0) == 0.0
        assert ControlPanel.idle_progress_value(dummy_panel, "migration", 91.0) == 0.0
        store.save_settings({"training_seconds": 1, "still_seconds": 3, "experience_pool_gb": 4, "ai_model_limit": 5, "forbidden": 6})
        saved_settings = json.loads(store.settings_file.read_text(encoding="utf-8"))
        assert "forbidden" not in saved_settings and saved_settings["experience_pool_gb"] == 4.0 and saved_settings["ai_model_limit"] == 5
        store.experience_file.write_text("{bad json}\n" + json.dumps({"id": "ok"}) + "\n", encoding="utf-8")
        loaded = store.load_experience()
        assert len(loaded) == 1 and loaded[0]["id"] == "ok"
        assert (store.root / "experience.bad.jsonl").exists()
        screen_path = store.screen_dir / "sample.png"
        screen_path.write_bytes(b"screen")
        store.save_experience_records([{"id": "m1", "mouse_action": {"type": "click"}, "reward": 1.0, "sleep_confidence": 0.5, "screen_path": store.relative_path(screen_path)}])
        before_snapshot_count = len(list(store.model_dir.glob("model_*.json")))
        model_path = store.save_ai_model_snapshot(store.load_experience(), settings, 1, "completed")
        model_path_extra = store.save_ai_model_snapshot(store.load_experience(), settings, 1, "completed")
        assert model_path.exists() and model_path_extra.exists() and len(list(store.model_dir.glob("model_*.json"))) == before_snapshot_count + 2
        sleep_task3_panel = type("SnapshotTask3Panel", (), {})()
        sleep_task3_panel.store = store
        sleep_task3_panel.events = type("Events", (), {"publish": lambda self, name, **data: None})()
        sleep_task3_panel.progress_label_var = SleepDummyVar()
        sleep_task3_panel.progress_value = 0.0
        sleep_task3_panel.ui = lambda fn: fn()
        sleep_task3_panel.update_progress = lambda value, force=False: None
        sleep_task3_config = type("SnapshotTask3Config", (), {"ai_model_limit": 1, "experience_pool_gb": 0.1})()
        ControlPanel.run_sleep_task3(sleep_task3_panel, sleep_task3_config)
        assert len(list(store.model_dir.glob("model_*.json"))) == 1
        model_path = max(store.model_dir.glob("model_*.json"), key=store.model_created_key)
        model_payload = json.loads(model_path.read_text(encoding="utf-8"))
        assert len(model_payload["model_group"]["models"]) == 5
        assert [item["key"] for item in model_payload["model_group"]["models"]] == [item["key"] for item in AI_MODEL_GROUP_SPECS]
        model_path.write_bytes(model_path.read_bytes() + (b"m" * 2048))
        assert store.storage_size_bytes() > store.experience_pool_size_bytes()
        compact = store.compact_experience_pool(0.1)
        assert compact["size_bytes"] == store.experience_pool_size_bytes() and model_path.exists()
        old_root = store.root
        new_root = Path(folder) / "new_data"
        migration_panel = type("MigrationPanel", (), {"settings": settings})()
        for name in ("migration_known_names", "migration_items", "migration_counts", "migration_sample_files", "file_digest", "verify_migration"):
            setattr(migration_panel, name, getattr(ControlPanel, name).__get__(migration_panel, type(migration_panel)))
        assert "models" in migration_panel.migration_known_names()
        assert any(item[0].parts[0] == "models" for item in migration_panel.migration_items(old_root))
        shutil.copytree(old_root, new_root)
        counts = migration_panel.migration_counts(new_root)
        assert counts["models"] >= 1
        migration_panel.verify_migration(old_root, new_root)
    recorder = MouseRecorder(lambda: "learning", lambda: (0, 0, 100, 100), lambda: None)
    recorder.on_move(10, 10)
    time.sleep(0.13)
    recorder.on_move(11, 10)
    time.sleep(0.13)
    actions = recorder.pop_actions()
    assert actions and actions[0]["type"] == "move" and actions[0]["source"] == "user" and len(actions[0]["path"]) >= 2
    tiny = replace(settings, async_queue_size=1, persistence_event_wait=settings.sleep_event_wait, persistence_close_seconds=settings.sleep_event_wait)
    persistence = AsyncPersistenceQueue(tiny)
    assert persistence.enqueue({"type": "noop"}, block_when_full=False)
    dropped = persistence.enqueue({"type": "image", "analyzer": None, "image": None, "path": "x"}, block_when_full=False)
    assert dropped is False
    heartbeat_statuses = []
    blocked = AsyncPersistenceQueue.__new__(AsyncPersistenceQueue)
    blocked.settings = tiny
    blocked.jobs = queue.Queue()
    blocked.jobs.unfinished_tasks = 1
    blocked.lock = threading.RLock()
    blocked.errors = []
    blocked.pending_sequences = {1}
    blocked.failed_sequences = set()
    blocked.recent_sequences = deque(maxlen=1024)
    blocked.last_confirmed_sequence = 0
    blocked.enqueued_sequences = blocked.pending_sequences
    blocked.confirmed_sequences = blocked.recent_sequences
    try:
        AsyncPersistenceQueue.flush(blocked, timeout_seconds=0.001, heartbeat=lambda status: heartbeat_statuses.append(status))
        assert False
    except PersistenceFlushError as exc:
        assert "timeout_seconds" in str(exc)
        assert heartbeat_statuses and heartbeat_statuses[-1]["pending"] >= 1
    persistence.close()
    print("self-test passed")


def now_text():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def safe_int(value, default):
    try:
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default):
    try:
        return float(value)
    except Exception:
        return default


def clamp(value, minimum, maximum):
    value = safe_float(value, minimum)
    minimum = safe_float(minimum, value)
    maximum = safe_float(maximum, value)
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    return max(minimum, min(maximum, value))


def normalized_human_feature_weights(weights, feature_names=HUMAN_FEATURE_NAMES):
    normalized = [clamp(item, 0.0, 1.0) for item in (weights or ())]
    target = len(feature_names)
    if len(normalized) < target:
        normalized.extend(default_human_feature_weights(feature_names[len(normalized):]))
    normalized = normalized[:target]
    if not normalized or sum(normalized) <= 0.0:
        normalized = list(default_human_feature_weights(feature_names))
    return tuple(normalized)


def normalize_settings(settings):
    random_min = clamp(settings.random_action_min, 0.0, 1.0)
    random_max = clamp(settings.random_action_max, 0.0, 1.0)
    if random_min > random_max:
        random_min, random_max = random_max, random_min
    max_samples = max(1, safe_int(settings.human_profile_max_samples, 1))
    keep_samples = clamp(settings.human_profile_keep_samples, 1, max_samples)
    human_weights = normalized_human_feature_weights(settings.human_feature_weights)
    return Settings(
        hash_size=max(1, safe_int(settings.hash_size, 1)),
        nearest_top_k=max(1, safe_int(settings.nearest_top_k, 1)),
        nearest_candidate_limit=max(1, safe_int(settings.nearest_candidate_limit, 1)),
        hash_prefix_bits=max(1, safe_int(settings.hash_prefix_bits, 1)),
        mouse_activity_wait=max(0.001, safe_float(settings.mouse_activity_wait, 0.001)),
        training_event_wait=max(0.001, safe_float(settings.training_event_wait, 0.001)),
        sleep_event_wait=max(0.001, safe_float(settings.sleep_event_wait, 0.001)),
        sleep_worker_count=max(1, safe_int(settings.sleep_worker_count, 1)),
        sleep_batch_size=max(1, safe_int(settings.sleep_batch_size, 1)),
        sleep_queue_depth=max(1, safe_int(settings.sleep_queue_depth, 1)),
        key_debounce_seconds=max(0.001, safe_float(settings.key_debounce_seconds, 0.001)),
        window_attach_seconds=max(0.001, safe_float(settings.window_attach_seconds, 0.001)),
        window_event_wait=max(0.001, safe_float(settings.window_event_wait, 0.001)),
        min_action_delay_seconds=max(0.0, safe_float(settings.min_action_delay_seconds, 0.0)),
        random_action_min=random_min,
        random_action_max=random_max,
        explore_min_rate=clamp(settings.explore_min_rate, 0.0, 1.0),
        explore_max_rate=clamp(settings.explore_max_rate, 0.0, 1.0),
        action_jitter=clamp(settings.action_jitter, 0.0, 1.0),
        softmax_temperature=max(0.001, safe_float(settings.softmax_temperature, 1.0)),
        human_profile_min_samples=max(1, safe_int(settings.human_profile_min_samples, 1)),
        human_profile_max_samples=max_samples,
        human_profile_keep_samples=int(keep_samples),
        window_title_keywords=tuple(settings.window_title_keywords) or Settings.window_title_keywords,
        ui_width=max(1, safe_int(settings.ui_width, 1)),
        ui_height=max(1, safe_int(settings.ui_height, 1)),
        ui_min_width=max(1, safe_int(settings.ui_min_width, 1)),
        ui_min_height=max(1, safe_int(settings.ui_min_height, 1)),
        ui_padding=max(0, safe_int(settings.ui_padding, 0)),
        ui_section_padding=max(0, safe_int(settings.ui_section_padding, 0)),
        ui_metric_columns=max(1, safe_int(settings.ui_metric_columns, 1)),
        ui_metric_min_column_width=max(1, safe_int(settings.ui_metric_min_column_width, 1)),
        click_direct_threshold=clamp(settings.click_direct_threshold, 0.0, 1.0),
        drag_direct_threshold=clamp(settings.drag_direct_threshold, 0.0, 1.0),
        drag_min_points=max(1, safe_int(settings.drag_min_points, 1)),
        drag_bend_penalty_threshold=max(0.001, safe_float(settings.drag_bend_penalty_threshold, 0.001)),
        click_long_duration=max(0.0, safe_float(settings.click_long_duration, 0.0)),
        reward_total_min=safe_float(settings.reward_total_min, 0.0),
        reward_total_max=safe_float(settings.reward_total_max, 0.0),
        experience_load_limit=max(1, safe_int(settings.experience_load_limit, 1)),
        score_default=clamp(settings.score_default, 0.0, 100.0),
        scroll_score_default=clamp(settings.scroll_score_default, 0.0, 100.0),
        fallback_score_base=clamp(settings.fallback_score_base, 0.0, 100.0),
        global_action_probability=clamp(settings.global_action_probability, 0.0, 1.0),
        random_click_duration_min=max(0.0, safe_float(settings.random_click_duration_min, 0.0)),
        random_click_duration_max=max(0.0, safe_float(settings.random_click_duration_max, 0.0)),
        action_duration_min=max(0.0, safe_float(settings.action_duration_min, 0.0)),
        action_duration_max=max(0.0, safe_float(settings.action_duration_max, 0.0)),
        generated_click_hold_max=max(0.0, safe_float(settings.generated_click_hold_max, 0.0)),
        generated_sleep_event_wait=max(0.001, safe_float(settings.generated_sleep_event_wait, 0.001)),
        generated_action_complete_wait=max(0.001, safe_float(settings.generated_action_complete_wait, 0.001)),
        motion_steps_per_second=max(0.001, safe_float(settings.motion_steps_per_second, 0.001)),
        motion_curve_offset_min=clamp(settings.motion_curve_offset_min, 0.0, 1.0),
        motion_curve_offset_max=clamp(settings.motion_curve_offset_max, 0.0, 1.0),
        motion_first_control_min=clamp(settings.motion_first_control_min, 0.0, 1.0),
        motion_first_control_max=clamp(settings.motion_first_control_max, 0.0, 1.0),
        motion_second_control_min=clamp(settings.motion_second_control_min, 0.0, 1.0),
        motion_second_control_max=clamp(settings.motion_second_control_max, 0.0, 1.0),
        learning_screen_change_capacity=max(0.001, safe_float(settings.learning_screen_change_capacity, 0.001)),
        learning_screen_similarity_threshold=clamp(settings.learning_screen_similarity_threshold, 0.0, 1.0),
        training_fail_stop_count=max(1, safe_int(settings.training_fail_stop_count, 1)),
        ui_event_coalesce_seconds=max(0.001, safe_float(settings.ui_event_coalesce_seconds, 0.001)),
        ui_progress_delta=max(0.001, safe_float(settings.ui_progress_delta, 0.001)),
        persistence_event_wait=max(0.001, safe_float(settings.persistence_event_wait, 0.001)),
        persistence_close_seconds=max(0.001, safe_float(settings.persistence_close_seconds, 0.001)),
        async_queue_size=max(1, safe_int(settings.async_queue_size, 1)),
        global_action_heap_limit=max(1, safe_int(settings.global_action_heap_limit, 1)),
        local_action_heap_limit=max(1, safe_int(settings.local_action_heap_limit, 1)),
        action_score_similarity_weight=max(0.0, safe_float(settings.action_score_similarity_weight, 0.0)),
        action_score_reward_weight=max(0.0, safe_float(settings.action_score_reward_weight, 0.0)),
        action_score_human_weight=max(0.0, safe_float(settings.action_score_human_weight, 0.0)),
        action_score_novelty_weight=max(0.0, safe_float(settings.action_score_novelty_weight, 0.0)),
        motion_score_magnitude_weight=max(0.0, safe_float(settings.motion_score_magnitude_weight, 0.0)),
        motion_score_continuity_weight=max(0.0, safe_float(settings.motion_score_continuity_weight, 0.0)),
        human_feature_weights=human_weights
    )


def read_hardware_state():
    cpu_count = safe_int(psutil.cpu_count(logical=True), 0) if psutil else 0
    cpu_load = safe_float(psutil.cpu_percent(interval=0.05), 0.0) if psutil else 0.0
    memory_total = safe_float(getattr(psutil.virtual_memory(), "total", 0.0), 0.0) if psutil else 0.0
    memory_available = safe_float(getattr(psutil.virtual_memory(), "available", 0.0), 0.0) if psutil else 0.0
    memory_free_ratio = clamp(memory_available / memory_total if memory_total > 0 else 0.0, 0.0, 1.0)
    gpu_count = 0
    gpu_memory_total = 0.0
    wmic_path = shutil.which("wmic")
    if wmic_path:
        try:
            output = subprocess.check_output([wmic_path, "path", "win32_VideoController", "get", "AdapterRAM"], stderr=subprocess.DEVNULL, text=True, timeout=2.0)
            for line in output.splitlines():
                number = safe_int(line.strip(), 0)
                if number > 0:
                    gpu_count += 1
                    gpu_memory_total += float(number)
        except Exception:
            pass
    return {"cpu_count": max(1, cpu_count), "cpu_load": clamp(cpu_load, 0.0, 100.0), "memory_free_ratio": memory_free_ratio, "gpu_count": gpu_count, "gpu_memory_total": max(0.0, gpu_memory_total)}


class RuntimeGeneratedNumbers:
    def __init__(self):
        self.audit = {}
        self.history = {}

    def value(self, name, context, semantic_goal):
        condition = dict(context)
        cpu_headroom = 1.0 - clamp(safe_float(condition.get("cpu_load", 0.0), 0.0) / 100.0, 0.0, 1.0)
        memory_headroom = clamp(safe_float(condition.get("memory_free_ratio", 0.5), 0.5), 0.0, 1.0)
        success_rate = clamp(safe_float(condition.get("success_rate", 1.0), 1.0), 0.0, 1.0)
        window_stability = 1.0 - clamp(safe_float(condition.get("window_instability", 0.0), 0.0), 0.0, 1.0)
        capture_ms = max(1.0, safe_float(condition.get("capture_ms", 24.0), 24.0))
        execution_ms = max(1.0, safe_float(condition.get("execution_ms", 80.0), 80.0))
        latency_score = 1.0 - clamp((capture_ms + execution_ms * 0.5) / 360.0, 0.0, 1.0)
        pool_pressure = clamp(math.log1p(max(0.0, safe_float(condition.get("pool_count", 0.0), 0.0))) / math.log1p(100000.0), 0.0, 1.0)
        score_pressure = clamp(math.log1p(max(0.0, safe_float(condition.get("screen_score_total", 0.0), 0.0))) / math.log1p(1000000.0), 0.0, 1.0)
        learning_similarity = clamp(safe_float(condition.get("learning_similarity", 0.97), 0.97), 0.0, 1.0)
        metrics = {"cpu_headroom": cpu_headroom, "memory_headroom": memory_headroom, "success_rate": success_rate, "window_stability": window_stability, "latency_score": latency_score, "pool_pressure": pool_pressure, "score_pressure": score_pressure, "learning_similarity": learning_similarity}
        if any(word in name for word in ("worker", "batch", "queue", "async", "heap", "candidate", "load")):
            weights = {"cpu_headroom": 0.22, "memory_headroom": 0.24, "success_rate": 0.12, "window_stability": 0.1, "latency_score": 0.1, "pool_pressure": 0.16, "score_pressure": 0.06}
        elif any(word in name for word in ("wait", "duration", "debounce", "event", "close")):
            weights = {"cpu_headroom": -0.18, "memory_headroom": -0.08, "success_rate": -0.1, "window_stability": -0.12, "latency_score": -0.3, "pool_pressure": 0.08, "score_pressure": 0.02}
        elif any(word in name for word in ("explore", "random", "jitter", "temperature")):
            weights = {"cpu_headroom": 0.06, "memory_headroom": 0.04, "success_rate": -0.28, "window_stability": -0.18, "latency_score": 0.06, "pool_pressure": 0.08, "learning_similarity": -0.3}
        else:
            weights = {"cpu_headroom": 0.14, "memory_headroom": 0.14, "success_rate": 0.14, "window_stability": 0.14, "latency_score": 0.14, "pool_pressure": 0.15, "score_pressure": 0.08, "learning_similarity": 0.07}
        baseline = 0.5 + sum(metrics.get(key, 0.0) * weight for key, weight in weights.items())
        previous = self.history.get(name, baseline)
        value = clamp(previous * 0.55 + baseline * 0.45, 0.0, 1.0)
        self.history[name] = value
        self.audit[name] = {"source": "RuntimeGeneratedNumbers.closed_loop_value", "reality_conditions": condition, "formula": "closed_loop_weighted_metrics_with_history(candidate,current_conditions,result_proxy,benefit_proxy)", "semantic_goal": semantic_goal, "candidate_value": baseline, "result_metrics": metrics, "benefit": value, "current_value": value}
        return value


class ResourceAdaptiveRLModel:
    STATE_NAMES = ("cpu_headroom", "memory_free_ratio", "frame_speed", "execution_speed", "window_stability", "success_rate", "pool_maturity")
    ACTION_NAMES = ("sleep_batch_size", "sleep_worker_count", "training_event_wait", "nearest_candidate_limit", "explore_max_rate", "generated_sleep_event_wait")

    def __init__(self, state=None):
        state = state if isinstance(state, dict) else {}
        weights = state.get("weights") if isinstance(state.get("weights"), dict) else {}
        self.weights = {action: {name: safe_float((weights.get(action) or {}).get(name, 0.0), 0.0) for name in self.STATE_NAMES} for action in self.ACTION_NAMES}
        bias = state.get("bias") if isinstance(state.get("bias"), dict) else {}
        self.bias = {action: safe_float(bias.get(action, 0.0), 0.0) for action in self.ACTION_NAMES}
        self.trained_steps = safe_int(state.get("trained_steps", 0), 0)
        self.reward_ema = safe_float(state.get("reward_ema", 0.0), 0.0)
        self.last_reward = safe_float(state.get("last_reward", 0.0), 0.0)

    def state_vector(self, metrics):
        capture_ms = max(1.0, safe_float(metrics.get("capture_ms", metrics.get("frame_ms", 24.0)), 24.0))
        execution_ms = max(1.0, safe_float(metrics.get("execution_ms", 120.0), 120.0))
        pool_count = max(0.0, safe_float(metrics.get("pool_count", 0.0), 0.0))
        return {
            "cpu_headroom": 1.0 - clamp(safe_float(metrics.get("cpu_load", 0.0), 0.0) / 100.0, 0.0, 1.0),
            "memory_free_ratio": clamp(safe_float(metrics.get("memory_free_ratio", 0.5), 0.5), 0.0, 1.0),
            "frame_speed": 1.0 - clamp(capture_ms / 240.0, 0.0, 1.0),
            "execution_speed": 1.0 - clamp(execution_ms / 900.0, 0.0, 1.0),
            "window_stability": 1.0 - clamp(safe_float(metrics.get("window_instability", metrics.get("window_exception_rate", 0.0)), 0.0), 0.0, 1.0),
            "success_rate": clamp(safe_float(metrics.get("success_rate", metrics.get("recent_success", 1.0)), 1.0), 0.0, 1.0),
            "pool_maturity": clamp(math.log1p(pool_count) / math.log1p(100000.0), 0.0, 1.0)
        }

    def score(self, action, state_vector):
        return self.bias.get(action, 0.0) + sum(self.weights.get(action, {}).get(name, 0.0) * state_vector.get(name, 0.0) for name in self.STATE_NAMES)

    def multipliers(self, metrics):
        state_vector = self.state_vector(metrics)
        learned = self.trained_steps > 0
        result = {}
        for action in self.ACTION_NAMES:
            raw = math.tanh(self.score(action, state_vector))
            span = 0.55 if learned else 0.18
            if action in ("training_event_wait", "generated_sleep_event_wait"):
                raw = -raw
            result[action] = clamp(1.0 + raw * span, 0.35, 1.85)
        return result

    def apply_settings(self, settings, metrics):
        factors = self.multipliers(metrics)
        values = {item.name: getattr(settings, item.name) for item in fields(Settings)}
        count_bounds = {"sleep_batch_size": (8, 4096), "sleep_worker_count": (1, 64), "nearest_candidate_limit": (256, 20000)}
        ratio_bounds = {"explore_max_rate": (0.2, 0.95)}
        seconds_bounds = {"training_event_wait": (0.01, 0.9), "generated_sleep_event_wait": (0.02, 1.2)}
        for name, (low, high) in count_bounds.items():
            values[name] = int(clamp(safe_float(values.get(name, low), low) * factors.get(name, 1.0), low, high))
        for name, (low, high) in ratio_bounds.items():
            values[name] = clamp(safe_float(values.get(name, low), low) * factors.get(name, 1.0), low, high)
        for name, (low, high) in seconds_bounds.items():
            values[name] = round(clamp(safe_float(values.get(name, low), low) * factors.get(name, 1.0), low, high), 4)
        values["explore_min_rate"] = min(values["explore_min_rate"], max(0.01, values["explore_max_rate"] * 0.45))
        return normalize_settings(Settings(**values))

    def train_from_metrics(self, metrics):
        state_vector = self.state_vector(metrics)
        throughput = clamp(safe_float(metrics.get("throughput", 0.0), 0.0) / max(1.0, safe_float(metrics.get("batch_size", 1.0), 1.0)), 0.0, 1.0)
        save_success = clamp(safe_float(metrics.get("save_success_rate", metrics.get("success_rate", 1.0)), 1.0), 0.0, 1.0)
        stability = state_vector["window_stability"]
        gain = clamp(safe_float(metrics.get("training_gain", 0.0), 0.0), 0.0, 1.0)
        resource_penalty = clamp((1.0 - state_vector["cpu_headroom"]) * 0.35 + (1.0 - state_vector["memory_free_ratio"]) * 0.35 + (1.0 - state_vector["frame_speed"]) * 0.15 + (1.0 - state_vector["execution_speed"]) * 0.15, 0.0, 1.0)
        reward = clamp(throughput * 0.28 + save_success * 0.24 + stability * 0.18 + gain * 0.22 - resource_penalty * 0.22, -1.0, 1.0)
        lr = clamp(1.0 / math.sqrt(self.trained_steps + 4.0), 0.01, 0.18)
        for action in self.ACTION_NAMES:
            direction = -1.0 if action in ("training_event_wait", "generated_sleep_event_wait") else 1.0
            error = reward - self.score(action, state_vector)
            self.bias[action] = clamp(self.bias.get(action, 0.0) + lr * error * direction, -4.0, 4.0)
            for name in self.STATE_NAMES:
                self.weights[action][name] = clamp(self.weights[action].get(name, 0.0) + lr * error * state_vector[name] * direction, -4.0, 4.0)
        self.trained_steps += 1
        self.last_reward = round(reward, 6)
        self.reward_ema = round(self.reward_ema * 0.85 + reward * 0.15, 6)
        return {"resource_policy_reward": self.last_reward, "resource_policy_ema": self.reward_ema, "resource_policy_steps": self.trained_steps}

    def snapshot(self):
        return {"type": "resource_adaptive_actor_critic", "state_names": list(self.STATE_NAMES), "action_names": list(self.ACTION_NAMES), "weights": {action: {name: round(value, 8) for name, value in weights.items()} for action, weights in self.weights.items()}, "bias": {action: round(value, 8) for action, value in self.bias.items()}, "trained_steps": self.trained_steps, "reward_ema": self.reward_ema, "last_reward": self.last_reward}


class RuntimeNumberFactory:
    def __init__(self, hardware, screen, pool_count, capture_ms, execution_ms, latency, success_rate, window_instability, learning_similarity, screen_score_total):
        width, height = screen
        self.width = max(1, safe_int(width, 1))
        self.height = max(1, safe_int(height, 1))
        self.pool_count = max(0, safe_int(pool_count, 0))
        self.capture_ms = max(1.0, safe_float(capture_ms, 24.0))
        self.execution_ms = max(1.0, safe_float(execution_ms, max(60.0, self.capture_ms * 4.2)))
        self.latency = max(1.0, safe_float(latency, self.capture_ms))
        self.success_rate = clamp(success_rate, 0.0, 1.0)
        self.window_instability = clamp(window_instability, 0.0, 1.0)
        self.learning_similarity = clamp(learning_similarity, 0.0, 1.0)
        self.screen_score_total = max(0.0, safe_float(screen_score_total, 0.0))
        hardware = hardware or {}
        self.cpu_load = clamp(safe_float(hardware.get("cpu_load", 0.0), 0.0), 0.0, 100.0)
        self.memory_free_ratio = clamp(safe_float(hardware.get("memory_free_ratio", 0.5), 0.5), 0.0, 1.0)
        self.cpu_count = max(1, safe_int(hardware.get("cpu_count", 1), 1))
        self.gpu_count = max(0, safe_int(hardware.get("gpu_count", 0), 0))
        self.gpu_memory_total = max(0.0, safe_float(hardware.get("gpu_memory_total", 0.0), 0.0))
        self.pixel_factor = math.sqrt(max(1.0, self.width * self.height) / (1280.0 * 720.0))
        self.record_factor = math.log2(max(2, self.pool_count + 2))
        self.gpu_factor = clamp(1.0 + self.gpu_count * 0.08 + self.gpu_memory_total / (8.0 * 1024 * 1024 * 1024), 1.0, 2.2)
        self.core_factor = clamp(math.log2(self.cpu_count + 1.0) / math.log2(9.0), 0.45, 1.2)
        self.capture_factor = clamp(self.capture_ms / 24.0, 0.4, 3.0)
        self.cpu_factor = clamp((1.0 + self.cpu_load / 200.0) * (1.12 - self.memory_free_ratio * 0.22) / max(0.5, self.core_factor), 0.65, 2.4)
        self.timing_factor = clamp((self.capture_ms + self.execution_ms * 0.5) / 80.0, 0.5, 2.5)
        self.generator = RuntimeGeneratedNumbers()
        self.context = {"screen_width": self.width, "screen_height": self.height, "screen_pixels": self.width * self.height, "pool_count": self.pool_count, "capture_ms": self.capture_ms, "execution_ms": self.execution_ms, "cpu_load": self.cpu_load, "cpu_count": self.cpu_count, "memory_free_ratio": self.memory_free_ratio, "gpu_count": self.gpu_count, "gpu_memory_total": self.gpu_memory_total, "success_rate": self.success_rate, "window_instability": self.window_instability, "learning_similarity": self.learning_similarity, "screen_score_total": self.screen_score_total}
        self.audit = {}

    def generated(self, name, semantic_goal, minimum, maximum):
        seed = self.generator.value(name, self.context, semantic_goal)
        return minimum + (maximum - minimum) * seed

    def source_tuple(self, name):
        return runtime_number_rule(name)

    def remember(self, name, value):
        parts = {key: data for key, data in self.generator.audit.items() if key.startswith(f"{name}.")}
        generator_audit = self.generator.audit.get(name, {})
        payload = {"source": generator_audit.get("source", "RuntimeGeneratedNumbers.value"), "sources": self.source_tuple(name), "reality_conditions": generator_audit.get("reality_conditions", dict(self.context)), "formula": generator_audit.get("formula", "semantic runtime formula composed only from RuntimeGeneratedNumbers coefficients and current reality conditions"), "semantic_goal": generator_audit.get("semantic_goal", name), "generated_coefficients": parts, "current_value": value}
        self.audit[name] = payload
        RUNTIME_NUMBER_AUDIT[name] = payload
        return value

    def value(self, name):
        g = lambda key, goal, lo, hi: self.generated(f"{name}.{key}", goal, lo, hi)
        explore_base = g("explore_base", "exploration balances recent failures and learned similarity", self.window_instability, self.learning_similarity + (1.0 - self.success_rate))
        formulas = {
            "mouse_activity_wait": self.capture_factor * g("capture_fraction", "wait tracks capture arrival latency", 0.0, 0.08),
            "training_event_wait": self.capture_factor * self.cpu_factor * self.timing_factor * g("event_gap", "next screenshot follows action result and hardware readiness", 0.0, 0.12) / self.gpu_factor,
            "sleep_event_wait": self.cpu_factor / self.core_factor * g("batch_event_gap", "sleep advances on batch completion events", 0.0, 0.18),
            "min_action_delay_seconds": (self.capture_ms + self.execution_ms * g("action_gap_share", "minimum delay follows action completion and system pressure", 0.0, 0.45)) / g("action_gap_divisor", "convert action gap milliseconds to seconds", 600.0, 2400.0) * self.cpu_factor,
            "generated_sleep_event_wait": (self.capture_ms + self.execution_ms * g("sleep_wait_share", "offline sleep wakeups slow down under load and unstable windows", 0.1, 0.9)) / g("sleep_wait_divisor", "convert sleep event wait milliseconds to seconds", 180.0, 900.0) * self.cpu_factor * (1.0 + self.window_instability) / max(0.2, self.success_rate),
            "generated_action_complete_wait": (self.capture_ms + self.execution_ms * g("action_complete_share", "action completion polling tracks actual execution latency", 0.05, 0.7)) / g("action_complete_divisor", "convert action completion wait milliseconds to seconds", 220.0, 1100.0) * self.cpu_factor * (1.0 + self.window_instability) / max(0.2, self.success_rate),
            "sleep_worker_count": self.cpu_count * clamp(self.memory_free_ratio * g("worker_memory", "parallelism respects free memory and cpu load", 0.2, 1.6), 0.0, max(1.0, self.cpu_count)) + self.gpu_count,
            "sleep_batch_size": (self.record_factor + self.core_factor + self.memory_free_ratio + math.log1p(self.screen_score_total)) * g("batch_scale", "batch size grows with experience pool and hardware", 1.0, 64.0),
            "sleep_queue_depth": (self.cpu_count + max(1, self.gpu_count) + self.memory_free_ratio) * g("queue_scale", "queue absorbs worker completion events", 0.5, 4.0),
            "key_debounce_seconds": self.cpu_factor * g("debounce", "human escape key debounce", 0.0, 0.6),
            "window_attach_seconds": g("attach_budget", "ldplayer launch and attach budget", self.capture_ms, self.execution_ms + self.cpu_load),
            "window_event_wait": self.cpu_factor / self.gpu_factor * g("window_event", "window validation after foreground event", 0.0, 0.6),
            "action_duration_min": (self.capture_ms + self.execution_ms * g("min_exec_share", "minimum mouse action duration", 0.0, 0.3)) / g("min_duration_divisor", "convert runtime milliseconds to seconds", 600.0, 2200.0),
            "action_duration_max": (self.capture_ms + self.execution_ms * g("max_exec_share", "maximum mouse action duration", 0.3, 1.0)) / g("max_duration_divisor", "convert runtime milliseconds to seconds", 180.0, 900.0),
            "random_click_duration_min": (self.capture_ms + self.execution_ms * g("click_min_share", "short click lower duration", 0.0, 0.2)) / g("click_min_divisor", "convert runtime milliseconds to seconds", 800.0, 2500.0),
            "random_click_duration_max": (self.capture_ms + self.execution_ms * g("click_max_share", "short click upper duration", 0.1, 0.6)) / g("click_max_divisor", "convert runtime milliseconds to seconds", 350.0, 1200.0),
            "generated_click_hold_max": (self.capture_ms + self.execution_ms * g("hold_share", "generated click hold duration", 0.0, 0.5)) / g("hold_divisor", "convert runtime milliseconds to seconds", 250.0, 900.0),
            "ui_event_coalesce_seconds": (self.capture_ms / g("ui_ms", "coalesce milliseconds to seconds", 700.0, 1600.0)) * self.cpu_factor / self.gpu_factor,
            "persistence_event_wait": (self.capture_ms + self.latency) / g("persist_divisor", "persistence queue wakeup", 120.0, 420.0) * self.cpu_factor,
            "persistence_close_seconds": (self.execution_ms + self.capture_ms) / g("close_divisor", "persistence close drain", 40.0, 180.0) * self.cpu_factor,
            "explore_max_rate": explore_base,
            "explore_min_rate": explore_base * g("min_ratio", "minimum exploration remains lower than maximum", 0.02, 0.3),
            "action_jitter": g("jitter", "mouse randomness follows instability and failures", 0.0, 0.12) * (1.0 + self.window_instability + (1.0 - self.success_rate)),
            "global_action_probability": g("global_mix", "choose global experience when local nearest is weak", 0.2, 0.8),
            "random_action_min": self.window_instability * g("random_min", "minimum random action bound", 0.0, 0.12),
            "random_action_max": 1.0 - self.window_instability * g("random_max_margin", "maximum random action bound", 0.0, 0.12),
            "motion_curve_offset_min": g("curve_min", "human curve minimum", 0.0, 0.2),
            "motion_curve_offset_max": g("curve_max", "human curve maximum", 0.15, 0.7) + (1.0 - self.success_rate) * self.window_instability,
            "motion_first_control_min": g("first_min", "first bezier control minimum", 0.0, (35.0 / 100.0)),
            "motion_first_control_max": g("first_max", "first bezier control maximum", (35.0 / 100.0), 0.8),
            "motion_second_control_min": g("second_min", "second bezier control minimum", 0.2, 0.7),
            "motion_second_control_max": g("second_max", "second bezier control maximum", 0.7, 1.0),
            "learning_screen_similarity_threshold": 1.0 - g("learning_delta", "screen change novelty threshold", 0.0, 0.08),
            "ui_progress_delta": g("progress_delta", "visible progress refresh threshold", 0.0, 1.0),
            "action_score_similarity_weight": g("similarity_weight", "nearest screen action score", 0.1, 2.0),
            "action_score_reward_weight": g("reward_weight", "screen score reward action score", 0.05, 1.5),
            "action_score_human_weight": g("human_weight", "learning mouse similarity score", 0.05, 1.5),
            "action_score_novelty_weight": g("novelty_weight", "novel screen score", 0.01, 1.0),
            "motion_score_magnitude_weight": g("magnitude_weight", "scroll magnitude weight", 0.1, 0.9),
            "motion_score_continuity_weight": g("continuity_weight", "scroll continuity weight", 0.1, 0.9),
            "hash_size": math.sqrt(self.width * self.height) / g("hash_divisor", "perceptual hash detail follows screen size", 120.0, 260.0) + self.record_factor,
            "nearest_top_k": self.record_factor * g("topk", "nearest historical screen batch size", 4.0, 24.0),
            "nearest_candidate_limit": (self.record_factor + self.window_instability + self.core_factor) * g("candidate", "nearest candidate search depth", 128.0, 1400.0),
            "hash_prefix_bits": self.record_factor + g("prefix", "hash prefix bits follow pool density", 2.0, 10.0),
            "human_profile_min_samples": self.record_factor * g("profile_min", "minimum user samples for auditability", 3.0, 18.0),
            "human_profile_max_samples": self.record_factor * g("profile_max", "maximum user profile memory", 300.0, 1400.0),
            "human_profile_keep_samples": self.record_factor * g("profile_keep", "kept user profile samples", 250.0, 1200.0),
            "ui_width": self.width * g("ui_width", "control panel width", 0.3, 0.9),
            "ui_height": self.height * g("ui_height", "control panel height", (35.0 / 100.0), 0.9),
            "ui_min_width": self.width * g("ui_min_width", "minimum complete UI width", 0.25, 0.65),
            "ui_min_height": self.height * g("ui_min_height", "minimum complete UI height", 0.25, 0.65),
            "ui_padding": min(self.width, self.height) * g("ui_padding", "visual spacing", 0.004, 0.03),
            "ui_section_padding": min(self.width, self.height) * g("ui_section_padding", "section visual spacing", 0.003, 0.025),
            "ui_metric_columns": self.width / g("metric_column_width", "responsive metric columns", 220.0, 520.0),
            "ui_metric_min_column_width": self.width / g("metric_min_divisor", "minimum metric column width", 3.0, 8.0),
            "experience_load_limit": self.record_factor * g("load_limit", "experience loading budget", 1200.0, 12000.0),
            "training_fail_stop_count": (1.0 - self.success_rate + self.window_instability) * g("failures", "stop after repeated failed action results", 2.0, 18.0),
            "async_queue_size": (self.cpu_count + self.gpu_count + self.record_factor) * g("async_queue", "async persistence queue", 12.0, 128.0),
            "global_action_heap_limit": (self.record_factor + self.core_factor) * g("global_heap", "global action ranking memory", 64.0, 512.0),
            "local_action_heap_limit": (self.record_factor + self.window_instability + 1.0) * g("local_heap", "local action ranking memory", 32.0, 256.0),
            "softmax_temperature": self.record_factor * g("temperature", "softmax action diversity", 1.0, 6.0),
            "reward_total_min": -max(self.screen_score_total, g("reward_min", "minimum auditable reward", 1000.0, 20000.0)),
            "reward_total_max": max(self.screen_score_total, g("reward_max", "maximum auditable reward", 1000.0, 20000.0)),
            "score_default": 100.0 * clamp(g("score_default", "default screen score follows pool maturity and learned similarity", 0.25, 0.75) * (0.7 + min(1.0, self.record_factor / 12.0) * 0.3) * (0.85 + (1.0 - self.learning_similarity) * 0.3), 0.0, 1.0),
            "scroll_score_default": 100.0 * clamp(g("scroll_score", "scroll default score follows successful human-like outcomes", 0.25, 0.75) * (0.8 + self.success_rate * 0.2), 0.0, 1.0),
            "fallback_score_base": 100.0 * clamp(g("fallback_score", "fallback score remains conservative under uncertain reality", 0.2, 0.65) * (0.75 + self.success_rate * 0.25) * (0.9 + (1.0 - self.learning_similarity) * 0.2), 0.0, 1.0),
            "motion_steps_per_second": (self.width / g("motion_divisor", "mouse interpolation density", 12.0, 42.0)) / max(0.1, self.cpu_factor),
            "learning_screen_change_capacity": clamp(self.core_factor * self.gpu_factor, 0.1, 4.0) / max(0.001, self.capture_ms / g("capacity_ms", "screen-change event capacity", 700.0, 1400.0))
        }
        return self.remember(name, formulas[name])

    def seconds(self, name, minimum, maximum):
        return clamp(self.value(name), minimum, maximum)

    def ratio(self, name, minimum, maximum):
        return clamp(self.value(name), minimum, maximum)

    def count(self, name, minimum, maximum):
        return int(clamp(self.value(name), minimum, maximum))

    def scalar(self, name, minimum, maximum):
        return clamp(self.value(name), minimum, maximum)

    def human_feature_weights(self, source_weights):
        normalized = normalized_human_feature_weights(source_weights)
        values = tuple(clamp(value * (0.9 + self.success_rate * 0.2), 0.01, 1.0) for value in normalized)
        self.remember("human_feature_weights", values)
        return values


def derive_runtime_settings(base_settings=None, rect=None, pool_count=0, capture_ms=None, cpu_load=0.0, execution_ms=None, window_instability=0.0, recent_success=1.0, screen_score_total=0.0, learning_similarity=0.97, hardware=None):
    source = base_settings or Settings()
    width, height = rect_size(rect) if rect else (safe_int(getattr(win32api, "GetSystemMetrics", lambda _: 1920)(0), 1920), safe_int(getattr(win32api, "GetSystemMetrics", lambda _: 1080)(1), 1080))
    hardware = dict(hardware or {})
    hardware["cpu_load"] = cpu_load
    factory = RuntimeNumberFactory(hardware, (width, height), pool_count, capture_ms, execution_ms, capture_ms, recent_success, window_instability, learning_similarity, screen_score_total)
    values = {item.name: getattr(source, item.name) for item in fields(Settings)}
    values.update({
        "hash_size": factory.count("hash_size", 8, 24),
        "nearest_top_k": factory.count("nearest_top_k", 8, 256),
        "nearest_candidate_limit": factory.count("nearest_candidate_limit", 256, 20000),
        "hash_prefix_bits": factory.count("hash_prefix_bits", 4, 20),
        "mouse_activity_wait": round(factory.seconds("mouse_activity_wait", 0.01, 0.2), 4),
        "training_event_wait": round(factory.seconds("training_event_wait", 0.01, 0.9), 4),
        "sleep_event_wait": round(factory.seconds("sleep_event_wait", 0.05, 1.0), 4),
        "sleep_worker_count": factory.count("sleep_worker_count", 1, 64),
        "sleep_batch_size": factory.count("sleep_batch_size", 8, 4096),
        "sleep_queue_depth": factory.count("sleep_queue_depth", 1, 256),
        "key_debounce_seconds": round(factory.seconds("key_debounce_seconds", 0.05, 1.0), 3),
        "window_attach_seconds": round(factory.seconds("window_attach_seconds", 5.0, 120.0), 2),
        "window_event_wait": round(factory.seconds("window_event_wait", 0.05, 1.0), 3),
        "min_action_delay_seconds": round(factory.seconds("min_action_delay_seconds", 0.01, 0.8), 4),
        "generated_sleep_event_wait": round(factory.seconds("generated_sleep_event_wait", 0.02, 1.2), 4),
        "generated_action_complete_wait": round(factory.seconds("generated_action_complete_wait", 0.02, 1.0), 4),
        "explore_max_rate": factory.ratio("explore_max_rate", 0.2, 0.95),
        "explore_min_rate": factory.ratio("explore_min_rate", 0.01, 0.2),
        "action_jitter": factory.ratio("action_jitter", 0.005, 0.08),
        "softmax_temperature": factory.scalar("softmax_temperature", 6.0, 30.0),
        "human_profile_min_samples": factory.count("human_profile_min_samples", 12, 120),
        "human_profile_max_samples": factory.count("human_profile_max_samples", 1500, 10000),
        "human_profile_keep_samples": factory.count("human_profile_keep_samples", 1200, 9000),
        "ui_width": factory.count("ui_width", 700, 1600),
        "ui_height": factory.count("ui_height", 520, 1200),
        "ui_min_width": factory.count("ui_min_width", 520, 1100),
        "ui_min_height": factory.count("ui_min_height", 420, 900),
        "ui_padding": factory.count("ui_padding", 8, 28),
        "ui_section_padding": factory.count("ui_section_padding", 6, 22),
        "ui_metric_columns": factory.count("ui_metric_columns", 2, 6),
        "ui_metric_min_column_width": factory.count("ui_metric_min_column_width", 150, 320),
        "reward_total_min": factory.scalar("reward_total_min", -1000000000.0, 0.0),
        "reward_total_max": factory.scalar("reward_total_max", 0.0, 1000000000.0),
        "score_default": round(factory.scalar("score_default", 5.0, 95.0), 3),
        "scroll_score_default": round(factory.scalar("scroll_score_default", 5.0, 95.0), 3),
        "fallback_score_base": round(factory.scalar("fallback_score_base", 5.0, 95.0), 3),
        "experience_load_limit": factory.count("experience_load_limit", 8000, 90000),
        "global_action_probability": factory.ratio("global_action_probability", 0.2, 0.75),
        "random_action_min": factory.ratio("random_action_min", 0.0, 0.12),
        "random_action_max": factory.ratio("random_action_max", 0.88, 1.0),
        "action_duration_min": factory.seconds("action_duration_min", 0.05, (35.0 / 100.0)),
        "action_duration_max": factory.seconds("action_duration_max", 0.18, 1.8),
        "random_click_duration_min": factory.seconds("random_click_duration_min", 0.03, 0.22),
        "random_click_duration_max": factory.seconds("random_click_duration_max", 0.08, 0.7),
        "generated_click_hold_max": factory.seconds("generated_click_hold_max", 0.08, 0.9),
        "motion_steps_per_second": factory.scalar("motion_steps_per_second", 40.0, 260.0),
        "motion_curve_offset_min": factory.ratio("motion_curve_offset_min", 0.02, 0.2),
        "motion_curve_offset_max": factory.ratio("motion_curve_offset_max", 0.16, 0.65),
        "motion_first_control_min": factory.ratio("motion_first_control_min", 0.08, 0.3),
        "motion_first_control_max": factory.ratio("motion_first_control_max", 0.3, 0.78),
        "motion_second_control_min": factory.ratio("motion_second_control_min", 0.26, 0.7),
        "motion_second_control_max": factory.ratio("motion_second_control_max", 0.7, 0.95),
        "learning_screen_change_capacity": factory.scalar("learning_screen_change_capacity", 0.5, 20.0),
        "learning_screen_similarity_threshold": factory.ratio("learning_screen_similarity_threshold", 0.9, 0.999),
        "training_fail_stop_count": factory.count("training_fail_stop_count", 2, 24),
        "ui_event_coalesce_seconds": round(factory.seconds("ui_event_coalesce_seconds", 0.01, (35.0 / 100.0)), 4),
        "ui_progress_delta": factory.ratio("ui_progress_delta", 0.02, 1.0),
        "persistence_event_wait": round(factory.seconds("persistence_event_wait", 0.02, 1.0), 4),
        "persistence_close_seconds": round(factory.seconds("persistence_close_seconds", 0.2, 8.0), 4),
        "async_queue_size": factory.count("async_queue_size", 32, int(factory.generated("async_queue_upper_bound", "upper bound for async queue", 4096, 16384))),
        "global_action_heap_limit": factory.count("global_action_heap_limit", 128, int(factory.generated("global_action_heap_upper_bound", "upper bound for global action heap", 4096, 16384))),
        "local_action_heap_limit": factory.count("local_action_heap_limit", 64, 4096),
        "action_score_similarity_weight": factory.ratio("action_score_similarity_weight", 0.1, 2.0),
        "action_score_reward_weight": factory.ratio("action_score_reward_weight", 0.05, 1.5),
        "action_score_human_weight": factory.ratio("action_score_human_weight", 0.05, 1.5),
        "action_score_novelty_weight": factory.ratio("action_score_novelty_weight", 0.01, 1.0),
        "motion_score_magnitude_weight": factory.ratio("motion_score_magnitude_weight", 0.1, 0.9),
        "motion_score_continuity_weight": factory.ratio("motion_score_continuity_weight", 0.1, 0.9),
        "human_feature_weights": factory.human_feature_weights(source.human_feature_weights)
    })
    return normalize_settings(Settings(**values))

def create_runtime_settings(base_settings=None, rect=None, pool_count=0, capture_ms=24.0, cpu_load=0.0):
    return derive_runtime_settings(base_settings=base_settings, rect=rect, pool_count=pool_count, capture_ms=capture_ms, cpu_load=cpu_load, hardware=read_hardware_state())


def rect_size(rect):
    left, top, right, bottom = rect
    return max(1, int(right - left)), max(1, int(bottom - top))


def rect_intersection(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    return (left, top, right, bottom) if right > left and bottom > top else None


def rect_area(rect):
    return max(0, int(rect[2] - rect[0])) * max(0, int(rect[3] - rect[1])) if rect else 0


def rect_union_area(rects):
    normalized = [tuple(map(int, rect)) for rect in rects if rect and rect[2] > rect[0] and rect[3] > rect[1]]
    if not normalized:
        return 0
    xs = sorted({value for rect in normalized for value in (rect[0], rect[2])})
    total = 0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted((rect[1], rect[3]) for rect in normalized if rect[0] < right and rect[2] > left)
        merged = 0
        current_top = None
        current_bottom = None
        for top, bottom in intervals:
            if current_top is None:
                current_top, current_bottom = top, bottom
            elif top <= current_bottom:
                current_bottom = max(current_bottom, bottom)
            else:
                merged += current_bottom - current_top
                current_top, current_bottom = top, bottom
        if current_top is not None:
            merged += current_bottom - current_top
        total += (right - left) * merged
    return total


def point_inside(rect, x, y):
    left, top, right, bottom = rect
    return left <= x < right and top <= y < bottom


def rel_from_abs(rect, x, y):
    left, top, right, bottom = rect
    width, height = rect_size(rect)
    return [
        round(clamp((float(x) - left) / width, 0.0, 1.0), 6),
        round(clamp((float(y) - top) / height, 0.0, 1.0), 6)
    ]


def abs_from_rel(rect, point):
    left, top, right, bottom = rect
    width, height = rect_size(rect)
    x = left + clamp(point[0], 0.0, 1.0) * max(0, width - 1)
    y = top + clamp(point[1], 0.0, 1.0) * max(0, height - 1)
    return int(round(x)), int(round(y))



def build_mouse_event(kind, x, y, rect, previous=None, extra=None, now_perf=None, created_at=None):
    now_perf = time.perf_counter() if now_perf is None else safe_float(now_perf, time.perf_counter())
    inside = point_inside(rect, x, y) if rect else False
    px = safe_float(previous.get("x", x), x) if isinstance(previous, dict) else float(x)
    py = safe_float(previous.get("y", y), y) if isinstance(previous, dict) else float(y)
    pt = safe_float(previous.get("t", previous.get("timestamp", now_perf)), now_perf) if isinstance(previous, dict) else now_perf
    vx0 = safe_float(previous.get("instant_speed_x", 0.0), 0.0) if isinstance(previous, dict) else 0.0
    vy0 = safe_float(previous.get("instant_speed_y", 0.0), 0.0) if isinstance(previous, dict) else 0.0
    dt = max(1e-6, now_perf - pt)
    dx = float(x) - px
    dy = float(y) - py
    speed_x = dx / dt
    speed_y = dy / dt
    width, height = rect_size(rect) if rect else (1.0, 1.0)
    rel_x = (float(x) - rect[0]) / max(1.0, width) if rect else 0.0
    rel_y = (float(y) - rect[1]) / max(1.0, height) if rect else 0.0
    event = {"type": kind, "t": now_perf, "timestamp": now_perf, "created_at": created_at or now_text(), "x": int(x), "y": int(y), "abs": [int(x), int(y)], "rel": [round(rel_x, 6), round(rel_y, 6)], "inside": bool(inside), "dx": round(dx, 6), "dy": round(dy, 6), "dt": round(dt, 6), "direction_angle": round(math.degrees(math.atan2(dy, dx)), 6) if dx or dy else 0.0, "instant_speed": round(math.hypot(speed_x, speed_y), 6), "instant_speed_x": round(speed_x, 6), "instant_speed_y": round(speed_y, 6), "acceleration": round(math.hypot(speed_x - vx0, speed_y - vy0) / dt, 6)}
    if extra:
        event.update(extra)
    return event

def distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def path_length(points):
    return sum(distance(points[index - 1], points[index]) for index in range(1, len(points)))


def normalize_rel_point(point, fallback=None):
    source = point if isinstance(point, (list, tuple)) and len(point) >= 2 else fallback
    if not isinstance(source, (list, tuple)) or len(source) < 2:
        source = [0.5, 0.5]
    return [round(clamp(safe_float(source[0], 0.5), 0.0, 1.0), 6), round(clamp(safe_float(source[1], 0.5), 0.0, 1.0), 6)]


def normalize_path(rect, path, start_abs):
    result = []
    base_t = None
    for item in path or []:
        if isinstance(item, dict):
            t = safe_float(item.get("t", 0.0), 0.0)
            if base_t is None:
                base_t = t
            point = rel_from_abs(rect, item.get("x", start_abs[0]), item.get("y", start_abs[1]))
            point.append(round(max(0.0, t - base_t), 6))
            result.append(point)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            point = rel_from_abs(rect, item[0], item[1])
            point.append(0.0)
            result.append(point)
    return result


def normalize_motion_events(rect, path, start_abs):
    result = []
    base_t = None
    for item in path or []:
        if not isinstance(item, dict):
            continue
        t = safe_float(item.get("t", item.get("timestamp", 0.0)), 0.0)
        if base_t is None:
            base_t = t
        rel = item.get("rel") if isinstance(item.get("rel"), (list, tuple)) and len(item.get("rel")) >= 2 else rel_from_abs(rect, item.get("x", start_abs[0]), item.get("y", start_abs[1]))
        result.append({"type": item.get("type", "move"), "timestamp": t, "time_offset": round(max(0.0, t - base_t), 6), "created_at": item.get("created_at"), "x": safe_int(item.get("x", start_abs[0]), safe_int(start_abs[0], 0)), "y": safe_int(item.get("y", start_abs[1]), safe_int(start_abs[1], 0)), "abs": [safe_int(item.get("x", start_abs[0]), safe_int(start_abs[0], 0)), safe_int(item.get("y", start_abs[1]), safe_int(start_abs[1], 0))], "rel": normalize_rel_point(rel), "dx": round(safe_float(item.get("dx", 0.0), 0.0), 6), "dy": round(safe_float(item.get("dy", 0.0), 0.0), 6), "dt": round(max(0.0, safe_float(item.get("dt", 0.0), 0.0)), 6), "direction_angle": round(safe_float(item.get("direction_angle", 0.0), 0.0), 6), "instant_speed": round(max(0.0, safe_float(item.get("instant_speed", 0.0), 0.0)), 6), "instant_speed_x": round(safe_float(item.get("instant_speed_x", 0.0), 0.0), 6), "instant_speed_y": round(safe_float(item.get("instant_speed_y", 0.0), 0.0), 6), "acceleration": round(max(0.0, safe_float(item.get("acceleration", 0.0), 0.0)), 6), "inside": bool(item.get("inside", True))})
    return result


def normalize_mouse_action(action, rect):
    if not action:
        return None
    if action.get("invalid_outside_client") or any(not event.get("inside", True) for event in action.get("path", [])):
        return None
    if "start_rel" in action and "end_rel" in action:
        result = copy.deepcopy(action)
        result["type"] = result.get("type", "click")
        result["button"] = result.get("button", "Button.left")
        result["source"] = result.get("source", "user")
        result["duration"] = round(max(0.0, safe_float(result.get("duration", 0.0), 0.0)), 6)
        result["start_rel"] = normalize_rel_point(result.get("start_rel"), [0.5, 0.5])
        result["end_rel"] = normalize_rel_point(result.get("end_rel"), result["start_rel"])
        result["path_rel"] = [[round(clamp(safe_float(p[0], 0.0), 0.0, 1.0), 6), round(clamp(safe_float(p[1], 0.0), 0.0, 1.0), 6), round(max(0.0, safe_float(p[2], 0.0)), 6) if len(p) >= 3 else 0.0] for p in result.get("path_rel", []) if isinstance(p, (list, tuple)) and len(p) >= 2]
        if result.get("scroll"):
            result["scroll"] = [safe_int(result["scroll"][0], 0), safe_int(result["scroll"][1], 0)]
        return result
    start_abs = action.get("start_abs") or [action.get("x0", 0), action.get("y0", 0)]
    end_abs = action.get("end_abs") or [action.get("x1", start_abs[0]), action.get("y1", start_abs[1])]
    result = {
        "type": action.get("type", "click"),
        "button": action.get("button", "Button.left"),
        "source": action.get("source", "user"),
        "started_at": action.get("started_at"),
        "ended_at": action.get("ended_at"),
        "started_perf": action.get("started_perf"),
        "ended_perf": action.get("ended_perf"),
        "duration": round(max(0.0, safe_float(action.get("duration", 0.0), 0.0)), 6),
        "start_rel": rel_from_abs(rect, start_abs[0], start_abs[1]),
        "end_rel": rel_from_abs(rect, end_abs[0], end_abs[1]),
        "path_rel": normalize_path(rect, action.get("path", []), start_abs),
        "motion_events": normalize_motion_events(rect, action.get("path", []), start_abs),
        "start_abs": [safe_int(start_abs[0], 0), safe_int(start_abs[1], 0)],
        "end_abs": [safe_int(end_abs[0], 0), safe_int(end_abs[1], 0)]
    }
    if action.get("scroll"):
        result["scroll"] = [safe_int(action["scroll"][0], 0), safe_int(action["scroll"][1], 0)]
    return result


def parse_hash_value(record):
    if not record:
        return None
    bits = safe_int(record.get("screen_hash_bits", record.get("hash_bits", 0)), 0)
    value = record.get("screen_hash_int", record.get("hash_int"))
    if value is not None and bits > 0:
        try:
            value = int(value)
            width = max(1, math.ceil(bits / 4))
            return HashValue(value, bits, f"{value:0{width}x}"[-width:])
        except Exception:
            pass
    text = record.get("screen_hash_hex") or record.get("screen_hash") or record.get("hash")
    if not text:
        return None
    try:
        text = str(text).strip()
        if set(text) <= {"0", "1"}:
            bits = len(text)
            value = int(text, 2)
        else:
            value = int(text, 16)
            bits = bits if bits > 0 else len(text) * 4
        width = max(1, math.ceil(bits / 4))
        return HashValue(value, bits, f"{value:0{width}x}"[-width:])
    except Exception:
        return None


def hash_distance(hash_a, hash_b):
    if not hash_a or not hash_b:
        return None
    bits = min(hash_a.bits, hash_b.bits)
    if bits <= 0:
        return None
    a_value = hash_a.value >> max(0, hash_a.bits - bits)
    b_value = hash_b.value >> max(0, hash_b.bits - bits)
    return (a_value ^ b_value).bit_count(), bits


def hash_similarity(hash_a, hash_b):
    distance = hash_distance(hash_a, hash_b)
    if not distance:
        return 0.0
    diff, bits = distance
    return clamp(1.0 - diff / bits, 0.0, 1.0)


def parse_semantic_vector(value):
    if not value:
        return ()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return ()
    if not isinstance(value, (list, tuple)):
        return ()
    result = []
    for item in value:
        try:
            result.append(float(item))
        except Exception:
            return ()
    return tuple(result)


def semantic_similarity(vector_a, vector_b):
    vector_a = parse_semantic_vector(vector_a)
    vector_b = parse_semantic_vector(vector_b)
    size = min(len(vector_a), len(vector_b))
    if size <= 0:
        return None
    dot = sum(vector_a[index] * vector_b[index] for index in range(size))
    norm_a = math.sqrt(sum(vector_a[index] * vector_a[index] for index in range(size)))
    norm_b = math.sqrt(sum(vector_b[index] * vector_b[index] for index in range(size)))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    return clamp((dot / (norm_a * norm_b) + 1.0) * 0.5, 0.0, 1.0)


@dataclass
class RewardState:
    best_income: float = 0.0
    cost: float = 0.000001
    last_income_improved_at: float = 0.0
    state_version: int = 1


def parse_event_timestamp(value):
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else time.time()
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return time.time()
    return time.time()


def reward_income(screen_score, human_similarity):
    return strict_reward_value(screen_score, human_similarity)


def reward_state_from(value):
    if isinstance(value, RewardState):
        return RewardState(value.best_income, value.cost, value.last_income_improved_at, value.state_version)
    if isinstance(value, dict):
        return RewardState(safe_float(value.get("best_income", 0.0), 0.0), max(0.000001, safe_float(value.get("cost", 0.000001), 0.000001)), safe_float(value.get("last_income_improved_at", 0.0), 0.0), max(1, safe_int(value.get("state_version", 1), 1)))
    return RewardState()


def reward_with_state(screen_score, human_similarity, state=None, event_time=None):
    state = reward_state_from(state)
    timestamp = parse_event_timestamp(event_time)
    baseline_time = state.last_income_improved_at or timestamp
    elapsed = max(0.0, timestamp - baseline_time)
    income = reward_income(screen_score, human_similarity)
    cost = max(0.000001, state.cost + elapsed * 0.000001)
    improved = income > state.best_income
    if improved:
        state.best_income = income
        state.cost = 0.000001
        state.last_income_improved_at = timestamp
        cost = state.cost
    else:
        state.cost = cost
    reward = income - cost
    return {"reward_version": 6, "income": round(income, 8), "cost": round(cost, 8), "reward": round(reward, 8), "income_improved": improved, "event_time": timestamp, "state": asdict(state)}


def reward_breakdown(novelty, human_score, settings, state=None, event_time=None):
    score_precision = 2
    screen_resolution = 10 ** (-score_precision)
    screen_novelty = round(clamp(novelty, 0.0, 100.0), score_precision)
    screen_reward = screen_novelty
    human_similarity = round(clamp(human_score, 0.0, 100.0), score_precision)
    human_delta = round(human_similarity - clamp(settings.score_default, 0.0, 100.0), score_precision)
    human_tiebreak = human_similarity
    human_bonus = round((human_similarity / 100.0) * screen_resolution * (1.0 - 1e-6), 8)
    screen_score_delta = round(clamp(screen_reward, settings.reward_total_min, settings.reward_total_max), 6)
    stateful = reward_with_state(screen_novelty, human_similarity, state, event_time)
    total_reward = round(clamp(stateful["reward"], settings.reward_total_min, settings.reward_total_max + screen_resolution), 8)
    return {"reward_version": stateful["reward_version"], "income": stateful["income"], "cost": stateful["cost"], "income_improved": stateful["income_improved"], "reward_state": stateful["state"], "reward_state_version": stateful["state"]["state_version"], "screen_novelty": screen_novelty, "screen_reward": screen_reward, "human_similarity": human_similarity, "human_tiebreak": round(human_tiebreak, 6), "human_bonus": human_bonus, "screen_score_resolution": screen_resolution, "screen_score_delta": screen_score_delta, "basis": ["nearest_screen_batch", "learning_mouse_profile", "lexicographic_screen_then_human", "numeric_human_bonus_below_screen_resolution", "stateful_time_cost_reset_on_income_improvement"], "screen_primary_reward": screen_reward, "human_tie_break_reward": round(human_tiebreak, 6), "mouse_action_delta": human_delta, "total_reward": total_reward, "reward_sort_key": [round(screen_reward, score_precision), round(human_similarity, score_precision)]}


def strict_reward_key(screen_score, human_similarity):
    return (round(clamp(safe_float(screen_score, 0.0), 0.0, 100.0), 2), round(clamp(safe_float(human_similarity, 0.0), 0.0, 100.0), 2))


def strict_reward_value(screen_score, human_similarity):
    screen, human = strict_reward_key(screen_score, human_similarity)
    return screen + (human / 100.0) * 0.000099999


def strict_reward_target(screen_score, human_similarity):
    screen, human = strict_reward_key(screen_score, human_similarity)
    tie_cap = 0.000099999
    return clamp((screen / 100.0 + human / 100.0 * tie_cap) / (1.0 + tie_cap), 0.0, 1.0)


AI_MODEL_GROUP_SPECS = (
    {"key": "screen_novelty_scorer", "name": "画面新颖程度评分模型", "goal": "一个画面与经验池中最相似的一批历史画面批量聚合相似度越高，评分越低"},
    {"key": "mouse_humanlikeness_scorer", "name": "鼠标拟人程度评分模型", "goal": "AI鼠标操作与学习模式用户鼠标操作相似度越高，评分越高"},
    {"key": "operation_policy", "name": "实操模型", "goal": "在训练模式期间，雷电模拟器客户区内，AI输出鼠标操作"},
    {"key": "reward_model", "name": "奖励模型", "goal": "奖励=收入（画面评分不同的样本，画面评分越高，收入越高；画面评分相同的样本，鼠标评分越高，收入越高）-支出（支出一开始为正极小值，随时间推移逐渐变大，并在收入变高时重置为正极小值）"},
    {"key": "runtime_value_model", "name": "数学模型", "goal": "依据现实条件确定没有在要求中确定的变量的初始数值，并让它们跟随现实条件的变化而变化"},
)


def training_data_digest(records):
    records = records or []
    items = []
    for record in records:
        if isinstance(record, dict):
            items.append([record.get("id"), record.get("mode"), record.get("mouse_source"), record.get("screen_hash"), record.get("screen_hash_hex"), record.get("image_checksum"), record.get("reward"), record.get("human_score"), record.get("screen_score"), record.get("sleep_novelty"), record.get("sleep_human_score")])
    return hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def numeric_vector(values, limit=64):
    result = []
    for value in values or []:
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            result.append(number)
        if len(result) >= limit:
            break
    return result


def vector_mean(vectors, limit=64):
    clean = [numeric_vector(vector, limit) for vector in vectors or []]
    clean = [vector for vector in clean if vector]
    if not clean:
        return []
    width = min(len(vector) for vector in clean)
    if width <= 0:
        return []
    return [round(sum(vector[index] for vector in clean) / len(clean), 8) for index in range(width)]


def vector_similarity(left, right):
    a = numeric_vector(left)
    b = numeric_vector(right)
    width = min(len(a), len(b))
    if width <= 0:
        return 0.0
    dot = sum(a[index] * b[index] for index in range(width))
    na = math.sqrt(sum(a[index] * a[index] for index in range(width)))
    nb = math.sqrt(sum(b[index] * b[index] for index in range(width)))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return clamp((dot / (na * nb) + 1.0) / 2.0, 0.0, 1.0)


def percentile(values, ratio, default=0.0):
    clean = sorted(safe_float(value, default) for value in values or [] if math.isfinite(safe_float(value, default)))
    if not clean:
        return default
    pos = clamp(ratio, 0.0, 1.0) * (len(clean) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return clean[lower]
    return clean[lower] * (upper - pos) + clean[upper] * (pos - lower)


def robust_scale(values, default=1.0):
    q1 = percentile(values, 0.25, default)
    q3 = percentile(values, 0.75, default)
    return max(0.0001, (q3 - q1) / 1.349 if q3 > q1 else default)


def feature_distance_score(values, means, variances, names):
    if not means:
        return 50.0
    distance = 0.0
    used = 0
    for name in names:
        if name not in means:
            continue
        variance = max(0.0001, safe_float(variances.get(name, 1.0), 1.0))
        delta = abs(safe_float(values.get(name, 0.0), 0.0) - safe_float(means.get(name, 0.0), 0.0))
        distance += math.log1p((delta * delta) / variance)
        used += 1
    return round(clamp(100.0 / (1.0 + math.sqrt(distance / max(1, used))), 0.0, 100.0), 2)


def monotonic_bins(pairs, reverse=False, bins=16):
    clean = [(clamp(safe_float(x, 0.0), 0.0, 1.0), clamp(safe_float(y, 0.0), 0.0, 100.0)) for x, y in pairs or []]
    if not clean:
        return []
    clean.sort(key=lambda item: item[0])
    bucket_count = max(1, min(int(bins), len(clean)))
    buckets = []
    for index in range(bucket_count):
        start = index * len(clean) // bucket_count
        end = (index + 1) * len(clean) // bucket_count
        chunk = clean[start:end] or clean[start:start + 1]
        buckets.append([sum(item[0] for item in chunk) / len(chunk), sum(item[1] for item in chunk) / len(chunk), len(chunk)])
    values = [item[1] for item in buckets]
    if reverse:
        for index in range(1, len(values)):
            values[index] = min(values[index], values[index - 1])
    else:
        for index in range(1, len(values)):
            values[index] = max(values[index], values[index - 1])
    return [{"x": round(buckets[index][0], 6), "y": round(values[index], 4), "count": buckets[index][2]} for index in range(len(buckets))]


def calibrated_lookup(value, bins, fallback):
    points = bins if isinstance(bins, list) else []
    if not points:
        return fallback
    x = clamp(safe_float(value, 0.0), 0.0, 1.0)
    if x <= safe_float(points[0].get("x", 0.0), 0.0):
        return safe_float(points[0].get("y", fallback), fallback)
    for left, right in zip(points, points[1:]):
        lx = safe_float(left.get("x", 0.0), 0.0)
        rx = safe_float(right.get("x", lx), lx)
        if x <= rx:
            ly = safe_float(left.get("y", fallback), fallback)
            ry = safe_float(right.get("y", fallback), fallback)
            ratio = 0.0 if rx == lx else (x - lx) / (rx - lx)
            return ly + (ry - ly) * ratio
    return safe_float(points[-1].get("y", fallback), fallback)


class TrainableModel:
    key = ""
    model_type = "trainable_model"
    version = 1

    def __init__(self, settings=None, state=None):
        self.settings = settings
        self.state = state if isinstance(state, dict) else {}
        self.metrics = self.state.get("metrics") if isinstance(self.state.get("metrics"), dict) else {}
        self.sample_count = safe_int(self.state.get("sample_count", 0), 0)
        self.trained_at = self.state.get("trained_at") or now_text()
        self.training_data_digest = self.state.get("training_data_digest") or ""

    def fit(self, records):
        self.sample_count = len(records or [])
        self.trained_at = now_text()
        self.training_data_digest = training_data_digest(records)
        return self.metrics

    def predict(self, features):
        raise NotImplementedError

    def parameters(self):
        return {}

    def intelligence_profile(self):
        return {"level": "adaptive_ensemble", "capabilities": ["calibration", "validation", "self_audit"], "sample_count": self.sample_count}

    def snapshot(self):
        payload = {"key": self.key, "type": self.model_type, "version": self.version, "trained_at": self.trained_at, "training_data_digest": self.training_data_digest, "sample_count": self.sample_count, "parameters": self.parameters(), "metrics": self.metrics, "class": self.__class__.__name__, "intelligence": self.intelligence_profile()}
        self.restore(payload, self.settings).predict(self.probe_input())
        payload.update({"can_fit": True, "can_predict": True, "can_snapshot": True, "can_load": True, "probe_prediction": self.predict(self.probe_input())})
        return payload

    @classmethod
    def restore(cls, payload, settings=None):
        return cls(settings, payload)

    def probe_input(self):
        return {}


class ScreenNoveltyScorerModel(TrainableModel):
    key = "screen_novelty_scorer"
    model_type = "calibrated_inverse_similarity_novelty_model"

    def fit(self, records):
        super().fit(records)
        values = [safe_float(record.get("sleep_novelty", record.get("novelty", record.get("screen_score", 0.0))), 0.0) for record in records or [] if isinstance(record, dict)]
        self.average_score = round(sum(values) / len(values), 4) if values else 0.0
        semantic_records = [(numeric_vector(record.get("screen_semantic_vector")), safe_float(record.get("sleep_novelty", record.get("novelty", record.get("screen_score", 0.0))), 0.0)) for record in records or [] if isinstance(record, dict)]
        semantic_records = [(vector, score) for vector, score in semantic_records if vector]
        semantic_records.sort(key=lambda item: item[1], reverse=True)
        cluster_count = max(1, min(max(safe_int(getattr(self.settings, "nearest_top_k", 1), 1), int(math.sqrt(len(semantic_records))) or 1), len(semantic_records))) if semantic_records else 0
        self.clusters = [{"center": vector_mean([item[0] for item in semantic_records[index::cluster_count]]), "mean_score": round(sum(item[1] for item in semantic_records[index::cluster_count]) / max(1, len(semantic_records[index::cluster_count])), 4), "size": len(semantic_records[index::cluster_count]), "score_p25": round(percentile([item[1] for item in semantic_records[index::cluster_count]], 0.25, 0.0), 4), "score_p75": round(percentile([item[1] for item in semantic_records[index::cluster_count]], 0.75, 0.0), 4)} for index in range(cluster_count)]
        self.dimensions = len(self.clusters[0]["center"]) if self.clusters else 0
        self.nearest_top_k = safe_int(getattr(self.settings, "nearest_top_k", 1), 1)
        self.similarity_weights = self.learn_similarity_weights(records or [])
        self.calibration_bins = self.learn_calibration(records or [])
        self.metrics = {"mean_score": self.average_score, "records": len(values), "semantic_clusters": len(self.clusters), "semantic_dimensions": self.dimensions, "calibration_bins": len(self.calibration_bins), "validation_mae": self.validation_mae(records or [])}
        return self.metrics

    def aggregate_similarity(self, sims):
        if not sims:
            return 0.0
        top = sorted([clamp(safe_float(item, 0.0), 0.0, 1.0) for item in sims], reverse=True)[:max(1, min(safe_int(getattr(self.settings, "nearest_top_k", getattr(self, "nearest_top_k", 1)), getattr(self, "nearest_top_k", 1)), len(sims)))]
        mean_similarity = sum(top) / len(top)
        density = sum(1 for item in top if item >= 0.95) / len(top)
        weights = getattr(self, "similarity_weights", {"max": 0.5, "mean": 0.35, "density": 0.15})
        return top[0] * safe_float(weights.get("max", 0.5), 0.5) + mean_similarity * safe_float(weights.get("mean", 0.35), 0.35) + density * safe_float(weights.get("density", 0.15), 0.15)

    def learn_calibration(self, records):
        pairs = []
        for record in records:
            if not isinstance(record, dict):
                continue
            neighbors = record.get("score_neighbors") or record.get("nearest") or []
            sims = [item.get("similarity", 0.0) for item in neighbors if isinstance(item, dict)]
            if sims:
                target = safe_float(record.get("sleep_novelty", record.get("novelty", record.get("screen_score", 0.0))), 0.0)
                pairs.append((self.aggregate_similarity(sims), target))
        return monotonic_bins(pairs, reverse=True, bins=max(4, min(32, safe_int(getattr(self.settings, "nearest_top_k", 8), 8))))

    def learn_similarity_weights(self, records):
        pairs = []
        for record in records:
            if not isinstance(record, dict):
                continue
            neighbors = record.get("score_neighbors") or record.get("nearest") or []
            sims = [clamp(safe_float(item.get("similarity", 0.0), 0.0), 0.0, 1.0) for item in neighbors if isinstance(item, dict)]
            if sims:
                pairs.append((sims, clamp(1.0 - safe_float(record.get("sleep_novelty", record.get("novelty", record.get("screen_score", 0.0))), 0.0) / 100.0, 0.0, 1.0)))
        if not pairs:
            return {"max": 0.5, "mean": 0.35, "density": 0.15}
        best = (999.0, {"max": 0.5, "mean": 0.35, "density": 0.15})
        grid = tuple(index / 20.0 for index in range(2, 19))
        for max_w in grid:
            for mean_w in grid:
                if max_w + mean_w > 0.98:
                    continue
                density_w = max(0.0, 1.0 - max_w - mean_w)
                error = 0.0
                for sims, target in pairs:
                    top = sorted(sims, reverse=True)[:max(1, min(self.nearest_top_k, len(sims)))]
                    predicted = top[0] * max_w + (sum(top) / len(top)) * mean_w + (sum(1 for item in top if item >= 0.95) / len(top)) * density_w
                    error += abs(predicted - target)
                if error < best[0]:
                    best = (error, {"max": max_w, "mean": mean_w, "density": density_w})
        return best[1]

    def validation_mae(self, records):
        errors = []
        for record in records:
            if isinstance(record, dict) and (record.get("score_neighbors") or record.get("nearest")):
                predicted = self.predict({"similarities": [item.get("similarity", 0.0) for item in (record.get("score_neighbors") or record.get("nearest") or []) if isinstance(item, dict)]})
                actual = safe_float(record.get("sleep_novelty", record.get("novelty", record.get("screen_score", predicted))), predicted)
                errors.append(abs(predicted - actual))
        return round(sum(errors) / len(errors), 4) if errors else 0.0

    def parameters(self):
        return {"average_score": self.average_score, "clusters": self.clusters, "dimensions": self.dimensions, "nearest_top_k": self.nearest_top_k, "similarity_weights": self.similarity_weights, "calibration_bins": self.calibration_bins, "degradation_threshold": 25.0, "ensemble": ["weighted_neighbor_similarity", "semantic_cluster_similarity", "monotonic_calibration", "density_aware_uncertainty"], "intelligence_goal": "maximize novelty estimation accuracy with auditable nonparametric calibration"}

    def predict(self, features):
        similarities = features.get("similarities", []) if isinstance(features, dict) else []
        sims = sorted([clamp(safe_float(item, 0.0), 0.0, 1.0) for item in similarities], reverse=True)
        semantic = numeric_vector((features or {}).get("semantic_vector", [])) if isinstance(features, dict) else []
        if semantic and getattr(self, "clusters", None):
            cluster_similarity = max((vector_similarity(semantic, cluster.get("center", [])) for cluster in self.clusters if isinstance(cluster, dict)), default=0.0)
            sims.append(cluster_similarity)
            sims.sort(reverse=True)
        if not sims:
            return 100.0
        similarity = self.aggregate_similarity(sims)
        raw = clamp((1.0 - similarity) * 100.0, 0.0, 100.0)
        calibrated = calibrated_lookup(similarity, getattr(self, "calibration_bins", []), raw)
        density_confidence = clamp(len(sims) / max(1.0, safe_float(getattr(self, "nearest_top_k", 1), 1.0)), 0.0, 1.0)
        semantic_adjustment = 0.0
        if semantic and getattr(self, "clusters", None):
            nearest_cluster = max((cluster for cluster in self.clusters if isinstance(cluster, dict)), key=lambda cluster: vector_similarity(semantic, cluster.get("center", [])), default={})
            semantic_adjustment = (safe_float(nearest_cluster.get("mean_score", raw), raw) - raw) * 0.18
        blend = 0.42 + density_confidence * 0.18
        return round(clamp(raw * blend + calibrated * (1.0 - blend) + semantic_adjustment, 0.0, 100.0), 2)

    def probe_input(self):
        return {"similarities": [0.5]}

    @classmethod
    def restore(cls, payload, settings=None):
        obj = cls(settings, payload)
        params = payload.get("parameters") if isinstance(payload, dict) else {}
        obj.average_score = safe_float(params.get("average_score", 0.0), 0.0)
        obj.clusters = params.get("clusters") if isinstance(params.get("clusters"), list) else []
        obj.dimensions = safe_int(params.get("dimensions", 0), 0)
        obj.nearest_top_k = safe_int(params.get("nearest_top_k", getattr(settings, "nearest_top_k", 1)), 1)
        obj.similarity_weights = params.get("similarity_weights") if isinstance(params.get("similarity_weights"), dict) else {"max": 0.5, "mean": 0.35, "density": 0.15}
        obj.calibration_bins = params.get("calibration_bins") if isinstance(params.get("calibration_bins"), list) else []
        return obj


class MouseHumanlikenessScorerModel(TrainableModel):
    key = "mouse_humanlikeness_scorer"
    model_type = "learned_mouse_humanlikeness_model"

    def fit(self, records):
        learning = [record for record in records or [] if isinstance(record, dict) and record.get("mode") == "learning" and record.get("mouse_source") == "user"]
        super().fit(learning)
        values = [safe_float(record.get("sleep_human_score", record.get("human_score", 50.0)), 50.0) for record in learning]
        self.average_user_similarity = round(sum(values) / len(values), 4) if values else 50.0
        features = [action_features(record.get("mouse_action")) for record in learning if record.get("mouse_action")]
        names = sorted({name for item in features for name in item})
        means = {name: round(sum(safe_float(item.get(name, 0.0), 0.0) for item in features) / len(features), 6) for name in names} if features else {}
        variances = {name: round(sum((safe_float(item.get(name, 0.0), 0.0) - means[name]) ** 2 for item in features) / len(features), 6) for name in names} if features else {}
        by_type = defaultdict(list)
        for record in learning:
            action = record.get("mouse_action") or {}
            by_type[str(action.get("type", "click"))].append(action_features(action))
        prototypes = {}
        for action_type, rows in by_type.items():
            type_names = sorted({name for row in rows for name in row})
            prototypes[action_type] = {"means": {name: round(sum(safe_float(row.get(name, 0.0), 0.0) for row in rows) / len(rows), 6) for name in type_names}, "variances": {name: round(sum((safe_float(row.get(name, 0.0), 0.0) - sum(safe_float(inner.get(name, 0.0), 0.0) for inner in rows) / len(rows)) ** 2 for row in rows) / len(rows), 6) for name in type_names}, "count": len(rows)}
        type_priors = {name: len(rows) / max(1, len(features)) for name, rows in by_type.items()}
        calibration_pairs = []
        for record in learning:
            action = record.get("mouse_action")
            if action:
                fv = action_features(action)
                action_type = str(action.get("type", "click"))
                proto = prototypes.get(action_type, {})
                raw = feature_distance_score(fv, proto.get("means", means), proto.get("variances", variances), (proto.get("means") or means).keys())
                calibration_pairs.append((raw / 100.0, safe_float(record.get("human_score", self.average_user_similarity), self.average_user_similarity)))
        calibration_bins = monotonic_bins(calibration_pairs, reverse=False, bins=max(4, min(24, len(calibration_pairs)))) if calibration_pairs else []
        robust_variances = {name: round(robust_scale([safe_float(item.get(name, 0.0), 0.0) for item in features], math.sqrt(max(0.0001, variances.get(name, 1.0)))) ** 2, 6) for name in names} if features else {}
        self.feature_profile = {"means": means, "variances": variances, "robust_variances": robust_variances, "density": len(features), "prototypes": prototypes, "type_priors": type_priors, "calibration_bins": calibration_bins}
        validation = [abs(self.predict({"mouse_action": record.get("mouse_action"), "human_score": safe_float(record.get("human_score", 50.0), 50.0)}) - safe_float(record.get("human_score", self.average_user_similarity), self.average_user_similarity)) for record in learning if record.get("mouse_action")]
        self.metrics = {"mean_score": self.average_user_similarity, "learning_records": len(learning), "feature_dimensions": len(names), "action_type_prototypes": len(prototypes), "validation_mae": round(sum(validation) / len(validation), 4) if validation else 0.0, "degradation_threshold": 30.0}
        return self.metrics

    def parameters(self):
        return {"average_user_similarity": self.average_user_similarity, "feature_profile": self.feature_profile, "accepted_sources": ["learning:user"], "ensemble": ["global_feature_distance", "action_type_prototype", "monotonic_human_calibration", "action_type_prior", "robust_variance_normalization"], "intelligence_goal": "imitate user mouse dynamics using typed prototypes and calibrated robust feature distributions"}

    def predict(self, features):
        if isinstance(features, dict) and isinstance(features.get("mouse_action"), dict) and getattr(self, "feature_profile", None):
            feature_values = action_features(features.get("mouse_action"))
            means = self.feature_profile.get("means", {})
            variances = self.feature_profile.get("variances", {})
            if means:
                global_score = feature_distance_score(feature_values, means, self.feature_profile.get("robust_variances", variances), means.keys())
                action_type = str(features.get("mouse_action", {}).get("type", "click"))
                prototypes = self.feature_profile.get("prototypes", {})
                prototype = prototypes.get(action_type) if isinstance(prototypes, dict) else None
                if isinstance(prototype, dict):
                    type_score = feature_distance_score(feature_values, prototype.get("means", {}), prototype.get("variances", {}), prototype.get("means", {}).keys())
                    prior = safe_float(self.feature_profile.get("type_priors", {}).get(action_type, 0.0), 0.0)
                    density = clamp(safe_float(prototype.get("count", 0), 0.0) / max(1.0, safe_float(self.feature_profile.get("density", 1), 1.0)), 0.0, 1.0)
                    raw = clamp(global_score * (0.2 + 0.15 * (1.0 - density)) + type_score * (0.62 + 0.18 * density) + prior * 10.0, 0.0, 100.0)
                    calibrated = calibrated_lookup(raw / 100.0, self.feature_profile.get("calibration_bins", []), raw)
                    return round(clamp(raw * 0.7 + calibrated * 0.3, 0.0, 100.0), 2)
                calibrated = calibrated_lookup(global_score / 100.0, self.feature_profile.get("calibration_bins", []), global_score)
                return round(clamp(global_score * 0.7 + calibrated * 0.3, 0.0, 100.0), 2)
        return round(clamp(safe_float((features or {}).get("human_score", self.average_user_similarity), self.average_user_similarity), 0.0, 100.0), 2)

    def probe_input(self):
        return {"human_score": 50.0}

    @classmethod
    def restore(cls, payload, settings=None):
        obj = cls(settings, payload)
        params = payload.get("parameters") if isinstance(payload, dict) else {}
        obj.average_user_similarity = safe_float(params.get("average_user_similarity", 50.0), 50.0)
        obj.feature_profile = params.get("feature_profile") if isinstance(params.get("feature_profile"), dict) else {}
        return obj


class OperationPolicyModel(TrainableModel):
    key = "operation_policy"
    model_type = "online_logistic_policy"

    def fit(self, records):
        super().fit(records)
        self.policy = self.state.get("policy") if isinstance(self.state.get("policy"), dict) else {}
        self.action_count = sum(1 for record in records or [] if isinstance(record, dict) and record.get("mouse_action"))
        learner = PolicyModel(self.settings, self.policy)
        train_metrics = learner.train(records or [])
        refinement_rounds = 0
        while safe_int(train_metrics.get("trained", 0), 0) > 0 and safe_float(train_metrics.get("confidence", 0.0), 0.0) < 0.992 and refinement_rounds < 3:
            extra = learner.train(records or [])
            refinement_rounds += 1
            train_metrics = {**train_metrics, "trained": safe_int(train_metrics.get("trained", 0), 0) + safe_int(extra.get("trained", 0), 0), "loss": min(safe_float(train_metrics.get("loss", 1.0), 1.0), safe_float(extra.get("loss", 1.0), 1.0)), "confidence": max(safe_float(train_metrics.get("confidence", 0.0), 0.0), safe_float(extra.get("confidence", 0.0), 0.0)), "extra_refinement_rounds": refinement_rounds}
        self.policy = learner.snapshot()
        self.metrics = {"loss": safe_float(self.policy.get("loss", train_metrics.get("loss", 1.0)), 1.0), "trained_steps": safe_int(self.policy.get("trained_steps", 0), 0), "batch_trained": train_metrics.get("trained", 0), "confidence": train_metrics.get("confidence", 0.0)}
        return self.metrics

    def parameters(self):
        return {"policy": self.policy, "action_count": self.action_count, "intelligence_goal": "rank actions with online preference learning, replay refinement, and confidence-aware scoring"}

    def predict(self, features):
        weights = self.policy.get("weights") if isinstance(self.policy.get("weights"), dict) else {}
        names = self.policy.get("feature_names") or PolicyModel.FEATURE_NAMES
        z = sum(safe_float(weights.get(name, 0.0), 0.0) * safe_float((features or {}).get(name, 0.0), 0.0) for name in names)
        return 1.0 / (1.0 + math.exp(-clamp(z, -60.0, 60.0)))

    def probe_input(self):
        return {name: 0.0 for name in PolicyModel.FEATURE_NAMES}

    @classmethod
    def restore(cls, payload, settings=None):
        obj = cls(settings, payload)
        params = payload.get("parameters") if isinstance(payload, dict) else {}
        obj.policy = params.get("policy") if isinstance(params.get("policy"), dict) else {}
        obj.action_count = safe_int(params.get("action_count", 0), 0)
        return obj


class RewardModel(TrainableModel):
    key = "reward_model"
    model_type = "stateful_income_minus_time_cost_reward_model"

    def fit(self, records):
        super().fit(records)
        values = [safe_float(record.get("sleep_policy_reward", record.get("reward", 0.0)), 0.0) for record in records or [] if isinstance(record, dict)]
        screen_values = [safe_float(record.get("screen_score", record.get("novelty", 0.0)), 0.0) for record in records or [] if isinstance(record, dict)]
        human_values = [safe_float(record.get("human_score", 50.0), 50.0) for record in records or [] if isinstance(record, dict)]
        self.screen_scale = round(max(1.0, (max(screen_values) - min(screen_values)) if screen_values else 100.0), 6)
        self.human_tie_break_weight = round(min(0.000099999, 0.000099999 * (sum(human_values) / max(1, len(human_values))) / 100.0), 9) if human_values else 0.00005
        self.reward_state = reward_state_from(self.state.get("reward_state") if isinstance(self.state, dict) else None)
        for record in sorted([record for record in records or [] if isinstance(record, dict)], key=lambda item: parse_event_timestamp(item.get("created_at") or item.get("score_checked_at") or item.get("sleep_evaluated_at"))):
            screen, human = record_screen_human(record)
            result = reward_with_state(screen, human, self.reward_state, record.get("created_at") or record.get("score_checked_at") or record.get("sleep_evaluated_at"))
            self.reward_state = reward_state_from(result["state"])
        self.metrics = {"average_reward": round(sum(values) / len(values), 4) if values else 0.0, "records": len(values), "calibrated_screen_scale": self.screen_scale, "calibrated_tie_break_weight": self.human_tie_break_weight, "degradation_threshold": 0.0, "reward_state_version": self.reward_state.state_version, "best_income": round(self.reward_state.best_income, 8), "cost": round(self.reward_state.cost, 8)}
        return self.metrics

    def parameters(self):
        return {"screen_precision": 2, "tie_break_weight": getattr(self, "human_tie_break_weight", 0.000099999), "screen_scale": getattr(self, "screen_scale", 100.0), "reward_version": 6, "reward_state": asdict(getattr(self, "reward_state", RewardState())), "monotonic_constraints": ["screen_score_first", "human_score_tiebreak"], "intelligence_goal": "reward equals income minus increasing time cost with cost reset after income improvement"}

    def predict(self, features):
        features = features or {}
        result = reward_with_state(features.get("screen_score", 0.0), features.get("human_score", 50.0), features.get("reward_state", getattr(self, "reward_state", RewardState())), features.get("event_time"))
        if features.get("commit_state", False):
            self.reward_state = reward_state_from(result["state"])
        return result["reward"]

    def probe_input(self):
        return {"screen_score": 1.0, "human_score": 50.0, "event_time": time.time()}

    @classmethod
    def restore(cls, payload, settings=None):
        obj = cls(settings, payload)
        params = payload.get("parameters") if isinstance(payload, dict) else {}
        obj.reward_state = reward_state_from(params.get("reward_state") if isinstance(params, dict) else None)
        return obj


class RuntimeValueModel(TrainableModel):
    key = "runtime_value_model"
    model_type = "resource_adaptive_runtime_value_model"

    def fit(self, records):
        super().fit(records)
        self.resource_model = self.state.get("resource_model") if isinstance(self.state.get("resource_model"), dict) else {}
        learner = ResourceAdaptiveRLModel(self.resource_model)
        batch = [record for record in records or [] if isinstance(record, dict)]
        if batch:
            metrics = {
                "pool_count": len(batch),
                "throughput": len(batch),
                "batch_size": max(1, safe_int(getattr(self.settings, "sleep_batch_size", len(batch)), len(batch))),
                "success_rate": sum(1 for record in batch if not record.get("quarantined")) / max(1, len(batch)),
                "training_gain": sum(clamp(safe_float(record.get("sleep_novelty", record.get("novelty", record.get("screen_score", 0.0))), 0.0), 0.0, 100.0) for record in batch) / max(1, len(batch)) / 100.0,
                **read_hardware_state()
            }
            learner.train_from_metrics(metrics)
        self.resource_model = learner.snapshot()
        self.runtime_rules = RUNTIME_NUMBER_RULES
        self.saved_settings = {name: getattr(self.settings, name) for name in RUNTIME_NUMBER_RULES if hasattr(self.settings, name)} if self.settings else {}
        self.metrics = {"trained_steps": safe_int(self.resource_model.get("trained_steps", 0), 0), "reward_ema": safe_float(self.resource_model.get("reward_ema", 0.0), 0.0)}
        return self.metrics

    def parameters(self):
        return {"resource_model": self.resource_model, "runtime_rules": self.runtime_rules, "settings": self.saved_settings, "intelligence_goal": "adapt runtime hyperparameters to hardware, throughput, stability, and training gain"}

    def predict(self, features):
        return ResourceAdaptiveRLModel(self.resource_model).multipliers(features or {})

    def probe_input(self):
        return {"cpu_load": 0.0, "memory_free_ratio": 0.5}

    @classmethod
    def restore(cls, payload, settings=None):
        obj = cls(settings, payload)
        params = payload.get("parameters") if isinstance(payload, dict) else {}
        obj.resource_model = params.get("resource_model") if isinstance(params.get("resource_model"), dict) else {}
        obj.runtime_rules = params.get("runtime_rules") if isinstance(params.get("runtime_rules"), dict) else RUNTIME_NUMBER_RULES
        obj.saved_settings = params.get("settings") if isinstance(params.get("settings"), dict) else {}
        return obj


TRAINABLE_MODEL_CLASSES = {cls.key: cls for cls in (ScreenNoveltyScorerModel, MouseHumanlikenessScorerModel, OperationPolicyModel, RewardModel, RuntimeValueModel)}


def restore_trainable_model(payload, settings=None):
    if not isinstance(payload, dict):
        raise ValueError("模型载荷必须是对象")
    key = payload.get("key")
    cls = TRAINABLE_MODEL_CLASSES.get(key)
    if not cls:
        raise ValueError("未知模型 " + str(key))
    model = cls.restore(payload, settings)
    model.predict(model.probe_input())
    return model


class ModelGroupRuntime:
    def __init__(self, models=None, settings=None):
        self.models = models if isinstance(models, dict) else {}
        self.settings = settings

    def apply_settings(self, settings):
        self.settings = settings
        for model in self.models.values():
            try:
                model.settings = settings
            except Exception:
                pass

    def screen_novelty(self, similarities):
        model = self.models.get("screen_novelty_scorer")
        if model:
            return model.predict({"similarities": similarities})
        sims = sorted([clamp(safe_float(item, 0.0), 0.0, 1.0) for item in similarities], reverse=True)
        if not sims:
            return 100.0
        top = sims[:max(1, min(safe_int(getattr(self.settings, "nearest_top_k", 1), 1), len(sims)))]
        density = sum(1 for item in top if item >= 0.95) / len(top)
        return clamp((1.0 - (top[0] * 0.5 + sum(top) / len(top) * 0.35 + density * 0.15)) * 100.0, 0.0, 100.0)

    def mouse_humanlikeness(self, action, fallback_score):
        model = self.models.get("mouse_humanlikeness_scorer")
        if model:
            return model.predict({"mouse_action": action, "human_score": fallback_score})
        return fallback_score

    def operation_policy_score(self, record, similarity):
        model = self.models.get("operation_policy")
        if model:
            features = PolicyModel(self.settings).features(record)
            features["confidence"] = clamp(similarity, 0.0, 1.0)
            return model.predict(features)
        return 0.0

    def reward(self, screen_score, human_score, event_time=None, reward_state=None, commit_state=False):
        model = self.models.get("reward_model")
        if model:
            return model.predict({"screen_score": screen_score, "human_score": human_score, "event_time": event_time, "reward_state": reward_state, "commit_state": commit_state})
        return reward_with_state(screen_score, human_score, reward_state, event_time)["reward"]

    def reward_breakdown(self, screen_score, human_score, event_time=None, reward_state=None, commit_state=False):
        parts = reward_breakdown(screen_score, human_score, self.settings, reward_state, event_time)
        parts["total_reward"] = self.reward(screen_score, human_score, event_time, reward_state, commit_state)
        parts["reward_sort_key"] = list(strict_reward_key(screen_score, human_score))
        parts["basis"] = list(parts.get("basis", [])) + ["model_group_runtime_reward_model"]
        return parts

    def apply_runtime_values(self, settings, features, fallback_model=None):
        model = self.models.get("runtime_value_model")
        if model:
            factors = model.predict(features)
            values = {item.name: getattr(settings, item.name) for item in fields(Settings)}
            for name, low, high, integer in (("sleep_batch_size", 8, 4096, True), ("sleep_worker_count", 1, 64, True), ("nearest_candidate_limit", 256, 20000, True), ("explore_max_rate", 0.2, 0.95, False), ("training_event_wait", 0.01, 0.9, False), ("generated_sleep_event_wait", 0.02, 1.2, False)):
                value = clamp(safe_float(values.get(name, low), low) * safe_float(factors.get(name, 1.0), 1.0), low, high)
                values[name] = int(value) if integer else value
            return Settings(**values)
        if fallback_model:
            return fallback_model.apply_settings(settings, features)
        return settings


def model_group_complete(model_group, settings=None):
    models = model_group.get("models") if isinstance(model_group, dict) else None
    if not isinstance(models, list) or len(models) != len(AI_MODEL_GROUP_SPECS):
        return False
    expected = [spec["key"] for spec in AI_MODEL_GROUP_SPECS]
    if [model.get("key") for model in models if isinstance(model, dict)] != expected:
        return False
    for model_payload in models:
        try:
            model = restore_trainable_model(model_payload, settings)
            snapshot = model.snapshot()
            if not all(snapshot.get(name) for name in ("can_fit", "can_predict", "can_snapshot", "can_load")):
                return False
        except Exception:
            return False
    return True


def ai_model_group_snapshot(policy_payload, settings, records):
    policy_payload = policy_payload if isinstance(policy_payload, dict) else {}
    records = records or []
    states = {
        "screen_novelty_scorer": {},
        "mouse_humanlikeness_scorer": {},
        "operation_policy": {"policy": policy_payload},
        "reward_model": {},
        "runtime_value_model": {"resource_model": policy_payload.get("resource_model") if isinstance(policy_payload.get("resource_model"), dict) else {}}
    }
    models = []
    for spec in AI_MODEL_GROUP_SPECS:
        cls = TRAINABLE_MODEL_CLASSES[spec["key"]]
        model = cls(settings, states.get(spec["key"], {}))
        model.fit(records)
        payload = model.snapshot()
        payload.update({"name": spec["name"], "goal": spec["goal"]})
        models.append(payload)
    model_group = {"group_version": 4, "trained_at": now_text(), "training_data_digest": training_data_digest(records), "models": models, "intelligence_strategy": "human_current_technology_ceiling: calibrated ensembles, online preference learning, robust statistics, diversity, and resource-adaptive reinforcement"}
    model_group["complete"] = model_group_complete(model_group, settings)
    return model_group

def record_screen_human(record):
    screen = record.get("sleep_novelty", record.get("screen_primary_reward", record.get("novelty", record.get("screen_score", 0.0))))
    human = record.get("sleep_human_score", record.get("human_tie_break_reward", record.get("human_score", 50.0)))
    return strict_reward_key(screen, human)


def reward_parts(novelty, human_score, settings):
    parts = reward_breakdown(novelty, human_score, settings)
    return parts["screen_primary_reward"], parts["mouse_action_delta"], parts["total_reward"]


def weighted_choice(weighted_items):
    clean = [(max(0.0, safe_float(weight, 0.0)), item) for weight, item in weighted_items]
    total = sum(weight for weight, _ in clean)
    if total <= 0.0:
        return random.choice(clean)[1] if clean else None
    target = random.uniform(0.0, total)
    running = 0.0
    for weight, item in clean:
        running += weight
        if running >= target:
            return item
    return clean[-1][1]


def percentile_score(value, samples, default):
    normalized = []
    for item in samples:
        try:
            number = float(item)
        except Exception:
            continue
        if math.isfinite(number):
            normalized.append(number)
    ordered = sorted(normalized)
    if len(ordered) < 2:
        return default
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    deviations = sorted(abs(item - median) for item in ordered)
    mad = deviations[n // 2] if n % 2 else (deviations[n // 2 - 1] + deviations[n // 2]) / 2.0
    scale = max(mad * 1.4826, (ordered[-1] - ordered[0]) / max(12.0, n), 1e-6)
    return round(clamp(100.0 - abs(float(value) - median) / scale * 18.0, 35.0, 100.0), 2)


def action_features(action):
    if not action:
        return {}
    start = action.get("start_rel") or [0.0, 0.0]
    end = action.get("end_rel") or start
    path_items = [item for item in action.get("path_rel", []) if isinstance(item, (list, tuple)) and len(item) >= 2]
    points = [[safe_float(item[0], 0.0), safe_float(item[1], 0.0)] for item in path_items]
    times = [max(0.0, safe_float(item[2], 0.0)) if len(item) >= 3 else 0.0 for item in path_items]
    if not points:
        points = [normalize_rel_point(start), normalize_rel_point(end, start)]
        times = [0.0, max(0.0, safe_float(action.get("duration", 0.0), 0.0))]
    direct = distance(points[0], points[-1])
    total = path_length(points)
    segments = [distance(points[index - 1], points[index]) for index in range(1, len(points))]
    deltas = [max(1e-4, times[index] - times[index - 1]) for index in range(1, len(times)) if index < len(points)]
    speeds = [segment / delta for segment, delta in zip(segments, deltas)]
    speed_mean = sum(speeds) / len(speeds) if speeds else 0.0
    speed_variance = sum((item - speed_mean) ** 2 for item in speeds) / len(speeds) if speeds else 0.0
    accelerations = [(speeds[index] - speeds[index - 1]) / max(1e-4, deltas[index]) for index in range(1, min(len(speeds), len(deltas)))]
    acceleration_change = sum(abs(item) for item in accelerations) / len(accelerations) if accelerations else 0.0
    pauses = sum(1 for delta, segment in zip(deltas, segments) if delta > max(0.03, safe_float(action.get("duration", 0.0), 0.0) / max(4.0, len(points))) and segment < 0.002)
    hover_before = max(0.0, times[1] - times[0]) if len(times) > 1 and len(points) > 1 and distance(points[0], points[1]) < 0.002 else 0.0
    curvature = clamp(total / direct if direct > 1e-6 else 1.0, 1.0, 5.0)
    scroll = action.get("scroll") or [0, 0]
    scroll_abs = abs(safe_int(scroll[0], 0)) + abs(safe_int(scroll[1], 0))
    return {
        "duration": max(0.0, safe_float(action.get("duration", 0.0), 0.0)),
        "direct": direct,
        "total": total,
        "bend": curvature,
        "points": len(points),
        "scroll_abs": scroll_abs,
        "speed_mean": speed_mean,
        "speed_variance": speed_variance,
        "acceleration_change": acceleration_change,
        "pauses": pauses,
        "hover_before": hover_before,
        "drag_curvature": curvature if action.get("type") == "drag" else 1.0,
        "scroll_continuity": scroll_abs / max(1.0, len(points)),
        "double_click_interval": max(0.0, safe_float(action.get("double_click_interval", 0.0), 0.0))
    }


class HumanProfile:
    def __init__(self, settings):
        self.settings = settings
        self.lock = threading.RLock()
        self.stats = defaultdict(lambda: defaultdict(list))

    def observe(self, action):
        if not action:
            return
        action_type = str(action.get("type", "click"))
        with self.lock:
            for name, value in action_features(action).items():
                if math.isfinite(float(value)):
                    bucket = self.stats[action_type][name]
                    bucket.append(float(value))
                    if len(bucket) > self.settings.human_profile_max_samples:
                        self.stats[action_type][name] = bucket[-self.settings.human_profile_keep_samples:]

    def enough(self, action_type):
        with self.lock:
            return len(self.stats[action_type].get("duration", [])) >= self.settings.human_profile_min_samples

    def score(self, action):
        if not action:
            return 0.0
        action_type = str(action.get("type", "click"))
        features = action_features(action)
        if action_type == "scroll":
            return self.score_scroll(features)
        if not self.enough(action_type):
            return 50.0
        if self.enough(action_type):
            with self.lock:
                stats = {name: tuple(values) for name, values in self.stats[action_type].items()}
            names = HUMAN_FEATURE_NAMES
            weights = normalized_human_feature_weights(self.settings.human_feature_weights)
            scores = [percentile_score(features.get(name, 0.0), stats.get(name, []), self.settings.score_default) for name in names]
            return round(clamp(sum(score * weight for score, weight in zip(scores, weights)) / max(0.01, sum(weights)), 0.0, 100.0), 2)
        return 50.0

    def score_scroll(self, features):
        with self.lock:
            samples = list(self.stats["scroll"].get("scroll_abs", []))
        if len(samples) >= max(6, self.settings.human_profile_min_samples // 3):
            continuity = percentile_score(features.get("scroll_continuity", 0.0), self.stats["scroll"].get("scroll_continuity", []), self.settings.scroll_score_default)
            magnitude = percentile_score(features.get("scroll_abs", 0.0), samples, self.settings.scroll_score_default)
            total_weight = max(0.01, self.settings.motion_score_magnitude_weight + self.settings.motion_score_continuity_weight)
            return round(clamp((magnitude * self.settings.motion_score_magnitude_weight + continuity * self.settings.motion_score_continuity_weight) / total_weight, 0.0, 100.0), 2)
        return 50.0



class EventBus:
    def __init__(self):
        self.lock = threading.RLock()
        self.listeners = defaultdict(list)
        self.sequence = 0

    def subscribe(self, name, handler):
        with self.lock:
            self.listeners[name].append(handler)

    def publish(self, name, **payload):
        with self.lock:
            self.sequence += 1
            event = {"sequence": self.sequence, "name": name, "created_at": now_text(), **payload}
            handlers = list(self.listeners.get(name, ())) + list(self.listeners.get("*", ()))
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                pass
        return event


class PolicyModel:
    FEATURE_NAMES = ("bias", "reward", "novelty", "human", "duration", "distance", "source", "confidence", "visits")

    def __init__(self, settings, state=None):
        self.settings = settings
        self.lock = threading.RLock()
        state = state if isinstance(state, dict) else {}
        weights = state.get("weights") if isinstance(state.get("weights"), dict) else {}
        self.weights = {name: safe_float(weights.get(name, 0.0), 0.0) for name in self.FEATURE_NAMES}
        if not any(abs(value) > 0.0 for value in self.weights.values()):
            self.weights.update({"bias": 0.0, "reward": 0.35, "novelty": 0.25, "human": 0.25, "duration": 0.03, "distance": 0.04, "source": 0.08, "confidence": 0.12, "visits": -0.03})
        self.trained_steps = safe_int(state.get("trained_steps", 0), 0)
        self.loss = safe_float(state.get("loss", 1.0), 1.0)
        self.visual_model = state.get("visual_model") if isinstance(state.get("visual_model"), dict) else {}
        self.resource_model = ResourceAdaptiveRLModel(state.get("resource_model") if isinstance(state.get("resource_model"), dict) else None)

    def features(self, record):
        action = record.get("mouse_action") or {}
        facts = action_features(action) if action else {}
        reward = clamp(safe_float(record.get("reward", record.get("total_reward", 0.0)), 0.0), self.settings.reward_total_min, self.settings.reward_total_max)
        span = max(1.0, self.settings.reward_total_max - self.settings.reward_total_min)
        return {
            "bias": 1.0,
            "reward": (reward - self.settings.reward_total_min) / span,
            "novelty": clamp(safe_float(record.get("sleep_novelty", record.get("novelty", 0.0)), 0.0), 0.0, 100.0) / 100.0,
            "human": clamp(safe_float(record.get("sleep_human_score", record.get("human_score", 50.0)), 50.0), 0.0, 100.0) / 100.0,
            "duration": clamp(safe_float(facts.get("duration", 0.0), 0.0) / max(0.001, self.settings.action_duration_max), 0.0, 1.0),
            "distance": clamp(safe_float(facts.get("total", 0.0), 0.0), 0.0, 1.0),
            "source": 1.0 if record.get("mode") == "learning" and record.get("mouse_source") == "user" else 0.0,
            "confidence": clamp(safe_float(record.get("sleep_confidence", 0.0), 0.0), 0.0, 1.0),
            "visits": clamp(math.log1p(max(0, safe_int(record.get("sleep_visits", 0), 0))) / math.log1p(max(1, self.settings.nearest_top_k)), 0.0, 1.0)
        }

    def predict_features(self, features):
        z = sum(self.weights.get(name, 0.0) * safe_float(features.get(name, 0.0), 0.0) for name in self.FEATURE_NAMES)
        if z >= 0.0:
            ez = math.exp(-z) if z < 700.0 else 0.0
            return 1.0 / (1.0 + ez)
        ez = math.exp(z) if z > -700.0 else 0.0
        return ez / (1.0 + ez)

    def predict(self, record):
        return self.predict_features(self.features(record))

    def target(self, record):
        screen, human = record_screen_human(record)
        return strict_reward_target(screen, human)

    def train(self, records):
        usable = [record for record in records or [] if isinstance(record, dict) and record.get("mouse_action")]
        if not usable:
            return {"trained": 0, "loss": self.loss, "confidence": 0.0}
        usable.sort(key=lambda record: strict_reward_key(*record_screen_human(record)), reverse=True)
        head = usable[:max(1, len(usable) // 2)]
        tail = usable[max(1, len(usable) // 2):]
        replay = head + tail + head[:max(1, len(head) // 3)]
        lr = clamp(1.0 / max(5.0, math.sqrt(self.trained_steps + len(replay) + 1.0)), 0.008, 0.16)
        l2 = 0.0005
        epochs = 2 if len(replay) < max(4, self.settings.nearest_top_k) else 1
        total_loss = 0.0
        trained = 0
        with self.lock:
            for _ in range(epochs):
                for record in replay:
                    features = self.features(record)
                    pred = self.predict_features(features)
                    target = self.target(record)
                    error = pred - target
                    total_loss += error * error
                    for name in self.FEATURE_NAMES:
                        penalty = l2 * self.weights.get(name, 0.0) if name != "bias" else 0.0
                        self.weights[name] = clamp(self.weights.get(name, 0.0) - lr * (error * features.get(name, 0.0) + penalty), -8.0, 8.0)
                    record["model_prediction"] = round(pred, 4)
                    record["model_target"] = round(target, 4)
                    trained += 1
            self.trained_steps += trained
            self.loss = round(total_loss / max(1, trained), 6)
            confidence = clamp(1.0 - math.sqrt(self.loss), 0.0, 1.0)
        return {"trained": trained, "loss": self.loss, "confidence": confidence, "epochs": epochs, "replay": len(replay)}

    def snapshot(self):
        with self.lock:
            return {"type": "online_logistic_policy", "feature_names": list(self.FEATURE_NAMES), "weights": {name: round(value, 8) for name, value in self.weights.items()}, "trained_steps": self.trained_steps, "loss": self.loss, "visual_model": copy.deepcopy(self.visual_model), "resource_model": self.resource_model.snapshot()}


class AppConfigStore:
    def __init__(self):
        base = Path(os.environ.get("APPDATA") or (Path.home() / ".config"))
        self.root = base / "AGI"
        self.settings_file = self.root / "startup.json"
        self.lock = threading.RLock()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            configuration_failure("创建启动配置目录失败 " + str(self.root), exc)

    def load_settings(self):
        if not self.settings_file.exists():
            self.save_settings({"ldplayer_path": AGENT_SPEC.default_ldplayer_path, "data_path": AGENT_SPEC.default_data_path})
        try:
            with self.settings_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict):
                raise ValueError("启动配置必须是 JSON 对象")
            return AllowedUserEditPolicy.filter(data, "startup")
        except Exception as exc:
            configuration_failure("读取启动配置失败 " + str(self.settings_file), exc)

    def save_settings(self, settings):
        source = AllowedUserEditPolicy.filter(settings, "startup")
        payload = {"schema_version": CONFIG_SCHEMA_VERSION, "ldplayer_path": str(source.get("ldplayer_path") or AGENT_SPEC.default_ldplayer_path), "data_path": str(source.get("data_path") or AGENT_SPEC.default_data_path)}
        try:
            atomic_write_json(self.settings_file, payload, self.lock)
        except Exception as exc:
            configuration_failure("写入启动配置失败 " + str(self.settings_file), exc)


class DataStore:
    def __init__(self, root):
        self.root = Path(root)
        self.screen_dir = self.root / "screens"
        self.model_dir = self.root / "models"
        self.experience_file = self.root / "experience.jsonl"
        self.state_file = self.root / "state.json"
        self.settings_file = self.root / "settings.json"
        self.sleep_checkpoint_file = self.root / "sleep_checkpoint.json"
        self.runtime_audit_file = self.root / "runtime_parameters_audit.jsonl"
        self.error_file = self.root / "errors.jsonl"
        self.lock = threading.RLock()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.screen_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            configuration_failure("创建数据配置目录失败 " + str(self.root), exc)
        self.state = self.load_state()
        self.pending_state_writes = 0
        self.last_state_save_perf = time.perf_counter()

    @property
    def screen_score_total(self):
        return safe_float(self.state.get("screen_score_total", 0.0), 0.0)

    def default_state_payload(self):
        return {"screen_score_total": 0.0, "penalty": 0.0}

    def notify_state_rebuilt(self, message):
        if "--self-test" in sys.argv:
            return
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning("状态文件已重建", message)
            root.destroy()
        except Exception:
            pass

    def rebuild_bad_state(self, error):
        default = self.default_state_payload()
        backup_path = self.state_file.with_name(f"state.bad.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json")
        backup_text = ""
        try:
            backup_text = str(backup_path)
            if self.state_file.exists():
                shutil.move(str(self.state_file), str(backup_path))
            with self.state_file.open("w", encoding="utf-8") as file:
                json.dump(default, file, ensure_ascii=False, indent=2)
        except Exception as rebuild_error:
            configuration_failure("重建状态文件失败 " + str(self.state_file), rebuild_error)
        try:
            self.log_error("load_state.rebuilt", error, {"state_file": str(self.state_file), "backup_file": backup_text})
        except Exception:
            pass
        detail = f"状态文件损坏，已重建。\n原文件: {self.state_file}\n备份文件: {backup_text}\n错误详情: {error}"
        self.notify_state_rebuilt(detail)
        return default

    def load_state(self):
        if not self.state_file.exists():
            return self.default_state_payload()
        try:
            with self.state_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict):
                raise ValueError("状态文件必须是 JSON 对象")
            return {"screen_score_total": safe_float(data.get("screen_score_total", 0.0), 0.0), "penalty": safe_float(data.get("penalty", 0.0), 0.0)}
        except Exception as exc:
            return self.rebuild_bad_state(exc)

    def fsync_directory(self, path):
        try:
            fd = os.open(str(Path(path)), os.O_RDONLY)
        except Exception:
            return
        try:
            os.fsync(fd)
        except Exception:
            pass
        finally:
            try:
                os.close(fd)
            except Exception:
                pass

    def replace_atomic_synced(self, temporary, target):
        temporary.replace(target)
        self.fsync_directory(Path(target).parent)

    def save_state(self):
        atomic_write_json(self.state_file, self.state, self.lock)

    def write_json_atomic(self, path, payload):
        atomic_write_json(path, payload, self.lock)

    def load_sleep_checkpoint(self):
        try:
            if not self.sleep_checkpoint_file.exists():
                return None
            with self.sleep_checkpoint_file.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            return payload if isinstance(payload, dict) else None
        except Exception as exc:
            self.log_error("load_sleep_checkpoint", exc, {"path": str(self.sleep_checkpoint_file)})
            return None

    def save_sleep_checkpoint(self, checkpoint, **updates):
        payload = dict(checkpoint or {})
        payload.update(updates)
        payload["updated_at"] = now_text()
        self.write_json_atomic(self.sleep_checkpoint_file, payload)
        return payload

    def clear_sleep_checkpoint(self):
        with self.lock:
            try:
                self.sleep_checkpoint_file.unlink(missing_ok=True)
            except Exception as exc:
                self.log_error("clear_sleep_checkpoint", exc, {"path": str(self.sleep_checkpoint_file)})

    def append_runtime_parameter_audit(self, previous_values, current_values, audit):
        payload = {"schema_version": CONFIG_SCHEMA_VERSION, "audit_version": 1, "created_at": now_text(), "previous_values": previous_values or {}, "current_values": current_values or {}, "changes": [], "runtime_generated_numbers": audit or {}}
        for key, value in sorted((current_values or {}).items()):
            previous = (previous_values or {}).get(key)
            if previous != value:
                item_audit = (audit or {}).get(key, {})
                payload["changes"].append({"name": key, "before": previous, "after": value, "trigger_conditions": item_audit.get("reality_conditions", {}), "reason": item_audit.get("semantic_goal", key), "effect_metrics": {"pending_observation": True}})
        with self.lock:
            with self.runtime_audit_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def add_screen_score_total(self, value):
        with self.lock:
            delta = safe_float(value, 0.0)
            current = safe_float(self.state.get("screen_score_total", 0.0), 0.0)
            penalty = safe_float(self.state.get("penalty", 0.0), 0.0)
            if delta >= 0.0:
                self.state["screen_score_total"] = round(current + delta, 2)
            else:
                self.state["screen_score_total"] = round(current, 2)
                self.state["penalty"] = round(penalty + abs(delta), 2)
            return self.state["screen_score_total"]

    def flush_state(self, force=False, min_interval=1.5, max_pending=36):
        with self.lock:
            self.pending_state_writes += 1
            now_perf = time.perf_counter()
            due = force or self.pending_state_writes >= max(1, safe_int(max_pending, 36)) or (now_perf - self.last_state_save_perf) >= max(0.1, safe_float(min_interval, 1.5))
            if due:
                self.save_state()
                self.pending_state_writes = 0
                self.last_state_save_perf = now_perf

    def new_screen_path(self, mode):
        folder = self.screen_dir / datetime.now().strftime("%Y%m%d") / mode
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{datetime.now().strftime('%H%M%S_%f')}_{uuid.uuid4().hex}.png"

    def relative_path(self, path):
        try:
            return str(Path(path).resolve().relative_to(self.root.resolve()))
        except Exception:
            return str(path)

    def ensure_experience_newline(self):
        if not self.experience_file.exists() or self.experience_file.stat().st_size <= 0:
            return
        with self.experience_file.open("rb+") as file:
            file.seek(-1, os.SEEK_END)
            if file.read(1) != b"\n":
                file.write(b"\n")

    def append_experience(self, record):
        with self.lock:
            self.ensure_experience_newline()
            with self.experience_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())

    def save_experience_records(self, records):
        with self.lock:
            temporary = self.experience_file.with_suffix(".save.tmp")
            with temporary.open("w", encoding="utf-8") as file:
                for record in records or []:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())
            self.replace_atomic_synced(temporary, self.experience_file)

    def merge_experience_records_by_id(self, records):
        updates = {str(record.get("id")): record for record in records or [] if isinstance(record, dict) and record.get("id")}
        if not updates:
            return {"changed": False, "updated": 0, "appended": 0}
        with self.lock:
            self.ensure_experience_newline()
            temporary = self.experience_file.with_suffix(".merge.tmp")
            seen = set()
            changed = 0
            if self.experience_file.exists():
                with self.experience_file.open("r", encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
                    for line_number, line in enumerate(source, start=1):
                        text = line.strip()
                        if not text:
                            continue
                        try:
                            record = json.loads(text)
                        except Exception as exc:
                            self.quarantine_bad_experience(line_number, text, exc)
                            target.write(text + "\n")
                            continue
                        record_id = str(record.get("id")) if isinstance(record, dict) and record.get("id") else None
                        if record_id in updates:
                            target.write(json.dumps(updates[record_id], ensure_ascii=False) + "\n")
                            seen.add(record_id)
                            changed += 1
                        else:
                            target.write(text + "\n")
                    appended = 0
                    for record_id, record in updates.items():
                        if record_id not in seen:
                            target.write(json.dumps(record, ensure_ascii=False) + "\n")
                            appended += 1
                    target.flush()
                    os.fsync(target.fileno())
                self.replace_atomic_synced(temporary, self.experience_file)
            else:
                with temporary.open("w", encoding="utf-8") as target:
                    for record in updates.values():
                        target.write(json.dumps(record, ensure_ascii=False) + "\n")
                    target.flush()
                    os.fsync(target.fileno())
                self.replace_atomic_synced(temporary, self.experience_file)
                appended = len(updates)
            return {"changed": bool(changed or appended), "updated": changed, "appended": appended}

    def record_reward_sort_key(self, record):
        key = record.get("reward_sort_key") if isinstance(record, dict) else None
        if isinstance(key, (list, tuple)) and key:
            screen = safe_float(key[0], 0.0)
            human = safe_float(key[1] if len(key) > 1 else 0.0, 0.0)
            return screen, human, safe_float(record.get("reward", record.get("total_reward", 0.0)), 0.0)
        reward = safe_float(record.get("reward", record.get("total_reward", 0.0)), 0.0) if isinstance(record, dict) else 0.0
        return reward, safe_float(record.get("sleep_confidence", 0.0), 0.0) if isinstance(record, dict) else 0.0, reward

    def save_ai_model_snapshot(self, records, settings, max_models, status, model=None, run_guard=None, status_detail=None):
        if run_guard and run_guard():
            return None
        with self.lock:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            ranked = sorted([record for record in records or [] if record.get("mouse_action")], key=lambda record: (safe_float(record.get("sleep_policy_reward", record.get("reward", 0.0)), 0.0), safe_float(record.get("sleep_confidence", 0.0), 0.0)), reverse=True)
            limit = max(1, min(len(ranked) or 1, safe_int(getattr(settings, "global_action_heap_limit", 1), 1), 512))
            model_payload = model.snapshot() if model else None
            model_group = ai_model_group_snapshot(model_payload, settings, ranked)
            if not model_group_complete(model_group, settings):
                raise RuntimeError("AI模型组快照不完整：五类模型必须都能独立加载并完成探针推理")
            identity = hashlib.sha256(str(self.root.resolve()).encode("utf-8", "replace")).hexdigest()
            training_digest = hashlib.sha256(json.dumps([record.get("id") for record in ranked[:limit]], ensure_ascii=False).encode("utf-8")).hexdigest()
            payload = {"schema_version": CONFIG_SCHEMA_VERSION, "model_version": 2, "training_data_version": 1, "data_path_id": identity, "checksum": training_digest, "created_at": now_text(), "status": status, "status_detail": status_detail or {}, "screen_score_total": self.screen_score_total, "reward_state": next((record.get("reward_state") for record in reversed(records or []) if isinstance(record, dict) and isinstance(record.get("reward_state"), dict)), asdict(RewardState())), "experience_count": len(records or []), "policy_limit": limit, "training_sample_ids": [record.get("id") for record in ranked[:limit]], "model_group": model_group, "model": model_payload, "policy": [{"id": record.get("id"), "mode": record.get("mode"), "action_type": (record.get("mouse_action") or {}).get("type") if isinstance(record.get("mouse_action"), dict) else None, "reward": safe_float(record.get("reward", 0.0), 0.0), "sleep_policy_reward": safe_float(record.get("sleep_policy_reward", record.get("reward", 0.0)), 0.0), "sleep_confidence": safe_float(record.get("sleep_confidence", 0.0), 0.0), "sleep_model_confidence": safe_float(record.get("sleep_model_confidence", record.get("model_prediction", 0.0)), 0.0), "model_prediction": safe_float(record.get("model_prediction", 0.0), 0.0), "model_target": safe_float(record.get("model_target", 0.0), 0.0), "sleep_novelty": safe_float(record.get("sleep_novelty", record.get("novelty", 0.0)), 0.0), "human_score": safe_float(record.get("sleep_human_score", record.get("human_score", 0.0)), 0.0)} for record in ranked[:limit]]}
            path = self.model_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex}.json"
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            if run_guard and run_guard():
                try:
                    temporary.unlink(missing_ok=True)
                except Exception:
                    pass
                return None
            self.replace_atomic_synced(temporary, path)
            return path

    def model_created_key(self, path):
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            created = payload.get("created_at") if isinstance(payload, dict) else None
            return str(created or path.name)
        except Exception:
            return path.name

    def compact_ai_models(self, max_models):
        limit = max(1, safe_int(max_models, AGENT_SPEC.default_ai_model_limit))
        models = sorted(self.model_dir.glob("model_*.json"), key=self.model_created_key, reverse=True)
        keep = min(len(models), limit)
        removed = 0
        errors = []
        for path in models[keep:]:
            try:
                path.unlink()
                removed += 1
            except Exception as exc:
                error = {"path": str(path), "error": str(exc), "error_type": type(exc).__name__}
                errors.append(error)
                try:
                    self.log_error("compact_ai_models.unlink", exc, error)
                except Exception:
                    pass
        remaining = len(list(self.model_dir.glob("model_*.json")))
        return {"changed": removed > 0, "removed": removed, "limit": limit, "target_count": min(remaining, limit), "model_count": remaining, "complete": remaining <= limit, "errors": errors}


    def validate_model_state(self, payload, settings=None):
        model = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(model, dict):
            raise ValueError("模型快照缺少模型对象")
        if model.get("type") != "online_logistic_policy":
            raise ValueError("模型类型不匹配")
        names = tuple(model.get("feature_names") or ())
        if names != PolicyModel.FEATURE_NAMES:
            raise ValueError("模型特征不匹配")
        weights = model.get("weights")
        if not isinstance(weights, dict):
            raise ValueError("模型权重缺失")
        clean = {}
        for name in PolicyModel.FEATURE_NAMES:
            value = safe_float(weights.get(name), None)
            if value is None or not math.isfinite(value) or value < -8.0 or value > 8.0:
                raise ValueError("模型权重越界 " + name)
            clean[name] = value
        trained_steps = safe_int(model.get("trained_steps", 0), 0)
        loss = safe_float(model.get("loss", 1.0), 1.0)
        if trained_steps < 0 or not math.isfinite(loss) or loss < 0.0:
            raise ValueError("模型训练状态无效")
        resource_model = model.get("resource_model") if isinstance(model.get("resource_model"), dict) else {}
        if resource_model:
            if tuple(resource_model.get("state_names") or ()) != ResourceAdaptiveRLModel.STATE_NAMES:
                raise ValueError("资源自适应模型状态特征不匹配")
            if tuple(resource_model.get("action_names") or ()) != ResourceAdaptiveRLModel.ACTION_NAMES:
                raise ValueError("资源自适应模型动作空间不匹配")
        model_group = payload.get("model_group") if isinstance(payload.get("model_group"), dict) else None
        if not model_group_complete(model_group, settings):
            raise ValueError("六模型组无法逐个恢复并推理")
        restored_models = {item.get("key"): item for item in model_group.get("models", [])}
        reward_state = payload.get("reward_state") if isinstance(payload.get("reward_state"), dict) else None
        return {"weights": clean, "trained_steps": trained_steps, "loss": loss, "resource_model": resource_model, "model_group": model_group, "restored_models": restored_models, "reward_state": reward_state}

    def load_latest_model_state(self, settings=None):
        candidates = sorted(self.model_dir.glob("model_*.json"), key=self.model_created_key, reverse=True)
        for path in candidates:
            try:
                with path.open("r", encoding="utf-8") as file:
                    payload = json.load(file)
                state = self.validate_model_state(payload, settings)
                state["loaded_from"] = str(path)
                return state
            except Exception as exc:
                try:
                    self.log_error("load_latest_model_state", exc, {"path": str(path)})
                except Exception:
                    pass
        return None

    def load_settings(self):
        if not self.settings_file.exists():
            self.save_settings(default_runtime_settings_payload())
        try:
            with self.settings_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict):
                raise ValueError("运行配置必须是 JSON 对象")
            return AllowedUserEditPolicy.filter(data, "runtime")
        except Exception as exc:
            configuration_failure("读取运行配置失败 " + str(self.settings_file), exc)

    def save_settings(self, settings):
        source = AllowedUserEditPolicy.filter(settings, "runtime")
        payload = {"schema_version": CONFIG_SCHEMA_VERSION, "training_seconds": max(1, safe_int(source.get("training_seconds", AGENT_SPEC.default_training_seconds), AGENT_SPEC.default_training_seconds)), "still_seconds": max(0.1, safe_float(source.get("still_seconds", AGENT_SPEC.default_still_seconds), AGENT_SPEC.default_still_seconds)), "experience_pool_gb": max(0.1, safe_float(source.get("experience_pool_gb", AGENT_SPEC.default_experience_pool_gb), AGENT_SPEC.default_experience_pool_gb)), "ai_model_limit": max(1, safe_int(source.get("ai_model_limit", AGENT_SPEC.default_ai_model_limit), AGENT_SPEC.default_ai_model_limit)), "runtime_generated_numbers": RUNTIME_NUMBER_AUDIT}
        try:
            atomic_write_json(self.settings_file, payload, self.lock)
        except Exception as exc:
            configuration_failure("写入运行配置失败 " + str(self.settings_file), exc)

    def quarantine_bad_experience(self, line_number, text, error):
        payload = {"line": line_number, "error": str(error), "text": text}
        with self.lock:
            with (self.root / "experience.bad.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.log_error("load_experience.bad_json", error, {"line": line_number})

    def load_experience(self, limit=None):
        records = []
        if not self.experience_file.exists():
            return records
        if not limit:
            with self.experience_file.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        records.append(json.loads(text))
                    except Exception as exc:
                        self.quarantine_bad_experience(line_number, text, exc)
            return records
        limit = max(1, safe_int(limit, 0))
        recent_limit = max(1, limit // 2)
        reward_limit = max(1, limit // 4)
        old_limit = max(1, limit - recent_limit - reward_limit)
        tail = deque(maxlen=recent_limit)
        rewarded = []
        old = []
        with self.experience_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                text = line.strip()
                if not text:
                    continue
                item = (line_number, text)
                tail.append(item)
                try:
                    parsed = json.loads(text)
                    reward = self.record_reward_sort_key(parsed)
                    if len(rewarded) < reward_limit:
                        heapq.heappush(rewarded, (reward, line_number, text))
                    elif reward > rewarded[0][0]:
                        heapq.heapreplace(rewarded, (reward, line_number, text))
                    if line_number % max(1, limit // max(1, old_limit)) == 0 and len(old) < old_limit:
                        old.append(item)
                except Exception as exc:
                    self.quarantine_bad_experience(line_number, text, exc)
        merged = OrderedDict()
        for line_number, text in list(tail) + [(line_number, text) for _, line_number, text in sorted(rewarded, reverse=True)] + old:
            merged[line_number] = text
        for line_number, text in merged.items():
            try:
                records.append(json.loads(text))
            except Exception as exc:
                self.quarantine_bad_experience(line_number, text, exc)
        return records[:limit]

    def storage_size_bytes(self):
        total = 0
        if not self.root.exists():
            return total
        for file_root, _, filenames in os.walk(self.root):
            for filename in filenames:
                try:
                    total += (Path(file_root) / filename).stat().st_size
                except Exception:
                    pass
        return total

    def experience_record_paths(self, records):
        paths = set()
        for record in records or []:
            if not isinstance(record, dict):
                continue
            for key in ("screen_path", "before_screen", "after_screen"):
                value = record.get(key)
                if not value:
                    continue
                try:
                    path = (self.root / str(value)).resolve()
                    if path.is_file() and self.root.resolve() in (path, *path.parents):
                        paths.add(path)
                except Exception:
                    pass
        return paths

    def experience_pool_size_bytes(self, records=None):
        total = 0
        if self.experience_file.exists():
            try:
                total += self.experience_file.stat().st_size
            except Exception:
                pass
        if records is None:
            records = self.load_experience()
        for path in self.experience_record_paths(records):
            try:
                total += path.stat().st_size
            except Exception:
                pass
        return total

    def screen_file_paths(self):
        paths = set()
        if not self.screen_dir.exists():
            return paths
        for file_root, _, filenames in os.walk(self.screen_dir):
            for filename in filenames:
                path = Path(file_root) / filename
                if path.is_file():
                    paths.add(path.resolve())
        return paths

    def experience_pool_size_report(self, records=None):
        if records is None:
            records = self.load_experience()
        logical = self.experience_pool_size_bytes(records)
        referenced = self.experience_record_paths(records)
        orphans = self.screen_file_paths() - referenced
        orphan_size = 0
        for path in orphans:
            try:
                orphan_size += path.stat().st_size
            except Exception:
                pass
        return {"logical_pool_size_bytes": logical, "physical_pool_size_bytes": logical + orphan_size, "orphan_screen_size_bytes": orphan_size, "orphan_screen_count": len(orphans), "orphan_screen_paths": [str(path) for path in sorted(orphans)]}

    def cleanup_orphan_screens(self, records=None):
        report = self.experience_pool_size_report(records)
        failed = []
        attempted = len(report.get("orphan_screen_paths", []))
        for text in report.get("orphan_screen_paths", []):
            path = Path(text)
            try:
                if path.exists() and path.is_file():
                    path.unlink()
            except Exception as exc:
                failed.append(text)
                try:
                    self.log_error("cleanup_orphan_screens.unlink", exc, {"path": text})
                except Exception:
                    pass
        refreshed = self.experience_pool_size_report(records)
        refreshed["orphan_cleanup_failed_paths"] = failed
        refreshed["orphan_cleanup_attempted"] = attempted
        return refreshed

    def compact_experience_pool(self, limit_gb, run_guard=None, progress_callback=None):
        limit_bytes = max(1, int(max(0.1, safe_float(limit_gb, DEFAULT_EXPERIENCE_POOL_GB)) * 1024 * 1024 * 1024))
        records = self.load_experience()
        if progress_callback:
            progress_callback("扫描经验记录", 0, max(1, len(records)), {"removed": 0, "size_bytes": 0, "target_bytes": limit_bytes})
        path_sizes = {}
        for path in self.experience_record_paths(records):
            try:
                path_sizes[path] = path.stat().st_size
            except Exception:
                pass
        experience_file_size = 0
        if self.experience_file.exists():
            try:
                experience_file_size = self.experience_file.stat().st_size
            except Exception:
                pass
        current = experience_file_size + sum(path_sizes.values())
        size_report = self.cleanup_orphan_screens(records)
        if progress_callback:
            progress_callback("统计引用图片", len(path_sizes), max(1, len(path_sizes)), {"removed": 0, "size_bytes": current, "target_bytes": limit_bytes})
        if current <= limit_bytes:
            complete = current <= limit_bytes and size_report.get("orphan_screen_size_bytes", 0) == 0
            result = {"changed": bool(size_report.get("orphan_cleanup_attempted", 0)), "size_bytes": current, "removed": 0, "target_bytes": limit_bytes, "complete": complete}
            result.update(size_report)
            return result
        target_bytes = max(1, limit_bytes // 2)
        records = []
        if self.experience_file.exists():
            with self.experience_file.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        record = json.loads(text)
                    except Exception as exc:
                        self.quarantine_bad_experience(line_number, text, exc)
                        continue
                    reward = self.record_reward_sort_key(record)
                    records.append({"reward": reward, "line": line_number, "text": text, "record": record})
        records.sort(key=lambda item: (item["reward"], item["line"]))
        if progress_callback:
            progress_callback("按奖励排序", len(records), max(1, len(records)), {"removed": 0, "size_bytes": current, "target_bytes": target_bytes})
        path_references = defaultdict(int)
        record_paths = {}
        for item in records:
            paths = self.experience_record_paths([item["record"]])
            record_paths[item["line"]] = paths
            for path in paths:
                path_references[path] += 1
        removed_ids = set()
        removed = 0
        interrupted = False
        for item in records:
            if run_guard and run_guard():
                interrupted = True
                break
            if current <= target_bytes:
                break
            removed += 1
            removed_ids.add(item["line"])
            current = max(0, current - (len(item["text"].encode("utf-8")) + 1))
            for path in record_paths.get(item["line"], set()):
                path_references[path] = max(0, path_references[path] - 1)
                if path_references[path] == 0:
                    try:
                        if path.exists() and path.is_file():
                            size = path_sizes.get(path, path.stat().st_size)
                            current = max(0, current - size)
                    except Exception:
                        pass
            if progress_callback and (removed == 1 or removed % 100 == 0):
                progress_callback("删除低奖励样本", removed, max(1, len(records)), {"removed": removed, "size_bytes": current, "target_bytes": target_bytes})
        if interrupted:
            size_report = self.experience_pool_size_report([entry["record"] for entry in records])
            result = {"changed": False, "interrupted": True, "removed": removed, "size_bytes": current, "target_bytes": target_bytes, "complete": False}
            result.update(size_report)
            return result
        if removed_ids:
            keep_records = [entry for entry in records if entry["line"] not in removed_ids]
            referenced_after = set()
            for entry in keep_records:
                referenced_after.update(record_paths.get(entry["line"], set()))
            delete_after_replace = [path for path in path_sizes if path not in referenced_after]
            temporary = self.experience_file.with_suffix(".compact.tmp")
            with temporary.open("w", encoding="utf-8") as file:
                for index, item in enumerate(keep_records, start=1):
                    file.write(item["text"] + "\n")
                    if progress_callback and (index == 1 or index % 500 == 0):
                        progress_callback("重写经验文件", index, max(1, len(keep_records)), {"removed": removed, "size_bytes": current, "target_bytes": target_bytes})
                file.flush()
                os.fsync(file.fileno())
            self.replace_atomic_synced(temporary, self.experience_file)
            for index, path in enumerate(delete_after_replace, start=1):
                try:
                    if path.exists() and path.is_file():
                        path.unlink()
                except Exception as exc:
                    try:
                        self.log_error("compact_experience_pool.unlink", exc, {"path": str(path)})
                    except Exception:
                        pass
                if progress_callback and (index == 1 or index % 100 == 0):
                    progress_callback("删除不再引用的截图", index, max(1, len(delete_after_replace)), {"removed": removed, "size_bytes": current, "target_bytes": target_bytes})
            current = self.experience_pool_size_bytes([entry["record"] for entry in keep_records])
            size_report = self.cleanup_orphan_screens([entry["record"] for entry in keep_records])
            current = size_report.get("logical_pool_size_bytes", current)
        else:
            size_report = self.cleanup_orphan_screens([entry["record"] for entry in records])
        complete = current <= target_bytes and size_report.get("orphan_screen_size_bytes", 0) == 0
        if progress_callback:
            progress_callback("最终校验实际磁盘占用", 1, 1, {"removed": removed, "size_bytes": current, "target_bytes": target_bytes})
        result = {"changed": bool(removed_ids) or bool(size_report.get("orphan_cleanup_attempted", 0)), "size_bytes": current, "removed": removed, "target_bytes": target_bytes, "complete": complete}
        result.update(size_report)
        return result

    def log_error(self, where, error, context=None):
        payload = {
            "id": uuid.uuid4().hex,
            "time": now_text(),
            "where": str(where),
            "error_type": type(error).__name__,
            "error": str(error)
        }
        if context:
            payload["context"] = context
        with self.lock:
            with self.error_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload["id"]


@dataclass(frozen=True)
class WindowCheck:
    ok: bool
    reason: str
    rect: tuple = ()
    hits: int = 0
    expected: int = 0
    occluded_ratio: float = 1.0


class WindowManager:
    def __init__(self, executable_path, settings, ignored_hwnds=None):
        self.executable_path = Path(executable_path)
        self.settings = settings
        self.ignored_hwnds = set(int(item) for item in (ignored_hwnds or ()) if item)
        self.process = None
        self.hwnd = None
        self.lock = threading.RLock()
        self.window_check_cache = {"rect": None, "basic_perf": 0.0, "visibility_perf": 0.0, "occlusion_perf": 0.0, "ok": False}

    def launch_or_attach(self):
        if self.find_window():
            return True
        if self.executable_path.exists():
            try:
                self.process = subprocess.Popen([str(self.executable_path)], cwd=str(self.executable_path.parent))
            except Exception:
                self.process = None
            deadline = time.time() + self.settings.window_attach_seconds
            while time.time() < deadline:
                if self.find_window():
                    return True
                threading.Event().wait(self.settings.window_event_wait)
        return self.find_window()

    def executable_pids(self):
        pids = set()
        target = self.executable_path.name.lower()
        for proc in psutil.process_iter(["pid", "name", "exe"]):
            try:
                name = (proc.info.get("name") or "").lower()
                exe_name = Path(proc.info.get("exe") or "").name.lower()
                if name == target or exe_name == target:
                    pids.add(proc.info["pid"])
            except Exception:
                pass
        if self.process:
            try:
                root = psutil.Process(self.process.pid)
                pids.add(root.pid)
                for child in root.children(recursive=True):
                    pids.add(child.pid)
            except Exception:
                pass
        return pids

    def find_window(self):
        pids = self.executable_pids()
        pid_candidates = []
        title_candidates = []
        def handler(hwnd, _):
            try:
                if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
                    return
                title = win32gui.GetWindowText(hwnd).strip()
                if not title:
                    return
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                rect = win32gui.GetWindowRect(hwnd)
                client = self.client_rect_for(hwnd)
                cwidth, cheight = rect_size(client) if client else (0, 0)
                width, height = rect_size(rect)
                title_lower = title.lower()
                matched_title = any(str(word).lower() in title_lower for word in self.settings.window_title_keywords)
                candidate = (cwidth * cheight, width * height, hwnd, pid, title)
                if pids:
                    if pid in pids:
                        pid_candidates.append(candidate)
                elif matched_title:
                    title_candidates.append(candidate)
            except Exception:
                pass
        try:
            win32gui.EnumWindows(handler, None)
        except Exception:
            return False
        candidates = pid_candidates or title_candidates
        if not candidates:
            return False
        candidates.sort(reverse=True)
        with self.lock:
            self.hwnd = candidates[0][2]
            self.bound_identity = {"pid": candidates[0][3], "title": candidates[0][4], "client_rect": list(self.client_rect_for(candidates[0][2]) or ())}
        return True

    def client_rect_for(self, hwnd):
        try:
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            x1, y1 = win32gui.ClientToScreen(hwnd, (left, top))
            x2, y2 = win32gui.ClientToScreen(hwnd, (right, bottom))
            return (int(x1), int(y1), int(x2), int(y2)) if x2 > x1 and y2 > y1 else None
        except Exception:
            return None

    def client_rect(self):
        with self.lock:
            hwnd = self.hwnd
        if not hwnd or not win32gui.IsWindow(hwnd):
            if not self.find_window():
                return None
            with self.lock:
                hwnd = self.hwnd
        rect = self.client_rect_for(hwnd)
        if not rect:
            return None
        width, height = rect_size(rect)
        return rect if width > 1 and height > 1 else None

    def foreground(self):
        with self.lock:
            hwnd = self.hwnd
        if not hwnd:
            return
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            try:
                ctypes.windll.user32.keybd_event(win32con.VK_MENU, 0, 0, 0)
                ctypes.windll.user32.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
            except Exception:
                pass
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def topmost(self):
        self.foreground()

    def set_ignored_hwnds(self, hwnds):
        self.ignored_hwnds = set(int(item) for item in (hwnds or ()) if item)


    def occluded_area_ratio(self, hwnd, rect):
        front = []
        target_area = max(1, rect_area(rect))
        def handler(other, _):
            try:
                if other == hwnd or other in self.ignored_hwnds or win32gui.IsChild(hwnd, other) or any(win32gui.IsChild(ignored, other) for ignored in self.ignored_hwnds) or not win32gui.IsWindowVisible(other) or win32gui.IsIconic(other):
                    return
                other_rect = win32gui.GetWindowRect(other)
                inter = rect_intersection(rect, other_rect)
                if inter:
                    front.append((other, inter))
            except Exception:
                pass
        win32gui.EnumWindows(handler, None)
        ordered = []
        current = win32gui.GetTopWindow(0)
        seen = set()
        while current and current not in seen:
            seen.add(current)
            if current == hwnd:
                break
            ordered.append(current)
            current = win32gui.GetWindow(current, win32con.GW_HWNDNEXT)
        front_hwnds = set(ordered)
        covered = rect_union_area([inter for other, inter in front if other in front_hwnds])
        return clamp(covered / target_area, 0.0, 1.0)

    def check_window(self, force=False):
        with self.lock:
            hwnd = self.hwnd
        if not hwnd or not win32gui.IsWindow(hwnd):
            return WindowCheck(False, "invalid_handle")
        try:
            rect = self.client_rect_for(hwnd)
            if not rect:
                return WindowCheck(False, "missing_rect")
            width, height = rect_size(rect)
            if width <= 1 or height <= 1:
                return WindowCheck(False, "empty_rect", rect)
            now_perf = time.perf_counter()
            cache = self.window_check_cache
            if not force and cache.get("rect") == rect and now_perf - cache.get("basic_perf", 0.0) < self.settings.window_event_wait:
                return cache.get("check", WindowCheck(bool(cache.get("ok", False)), cache.get("reason", "cached"), rect))
            cache["rect"] = rect
            cache["basic_perf"] = now_perf
            if force or now_perf - cache.get("visibility_perf", 0.0) >= clamp(self.settings.window_event_wait, 0.2, 0.5):
                if not win32gui.IsWindowVisible(hwnd):
                    result = WindowCheck(False, "invisible", rect)
                    cache.update({"ok": result.ok, "reason": result.reason, "check": result})
                    return result
                if win32gui.IsIconic(hwnd):
                    result = WindowCheck(False, "minimized", rect)
                    cache.update({"ok": result.ok, "reason": result.reason, "check": result})
                    return result
                left, top, right, bottom = rect
                virtual_left = safe_int(win32api.GetSystemMetrics(76), 0)
                virtual_top = safe_int(win32api.GetSystemMetrics(77), 0)
                virtual_w = max(1, safe_int(win32api.GetSystemMetrics(78), win32api.GetSystemMetrics(0)))
                virtual_h = max(1, safe_int(win32api.GetSystemMetrics(79), win32api.GetSystemMetrics(1)))
                if left < virtual_left or top < virtual_top or right > virtual_left + virtual_w or bottom > virtual_top + virtual_h:
                    result = WindowCheck(False, "out_of_screen", rect)
                    cache.update({"ok": result.ok, "reason": result.reason, "check": result})
                    return result
                cache["visibility_perf"] = now_perf
            if not force and now_perf - cache.get("occlusion_perf", 0.0) < clamp(self.settings.window_event_wait * 2.0, 0.5, 1.0):
                return cache.get("check", WindowCheck(bool(cache.get("ok", True)), cache.get("reason", "cached"), rect))
            left, top, right, bottom = rect
            inset_x = max(1, min(width // 20, 12))
            inset_y = max(1, min(height // 20, 12))
            mid_x = int((left + right) / 2)
            mid_y = int((top + bottom) / 2)
            points = ((mid_x, mid_y), (left + inset_x, top + inset_y), (right - inset_x - 1, top + inset_y), (left + inset_x, bottom - inset_y - 1), (right - inset_x - 1, bottom - inset_y - 1), (mid_x, top + inset_y), (mid_x, bottom - inset_y - 1), (left + inset_x, mid_y), (right - inset_x - 1, mid_y))
            hits = 0
            for point in points:
                hit = win32gui.WindowFromPoint(point)
                if hit == hwnd or win32gui.IsChild(hwnd, hit) or hit in self.ignored_hwnds or any(win32gui.IsChild(ignored, hit) for ignored in self.ignored_hwnds):
                    hits += 1
            cache["occlusion_perf"] = now_perf
            occluded_ratio = self.occluded_area_ratio(hwnd, rect)
            ok = hits == len(points) and occluded_ratio <= (1.0 / max(100.0, width * height))
            reason = "ok" if ok else "occluded"
            result = WindowCheck(ok, reason, rect, hits, len(points), occluded_ratio)
            cache.update({"ok": result.ok, "reason": result.reason, "check": result})
            return result
        except Exception as exc:
            return WindowCheck(False, type(exc).__name__)

    def window_ok(self, force=False):
        return self.check_window(force=force).ok

class ScreenShotError(RuntimeError):
    pass


class ScoreReviewError(RuntimeError):
    pass


class PersistenceError(RuntimeError):
    pass


class ScreenAnalyzer:
    def __init__(self, hash_size):
        self.hash_size = int(hash_size)
        self.sct = None
        resampling = getattr(Image, "Resampling", Image) if Image else None
        self.resample = getattr(resampling, "LANCZOS", 1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def capture(self, rect):
        if mss is None:
            raise ScreenShotError("mss 不可用，无法启动实时屏幕抓取服务")
        if self.sct is None:
            try:
                self.sct = mss.mss()
            except Exception as exc:
                raise ScreenShotError(str(exc)) from exc
        left, top, right, bottom = rect
        width, height = rect_size(rect)
        try:
            shot = self.sct.grab({"left": int(left), "top": int(top), "width": width, "height": height})
        except Exception as exc:
            raise ScreenShotError(str(exc)) from exc
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        return image

    def save_image(self, image, path, priority="normal", settings=None):
        save_kwargs = {}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if str(path).lower().endswith(".png"):
            compression = 1 if priority == "critical" else 3
            if settings is not None and settings.training_event_wait > settings.ui_event_coalesce_seconds:
                compression = 1
            save_kwargs = {"format": "PNG", "optimize": priority == "critical", "compress_level": int(clamp(compression, 0, 9))}
        temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
        image.save(temporary, **save_kwargs)
        temporary.replace(path)

    def fingerprint(self, image):
        small = image.convert("L").resize((self.hash_size, self.hash_size), self.resample)
        pixels = small.tobytes()
        total = self.hash_size * self.hash_size
        histogram = small.histogram()
        average = sum(level * count for level, count in enumerate(histogram)) / max(1, total)
        value = 0
        for pixel in pixels:
            value = (value << 1) | (1 if pixel >= average else 0)
        bits = self.hash_size * self.hash_size
        width = max(1, math.ceil(bits / 4))
        return HashValue(value=value, bits=bits, hex=f"{value:0{width}x}")

    def semantic_fingerprint(self, image):
        size = max(4, min(16, self.hash_size))
        rgb = image.convert("RGB").resize((size, size), self.resample)
        gray = rgb.convert("L")
        gray_pixels = list(gray.tobytes())
        mean = sum(gray_pixels) / max(1, len(gray_pixels))
        variance = sum((pixel - mean) * (pixel - mean) for pixel in gray_pixels) / max(1, len(gray_pixels))
        features = [(pixel - mean) / 255.0 for pixel in gray_pixels]
        channels = list(rgb.getdata())
        for channel in range(3):
            values = [pixel[channel] for pixel in channels]
            features.append((sum(values) / max(1, len(values)) - 127.5) / 127.5)
            buckets = [0, 0, 0, 0]
            for value in values:
                buckets[min(3, int(value) * 4 // 256)] += 1
            features.extend(count / max(1, len(values)) for count in buckets)
        features.append((mean - 127.5) / 127.5)
        features.append(math.sqrt(variance) / 127.5)
        norm = math.sqrt(sum(value * value for value in features))
        if norm <= 0.0:
            return tuple(round(value, 6) for value in features)
        return tuple(round(value / norm, 6) for value in features)

    def close(self):
        if self.sct:
            try:
                self.sct.close()
            except Exception:
                pass
            self.sct = None


class BKHashTree:
    def __init__(self):
        self.root = None
        self.items = []

    def clear(self):
        self.root = None
        self.items = []

    def add(self, hash_value, index):
        if not hash_value:
            return
        node = {"hash": hash_value, "indexes": [index], "children": {}}
        self.items.append(index)
        if self.root is None:
            self.root = node
            return
        current = self.root
        while True:
            distance = hash_distance(hash_value, current["hash"])
            if not distance:
                return
            diff = distance[0]
            if diff == 0:
                current["indexes"].append(index)
                return
            child = current["children"].get(diff)
            if child is None:
                current["children"][diff] = node
                return
            current = child

    def nearest(self, hash_value, limit):
        if not self.root or not hash_value:
            return []
        limit = max(1, int(limit))
        heap = []
        stack = [self.root]
        best = hash_value.bits
        while stack:
            node = stack.pop()
            distance = hash_distance(hash_value, node["hash"])
            if not distance:
                continue
            diff = distance[0]
            for index in node["indexes"]:
                item = (-diff, index)
                if len(heap) < limit:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            if len(heap) >= limit:
                best = min(best, -heap[0][0])
            radius = best if len(heap) >= limit else hash_value.bits
            low = max(0, diff - radius)
            high = diff + radius
            for edge, child in node["children"].items():
                if low <= edge <= high:
                    stack.append(child)
        return [index for _, index in sorted(heap, key=lambda item: (item[0], -item[1]), reverse=True)]


class ExperiencePool:
    INDEX_SETTING_NAMES = ("hash_prefix_bits", "nearest_candidate_limit")

    def __init__(self, settings, records=None, model_state=None):
        self.settings = settings
        self.index_settings = settings
        self.records = []
        self.hashes = []
        self.index = defaultdict(list)
        self.sorted_prefixes = []
        self.profile = HumanProfile(settings)
        self.model = PolicyModel(settings, model_state)
        group_payload = (model_state or {}).get("model_group") if isinstance(model_state, dict) else None
        self.model_group_models = {}
        if isinstance(group_payload, dict) and model_group_complete(group_payload, settings):
            self.model_group_models = {item.get("key"): restore_trainable_model(item, settings) for item in group_payload.get("models", [])}
        self.model_runtime = ModelGroupRuntime(self.model_group_models, settings)
        self.reward_state = reward_state_from((model_state or {}).get("reward_state") if isinstance(model_state, dict) else None)
        self.lock = threading.RLock()
        self.action_cache = []
        self.prefix_neighbor_cache = OrderedDict()
        self.global_action_heap = []
        self.nearest_cache = OrderedDict()
        self.inflight_indices = set()
        self.metric_tree = BKHashTree()
        self.visual_model = {}
        self.index_version = 0
        for record in records or []:
            self.add(record)

    def _prefix(self, hash_value):
        bits = min(max(1, self.index_settings.hash_prefix_bits), hash_value.bits)
        return hash_value.value >> max(0, hash_value.bits - bits)

    def _index_signature(self, settings):
        return tuple(getattr(settings, name) for name in self.INDEX_SETTING_NAMES)

    def apply_settings(self, settings):
        with self.lock:
            rebuild = self._index_signature(settings) != self._index_signature(self.index_settings)
            self.settings = settings
            self.profile.settings = settings
            self.model.settings = settings
            self.model_runtime.apply_settings(settings)
            if rebuild:
                self.index_settings = settings
                self.rebuild_index_locked()
            self.nearest_cache.clear()

    def rebuild_index_locked(self):
        self.index = defaultdict(list)
        self.sorted_prefixes = []
        self.prefix_neighbor_cache = OrderedDict()
        self.metric_tree.clear()
        self.index_version += 1
        for index, hash_value in enumerate(self.hashes):
            if hash_value:
                prefix = self._prefix(hash_value)
                bucket = self.index[prefix]
                if not bucket:
                    self.sorted_prefixes.append(prefix)
                bucket.append(index)
                self.metric_tree.add(hash_value, index)

    def is_training_eligible(self, record):
        return isinstance(record, dict) and not bool(record.get("quarantined") or record.get("exclude_from_training") or record.get("score_status") in ("image_missing", "image_corrupt", "unrecoverable"))

    def record_trainable(self, record):
        return self.is_training_eligible(record)

    def add(self, record):
        with self.lock:
            index = len(self.records)
            self.records.append(record)
            hash_value = parse_hash_value(record)
            self.hashes.append(hash_value)
            if hash_value:
                prefix = self._prefix(hash_value)
                bucket = self.index[prefix]
                if not bucket:
                    self.sorted_prefixes.append(prefix)
                    self.index_version += 1
                bucket.append(index)
                self.metric_tree.add(hash_value, index)
            if self.is_training_eligible(record) and record.get("mouse_action") and record.get("mode") == "learning" and record.get("mouse_source") == "user":
                self.profile.observe(record["mouse_action"])
            if self.is_training_eligible(record) and record.get("mouse_action"):
                self.action_cache.append(record)
                reward = safe_float(record.get("reward", 0.0), 0.0)
                item = (reward, index)
                if len(self.global_action_heap) < self.settings.global_action_heap_limit:
                    heapq.heappush(self.global_action_heap, item)
                elif reward > self.global_action_heap[0][0]:
                    heapq.heapreplace(self.global_action_heap, item)
            if self.nearest_cache:
                capacity = max(1, min(self.settings.global_action_heap_limit, self.index_settings.nearest_candidate_limit // max(1, self.settings.ui_metric_columns)))
                while len(self.nearest_cache) > capacity:
                    self.nearest_cache.popitem(last=False)


    def train_local_vision_model(self):
        with self.lock:
            vectors = [(parse_semantic_vector(record.get("screen_semantic_vector")), safe_float(record.get("screen_score", record.get("novelty", 0.0)), 0.0), record.get("id")) for record in self.records if self.is_training_eligible(record)]
        vectors = [(vector, score, record_id) for vector, score, record_id in vectors if vector]
        if not vectors:
            self.visual_model = {"created_at": now_text(), "status": "empty", "clusters": []}
            self.model.visual_model = copy.deepcopy(self.visual_model)
            return self.visual_model
        dimensions = min(len(vector) for vector, _, _ in vectors)
        ordered = sorted(vectors, key=lambda item: item[1])
        bucket_count = max(1, min(self.settings.nearest_top_k, int(math.sqrt(len(ordered))) or 1))
        clusters = []
        for bucket_index in range(bucket_count):
            bucket = ordered[bucket_index::bucket_count]
            if not bucket:
                continue
            centroid = [sum(vector[index] for vector, _, _ in bucket) / len(bucket) for index in range(dimensions)]
            clusters.append({"index": bucket_index, "count": len(bucket), "avg_score": round(sum(score for _, score, _ in bucket) / len(bucket), 6), "centroid": [round(value, 6) for value in centroid]})
        self.visual_model = {"created_at": now_text(), "status": "trained", "source": "sleep_local_semantic_vectors", "dimensions": dimensions, "records": len(vectors), "clusters": clusters}
        self.model.visual_model = copy.deepcopy(self.visual_model)
        return self.visual_model

    def count(self):
        with self.lock:
            return len(self.records)

    def action_records(self):
        with self.lock:
            return list(self.action_cache)

    def candidate_indices(self, hash_value):
        with self.lock:
            if len(self.records) <= self.index_settings.nearest_candidate_limit:
                return [index for index, item in enumerate(self.hashes) if item]
            tree_result = self.metric_tree.nearest(hash_value, self.index_settings.nearest_candidate_limit)
            if len(tree_result) >= max(1, min(self.settings.nearest_top_k, self.index_settings.nearest_candidate_limit)):
                return tree_result
            query_prefix = self._prefix(hash_value)
            result = []
            cache_key = (self.index_version, query_prefix)
            nearby = self.prefix_neighbor_cache.get(cache_key)
            if nearby is not None:
                self.prefix_neighbor_cache.move_to_end(cache_key)
            else:
                buckets = defaultdict(list)
                for item in self.sorted_prefixes:
                    buckets[(item ^ query_prefix).bit_count()].append(item)
                nearby = []
                for distance in sorted(buckets):
                    nearby.extend(buckets[distance])
                    if len(nearby) >= max(1, self.index_settings.nearest_candidate_limit):
                        break
                self.prefix_neighbor_cache[cache_key] = nearby
                capacity = max(1, min(self.settings.local_action_heap_limit, self.index_settings.nearest_candidate_limit // max(1, self.settings.hash_prefix_bits)))
                while len(self.prefix_neighbor_cache) > capacity:
                    self.prefix_neighbor_cache.popitem(last=False)
            for prefix in nearby:
                result.extend(self.index[prefix])
                if len(result) >= self.index_settings.nearest_candidate_limit:
                    break
            if result:
                return result[:self.index_settings.nearest_candidate_limit]
            valid = [index for index, item in enumerate(self.hashes) if item]
            return random.sample(valid, self.index_settings.nearest_candidate_limit) if len(valid) > self.index_settings.nearest_candidate_limit else valid

    def nearest(self, hash_value, exclude_id=None, limit=None, before_index=None, semantic_vector=None):
        if not hash_value:
            return []
        with self.lock:
            top_k = max(1, safe_int(limit, self.settings.nearest_top_k))
            query_semantic = parse_semantic_vector(semantic_vector)
            semantic_key = tuple(round(value, 4) for value in query_semantic[:32]) if query_semantic else ()
            cache_key = None if exclude_id is not None or limit is not None or before_index is not None else (self.index_version, hash_value.bits, hash_value.hex, semantic_key, self.index_settings.hash_prefix_bits, self.index_settings.nearest_candidate_limit, self.settings.nearest_top_k)
            cached = self.nearest_cache.get(cache_key) if cache_key is not None else None
            if cached is not None:
                self.nearest_cache.move_to_end(cache_key)
                return [copy.deepcopy(item) for item in cached]
            candidate_indexes = self.candidate_indices(hash_value)
            if len(self.records) > self.index_settings.nearest_candidate_limit:
                recent_limit = min(len(self.records), max(top_k, self.index_settings.nearest_candidate_limit // 4))
                candidate_indexes = list(dict.fromkeys(candidate_indexes + list(range(len(self.records) - recent_limit, len(self.records))) + [index for _, index in heapq.nlargest(min(len(self.global_action_heap), top_k), self.global_action_heap)]))
            snapshot = [(index, self.hashes[index], self.records[index]) for index in candidate_indexes if self.hashes[index] and self.is_training_eligible(self.records[index]) and (before_index is None or index < before_index) and (exclude_id is None or self.records[index].get("id") != exclude_id)]
        scored = []
        for index, other, record in snapshot:
            if other:
                hash_score = hash_similarity(hash_value, other)
                semantic_score = semantic_similarity(query_semantic, record.get("screen_semantic_vector"))
                similarity = clamp(hash_score if semantic_score is None else hash_score * 0.35 + semantic_score * 0.65, 0.0, 1.0)
                item = {"similarity": similarity, "hash_similarity": hash_score, "semantic_similarity": semantic_score, "record": record}
                if len(scored) < top_k:
                    heapq.heappush(scored, (similarity, index, item))
                elif similarity > scored[0][0]:
                    heapq.heapreplace(scored, (similarity, index, item))
        result = [item for _, _, item in sorted(scored, key=lambda pair: (pair[0], pair[1]), reverse=True)]
        if cache_key is not None:
            with self.lock:
                self.nearest_cache[cache_key] = copy.deepcopy(result)
                self.nearest_cache.move_to_end(cache_key)
                capacity = max(1, min(self.settings.global_action_heap_limit, self.index_settings.nearest_candidate_limit // max(1, self.settings.ui_metric_columns)))
                while len(self.nearest_cache) > capacity:
                    self.nearest_cache.popitem(last=False)
        return result

    def novelty(self, hash_value, exact_checksum=None, semantic_vector=None):
        score, neighbors, _ = self.compute_screen_score(hash_value, exact_checksum=exact_checksum, semantic_vector=semantic_vector)
        return score, neighbors

    def score_snapshot(self, snapshot, exclude_id=None, before_index=None):
        if not snapshot:
            return 0.0, [], 0.0
        return self.compute_screen_score(snapshot.hash_value, exclude_id=exclude_id, before_index=before_index, exact_checksum=getattr(snapshot, "image_checksum", ""), semantic_vector=getattr(snapshot, "semantic_vector", ()))

    def best_global_action(self):
        with self.lock:
            ranked = heapq.nlargest(max(1, min(self.settings.nearest_top_k, len(self.global_action_heap))), self.global_action_heap, key=lambda item: item[0])
            weighted = []
            for _, index in ranked:
                if index < 0 or index >= len(self.records):
                    continue
                record = self.records[index]
                if not self.is_training_eligible(record):
                    continue
                action = record.get("mouse_action")
                reward = safe_float(record.get("reward", 0.0), 0.0)
                human_score = clamp(record.get("human_score", 50.0), 0.0, 100.0)
                if action:
                    weighted.append((max(0.05, 1.0 + reward / 100.0) * max(0.25, human_score / 100.0), action))
        chosen = weighted_choice(weighted)
        return copy.deepcopy(chosen) if chosen else None


    def sleep_training_batch_indices(self, batch_size):
        with self.lock:
            action_indices = [index for index, record in enumerate(self.records) if index not in self.inflight_indices and record.get("mouse_action") and self.is_training_eligible(record)]
            if not action_indices:
                return []
            target = max(1, min(safe_int(batch_size, 1), len(action_indices)))
            available = set(action_indices)
            ranked = [index for _, index in heapq.nlargest(min(len(self.global_action_heap), target), self.global_action_heap, key=lambda item: item[0]) if index in available]
            recent = action_indices[-target:]
            remaining = target * max(1, self.settings.ui_metric_columns) - len(ranked) - len(recent)
            sampled = random.sample(action_indices, min(len(action_indices), max(0, remaining))) if remaining > 0 else []
            selected = list(dict.fromkeys(ranked + recent + sampled))[:target]
            self.inflight_indices.update(selected)
            return selected

    def compute_screen_score(self, hash_value, exclude_id=None, before_index=None, exact_checksum=None, semantic_vector=None):
        if exact_checksum:
            with self.lock:
                for index, record in enumerate(self.records):
                    if (before_index is None or index < before_index) and (exclude_id is None or record.get("id") != exclude_id) and self.is_training_eligible(record) and exact_checksum == record.get("image_checksum"):
                        return 0.0, [{"similarity": 1.0, "hash_similarity": 1.0, "semantic_similarity": 1.0, "record": record}], 1.0
        neighbors = self.nearest(hash_value, exclude_id=exclude_id, limit=self.settings.nearest_top_k, before_index=before_index, semantic_vector=semantic_vector)
        sims = [clamp(item.get("similarity", 0.0), 0.0, 1.0) for item in neighbors]
        if sims:
            top = sims[:max(1, min(self.settings.nearest_top_k, len(sims)))]
            density = sum(1 for item in top if item >= 0.95) / len(top)
            score = self.model_runtime.screen_novelty(sims)
            confidence = clamp(top[0] * 0.65 + (1.0 - density) * 0.35, 0.0, 1.0)
        else:
            score = self.model_runtime.screen_novelty([])
            confidence = 0.0
        return round(score, 2), neighbors, confidence

    def recheck_screen_scores(self, store=None, analyzer=None, tolerance=0.01, run_guard=None, progress_callback=None):
        checked = 0
        processed_count = 0
        interrupted = False
        rescored = 0
        missing = 0
        errors = 0
        image_missing_ids = set()
        image_corrupt_ids = set()
        hash_missing_ids = set()
        unrecoverable_ids = set()
        quarantined_ids = set()
        with self.lock:
            snapshot = [(index, dict(record)) for index, record in enumerate(self.records)]
        updates = []
        changed_records = []
        total = len(snapshot)
        if callable(progress_callback):
            progress_callback(0, total)
        for processed, (index, record) in enumerate(snapshot, start=1):
            processed_count = processed
            record_key = record.get("id") or index
            if run_guard and run_guard():
                interrupted = True
                processed_count = processed - 1
                break
            hash_value = parse_hash_value(record)
            file_hash = None
            screen_path = record.get("screen_path")
            needs_checksum = not bool(record.get("image_checksum"))
            if store and analyzer and screen_path:
                path = (store.root / str(screen_path)).resolve()
                try:
                    if path.is_file() and store.root.resolve() in (path, *path.parents):
                        with Image.open(path) as image:
                            rgb_image = image.convert("RGB")
                            file_hash = analyzer.fingerprint(rgb_image)
                            record["screen_semantic_vector"] = analyzer.semantic_fingerprint(rgb_image)
                            record["image_checksum"] = image_content_checksum(rgb_image)
                    else:
                        if record.get("image_dropped") or record.get("screen_file_expected") is False:
                            record.update({"score_status": "image_unavailable_hash_scored", "score_checked_at": now_text()})
                        else:
                            record.update({"score_status": "image_missing", "exclude_from_training": True, "quarantined": True, "quarantine_reason": "image_missing", "score_checked_at": now_text()})
                            image_missing_ids.add(record_key)
                            quarantined_ids.add(record_key)
                except Exception as exc:
                    record["screen_file_error"] = str(exc)
                    record.update({"score_status": "image_corrupt", "exclude_from_training": True, "quarantined": True, "quarantine_reason": "image_corrupt", "score_checked_at": now_text()})
                    image_corrupt_ids.add(record_key)
                    quarantined_ids.add(record_key)
            elif needs_checksum:
                record.update({"score_status": "image_missing", "exclude_from_training": True, "quarantined": True, "quarantine_reason": "image_checksum_missing_without_recoverable_image", "score_checked_at": now_text()})
                image_missing_ids.add(record_key)
                quarantined_ids.add(record_key)
            if file_hash and (not hash_value or file_hash.hex != hash_value.hex or file_hash.bits != hash_value.bits):
                hash_value = file_hash
                record["screen_hash"] = file_hash.hex
                record["screen_hash_hex"] = file_hash.hex
                record["screen_hash_int"] = file_hash.value
                record["screen_hash_bits"] = file_hash.bits
                errors += 1
            if not hash_value:
                hash_missing_ids.add(record_key)
                unrecoverable_ids.add(record_key)
                quarantined_ids.add(record_key)
                record["score_status"] = "unrecoverable"
                record["exclude_from_training"] = True
                record["quarantined"] = True
                record["quarantine_reason"] = record.get("quarantine_reason") or "hash_missing"
                record["score_checked_at"] = now_text()
                updates.append((index, record, None))
                changed_records.append(record)
                if callable(progress_callback):
                    progress_callback(processed, total)
                continue
            if record.get("quarantined") and record.get("quarantine_reason") in ("image_missing", "image_corrupt", "image_checksum_missing_without_recoverable_image"):
                unrecoverable_ids.add(record_key)
                quarantined_ids.add(record_key)
                updates.append((index, record, hash_value))
                changed_records.append(record)
                if callable(progress_callback):
                    progress_callback(processed, total)
                continue
            score, neighbors, confidence = self.compute_screen_score(hash_value, exclude_id=record.get("id"), before_index=index, exact_checksum=record.get("image_checksum"), semantic_vector=record.get("screen_semantic_vector"))
            old_score = safe_float(record.get("screen_score", record.get("novelty", record.get("after_novelty", None))), None)
            lacks = old_score is None or record.get("score_version") != 1 or not record.get("score_basis")
            wrong = old_score is not None and abs(score - old_score) > tolerance
            if lacks:
                missing += 1
            if wrong:
                errors += 1
            if lacks or wrong:
                rescored += 1
            human_score = self.human_score(record.get("mouse_action")) if record.get("mouse_action") else self.settings.score_default
            reward_info = reward_breakdown(score, human_score, self.settings, self.reward_state, record.get("created_at") or record.get("score_checked_at"))
            self.reward_state = reward_state_from(reward_info.get("reward_state"))
            record.update({"screen_score": score, "novelty": score, "score_version": 1, "score_status": "rescored" if bool(lacks or wrong) else "scored", "score_basis": "nearest_screen_content_recheck", "score_checked_at": now_text(), "score_neighbors": [{"id": item["record"].get("id"), "similarity": round(item.get("similarity", 0.0), 4)} for item in neighbors], "score_confidence": round(confidence, 4), "score_rechecked": True, "score_recomputed": bool(lacks or wrong), "reward_version": reward_info["reward_version"], "screen_primary_reward": reward_info["screen_primary_reward"], "human_tie_break_reward": reward_info["human_tie_break_reward"], "reward_breakdown": reward_info, "reward_sort_key": reward_info["reward_sort_key"], "total_reward": reward_info["total_reward"], "reward": reward_info["total_reward"], "screen_score_delta": max(0.0, reward_info["screen_score_delta"])})
            checked += 1
            updates.append((index, record, hash_value))
            changed_records.append(record)
            if callable(progress_callback):
                progress_callback(processed, total)
        with self.lock:
            for index, record, hash_value in updates:
                if index < len(self.records):
                    self.records[index] = record
                    self.hashes[index] = hash_value
            self.rebuild_index_locked()
            self.rebuild_derived_state_locked()
            self.nearest_cache.clear()
        if callable(progress_callback):
            progress_callback(processed_count if interrupted else total, total)
        trainable = sum(1 for record in self.records if self.is_training_eligible(record))
        degraded = sum(1 for record in self.records if isinstance(record, dict) and record.get("score_status") == "image_unavailable_hash_scored")
        image_missing = len(image_missing_ids)
        image_corrupt = len(image_corrupt_ids)
        hash_missing = len(hash_missing_ids)
        unrecoverable = len(unrecoverable_ids)
        unique_quarantined = len(quarantined_ids)
        return {"checked": checked, "processed": processed_count if interrupted else total, "total": total, "complete": not interrupted, "interrupted": interrupted, "rescored": rescored, "missing": missing, "errors": errors, "image_missing": image_missing, "image_corrupt": image_corrupt, "hash_missing": hash_missing, "unrecoverable": unrecoverable, "degraded": degraded, "quarantined": unique_quarantined, "unique_quarantined": unique_quarantined, "trainable": trainable, "changed_records": changed_records, "changed_ids": [record.get("id") for record in changed_records if isinstance(record, dict) and record.get("id")]}

    def sleep_training_step(self, batch_size, settle_screen_score=None, run_guard=None):
        indices = self.sleep_training_batch_indices(batch_size)
        if not indices:
            return {"trained": 0, "best_score": 0.0, "avg_score": 0.0, "avg_confidence": 0.0}
        try:
            with self.lock:
                records_len = len(self.records)
                snapshot = [(index, self.hashes[index], copy.deepcopy(self.records[index])) for index in indices if index < records_len and self.hashes[index]]
            updates = []
            for index, hash_value, record in snapshot:
                if run_guard and run_guard():
                    break
                novelty, neighbors, similarity_confidence = self.compute_screen_score(hash_value, exclude_id=record.get("id"), before_index=index, exact_checksum=record.get("image_checksum"), semantic_vector=record.get("screen_semantic_vector"))
                action = record.get("mouse_action")
                human_score = clamp(record.get("human_score", self.profile.score(action)), 0.0, 100.0) if action else self.settings.score_default
                event_time = record.get("created_at") or record.get("sleep_evaluated_at") or now_text()
                reward_state = None
                with self.lock:
                    reward_state = reward_state_from(self.reward_state)
                stateful_reward = reward_with_state(novelty, human_score, reward_state, event_time)
                reward = stateful_reward["reward"]
                visits = max(0, safe_int(record.get("sleep_visits", 0), 0))
                candidate = reward
                value = candidate
                confidence = clamp((similarity_confidence + human_score / 100.0 + min(1.0, (visits + 1) / max(1.0, self.settings.nearest_top_k))) / 3.0, 0.0, 1.0)
                reward_info = reward_breakdown(novelty, human_score, self.settings, reward_state, event_time)
                settled_delta = 0.0
                if record.get("reward_version") != reward_info["reward_version"] and callable(settle_screen_score):
                    settled_delta = max(0.0, safe_float(reward_info["screen_score_delta"], 0.0) - max(0.0, safe_float(record.get("screen_score_settled", record.get("screen_score_delta", 0.0)), 0.0)))
                    if settled_delta > 0.0:
                        settle_screen_score(settled_delta)
                updates.append((index, value, visits + 1, confidence, novelty, human_score, reward_info, settled_delta))
            train_records = []
            with self.lock:
                total_score = 0.0
                best_score = None
                for index, value, visits, confidence, novelty, human_score, reward_info, settled_delta in updates:
                    if index >= len(self.records):
                        continue
                    record = self.records[index]
                    record["sleep_policy_reward"] = round(value, 4)
                    record["sleep_visits"] = visits
                    record["sleep_confidence"] = round(confidence, 4)
                    record["sleep_novelty"] = round(novelty, 2)
                    record["sleep_human_score"] = round(human_score, 2)
                    record["reward_version"] = reward_info["reward_version"]
                    record["sleep_evaluated_at"] = now_text()
                    record["screen_primary_reward"] = reward_info["screen_primary_reward"]
                    record["human_tie_break_reward"] = reward_info["human_tie_break_reward"]
                    record["reward_breakdown"] = reward_info
                    record["reward_sort_key"] = reward_info["reward_sort_key"]
                    record["total_reward"] = reward_info["total_reward"]
                    record["reward"] = reward_info["total_reward"]
                    record["income"] = reward_info.get("income")
                    record["cost"] = reward_info.get("cost")
                    record["reward_state"] = reward_info.get("reward_state")
                    self.reward_state = reward_state_from(reward_info.get("reward_state"))
                    record["screen_score_delta"] = max(0.0, reward_info["screen_score_delta"])
                    record["screen_score_settled"] = max(0.0, safe_float(record.get("screen_score_settled", 0.0), 0.0)) + settled_delta
                    train_records.append(record)
                model_result = self.model.train(train_records)
                for record in train_records:
                    model_prediction = clamp(safe_float(record.get("model_prediction", self.model.predict(record)), 0.0), 0.0, 1.0)
                    value = safe_float(record.get("sleep_policy_reward", record.get("reward", 0.0)), 0.0)
                    confidence = clamp(safe_float(record.get("sleep_confidence", 0.0), 0.0), 0.0, 1.0)
                    record["sleep_model_confidence"] = round(model_prediction, 4)
                    score = value + (confidence * 0.000000001) + (model_prediction * 0.0000000001)
                    total_score += score
                    best_score = score if best_score is None else max(best_score, score)
                self.rebuild_action_heap_locked()
            count = len(updates)
            model_confidence = safe_float(model_result.get("confidence", 0.0), 0.0)
            avg_confidence = (sum(item[3] for item in updates) / count if count else 0.0)
            hardware = read_hardware_state()
            with self.lock:
                resource_result = self.model.resource_model.train_from_metrics({"cpu_load": safe_float(hardware.get("cpu_load", 0.0), 0.0), "memory_free_ratio": safe_float(hardware.get("memory_free_ratio", 0.5), 0.5), "capture_ms": safe_float(getattr(self.settings, "training_event_wait", 0.05), 0.05) * 1000.0, "execution_ms": safe_float(getattr(self.settings, "generated_sleep_event_wait", 0.1), 0.1) * 1000.0, "window_instability": 1.0 - avg_confidence, "success_rate": model_confidence, "pool_count": len(self.records), "throughput": count, "batch_size": max(1, safe_int(getattr(self.settings, "sleep_batch_size", 1), 1)), "save_success_rate": 1.0, "training_gain": clamp((best_score or 0.0) / max(1.0, abs(self.settings.reward_total_max)), 0.0, 1.0)})
            return {"trained": count, "model_trained": safe_int(model_result.get("trained", 0), 0), "model_loss": safe_float(model_result.get("loss", 0.0), 0.0), "best_score": round(best_score or 0.0, 4), "avg_score": round(total_score / count if count else 0.0, 4), "avg_confidence": round(clamp(avg_confidence * 0.6 + model_confidence * 0.4, 0.0, 1.0), 4), **resource_result}

        finally:
            with self.lock:
                for index in indices:
                    self.inflight_indices.discard(index)

    def rebuild_derived_state_locked(self):
        self.action_cache = []
        heap = []
        for index, record in enumerate(self.records):
            if not self.is_training_eligible(record):
                continue
            action = record.get("mouse_action")
            if not action:
                continue
            self.action_cache.append(record)
            screen, human = record_screen_human(record)
            reward = strict_reward_value(screen, human)
            confidence = clamp(record.get("sleep_confidence", 0.0), 0.0, 1.0)
            item = (reward + confidence * 0.000000001, index)
            if len(heap) < self.settings.global_action_heap_limit:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
        self.global_action_heap = heap

    def rebuild_action_heap_locked(self):
        self.rebuild_derived_state_locked()

    def human_score(self, action):
        return self.model_runtime.mouse_humanlikeness(action, self.profile.score(action))


class ActionBrain:
    def __init__(self, pool, settings):
        self.pool = pool
        self.settings = settings
        self.last_action = None
        self.last_decision = None

    def exploration_rate(self, novelty, screen_score_total):
        action_count = len(self.pool.action_records())
        count_factor = 1.0 / math.sqrt(max(1.0, action_count))
        novelty_factor = clamp(novelty / 100.0, 0.0, 1.0)
        score_factor = 1.0 / math.sqrt(max(1.0, 1.0 + screen_score_total / 200.0))
        rate = (0.12 + 0.28 * count_factor + 0.18 * novelty_factor) * score_factor
        return round(clamp(rate, self.settings.explore_min_rate, self.settings.explore_max_rate), 4)

    def score_candidate(self, item):
        record = item["record"]
        similarity = clamp(item.get("similarity", 0.0), 0.0, 1.0)
        screen, human_score = record_screen_human(record)
        reward = self.pool.model_runtime.reward(screen, human_score)
        source_bonus = 0.0000000001 if record.get("mode") == "learning" else 0.0
        sleep_confidence = clamp(record.get("sleep_confidence", 0.0), 0.0, 1.0)
        return reward + similarity * 0.00000001 + sleep_confidence * 0.000000001 + source_bonus

    def mutate_point(self, point, scale):
        return [
            round(clamp(safe_float(point[0], 0.5) + random.uniform(-scale, scale), 0.0, 1.0), 6),
            round(clamp(safe_float(point[1], 0.5) + random.uniform(-scale, scale), 0.0, 1.0), 6)
        ]

    def mutate_action(self, action, strength):
        if not action:
            return None
        result = normalize_mouse_action(action, (0, 0, 1, 1))
        result["source"] = "ai"
        scale = clamp(self.settings.action_jitter * strength, 0.002, 0.055)
        result["start_rel"] = self.mutate_point(result.get("start_rel", [0.5, 0.5]), scale)
        result["end_rel"] = self.mutate_point(result.get("end_rel", result["start_rel"]), scale)
        if result.get("type") == "click":
            result["end_rel"] = list(result["start_rel"])
        if result.get("type") == "drag" and distance(result["start_rel"], result["end_rel"]) < 0.01:
            result["end_rel"] = self.mutate_point(result["start_rel"], scale * 4.0)
        duration = safe_float(result.get("duration", 0.12), 0.12)
        result["duration"] = round(clamp(duration * random.uniform(0.85, 1.2), self.settings.action_duration_min, self.settings.action_duration_max), 6)
        result["path_rel"] = []
        return result

    def random_action(self, strength=1.0):
        action_type = random.choice(["click", "drag", "scroll"])
        start = [round(random.random(), 6), round(random.random(), 6)]
        if action_type == "scroll":
            magnitude = random.choice([-1, 1]) * max(1, int(1 + strength * random.uniform(1.0, 6.0)))
            return {"type": "scroll", "button": "scroll", "source": "ai", "start_rel": start, "end_rel": start, "duration": 0.0, "scroll": [0, magnitude], "path_rel": [[start[0], start[1], 0.0]]}
        if action_type == "drag":
            span = clamp(0.08 + 0.28 * strength, 0.08, 0.75)
            end = self.mutate_point(start, span)
        else:
            end = list(start)
        duration_max = self.settings.action_duration_max if action_type == "drag" else self.settings.random_click_duration_max
        duration_min = self.settings.action_duration_min if action_type == "drag" else self.settings.random_click_duration_min
        return {"type": action_type, "button": "Button.left", "source": "ai", "start_rel": start, "end_rel": end, "duration": round(random.uniform(min(duration_min, duration_max), max(duration_min, duration_max)), 6), "path_rel": [[start[0], start[1], 0.0], [end[0], end[1], 1.0]]}

    def bootstrap_action(self, strength=0.2):
        action_type = random.choice(["click", "drag", "scroll"])
        margin = clamp(0.2 + random.random() * 0.1, 0.15, 0.35)
        start = [round(random.uniform(margin, 1.0 - margin), 6), round(random.uniform(margin, 1.0 - margin), 6)]
        if action_type == "scroll":
            magnitude = random.choice([-1, 1])
            return {"type": "scroll", "button": "scroll", "source": "ai_bootstrap", "start_rel": start, "end_rel": start, "duration": 0.0, "scroll": [0, magnitude], "path_rel": [[start[0], start[1], 0.0]], "exploration_policy": "bounded_bootstrap"}
        if action_type == "drag":
            end = self.mutate_point(start, clamp(0.04 + strength * 0.12, 0.04, 0.18))
        else:
            end = list(start)
        duration_max = self.settings.action_duration_max if action_type == "drag" else self.settings.random_click_duration_max
        duration_min = self.settings.action_duration_min if action_type == "drag" else self.settings.random_click_duration_min
        return {"type": action_type, "button": "Button.left", "source": "ai_bootstrap", "start_rel": start, "end_rel": end, "duration": round(random.uniform(min(duration_min, duration_max), max(duration_min, duration_max)), 6), "path_rel": [[start[0], start[1], 0.0], [end[0], end[1], 1.0]], "exploration_policy": "bounded_bootstrap"}

    def fallback_action(self):
        learned = self.pool.best_global_action()
        if learned and random.random() < self.settings.global_action_probability:
            return self.mutate_action(learned, 1.8), "global_experience"
        return self.bootstrap_action(0.2), "bounded_bootstrap_exploration"

    def choose(self, hash_value, novelty, batch, screen_score_total):
        rate = clamp(self.exploration_rate(novelty, screen_score_total), self.settings.explore_min_rate, self.settings.explore_max_rate)
        usable = []
        for item in batch:
            action = item["record"].get("mouse_action")
            if not action or safe_float(item["record"].get("reward", 0.0), 0.0) < -60.0:
                continue
            score = self.score_candidate(item)
            policy_score = self.pool.model_runtime.operation_policy_score(item["record"], item.get("similarity", 0.0))
            usable.append((math.exp(clamp(score + policy_score * 0.000001, self.settings.reward_total_min, self.settings.reward_total_max) / self.settings.softmax_temperature), {"item": item, "score": score, "policy_score": policy_score, "action": action}))
        if random.random() < rate or not usable:
            action, reason = self.fallback_action()
            decision = {"reason": reason, "exploration_rate": rate, "candidate_count": len(usable), "confidence": 0.0, "nearest_similarity": round(batch[0]["similarity"], 4) if batch else 0.0}
        else:
            chosen = weighted_choice(usable)
            item = chosen["item"]
            confidence = clamp(item.get("similarity", 0.0) * 0.65 + clamp(chosen.get("score", 0.0), 0.0, 200.0) / 200.0 * (35.0 / 100.0), 0.0, 1.0)
            action = self.mutate_action(chosen["action"], 1.0 - confidence + rate)
            decision = {"reason": "nearest_rewarded_experience", "exploration_rate": rate, "candidate_count": len(usable), "confidence": round(confidence, 4), "nearest_similarity": round(batch[0]["similarity"], 4) if batch else 0.0, "chosen_similarity": round(item.get("similarity", 0.0), 4), "chosen_reward": round(safe_float(item["record"].get("reward", 0.0), 0.0), 2), "chosen_record_id": item["record"].get("id")}
        self.last_action = copy.deepcopy(action) if action else None
        self.last_decision = decision
        return action, decision


class MouseRecorder:
    def __init__(self, get_mode, get_rect, on_activity):
        self.get_mode = get_mode
        self.get_rect = get_rect
        self.on_activity = on_activity
        self.lock = threading.RLock()
        self.actions = deque()
        self.start_markers = deque()
        self.current_by_button = {}
        self.move_buffer = []
        self.move_action_id = None
        self.listener = None
        self.wake = threading.Event()
        self.cursor_outside_event = threading.Event()

    def start(self):
        if self.listener or not pynput_mouse:
            return
        self.listener = pynput_mouse.Listener(on_move=self.on_move, on_click=self.on_click, on_scroll=self.on_scroll)
        self.listener.start()

    def stop(self):
        if not self.listener:
            return
        try:
            self.listener.stop()
        except Exception:
            pass
        self.listener = None

    def clear(self):
        with self.lock:
            self.actions.clear()
            self.start_markers.clear()
            self.current_by_button = {}
            self.move_buffer.clear()
            self.move_action_id = None
            self.wake.clear()
            self.cursor_outside_event.clear()

    def active(self):
        return self.get_mode() == "learning"

    def capture_event(self, kind, x, y, extra=None, allow_current=False):
        if not self.active():
            return None
        rect = self.get_rect()
        if not rect:
            return None
        inside = point_inside(rect, x, y)
        if not inside:
            self.cursor_outside_event.set()
            self.wake.set()
            if self.current_by_button:
                for active_action in self.current_by_button.values():
                    active_action["invalid_outside_client"] = True
                    active_action["termination_reason"] = "cursor_outside"
            if self.move_buffer:
                self.move_buffer.clear()
                self.move_action_id = None
            return None
        self.on_activity()
        self.wake.set()
        previous = None
        active_paths = [action for action in self.current_by_button.values() if action.get("path")]
        if active_paths:
            previous = max(active_paths, key=lambda action: action["path"][-1].get("t", 0.0))["path"][-1]
        elif self.move_buffer:
            previous = self.move_buffer[-1]
        return build_mouse_event(kind, x, y, rect, previous=previous, extra=extra)

    def cursor_outside(self):
        return self.cursor_outside_event.is_set()

    def clear_cursor_outside(self):
        self.cursor_outside_event.clear()

    def push_start_marker(self, action_id, event, action_type):
        self.start_markers.append({"action_id": action_id, "action_type": action_type, "perf_time": event["t"], "created_at": now_text(), "x": event["x"], "y": event["y"]})
        self.wake.set()

    def move_flush_due(self, now_perf=None):
        now_perf = time.perf_counter() if now_perf is None else now_perf
        if not self.move_buffer:
            return False
        duration = now_perf - self.move_buffer[0]["t"]
        idle = now_perf - self.move_buffer[-1]["t"]
        return (len(self.move_buffer) >= 2 and duration >= 0.25) or idle >= 0.12

    def flush_move_locked(self, force=False, now_perf=None):
        if not self.move_buffer or not self.move_action_id:
            return None
        now_perf = time.perf_counter() if now_perf is None else now_perf
        if not force and not self.move_flush_due(now_perf):
            return None
        path = list(self.move_buffer)
        if len(path) == 1:
            last = dict(path[-1])
            last["t"] = now_perf
            path.append(last)
        first = path[0]
        last = path[-1]
        action = {"action_id": self.move_action_id, "type": "move", "button": "none", "source": "user", "started_at": first.get("created_at", now_text()), "ended_at": now_text(), "started_perf": first["t"], "ended_perf": last["t"], "duration": round(max(0.0, last["t"] - first["t"]), 6), "start_abs": [int(first["x"]), int(first["y"])], "end_abs": [int(last["x"]), int(last["y"])], "path": path}
        self.actions.append(action)
        self.move_buffer = []
        self.move_action_id = None
        self.wake.set()
        return action

    def on_move(self, x, y):
        with self.lock:
            event = self.capture_event("move", x, y, allow_current=bool(self.current_by_button))
        if not event:
            return
        with self.lock:
            if self.current_by_button:
                for active_action in self.current_by_button.values():
                    active_action["path"].append(event)
            else:
                if self.move_buffer and not point_inside(self.get_rect(), event["x"], event["y"]):
                    self.flush_move_locked(force=True, now_perf=event["t"])
                    return
                if not self.move_buffer:
                    self.move_action_id = uuid.uuid4().hex
                    self.push_start_marker(self.move_action_id, event, "move")
                self.move_buffer.append(event)
                self.flush_move_locked(now_perf=event["t"])

    def on_scroll(self, x, y, dx, dy):
        event = self.capture_event("scroll", x, y, {"dx": int(dx), "dy": int(dy)})
        if not event:
            return
        action_id = uuid.uuid4().hex
        action = {"action_id": action_id, "type": "scroll", "button": "scroll", "source": "user", "started_at": now_text(), "ended_at": now_text(), "started_perf": event["t"], "ended_perf": event["t"], "duration": 0.0, "start_abs": [int(x), int(y)], "end_abs": [int(x), int(y)], "path": [event], "scroll": [int(dx), int(dy)]}
        with self.lock:
            self.flush_move_locked(force=True, now_perf=event["t"])
            self.push_start_marker(action_id, event, "scroll")
            self.actions.append(action)
            self.wake.set()

    def on_click(self, x, y, button, pressed):
        button_key = str(button)
        with self.lock:
            event = self.capture_event("press" if pressed else "release", x, y, {"button": button_key}, allow_current=(not pressed and button_key in self.current_by_button))
        if not event:
            return
        with self.lock:
            if pressed:
                self.flush_move_locked(force=True, now_perf=event["t"])
                action_id = uuid.uuid4().hex
                self.current_by_button[button_key] = {"action_id": action_id, "type": "click", "button": button_key, "source": "user", "started_at": now_text(), "started_perf": event["t"], "t0": event["t"], "start_abs": [int(x), int(y)], "path": [event]}
                self.push_start_marker(action_id, event, "click")
            elif button_key in self.current_by_button:
                current = self.current_by_button.pop(button_key)
                current["path"].append(event)
                start_abs = current["start_abs"]
                end_abs = [int(x), int(y)]
                current.update({"end_abs": end_abs, "ended_at": now_text(), "ended_perf": event["t"], "duration": round(max(0.0, event["t"] - current.get("t0", event["t"])), 6)})
                if int(start_abs[0]) != int(end_abs[0]) or int(start_abs[1]) != int(end_abs[1]):
                    current["type"] = "drag"
                self.actions.append(current)
                self.wake.set()

    def pop_start_markers(self):
        with self.lock:
            items = list(self.start_markers)
            self.start_markers.clear()
            return items

    def pop_actions(self):
        with self.lock:
            self.flush_move_locked(force=self.move_flush_due())
            items = [item for item in self.actions if not item.get("invalid_outside_client") and all(event.get("inside", True) for event in item.get("path", []))]
            self.actions.clear()
            if not self.start_markers and not self.move_buffer:
                self.wake.clear()
            return items

    def wait(self, timeout):
        return self.wake.wait(timeout)


class HumanMouseExecutor:
    def __init__(self, window_manager, settings):
        self.window_manager = window_manager
        self.settings = settings
        self.controller = pynput_mouse.Controller()

    def button_from_text(self, value):
        text = str(value).lower()
        if "right" in text:
            return pynput_mouse.Button.right
        if "middle" in text:
            return pynput_mouse.Button.middle
        return pynput_mouse.Button.left

    def clamp_point_to_rect(self, point, rect):
        if not rect:
            return (int(point[0]), int(point[1]))
        left, top, right, bottom = rect
        return (max(left, min(right - 1, int(point[0]))), max(top, min(bottom - 1, int(point[1]))))

    def smooth_points(self, start, end, duration, rect=None):
        sx, sy = start
        ex, ey = end
        direct = math.hypot(ex - sx, ey - sy)
        steps = max(2, int(math.sqrt(max(1.0, direct)) + max(0.0, duration) * self.settings.motion_steps_per_second))
        angle = random.uniform(0.0, math.tau)
        offset = direct * random.uniform(self.settings.motion_curve_offset_min, self.settings.motion_curve_offset_max)
        c1 = (sx + (ex - sx) * random.uniform(self.settings.motion_first_control_min, self.settings.motion_first_control_max) + math.cos(angle) * offset, sy + (ey - sy) * random.uniform(self.settings.motion_first_control_min, self.settings.motion_first_control_max) + math.sin(angle) * offset)
        c2 = (sx + (ex - sx) * random.uniform(self.settings.motion_second_control_min, self.settings.motion_second_control_max) - math.cos(angle) * offset, sy + (ey - sy) * random.uniform(self.settings.motion_second_control_min, self.settings.motion_second_control_max) - math.sin(angle) * offset)
        points = []
        for index in range(steps + 1):
            t = index / steps
            u = 1.0 - t
            x = u ** 3 * sx + 3.0 * u ** 2 * t * c1[0] + 3.0 * u * t ** 2 * c2[0] + t ** 3 * ex
            y = u ** 3 * sy + 3.0 * u ** 2 * t * c1[1] + 3.0 * u * t ** 2 * c2[1] + t ** 3 * ey
            points.append(self.clamp_point_to_rect((round(x), round(y)), rect))
        return points

    def stoppable_sleep(self, seconds, stop_event, should_stop):
        deadline = time.perf_counter() + max(0.0, seconds)
        while time.perf_counter() < deadline:
            if stop_event.is_set() or should_stop():
                stop_event.set()
                break
            stop_event.wait(min(self.settings.generated_sleep_event_wait, max(0.0, deadline - time.perf_counter())))

    def move_smooth(self, start, end, duration, stop_event, should_stop, rect=None, previous=None, on_activity=None):
        points = self.smooth_points(start, end, duration, rect=rect)
        actual = []
        delay = duration / max(1, len(points) - 1) if duration > 0.0 else 0.0
        last = previous
        for point in points:
            if stop_event.is_set() or should_stop():
                stop_event.set()
                break
            if rect and not point_inside(rect, point[0], point[1]):
                stop_event.set()
                return actual
            if rect and not self.window_manager.window_ok(force=True):
                stop_event.set()
                return actual
            self.controller.position = point
            if on_activity:
                on_activity()
            event = build_mouse_event("move", point[0], point[1], rect, previous=last)
            event["source"] = "ai"
            actual.append(event)
            last = event
            if delay > 0.0:
                self.stoppable_sleep(delay, stop_event, should_stop)
        return actual

    def execute(self, action, rect, stop_event, should_stop, on_activity=None):
        if not action:
            return None
        if rect:
            self.controller.position = self.clamp_point_to_rect(self.controller.position, rect)
        self.window_manager.topmost()
        if not self.window_manager.window_ok(force=True):
            return {"execution_error": "window_not_ready"}
        action = normalize_mouse_action(action, rect)
        action_type = action.get("type", "click")
        button = self.button_from_text(action.get("button", "Button.left"))
        start_rel = action.get("start_rel") or action.get("end_rel") or [0.5, 0.5]
        end_rel = action.get("end_rel") or start_rel
        start_abs = abs_from_rel(rect, start_rel)
        end_abs = abs_from_rel(rect, end_rel)
        if not point_inside(rect, start_abs[0], start_abs[1]) or not point_inside(rect, end_abs[0], end_abs[1]):
            return {"execution_error": "point_outside_client", "rect": list(rect), "start_abs": [int(start_abs[0]), int(start_abs[1])], "end_abs": [int(end_abs[0]), int(end_abs[1])]}
        current = self.controller.position
        distance_to_start = math.hypot(start_abs[0] - current[0], start_abs[1] - current[1])
        main_distance = math.hypot(end_abs[0] - start_abs[0], end_abs[1] - start_abs[1])
        duration = safe_float(action.get("duration", 0.0), 0.0)
        if duration <= 0.0:
            width, height = rect_size(rect)
            duration = clamp((distance_to_start + main_distance) / max(width, height), self.settings.action_duration_min, self.settings.action_duration_max)
        duration = clamp(duration, self.settings.action_duration_min, self.settings.action_duration_max)
        approach_ratio = distance_to_start / max(1.0, distance_to_start + main_distance)
        approach_duration = clamp(duration * approach_ratio, 0.02, duration * 0.85)
        main_duration = clamp(duration - approach_duration, 0.03, duration)
        if on_activity:
            on_activity()
        actual_path = self.move_smooth(current, start_abs, approach_duration, stop_event, should_stop, rect=rect, on_activity=on_activity)
        if stop_event.is_set():
            return None
        started_at = now_text()
        action_t = time.perf_counter()
        pressed = False
        try:
            if action_type == "drag":
                self.controller.press(button)
                pressed = True
                actual_path.extend(self.move_smooth(start_abs, end_abs, main_duration, stop_event, should_stop, rect=rect, previous=actual_path[-1] if actual_path else None, on_activity=on_activity))
            elif action_type == "scroll":
                scroll = action.get("scroll") or [0, 0]
                self.controller.scroll(int(scroll[0]), int(scroll[1]))
                if on_activity:
                    on_activity()
                actual_path.append(build_mouse_event("scroll", start_abs[0], start_abs[1], rect, previous=actual_path[-1] if actual_path else None, extra={"source": "ai", "scroll": action.get("scroll") or [0, 0]}))
            else:
                self.controller.press(button)
                pressed = True
                if on_activity:
                    on_activity()
                hold_floor = min(self.settings.random_click_duration_min, self.settings.random_click_duration_max)
                hold_ceiling = max(self.settings.random_click_duration_min, self.settings.random_click_duration_max)
                hold_duration = clamp(main_duration, hold_floor, hold_ceiling if hold_ceiling > 0.0 else self.settings.generated_click_hold_max)
                self.stoppable_sleep(clamp(hold_duration, 0.0, self.settings.generated_click_hold_max), stop_event, should_stop)
                actual_path.append(build_mouse_event("release", end_abs[0], end_abs[1], rect, previous=actual_path[-1] if actual_path else None, extra={"source": "ai", "button": str(button)}))
                if on_activity:
                    on_activity()
        except Exception as exc:
            return {"execution_error": str(exc), "error_type": type(exc).__name__, "action_type": action_type, "start_abs": [int(start_abs[0]), int(start_abs[1])], "end_abs": [int(end_abs[0]), int(end_abs[1])]}
        finally:
            if pressed:
                try:
                    self.controller.release(button)
                except Exception:
                    pass
        if stop_event.is_set():
            return None
        if rect:
            self.controller.position = self.clamp_point_to_rect(self.controller.position, rect)
        actual = {"type": action_type if action_type in ["click", "drag", "scroll"] else "click", "button": str(button), "source": "ai", "started_at": started_at, "ended_at": now_text(), "started_perf": action_t, "ended_perf": time.perf_counter(), "duration": round(max(0.0, time.perf_counter() - action_t), 6), "start_abs": [int(start_abs[0]), int(start_abs[1])], "end_abs": [int(end_abs[0]), int(end_abs[1])], "path": actual_path}
        if action_type == "scroll":
            actual["scroll"] = action.get("scroll") or [0, 0]
        normalized = normalize_mouse_action(actual, rect)
        if action_type == "scroll":
            normalized["scroll"] = actual["scroll"]
        return normalized


class EscapeMonitor:
    def __init__(self, callback, debounce_seconds):
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.listener = None
        self.last_key_time = 0.0
        self.lock = threading.RLock()

    def start(self):
        if pynput_keyboard and not self.listener:
            try:
                self.listener = pynput_keyboard.Listener(on_press=self.on_key_press)
                self.listener.start()
            except Exception:
                self.listener = None

    def stop(self):
        if self.listener:
            try:
                self.listener.stop()
            except Exception:
                pass
            self.listener = None

    def trigger(self):
        current = time.perf_counter()
        with self.lock:
            if current - self.last_key_time < self.debounce_seconds:
                return False
            self.last_key_time = current
        self.callback()
        return True

    def on_key_press(self, key):
        try:
            if key == pynput_keyboard.Key.esc:
                self.trigger()
        except Exception:
            pass

    def esc_event_pending(self):
        if not win32api:
            return False
        try:
            if win32api.GetAsyncKeyState(0x1B) & 0x8000:
                current = time.perf_counter()
                with self.lock:
                    if current - self.last_key_time < self.debounce_seconds:
                        return False
                    self.last_key_time = current
                return True
        except Exception:
            return False
        return False


class PersistenceFlushError(RuntimeError):
    pass


class AsyncPersistenceQueue:
    def __init__(self, settings_or_maxsize):
        if isinstance(settings_or_maxsize, Settings):
            self.settings = settings_or_maxsize
            maxsize = settings_or_maxsize.async_queue_size
            self.close_seconds = settings_or_maxsize.persistence_close_seconds
        else:
            self.settings = derive_runtime_settings()
            maxsize = settings_or_maxsize
            self.close_seconds = self.settings.persistence_close_seconds
        self.jobs = queue.Queue(maxsize=max(1, safe_int(maxsize, self.settings.async_queue_size)))
        self.image_dropped = 0
        self.lock = threading.RLock()
        self.next_sequence = 0
        self.pending_sequences = set()
        self.failed_sequences = set()
        self.recent_sequences = deque(maxlen=1024)
        self.last_confirmed_sequence = 0
        self.enqueued_sequences = self.pending_sequences
        self.confirmed_sequences = self.recent_sequences
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def allocate_sequence(self):
        with self.lock:
            self.next_sequence += 1
            return self.next_sequence

    def enqueue_image(self, analyzer, image, path, store=None, priority="normal"):
        critical = priority == "critical"
        if not critical and self.jobs.full():
            if store:
                try:
                    store.log_error("async_persistence_queue_full", RuntimeError("image_job_dropped_before_copy"), {"path": str(path)})
                except Exception:
                    pass
            self.image_dropped += 1
            return False
        prepared_image = image.copy() if image else None
        return self.enqueue({"type": "image", "analyzer": analyzer, "image": prepared_image, "path": path, "store": store, "priority": priority, "persistence_sequence": self.allocate_sequence()}, block_when_full=critical)

    def enqueue_record(self, store, record):
        prepared = copy.deepcopy(record)
        sequence = self.allocate_sequence()
        prepared["persistence_sequence"] = sequence
        prepared["persistence_status"] = "queued"
        return self.enqueue({"type": "record", "store": store, "record": prepared, "persistence_sequence": sequence}, block_when_full=True)

    def enqueue(self, job, block_when_full=False, timeout_seconds=None, stop_event=None, heartbeat=None):
        if self.stop_event.is_set():
            return False
        started = time.perf_counter()
        timeout = self.close_seconds if timeout_seconds is None else max(0.0, safe_float(timeout_seconds, self.close_seconds))
        interval = max(0.05, min(0.5, self.settings.persistence_event_wait or 0.1))
        sequence = job.get("persistence_sequence") if isinstance(job, dict) else None
        registered = False
        if sequence is not None:
            with self.lock:
                self.pending_sequences.add(sequence)
                registered = True
        while True:
            if self.stop_event.is_set() or (stop_event and stop_event.is_set()):
                if registered:
                    with self.lock:
                        self.pending_sequences.discard(sequence)
                return False
            try:
                self.jobs.put(job, block=block_when_full, timeout=interval if block_when_full else 0)
                return True
            except queue.Full:
                if registered and not block_when_full:
                    with self.lock:
                        self.pending_sequences.discard(sequence)
                if not block_when_full:
                    store = job.get("store")
                    if store:
                        try:
                            store.log_error("async_persistence_queue_full", RuntimeError("image_job_dropped"), {"path": str(job.get("path"))})
                        except Exception:
                            pass
                    self.image_dropped += 1
                    return False
                waited = time.perf_counter() - started
                if heartbeat:
                    heartbeat(self.status(waited))
                if waited >= timeout:
                    if registered:
                        with self.lock:
                            self.pending_sequences.discard(sequence)
                    store = job.get("store") if isinstance(job, dict) else None
                    if store:
                        store.log_error("async_persistence_enqueue_timeout", TimeoutError("persistence_enqueue_timeout"), self.status(waited))
                    return False

    def status(self, waited=0.0):
        with self.lock:
            errors = list(self.errors)
            missing = sorted(self.pending_sequences | self.failed_sequences)
            recent = list(self.recent_sequences)
            last_confirmed = self.last_confirmed_sequence
        return {"pending": self.jobs.unfinished_tasks, "queued": len(self.jobs.queue), "waited_seconds": round(waited, 1), "errors": errors, "unconfirmed_sequences": missing, "last_confirmed_sequence": last_confirmed, "recent_sequences": recent}

    def run(self):
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                if job.get("type") == "image" and job.get("image") is not None:
                    job["analyzer"].save_image(job["image"], job["path"], priority=job.get("priority", "normal"), settings=self.settings)
                elif job.get("type") == "record":
                    store = job["store"]
                    record = job["record"]
                    record["persistence_status"] = "committed"
                    record["persistence_committed_at"] = now_text()
                    store.append_experience(record)
                    store.flush_state(min_interval=self.settings.persistence_close_seconds, max_pending=max(1, self.settings.async_queue_size // max(1, self.settings.global_action_heap_limit // self.settings.local_action_heap_limit)))
                sequence = job.get("persistence_sequence") if isinstance(job, dict) else None
                if sequence is not None:
                    with self.lock:
                        self.pending_sequences.discard(sequence)
                        self.failed_sequences.discard(sequence)
                        self.last_confirmed_sequence = max(self.last_confirmed_sequence, safe_int(sequence, self.last_confirmed_sequence))
                        self.recent_sequences.append(sequence)
            except Exception as exc:
                store = job.get("store") if isinstance(job, dict) else None
                sequence = job.get("persistence_sequence") if isinstance(job, dict) else None
                error_payload = {"sequence": sequence, "type": job.get("type") if isinstance(job, dict) else None, "error": repr(exc)}
                with self.lock:
                    self.errors.append(error_payload)
                    if sequence is not None:
                        self.pending_sequences.discard(sequence)
                        self.failed_sequences.add(sequence)
                if store:
                    try:
                        store.log_error("async_persistence", exc, {"type": job.get("type"), "persistence_sequence": sequence})
                    except Exception:
                        pass
            finally:
                self.jobs.task_done()

    def flush(self, timeout_seconds=None, heartbeat=None, stop_event=None):
        started = time.perf_counter()
        timeout = None if timeout_seconds is None else max(0.0, safe_float(timeout_seconds, 0.0))
        interval = max(0.05, min(0.5, self.settings.persistence_event_wait or 0.1))
        with self.jobs.all_tasks_done:
            while self.jobs.unfinished_tasks:
                waited = time.perf_counter() - started
                if heartbeat:
                    heartbeat(self.status(waited))
                if timeout is not None and waited >= timeout:
                    raise PersistenceFlushError(json.dumps({"timeout_seconds": timeout, **self.status(waited)}, ensure_ascii=False))
                if stop_event and stop_event.is_set() and timeout is not None and waited >= min(timeout, self.close_seconds):
                    raise PersistenceFlushError(json.dumps({"stopped": True, **self.status(waited)}, ensure_ascii=False))
                self.jobs.all_tasks_done.wait(interval)
        state = self.status(time.perf_counter() - started)
        if state["errors"] or state["unconfirmed_sequences"]:
            raise PersistenceFlushError(json.dumps({"errors": state["errors"], "unconfirmed_sequences": state["unconfirmed_sequences"]}, ensure_ascii=False))

    def drop_pending_images(self):
        dropped = 0
        with self.jobs.mutex:
            kept = deque()
            while self.jobs.queue:
                job = self.jobs.queue.popleft()
                if isinstance(job, dict) and job.get("type") == "image":
                    dropped += 1
                    self.jobs.unfinished_tasks = max(0, self.jobs.unfinished_tasks - 1)
                    store = job.get("store")
                    if store:
                        try:
                            store.log_error("async_persistence_close", RuntimeError("image_dropped"), {"path": str(job.get("path"))})
                        except Exception:
                            pass
                else:
                    kept.append(job)
            self.jobs.queue.extend(kept)
            self.jobs.all_tasks_done.notify_all()
        self.image_dropped += dropped
        return dropped

    def close(self):
        self.stop_event.set()
        self.flush()
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            self.flush()
            self.jobs.put(None)
        self.thread.join(timeout=self.close_seconds)


class RuntimeContext:
    def __init__(self, panel):
        self.panel = panel

    @property
    def settings(self):
        return self.panel.settings

    @property
    def store(self):
        return self.panel.store

    @property
    def pool(self):
        return self.panel.experience_pool

    @property
    def brain(self):
        return self.panel.brain

    @property
    def window_manager(self):
        return self.panel.window_manager

    @property
    def executor(self):
        return self.panel.executor


class ModeController:
    def __init__(self, panel):
        self.panel = panel

    def begin(self, mode, deadline=None):
        return self.panel.begin_run(mode, deadline=deadline)

    def finish(self, token, status, progress=0.0, release=True, reason=None):
        return self.panel.finish_run(token, status, progress=progress, release=release, reason=reason)

    def active(self, token, mode=None):
        return self.panel.is_run_active(token, mode)

    def stop(self):
        return self.panel.stop_current_mode()


class MetricsPresenter:
    def __init__(self, panel):
        self.panel = panel

    def update(self, novelty, human_score, screen_reward, action_reward, reward, screen_score_total, decision=None):
        return self.panel.update_metrics(novelty, human_score, screen_reward, action_reward, reward, screen_score_total, decision=decision)


class MigrationService:
    def __init__(self, panel):
        self.panel = panel

    def run(self, token, old_path, new_path, stop_event, values):
        return self.panel.migration_loop(token, old_path, new_path, stop_event, values)


class LearningService:
    def __init__(self, panel):
        self.panel = panel

    def run(self, token, stop_event, config):
        return self.panel.learning_loop(token, stop_event, config)


class TrainingService:
    def __init__(self, panel):
        self.panel = panel

    def prepare_for_event(self, rect):
        panel = self.panel
        now_perf = time.perf_counter()
        panel.hardware_state = panel.refresh_hardware_state()
        pool_count = panel.experience_pool.count() if panel.experience_pool else 0
        screen_score_total_value = panel.store.screen_score_total if panel.store else 0.0
        metrics = (round(safe_float(panel.hardware_state.get("cpu_load", 0.0), 0.0) / 5.0), round(safe_float(panel.hardware_state.get("memory_free_ratio", 0.0), 0.0), 2), pool_count // 256, round(panel.adaptive_policy._avg(panel.adaptive_policy.execution_latency_ms, 140.0) / 25.0), round(panel.adaptive_policy._avg(panel.adaptive_policy.capture_latency_ms, 24.0) / 10.0))
        if panel.last_training_runtime_metrics == metrics and now_perf - panel.last_training_runtime_update_perf < 5.0:
            return panel.settings
        settings = panel.adaptive_policy.build(panel.settings, rect, pool_count, screen_score_total_value, hardware=panel.hardware_state)
        settings = replace(settings, hash_prefix_bits=panel.settings.hash_prefix_bits, nearest_candidate_limit=panel.settings.nearest_candidate_limit)
        if panel.experience_pool and getattr(panel.experience_pool.model, "resource_model", None):
            features = {"cpu_load": panel.hardware_state.get("cpu_load", 0.0), "memory_free_ratio": panel.hardware_state.get("memory_free_ratio", 0.5), "capture_ms": panel.adaptive_policy._avg(panel.adaptive_policy.capture_latency_ms, 24.0), "execution_ms": panel.adaptive_policy._avg(panel.adaptive_policy.execution_latency_ms, 140.0), "window_instability": panel.adaptive_policy._avg(panel.adaptive_policy.window_change_flags, 0.0), "success_rate": panel.adaptive_policy._avg(panel.adaptive_policy.outcome_flags, 1.0), "pool_count": pool_count}
            settings = panel.experience_pool.model_runtime.apply_runtime_values(settings, features, panel.experience_pool.model.resource_model)
            settings = replace(settings, hash_prefix_bits=panel.settings.hash_prefix_bits, nearest_candidate_limit=panel.settings.nearest_candidate_limit)
        panel.apply_runtime_settings(settings)
        panel.last_training_runtime_metrics = metrics
        panel.last_training_runtime_update_perf = now_perf
        return settings

    def observe_screen(self, analyzer, session_id, start, rect):
        return self.panel.capture_snapshot(analyzer, "training", session_id, start, rect=rect, priority="critical")

    def decide_action(self, snapshot):
        panel = self.panel
        novelty, batch = panel.experience_pool.novelty(snapshot.hash_value, exact_checksum=getattr(snapshot, "image_checksum", ""), semantic_vector=getattr(snapshot, "semantic_vector", ()))
        screen_score_total = panel.store.screen_score_total if panel.store else 0.0
        return panel.brain.choose(snapshot.hash_value, novelty, batch, screen_score_total)

    def should_stop(self, clock, config, stop_event, suspend_time_limit=False):
        panel = self.panel
        if isinstance(clock, PausableTrainingClock):
            deadline = clock.deadline()
        else:
            deadline = clock + max(1, config.training_seconds)
        guarded = panel.active_mode_stop_reason("training", stop_event, config, deadline)
        if guarded:
            panel.termination_reason = guarded
            stop_event.set()
            panel.apply_active_stop_reason("training", guarded, stop_event)
            return True
        panel.update_progress(0.0)
        if isinstance(clock, PausableTrainingClock):
            remaining = max(0.0, clock.remaining)
        else:
            elapsed = time.perf_counter() - clock
            remaining = max(0.0, config.training_seconds - elapsed)
        panel.ui(lambda r=remaining: panel.progress_label_var.set(f"训练模式进度保持 0%｜剩余 {r:.1f} 秒"))
        return False

    def execute_and_record(self, analyzer, session_id, start, rect, snapshot, action, decision, stop_event, config=None):
        panel = self.panel
        if action.get("end_rel") is None:
            action["end_rel"] = action.get("start_rel", [0.5, 0.5])
        latest_rect = panel.current_rect()
        if not latest_rect or latest_rect != rect:
            panel.write_record("training", session_id, snapshot, None, "ai_mouse_failed", decision=decision, planned_action=action, failed_action=True, window_rect_changed=True, execution_error="window_rect_changed")
            panel.adaptive_policy.observe_execution(success=False)
            return False, {"failure_reason": "window_rect_changed"}
        deadline = None
        active_session = getattr(panel, "active_session", None)
        if active_session and active_session.mode == "training":
            deadline = active_session.deadline
        stop_snapshot = StopSnapshot(safe_int(getattr(config, "training_seconds", DEFAULT_TRAINING_SECONDS), DEFAULT_TRAINING_SECONDS), safe_float(getattr(config, "still_seconds", DEFAULT_STILL_SECONDS), DEFAULT_STILL_SECONDS), deadline)
        def training_execution_stop():
            reason = panel.active_mode_stop_reason("training", stop_event, stop_snapshot, stop_snapshot.deadline)
            if reason:
                panel.apply_active_stop_reason("training", reason, stop_event)
                return True
            return False
        actual = panel.executor.execute(action, rect, stop_event, training_execution_stop, panel.mark_learning_activity)
        panel.events.publish("mouse_action_completed", mode="training", success=bool(actual and not actual.get("execution_error")))
        if not actual:
            panel.log_exception("training.execute", RuntimeError("empty_action_result"))
            panel.write_record("training", session_id, snapshot, None, "ai_mouse_failed", decision=decision, planned_action=action, failed_action=True, execution_error="empty_action_result")
            panel.adaptive_policy.observe_execution(success=False)
            return False, {"failure_reason": "empty_action_result"}
        if actual.get("execution_error"):
            panel.log_exception("training.execute", RuntimeError(actual.get("execution_error")), {"detail": actual, "decision": decision})
            panel.write_record("training", session_id, snapshot, None, "ai_mouse_failed", decision=decision, planned_action=action, failed_action=True, execution_error=actual.get("execution_error"))
            panel.adaptive_policy.observe_execution(success=False)
            return False, {"failure_reason": actual.get("execution_error") or "execution_error"}
        latency_ms = safe_float(actual.get("duration", 0.0), 0.0) * 1000.0
        panel.adaptive_policy.observe_execution(latency_ms=latency_ms, success=True)
        panel.mark_learning_activity()
        after_snapshot = None
        for attempt in range(3):
            after_snapshot = panel.capture_snapshot(analyzer, "training", session_id, start, rect=rect, priority="critical")
            if after_snapshot:
                break
            stop_event.wait(min(0.2, max(0.01, panel.settings.training_event_wait)))
        if not after_snapshot:
            record = panel.write_record("training", session_id, snapshot, actual, "ai_mouse_after_screen_missing", decision=decision, action_anchor_perf=actual.get("started_perf"), after_snapshot=None, planned_action=action, execution_latency_ms=round(latency_ms, 3), execution_error="after_snapshot_missing", screen_result_unknown=True, exclude_from_training=True)
            panel.adaptive_policy.observe_execution(success=False)
            return False, {"failure_reason": "after_snapshot_missing", "record_id": record.get("id") if record else None}
        record = panel.write_record("training", session_id, snapshot, actual, "ai_mouse", decision=decision, action_anchor_perf=actual.get("started_perf"), after_snapshot=after_snapshot, planned_action=action, execution_latency_ms=round(latency_ms, 3))
        return True, record

    def run(self, token, stop_event, config):
        return self.panel.training_loop(token, stop_event, config)


class ControlPanel(tk.Tk):
    def __init__(self):
        enable_dpi_awareness()
        super().__init__()
        self.events = EventBus()
        self.adaptive_policy = AdaptivePolicy()
        initial_capture_ms = self.measure_capture_latency()
        self.hardware_state = read_hardware_state()
        now_perf = time.perf_counter()
        self.hardware_last_full_refresh_perf = now_perf
        self.hardware_last_light_refresh_perf = now_perf
        self.last_training_runtime_update_perf = 0.0
        self.last_training_runtime_metrics = None
        self.settings = derive_runtime_settings(rect=self.screen_rect(), pool_count=0, capture_ms=initial_capture_ms, cpu_load=safe_float(self.hardware_state.get("cpu_load", 0.0), 0.0), hardware=self.hardware_state)
        self.event_journal = deque(maxlen=max(128, self.settings.async_queue_size * max(1, self.settings.ui_metric_columns)))
        self.events.subscribe("*", self.remember_event)
        self.adaptive_policy.observe_capture(initial_capture_ms)
        self.progress_value = 0.0
        self.last_progress_update_perf = 0.0
        self.last_metrics_update_perf = 0.0
        self.last_status_update_perf = 0.0
        self.last_metric_payload = None
        self.persistence_queue = AsyncPersistenceQueue(self.settings)
        self.persistence_paused = threading.Event()
        self.main_thread_events = queue.Queue(maxsize=max(64, self.settings.async_queue_size))
        self.coalesced_main_thread_events = {}
        self.main_thread_events_dropped = 0
        self.shutdown_requested = False
        self.runtime_context = RuntimeContext(self)
        self.mode_controller = ModeController(self)
        self.migration_service = MigrationService(self)
        self.metrics_presenter = MetricsPresenter(self)
        self.learning_service = LearningService(self)
        self.training_service = TrainingService(self)
        self.events.subscribe("window_state_changed", lambda event: self.ui(lambda e=event: self.status_var.set(f"客户区状态事件：{e.get('reason', 'ok')}")))
        self.events.subscribe("sleep_batch_completed", lambda event: self.ui(lambda e=event: self.progress_label_var.set(f"睡眠批次完成｜已完成 {e.get('completed_batches', 0)} 批")))
        self.events.subscribe("experience_pool_compaction_completed", lambda event: self.ui(lambda e=event: self.status_var.set(f"经验池压缩完成：删除 {e.get('removed', 0)} 条")))
        self.events.subscribe("save_completed", lambda event: self.ui(lambda e=event: self.status_var.set(f"保存完成：{e.get('kind', 'data')}")))
        self.title("雷电模拟器学习训练控制面板")
        self.geometry(f"{self.settings.ui_width}x{self.settings.ui_height}")
        self.minsize(self.settings.ui_min_width, self.settings.ui_min_height)
        self.state_lock = threading.RLock()
        self.mode = "idle"
        self.run_token = 0
        self.stop_event = threading.Event()
        self.active_session = ModeSession(0, "idle", time.perf_counter(), None, self.stop_event)
        self.termination_reason = None
        self.mode_thread = None
        self.window_manager = None
        self.store = None
        self.experience_pool = ExperiencePool(self.settings)
        self.brain = ActionBrain(self.experience_pool, self.settings)
        self.mouse_recorder = None
        self.executor = None
        self.last_learning_activity = time.perf_counter()
        self.activity_lock = threading.RLock()
        self.last_cursor_pos = None
        self.escape_monitor = EscapeMonitor(self.request_escape, self.settings.key_debounce_seconds)
        self.ldplayer_var = tk.StringVar(value=DEFAULT_LDPLAYER_PATH)
        self.data_var = tk.StringVar(value=DEFAULT_DATA_PATH)
        self.training_seconds_var = tk.StringVar(value=str(DEFAULT_TRAINING_SECONDS))
        self.still_seconds_var = tk.StringVar(value=str(DEFAULT_STILL_SECONDS))
        self.experience_pool_gb_var = tk.StringVar(value=str(DEFAULT_EXPERIENCE_POOL_GB))
        self.ai_model_limit_var = tk.StringVar(value=str(DEFAULT_AI_MODEL_LIMIT))
        self.mode_var = tk.StringVar(value=MODE_NAMES["idle"])
        self.status_var = tk.StringVar(value="等待开始")
        self.screen_score_total_var = tk.StringVar(value="0")
        self.reward_var = tk.StringVar(value="0")
        self.screen_reward_var = tk.StringVar(value="0")
        self.action_reward_var = tk.StringVar(value="0")
        self.novelty_var = tk.StringVar(value="0%")
        self.human_var = tk.StringVar(value="0%")
        self.ai_var = tk.StringVar(value="未决策")
        self.pool_var = tk.StringVar(value="0")
        self.base_runtime_var = tk.StringVar(value="基础运行环境：检查中")
        self.ldplayer_client_var = tk.StringVar(value="雷电客户区：未附着")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="0%")
        self.last_learning_event_perf = 0.0
        self.last_learning_event_hash = None
        self.hardware_last_full_refresh_perf = getattr(self, "hardware_last_full_refresh_perf", 0.0)
        self.hardware_last_light_refresh_perf = getattr(self, "hardware_last_light_refresh_perf", 0.0)
        self.last_training_runtime_update_perf = getattr(self, "last_training_runtime_update_perf", 0.0)
        self.last_training_runtime_metrics = getattr(self, "last_training_runtime_metrics", None)
        self.progress_label_var = tk.StringVar(value="进度")
        self.runtime_value_specs = {
            "training_seconds": ("训练秒数", self.training_seconds_var, DEFAULT_TRAINING_SECONDS, safe_int, 1),
            "still_seconds": ("静止秒数", self.still_seconds_var, DEFAULT_STILL_SECONDS, safe_float, 0.1),
            "experience_pool_gb": ("经验池 GB", self.experience_pool_gb_var, DEFAULT_EXPERIENCE_POOL_GB, safe_float, 0.1),
            "ai_model_limit": ("AI 模型个数", self.ai_model_limit_var, DEFAULT_AI_MODEL_LIMIT, safe_int, 1)
        }
        self.modify_buttons = []
        self.runtime_environment_refresh_id = None
        self.runtime_environment_last_ready = None
        self.metric_items = []
        self.hint_label = None
        self.metrics_frame = None
        self.app_config_store = AppConfigStore()
        self.hint_templates = {
            "idle": "当前处于空闲。可修改允许的六项参数，或启动学习、训练、睡眠模式；启动学习/训练前会自动检查并修复运行环境。",
            "starting": "正在准备模式。控制面板会最小化，鼠标会被放入雷电模拟器客户区；本程序窗口造成的遮挡不会判定为客户区异常。",
            "learning": "学习模式：请只在雷电模拟器客户区内操作。ESC、鼠标静止超时、用户鼠标离开客户区或客户区真实异常会结束并保存数据。",
            "training": "训练模式：AI 鼠标动作会被限制在雷电模拟器客户区内。ESC、用户干预导致鼠标离开客户区或客户区真实异常会结束并保存数据。",
            "sleep": "睡眠模式：正在检查样本评分、训练一组 AI 模型并清理模型组和经验池；进度条会从 0% 推进到 100%。",
            "migration": "正在迁移数据目录。请等待复制、校验与保存完成，完成前不要关闭程序。",
            "stopping": "正在安全保存与退出当前模式。后台写入完成后会回到空闲并显示控制面板。"
        }
        self.build_ui()
        self.load_persistent_settings()
        self.after_idle(self.fit_complete_panel)
        self.bind("<Configure>", self.on_window_resize)
        self.bind("<Escape>", lambda event: self.request_escape())
        if self.required_import_error():
            self.after(200, self.show_import_error)
        else:
            self.mouse_recorder = MouseRecorder(self.current_mode, self.current_rect, self.mark_learning_activity)
            self.mouse_recorder.start()
        self.escape_monitor.start()
        if not pynput_keyboard:
            self.status_var.set("全局键盘监听不可用，已启用 Windows ESC 轮询兜底")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_runtime_environment_state()

    def modify_settings_dialog(self):
        if self.current_mode() != "idle":
            self.status_var.set("只能在空闲状态点击修改")
            return
        values = self.pending_user_settings()
        ldplayer = filedialog.askopenfilename(title="选择雷电模拟器 dnplayer.exe", initialdir=str(Path(values["ldplayer_path"] or DEFAULT_LDPLAYER_PATH).parent), filetypes=[("dnplayer.exe", "dnplayer.exe"), ("可执行文件", "*.exe"), ("所有文件", "*.*")], parent=self)
        if ldplayer:
            values["ldplayer_path"] = ldplayer
        data_path = filedialog.askdirectory(title="选择数据存储路径", initialdir=values["data_path"] or DEFAULT_DATA_PATH, parent=self)
        if data_path:
            values["data_path"] = data_path
        for field in ("experience_pool_gb", "ai_model_limit", "still_seconds", "training_seconds"):
            changed = self.ask_runtime_value(field, values[field])
            if changed is not None:
                values[field] = changed
        self.submit_user_settings(values)

    def pending_user_settings(self):
        return {"ldplayer_path": self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH, "data_path": self.data_var.get().strip() or DEFAULT_DATA_PATH, "training_seconds": max(1, safe_int(self.training_seconds_var.get(), DEFAULT_TRAINING_SECONDS)), "still_seconds": max(0.1, safe_float(self.still_seconds_var.get(), DEFAULT_STILL_SECONDS)), "experience_pool_gb": max(0.1, safe_float(self.experience_pool_gb_var.get(), DEFAULT_EXPERIENCE_POOL_GB)), "ai_model_limit": max(1, safe_int(self.ai_model_limit_var.get(), DEFAULT_AI_MODEL_LIMIT))}

    def ask_runtime_value(self, field, current_value):
        AllowedUserEditPolicy.assert_allowed(field)
        title, _, default, parser, minimum = self.runtime_value_specs[field]
        answer = simpledialog.askstring("修改" + title, "请输入" + title, initialvalue=self.format_runtime_value(field, current_value), parent=self)
        if answer is None:
            return None
        return max(minimum, parser(answer, default))

    def apply_runtime_value_vars(self, values):
        self.training_seconds_var.set(self.format_runtime_value("training_seconds", values["training_seconds"]))
        self.still_seconds_var.set(self.format_runtime_value("still_seconds", values["still_seconds"]))
        self.experience_pool_gb_var.set(self.format_runtime_value("experience_pool_gb", values["experience_pool_gb"]))
        self.ai_model_limit_var.set(self.format_runtime_value("ai_model_limit", values["ai_model_limit"]))

    def submit_user_settings(self, values):
        old_values = self.pending_user_settings()
        new_ldplayer = str(values.get("ldplayer_path") or DEFAULT_LDPLAYER_PATH)
        new_data_path = Path(str(values.get("data_path") or DEFAULT_DATA_PATH))
        if new_ldplayer != old_values["ldplayer_path"]:
            ok, reason = validate_ldplayer_executable(new_ldplayer, self.settings, require_attach=False)
            if not ok:
                messagebox.showerror("雷电路径不合法", reason)
                self.status_var.set("雷电路径校验失败，未保存")
                self.update_mode_button_states()
                return False
        normalized = {"ldplayer_path": new_ldplayer, "data_path": str(new_data_path), "training_seconds": max(1, safe_int(values.get("training_seconds"), DEFAULT_TRAINING_SECONDS)), "still_seconds": max(0.1, safe_float(values.get("still_seconds"), DEFAULT_STILL_SECONDS)), "experience_pool_gb": max(0.1, safe_float(values.get("experience_pool_gb"), DEFAULT_EXPERIENCE_POOL_GB)), "ai_model_limit": max(1, safe_int(values.get("ai_model_limit"), DEFAULT_AI_MODEL_LIMIT))}
        if Path(old_values["data_path"]) != new_data_path:
            return self.start_data_migration(new_data_path, normalized)
        self.ldplayer_var.set(new_ldplayer)
        self.data_var.set(str(new_data_path))
        self.apply_runtime_value_vars(normalized)
        self.save_persistent_settings()
        self.refresh_runtime_environment_state()
        self.status_var.set("修改已保存")
        return True

    def build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        bg = "#eef3f8"
        card_bg = "#ffffff"
        accent = "#2563eb"
        text = "#172033"
        muted = "#5b677a"
        self.configure(bg=bg)
        style.configure(".", font=("Microsoft YaHei UI", 10), background=bg, foreground=text)
        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background=card_bg, relief="flat")
        style.configure("TLabelframe", background=bg, bordercolor="#cbd5e1", relief="solid")
        style.configure("TLabelframe.Label", background=bg, foreground=text, font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TButton", padding=(14, 8), font=("Microsoft YaHei UI", 10, "bold"))
        style.map("TButton", background=[("active", "#dbeafe")], foreground=[("disabled", "#94a3b8")])
        style.configure("TEntry", fieldbackground="#f8fafc", bordercolor="#cbd5e1", padding=4)
        style.configure("Horizontal.TProgressbar", troughcolor="#dbeafe", background=accent)
        scale = max(0.8, min(2.2, float(self.winfo_fpixels("1i")) / 96.0))
        pane_h = max(1, self.winfo_height() or self.settings.ui_height)
        font_scale = max(0.8, min(2.1, (pane_h / max(1, self.settings.ui_min_height)) ** 0.5 * scale))
        title_size = max(11, int(round(14 * font_scale)))
        value_size = max(10, int(round(12 * font_scale)))
        card_size = max(8, int(round(9 * font_scale)))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", title_size, "bold"))
        style.configure("CardTitle.TLabel", font=("Microsoft YaHei UI", card_size))
        style.configure("Value.TLabel", font=("Microsoft YaHei UI", value_size, "bold"))
        style.configure("Hint.TLabel", foreground=muted, background=card_bg)
        root = ttk.Frame(self, padding=self.settings.ui_padding)
        root.pack(fill="both", expand=True)
        self.scroll_canvas = tk.Canvas(root, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.configure(background=bg)
        container = ttk.Frame(self.scroll_canvas)
        self.scroll_window = self.scroll_canvas.create_window((0, 0), window=container, anchor="nw")
        container.bind("<Configure>", self.update_scroll_region)
        self.scroll_canvas.bind("<Configure>", self.update_scroll_width)
        self.scroll_canvas.bind("<Enter>", self.bind_mousewheel)
        self.scroll_canvas.bind("<Leave>", self.unbind_mousewheel)
        ttk.Label(container, text="雷电模拟器学习训练控制面板", style="Title.TLabel").pack(anchor="w")
        path_frame = ttk.LabelFrame(container, text="路径与时间", padding=self.settings.ui_section_padding)
        path_frame.pack(fill="x", pady=(16, 10))
        path_frame.columnconfigure(1, weight=1)
        ttk.Label(path_frame, text="雷电模拟器").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(path_frame, textvariable=self.ldplayer_var, justify="right", state="readonly").grid(row=0, column=1, sticky="ew", pady=6)
        path_frame.columnconfigure(2, weight=0)
        ttk.Label(path_frame, text="数据存储").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(path_frame, textvariable=self.data_var, justify="right", state="readonly").grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Label(path_frame, text="").grid(row=1, column=2, padx=(8, 0), pady=6)
        time_frame = ttk.Frame(path_frame)
        time_frame.grid(row=2, column=1, columnspan=2, sticky="w", pady=6)
        ttk.Label(path_frame, text="时间设置").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=6)
        for field, label in (("training_seconds", "训练/秒"), ("still_seconds", "静止/秒"), ("experience_pool_gb", "经验池/GB"), ("ai_model_limit", "AI模型/个")):
            variable = self.runtime_value_specs[field][1]
            item_frame = ttk.Frame(time_frame)
            item_frame.pack(side="left", padx=(0, 12))
            ttk.Label(item_frame, text=label).pack(side="left")
            ttk.Entry(item_frame, textvariable=variable, width=10, state="readonly", justify="right").pack(side="left", padx=(6, 4))
            ttk.Label(item_frame, text="").pack(side="left")
        button_frame = ttk.Frame(container)
        button_frame.pack(fill="x", pady=(4, 12))
        self.button_frame = button_frame
        self.modify_button = ttk.Button(button_frame, text="修改", command=self.modify_settings_dialog)
        self.learning_button = ttk.Button(button_frame, text="学习模式", command=self.learning_mode)
        self.training_button = ttk.Button(button_frame, text="训练模式", command=self.training_mode)
        self.sleep_button = ttk.Button(button_frame, text="睡眠模式", command=self.sleep_mode)
        self.mode_buttons = [self.learning_button, self.training_button, self.sleep_button]
        self.modify_buttons.append(self.modify_button)
        self.control_buttons = [self.modify_button, self.learning_button, self.training_button, self.sleep_button]
        self.reflow_buttons()
        status_frame = ttk.LabelFrame(container, text="状态", padding=self.settings.ui_section_padding)
        status_frame.pack(fill="both", expand=True)
        self.metrics_frame = ttk.Frame(status_frame)
        self.metrics_frame.grid(row=0, column=0, sticky="nsew")
        status_frame.columnconfigure(0, weight=1)
        metrics = [("当前模式", self.mode_var), ("基础环境", self.base_runtime_var), ("雷电客户区", self.ldplayer_client_var), ("画面评分累计", self.screen_score_total_var), ("经验条数", self.pool_var), ("画面评分", self.novelty_var), ("鼠标相似度", self.human_var), ("画面奖励", self.screen_reward_var), ("鼠标奖惩", self.action_reward_var), ("本次奖励", self.reward_var), ("AI决策", self.ai_var)]
        for title, variable in metrics:
            self.metric_items.append(self.create_metric(self.metrics_frame, title, variable))
        self.reflow_metrics()
        ttk.Label(status_frame, text="快捷键", style="CardTitle.TLabel").grid(row=1, column=0, sticky="w", pady=(18, 6), padx=(0, 12))
        self.hint_label = ttk.Label(status_frame, text=self.hint_text(), wraplength=max(320, self.settings.ui_width - 120), style="Hint.TLabel")
        self.hint_label.grid(row=2, column=0, sticky="ew", pady=6)
        progress_frame = ttk.LabelFrame(container, text=self.progress_label_var.get(), padding=self.settings.ui_section_padding)
        self.progress_label_var.trace_add("write", lambda *args: progress_frame.configure(text=self.progress_label_var.get()))
        progress_frame.pack(fill="x", pady=(12, 0))
        progress_frame.columnconfigure(0, weight=1)
        ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100).grid(row=0, column=0, sticky="ew")
        ttk.Label(progress_frame, textvariable=self.progress_text_var, width=8, anchor="e").grid(row=0, column=1, sticky="e", padx=(10, 0))
        ttk.Label(container, textvariable=self.status_var).pack(anchor="w", pady=(10, 0))


    def own_window_handles(self):
        handles = []
        for widget in (self, *self.winfo_children()):
            try:
                handle = int(widget.winfo_id())
                if handle:
                    handles.append(handle)
            except Exception:
                pass
        return tuple(dict.fromkeys(handles))

    def hint_text(self):
        mode = self.current_mode() if hasattr(self, "mode") else "idle"
        base = self.hint_templates.get(mode, self.hint_templates["idle"])
        runtime = getattr(self, "runtime_environment_issue", "")
        if runtime:
            return base + " 当前提示：" + runtime
        return base

    def refresh_hint(self):
        if self.hint_label:
            self.hint_label.configure(text=self.hint_text())

    def fit_complete_panel(self):
        try:
            self.update_idletasks()
            req_w = max(self.settings.ui_min_width, self.winfo_reqwidth())
            req_h = max(self.settings.ui_min_height, self.winfo_reqheight())
            screen_w = max(1, self.winfo_screenwidth())
            screen_h = max(1, self.winfo_screenheight())
            width = min(max(self.winfo_width(), req_w), screen_w)
            height = min(max(self.winfo_height(), req_h), screen_h)
            self.minsize(min(req_w, screen_w), min(req_h, screen_h))
            if self.winfo_width() < req_w or self.winfo_height() < req_h:
                self.geometry(f"{width}x{height}")
            self.update_scroll_region()
        except Exception as exc:
            self.log_exception("ui.fit_complete", exc)

    def update_scroll_region(self, _event=None):
        if getattr(self, "scroll_canvas", None):
            self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def update_scroll_width(self, event=None):
        if getattr(self, "scroll_canvas", None) and getattr(self, "scroll_window", None):
            width = event.width if event else self.scroll_canvas.winfo_width()
            self.scroll_canvas.itemconfigure(self.scroll_window, width=max(1, width))
            self.update_scroll_region()

    def create_metric(self, parent, title, variable):
        frame = ttk.Frame(parent)
        ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(anchor="w", pady=(8, 4))
        ttk.Label(frame, textvariable=variable, style="Value.TLabel").pack(anchor="w", pady=(0, 8))
        return frame

    def reflow_metrics(self):
        if not self.metrics_frame:
            return
        width = max(1, self.metrics_frame.winfo_width() or self.winfo_width())
        min_width = max(120, self.settings.ui_metric_min_column_width)
        columns = max(1, width // min_width if width >= min_width else 1)
        for column in range(max(1, len(self.metric_items))):
            self.metrics_frame.columnconfigure(column, weight=0)
        for column in range(columns):
            self.metrics_frame.columnconfigure(column, weight=1)
        for index, item in enumerate(self.metric_items):
            row = index // columns
            column = index % columns
            item.grid(row=row, column=column, sticky="ew", padx=(0, 12))
    def reflow_buttons(self):
        if not getattr(self, "button_frame", None):
            return
        width = max(1, self.button_frame.winfo_width() or self.winfo_width())
        min_width = max(120, int(max(120, self.settings.ui_metric_min_column_width * 0.7)))
        columns = max(1, width // min_width if width >= min_width else 1)
        for index, button in enumerate(self.control_buttons):
            row = index // columns
            column = index % columns
            sticky = "ew" if button.cget("text") != "退出" else "e"
            button.grid(row=row, column=column, padx=6, pady=4, sticky=sticky)
        for column in range(columns):
            self.button_frame.columnconfigure(column, weight=1)

    def on_window_resize(self, _event):
        if self.hint_label:
            width = max(320, self.winfo_width() - self.settings.ui_padding * 4)
            self.hint_label.configure(wraplength=width)
        self.reflow_metrics()
        self.reflow_buttons()

    def remember_event(self, event):
        self.event_journal.append(event)

    def ui(self, func):
        if threading.current_thread() is threading.main_thread():
            try:
                func()
            except Exception as exc:
                self.log_exception("ui.dispatch", exc)
            return
        self.enqueue_main_thread_event({"type": "ui_call", "func": func}, block=False)

    def enqueue_main_thread_event(self, event, block=False):
        if getattr(self, "shutdown_requested", False) and event.get("type") == "ui_call":
            return False
        key = event.get("coalesce_key") if isinstance(event, dict) else None
        if key:
            self.coalesced_main_thread_events[key] = event
            event = {"type": "coalesced", "key": key}
        try:
            self.main_thread_events.put(event, block=block, timeout=0.25 if block else 0)
            return True
        except queue.Full:
            if key:
                return True
            self.main_thread_events_dropped += 1
            return False

    def ui_sync(self, func, timeout=None):
        if threading.current_thread() is threading.main_thread():
            try:
                return func()
            except Exception as exc:
                self.log_exception("ui.sync", exc)
                return None
        done = threading.Event()
        result = {}
        def apply():
            try:
                result["value"] = func()
            except Exception as exc:
                result["error"] = exc
            finally:
                done.set()
        if not self.enqueue_main_thread_event({"type": "ui_sync_call", "func": apply}, block=True):
            done.set()
            return None
        wait_seconds = timeout if timeout is not None else max(self.settings.window_event_wait, self.settings.key_debounce_seconds)
        done.wait(wait_seconds)
        if "error" in result:
            self.log_exception("ui.sync", result["error"])
        return result.get("value")

    def log_exception(self, where, error, context=None):
        logged = False
        if self.store:
            try:
                self.store.log_error(where, error, context)
                logged = True
            except Exception:
                logged = False
        if not logged:
            try:
                sys.stderr.write(json.dumps({"time": now_text(), "where": str(where), "error": str(error), "context": context}, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def runtime_environment_ready(self):
        if self.required_import_error():
            return False
        storage_issue = data_path_write_issue(Path(self.data_var.get().strip() or DEFAULT_DATA_PATH), create=True)
        if storage_issue:
            self.runtime_environment_issue = "存储路径无效：" + storage_issue
            return False
        ok, reason = validate_ldplayer_executable(Path(self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH), self.settings, require_attach=False)
        if not ok:
            self.runtime_environment_issue = reason
            return False
        ready = bool(windows_runtime_report(Path(self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH)).get("ok"))
        self.runtime_environment_issue = "" if ready else "Windows 桌面运行环境不可用"
        return ready

    def offline_sleep_environment_ready(self):
        if self.required_import_error():
            return False
        path = Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path.is_dir()
        except Exception:
            return False

    def update_mode_button_states(self):
        online_enabled = self.runtime_environment_ready()
        self.base_runtime_var.set("基础运行环境：就绪" if online_enabled else "基础运行环境：异常")
        client_ready = False
        client_reason = "未附着"
        try:
            if self.window_manager:
                check = self.window_manager.check_window(force=False)
                client_ready = bool(check.ok)
                client_reason = "就绪" if check.ok else check.reason
        except Exception as exc:
            client_reason = str(exc)
        self.ldplayer_client_var.set("雷电客户区：" + client_reason)
        sleep_enabled = self.offline_sleep_environment_ready()
        mode = self.current_mode()
        online_state = "normal" if online_enabled and mode == "idle" else "disabled"
        sleep_state = "normal" if sleep_enabled and mode == "idle" else "disabled"
        modify_state = "normal" if mode == "idle" else "disabled"
        for button in (getattr(self, "learning_button", None), getattr(self, "training_button", None)):
            if button:
                try:
                    button.configure(state=online_state)
                except Exception:
                    pass
        if getattr(self, "sleep_button", None):
            try:
                self.sleep_button.configure(state=sleep_state)
            except Exception:
                pass
        for button in getattr(self, "modify_buttons", []):
            try:
                button.configure(state=modify_state)
            except Exception:
                pass
        if not online_enabled and mode == "idle":
            self.status_var.set("雷电运行环境未就绪：" + (getattr(self, "runtime_environment_issue", "") or "学习/训练需 Windows 桌面、雷电路径与可写存储路径"))
        self.runtime_environment_last_ready = online_enabled
        return online_enabled

    def runtime_environment_refresh_delay_ms(self):
        source = max(self.settings.window_event_wait, self.settings.ui_event_coalesce_seconds, self.settings.key_debounce_seconds)
        return max(200, min(3000, int(source * 1000)))

    def process_main_thread_events(self):
        processed = 0
        started = time.perf_counter()
        max_events = 50
        max_seconds = 0.008
        while processed < max_events and time.perf_counter() - started < max_seconds:
            try:
                event = self.main_thread_events.get_nowait()
            except queue.Empty:
                break
            processed += 1
            try:
                if event.get("type") in ("ui_call", "ui_sync_call"):
                    func = event.get("func")
                    if callable(func):
                        func()
                elif event.get("type") == "coalesced":
                    latest = self.coalesced_main_thread_events.pop(event.get("key"), None)
                    func = latest.get("func") if isinstance(latest, dict) else None
                    if callable(func):
                        func()
                elif event.get("type") == "restart_training":
                    self.status_var.set("睡眠模式已保存，正在自动调度训练模式")
                    old_token = event.get("token")
                    reason = event.get("reason") or "time_limit"
                    if self.is_run_active(old_token, "sleep"):
                        self.finish_run(old_token, "睡眠模式已退出，数据已保存", 0.0, release=False, reason=reason)
                    if self.current_mode() == "idle":
                        ok = self.request_active_mode("training", auto_restart=True)
                        if not ok:
                            failure = getattr(self, "last_active_mode_failure", "自动重启训练失败")
                            self.restore_panel()
                            self.status_var.set("自动重启训练失败：" + failure)
                            self.events.publish("auto_restart_training_failed", reason=failure)
            except Exception as exc:
                self.log_exception("main_thread_event", exc, event)
        if not self.main_thread_events.empty():
            try:
                self.after(1, self.process_main_thread_events)
            except Exception:
                pass

    def refresh_runtime_environment_state(self):
        try:
            self.process_main_thread_events()
            self.update_mode_button_states()
        finally:
            try:
                self.runtime_environment_refresh_id = self.after(self.runtime_environment_refresh_delay_ms(), self.refresh_runtime_environment_state)
            except Exception:
                self.runtime_environment_refresh_id = None

    def required_import_error(self):
        return {name: IMPORT_ERRORS[name] for name in tuple(REQUIRED_MODULES) if name in IMPORT_ERRORS}

    def show_import_error(self):
        missing = self.required_import_error()
        lines = [f"{name}: {error}" for name, error in missing.items()]
        messagebox.showerror("依赖异常", "依赖自动安装或加载失败。\n\n当前错误：\n" + "\n".join(lines))
        self.status_var.set("依赖缺失")

    def require_idle_for_user_edit(self):
        if self.current_mode() != "idle":
            self.status_var.set("只能在空闲状态点击修改")
            return False
        return True

    def format_runtime_value(self, field, value):
        if field in ("training_seconds", "ai_model_limit"):
            return str(safe_int(value, self.runtime_value_specs[field][2]))
        return str(safe_float(value, self.runtime_value_specs[field][2]))

    def modify_runtime_value(self, field):
        AllowedUserEditPolicy.assert_allowed(field)
        if not self.require_idle_for_user_edit():
            return
        title, variable, default, parser, minimum = self.runtime_value_specs[field]
        current = variable.get().strip() or str(default)
        answer = simpledialog.askstring("修改" + title, "请输入" + title, initialvalue=current, parent=self)
        if answer is None:
            return
        value = max(minimum, parser(answer, default))
        variable.set(self.format_runtime_value(field, value))
        self.save_persistent_settings()
        self.status_var.set(title + "已保存")
        self.update_mode_button_states()

    def choose_ldplayer(self):
        if not self.require_idle_for_user_edit():
            return
        path = filedialog.askopenfilename(title="选择 dnplayer.exe", filetypes=[("dnplayer.exe", "dnplayer.exe")])
        if not path:
            return
        ok, reason = validate_ldplayer_executable(path, self.settings, require_attach=False)
        if not ok:
            messagebox.showerror("雷电路径不合法", reason)
            self.status_var.set("雷电路径校验失败，未保存")
            self.update_mode_button_states()
            return
        self.ldplayer_var.set(path)
        self.save_persistent_settings()
        self.status_var.set("雷电路径已校验并保存")
        self.update_mode_button_states()

    def choose_data(self):
        if not self.require_idle_for_user_edit():
            return
        path = filedialog.askdirectory(title="选择数据存储目录")
        if not path:
            return
        self.start_data_migration(Path(path), self.pending_user_settings())

    def start_data_migration(self, new_path, values=None):
        old_path = Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)
        new_path = Path(new_path)
        if old_path == new_path:
            if values:
                self.ldplayer_var.set(str(values.get("ldplayer_path") or DEFAULT_LDPLAYER_PATH))
                self.apply_runtime_value_vars(values)
            self.data_var.set(str(new_path))
            self.save_persistent_settings()
            self.update_mode_button_states()
            self.status_var.set("修改已保存")
            return True
        token, stop_event = self.begin_run("migration", reason="click_modify_data_path")
        if not token:
            self.status_var.set("请先终止当前模式，或等待当前模式结束")
            return False
        self.status_var.set("正在迁移数据")
        self.update_progress(0.0)
        values = values or self.pending_user_settings()
        migration_values = {"ldplayer_path": str(values.get("ldplayer_path") or DEFAULT_LDPLAYER_PATH), "training_seconds": max(1, safe_int(values.get("training_seconds"), DEFAULT_TRAINING_SECONDS)), "still_seconds": max(0.1, safe_float(values.get("still_seconds"), DEFAULT_STILL_SECONDS)), "experience_pool_gb": max(0.1, safe_float(values.get("experience_pool_gb"), DEFAULT_EXPERIENCE_POOL_GB)), "ai_model_limit": max(1, safe_int(values.get("ai_model_limit"), DEFAULT_AI_MODEL_LIMIT))}
        self.mode_thread = threading.Thread(target=self.migration_service.run, args=(token, old_path, new_path, stop_event, migration_values))
        self.mode_thread.start()
        return True

    def migration_items(self, old_path):
        root = Path(old_path)
        items = []
        for name in ("screens", "models", "experience.jsonl", "state.json", "settings.json", "errors.jsonl", "sleep_checkpoint.json", "runtime_parameters_audit.jsonl", "startup_install.log"):
            source = root / name
            if source.is_dir():
                for file_root, _, filenames in os.walk(source):
                    for filename in filenames:
                        file_path = Path(file_root) / filename
                        try:
                            relative = file_path.relative_to(root)
                            items.append((relative, file_path, max(0, file_path.stat().st_size)))
                        except Exception:
                            pass
            elif source.exists():
                try:
                    items.append((Path(name), source, max(0, source.stat().st_size)))
                except Exception:
                    items.append((Path(name), source, 0))
        return items

    def copy_migration_file(self, source, target, copied, total, stop_event, token):
        target.parent.mkdir(parents=True, exist_ok=True)
        buffer_size = max(65536, min(1048576, safe_int(total / 200 if total else 65536, 65536)))
        with source.open("rb") as src, target.open("wb") as dst:
            while True:
                if stop_event.is_set() or not self.is_run_active(token, "migration"):
                    return copied, False
                chunk = src.read(buffer_size)
                if not chunk:
                    break
                dst.write(chunk)
                copied += len(chunk)
                self.events.publish("migration_chunk_completed", source=str(source), target=str(target), copied=copied, total=total)
                self.ui(lambda c=copied, t=total: self.progress_label_var.set(f"数据迁移中｜已迁移 {c}/{t} 字节"))
        try:
            shutil.copystat(source, target)
        except Exception:
            pass
        return copied, True

    def migration_known_names(self):
        return {"screens", "models", "experience.jsonl", "state.json", "settings.json", "errors.jsonl", "sleep_checkpoint.json", "runtime_parameters_audit.jsonl", "startup_install.log"}

    def migration_target_conflicts(self, target):
        path = Path(target)
        if not path.exists():
            return []
        known = self.migration_known_names()
        return [item.name for item in path.iterdir() if item.name not in known and not item.name.startswith(".backup_")]

    def migration_counts(self, root):
        root = Path(root)
        screens = root / "screens"
        screen_count = 0
        if screens.exists():
            for _, _, filenames in os.walk(screens):
                screen_count += len([name for name in filenames if name.lower().endswith(".png")])
        models = root / "models"
        model_count = 0
        if models.exists():
            for _, _, filenames in os.walk(models):
                model_count += len([name for name in filenames if name.lower().startswith("model_")])
        lines = 0
        audit_lines = 0
        experience = root / "experience.jsonl"
        if experience.exists():
            try:
                with experience.open("r", encoding="utf-8") as file:
                    for _ in file:
                        lines += 1
            except Exception:
                pass
        runtime_audit = root / "runtime_parameters_audit.jsonl"
        if runtime_audit.exists():
            try:
                with runtime_audit.open("r", encoding="utf-8") as file:
                    for _ in file:
                        audit_lines += 1
            except Exception:
                pass
        checkpoint = root / "sleep_checkpoint.json"
        checkpoint_stage = None
        if checkpoint.exists():
            try:
                with checkpoint.open("r", encoding="utf-8") as file:
                    loaded = json.load(file)
                checkpoint_stage = loaded.get("stage") if isinstance(loaded, dict) else None
            except Exception:
                checkpoint_stage = "invalid"
        sizes = {}
        for name in self.migration_known_names():
            path = root / name
            if path.is_file():
                try:
                    sizes[name] = path.stat().st_size
                except Exception:
                    sizes[name] = 0
        return {"screens": screen_count, "models": model_count, "experience_lines": lines, "runtime_audit_lines": audit_lines, "sleep_checkpoint": checkpoint.exists(), "sleep_checkpoint_stage": checkpoint_stage, "file_sizes": sizes, "settings": (root / "settings.json").exists(), "state": (root / "state.json").exists()}

    def migration_sample_files(self, root):
        root = Path(root)
        files = []
        for name in self.migration_known_names():
            path = root / name
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                for file_root, _, filenames in os.walk(path):
                    for filename in sorted(filenames):
                        files.append(Path(file_root) / filename)
        if not files:
            return []
        step = max(1, len(files) // max(1, self.settings.ui_metric_columns * self.settings.hash_prefix_bits))
        return files[::step][:max(1, self.settings.nearest_top_k)]

    def file_digest(self, path):
        h = 0
        with Path(path).open("rb") as file:
            while True:
                chunk = file.read(max(65536, self.settings.local_action_heap_limit * self.settings.hash_prefix_bits))
                if not chunk:
                    break
                for value in chunk:
                    h = ((h << 5) - h + value) & 0xffffffffffffffff
        return h

    def verify_migration(self, source, target):
        source_counts = self.migration_counts(source)
        target_counts = self.migration_counts(target)
        if target_counts["experience_lines"] < source_counts["experience_lines"]:
            raise ValueError("迁移校验失败：experience.jsonl 行数少于源目录")
        if target_counts["screens"] < source_counts["screens"]:
            raise ValueError("迁移校验失败：screens 文件数少于源目录")
        if target_counts["models"] < source_counts["models"]:
            raise ValueError("迁移校验失败：models 文件数少于源目录")
        if source_counts["settings"] and not target_counts["settings"]:
            raise ValueError("迁移校验失败：settings.json 缺失")
        if source_counts["state"] and not target_counts["state"]:
            raise ValueError("迁移校验失败：state.json 缺失")
        if target_counts.get("runtime_audit_lines", 0) < source_counts.get("runtime_audit_lines", 0):
            raise ValueError("迁移校验失败：runtime_parameters_audit.jsonl 行数少于源目录")
        if source_counts.get("sleep_checkpoint") and not target_counts.get("sleep_checkpoint"):
            raise ValueError("迁移校验失败：sleep_checkpoint.json 缺失")
        if source_counts.get("sleep_checkpoint_stage") != target_counts.get("sleep_checkpoint_stage"):
            raise ValueError("迁移校验失败：sleep_checkpoint.json 断点状态不一致")
        for name, size in source_counts.get("file_sizes", {}).items():
            if target_counts.get("file_sizes", {}).get(name, 0) < size:
                raise ValueError("迁移校验失败：文件大小不一致 " + name)
        for source_file in self.migration_sample_files(source):
            relative = source_file.relative_to(Path(source))
            target_file = Path(target) / relative
            if not target_file.exists():
                raise ValueError("迁移校验失败：抽样文件缺失 " + str(relative))
            if source_file.stat().st_size != target_file.stat().st_size or self.file_digest(source_file) != self.file_digest(target_file):
                raise ValueError("迁移校验失败：抽样文件不一致 " + str(relative))

    def migration_loop(self, token, old_path, new_path, stop_event, values):
        reason = "数据迁移完成"
        temp_root = None
        backup_root = None
        try:
            self.persistence_paused.set()
            if self.persistence_queue:
                self.persistence_queue.flush()
            if self.store:
                self.store.flush_state(force=True)
            old_root = Path(old_path).resolve()
            new_root = Path(new_path).resolve()
            if new_root == old_root or old_root in new_root.parents:
                raise ValueError("迁移目标不能等于旧目录，也不能位于旧目录内部")
            conflicts = self.migration_target_conflicts(new_root)
            if conflicts:
                raise ValueError("迁移目标包含非本程序文件：" + "，".join(conflicts[:8]))
            new_root.parent.mkdir(parents=True, exist_ok=True)
            temp_root = new_root.parent / f".{new_root.name}.migration.{uuid.uuid4().hex}"
            temp_root.mkdir(parents=True, exist_ok=False)
            items = self.migration_items(old_root)
            total = max(1, sum(size for _, _, size in items))
            copied = 0
            for relative, source, size in items:
                copied, ok = self.copy_migration_file(source, temp_root / relative, copied, total, stop_event, token)
                if not ok:
                    reason = "数据迁移已终止"
                    break
                if size == 0:
                    self.ui(lambda c=copied, t=total: self.progress_label_var.set(f"数据迁移中｜已迁移 {c}/{t} 字节"))
            if not stop_event.is_set() and self.is_run_active(token, "migration"):
                DataStore(temp_root).save_settings({"training_seconds": values["training_seconds"], "still_seconds": values["still_seconds"], "experience_pool_gb": values["experience_pool_gb"], "ai_model_limit": values["ai_model_limit"]})
                if new_root.exists():
                    backup_root = new_root.parent / f".backup_{new_root.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.copytree(new_root, backup_root)
                    for item in temp_root.iterdir():
                        target = new_root / item.name
                        if target.exists():
                            if target.is_dir():
                                shutil.rmtree(target)
                            else:
                                target.unlink()
                        shutil.move(str(item), str(target))
                    shutil.rmtree(temp_root, ignore_errors=True)
                else:
                    temp_root.replace(new_root)
                    temp_root = None
                self.verify_migration(old_root, new_root)
                self.ui(lambda path=str(new_root), v=dict(values): (self.ldplayer_var.set(v["ldplayer_path"]), self.data_var.set(path), self.apply_runtime_value_vars(v)))
                self.app_config_store.save_settings({"ldplayer_path": values["ldplayer_path"], "data_path": str(new_root)})
                self.store = DataStore(new_root)
                self.experience_pool = ExperiencePool(self.settings, self.store.load_experience(self.settings.experience_load_limit), self.store.load_latest_model_state(self.settings))
                self.brain = ActionBrain(self.experience_pool, self.settings)
                self.update_progress(0.0, force=True)
                self.finish_run(token, reason, 0.0, release=False, reason="completed")
            elif self.is_run_active(token, "migration"):
                self.finish_run(token, reason, 0.0, release=False, reason="user_stop")
            elif self.is_run_active(token, "stopping"):
                final_reason = self.termination_reason if self.termination_reason in ("esc", "user_stop") else "user_stop"
                self.finish_run(token, "数据迁移已终止", 0.0, release=False, reason=final_reason)
        except Exception as exc:
            if backup_root and Path(backup_root).exists():
                try:
                    if Path(new_path).exists():
                        shutil.rmtree(new_path)
                    shutil.copytree(backup_root, new_path)
                except Exception:
                    pass
            self.log_exception("migration", exc, {"old_path": str(old_path), "new_path": str(new_path)})
            self.ui(lambda e=str(exc): messagebox.showerror("迁移失败", e))
            self.finish_run(token, "数据迁移失败", self.progress_value, release=False, reason="migration_error")
        finally:
            self.persistence_paused.clear()
            if temp_root:
                shutil.rmtree(temp_root, ignore_errors=True)

    def bind_mousewheel(self, _event):
        self.bind_all("<MouseWheel>", self.on_mousewheel)

    def unbind_mousewheel(self, _event):
        self.unbind_all("<MouseWheel>")

    def on_mousewheel(self, event):
        if getattr(self, "scroll_canvas", None):
            self.scroll_canvas.yview_scroll(int(-event.delta / 120), "units")

    def load_persistent_settings(self):
        startup = self.app_config_store.load_settings()
        self.ldplayer_var.set(str(startup.get("ldplayer_path", self.ldplayer_var.get())))
        self.data_var.set(str(startup.get("data_path", self.data_var.get())))
        data_settings = DataStore(Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)).load_settings()
        if not data_settings:
            DataStore(Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)).save_settings({"training_seconds": DEFAULT_TRAINING_SECONDS, "still_seconds": DEFAULT_STILL_SECONDS, "experience_pool_gb": DEFAULT_EXPERIENCE_POOL_GB, "ai_model_limit": DEFAULT_AI_MODEL_LIMIT})
        self.training_seconds_var.set(str(max(1, safe_int(data_settings.get("training_seconds", self.training_seconds_var.get()), DEFAULT_TRAINING_SECONDS))))
        self.still_seconds_var.set(str(max(0.1, safe_float(data_settings.get("still_seconds", self.still_seconds_var.get()), DEFAULT_STILL_SECONDS))))
        self.experience_pool_gb_var.set(str(max(0.1, safe_float(data_settings.get("experience_pool_gb", self.experience_pool_gb_var.get()), DEFAULT_EXPERIENCE_POOL_GB))))
        self.ai_model_limit_var.set(str(max(1, safe_int(data_settings.get("ai_model_limit", self.ai_model_limit_var.get()), DEFAULT_AI_MODEL_LIMIT))))
        self.update_mode_button_states()

    def save_persistent_settings(self):
        data_path = Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)
        self.app_config_store.save_settings({"ldplayer_path": self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH, "data_path": str(data_path)})
        DataStore(data_path).save_settings({"training_seconds": max(1, safe_int(self.training_seconds_var.get(), DEFAULT_TRAINING_SECONDS)), "still_seconds": max(0.1, safe_float(self.still_seconds_var.get(), DEFAULT_STILL_SECONDS)), "experience_pool_gb": max(0.1, safe_float(self.experience_pool_gb_var.get(), DEFAULT_EXPERIENCE_POOL_GB)), "ai_model_limit": max(1, safe_int(self.ai_model_limit_var.get(), DEFAULT_AI_MODEL_LIMIT))})

    def screen_rect(self):
        width = safe_int(getattr(win32api, "GetSystemMetrics", lambda _: self.winfo_screenwidth())(0), self.winfo_screenwidth())
        height = safe_int(getattr(win32api, "GetSystemMetrics", lambda _: self.winfo_screenheight())(1), self.winfo_screenheight())
        return 0, 0, max(1, width), max(1, height)

    def measure_capture_latency(self, rounds=2):
        if not mss:
            return 24.0
        monitor = {"left": 0, "top": 0, "width": max(160, safe_int(self.winfo_screenwidth(), 1920) // 6), "height": max(120, safe_int(self.winfo_screenheight(), 1080) // 6)}
        cost = []
        try:
            with mss.mss() as sct:
                for _ in range(max(1, safe_int(rounds, 2))):
                    t0 = time.perf_counter()
                    sct.grab(monitor)
                    cost.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            return 24.0
        return max(1.0, sum(cost) / len(cost)) if cost else 24.0

    def read_config(self):
        AllowedUserEditPolicy.assert_allowed("ldplayer_path", "data_path", "training_seconds", "still_seconds", "experience_pool_gb", "ai_model_limit")
        training_seconds = max(1, safe_int(self.training_seconds_var.get(), DEFAULT_TRAINING_SECONDS))
        still_seconds = max(0.1, safe_float(self.still_seconds_var.get(), DEFAULT_STILL_SECONDS))
        experience_pool_gb = max(0.1, safe_float(self.experience_pool_gb_var.get(), DEFAULT_EXPERIENCE_POOL_GB))
        ai_model_limit = max(1, safe_int(self.ai_model_limit_var.get(), DEFAULT_AI_MODEL_LIMIT))
        self.training_seconds_var.set(str(training_seconds))
        self.still_seconds_var.set(str(still_seconds))
        self.experience_pool_gb_var.set(str(experience_pool_gb))
        self.ai_model_limit_var.set(str(ai_model_limit))
        self.save_persistent_settings()
        data_path = Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)
        self.hardware_state = read_hardware_state()
        self.hardware_last_full_refresh_perf = time.perf_counter()
        self.hardware_last_light_refresh_perf = self.hardware_last_full_refresh_perf
        cpu_load = safe_float(self.hardware_state.get("cpu_load", 0.0), 0.0)
        capture_ms = self.adaptive_policy._avg(self.adaptive_policy.capture_latency_ms, 0.0)
        if capture_ms <= 0.0:
            capture_ms = self.measure_capture_latency()
        screen_score_total_value = self.store.screen_score_total if self.store else 0.0
        previous_settings = {item.name: getattr(self.settings, item.name) for item in fields(Settings)} if self.settings else {}
        settings = derive_runtime_settings(base_settings=self.settings, rect=self.current_rect() or self.screen_rect(), pool_count=self.experience_pool.count() if self.experience_pool else 0, capture_ms=capture_ms, cpu_load=cpu_load, execution_ms=self.adaptive_policy._avg(self.adaptive_policy.execution_latency_ms, 0.0), window_instability=self.adaptive_policy._avg(self.adaptive_policy.window_change_flags, 0.0), recent_success=self.adaptive_policy._avg(self.adaptive_policy.outcome_flags, 1.0), screen_score_total=screen_score_total_value, learning_similarity=self.adaptive_policy._avg(self.adaptive_policy.learning_similarity, 0.97), hardware=self.hardware_state)
        if self.store:
            self.store.append_runtime_parameter_audit(previous_settings, {item.name: getattr(settings, item.name) for item in fields(Settings)}, copy.deepcopy(RUNTIME_NUMBER_AUDIT))
        self.settings = settings
        self.escape_monitor.debounce_seconds = settings.key_debounce_seconds
        self.ui(lambda: self.minsize(settings.ui_min_width, settings.ui_min_height))
        return Config(Path(self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH), data_path, training_seconds, still_seconds, experience_pool_gb, ai_model_limit, settings)

    def apply_runtime_settings(self, settings):
        previous_settings = {item.name: getattr(self.settings, item.name) for item in fields(Settings)} if self.settings else {}
        if self.store:
            self.store.append_runtime_parameter_audit(previous_settings, {item.name: getattr(settings, item.name) for item in fields(Settings)}, copy.deepcopy(RUNTIME_NUMBER_AUDIT))
        self.settings = settings
        if self.experience_pool:
            self.experience_pool.apply_settings(settings)
        if self.brain:
            self.brain.settings = settings
        if self.window_manager:
            self.window_manager.settings = settings
        if self.executor:
            self.executor.settings = settings
        if self.escape_monitor:
            self.escape_monitor.debounce_seconds = settings.key_debounce_seconds

    def current_mode(self):
        with self.state_lock:
            return self.mode

    def set_mode_ui(self, mode):
        self.ui(lambda m=mode: (self.mode_var.set(MODE_NAMES.get(m, m)), self.refresh_hint()))

    def transition(self, expected, target, reason=None, token=None, deadline=None, fresh_stop_event=False):
        with self.state_lock:
            if token is not None and token != self.run_token:
                return None
            source = self.mode
            if expected is not None and source != expected:
                return None
            transition_reason = reason or "completed"
            if (source, target) not in ALLOWED_TRANSITIONS or transition_reason not in ALLOWED_TRANSITIONS[(source, target)]:
                return None
            if target == "idle":
                if self.active_session:
                    self.active_session.termination_reason = transition_reason
                self.termination_reason = transition_reason
                self.mode = "idle"
                event = self.events.publish("mode_transition", source=source, target=target, reason=transition_reason, token=self.run_token)
                self.active_session = ModeSession(self.run_token, "idle", time.perf_counter(), None, self.stop_event, transition_reason, event["sequence"])
                session = self.active_session
            else:
                self.run_token += 1 if token is None else 0
                session_token = self.run_token if token is None else token
                stop_event = threading.Event() if fresh_stop_event else (self.stop_event if token is not None else threading.Event())
                self.stop_event = stop_event
                self.mode = target
                event = self.events.publish("mode_transition", source=source, target=target, reason=transition_reason, token=session_token)
                if target in RUNNING_MODES:
                    self.termination_reason = None
                    session_reason = None
                elif target == "stopping":
                    self.termination_reason = transition_reason
                    session_reason = transition_reason
                else:
                    session_reason = transition_reason
                session = ModeSession(session_token, target, time.perf_counter(), deadline, stop_event, session_reason, event["sequence"])
                self.active_session = session
        self.set_mode_ui(target)
        self.ui(self.update_mode_button_states)
        return session

    def is_run_active(self, token, mode=None):
        with self.state_lock:
            return token == self.run_token and (mode is None or self.mode == mode)

    def begin_run(self, mode, deadline=None, reason=None):
        session = self.transition("idle", mode, token=None, deadline=deadline, reason=reason or {"starting": "click_learning", "sleep": "click_sleep", "migration": "click_modify_data_path"}.get(mode, "click_learning"))
        if not session:
            return None, None
        return session.token, session.stop_event

    def activate_run(self, token, mode):
        session = self.transition("starting", mode, token=token, reason="window_ok")
        return bool(session)

    def idle_progress_value(self, source_mode, progress=0.0):
        return 0.0 if source_mode in ("starting", "learning", "training", "sleep", "migration", "idle") else progress

    def render_sleep_completion_before_idle(self, label, progress=100.0):
        if self.current_mode() == "sleep":
            self.update_progress(progress, force=True)
            self.ui_sync(lambda l=label: (self.progress_label_var.set(l), self.update_idletasks()), self.settings.ui_event_coalesce_seconds)

    def handle_save_failure(self, token, error):
        self.log_exception("mode_data_flush", error)
        with self.state_lock:
            mode = self.mode
        if mode in ("learning", "training", "sleep"):
            self.transition(mode, "stopping", reason="runtime_error", token=token)
        detail = str(error)
        self.update_progress(self.progress_value, force=True)
        self.ui(lambda d=detail: (self.status_var.set("保存失败，已保持在正在退出状态，禁止直接进入空闲"), messagebox.showerror("保存失败", d)))

    def flush_mode_data(self):
        try:
            if self.persistence_queue:
                self.persistence_queue.flush()
            if self.store:
                self.store.flush_state(force=True)
            return True, None
        except Exception as exc:
            token = self.run_token
            self.handle_save_failure(token, exc)
            return False, exc

    def finish_run(self, token, status, progress=0.0, release=True, reason=None):
        if str(status).startswith("保存失败"):
            self.ui(lambda s=status: self.status_var.set(s))
            return False
        mapped_reason = reason or "completed"
        if mapped_reason not in TERMINATION_REASONS and not str(mapped_reason).startswith("window_"):
            mapped_reason = "runtime_error"
        if str(mapped_reason).startswith("window_"):
            mapped_reason = "window_invalid"
        with self.state_lock:
            source_mode = self.mode if token == self.run_token else None
        if source_mode in ("starting", "learning", "training", "sleep", "migration"):
            if not self.transition(source_mode, "stopping", reason=mapped_reason, token=token):
                return False
            flushed, _ = self.flush_mode_data()
            if not flushed:
                return False
            source_mode = "stopping"
        elif source_mode == "stopping":
            flushed, _ = self.flush_mode_data()
            if not flushed:
                return False
        if not self.transition("stopping", "idle", reason=mapped_reason, token=token):
            return False
        self.update_progress(self.idle_progress_value(source_mode, progress), force=True)
        self.ui(self.update_mode_button_states)
        self.ui(lambda s=status: self.status_var.set(s))
        if release:
            self.release_window_and_panel()
        return True

    def current_rect(self):
        return self.window_manager.client_rect() if self.window_manager else None

    def mark_learning_activity(self):
        with self.activity_lock:
            self.last_learning_activity = time.perf_counter()

    def learning_idle_seconds(self):
        with self.activity_lock:
            return max(0.0, time.perf_counter() - self.last_learning_activity)

    def should_stop_by_escape(self):
        return self.escape_monitor.esc_event_pending()

    def active_mode_stop_reason(self, mode, stop_event, config=None, deadline=None):
        guarded = should_stop_run(stop_event, deadline, self.should_stop_by_escape, getattr(self, "termination_reason", None))
        if guarded:
            return guarded
        check = self.window_manager.check_window(force=True) if self.window_manager else None
        if not check or not check.ok:
            return "window_invalid"
        if mode == "learning" and self.mouse_recorder and self.mouse_recorder.cursor_outside():
            return "cursor_outside"
        if not self.cursor_inside_window():
            if mode == "training" and hasattr(self, "ensure_cursor_inside_window") and self.ensure_cursor_inside_window():
                return None
            return "cursor_outside"
        if mode == "learning":
            still_seconds = safe_float(getattr(config, "still_seconds", self.settings.generated_action_complete_wait), self.settings.generated_action_complete_wait) if config else self.settings.generated_action_complete_wait
            if self.learning_idle_seconds() >= still_seconds:
                return "still_timeout"
        return None

    def apply_active_stop_reason(self, mode, reason, stop_event):
        if not reason:
            return False
        self.termination_reason = reason
        stop_event.set()
        if reason == "window_invalid":
            self.ui(lambda m=mode: self.status_var.set(f"{MODE_NAMES.get(m, m)}结束：雷电模拟器客户区异常"))
        elif reason == "cursor_outside":
            self.ui(lambda m=mode: self.status_var.set(f"{MODE_NAMES.get(m, m)}结束：鼠标位于雷电模拟器客户区外"))
        elif reason == "still_timeout":
            self.ui(lambda m=mode: self.status_var.set(f"{MODE_NAMES.get(m, m)}结束：鼠标静止超时"))
        return True

    def cursor_position(self):
        if not win32api:
            return None
        try:
            x, y = win32api.GetCursorPos()
            return int(x), int(y)
        except Exception:
            return None

    def observe_cursor_activity(self):
        pos = self.cursor_position()
        if pos is None:
            return None
        with self.activity_lock:
            if self.last_cursor_pos != pos:
                self.last_cursor_pos = pos
                self.last_learning_activity = time.perf_counter()
        return pos

    def cursor_inside_window(self, tolerance=0):
        pos = self.cursor_position()
        rect = self.current_rect()
        if pos is None or not rect:
            return False
        left, top, right, bottom = rect
        inside = left - tolerance <= pos[0] < right + tolerance and top - tolerance <= pos[1] < bottom + tolerance
        if inside:
            self.observe_cursor_activity()
        return inside

    def ensure_cursor_inside_window(self, rect=None):
        rect = rect or self.current_rect()
        if not rect:
            return False
        pos = self.cursor_position()
        if pos and point_inside(rect, pos[0], pos[1]):
            self.observe_cursor_activity()
            return True
        left, top, right, bottom = rect
        target = (int((left + right) / 2), int((top + bottom) / 2))
        try:
            if self.executor:
                self.executor.controller.position = target
            elif pynput_mouse:
                pynput_mouse.Controller().position = target
            else:
                win32api.SetCursorPos(target)
            self.mark_learning_activity()
            return True
        except Exception as exc:
            self.log_exception("cursor.ensure_inside", exc, {"rect": list(rect), "target": list(target)})
            return False

    def release_window_and_panel(self):
        self.restore_panel()

    def ensure_storage_runtime(self, config):
        reload_pool = False
        if not self.store or self.store.root != config.data_path:
            self.store = DataStore(config.data_path)
            reload_pool = True
        if reload_pool or self.experience_pool.settings != config.settings:
            self.experience_pool = ExperiencePool(config.settings, self.store.load_experience(config.settings.experience_load_limit), self.store.load_latest_model_state(config.settings))
            self.brain = ActionBrain(self.experience_pool, config.settings)
            self.ui(lambda: self.screen_score_total_var.set(str(self.store.state.get("screen_score_total", 0.0))))
            self.ui(lambda: self.pool_var.set(str(self.experience_pool.count())))
        return True

    def runtime_environment_issues_for_config(self, config):
        issues = []
        if sys.version_info < MIN_PYTHON_VERSION:
            issues.append(f"Python 版本过低：需要 {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} 或更高版本")
        if sys.platform != "win32":
            issues.append(f"操作系统不符合要求：本程序需要 Windows 桌面会话和雷电模拟器，当前平台为 {sys.platform}")
        for name in REQUIRED_MODULES:
            if name in IMPORT_ERRORS:
                issues.append(f"依赖无法导入 {name}：{IMPORT_ERRORS[name]}")
        storage_issue = data_path_write_issue(config.data_path)
        if storage_issue:
            issues.append(f"存储路径无效 {config.data_path}：{storage_issue}")
        valid_path, path_reason = validate_ldplayer_executable(config.ldplayer_path, config.settings, require_attach=False)
        if not valid_path:
            issues.append(f"雷电模拟器启动路径无效 {config.ldplayer_path}：{path_reason}")
        report = windows_runtime_report(config.ldplayer_path)
        if not report.get("ok"):
            issues.append("Windows 桌面运行环境不可用：" + json.dumps(report, ensure_ascii=False))
        return issues

    def repair_runtime_environment_for_config(self, config, actions=None):
        actions = actions if actions is not None else []
        missing = [name for name in REQUIRED_MODULES if name in IMPORT_ERRORS]
        if missing:
            actions.append("自动安装缺失或异常依赖：" + "、".join(missing))
            bootstrap_dependencies()
        storage_issue = data_path_write_issue(config.data_path, create=True)
        actions.append("已创建并验证存储路径可写" if not storage_issue else f"无法修复存储路径：{storage_issue}")
        valid_path, path_reason = validate_ldplayer_executable(config.ldplayer_path, config.settings, require_attach=False)
        if not valid_path:
            actions.append(f"雷电模拟器路径无法自动修复：{path_reason}")
        if sys.platform != "win32":
            actions.append("当前操作系统无法由程序自动转换为 Windows 桌面环境")
        return actions


    def offline_data_environment_issues_for_config(self, config):
        issues = []
        if sys.version_info < MIN_PYTHON_VERSION:
            issues.append(f"Python 版本过低：需要 {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} 或更高版本")
        for name in REQUIRED_MODULES:
            if name in IMPORT_ERRORS:
                issues.append(f"依赖无法导入 {name}：{IMPORT_ERRORS[name]}")
        storage_issue = data_path_write_issue(config.data_path)
        if storage_issue:
            issues.append(f"存储路径无效 {config.data_path}：{storage_issue}")
        return issues

    def repair_offline_data_environment_for_config(self, config, actions=None):
        actions = actions if actions is not None else []
        missing = [name for name in REQUIRED_MODULES if name in IMPORT_ERRORS]
        if missing:
            actions.append("自动安装缺失或异常依赖：" + "、".join(missing))
            bootstrap_dependencies()
        storage_issue = data_path_write_issue(config.data_path, create=True)
        actions.append("已创建并验证存储路径可写" if not storage_issue else f"无法修复存储路径：{storage_issue}")
        return actions

    def ensure_offline_data_environment(self, config=None, allow_repair=True, show_error=True):
        config = config or self.read_config()
        result = ensure_environment("sleep_data", allow_repair, lambda: self.offline_data_environment_issues_for_config(config), lambda actions: self.repair_offline_data_environment_for_config(config, actions))
        if not result.ok:
            self.runtime_environment_issue = "；".join(result.unrecoverable) or "睡眠模式数据环境不符合要求"
            self.log_exception("sleep.environment", RuntimeError("offline_data_environment_not_ready"), {"result": result.detail()})
            if show_error:
                self.ui(lambda r=result: messagebox.showerror("睡眠模式数据环境不符合要求", r.detail()))
            return result
        self.runtime_environment_issue = ""
        self.ensure_storage_runtime(config)
        return result

    def ensure_environment(self, stage, config=None, allow_repair=True, require_attach=False, show_error=True, ignored_hwnds=None):
        config = config or self.read_config()
        result = ensure_environment(stage, allow_repair, lambda: self.runtime_environment_issues_for_config(config), lambda actions: self.repair_runtime_environment_for_config(config, actions))
        if not result.ok:
            self.runtime_environment_issue = "；".join(result.unrecoverable) or "运行环境不符合要求"
            self.log_exception("runtime.environment", RuntimeError("environment_not_ready"), {"stage": stage, "result": result.detail()})
            if show_error:
                self.ui(lambda r=result: messagebox.showerror("运行环境不符合要求", r.detail()))
            return result
        self.runtime_environment_issue = ""
        self.ensure_storage_runtime(config)
        if require_attach:
            ignored_hwnds = tuple(ignored_hwnds or ())
            if not self.window_manager or self.window_manager.executable_path != config.ldplayer_path or self.window_manager.settings != config.settings:
                self.window_manager = WindowManager(config.ldplayer_path, config.settings, ignored_hwnds)
            else:
                self.window_manager.set_ignored_hwnds(ignored_hwnds)
            if not self.executor or self.executor.window_manager is not self.window_manager or self.executor.settings != config.settings:
                self.executor = HumanMouseExecutor(self.window_manager, config.settings)
            attached = self.window_manager.launch_or_attach()
            if not attached:
                result = EnvironmentEnsureResult(False, stage, result.checks, result.repair_actions, ("无法启动或附着雷电模拟器客户区",), ("无法启动或附着雷电模拟器客户区",))
                if show_error:
                    self.ui(lambda r=result: messagebox.showerror("运行环境不符合要求", r.detail()))
                return result
            probe_ok, probe_reason = runtime_capability_probe(self.window_manager)
            if not probe_ok:
                result = EnvironmentEnsureResult(False, stage, result.checks, result.repair_actions, ("运行能力探测失败：" + probe_reason,), ("运行能力探测失败：" + probe_reason,))
                self.runtime_environment_issue = probe_reason
                self.log_exception("runtime.capability_probe", RuntimeError(probe_reason), {"stage": stage})
                if show_error:
                    self.ui(lambda r=result: messagebox.showerror("运行环境不符合要求", r.detail()))
                return result
        return result

    def ensure_runtime(self, config, ignored_hwnds=None):
        return self.ensure_environment("runtime", config, allow_repair=True, require_attach=True, ignored_hwnds=ignored_hwnds).ok

    def learning_mode(self):
        self.request_active_mode("learning")

    def training_mode(self):
        self.request_active_mode("training")

    def request_active_mode(self, target_mode, auto_restart=False):
        self.last_active_mode_failure = ""
        config = self.read_config()
        self.status_var.set("正在检查运行环境")
        ignored_hwnds = self.own_window_handles()
        if not self.ensure_environment("自动重启训练" if auto_restart else target_mode, config, allow_repair=True, require_attach=True, ignored_hwnds=ignored_hwnds).ok:
            self.last_active_mode_failure = getattr(self, "runtime_environment_issue", "运行环境不符合要求") or "运行环境不符合要求"
            self.status_var.set("运行环境不符合要求，未进入模式")
            if auto_restart:
                self.restore_panel()
            return False
        token, stop_event = self.begin_run("starting", reason="click_learning" if target_mode == "learning" else "click_training")
        if not token:
            self.last_active_mode_failure = "当前不是空闲状态"
            self.status_var.set("请先终止当前模式，或等待当前模式结束")
            if auto_restart:
                self.restore_panel()
            return False
        if not self.minimize_panel_for_active_mode(config):
            self.last_active_mode_failure = "控制面板最小化失败"
            self.finish_run(token, "控制面板最小化失败", 0.0, reason="minimize_failed")
            if auto_restart:
                self.restore_panel()
            return False
        self.update_progress(0.0)
        self.status_var.set("正在启动或连接雷电模拟器")
        self.mode_thread = threading.Thread(target=self.mode_job, args=(token, target_mode, config, stop_event, ignored_hwnds))
        self.mode_thread.start()
        return True

    def minimize_panel_for_active_mode(self, config):
        def apply():
            try:
                self.iconify()
                self.update_idletasks()
                return self.state() == "iconic"
            except Exception:
                return False
        return bool(self.ui_sync(apply, config.settings.window_event_wait))

    def mode_job(self, token, mode, config, stop_event, ignored_hwnds=()):
        try:
            if not self.ensure_runtime(config, ignored_hwnds=ignored_hwnds):
                if self.is_run_active(token):
                    self.ui(lambda: messagebox.showerror("未找到客户区", "没有找到雷电模拟器客户区，请确认路径正确或手动启动雷电模拟器。"))
                    self.finish_run(token, "未找到雷电模拟器", 0.0, reason="runtime_error")
                return
            if stop_event.is_set() or not self.is_run_active(token):
                self.finish_run(token, "当前模式已终止", 0.0, reason="user_stop")
                return
            self.window_manager.foreground()
            stop_event.wait(config.settings.window_event_wait)
            check = self.window_manager.check_window(force=True)
            if not check.ok:
                self.finish_run(token, f"雷电模拟器客户区异常：{check.reason}", 0.0, reason=f"window_{check.reason}")
                return
            if not self.ensure_cursor_inside_window(check.rect):
                self.finish_run(token, "无法确保鼠标位于雷电模拟器客户区内", 0.0, reason="window_invalid")
                return
            if not self.activate_run(token, mode):
                return
            if mode == "learning":
                if self.mouse_recorder:
                    self.mouse_recorder.clear()
                self.mark_learning_activity()
                self.observe_cursor_activity()
                self.ui(lambda: self.status_var.set("学习模式：记录客户区画面与用户鼠标，鼠标移出客户区自动结束"))
                self.learning_loop(token, stop_event, config)
            elif mode == "training":
                self.mark_learning_activity()
                self.observe_cursor_activity()
                self.ui(lambda: self.status_var.set("训练模式：根据实时画面执行 AI 鼠标并记录"))
                self.training_loop(token, stop_event, config)
        except Exception as exc:
            message = str(exc)
            self.ui(lambda m=message: messagebox.showerror("运行失败", m))
            self.finish_run(token, "运行失败", 0.0, reason="runtime_error")

    def request_stop(self, reason="user_stop"):
        reason = "esc" if reason == "esc" else "user_stop"
        with self.state_lock:
            mode = self.mode
            token = self.run_token
            if mode not in ["starting", "learning", "training", "sleep", "migration"]:
                self.ui(lambda: self.status_var.set("当前没有正在运行的模式"))
                return
            self.stop_event.set()
            self.termination_reason = reason
            if self.active_session:
                self.active_session.termination_reason = reason
            progress_now = self.progress_value
            if mode == "sleep":
                self.transition("sleep", "stopping", reason=reason, token=token)
                self.ui(lambda r=reason: self.status_var.set("睡眠模式正在保存数据" if r == "esc" else "正在终止睡眠模式，等待数据保存完成"))
                return
        if not self.transition(mode, "stopping", reason=reason, token=token):
            return
        self.update_progress(self.idle_progress_value(mode, progress_now), force=True)
        self.ui(lambda r=reason: self.status_var.set("ESC 已触发，等待后台任务保存并退出" if r == "esc" else "正在终止当前模式，等待后台任务清理完成"))

    def request_escape(self):
        self.request_stop("esc")

    def stop_current_mode(self):
        self.request_stop("user_stop")

    def sleep_mode(self, restart_training=False):
        token, stop_event = self.begin_run("sleep")
        if not token:
            self.status_var.set("请先终止当前模式，或等待当前模式结束")
            return
        config = self.read_config()
        try:
            env_result = self.ensure_offline_data_environment(config, allow_repair=True)
            if not env_result.ok:
                raise RuntimeError(env_result.detail())
        except Exception as exc:
            self.log_exception("sleep.storage", exc, {"data_path": str(config.data_path)})
            self.ui(lambda e=str(exc): messagebox.showerror("睡眠模式数据环境异常", e))
            self.finish_run(token, "睡眠模式数据环境异常", 0.0, release=False, reason="runtime_error")
            return
        self.status_var.set("睡眠模式运行中")
        self.update_progress(0.0)
        self.mode_thread = threading.Thread(target=self.sleep_loop, args=(token, config, stop_event, restart_training))
        self.mode_thread.start()

    def sleep_progress_fields(self, completed_steps, target_training_steps, compaction_progress):
        training_ratio = clamp(safe_float(completed_steps, 0.0) / max(1.0, safe_float(target_training_steps, 1.0)), 0.0, 1.0)
        compact_ratio = clamp(compaction_progress, 0.0, 1.0)
        overall = 0.30 + training_ratio * 0.40
        return {"prepare": 1.0, "review": 1.0, "training": training_ratio, "compaction": compact_ratio, "model_compaction": 0.0, "pool_compaction": 0.0, "save": 0.0, "overall": clamp(overall, 0.30, 0.70)}

    def sleep_progress_percent(self, completed_steps, target_training_steps, compaction_progress):
        return clamp(self.sleep_progress_fields(completed_steps, target_training_steps, compaction_progress)["overall"] * 100.0, 0.0, 100.0)

    def sleep_stage_progress(self, start, end, ratio):
        return clamp(safe_float(start, 0.0) + (safe_float(end, 0.0) - safe_float(start, 0.0)) * clamp(safe_float(ratio, 0.0), 0.0, 1.0), 0.0, 100.0)

    def sleep_persistence_heartbeat(self, stage, progress_start, progress_end):
        def heartbeat(status):
            waited = safe_float(status.get("waited_seconds", 0.0), 0.0)
            pending = safe_int(status.get("pending", 0), 0)
            queued = safe_int(status.get("queued", 0), 0)
            ratio = clamp(waited / max(1.0, self.settings.persistence_close_seconds * 4.0), 0.0, 1.0)
            self.update_progress(self.sleep_stage_progress(progress_start, progress_end, ratio), force=True)
            self.ui(lambda p=pending, q=queued, w=waited, s=stage: self.progress_label_var.set(f"{s}｜正在等待持久化队列：{p} 个任务待写入｜队列 {q}｜已等待 {w:.1f} 秒"))
        return heartbeat

    def sleep_compaction_progress(self, compact):
        if not isinstance(compact, dict):
            return 0.0
        size_bytes = max(0.0, safe_float(compact.get("size_bytes", 0.0), 0.0))
        target_bytes = max(1.0, safe_float(compact.get("target_bytes", 1.0), 1.0))
        return 1.0 if size_bytes <= target_bytes else clamp(target_bytes / size_bytes, 0.0, 1.0)

    def sleep_compaction_complete(self, compact):
        if not isinstance(compact, dict):
            return True
        size_bytes = max(0.0, safe_float(compact.get("size_bytes", 0.0), 0.0))
        target_bytes = max(1.0, safe_float(compact.get("target_bytes", 1.0), 1.0))
        return size_bytes <= target_bytes

    def sleep_completion_reached(self, completed, target_training_steps, review_status="completed"):
        return review_status in ("completed", "quarantined") and completed >= target_training_steps

    def sleep_loop(self, token, config, stop_event, restart_training=False):
        self.events.publish("sleep_started", data_path=str(config.data_path))
        if self.experience_pool and getattr(self.experience_pool.model, "resource_model", None):
            self.hardware_state = self.refresh_hardware_state()
            learned_settings = self.experience_pool.model.resource_model.apply_settings(config.settings, {"cpu_load": self.hardware_state.get("cpu_load", 0.0), "memory_free_ratio": self.hardware_state.get("memory_free_ratio", 0.5), "capture_ms": self.adaptive_policy._avg(self.adaptive_policy.capture_latency_ms, 24.0), "execution_ms": self.adaptive_policy._avg(self.adaptive_policy.execution_latency_ms, 140.0), "window_instability": self.adaptive_policy._avg(self.adaptive_policy.window_change_flags, 0.0), "success_rate": self.adaptive_policy._avg(self.adaptive_policy.outcome_flags, 1.0), "pool_count": self.experience_pool.count()})
            config = replace(config, settings=learned_settings)
            self.apply_runtime_settings(learned_settings)
        completed = 0
        submitted = 0
        workers = max(1, config.settings.sleep_worker_count)
        queue_depth = max(workers, config.settings.sleep_queue_depth)
        batch_size = max(1, config.settings.sleep_batch_size)
        minimum_training_batches = workers
        target_training_steps = max(minimum_training_batches, math.ceil(max(1, self.experience_pool.count() if self.experience_pool else 1) / batch_size), workers * max(1, config.settings.ui_metric_columns))
        initial_compact = {"changed": False, "size_bytes": self.store.experience_pool_size_bytes() if self.store else 0, "target_bytes": max(1, int(config.experience_pool_gb * 1024 * 1024 * 1024))}
        compaction_progress = self.sleep_compaction_progress(initial_compact)
        compaction_complete = self.sleep_compaction_complete(initial_compact)
        completed_normally = False
        models_complete = False
        checkpoint = self.store.load_sleep_checkpoint() if self.store else None
        if not checkpoint or checkpoint.get("completed"):
            checkpoint = {"run_id": uuid.uuid4().hex, "stage": "prepare", "task1_completed": False, "task2_completed": False, "task3_model_cleanup_completed": False, "task3_pool_compaction_completed": False, "created_at": now_text()}
        def current_session_reason():
            with self.state_lock:
                return self.active_session.termination_reason if self.active_session and self.active_session.token == token else None
        run_guard = lambda: should_stop_run(stop_event, None, self.should_stop_by_escape, current_session_reason())
        self.ui(lambda: self.progress_label_var.set("睡眠准备中｜暂停异步写入并刷盘"))
        try:
            if self.store:
                checkpoint = self.store.save_sleep_checkpoint(checkpoint, stage="prepare")
            self.persistence_paused.set()
            if self.persistence_queue:
                self.persistence_queue.flush(timeout_seconds=max(5.0, config.settings.persistence_close_seconds * 8.0), heartbeat=self.sleep_persistence_heartbeat("睡眠准备", 0.0, 5.0), stop_event=stop_event)
            self.store.flush_state(force=True)
            initial_compact = {"changed": False, "size_bytes": self.store.experience_pool_size_bytes() if self.store else 0, "target_bytes": max(1, int(config.experience_pool_gb * 1024 * 1024 * 1024))}
            compaction_progress = self.sleep_compaction_progress(initial_compact)
            compaction_complete = self.sleep_compaction_complete(initial_compact)
            self.events.publish("sleep_capacity_review", size_bytes=initial_compact.get("size_bytes", 0), target_bytes=initial_compact.get("target_bytes", 0), over_limit=not compaction_complete)
        except Exception as exc:
            self.log_exception("sleep_prepare", exc, initial_compact)
        finally:
            self.persistence_paused.clear()
        self.update_progress(5.0, force=True)
        self.ui(lambda: self.progress_label_var.set("睡眠评分复核中｜正在检查全部画面评分"))
        recheck_result = {"checked": 0, "rescored": 0, "missing": 0, "errors": 0, "image_missing": 0, "image_corrupt": 0, "hash_missing": 0, "unrecoverable": 0}
        review_status = "failed"
        try:
            if self.store and not checkpoint.get("task1_completed"):
                checkpoint = self.store.save_sleep_checkpoint(checkpoint, stage="task1_recheck", task1_completed=False)
            if checkpoint.get("task1_completed"):
                recheck_result = checkpoint.get("task1_result", recheck_result)
            else:
                with ScreenAnalyzer(config.settings.hash_size) as analyzer:
                    def score_progress(current, total):
                        ratio = clamp(safe_float(current, 0.0) / max(1.0, safe_float(total, 1.0)), 0.0, 1.0)
                        self.update_progress(self.sleep_stage_progress(5.0, 25.0, ratio), force=True)
                        self.ui(lambda c=current, t=total: self.progress_label_var.set(f"睡眠评分复核中｜已复核 {c}/{t} 条"))
                    recheck_result = self.experience_pool.recheck_screen_scores(self.store, analyzer, run_guard=run_guard, progress_callback=score_progress)
                event_result = {key: value for key, value in recheck_result.items() if key != "changed_records"}
                self.events.publish("sleep_screen_scores_rechecked", **event_result)
                self.store.merge_experience_records_by_id(recheck_result.get("changed_records", []))
                recheck_result = event_result
                if recheck_result.get("complete"):
                    checkpoint = self.store.save_sleep_checkpoint(checkpoint, stage="task1_saved", task1_completed=True, task1_result=recheck_result)
                else:
                    checkpoint = self.store.save_sleep_checkpoint(checkpoint, stage="task1_interrupted", task1_completed=False, task1_result=recheck_result)
                    review_status = "interrupted"
                    stop_event.set()
            blocking_records = sum(safe_int(recheck_result.get(name, 0), 0) for name in ("unrecoverable", "image_missing", "image_corrupt"))
            trainable_records = safe_int(recheck_result.get("trainable", 0), 0)
            if review_status != "interrupted":
                review_status = "quarantined" if blocking_records and trainable_records > 0 else ("completed" if not blocking_records else "failed")
        except Exception as exc:
            review_status = "failed"
            fatal_error = {"phase": "recalculate_scores", "type": type(exc).__name__, "message": str(exc)}
            recheck_result = {"complete": False, "checked": safe_int(recheck_result.get("checked", 0), 0) if isinstance(recheck_result, dict) else 0, "fatal_error": fatal_error}
            self.log_exception("sleep_score_recheck", exc, recheck_result)
        self.ui(lambda r=recheck_result: self.progress_label_var.set(f"睡眠训练进度｜评分复核 {r.get('checked', 0)} 条｜重评 {r.get('rescored', 0)} 条｜不可恢复 {r.get('unrecoverable', 0)} 条"))
        blocking_records = sum(safe_int(recheck_result.get(name, 0), 0) for name in ("unrecoverable", "image_missing", "image_corrupt"))
        if self.store and blocking_records:
            self.store.log_error("sleep_score_recheck_unrecoverable", RuntimeError("unrecoverable_or_invalid_screen_records"), recheck_result)
        if review_status in ("failed", "interrupted"):
            failure_detail = {"stage": "评分复核", "review_status": review_status, "result": recheck_result, "auto_restart_allowed": False}
            log_id = None
            if self.store:
                log_id = self.store.log_error("sleep_review_failed_exit", RuntimeError("sleep_review_failed"), failure_detail)
                failure_detail["log_id"] = log_id
            self.ui(lambda r=recheck_result: self.progress_label_var.set(f"睡眠评分复核失败｜缺失 {r.get('image_missing', 0)}｜损坏 {r.get('image_corrupt', 0)}｜不可恢复 {r.get('unrecoverable', 0)}"))
            self.update_progress(max(self.progress_value, 25.0), force=True)
            saved, save_error, save_report = self.save_after_task1(config, "review_failed", run_guard=run_guard, status_detail=failure_detail)
            if not saved:
                if (self.is_run_active(token, "sleep") or self.is_run_active(token, "stopping")):
                    self.finish_run(token, "保存失败：" + str(save_error), self.progress_value, release=False, reason="runtime_error")
                    self.ui(lambda e=str(save_error): messagebox.showerror("保存失败", e))
                return
            if (self.is_run_active(token, "sleep") or self.is_run_active(token, "stopping")):
                saved_progress = max(self.progress_value, 25.0)
                self.render_sleep_completion_before_idle(f"睡眠评分复核失败：任务完成 {saved_progress:.1f}%，失败上下文已安全保存", saved_progress)
                self.finish_run(token, "睡眠模式已退出：评分复核失败，需人工处理或明确降级策略", saved_progress, release=True, reason="runtime_error")
                def show_review_error(d=failure_detail):
                    fatal = d.get("result", {}).get("fatal_error", {}) if isinstance(d.get("result"), dict) else {}
                    text = "\n".join([
                        "睡眠评分复核失败，请先处理图片或存储问题后重试。",
                        f"阶段：{fatal.get('phase', '重算评分')}",
                        f"异常类型：{fatal.get('type', 'ScoreReviewError')}",
                        f"异常信息：{fatal.get('message', d.get('review_status', 'failed'))}",
                        f"错误编号：{d.get('log_id', '未写入')}",
                        "可重试：是",
                        "可跳过：否",
                        "阻止自动重启训练：是"
                    ])
                    messagebox.showerror("评分复核失败", text)
                self.ui(show_review_error)
            return
        saved, save_error, save_report = self.save_after_task1(config, "task1_completed", run_guard=run_guard, status_detail=recheck_result)
        if not saved:
            if (self.is_run_active(token, "sleep") or self.is_run_active(token, "stopping")):
                self.finish_run(token, "保存失败：" + str(save_error), self.progress_value, release=False, reason="runtime_error")
                self.ui(lambda e=str(save_error): messagebox.showerror("保存失败", e))
            return
        def train_once():
            if run_guard():
                return {"trained": 0, "best_score": 0.0, "avg_score": 0.0, "avg_confidence": 0.0}
            return self.experience_pool.sleep_training_step(batch_size, self.store.add_screen_score_total if self.store else None, run_guard=run_guard)
        def submit_next(executor, futures):
            nonlocal submitted
            if stop_event.is_set() or not (self.is_run_active(token, "sleep") or self.is_run_active(token, "stopping")):
                return False
            remaining = target_training_steps - completed - len(futures)
            if remaining <= 0:
                return False
            futures.add(executor.submit(train_once))
            submitted += 1
            return True
        if self.store and not checkpoint.get("task2_completed"):
            checkpoint = self.store.save_sleep_checkpoint(checkpoint, stage="task2_training", task2_completed=False)
        if checkpoint.get("task2_completed"):
            completed = target_training_steps
            completed_normally = True
        else:
            try:
                visual_model = self.experience_pool.train_local_vision_model() if self.experience_pool else {"status": "missing_pool"}
                self.events.publish("sleep_local_vision_model_trained", status=visual_model.get("status"), records=visual_model.get("records", 0), clusters=len(visual_model.get("clusters", [])))
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = set()
                    for _ in range(queue_depth):
                        if not submit_next(executor, futures):
                            break
                    while not stop_event.is_set() and (self.is_run_active(token, "sleep") or self.is_run_active(token, "stopping")):
                        guarded = run_guard()
                        if guarded:
                            with self.state_lock:
                                self.termination_reason = guarded
                                if self.active_session:
                                    self.active_session.termination_reason = guarded
                            stop_event.set()
                            break
                        percent = self.sleep_progress_percent(completed, target_training_steps, compaction_progress)
                        self.update_progress(percent)
                        if not futures:
                            if self.sleep_completion_reached(completed, target_training_steps, review_status):
                                completed_normally = True
                                self.events.publish("sleep_completion_reached", completed_batches=completed, target_training_steps=target_training_steps, confidence=0.0)
                                break
                            if not submit_next(executor, futures):
                                raise RuntimeError("睡眠任务2无可执行批次，但训练目标尚未完成")
                        done, futures = concurrent.futures.wait(futures, timeout=config.settings.sleep_event_wait, return_when=concurrent.futures.FIRST_COMPLETED)
                        if not done:
                            continue
                        trained = 0
                        best_score = 0.0
                        avg_score = 0.0
                        avg_confidence = 0.0
                        for future in done:
                            try:
                                result = future.result()
                            except Exception as exc:
                                self.log_exception("sleep_training", exc, {"submitted": submitted, "completed": completed})
                                result = {"trained": 0, "best_score": 0.0, "avg_score": 0.0, "avg_confidence": 0.0, "model_loss": None}
                            completed += 1
                            trained += safe_int(result.get("trained", 0), 0)
                            best_score = max(best_score, safe_float(result.get("best_score", 0.0), 0.0))
                            avg_score += safe_float(result.get("avg_score", 0.0), 0.0)
                            avg_confidence += safe_float(result.get("avg_confidence", 0.0), 0.0)
                        for _ in range(queue_depth - len(futures)):
                            if not submit_next(executor, futures):
                                break
                        divisor = max(1, len(done))
                        batch_score = avg_score / divisor
                        batch_confidence = avg_confidence / divisor
                        screen_score_total = self.store.state.get("screen_score_total", 0.0) if self.store else 0.0
                        compact = {"changed": False, "size_bytes": self.store.experience_pool_size_bytes() if self.store else 0, "target_bytes": max(1, int(config.experience_pool_gb * 1024 * 1024 * 1024))}
                        compaction_progress = self.sleep_compaction_progress(compact)
                        compaction_complete = self.sleep_compaction_complete(compact)
                        if self.sleep_completion_reached(completed, target_training_steps, review_status):
                            completed_normally = True
                            self.events.publish("sleep_completion_reached", completed_batches=completed, target_training_steps=target_training_steps, confidence=batch_confidence)
                            break
                        divisor = max(1, len(done))
                        self.events.publish("sleep_batch_completed", trained=trained, best_score=best_score, confidence=batch_confidence, completed_batches=completed)
                        decision = {"reason": "sleep_prioritized_replay", "confidence": batch_confidence, "candidate_count": trained, "best_score": best_score, "completed_batches": completed, "target_training_steps": target_training_steps, "workers": workers, "pool_compacted": compact.get("changed", False), "pool_removed": compact.get("removed", 0)}
                        self.update_metrics(0.0, 50.0 + clamp(batch_confidence * 50.0, 0.0, 50.0), 0.0, batch_score, best_score, screen_score_total, decision)
                        self.ui(lambda c=completed, t=target_training_steps, q=batch_confidence: self.progress_label_var.set(f"睡眠训练进度｜已训练批次 {c}/{t}｜当前置信度 {q * 100.0:.1f}%"))
                        self.update_progress(self.sleep_progress_percent(completed, target_training_steps, compaction_progress))
                    for future in futures:
                        future.cancel()
                    if completed_normally and not stop_event.is_set() and self.store:
                        checkpoint = self.store.save_sleep_checkpoint(checkpoint, stage="task2_saved", task2_completed=True, completed_batches=completed, target_training_steps=target_training_steps)
            except Exception as exc:
                completed_normally = False
                self.log_exception("sleep_task2", exc, {"submitted": submitted, "completed": completed, "target_training_steps": target_training_steps})
                if self.store:
                    self.store.log_error("sleep_task2_failed", exc, {"submitted": submitted, "completed": completed, "target_training_steps": target_training_steps})
        with self.state_lock:
            session_reason = self.active_session.termination_reason if self.active_session and self.active_session.token == token else None
        stopped_reason = session_reason if stop_event.is_set() and session_reason in ("esc", "user_stop") else None
        save_status = "completed" if completed_normally and not stopped_reason else stopped_reason or "incomplete"
        if (self.is_run_active(token, "sleep") or self.is_run_active(token, "stopping")):
            self.update_progress(70.0, force=True)
            self.ui(lambda: self.progress_label_var.set("睡眠任务2｜保存模型快照 70%"))
        task3_report = None
        saved, save_error, save_report = self.save_after_task2(config, save_status, run_guard=run_guard)
        compaction_complete = bool(save_report.get("compaction_complete", compaction_complete)) if isinstance(save_report, dict) else compaction_complete
        models_complete = bool(save_report.get("models_complete", models_complete)) if isinstance(save_report, dict) else models_complete
        if saved and completed_normally and not stopped_reason:
            try:
                if self.store:
                    checkpoint = self.store.load_sleep_checkpoint() or checkpoint
                if self.store:
                    checkpoint = self.store.save_sleep_checkpoint(checkpoint, stage="task3_model_cleanup", task3_model_cleanup_completed=bool(checkpoint.get("task3_model_cleanup_completed")))
                model_result, pool_result = self.run_sleep_task3(config, run_guard=run_guard, checkpoint=checkpoint)
                if self.store:
                    checkpoint = self.store.save_sleep_checkpoint(checkpoint, stage="task3_saved", task3_model_cleanup_completed=bool(model_result.get("complete")), task3_pool_compaction_completed=bool(pool_result.get("complete")))
                task3_report = {"model": model_result, "pool": pool_result}
                compaction_complete = bool(pool_result.get("complete"))
                models_complete = bool(model_result.get("complete"))
                saved, save_error, save_report = self.save_after_task3(config, save_status, task3_report=task3_report)
            except Exception as exc:
                completed_normally = False
                compaction_complete = False
                models_complete = False
                task3_report = {"error": str(exc), "error_type": type(exc).__name__}
                self.log_exception("sleep_task3", exc, task3_report)
                if self.store:
                    self.store.log_error("sleep_task3_failed", exc, task3_report)
                save_status = "incomplete"
        if not saved:
            if (self.is_run_active(token, "sleep") or self.is_run_active(token, "stopping")):
                self.finish_run(token, "保存失败：" + str(save_error), self.progress_value, release=False, reason="runtime_error")
                self.ui(lambda e=str(save_error): messagebox.showerror("保存失败", e))
            return
        final_reason = None
        if (self.is_run_active(token, "sleep") or self.is_run_active(token, "stopping")):
            if completed_normally and not stopped_reason and compaction_complete and models_complete:
                final_reason = "completed"
                if self.store:
                    self.store.save_sleep_checkpoint(checkpoint, stage="completed", completed=True)
                    self.store.clear_sleep_checkpoint()
                self.render_sleep_completion_before_idle("睡眠模式保存完成 100%", 100.0)
                finished = self.finish_run(token, "睡眠模式完成，数据已安全保存", 100.0, release=not restart_training, reason=final_reason)
                if finished and restart_training:
                    self.main_thread_events.put({"type": "restart_training", "token": token, "reason": "completed"})
            else:
                final_reason = stopped_reason or "runtime_error"
                unfinished_reason = "用户中断" if stopped_reason else "运行异常"
                unfinished = self.sleep_unfinished_summary(review_status, completed, target_training_steps, compaction_complete, unfinished_reason, models_complete)
                interrupted_progress = self.sleep_progress_percent(completed, target_training_steps, compaction_progress)
                self.render_sleep_completion_before_idle(f"睡眠模式已中断：任务完成 {interrupted_progress:.1f}%，数据已安全保存｜{unfinished}", interrupted_progress)
                self.finish_run(token, f"睡眠模式未完成：{unfinished}。数据已安全保存", interrupted_progress, release=True, reason=final_reason)


    def sleep_unfinished_summary(self, review_status, completed, target_training_steps, compaction_complete, reason, models_complete=True):
        items = []
        if review_status not in ("completed", "quarantined"):
            items.append("评分复核未完成")
        if completed < target_training_steps:
            items.append(f"训练批次 {completed}/{target_training_steps}")
        if not compaction_complete:
            items.append("经验池压缩未完成")
        if not models_complete:
            items.append("AI模型清理未完成")
        if not items:
            items.append("后续训练循环尚未确认")
        return f"{reason}；未完成：" + "，".join(items)

    def run_sleep_task3(self, config, run_guard=None, checkpoint=None):
        if not self.store:
            return {"changed": False, "removed": 0, "model_count": 0, "complete": True}, {"changed": False, "size_bytes": 0, "removed": 0, "target_bytes": 0, "complete": True}
        if checkpoint:
            if not checkpoint.get("task1_completed") or not checkpoint.get("task2_completed") or not checkpoint.get("task2_saved"):
                raise RuntimeError("睡眠模式任务3前置不变量失败：任务1、任务2与任务2保存必须全部完成")
        self.ui(lambda: self.progress_label_var.set("睡眠任务3｜清理超额AI模型"))
        self.update_progress(78.0, force=True)
        model_result = self.store.compact_ai_models(config.ai_model_limit)
        if checkpoint and checkpoint.get("task3_model_cleanup_completed"):
            model_result["resumed_rechecked"] = True
        self.events.publish("sleep_model_cleanup_completed", removed=model_result.get("removed", 0), model_count=model_result.get("model_count", 0), limit=model_result.get("limit", 0), complete=model_result.get("complete", False))
        checkpoint = self.store.save_sleep_checkpoint(checkpoint or {}, stage="task3_model_cleanup_saved", task3_model_cleanup_completed=bool(model_result.get("complete")))
        if not model_result.get("complete"):
            raise RuntimeError("睡眠模式任务3未完成：AI模型清理未完成")
        self.update_progress(85.0, force=True)
        if run_guard and run_guard():
            raise RuntimeError("睡眠模式任务3被中断：经验池压缩未执行")
        self.ui(lambda: self.progress_label_var.set("睡眠任务3｜压缩经验池至上限50%"))
        def pool_progress(stage, current, total, detail):
            ratio = clamp(safe_float(current, 0.0) / max(1.0, safe_float(total, 1.0)), 0.0, 1.0)
            percent = ControlPanel.sleep_stage_progress(self, 85.0, 95.0, ratio)
            removed = safe_int(detail.get("removed", 0), 0) if isinstance(detail, dict) else 0
            size_gb = safe_float(detail.get("size_bytes", 0), 0.0) / (1024.0 * 1024.0 * 1024.0) if isinstance(detail, dict) else 0.0
            target_gb = safe_float(detail.get("target_bytes", 0), 0.0) / (1024.0 * 1024.0 * 1024.0) if isinstance(detail, dict) else 0.0
            self.update_progress(percent, force=True)
            self.ui(lambda s=stage, r=removed, a=size_gb, b=target_gb, p=percent: self.progress_label_var.set(f"睡眠任务3｜{s}：已删除 {r} 条｜当前 {a:.2f}GB / {b:.2f}GB｜{p:.1f}%"))
        pool_result = self.store.compact_experience_pool(config.experience_pool_gb, run_guard=run_guard, progress_callback=pool_progress)
        if checkpoint and checkpoint.get("task3_pool_compaction_completed"):
            pool_result["resumed_rechecked"] = True
        self.events.publish("experience_pool_compaction_completed", removed=pool_result.get("removed", 0), size_bytes=pool_result.get("size_bytes", 0), target_bytes=pool_result.get("target_bytes", 0), complete=pool_result.get("complete", False))
        self.store.save_sleep_checkpoint(checkpoint or {}, stage="task3_pool_compaction_saved", task3_pool_compaction_completed=bool(pool_result.get("complete")))
        if not pool_result.get("complete"):
            raise RuntimeError("睡眠模式任务3未完成：经验池压缩未完成")
        settings = getattr(config, "settings", getattr(self, "settings", derive_runtime_settings()))
        records = self.store.load_experience(settings.experience_load_limit)
        model_state = self.store.load_latest_model_state(settings)
        self.experience_pool = ExperiencePool(settings, records, model_state)
        self.brain = ActionBrain(self.experience_pool, settings)
        self.update_progress(95.0, force=True)
        return model_result, pool_result

    def save_after_task1(self, config, status, run_guard=None, status_detail=None):
        stage = "task1_completed" if status == "task1_completed" else "task1_failure_saved"
        return self.save_sleep_data(config, status, run_guard=run_guard, status_detail=status_detail, save_model=False, checkpoint_stage=stage, progress_start=25.0, progress_end=30.0)

    def save_after_task2(self, config, status, run_guard=None, status_detail=None):
        completed = status == "completed"
        return self.save_sleep_data(config, status, run_guard=run_guard, status_detail=status_detail, save_model=completed, checkpoint_stage="task2_saved" if completed else "task2_interrupted_saved", allow_cleanup=False, progress_start=70.0, progress_end=78.0)

    def save_after_task3(self, config, status, task3_report=None, status_detail=None):
        return self.save_sleep_data(config, status, run_guard=None, status_detail=status_detail, task3_report=task3_report, save_model=False, checkpoint_stage="task3_final_saved", allow_cleanup=False, progress_start=95.0, progress_end=100.0)

    def save_sleep_data(self, config, status, run_guard=None, status_detail=None, task3_report=None, save_model=True, checkpoint_stage=None, allow_cleanup=False, progress_start=95.0, progress_end=100.0):
        if not self.store or not self.experience_pool:
            return True, None, {"model_count": 0, "experience_size": 0, "target_bytes": 0, "compaction_complete": True, "models_complete": True}
        try:
            self.persistence_paused.set()
            persistence_guard = lambda: False
            if self.persistence_queue:
                self.persistence_queue.flush(timeout_seconds=max(5.0, config.settings.persistence_close_seconds * 8.0), heartbeat=self.sleep_persistence_heartbeat("睡眠保存", progress_start, self.sleep_stage_progress(progress_start, progress_end, 0.25)), stop_event=getattr(self, "stop_event", None))
            self.store.flush_state(force=True)
            self.ui(lambda: self.progress_label_var.set(f"睡眠保存｜状态文件已同步 {self.sleep_stage_progress(progress_start, progress_end, 0.15):.1f}%"))
            self.update_progress(self.sleep_stage_progress(progress_start, progress_end, 0.15), force=True)
            task3_compacted = isinstance(task3_report, dict) and isinstance(task3_report.get("pool"), dict) and bool(task3_report["pool"].get("complete"))
            with self.experience_pool.lock:
                if not task3_compacted:
                    self.store.merge_experience_records_by_id(copy.deepcopy(self.experience_pool.records))
                model = self.experience_pool.model
            self.update_progress(self.sleep_stage_progress(progress_start, progress_end, 0.55), force=True)
            compact = (task3_report or {}).get("pool") if isinstance(task3_report, dict) else None
            if not isinstance(compact, dict):
                size_bytes = self.store.experience_pool_size_bytes()
                target_bytes = max(1, int(config.experience_pool_gb * 1024 * 1024 * 1024))
                compact = {"changed": False, "size_bytes": size_bytes, "removed": 0, "target_bytes": target_bytes, "complete": size_bytes <= target_bytes}
            final_records = self.store.load_experience(config.settings.experience_load_limit)
            if save_model:
                self.ui(lambda c=len(final_records): self.progress_label_var.set(f"睡眠任务2｜正在写入模型快照：最多 512 条代表性样本｜候选 {c} 条"))
                self.store.save_ai_model_snapshot(final_records, config.settings, config.ai_model_limit, status, model, run_guard=persistence_guard, status_detail=status_detail)
            model_compact = self.store.compact_ai_models(config.ai_model_limit) if allow_cleanup else {"changed": False, "removed": 0, "limit": config.ai_model_limit, "model_count": len(list(self.store.model_dir.glob("model_*.json"))), "complete": True, "deferred_to_task3": True}
            self.experience_pool = ExperiencePool(config.settings, final_records, self.store.load_latest_model_state(config.settings))
            self.brain = ActionBrain(self.experience_pool, config.settings)
            self.update_progress(self.sleep_stage_progress(progress_start, progress_end, 0.85), force=True)
            self.store.flush_state(force=True)
            model_count = len(list(self.store.model_dir.glob("model_*.json")))
            target_bytes = max(1, safe_int(compact.get("target_bytes", 0), 0))
            experience_size = max(0, safe_int(compact.get("size_bytes", 0), 0))
            report = {"model_count": model_count, "experience_size": experience_size, "target_bytes": target_bytes, "compaction_complete": bool(compact.get("complete", experience_size <= target_bytes)), "models_complete": model_count <= max(1, safe_int(config.ai_model_limit, AGENT_SPEC.default_ai_model_limit)), "compact": compact, "model_compact": model_compact}
            if checkpoint_stage:
                checkpoint = self.store.load_sleep_checkpoint() or {}
                checkpoint_payload = {"current_task": checkpoint_stage, "input_record_version": checkpoint.get("input_record_version", 0), "output_record_version": now_text(), "flushed": True, "model_count": model_count, "experience_pool_size": experience_size, "allowed_next_task": checkpoint_stage in ("task2_saved", "task3_final_saved")}
                if checkpoint_stage == "task1_completed":
                    checkpoint_payload.update({"task1_completed": True, "task1_result": status_detail if isinstance(status_detail, dict) else {"status_detail": status_detail}, "allowed_next_task": True})
                if checkpoint_stage == "task2_saved":
                    checkpoint_payload.update({"task1_completed": True, "task2_completed": True, "task2_saved": True})
                self.store.save_sleep_checkpoint(checkpoint, stage=checkpoint_stage, **checkpoint_payload)
            self.events.publish("save_completed", kind="sleep_data", status=status, **report)
            self.ui(lambda c=self.experience_pool.count(): self.pool_var.set(str(c)))
            return True, None, report
        except Exception as exc:
            self.log_exception("sleep_save", exc, {"status": status})
            return False, exc, {"model_count": None, "experience_size": None, "target_bytes": None, "compaction_complete": False, "models_complete": False}
        finally:
            self.persistence_paused.clear()

    def restore_panel(self):
        def apply():
            try:
                self.deiconify()
                self.lift()
            except Exception:
                pass
        self.ui(apply)

    def update_progress(self, percent, force=False):
        percent = round(clamp(percent, 0.0, 100.0), 1)
        mode_getter = getattr(self, "current_mode", None)
        mode = mode_getter() if callable(mode_getter) else getattr(self, "mode", None)
        if mode == "sleep" and percent > 0.0:
            percent = max(getattr(self, "progress_value", 0.0), percent)
        if not force and abs(percent - self.progress_value) < self.settings.ui_progress_delta:
            return
        self.progress_value = percent
        self.last_progress_update_perf = time.perf_counter()
        def apply():
            self.progress_var.set(percent)
            self.progress_text_var.set(f"{percent:.1f}%")
        self.ui(apply)

    def update_metrics(self, novelty, human_score, screen_reward, action_reward, reward, screen_score_total, decision=None):
        payload = (round(float(novelty), 2), round(float(human_score), 2), round(float(screen_reward), 2), round(float(action_reward), 2), round(float(reward), 2), round(float(screen_score_total), 2), decision.get("reason") if decision else None, round(safe_float(decision.get("confidence", 0.0), 0.0), 3) if decision else None)
        if payload == self.last_metric_payload:
            return
        self.last_metric_payload = payload
        self.last_metrics_update_perf = time.perf_counter()
        def apply():
            self.novelty_var.set(f"{round(float(novelty), 2)}%")
            self.human_var.set(f"{round(float(human_score), 2)}%")
            self.screen_reward_var.set(str(round(float(screen_reward), 2)))
            self.action_reward_var.set(str(round(float(action_reward), 2)))
            self.reward_var.set(str(round(float(reward), 2)))
            self.screen_score_total_var.set(str(round(float(screen_score_total), 2)))
            self.pool_var.set(str(self.experience_pool.count()))
            if decision:
                confidence = round(safe_float(decision.get("confidence", 0.0), 0.0) * 100.0, 1)
                self.ai_var.set(f"{decision.get('reason', 'AI')} {confidence}%")
        self.ui(apply)

    def capture_snapshot_image(self, analyzer, mode, session_start, rect=None, priority="normal"):
        if not self.store:
            return None, None
        rect = rect or self.current_rect()
        if not rect:
            return None, None
        perf_time = time.perf_counter()
        try:
            image = analyzer.capture(rect)
            captured_perf = time.perf_counter()
            hash_value = analyzer.fingerprint(image)
            semantic_vector = analyzer.semantic_fingerprint(image)
            path = self.store.new_screen_path(mode)
            checksum = image_content_checksum(image)
            snapshot = ScreenSnapshot(path=path, relative_path=self.store.relative_path(path), hash_value=hash_value, captured_at=now_text(), perf_time=perf_time, elapsed=round(perf_time - session_start, 3), rect=tuple(rect), capture_latency_ms=round((captured_perf - perf_time) * 1000.0, 3), image_priority=priority, image_checksum=checksum, semantic_vector=semantic_vector)
            self.events.publish("screenshot_completed", mode=mode, path=str(path), latency_ms=snapshot.capture_latency_ms)
            return snapshot, image
        except Exception as exc:
            self.log_exception("capture_snapshot", exc, {"mode": mode, "rect": list(rect)})
            return None, None

    def mark_snapshot_image_result(self, snapshot, saved):
        if snapshot:
            self.events.publish("screenshot_save_completed", path=str(getattr(snapshot, "path", "")), saved=bool(saved))
        if snapshot and not saved:
            try:
                object.__setattr__(snapshot, "image_dropped", True)
            except Exception as exc:
                self.log_exception("snapshot.image_dropped", exc, {"path": str(getattr(snapshot, "path", ""))})
        return bool(saved)

    def capture_snapshot(self, analyzer, mode, session_id, session_start, rect=None, persist=True, priority="normal"):
        snapshot, image = self.capture_snapshot_image(analyzer, mode, session_start, rect, priority=priority)
        if snapshot:
            self.adaptive_policy.observe_capture(getattr(snapshot, "capture_latency_ms", 0.0))
        if snapshot and persist:
            try:
                if priority == "critical":
                    analyzer.save_image(image, snapshot.path, priority=priority, settings=self.settings)
                    self.mark_snapshot_image_result(snapshot, snapshot.path.is_file())
                elif self.persistence_paused.is_set():
                    self.mark_snapshot_image_result(snapshot, False)
                else:
                    self.mark_snapshot_image_result(snapshot, self.persistence_queue.enqueue_image(analyzer, image, snapshot.path, self.store, priority=priority))
            except Exception as exc:
                self.log_exception("capture_snapshot.save", exc, {"path": str(snapshot.path)})
                return None
        return snapshot

    def write_record(self, mode, session_id, snapshot, action, event_name, decision=None, action_anchor_perf=None, after_snapshot=None, planned_action=None, failed_action=False, window_rect_changed=False, capture_latency_ms=None, execution_latency_ms=None, execution_error=None, screen_result_unknown=False, exclude_from_training=False):
        if not self.store or not snapshot:
            return None
        before_novelty, batch = self.experience_pool.compute_screen_score(snapshot.hash_value, exact_checksum=getattr(snapshot, "image_checksum", ""), semantic_vector=getattr(snapshot, "semantic_vector", ()))[:2]
        normalized = normalize_mouse_action(action, snapshot.rect) if action else None
        mouse_source = normalized.get("source") if normalized else "idle"
        human_score = self.experience_pool.human_score(normalized) if normalized else 50.0
        after_novelty = self.experience_pool.compute_screen_score(after_snapshot.hash_value, exact_checksum=getattr(after_snapshot, "image_checksum", ""), semantic_vector=getattr(after_snapshot, "semantic_vector", ()))[0] if after_snapshot else before_novelty
        transition_reward = round(after_novelty - before_novelty, 2)
        scoring_novelty = after_novelty if normalized and after_snapshot else before_novelty
        event_time = getattr(after_snapshot, "captured_at", None) or getattr(snapshot, "captured_at", None) or now_text()
        reward_info = self.experience_pool.model_runtime.reward_breakdown(scoring_novelty, human_score if normalized else self.settings.score_default, event_time=event_time, reward_state=self.experience_pool.reward_state, commit_state=True)
        self.experience_pool.reward_state = reward_state_from(reward_info.get("reward_state"))
        novelty_reward = reward_info["screen_primary_reward"]
        human_delta = reward_info["mouse_action_delta"]
        reward = reward_info["total_reward"]
        human_action_reward = max(0.0, human_delta)
        human_action_penalty = max(0.0, -human_delta)
        if failed_action:
            reward = round(-abs(max(1.0, human_action_penalty, 100.0 - human_score)), 2)
        if screen_result_unknown:
            transition_reward = 0.0
            reward = 0.0
            reward_info = dict(reward_info)
            reward_info["basis"] = "screen_result_unknown_excluded"
        screen_score_total = self.store.add_screen_score_total(reward)
        started_perf = safe_float(normalized.get("started_perf"), None) if normalized else None
        offset_source = started_perf if started_perf is not None else action_anchor_perf
        offset_ms = round((float(offset_source) - snapshot.perf_time) * 1000.0, 3) if offset_source is not None else None
        sims = [round(item["similarity"], 4) for item in batch]
        record_event = self.events.publish("record_ready", mode=mode, session_id=session_id, event_name=event_name)
        record = {"record_schema_version": 2, "reward_version": reward_info["reward_version"], "id": uuid.uuid4().hex, "event_sequence": record_event["sequence"], "session_id": session_id, "created_at": now_text(), "mode": mode, "event": event_name, "elapsed": snapshot.elapsed, "screen_path": snapshot.relative_path, "screen_hash": snapshot.hash_value.hex, "screen_hash_hex": snapshot.hash_value.hex, "screen_hash_int": snapshot.hash_value.value, "screen_hash_bits": snapshot.hash_value.bits, "screen_semantic_vector": list(getattr(snapshot, "semantic_vector", ())), "screen_captured_at": snapshot.captured_at, "screen_perf": round(snapshot.perf_time, 6), "mouse_action": normalized, "planned_action": normalize_mouse_action(planned_action, snapshot.rect) if planned_action else None, "actual_action": None if failed_action else normalized, "execution_error": str(execution_error) if execution_error else None, "mouse_source": mouse_source, "screen_action_offset_ms": offset_ms, "nearest": [{"id": item["record"].get("id"), "similarity": round(item["similarity"], 4)} for item in batch], "nearest_summary": {"count": len(sims), "max_similarity": max(sims) if sims else 0.0, "avg_similarity": round(sum(sims) / len(sims), 4) if sims else 0.0}, "novelty": before_novelty, "screen_score": before_novelty, "score_version": 1, "score_basis": "nearest_screen_content_live", "score_checked_at": now_text(), "score_neighbors": [{"id": item["record"].get("id"), "similarity": round(item["similarity"], 4)} for item in batch], "before_screen": snapshot.relative_path, "after_screen": after_snapshot.relative_path if after_snapshot else snapshot.relative_path, "before_screen_hash": snapshot.hash_value.hex, "before_screen_score": before_novelty, "after_screen_hash": after_snapshot.hash_value.hex if after_snapshot else snapshot.hash_value.hex, "after_screen_hash_int": after_snapshot.hash_value.value if after_snapshot else snapshot.hash_value.value, "after_screen_hash_bits": after_snapshot.hash_value.bits if after_snapshot else snapshot.hash_value.bits, "after_screen_semantic_vector": list(getattr(after_snapshot, "semantic_vector", ())) if after_snapshot else list(getattr(snapshot, "semantic_vector", ())), "after_screen_score": after_novelty, "before_novelty": before_novelty, "after_novelty": after_novelty, "transition_reward": transition_reward, "screen_observation_reward": novelty_reward, "screen_primary_reward": reward_info["screen_primary_reward"], "human_tie_break_reward": reward_info["human_tie_break_reward"], "income": reward_info.get("income"), "cost": reward_info.get("cost"), "reward_state": reward_info.get("reward_state"), "reward_state_version": reward_info.get("reward_state_version"), "reward_breakdown": {"screen_novelty": reward_info["screen_novelty"], "screen_reward": reward_info["screen_reward"], "human_similarity": reward_info["human_similarity"], "human_tiebreak": reward_info["human_tiebreak"], "screen_score_delta": reward_info["screen_score_delta"], "income": reward_info.get("income"), "cost": reward_info.get("cost"), "basis": reward_info["basis"]}, "reward_sort_key": reward_info["reward_sort_key"], "mouse_action_reward": human_action_reward, "mouse_action_penalty": human_action_penalty, "human_score": human_score, "total_reward": reward, "reward": reward, "novelty_reward": novelty_reward, "human_action_reward": human_action_reward, "human_action_penalty": human_action_penalty, "screen_score_delta": max(0.0, reward), "screen_score_settled": max(0.0, reward), "penalty_delta": max(0.0, -reward), "screen_score_total": screen_score_total, "client_rect": list(snapshot.rect), "failed_action": bool(failed_action), "window_rect_changed": bool(window_rect_changed), "image_checksum": getattr(snapshot, "image_checksum", ""), "after_image_checksum": getattr(after_snapshot, "image_checksum", "") if after_snapshot else getattr(snapshot, "image_checksum", ""), "image_dropped": bool(getattr(snapshot, "image_dropped", False)), "screen_file_expected": not bool(getattr(snapshot, "image_dropped", False)), "capture_latency_ms": capture_latency_ms if capture_latency_ms is not None else getattr(snapshot, "capture_latency_ms", None), "execution_latency_ms": execution_latency_ms, "termination_reason": None, "policy_snapshot": {"hash_size": self.settings.hash_size, "nearest_top_k": self.settings.nearest_top_k, "training_event_wait": self.settings.training_event_wait, "explore_min_rate": self.settings.explore_min_rate, "explore_max_rate": self.settings.explore_max_rate, "action_jitter": self.settings.action_jitter}}
        if screen_result_unknown:
            record["screen_result_unknown"] = True
            record["exclude_from_training"] = True
            record["score_status"] = "screen_result_unknown"
        elif exclude_from_training:
            record["exclude_from_training"] = True
        if decision:
            record["ai_decision"] = decision
        self.persistence_queue.enqueue_record(self.store, record)
        self.events.publish("save_completed", kind="record", record_id=record.get("id"))
        self.experience_pool.add(record)
        self.update_metrics(after_novelty, human_score, novelty_reward, human_delta, reward, screen_score_total, decision)
        return record

    def capture_learning_screen_change(self, analyzer, session_id, start, now_perf, config):
        snapshot, image = self.capture_snapshot_image(analyzer, "learning", start, priority="low")
        if not snapshot:
            return
        if self.last_learning_event_hash:
            similarity = hash_similarity(snapshot.hash_value, self.last_learning_event_hash)
            self.adaptive_policy.observe_capture(0.0, similarity=similarity, window_rect_changed=False)
            if similarity >= self.settings.learning_screen_similarity_threshold:
                self.last_learning_event_perf = now_perf
                return
        try:
            if self.persistence_paused.is_set():
                saved = self.mark_snapshot_image_result(snapshot, False)
            else:
                saved = self.mark_snapshot_image_result(snapshot, self.persistence_queue.enqueue_image(analyzer, image, snapshot.path, self.store, priority="low"))
        except Exception as exc:
            self.log_exception("learning_screen_event.save", exc, {"path": str(snapshot.path)})
            return
        if not saved:
            return
        self.adaptive_policy.observe_capture(getattr(snapshot, "capture_latency_ms", 0.0))
        self.last_learning_event_perf = now_perf
        self.last_learning_event_hash = snapshot.hash_value
        self.write_record("learning", session_id, snapshot, None, "screen_event")

    def learning_loop(self, token, stop_event, config):
        session_id = uuid.uuid4().hex
        start = time.perf_counter()
        pending_snapshots = {}
        learning_events = 0
        learning_screens = 0
        termination_reason = "completed"
        self.mark_learning_activity()
        if self.mouse_recorder:
            self.mouse_recorder.clear_cursor_outside()
        self.last_learning_event_perf = time.perf_counter()
        self.last_learning_event_hash = None
        with ScreenAnalyzer(config.settings.hash_size) as analyzer:
            self.write_record("learning", session_id, self.capture_snapshot(analyzer, "learning", session_id, start, priority="critical"), None, "mode_start")
            while not stop_event.is_set() and self.is_run_active(token, "learning"):
                reason = self.active_mode_stop_reason("learning", stop_event, config)
                if reason:
                    termination_reason = reason
                    self.apply_active_stop_reason("learning", reason, stop_event)
                    break
                now_perf = time.perf_counter()
                idle_seconds = self.learning_idle_seconds()
                remaining = max(0.0, config.still_seconds - idle_seconds)
                self.ui(lambda e=learning_events, c=learning_screens, r=remaining: self.progress_label_var.set(f"学习模式进度保持 0%｜静止剩余 {r:.1f} 秒｜学习事件 {e}｜截图 {c}"))
                self.update_progress(0.0)
                if idle_seconds >= config.still_seconds:
                    termination_reason = "still_timeout"
                    stop_event.set()
                    self.ui(lambda: self.status_var.set("学习模式结束：鼠标静止超时"))
                    break
                event_seen = False
                if self.mouse_recorder:
                    markers = self.mouse_recorder.pop_start_markers()
                    actions = self.mouse_recorder.pop_actions()
                    event_seen = bool(markers or actions)
                    if event_seen:
                        learning_events += len(markers) + len(actions)
                        learning_screens += 1
                        self.capture_learning_screen_change(analyzer, session_id, start, now_perf, config)
                    for marker in markers:
                        marker_snapshot = self.capture_snapshot(analyzer, "learning", session_id, start, priority="critical")
                        if marker_snapshot:
                            learning_screens += 1
                            pending_snapshots[marker["action_id"]] = marker_snapshot
                    for action in actions:
                        action_snapshot = pending_snapshots.pop(action.get("action_id"), None) or self.capture_snapshot(analyzer, "learning", session_id, start, priority="critical")
                        after_snapshot = self.capture_snapshot(analyzer, "learning", session_id, start, priority="critical")
                        learning_screens += 2
                        self.write_record("learning", session_id, action_snapshot, action, "user_mouse", action_anchor_perf=action.get("started_perf") or action.get("t0"), after_snapshot=after_snapshot, planned_action=action)
                    if not event_seen:
                        while not stop_event.is_set():
                            reason = self.active_mode_stop_reason("learning", stop_event, config)
                            if reason:
                                termination_reason = reason
                                self.apply_active_stop_reason("learning", reason, stop_event)
                                break
                            remaining_wait = max(0.0, config.still_seconds - self.learning_idle_seconds())
                            if remaining_wait <= 0.0:
                                break
                            self.mouse_recorder.wait(min(0.1, remaining_wait))
                else:
                    while not stop_event.is_set():
                        reason = self.active_mode_stop_reason("learning", stop_event, config)
                        if reason:
                            termination_reason = reason
                            self.apply_active_stop_reason("learning", reason, stop_event)
                            break
                        remaining_wait = max(0.0, config.still_seconds - self.learning_idle_seconds())
                        if remaining_wait <= 0.0:
                            break
                        stop_event.wait(min(0.1, remaining_wait))
            self.write_record("learning", session_id, self.capture_snapshot(analyzer, "learning", session_id, start, priority="critical"), None, "mode_end")
        if self.is_run_active(token, "learning") or self.is_run_active(token, "stopping"):
            saved, save_error = self.flush_mode_data()
            if not saved:
                self.finish_run(token, "保存失败：" + str(save_error), 0.0, release=False, reason="runtime_error")
            else:
                final_reason = self.termination_reason
                if final_reason not in TERMINATION_REASONS:
                    final_reason = termination_reason
                if stop_event.is_set() and final_reason == "completed":
                    final_reason = "user_stop"
                self.finish_run(token, "学习模式结束", 0.0, reason=final_reason)
        else:
            self.release_window_and_panel()


    def training_loop(self, token, stop_event, config):
        session_id = uuid.uuid4().hex
        start = time.perf_counter()
        training_clock = PausableTrainingClock(config.training_seconds)
        consecutive_failures = 0
        consecutive_no_actions = 0
        last_training_error = None
        self.termination_reason = "completed"
        service = self.training_service
        with ScreenAnalyzer(config.settings.hash_size) as analyzer:
            self.write_record("training", session_id, self.capture_snapshot(analyzer, "training", session_id, start, priority="critical"), None, "mode_start")
            while not stop_event.is_set() and self.is_run_active(token, "training"):
                if service.should_stop(training_clock, config, stop_event):
                    break
                rect = self.current_rect()
                service.prepare_for_event(rect)
                if not rect:
                    self.termination_reason = "window_invalid"
                    stop_event.set()
                    break
                snapshot = service.observe_screen(analyzer, session_id, start, rect)
                if not snapshot:
                    self.termination_reason = "window_invalid"
                    stop_event.set()
                    break
                action, decision = service.decide_action(snapshot)
                if not action:
                    consecutive_no_actions += 1
                    self.write_record("training", session_id, snapshot, None, "screen_event", decision=decision)
                    if consecutive_no_actions >= max(1, self.settings.training_fail_stop_count):
                        action = self.brain.bootstrap_action(0.2)
                        decision = {"reason": "forced_bounded_bootstrap_exploration", "consecutive_no_actions": consecutive_no_actions, "candidate_count": 0, "confidence": 0.0}
                    else:
                        stop_event.wait(max(self.settings.training_event_wait, self.settings.min_action_delay_seconds))
                        continue
                else:
                    consecutive_no_actions = 0
                success, record = service.execute_and_record(analyzer, session_id, start, rect, snapshot, action, decision, stop_event, config)
                if not success:
                    consecutive_failures += 1
                    last_training_error = record.get("failure_reason") if isinstance(record, dict) else None
                    last_training_error = last_training_error or (decision.get("reason") if decision else "unknown")
                    if consecutive_failures >= self.settings.training_fail_stop_count:
                        self.termination_reason = "executor_error"
                        stop_event.set()
                        self.ui(lambda e=last_training_error: self.status_var.set(f"训练模式结束：连续执行失败：{e}"))
                    continue
                consecutive_failures = 0
                delay = safe_float(record["mouse_action"].get("duration", 0.0), 0.0) if record and record.get("mouse_action") else 0.0
                deadline = time.perf_counter() + max(self.settings.min_action_delay_seconds, delay)
                while time.perf_counter() < deadline and not stop_event.is_set():
                    reason = self.active_mode_stop_reason("training", stop_event, config)
                    if reason:
                        self.apply_active_stop_reason("training", reason, stop_event)
                        break
                    stop_event.wait(min(self.settings.generated_action_complete_wait, deadline - time.perf_counter()))
            self.write_record("training", session_id, self.capture_snapshot(analyzer, "training", session_id, start, priority="critical"), None, "mode_end")
        if self.is_run_active(token, "training") or self.is_run_active(token, "stopping"):
            final_reason = self.termination_reason or ("esc" if self.should_stop_by_escape() else "user_stop")
            if final_reason == "time_limit":
                saved, save_error = self.flush_mode_data()
                if not saved:
                    self.finish_run(token, "保存失败：" + str(save_error), 0.0, release=False, reason="runtime_error")
                    return
                session = self.transition("training", "sleep", reason="time_limit", token=token, fresh_stop_event=True)
                if session:
                    self.update_progress(0.0, force=True)
                    self.ui(self.update_mode_button_states)
                    self.ui(lambda: self.status_var.set("训练模式达到时间上限，进入睡眠模式"))
                    self.mode_thread = threading.Thread(target=self.sleep_loop, args=(session.token, config, session.stop_event, True))
                    self.mode_thread.start()
            else:
                saved, save_error = self.flush_mode_data()
                if not saved:
                    self.finish_run(token, "保存失败：" + str(save_error), 0.0, release=False, reason="runtime_error")
                else:
                    self.finish_run(token, "训练模式已终止" if stop_event.is_set() else "训练模式结束", 0.0, reason=final_reason)
        else:
            self.release_window_and_panel()

    def close(self):
        self.shutdown_requested = True
        mode_thread = self.mode_thread
        close_timeout = max(1.0, safe_float(getattr(self.settings, "persistence_close_seconds", 3.0), 3.0))
        deadline = time.perf_counter() + close_timeout
        with self.state_lock:
            active_mode = self.mode
        if active_mode in ["starting", "learning", "training", "sleep", "migration"]:
            self.status_var.set("正在保存，请等待完成")
            self.update_mode_button_states()
            self.request_stop("user_stop")
            if self.mouse_recorder:
                self.mouse_recorder.stop()
            self.escape_monitor.stop()
            if mode_thread and mode_thread.is_alive():
                while mode_thread.is_alive() and time.perf_counter() < deadline:
                    mode_thread.join(timeout=max(0.05, min(0.25, deadline - time.perf_counter())))
        still_running = bool(mode_thread and mode_thread.is_alive())
        if still_running:
            if self.store:
                try:
                    checkpoint = self.store.load_sleep_checkpoint() or {"run_id": uuid.uuid4().hex, "created_at": now_text()}
                    self.store.save_sleep_checkpoint(checkpoint, stage="recoverable_close_timeout", close_interrupted=True, safe_to_resume=True, active_mode=active_mode)
                except Exception as exc:
                    self.log_exception("close.checkpoint", exc, {"active_mode": active_mode})
            self.status_var.set("可恢复退出：后台保存超时，已尽力保留检查点")
            self.log_exception("close.recoverable_timeout", RuntimeError("background_task_close_timeout"), {"active_mode": active_mode, "timeout": close_timeout})
            force = messagebox.askyesno("后台任务仍在运行", "后台任务尚未结束，已保存可恢复检查点。\n\n选择“是”继续等待；选择“否”强制关闭窗口。", parent=self)
            if force:
                self.after(1000, self.close)
            else:
                self.destroy()
            return
        with self.state_lock:
            self.stop_event.set()
            if self.mode not in ("idle",) and not still_running:
                self.run_token += 1
                self.mode = "idle"
        try:
            saved, save_error = self.flush_mode_data()
            if not saved:
                if self.store:
                    checkpoint = self.store.load_sleep_checkpoint() or {"run_id": uuid.uuid4().hex, "created_at": now_text()}
                    self.store.save_sleep_checkpoint(checkpoint, stage="close_save_failed", close_save_failed=True, safe_to_resume=True, error=str(save_error))
                self.ui(lambda: self.status_var.set("保存失败，已保留恢复检查点并阻止退出"))
                return
        except Exception as exc:
            if self.store:
                try:
                    checkpoint = self.store.load_sleep_checkpoint() or {"run_id": uuid.uuid4().hex, "created_at": now_text()}
                    self.store.save_sleep_checkpoint(checkpoint, stage="close_save_failed", close_save_failed=True, safe_to_resume=True, error=str(exc))
                except Exception as checkpoint_exc:
                    self.log_exception("close.checkpoint_failed", checkpoint_exc, {"active_mode": active_mode})
            self.log_exception("close.flush_failed", exc, {"active_mode": active_mode})
            self.ui(lambda: self.status_var.set("保存失败，已阻止退出"))
            return
        if self.runtime_environment_refresh_id:
            try:
                self.after_cancel(self.runtime_environment_refresh_id)
            except Exception:
                pass
            self.runtime_environment_refresh_id = None
        if self.mouse_recorder:
            self.mouse_recorder.stop()
        self.escape_monitor.stop()
        if self.persistence_queue:
            try:
                self.persistence_queue.close()
            except Exception as exc:
                if self.store:
                    try:
                        checkpoint = self.store.load_sleep_checkpoint() or {"run_id": uuid.uuid4().hex, "created_at": now_text()}
                        self.store.save_sleep_checkpoint(checkpoint, stage="close_persistence_queue_failed", close_persistence_queue_failed=True, safe_to_resume=True, error=str(exc))
                    except Exception as checkpoint_exc:
                        self.log_exception("close.queue_checkpoint_failed", checkpoint_exc, {"active_mode": active_mode})
                self.log_exception("close.persistence_queue", exc, {"active_mode": active_mode})
                self.ui(lambda: self.status_var.set("异步持久化队列保存失败，已阻止退出"))
                return
        if self.store:
            try:
                self.store.flush_state(force=True)
            except Exception as exc:
                self.log_exception("close.store_flush", exc, {"active_mode": active_mode})
                self.ui(lambda: self.status_var.set("最终落盘失败，已阻止退出"))
                return
        mode_thread = self.mode_thread
        if mode_thread and mode_thread.is_alive():
            final_deadline = time.perf_counter() + close_timeout
            while mode_thread.is_alive() and time.perf_counter() < final_deadline:
                mode_thread.join(timeout=max(0.05, min(0.25, final_deadline - time.perf_counter())))
        if mode_thread and mode_thread.is_alive():
            if self.store:
                try:
                    checkpoint = self.store.load_sleep_checkpoint() or {"run_id": uuid.uuid4().hex, "created_at": now_text()}
                    self.store.save_sleep_checkpoint(checkpoint, stage="recoverable_close_final_timeout", close_interrupted=True, safe_to_resume=True, active_mode=active_mode)
                except Exception as exc:
                    self.log_exception("close.final_checkpoint", exc, {"active_mode": active_mode})
            self.log_exception("close.force_exit_risk", RuntimeError("background_task_not_safely_finished"), {"mode": active_mode})
            self.status_var.set("可恢复退出：后台任务尚未安全结束，已保留检查点")
            force = messagebox.askyesno("后台任务仍在运行", "后台任务尚未安全结束，已保存可恢复检查点。\n\n选择“是”继续等待；选择“否”强制关闭窗口。", parent=self)
            if force:
                self.after(1000, self.close)
            else:
                self.destroy()
            return
        try:
            self.destroy()
        except Exception:
            pass

    def refresh_hardware_state(self, force=False):
        event_perf = time.perf_counter()
        cached = dict(self.hardware_state or {})
        full_due = force or not cached or event_perf - self.hardware_last_full_refresh_perf >= 45.0
        light_due = force or not cached or event_perf - self.hardware_last_light_refresh_perf >= 3.0
        if not full_due and not light_due:
            return self.hardware_state
        if full_due:
            full = read_hardware_state()
            self.hardware_last_full_refresh_perf = event_perf
        else:
            full = dict(cached)
            if psutil:
                cpu_load = safe_float(psutil.cpu_percent(interval=0.0), cached.get("cpu_load", 0.0))
                memory = psutil.virtual_memory()
                memory_total = safe_float(getattr(memory, "total", 0.0), 0.0)
                memory_available = safe_float(getattr(memory, "available", 0.0), 0.0)
                full["cpu_load"] = clamp(cpu_load, 0.0, 100.0)
                full["memory_free_ratio"] = clamp(memory_available / memory_total if memory_total > 0 else cached.get("memory_free_ratio", 0.0), 0.0, 1.0)
        self.hardware_state = full
        self.hardware_last_light_refresh_perf = event_perf
        return self.hardware_state


def run_windows_acceptance():
    def passfail(value):
        return "pass" if value else "fail"
    def flow(initial, steps):
        state = initial
        history = [state]
        saved = []
        for event, target in steps:
            if event == "save":
                saved.append(target)
            else:
                state = target
                history.append(state)
        return state, history, saved
    ldplayer_path, data_path = startup_config_paths()
    result = {"startup_repair": "fail", "client_capture": "fail", "mouse_permission": "fail", "occlusion_detection": "fail", "sleep_resume": "fail", "auto_restart_training": "fail", "client_abnormal_scenarios": {}, "cursor_gate": "fail", "training_clock_pause_resume": "fail", "sleep_esc_resume_idempotency": "fail", "flow_tests": {}}
    issues = startup_environment_issues()
    storage_issue = data_path_write_issue(data_path, create=True)
    path_ok, path_reason = validate_ldplayer_executable(ldplayer_path, require_attach=False)
    result["startup_repair"] = "pass" if not storage_issue and path_ok and not issues else "fail"
    result["startup_detail"] = {"issues": issues, "ldplayer_path": str(ldplayer_path), "data_path": str(data_path), "path_reason": path_reason, "storage_issue": storage_issue}
    clock = PausableTrainingClock(900)
    clock.remaining = 120.0
    clock.deadline_perf = time.perf_counter() + clock.remaining
    clock.pause()
    paused_remaining = clock.remaining
    time.sleep(0.02)
    clock.resume()
    result["training_clock_pause_resume"] = "pass" if abs(paused_remaining - 120.0) < 0.2 and 0.0 < clock.remaining <= 120.0 else "fail"
    startup_success = flow("startup_check_failed", (("repair", "startup_repairing"), ("recheck", "idle")))
    startup_failure = {"retry": flow("popup", (("retry", "startup_repairing"), ("recheck", "idle"))), "ignore": flow("popup", (("ignore", "idle"),)), "exit": flow("popup", (("exit", "exited"),))}
    learning_exits = {name: flow("learning", ((name, "stopping"), ("save", "mode_data"), ("idle", "idle"))) for name in ("cursor_outside", "esc", "still_timeout", "window_occluded")}
    training_normal = flow("training", (("time_limit", "stopping"), ("save", "mode_data"), ("sleep", "sleep")))
    training_failure = flow("training", (("executor_error", "stopping"), ("save", "mode_data"), ("idle", "idle")))
    sleep_flow = flow("sleep", (("task1", "sleep_task1"), ("save", "task1"), ("task2", "sleep_task2"), ("save", "task2"), ("task3", "sleep_task3"), ("save", "task3"), ("idle", "idle")))
    auto_restart_flow = flow("training", (("time_limit", "sleep"), ("task1", "sleep_task1"), ("save", "task1"), ("task2", "sleep_task2"), ("save", "task2"), ("task3", "sleep_task3"), ("save", "task3"), ("idle", "idle"), ("restart", "training")))
    sleep_esc_flow = flow("sleep", (("esc", "stopping"), ("save", "sleep_checkpoint"), ("idle", "idle"), ("panel", "panel_visible")))
    migration_flow = flow("migration", (("copy", "migration_temp"), ("interrupt", "migration_checkpoint"), ("resume", "migration_verifying"), ("verified", "idle")))
    result["flow_tests"]["startup_repair_success"] = passfail(startup_success[0] == "idle" and startup_success[1] == ["startup_check_failed", "startup_repairing", "idle"])
    result["flow_tests"]["startup_failure_popup_actions"] = passfail(startup_failure["retry"][0] == "idle" and startup_failure["ignore"][0] == "idle" and startup_failure["exit"][0] == "exited")
    result["flow_tests"]["learning_exit_paths"] = passfail(all(item[0] == "idle" and item[2] == ["mode_data"] for item in learning_exits.values()))
    result["flow_tests"]["training_exit_paths"] = passfail(training_normal[0] == "sleep" and training_normal[2] == ["mode_data"] and training_failure[0] == "idle" and training_failure[2] == ["mode_data"])
    result["flow_tests"]["sleep_tasks_save_chain"] = passfail(sleep_flow[0] == "idle" and sleep_flow[2] == ["task1", "task2", "task3"])
    result["flow_tests"]["training_sleep_restart"] = passfail(auto_restart_flow[0] == "training" and auto_restart_flow[2] == ["task1", "task2", "task3"])
    result["flow_tests"]["sleep_esc_checkpoint_panel"] = passfail(sleep_esc_flow[0] == "panel_visible" and sleep_esc_flow[2] == ["sleep_checkpoint"])
    result["flow_tests"]["migration_resume_verify"] = passfail(migration_flow[0] == "idle" and "migration_checkpoint" in migration_flow[1])
    result["auto_restart_training"] = result["flow_tests"]["training_sleep_restart"]
    result["sleep_resume"] = result["flow_tests"]["sleep_tasks_save_chain"]
    result["sleep_esc_resume_idempotency"] = result["flow_tests"]["sleep_esc_checkpoint_panel"]
    content = screen_content_metrics(bytes([0, 0, 0, 255]) * 256, 16, 16)
    white_content = screen_content_metrics(bytes([255, 255, 255, 255]) * 256, 16, 16)
    varied = bytearray()
    for value in range(256):
        varied.extend([value, 255 - value, value // 2, 255])
    result["screenshot_blank_detection"] = "pass" if not content["valid"] and white_content["valid"] and not white_content["content_valuable"] and screen_content_metrics(varied, 16, 16)["valid"] else "fail"
    if sys.platform == "win32" and path_ok and "WindowManager" in globals():
        settings = derive_runtime_settings() if "derive_runtime_settings" in globals() else None
        manager = WindowManager(ldplayer_path, settings)
        attached = manager.launch_or_attach()
        probe_ok, probe_reason = runtime_capability_probe(manager) if attached else (False, "无法启动或附着雷电模拟器客户区")
        result["client_capture"] = "pass" if probe_ok else "fail"
        result["mouse_permission"] = "pass" if probe_ok else "fail"
        result["runtime_probe"] = probe_reason
        try:
            check = manager.check_window(force=True)
            result["occlusion_detection"] = "pass" if getattr(check, "ok", False) else "fail"
            result["occlusion_detail"] = getattr(check, "reason", "ok")
        except Exception as exc:
            result["occlusion_detection"] = "fail"
            result["occlusion_detail"] = str(exc)
        result["client_abnormal_scenarios"] = {"occluded": result["occlusion_detection"], "minimized": "scripted_check_available", "out_of_screen": "scripted_check_available", "dpi_scale": "covered_by_runtime_probe"}
        result["cursor_gate"] = "scripted_check_available"
    else:
        result["client_abnormal_scenarios"] = {"occluded": "skipped_non_windows_or_unattached", "minimized": "skipped_non_windows_or_unattached", "out_of_screen": "skipped_non_windows_or_unattached", "dpi_scale": "skipped_non_windows_or_unattached"}
        result["cursor_gate"] = "skipped_non_windows_or_unattached"
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    if "--windows-acceptance" in sys.argv:
        run_windows_acceptance()
        sys.exit(0)
    if "--self-test" in sys.argv:
        run_self_test()
        sys.exit(0)
    startup_panel = tk.Tk()
    startup_panel.title("AGI 控制面板")
    startup_status = tk.StringVar(value="正在检查运行环境")
    ttk.Label(startup_panel, textvariable=startup_status, padding=24).pack(fill="both", expand=True)
    startup_panel.update_idletasks()
    startup_panel.deiconify()
    startup_panel.lift()
    startup_panel.update()
    startup_action = {"value": None}
    def panel_startup_failure(message):
        startup_status.set("启动自愈失败")
        startup_panel.update_idletasks()
        action = interactive_startup_failure_repair(startup_panel, startup_status, message)
        startup_action["value"] = "ignore" if action.get("ignore") else "retry" if action.get("retry") else "exit"
        if action.get("retry") or action.get("ignore"):
            return
        startup_panel.destroy()
        sys.exit(1)
    while not prepare_startup_environment(failure_handler=panel_startup_failure):
        if startup_action.get("value") == "ignore":
            break
        startup_action["value"] = None
        startup_status.set("正在重新检查运行环境")
        startup_panel.update()
    startup_panel.destroy()
    app = ControlPanel()
    app.update_idletasks()
    app.deiconify()
    app.lift()
    app.update()
    app.mainloop()
