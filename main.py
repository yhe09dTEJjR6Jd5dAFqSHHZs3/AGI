import concurrent.futures
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
from dataclasses import dataclass, fields, replace
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
    default_sleep_seconds: int
    default_still_seconds: float
    default_experience_pool_gb: float
    default_ai_model_limit: int
    editable_fields: tuple


AGENT_SPEC = AgentSpec(
    default_ldplayer_path=r"D:\LDPlayer9\dnplayer.exe",
    default_data_path=r"C:\Users\Administrator\Desktop\AAA",
    default_training_seconds=900,
    default_sleep_seconds=1800,
    default_still_seconds=10.0,
    default_experience_pool_gb=10.0,
    default_ai_model_limit=10,
    editable_fields=("ldplayer_path", "data_path", "training_seconds", "sleep_seconds", "still_seconds", "experience_pool_gb", "ai_model_limit")
)


def should_stop_run(stop_event, deadline, escape_check):
    if stop_event and stop_event.is_set():
        return "esc"
    if escape_check and escape_check():
        return "esc"
    if deadline is not None and time.perf_counter() >= deadline:
        return "time_limit"
    return None


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
    base_command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--user"]
    mirrors = [None, "https://pypi.tuna.tsinghua.edu.cn/simple", "https://mirrors.aliyun.com/pypi/simple/"]
    commands = [base_command + (["-i", mirror, "--trusted-host", urlparse(mirror).hostname] if mirror else []) + missing for mirror in mirrors]
    install_timeout = max(1, (os.cpu_count() or 1) * 30)
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
    return None


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
    storage_issue = data_path_write_issue(data_path)
    if storage_issue:
        issues.append(f"存储路径无效 {data_path}：{storage_issue}")
    return issues


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
        actions.append("启动检查已跳过雷电模拟器窗口状态")
    elif not valid_path:
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
    sections = ["初次检查：", *[f"- {item}" for item in initial_issues], "", "修复过程："]
    sections.extend(f"- {item}" for item in repair_actions)
    if repair_error:
        sections.append(f"- 修复失败：{repair_error}")
    sections.extend(["", "修复后复检：", *[f"- {item}" for item in remaining_issues]])
    return "\n".join(sections)


def prepare_startup_environment(check_environment=None, repair_environment=None, failure_handler=None):
    check_environment = check_environment or startup_environment_issues
    repair_environment = repair_environment or attempt_startup_environment_repair
    failure_handler = failure_handler or fail_and_exit
    initial_issues = check_environment()
    if not initial_issues:
        return True
    repair_actions = []
    repair_error = None
    try:
        repair_environment(repair_actions)
    except Exception as exc:
        repair_error = str(exc)
    remaining_issues = check_environment()
    if not remaining_issues:
        return True
    if not repair_actions:
        repair_actions.append("自动修复未能完成")
    failure_handler(startup_failure_detail(initial_issues, repair_actions, repair_error, remaining_issues))
    return False


def configuration_failure(area, error):
    detail = f"{area}：{error}"
    if "--self-test" in sys.argv:
        raise RuntimeError(detail) from error
    fail_and_exit("配置文件生成失败或读取失败。\n" + detail)


def default_runtime_settings_payload():
    return {"training_seconds": AGENT_SPEC.default_training_seconds, "sleep_seconds": AGENT_SPEC.default_sleep_seconds, "still_seconds": AGENT_SPEC.default_still_seconds, "experience_pool_gb": AGENT_SPEC.default_experience_pool_gb, "ai_model_limit": AGENT_SPEC.default_ai_model_limit}


DEFAULT_LDPLAYER_PATH = AGENT_SPEC.default_ldplayer_path
DEFAULT_DATA_PATH = AGENT_SPEC.default_data_path
DEFAULT_TRAINING_SECONDS = AGENT_SPEC.default_training_seconds
DEFAULT_SLEEP_SECONDS = AGENT_SPEC.default_sleep_seconds
DEFAULT_STILL_SECONDS = AGENT_SPEC.default_still_seconds
DEFAULT_EXPERIENCE_POOL_GB = AGENT_SPEC.default_experience_pool_gb
DEFAULT_AI_MODEL_LIMIT = AGENT_SPEC.default_ai_model_limit
MODE_NAMES = {"idle": "空闲", "starting": "准备中", "learning": "学习模式", "training": "训练模式", "sleep": "睡眠模式", "migration": "数据迁移"}
CONFIG_SCHEMA_VERSION = 1
USER_EDITABLE_STARTUP_FIELDS = ("ldplayer_path", "data_path")
USER_EDITABLE_RUNTIME_FIELDS = ("training_seconds", "sleep_seconds", "still_seconds", "experience_pool_gb", "ai_model_limit")
USER_EDITABLE_FIELDS = AGENT_SPEC.editable_fields


class AllowedUserEditPolicy:
    ALLOWED_FIELDS = frozenset(AGENT_SPEC.editable_fields)
    STARTUP_FIELDS = frozenset(("ldplayer_path", "data_path"))
    RUNTIME_FIELDS = frozenset(("training_seconds", "sleep_seconds", "still_seconds", "experience_pool_gb", "ai_model_limit"))

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
    ("starting", "idle"): {"window_invalid", "user_stop", "runtime_error", "minimize_failed"},
    ("learning", "idle"): {"esc", "still_timeout", "window_invalid", "user_stop", "runtime_error"},
    ("training", "idle"): {"esc", "still_timeout", "window_invalid", "user_stop", "runtime_error", "executor_error"},
    ("training", "sleep"): {"time_limit"},
    ("sleep", "idle"): {"completed", "esc", "time_limit", "poor_optimization", "user_stop", "runtime_error"},
    ("migration", "idle"): {"completed", "migration_error", "user_stop"}
}


TERMINATION_REASONS = ("window_invalid", "rect_changed", "empty_action", "executor_error", "time_limit", "esc", "still_timeout", "user_stop", "migration_error", "completed", "poor_optimization")
HUMAN_FEATURE_NAMES = ("duration", "direct", "bend", "points", "speed_mean", "speed_variance", "acceleration_change", "pauses", "hover_before", "drag_curvature", "double_click_interval")

RUNTIME_NUMBER_RULES = {
    "hash_size": ("screen_pixels", "pool_count", "capture_ms"),
    "nearest_top_k": ("pool_count", "cpu_count", "screen_score_total"),
    "nearest_candidate_limit": ("pool_count", "window_instability", "cpu_count", "gpu_count"),
    "hash_prefix_bits": ("pool_count", "screen_pixels"),
    "mouse_activity_wait": ("capture_ms", "cpu_load"),
    "training_event_wait": ("capture_ms", "execution_ms", "cpu_load", "gpu_factor", "window_instability"),
    "sleep_event_wait": ("cpu_load", "cpu_count", "memory_free_ratio"),
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
class Config:
    ldplayer_path: Path
    data_path: Path
    training_seconds: int
    sleep_seconds: int
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


def windows_runtime_report(ldplayer_path=None):
    report = {"platform": sys.platform, "desktop_session": bool(os.environ.get("SESSIONNAME")), "dpi_awareness_checked": False, "admin_or_ui_access_checked": False, "ldplayer_path_exists": None}
    if sys.platform == "win32":
        try:
            report["dpi_awareness_checked"] = ctypes.windll.user32.GetDpiForSystem() > 0
        except Exception:
            report["dpi_awareness_checked"] = False
        try:
            report["admin_or_ui_access_checked"] = ctypes.windll.user32.OpenInputDesktop(0, False, 0x0100) != 0
        except Exception:
            report["admin_or_ui_access_checked"] = False
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
            return False, "无法通过该 dnplayer.exe 启动或附着雷电模拟器窗口"
        check = manager.check_window(force=True)
        if not check.ok:
            return False, f"已找到雷电模拟器但窗口异常：{check.reason}"
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
    assert len(startup_events) == 1 and "初次检查" in startup_events[0] and "修复后复检" in startup_events[0]
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
    assert set(USER_EDITABLE_FIELDS) == {"ldplayer_path", "data_path", "training_seconds", "sleep_seconds", "still_seconds", "experience_pool_gb", "ai_model_limit"}
    assert {"esc", "still_timeout", "window_invalid"}.issubset(ALLOWED_TRANSITIONS[("learning", "idle")])
    assert {"esc", "still_timeout", "window_invalid"}.issubset(ALLOWED_TRANSITIONS[("training", "idle")])
    assert "time_limit" not in ALLOWED_TRANSITIONS[("training", "idle")]
    assert "time_limit" in ALLOWED_TRANSITIONS[("training", "sleep")]
    assert ("sleep", "training") not in ALLOWED_TRANSITIONS
    assert ("sleep", "starting") not in ALLOWED_TRANSITIONS
    assert {"completed", "time_limit", "poor_optimization"}.issubset(ALLOWED_TRANSITIONS[("sleep", "idle")])
    assert {"completed", "migration_error", "user_stop"}.issubset(ALLOWED_TRANSITIONS[("migration", "idle")])
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
    sleep_result = pool.sleep_training_step(changed.sleep_batch_size)
    assert sleep_result["trained"] >= 1
    assert pool.records[0].get("sleep_visits", 0) >= 1
    brain = ActionBrain(pool, changed)
    _, decision = brain.choose(a, novelty, batch, 0.0)
    assert isinstance(decision, dict)
    random_action, random_decision = brain.choose(a, novelty, [], 0.0, zero_score_factor=1.0)
    assert random_action and random_decision["reason"] == "zero_score_random_exploration" and random_decision["zero_score_factor"] == 1.0
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
        compacted = store.compact_experience_pool(0.1)
        retained = store.load_experience()
        assert compacted["changed"] and shared.exists() and not unique.exists()
        assert [record["id"] for record in retained] == ["high"]
        dummy_panel = type("DummyPanel", (), {"store": store})()
        assert ControlPanel.sleep_compaction_progress(dummy_panel, {"size_bytes": 200, "target_bytes": 100}) == 0.5
        assert ControlPanel.sleep_compaction_progress(dummy_panel, {"size_bytes": 100, "target_bytes": 100}) == 1.0
        assert ControlPanel.sleep_progress_fields(dummy_panel, time.perf_counter(), 10, 0, 10, 1.0)["compaction"] == 1.0
        assert ControlPanel.sleep_compaction_complete(dummy_panel, {"size_bytes": 100, "target_bytes": 100})
        assert ControlPanel.sleep_completion_reached(dummy_panel, 3, 3, deque([0.02, 0.02]), 0.01, 0.9, 0.8, True)
        class DummyVar:
            def __init__(self):
                self.value = None
            def set(self, value):
                self.value = value
        dummy_panel.progress_value = 0.4
        dummy_panel.last_progress_update_perf = 0.0
        dummy_panel.progress_var = DummyVar()
        dummy_panel.progress_text_var = DummyVar()
        dummy_panel.ui = lambda fn: fn()
        dummy_panel.update_mode_button_states = lambda: None
        dummy_panel.settings = replace(settings, ui_progress_delta=1.0)
        ControlPanel.update_progress(dummy_panel, ControlPanel.idle_progress_value(dummy_panel, "learning", 87.0), force=True)
        assert dummy_panel.progress_value == 0.0 and dummy_panel.progress_var.value == 0.0
        assert ControlPanel.idle_progress_value(dummy_panel, "training", 91.0) == 0.0
        assert ControlPanel.idle_progress_value(dummy_panel, "migration", 91.0) == 0.0
        store.save_settings({"training_seconds": 1, "sleep_seconds": 2, "still_seconds": 3, "experience_pool_gb": 4, "ai_model_limit": 5, "forbidden": 6})
        saved_settings = json.loads(store.settings_file.read_text(encoding="utf-8"))
        assert "forbidden" not in saved_settings and saved_settings["experience_pool_gb"] == 4.0 and saved_settings["ai_model_limit"] == 5
        store.experience_file.write_text("{bad json}\n" + json.dumps({"id": "ok"}) + "\n", encoding="utf-8")
        loaded = store.load_experience()
        assert len(loaded) == 1 and loaded[0]["id"] == "ok"
        assert (store.root / "experience.bad.jsonl").exists()
        screen_path = store.screen_dir / "sample.png"
        screen_path.write_bytes(b"screen")
        store.save_experience_records([{"id": "m1", "mouse_action": {"type": "click"}, "reward": 1.0, "sleep_confidence": 0.5, "screen_path": store.relative_path(screen_path)}])
        model_path = store.save_ai_model_snapshot(store.load_experience(), settings, 1, "completed")
        assert model_path.exists() and len(list(store.model_dir.glob("model_*.json"))) == 1
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
    try:
        output = subprocess.check_output(["wmic", "path", "win32_VideoController", "get", "AdapterRAM"], stderr=subprocess.DEVNULL, text=True, timeout=2.0)
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

    def value(self, name, context, semantic_goal):
        stable = sum((index + 1) * ord(char) for index, char in enumerate(f"{name}|{semantic_goal}"))
        context_values = [abs(safe_float(value, 0.0)) for value in context.values() if isinstance(value, (int, float))]
        condition = sum(context_values) / max(1, len(context_values))
        spread = math.log1p(condition + stable % max(2, len(str(name)) + len(str(semantic_goal))))
        normalized = (math.sin(spread + stable) + 1.0) / 2.0
        value = clamp(normalized, 0.0, 1.0)
        self.audit[name] = {"source": "RuntimeGeneratedNumbers.value", "reality_conditions": dict(context), "formula": "clamp((sin(log1p(mean(abs(context))+stable_mod)+stable)+1)/2,0,1)", "semantic_goal": semantic_goal, "current_value": value}
        return value


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


def reward_breakdown(novelty, human_score, settings):
    score_precision = 2
    screen_resolution = 10 ** (-score_precision)
    screen_novelty = round(clamp(novelty, 0.0, 100.0), score_precision)
    screen_reward = screen_novelty
    human_similarity = round(clamp(human_score, 0.0, 100.0), score_precision)
    human_delta = round(human_similarity - clamp(settings.score_default, 0.0, 100.0), score_precision)
    human_tiebreak = human_similarity
    human_bonus = round((human_similarity / 100.0) * screen_resolution * (1.0 - 1e-6), 8)
    screen_score_delta = round(clamp(screen_reward, settings.reward_total_min, settings.reward_total_max), 6)
    total_reward = round(clamp(screen_score_delta + human_bonus, settings.reward_total_min, settings.reward_total_max + screen_resolution), 8)
    return {"reward_version": 4, "screen_novelty": screen_novelty, "screen_reward": screen_reward, "human_similarity": human_similarity, "human_tiebreak": round(human_tiebreak, 6), "human_bonus": human_bonus, "screen_score_resolution": screen_resolution, "screen_score_delta": screen_score_delta, "basis": ["nearest_screen_batch", "learning_mouse_profile", "lexicographic_screen_then_human", "numeric_human_bonus_below_screen_resolution"], "screen_primary_reward": screen_reward, "human_tie_break_reward": round(human_tiebreak, 6), "mouse_action_delta": human_delta, "total_reward": total_reward, "reward_sort_key": [round(screen_reward, score_precision), round(human_similarity, score_precision)]}


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
        reward = clamp(safe_float(record.get("reward", record.get("total_reward", 0.0)), 0.0), self.settings.reward_total_min, self.settings.reward_total_max)
        base = (reward - self.settings.reward_total_min) / max(1.0, self.settings.reward_total_max - self.settings.reward_total_min)
        human = clamp(safe_float(record.get("human_score", 50.0), 50.0), 0.0, 100.0) / 100.0
        novelty = clamp(safe_float(record.get("novelty", 0.0), 0.0), 0.0, 100.0) / 100.0
        return clamp(base * 0.55 + human * 0.25 + novelty * 0.2, 0.0, 1.0)

    def train(self, records):
        usable = [record for record in records or [] if isinstance(record, dict) and record.get("mouse_action")]
        if not usable:
            return {"trained": 0, "loss": self.loss, "confidence": 0.0}
        lr = clamp(1.0 / max(4.0, math.sqrt(self.trained_steps + len(usable) + 1.0)), 0.01, 0.2)
        total_loss = 0.0
        with self.lock:
            for record in usable:
                features = self.features(record)
                pred = self.predict_features(features)
                target = self.target(record)
                error = pred - target
                total_loss += error * error
                for name in self.FEATURE_NAMES:
                    self.weights[name] = clamp(self.weights.get(name, 0.0) - lr * error * features.get(name, 0.0), -8.0, 8.0)
                record["model_prediction"] = round(pred, 4)
                record["model_target"] = round(target, 4)
            self.trained_steps += len(usable)
            self.loss = round(total_loss / len(usable), 6)
            confidence = clamp(1.0 - math.sqrt(self.loss), 0.0, 1.0)
        return {"trained": len(usable), "loss": self.loss, "confidence": confidence}

    def snapshot(self):
        with self.lock:
            return {"type": "online_logistic_policy", "feature_names": list(self.FEATURE_NAMES), "weights": {name: round(value, 8) for name, value in self.weights.items()}, "trained_steps": self.trained_steps, "loss": self.loss}


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
            with self.lock:
                self.root.mkdir(parents=True, exist_ok=True)
                temporary = self.settings_file.with_suffix(".tmp")
                with temporary.open("w", encoding="utf-8") as file:
                    json.dump(payload, file, ensure_ascii=False, indent=2)
                temporary.replace(self.settings_file)
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

    def save_state(self):
        with self.lock:
            temporary = self.state_file.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(self.state, file, ensure_ascii=False, indent=2)
            temporary.replace(self.state_file)

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

    def save_experience_records(self, records):
        with self.lock:
            temporary = self.experience_file.with_suffix(".save.tmp")
            with temporary.open("w", encoding="utf-8") as file:
                for record in records or []:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary.replace(self.experience_file)

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
                temporary.replace(self.experience_file)
            else:
                with temporary.open("w", encoding="utf-8") as target:
                    for record in updates.values():
                        target.write(json.dumps(record, ensure_ascii=False) + "\n")
                temporary.replace(self.experience_file)
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

    def save_ai_model_snapshot(self, records, settings, max_models, status, model=None, run_guard=None):
        if run_guard and run_guard():
            return None
        with self.lock:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            ranked = sorted([record for record in records or [] if record.get("mouse_action")], key=lambda record: (safe_float(record.get("sleep_policy_reward", record.get("reward", 0.0)), 0.0), safe_float(record.get("sleep_confidence", 0.0), 0.0)), reverse=True)
            limit = max(1, min(len(ranked) or 1, safe_int(getattr(settings, "global_action_heap_limit", 1), 1)))
            model_payload = model.snapshot() if model else None
            identity = hashlib.sha256(str(self.root.resolve()).encode("utf-8", "replace")).hexdigest()
            training_digest = hashlib.sha256(json.dumps([record.get("id") for record in ranked[:limit]], ensure_ascii=False).encode("utf-8")).hexdigest()
            payload = {"schema_version": CONFIG_SCHEMA_VERSION, "model_version": 1, "training_data_version": 1, "data_path_id": identity, "checksum": training_digest, "created_at": now_text(), "status": status, "screen_score_total": self.screen_score_total, "experience_count": len(records or []), "model": model_payload, "policy": [{"id": record.get("id"), "mode": record.get("mode"), "mouse_action": record.get("mouse_action"), "reward": safe_float(record.get("reward", 0.0), 0.0), "sleep_policy_reward": safe_float(record.get("sleep_policy_reward", record.get("reward", 0.0)), 0.0), "sleep_confidence": safe_float(record.get("sleep_confidence", 0.0), 0.0), "sleep_model_confidence": safe_float(record.get("sleep_model_confidence", record.get("model_prediction", 0.0)), 0.0), "model_prediction": safe_float(record.get("model_prediction", 0.0), 0.0), "model_target": safe_float(record.get("model_target", 0.0), 0.0), "sleep_novelty": safe_float(record.get("sleep_novelty", record.get("novelty", 0.0)), 0.0), "human_score": safe_float(record.get("sleep_human_score", record.get("human_score", 0.0)), 0.0)} for record in ranked[:limit]]}
            path = self.model_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex}.json"
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            if run_guard and run_guard():
                try:
                    temporary.unlink(missing_ok=True)
                except Exception:
                    pass
                return None
            temporary.replace(path)
            self.compact_ai_models(max_models)
            return path

    def compact_ai_models(self, max_models):
        limit = max(1, safe_int(max_models, AGENT_SPEC.default_ai_model_limit))
        models = sorted(self.model_dir.glob("model_*.json"), key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
        keep = min(len(models), limit)
        removed = 0
        for path in models[keep:]:
            try:
                path.unlink()
                removed += 1
            except Exception:
                pass
        return {"changed": removed > 0, "removed": removed, "limit": limit, "target_count": keep}


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
        return {"weights": clean, "trained_steps": trained_steps, "loss": loss}

    def load_latest_model_state(self, settings=None):
        candidates = sorted(self.model_dir.glob("model_*.json"), key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
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
        payload = {"schema_version": CONFIG_SCHEMA_VERSION, "training_seconds": max(1, safe_int(source.get("training_seconds", AGENT_SPEC.default_training_seconds), AGENT_SPEC.default_training_seconds)), "sleep_seconds": max(1, safe_int(source.get("sleep_seconds", AGENT_SPEC.default_sleep_seconds), AGENT_SPEC.default_sleep_seconds)), "still_seconds": max(0.1, safe_float(source.get("still_seconds", AGENT_SPEC.default_still_seconds), AGENT_SPEC.default_still_seconds)), "experience_pool_gb": max(0.1, safe_float(source.get("experience_pool_gb", AGENT_SPEC.default_experience_pool_gb), AGENT_SPEC.default_experience_pool_gb)), "ai_model_limit": max(1, safe_int(source.get("ai_model_limit", AGENT_SPEC.default_ai_model_limit), AGENT_SPEC.default_ai_model_limit)), "runtime_generated_numbers": RUNTIME_NUMBER_AUDIT}
        try:
            with self.lock:
                temporary = self.settings_file.with_suffix(".tmp")
                with temporary.open("w", encoding="utf-8") as file:
                    json.dump(payload, file, ensure_ascii=False, indent=2)
                temporary.replace(self.settings_file)
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

    def compact_experience_pool(self, limit_gb, run_guard=None):
        limit_bytes = max(1, int(max(0.1, safe_float(limit_gb, DEFAULT_EXPERIENCE_POOL_GB)) * 1024 * 1024 * 1024))
        records = self.load_experience()
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
        if current <= limit_bytes:
            return {"changed": False, "size_bytes": current, "removed": 0, "target_bytes": limit_bytes}
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
        path_references = defaultdict(int)
        record_paths = {}
        for item in records:
            paths = self.experience_record_paths([item["record"]])
            record_paths[item["line"]] = paths
            for path in paths:
                path_references[path] += 1
        removed_ids = set()
        removed = 0
        for item in records:
            if run_guard and run_guard():
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
                            path.unlink()
                            current = max(0, current - size)
                            path_sizes.pop(path, None)
                    except Exception:
                        pass
        if removed_ids:
            temporary = self.experience_file.with_suffix(".compact.tmp")
            with temporary.open("w", encoding="utf-8") as file:
                for item in records:
                    if item["line"] not in removed_ids:
                        file.write(item["text"] + "\n")
            temporary.replace(self.experience_file)
            current = self.experience_pool_size_bytes([entry["record"] for entry in records if entry["line"] not in removed_ids])
        return {"changed": bool(removed_ids), "size_bytes": current, "removed": removed, "target_bytes": target_bytes}

    def log_error(self, where, error, context=None):
        payload = {
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


@dataclass(frozen=True)
class WindowCheck:
    ok: bool
    reason: str
    rect: tuple = ()
    hits: int = 0
    expected: int = 0


class WindowManager:
    def __init__(self, executable_path, settings):
        self.executable_path = Path(executable_path)
        self.settings = settings
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
        candidates = []
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
                if (pids and pid in pids) or matched_title:
                    candidates.append((cwidth * cheight, width * height, hwnd))
            except Exception:
                pass
        try:
            win32gui.EnumWindows(handler, None)
        except Exception:
            return False
        if not candidates:
            return False
        candidates.sort(reverse=True)
        with self.lock:
            self.hwnd = candidates[0][2]
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
                if hit == hwnd or win32gui.IsChild(hwnd, hit):
                    hits += 1
            cache["occlusion_perf"] = now_perf
            result = WindowCheck(hits == len(points), "ok" if hits == len(points) else "occluded", rect, hits, len(points))
            cache.update({"ok": result.ok, "reason": result.reason, "check": result})
            return result
        except Exception as exc:
            return WindowCheck(False, type(exc).__name__)

    def window_ok(self, force=False):
        return self.check_window(force=force).ok

class ScreenAnalyzer:
    def __init__(self, hash_size):
        self.hash_size = int(hash_size)
        self.sct = None
        self.resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

    def __enter__(self):
        self.sct = mss.mss()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def capture(self, rect):
        left, top, right, bottom = rect
        width, height = rect_size(rect)
        shot = self.sct.grab({"left": int(left), "top": int(top), "width": width, "height": height})
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
            save_kwargs = {"optimize": priority == "critical", "compress_level": int(clamp(compression, 0, 9))}
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
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
        self.lock = threading.RLock()
        self.action_cache = []
        self.prefix_neighbor_cache = OrderedDict()
        self.global_action_heap = []
        self.nearest_cache = OrderedDict()
        self.metric_tree = BKHashTree()
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
            if record.get("mouse_action") and record.get("mode") == "learning" and record.get("mouse_source") == "user":
                self.profile.observe(record["mouse_action"])
            if record.get("mouse_action"):
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

    def nearest(self, hash_value, exclude_id=None, limit=None, before_index=None):
        if not hash_value:
            return []
        with self.lock:
            top_k = max(1, safe_int(limit, self.settings.nearest_top_k))
            cache_key = None if exclude_id is not None or limit is not None or before_index is not None else (self.index_version, hash_value.bits, hash_value.hex, self.index_settings.hash_prefix_bits, self.index_settings.nearest_candidate_limit, self.settings.nearest_top_k)
            cached = self.nearest_cache.get(cache_key) if cache_key is not None else None
            if cached is not None:
                self.nearest_cache.move_to_end(cache_key)
                return [copy.deepcopy(item) for item in cached]
            candidate_indexes = self.candidate_indices(hash_value)
            if len(self.records) > self.index_settings.nearest_candidate_limit:
                recent_limit = min(len(self.records), max(top_k, self.index_settings.nearest_candidate_limit // 4))
                candidate_indexes = list(dict.fromkeys(candidate_indexes + list(range(len(self.records) - recent_limit, len(self.records))) + [index for _, index in heapq.nlargest(min(len(self.global_action_heap), top_k), self.global_action_heap)]))
            snapshot = [(index, self.hashes[index], self.records[index]) for index in candidate_indexes if self.hashes[index] and (before_index is None or index < before_index) and (exclude_id is None or self.records[index].get("id") != exclude_id)]
        scored = []
        for index, other, record in snapshot:
            if other:
                similarity = hash_similarity(hash_value, other)
                item = {"similarity": similarity, "record": record}
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

    def novelty(self, hash_value):
        batch = self.nearest(hash_value)
        if not batch:
            return 100.0, []
        top = batch[:max(1, min(self.settings.nearest_top_k, len(batch)))]
        sims = [clamp(item.get("similarity", 0.0), 0.0, 1.0) for item in top]
        if any(sim >= 0.9999 for sim in sims):
            return 0.0, batch
        weight_total = 0.0
        score_total = 0.0
        for index, sim in enumerate(sims):
            weight = 1.0 / (1.0 + index)
            score_total += sim * weight
            weight_total += weight
        avg_sim = score_total / weight_total if weight_total > 0.0 else sims[0]
        peak_sim = sims[0]
        density = sum(1 for sim in sims if sim >= 0.95) / len(sims)
        fused = clamp(peak_sim * 0.45 + avg_sim * 0.4 + density * 0.15, 0.0, 1.0)
        return round(clamp((1.0 - fused) * 100.0, 0.0, 100.0), 2), batch

    def best_global_action(self):
        with self.lock:
            ranked = heapq.nlargest(max(1, min(self.settings.nearest_top_k, len(self.global_action_heap))), self.global_action_heap, key=lambda item: item[0])
            weighted = []
            for _, index in ranked:
                if index < 0 or index >= len(self.records):
                    continue
                record = self.records[index]
                action = record.get("mouse_action")
                reward = safe_float(record.get("reward", 0.0), 0.0)
                human_score = clamp(record.get("human_score", 50.0), 0.0, 100.0)
                if action:
                    weighted.append((max(0.05, 1.0 + reward / 100.0) * max(0.25, human_score / 100.0), action))
        chosen = weighted_choice(weighted)
        return copy.deepcopy(chosen) if chosen else None


    def sleep_training_batch_indices(self, batch_size):
        with self.lock:
            action_indices = [index for index, record in enumerate(self.records) if record.get("mouse_action")]
            if not action_indices:
                return []
            target = max(1, min(safe_int(batch_size, 1), len(action_indices)))
            ranked = [index for _, index in heapq.nlargest(min(len(self.global_action_heap), target), self.global_action_heap, key=lambda item: item[0])]
            recent = action_indices[-target:]
            remaining = target * max(1, self.settings.ui_metric_columns) - len(ranked) - len(recent)
            sampled = random.sample(action_indices, min(len(action_indices), max(0, remaining))) if remaining > 0 else []
            return list(dict.fromkeys(ranked + recent + sampled))[:target]

    def compute_screen_score(self, hash_value, exclude_id=None, before_index=None, exact_checksum=None):
        neighbors = self.nearest(hash_value, exclude_id=exclude_id, limit=self.settings.nearest_top_k, before_index=before_index)
        sims = [clamp(item.get("similarity", 0.0), 0.0, 1.0) for item in neighbors]
        if sims:
            top = sims[:max(1, min(self.settings.nearest_top_k, len(sims)))]
            density = sum(1 for item in top if item >= 0.95) / len(top)
            score = clamp((1.0 - (top[0] * 0.5 + sum(top) / len(top) * 0.35 + density * 0.15)) * 100.0, 0.0, 100.0)
            if score <= 0.0 and exact_checksum:
                exact_match = any(exact_checksum and exact_checksum == item.get("record", {}).get("image_checksum") for item in neighbors)
                if not exact_match:
                    score = 0.01
            confidence = clamp(top[0] * 0.65 + (1.0 - density) * 0.35, 0.0, 1.0)
        else:
            score = 100.0
            confidence = 0.0
        return round(score, 2), neighbors, confidence

    def recheck_screen_scores(self, store=None, analyzer=None, tolerance=0.01, run_guard=None):
        checked = 0
        rescored = 0
        missing = 0
        errors = 0
        image_missing = 0
        image_corrupt = 0
        hash_missing = 0
        unrecoverable = 0
        with self.lock:
            snapshot = [(index, copy.deepcopy(record)) for index, record in enumerate(self.records)]
        updates = []
        for index, record in snapshot:
            if run_guard and run_guard():
                break
            hash_value = parse_hash_value(record)
            file_hash = None
            screen_path = record.get("screen_path")
            if store and analyzer and screen_path:
                path = (store.root / str(screen_path)).resolve()
                try:
                    if path.is_file() and store.root.resolve() in (path, *path.parents):
                        with Image.open(path) as image:
                            rgb_image = image.convert("RGB")
                            file_hash = analyzer.fingerprint(rgb_image)
                            record["image_checksum"] = image_content_checksum(rgb_image)
                    else:
                        record["score_status"] = "image_missing"
                        image_missing += 1
                except Exception as exc:
                    record["screen_file_error"] = str(exc)
                    record["score_status"] = "image_corrupt"
                    image_corrupt += 1
            if file_hash and (not hash_value or file_hash.hex != hash_value.hex or file_hash.bits != hash_value.bits):
                hash_value = file_hash
                record["screen_hash"] = file_hash.hex
                record["screen_hash_hex"] = file_hash.hex
                record["screen_hash_int"] = file_hash.value
                record["screen_hash_bits"] = file_hash.bits
                errors += 1
            if not hash_value:
                hash_missing += 1
                unrecoverable += 1
                record["score_status"] = "unrecoverable"
                record["score_checked_at"] = now_text()
                updates.append((index, record, None))
                continue
            score, neighbors, confidence = self.compute_screen_score(hash_value, exclude_id=record.get("id"), before_index=index, exact_checksum=record.get("image_checksum"))
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
            reward_info = reward_breakdown(score, human_score, self.settings)
            record.update({"screen_score": score, "novelty": score, "score_version": 1, "score_status": "rescored" if bool(lacks or wrong) else "scored", "score_basis": "nearest_screen_content_recheck", "score_checked_at": now_text(), "score_neighbors": [{"id": item["record"].get("id"), "similarity": round(item.get("similarity", 0.0), 4)} for item in neighbors], "score_confidence": round(confidence, 4), "score_rechecked": True, "score_recomputed": bool(lacks or wrong), "reward_version": reward_info["reward_version"], "screen_primary_reward": reward_info["screen_primary_reward"], "human_tie_break_reward": reward_info["human_tie_break_reward"], "reward_breakdown": reward_info, "reward_sort_key": reward_info["reward_sort_key"], "total_reward": reward_info["total_reward"], "reward": reward_info["total_reward"], "screen_score_delta": max(0.0, reward_info["screen_score_delta"])})
            checked += 1
            updates.append((index, record, hash_value))
        with self.lock:
            for index, record, hash_value in updates:
                if index < len(self.records):
                    self.records[index] = record
                    self.hashes[index] = hash_value
            self.rebuild_index_locked()
            self.rebuild_action_heap_locked()
            self.nearest_cache.clear()
        return {"checked": checked, "rescored": rescored, "missing": missing, "errors": errors, "image_missing": image_missing, "image_corrupt": image_corrupt, "hash_missing": hash_missing, "unrecoverable": unrecoverable}

    def sleep_training_step(self, batch_size, settle_screen_score=None, run_guard=None):
        indices = self.sleep_training_batch_indices(batch_size)
        if not indices:
            return {"trained": 0, "best_score": 0.0, "avg_score": 0.0, "avg_confidence": 0.0}
        with self.lock:
            records_len = len(self.records)
            snapshot = [(index, self.hashes[index], copy.deepcopy(self.records[index])) for index in indices if index < records_len and self.hashes[index]]
        updates = []
        for index, hash_value, record in snapshot:
            if run_guard and run_guard():
                break
            novelty, neighbors, similarity_confidence = self.compute_screen_score(hash_value, exclude_id=record.get("id"), before_index=index)
            action = record.get("mouse_action")
            human_score = clamp(record.get("human_score", self.profile.score(action)), 0.0, 100.0) if action else self.settings.score_default
            reward = safe_float(record.get("reward", 0.0), 0.0)
            transition = safe_float(record.get("transition_reward", 0.0), 0.0)
            source = 1.0 if record.get("mode") == "learning" and record.get("mouse_source") == "user" else 0.0
            visits = max(0, safe_int(record.get("sleep_visits", 0), 0))
            learned = safe_float(record.get("sleep_policy_reward", reward), reward)
            candidate = reward + novelty * self.settings.action_score_novelty_weight + (human_score - self.settings.score_default) * self.settings.action_score_human_weight + transition * self.settings.action_score_reward_weight + source * self.settings.score_default
            value = (learned * visits + candidate) / (visits + 1)
            confidence = clamp((similarity_confidence + human_score / 100.0 + min(1.0, (visits + 1) / max(1.0, self.settings.nearest_top_k))) / 3.0, 0.0, 1.0)
            reward_info = reward_breakdown(novelty, human_score, self.settings)
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
                record["screen_score_delta"] = max(0.0, reward_info["screen_score_delta"])
                record["screen_score_settled"] = max(0.0, safe_float(record.get("screen_score_settled", 0.0), 0.0)) + settled_delta
                train_records.append(record)
            model_result = self.model.train(train_records)
            for record in train_records:
                model_prediction = clamp(safe_float(record.get("model_prediction", self.model.predict(record)), 0.0), 0.0, 1.0)
                value = safe_float(record.get("sleep_policy_reward", record.get("reward", 0.0)), 0.0)
                confidence = clamp(safe_float(record.get("sleep_confidence", 0.0), 0.0), 0.0, 1.0)
                record["sleep_model_confidence"] = round(model_prediction, 4)
                score = value * (0.65 + confidence * 0.2 + model_prediction * 0.15)
                total_score += score
                best_score = score if best_score is None else max(best_score, score)
            self.rebuild_action_heap_locked()
        count = len(updates)
        model_confidence = safe_float(model_result.get("confidence", 0.0), 0.0)
        avg_confidence = (sum(item[3] for item in updates) / count if count else 0.0)
        return {"trained": count, "model_trained": safe_int(model_result.get("trained", 0), 0), "model_loss": safe_float(model_result.get("loss", 0.0), 0.0), "best_score": round(best_score or 0.0, 4), "avg_score": round(total_score / count if count else 0.0, 4), "avg_confidence": round(clamp(avg_confidence * 0.6 + model_confidence * 0.4, 0.0, 1.0), 4)}

    def rebuild_action_heap_locked(self):
        heap = []
        for index, record in enumerate(self.records):
            if not record.get("mouse_action"):
                continue
            reward = safe_float(record.get("sleep_policy_reward", record.get("reward", 0.0)), 0.0)
            confidence = clamp(record.get("sleep_confidence", 0.0), 0.0, 1.0)
            item = (reward * (0.75 + confidence * 0.25), index)
            if len(heap) < self.settings.global_action_heap_limit:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
        self.global_action_heap = heap

    def human_score(self, action):
        return self.profile.score(action)


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
        reward = max(-80.0, safe_float(record.get("sleep_policy_reward", record.get("reward", 0.0)), 0.0))
        human_score = clamp(record.get("human_score", 50.0), 0.0, 100.0)
        novelty = clamp(record.get("sleep_novelty", record.get("novelty", 50.0)), 0.0, 100.0)
        source_bonus = 4.0 if record.get("mode") == "learning" else 0.0
        sleep_confidence = clamp(record.get("sleep_confidence", 0.0), 0.0, 1.0)
        return similarity * 100.0 * self.settings.action_score_similarity_weight + reward * self.settings.action_score_reward_weight * (1.0 + sleep_confidence * (35.0 / 100.0)) + human_score * self.settings.action_score_human_weight + novelty * self.settings.action_score_novelty_weight + source_bonus

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

    def fallback_action(self, randomness=0.0):
        learned = self.pool.best_global_action()
        if randomness > 0.0 and random.random() < clamp(randomness, 0.0, 1.0):
            return self.random_action(1.0 + randomness), "zero_score_random_exploration"
        if learned and random.random() < self.settings.global_action_probability:
            return self.mutate_action(learned, 1.8 + randomness), "global_experience"
        if randomness > 0.0:
            return self.random_action(1.0 + randomness), "zero_score_random_exploration"
        return None, "observe_only"

    def choose(self, hash_value, novelty, batch, screen_score_total, zero_score_factor=0.0):
        rate = clamp(self.exploration_rate(novelty, screen_score_total) + clamp(zero_score_factor, 0.0, 1.0) * (1.0 - self.settings.explore_min_rate), self.settings.explore_min_rate, self.settings.explore_max_rate)
        usable = []
        for item in batch:
            action = item["record"].get("mouse_action")
            if not action or safe_float(item["record"].get("reward", 0.0), 0.0) < -60.0:
                continue
            score = self.score_candidate(item)
            usable.append((math.exp(clamp(score, self.settings.reward_total_min, self.settings.reward_total_max) / self.settings.softmax_temperature), {"item": item, "score": score, "action": action}))
        if random.random() < rate or not usable:
            action, reason = self.fallback_action(zero_score_factor)
            decision = {"reason": reason, "exploration_rate": rate, "zero_score_factor": round(clamp(zero_score_factor, 0.0, 1.0), 4), "candidate_count": len(usable), "confidence": 0.0, "nearest_similarity": round(batch[0]["similarity"], 4) if batch else 0.0}
        else:
            chosen = weighted_choice(usable)
            item = chosen["item"]
            confidence = clamp(item.get("similarity", 0.0) * 0.65 + clamp(chosen.get("score", 0.0), 0.0, 200.0) / 200.0 * (35.0 / 100.0), 0.0, 1.0)
            action = self.mutate_action(chosen["action"], 1.0 - confidence + rate + zero_score_factor)
            decision = {"reason": "nearest_rewarded_experience", "exploration_rate": rate, "zero_score_factor": round(clamp(zero_score_factor, 0.0, 1.0), 4), "candidate_count": len(usable), "confidence": round(confidence, 4), "nearest_similarity": round(batch[0]["similarity"], 4) if batch else 0.0, "chosen_similarity": round(item.get("similarity", 0.0), 4), "chosen_reward": round(safe_float(item["record"].get("reward", 0.0), 0.0), 2), "chosen_record_id": item["record"].get("id")}
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
        self.current = None
        self.move_buffer = []
        self.move_action_id = None
        self.listener = None
        self.wake = threading.Event()

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
            self.current = None
            self.move_buffer.clear()
            self.move_action_id = None
            self.wake.clear()

    def active(self):
        return self.get_mode() == "learning"

    def capture_event(self, kind, x, y, extra=None, allow_current=False):
        if not self.active():
            return None
        self.on_activity()
        self.wake.set()
        rect = self.get_rect()
        if not rect:
            return None
        inside = point_inside(rect, x, y)
        if not inside and not allow_current:
            return None
        previous = None
        if self.current and self.current.get("path"):
            previous = self.current["path"][-1]
        elif self.move_buffer:
            previous = self.move_buffer[-1]
        return build_mouse_event(kind, x, y, rect, previous=previous, extra=extra)

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
            event = self.capture_event("move", x, y, allow_current=bool(self.current))
        if not event:
            return
        with self.lock:
            if self.current:
                self.current["path"].append(event)
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
        with self.lock:
            event = self.capture_event("press" if pressed else "release", x, y, {"button": str(button)}, allow_current=(not pressed and bool(self.current)))
        if not event:
            return
        with self.lock:
            if pressed:
                self.flush_move_locked(force=True, now_perf=event["t"])
                action_id = uuid.uuid4().hex
                self.current = {"action_id": action_id, "type": "click", "button": str(button), "source": "user", "started_at": now_text(), "started_perf": event["t"], "t0": event["t"], "start_abs": [int(x), int(y)], "path": [event]}
                self.push_start_marker(action_id, event, "click")
            elif self.current:
                self.current["path"].append(event)
                start_abs = self.current["start_abs"]
                end_abs = [int(x), int(y)]
                self.current.update({"end_abs": end_abs, "ended_at": now_text(), "ended_perf": event["t"], "duration": round(max(0.0, event["t"] - self.current.get("t0", event["t"])), 6)})
                if int(start_abs[0]) != int(end_abs[0]) or int(start_abs[1]) != int(end_abs[1]):
                    self.current["type"] = "drag"
                self.actions.append(self.current)
                self.current = None
                self.wake.set()

    def pop_start_markers(self):
        with self.lock:
            items = list(self.start_markers)
            self.start_markers.clear()
            return items

    def pop_actions(self):
        with self.lock:
            self.flush_move_locked(force=self.move_flush_due())
            items = list(self.actions)
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

    def smooth_points(self, start, end, duration):
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
            points.append((int(round(x)), int(round(y))))
        return points

    def stoppable_sleep(self, seconds, stop_event, should_stop):
        deadline = time.perf_counter() + max(0.0, seconds)
        while time.perf_counter() < deadline:
            if stop_event.is_set() or should_stop():
                stop_event.set()
                break
            stop_event.wait(min(self.settings.generated_sleep_event_wait, max(0.0, deadline - time.perf_counter())))

    def move_smooth(self, start, end, duration, stop_event, should_stop, rect=None, previous=None):
        points = self.smooth_points(start, end, duration)
        actual = []
        delay = duration / max(1, len(points) - 1) if duration > 0.0 else 0.0
        last = previous
        for point in points:
            if stop_event.is_set() or should_stop():
                stop_event.set()
                break
            self.controller.position = point
            event = build_mouse_event("move", point[0], point[1], rect, previous=last)
            event["source"] = "ai"
            actual.append(event)
            last = event
            if delay > 0.0:
                self.stoppable_sleep(delay, stop_event, should_stop)
        return actual

    def execute(self, action, rect, stop_event, should_stop):
        if not action:
            return None
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
        actual_path = self.move_smooth(current, start_abs, approach_duration, stop_event, should_stop, rect=rect)
        if stop_event.is_set():
            return None
        started_at = now_text()
        action_t = time.perf_counter()
        pressed = False
        try:
            if action_type == "drag":
                self.controller.press(button)
                pressed = True
                actual_path.extend(self.move_smooth(start_abs, end_abs, main_duration, stop_event, should_stop, rect=rect, previous=actual_path[-1] if actual_path else None))
            elif action_type == "scroll":
                scroll = action.get("scroll") or [0, 0]
                self.controller.scroll(int(scroll[0]), int(scroll[1]))
                actual_path.append(build_mouse_event("scroll", start_abs[0], start_abs[1], rect, previous=actual_path[-1] if actual_path else None, extra={"source": "ai", "scroll": action.get("scroll") or [0, 0]}))
            else:
                self.controller.press(button)
                pressed = True
                hold_floor = min(self.settings.random_click_duration_min, self.settings.random_click_duration_max)
                hold_ceiling = max(self.settings.random_click_duration_min, self.settings.random_click_duration_max)
                hold_duration = clamp(main_duration, hold_floor, hold_ceiling if hold_ceiling > 0.0 else self.settings.generated_click_hold_max)
                self.stoppable_sleep(clamp(hold_duration, 0.0, self.settings.generated_click_hold_max), stop_event, should_stop)
                actual_path.append(build_mouse_event("release", end_abs[0], end_abs[1], rect, previous=actual_path[-1] if actual_path else None, extra={"source": "ai", "button": str(button)}))
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
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def enqueue_image(self, analyzer, image, path, store=None, priority="normal"):
        critical = priority == "critical"
        return self.enqueue({"type": "image", "analyzer": analyzer, "image": image.copy() if image else None, "path": path, "store": store, "priority": priority}, block_when_full=critical)

    def enqueue_record(self, store, record):
        return self.enqueue({"type": "record", "store": store, "record": copy.deepcopy(record)}, block_when_full=True)

    def enqueue(self, job, block_when_full=False):
        if self.stop_event.is_set():
            return False
        try:
            self.jobs.put_nowait(job)
            return True
        except queue.Full:
            if block_when_full:
                self.jobs.put(job)
                return True
            store = job.get("store")
            if store:
                try:
                    store.log_error("async_persistence_queue_full", RuntimeError("image_job_dropped"), {"path": str(job.get("path"))})
                except Exception:
                    pass
            self.image_dropped += 1
            return False

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
                    store.append_experience(job["record"])
                    store.flush_state(min_interval=self.settings.persistence_close_seconds, max_pending=max(1, self.settings.async_queue_size // max(1, self.settings.global_action_heap_limit // self.settings.local_action_heap_limit)))
            except Exception as exc:
                store = job.get("store") if isinstance(job, dict) else None
                if store:
                    try:
                        store.log_error("async_persistence", exc, {"type": job.get("type")})
                    except Exception:
                        pass
            finally:
                self.jobs.task_done()

    def flush(self):
        self.jobs.join()

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
        pool_count = panel.experience_pool.count() if panel.experience_pool else 0
        screen_score_total_value = panel.store.screen_score_total if panel.store else 0.0
        panel.hardware_state = panel.refresh_hardware_state()
        settings = panel.adaptive_policy.build(panel.settings, rect, pool_count, screen_score_total_value, hardware=panel.hardware_state)
        panel.apply_runtime_settings(settings)
        return settings

    def observe_screen(self, analyzer, session_id, start, rect):
        return self.panel.capture_snapshot(analyzer, "training", session_id, start, rect=rect, priority="critical")

    def decide_action(self, snapshot, zero_score_factor=0.0):
        panel = self.panel
        novelty, batch = panel.experience_pool.novelty(snapshot.hash_value)
        screen_score_total = panel.store.screen_score_total if panel.store else 0.0
        return panel.brain.choose(snapshot.hash_value, novelty, batch, screen_score_total, zero_score_factor=zero_score_factor)

    def should_stop(self, start, config, stop_event):
        panel = self.panel
        deadline = start + max(1, config.training_seconds)
        guarded = should_stop_run(stop_event, deadline, panel.should_stop_by_escape)
        if guarded:
            panel.termination_reason = guarded
            stop_event.set()
            return True
        elapsed = time.perf_counter() - start
        check = panel.window_manager.check_window(force=True)
        if not check.ok:
            panel.termination_reason = "window_invalid"
            stop_event.set()
            panel.ui(lambda r=check.reason: panel.status_var.set(f"训练模式结束：雷电模拟器窗口异常：{r}"))
            return True
        if not panel.cursor_inside_window(2):
            panel.termination_reason = "window_invalid"
            stop_event.set()
            panel.ui(lambda: panel.status_var.set("训练模式结束：鼠标位于雷电模拟器窗口外"))
            return True
        panel.ensure_cursor_inside_window(check.rect)
        idle_seconds = panel.learning_idle_seconds()
        if idle_seconds >= config.still_seconds:
            panel.termination_reason = "still_timeout"
            stop_event.set()
            panel.ui(lambda: panel.status_var.set("训练模式结束：鼠标静止超时"))
            return True
        panel.update_progress(0.0)
        remaining = max(0.0, config.training_seconds - elapsed)
        panel.ui(lambda r=remaining: panel.progress_label_var.set(f"训练模式进度保持 0%｜剩余 {r:.1f} 秒"))
        return False

    def execute_and_record(self, analyzer, session_id, start, rect, snapshot, action, decision, stop_event):
        panel = self.panel
        if action.get("end_rel") is None:
            action["end_rel"] = action.get("start_rel", [0.5, 0.5])
        latest_rect = panel.current_rect()
        if not latest_rect or latest_rect != rect:
            panel.write_record("training", session_id, snapshot, None, "ai_mouse_failed", decision=decision, planned_action=action, failed_action=True, window_rect_changed=True, execution_error="window_rect_changed")
            panel.adaptive_policy.observe_execution(success=False)
            return False, {"failure_reason": "window_rect_changed"}
        actual = panel.executor.execute(action, rect, stop_event, panel.should_stop_by_escape)
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
        after_snapshot = panel.capture_snapshot(analyzer, "training", session_id, start, rect=rect, priority="critical")
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
        self.main_thread_events = queue.Queue()
        self.runtime_context = RuntimeContext(self)
        self.mode_controller = ModeController(self)
        self.migration_service = MigrationService(self)
        self.metrics_presenter = MetricsPresenter(self)
        self.learning_service = LearningService(self)
        self.training_service = TrainingService(self)
        self.events.subscribe("window_state_changed", lambda event: self.ui(lambda e=event: self.status_var.set(f"窗口状态事件：{e.get('reason', 'ok')}")))
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
        self.sleep_seconds_var = tk.StringVar(value=str(DEFAULT_SLEEP_SECONDS))
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
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="0%")
        self.last_learning_event_perf = 0.0
        self.last_learning_event_hash = None
        self.hardware_last_full_refresh_perf = 0.0
        self.hardware_last_light_refresh_perf = 0.0
        self.progress_label_var = tk.StringVar(value="进度")
        self.runtime_value_specs = {
            "training_seconds": ("训练秒数", self.training_seconds_var, DEFAULT_TRAINING_SECONDS, safe_int, 1),
            "sleep_seconds": ("睡眠秒数", self.sleep_seconds_var, DEFAULT_SLEEP_SECONDS, safe_int, 1),
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

    def build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        scale = max(0.8, min(2.2, float(self.winfo_fpixels("1i")) / 96.0))
        pane_h = max(1, self.winfo_height() or self.settings.ui_height)
        font_scale = max(0.8, min(2.1, (pane_h / max(1, self.settings.ui_min_height)) ** 0.5 * scale))
        title_size = max(11, int(round(14 * font_scale)))
        value_size = max(10, int(round(12 * font_scale)))
        card_size = max(8, int(round(9 * font_scale)))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", title_size, "bold"))
        style.configure("CardTitle.TLabel", font=("Microsoft YaHei UI", card_size))
        style.configure("Value.TLabel", font=("Microsoft YaHei UI", value_size, "bold"))
        style.configure("Hint.TLabel", foreground="#555555")
        root = ttk.Frame(self, padding=self.settings.ui_padding)
        root.pack(fill="both", expand=True)
        self.scroll_canvas = tk.Canvas(root, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
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
        ldplayer_modify = ttk.Button(path_frame, text="修改", command=self.choose_ldplayer)
        ldplayer_modify.grid(row=0, column=2, padx=(8, 0), pady=6)
        self.modify_buttons.append(ldplayer_modify)
        ttk.Label(path_frame, text="数据存储").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(path_frame, textvariable=self.data_var, justify="right", state="readonly").grid(row=1, column=1, sticky="ew", pady=6)
        data_modify = ttk.Button(path_frame, text="修改", command=self.choose_data)
        data_modify.grid(row=1, column=2, padx=(8, 0), pady=6)
        self.modify_buttons.append(data_modify)
        time_frame = ttk.Frame(path_frame)
        time_frame.grid(row=2, column=1, columnspan=2, sticky="w", pady=6)
        ttk.Label(path_frame, text="时间设置").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=6)
        for field, label in (("training_seconds", "训练/秒"), ("sleep_seconds", "睡眠/秒"), ("still_seconds", "静止/秒"), ("experience_pool_gb", "经验池/GB"), ("ai_model_limit", "AI模型/个")):
            variable = self.runtime_value_specs[field][1]
            item_frame = ttk.Frame(time_frame)
            item_frame.pack(side="left", padx=(0, 12))
            ttk.Label(item_frame, text=label).pack(side="left")
            ttk.Entry(item_frame, textvariable=variable, width=10, state="readonly", justify="right").pack(side="left", padx=(6, 4))
            button = ttk.Button(item_frame, text="修改", command=lambda name=field: self.modify_runtime_value(name))
            button.pack(side="left")
            self.modify_buttons.append(button)
        button_frame = ttk.Frame(container)
        button_frame.pack(fill="x", pady=(4, 12))
        self.button_frame = button_frame
        self.learning_button = ttk.Button(button_frame, text="学习模式", command=self.learning_mode)
        self.training_button = ttk.Button(button_frame, text="训练模式", command=self.training_mode)
        self.sleep_button = ttk.Button(button_frame, text="睡眠模式", command=self.sleep_mode)
        self.mode_buttons = [self.learning_button, self.training_button, self.sleep_button]
        self.control_buttons = [
            self.learning_button,
            self.training_button,
            self.sleep_button,
            ttk.Button(button_frame, text="终止当前模式", command=self.stop_current_mode),
            ttk.Button(button_frame, text="退出", command=self.close)
        ]
        self.reflow_buttons()
        status_frame = ttk.LabelFrame(container, text="状态", padding=self.settings.ui_section_padding)
        status_frame.pack(fill="both", expand=True)
        self.metrics_frame = ttk.Frame(status_frame)
        self.metrics_frame.grid(row=0, column=0, sticky="nsew")
        status_frame.columnconfigure(0, weight=1)
        metrics = [("当前模式", self.mode_var), ("画面评分累计", self.screen_score_total_var), ("经验条数", self.pool_var), ("画面评分", self.novelty_var), ("鼠标相似度", self.human_var), ("画面奖励", self.screen_reward_var), ("鼠标奖惩", self.action_reward_var), ("本次奖励", self.reward_var), ("AI决策", self.ai_var)]
        for title, variable in metrics:
            self.metric_items.append(self.create_metric(self.metrics_frame, title, variable))
        self.reflow_metrics()
        ttk.Label(status_frame, text="快捷键", style="CardTitle.TLabel").grid(row=1, column=0, sticky="w", pady=(18, 6), padx=(0, 12))
        hint = "学习模式期间，全局鼠标静止超时会自动结束。雷电模拟器窗口外的新动作不会记录为学习动作，但会重置静止倒计时。ESC 终止当前学习、训练或睡眠。截图与坐标均使用雷电客户区。"
        self.hint_label = ttk.Label(status_frame, text=hint, wraplength=max(320, self.settings.ui_width - 120), style="Hint.TLabel")
        self.hint_label.grid(row=2, column=0, sticky="ew", pady=6)
        progress_frame = ttk.LabelFrame(container, text=self.progress_label_var.get(), padding=self.settings.ui_section_padding)
        self.progress_label_var.trace_add("write", lambda *args: progress_frame.configure(text=self.progress_label_var.get()))
        progress_frame.pack(fill="x", pady=(12, 0))
        progress_frame.columnconfigure(0, weight=1)
        ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100).grid(row=0, column=0, sticky="ew")
        ttk.Label(progress_frame, textvariable=self.progress_text_var, width=8, anchor="e").grid(row=0, column=1, sticky="e", padx=(10, 0))
        ttk.Label(container, textvariable=self.status_var).pack(anchor="w", pady=(10, 0))


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
        try:
            self.after(0, func)
        except Exception as exc:
            self.log_exception("ui.dispatch", exc)

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
        try:
            self.after(0, apply)
        except Exception as exc:
            self.log_exception("ui.sync.dispatch", exc)
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
        ok, _ = validate_ldplayer_executable(Path(self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH), self.settings, require_attach=False)
        return ok and bool(windows_runtime_report(Path(self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH)).get("ok"))

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
            self.status_var.set("雷电运行环境未就绪：学习/训练需 Windows 桌面与雷电窗口；睡眠模式仅需数据存储可用")
        self.runtime_environment_last_ready = online_enabled
        return online_enabled

    def runtime_environment_refresh_delay_ms(self):
        source = max(self.settings.window_event_wait, self.settings.ui_event_coalesce_seconds, self.settings.key_debounce_seconds)
        return max(200, min(3000, int(source * 1000)))

    def process_main_thread_events(self):
        while True:
            try:
                event = self.main_thread_events.get_nowait()
            except queue.Empty:
                break
            try:
                if event.get("type") == "restart_training":
                    self.status_var.set("睡眠模式已保存，准备重新进入训练模式")
                    config = event.get("config") or self.read_config()
                    old_token = event.get("token")
                    reason = event.get("reason") or "completed"
                    if self.is_run_active(old_token, "sleep"):
                        self.finish_run(old_token, "睡眠模式已退出，数据已保存", 0.0, release=False, reason=reason)
                    if self.current_mode() == "idle":
                        self.request_active_mode("training")
            except Exception as exc:
                self.log_exception("main_thread_event", exc, event)

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
        if field in ("training_seconds", "sleep_seconds", "ai_model_limit"):
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
        ok, reason = validate_ldplayer_executable(path, self.settings, require_attach=True)
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
        old_path = Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)
        new_path = Path(path)
        if old_path == new_path:
            self.data_var.set(str(new_path))
            self.save_persistent_settings()
            self.update_mode_button_states()
            return
        token, stop_event = self.begin_run("migration", reason="click_modify_data_path")
        if not token:
            self.status_var.set("请先终止当前模式，或等待当前模式结束")
            return
        self.status_var.set("正在迁移数据")
        self.update_progress(0.0)
        values = {"ldplayer_path": self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH, "training_seconds": max(1, safe_int(self.training_seconds_var.get(), DEFAULT_TRAINING_SECONDS)), "sleep_seconds": max(1, safe_int(self.sleep_seconds_var.get(), DEFAULT_SLEEP_SECONDS)), "still_seconds": max(0.1, safe_float(self.still_seconds_var.get(), DEFAULT_STILL_SECONDS)), "experience_pool_gb": max(0.1, safe_float(self.experience_pool_gb_var.get(), DEFAULT_EXPERIENCE_POOL_GB)), "ai_model_limit": max(1, safe_int(self.ai_model_limit_var.get(), DEFAULT_AI_MODEL_LIMIT))}
        self.mode_thread = threading.Thread(target=self.migration_service.run, args=(token, old_path, new_path, stop_event, values), daemon=True)
        self.mode_thread.start()

    def migration_items(self, old_path):
        root = Path(old_path)
        items = []
        for name in ("screens", "models", "experience.jsonl", "state.json", "settings.json", "errors.jsonl"):
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
                self.update_progress(clamp(copied / max(1, total) * 100.0, 0.0, 99.0))
        try:
            shutil.copystat(source, target)
        except Exception:
            pass
        return copied, True

    def migration_known_names(self):
        return {"screens", "models", "experience.jsonl", "state.json", "settings.json", "errors.jsonl"}

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
        experience = root / "experience.jsonl"
        if experience.exists():
            try:
                with experience.open("r", encoding="utf-8") as file:
                    for _ in file:
                        lines += 1
            except Exception:
                pass
        return {"screens": screen_count, "models": model_count, "experience_lines": lines, "settings": (root / "settings.json").exists(), "state": (root / "state.json").exists()}

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
                    self.update_progress(clamp(copied / total * 100.0, 0.0, 99.0))
            if not stop_event.is_set() and self.is_run_active(token, "migration"):
                DataStore(temp_root).save_settings({"training_seconds": values["training_seconds"], "sleep_seconds": values["sleep_seconds"], "still_seconds": values["still_seconds"], "experience_pool_gb": values["experience_pool_gb"], "ai_model_limit": values["ai_model_limit"]})
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
                self.ui(lambda path=str(new_root): self.data_var.set(path))
                self.app_config_store.save_settings({"ldplayer_path": values["ldplayer_path"], "data_path": str(new_root)})
                self.store = DataStore(new_root)
                self.experience_pool = ExperiencePool(self.settings, self.store.load_experience(self.settings.experience_load_limit), self.store.load_latest_model_state(self.settings))
                self.brain = ActionBrain(self.experience_pool, self.settings)
                self.update_progress(100.0)
                self.finish_run(token, reason, 100.0, release=False, reason="completed")
            elif self.is_run_active(token, "migration"):
                self.finish_run(token, reason, self.progress_value, release=False, reason="user_stop")
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
            DataStore(Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)).save_settings({"training_seconds": DEFAULT_TRAINING_SECONDS, "sleep_seconds": DEFAULT_SLEEP_SECONDS, "still_seconds": DEFAULT_STILL_SECONDS, "experience_pool_gb": DEFAULT_EXPERIENCE_POOL_GB, "ai_model_limit": DEFAULT_AI_MODEL_LIMIT})
        self.training_seconds_var.set(str(max(1, safe_int(data_settings.get("training_seconds", self.training_seconds_var.get()), DEFAULT_TRAINING_SECONDS))))
        self.sleep_seconds_var.set(str(max(1, safe_int(data_settings.get("sleep_seconds", self.sleep_seconds_var.get()), DEFAULT_SLEEP_SECONDS))))
        self.still_seconds_var.set(str(max(0.1, safe_float(data_settings.get("still_seconds", self.still_seconds_var.get()), DEFAULT_STILL_SECONDS))))
        self.experience_pool_gb_var.set(str(max(0.1, safe_float(data_settings.get("experience_pool_gb", self.experience_pool_gb_var.get()), DEFAULT_EXPERIENCE_POOL_GB))))
        self.ai_model_limit_var.set(str(max(1, safe_int(data_settings.get("ai_model_limit", self.ai_model_limit_var.get()), DEFAULT_AI_MODEL_LIMIT))))
        self.update_mode_button_states()

    def save_persistent_settings(self):
        data_path = Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)
        self.app_config_store.save_settings({"ldplayer_path": self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH, "data_path": str(data_path)})
        DataStore(data_path).save_settings({"training_seconds": max(1, safe_int(self.training_seconds_var.get(), DEFAULT_TRAINING_SECONDS)), "sleep_seconds": max(1, safe_int(self.sleep_seconds_var.get(), DEFAULT_SLEEP_SECONDS)), "still_seconds": max(0.1, safe_float(self.still_seconds_var.get(), DEFAULT_STILL_SECONDS)), "experience_pool_gb": max(0.1, safe_float(self.experience_pool_gb_var.get(), DEFAULT_EXPERIENCE_POOL_GB)), "ai_model_limit": max(1, safe_int(self.ai_model_limit_var.get(), DEFAULT_AI_MODEL_LIMIT))})

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
        AllowedUserEditPolicy.assert_allowed("ldplayer_path", "data_path", "training_seconds", "sleep_seconds", "still_seconds", "experience_pool_gb", "ai_model_limit")
        training_seconds = max(1, safe_int(self.training_seconds_var.get(), DEFAULT_TRAINING_SECONDS))
        sleep_seconds = max(1, safe_int(self.sleep_seconds_var.get(), DEFAULT_SLEEP_SECONDS))
        still_seconds = max(0.1, safe_float(self.still_seconds_var.get(), DEFAULT_STILL_SECONDS))
        experience_pool_gb = max(0.1, safe_float(self.experience_pool_gb_var.get(), DEFAULT_EXPERIENCE_POOL_GB))
        ai_model_limit = max(1, safe_int(self.ai_model_limit_var.get(), DEFAULT_AI_MODEL_LIMIT))
        self.training_seconds_var.set(str(training_seconds))
        self.sleep_seconds_var.set(str(sleep_seconds))
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
        settings = derive_runtime_settings(base_settings=self.settings, rect=self.current_rect() or self.screen_rect(), pool_count=self.experience_pool.count() if self.experience_pool else 0, capture_ms=capture_ms, cpu_load=cpu_load, execution_ms=self.adaptive_policy._avg(self.adaptive_policy.execution_latency_ms, 0.0), window_instability=self.adaptive_policy._avg(self.adaptive_policy.window_change_flags, 0.0), recent_success=self.adaptive_policy._avg(self.adaptive_policy.outcome_flags, 1.0), screen_score_total=screen_score_total_value, learning_similarity=self.adaptive_policy._avg(self.adaptive_policy.learning_similarity, 0.97), hardware=self.hardware_state)
        self.settings = settings
        self.escape_monitor.debounce_seconds = settings.key_debounce_seconds
        self.ui(lambda: self.minsize(settings.ui_min_width, settings.ui_min_height))
        return Config(Path(self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH), data_path, training_seconds, sleep_seconds, still_seconds, experience_pool_gb, ai_model_limit, settings)

    def apply_runtime_settings(self, settings):
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
        self.ui(lambda m=mode: self.mode_var.set(MODE_NAMES.get(m, m)))

    def transition(self, expected, target, reason=None, token=None, deadline=None):
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
                stop_event = self.stop_event if token is not None else threading.Event()
                self.stop_event = stop_event
                self.mode = target
                event = self.events.publish("mode_transition", source=source, target=target, reason=transition_reason, token=session_token)
                session = ModeSession(session_token, target, time.perf_counter(), deadline, stop_event, transition_reason, event["sequence"])
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

    def finish_run(self, token, status, progress=0.0, release=True, reason=None):
        mapped_reason = reason or "completed"
        if mapped_reason not in TERMINATION_REASONS and not str(mapped_reason).startswith("window_"):
            mapped_reason = "runtime_error"
        if str(mapped_reason).startswith("window_"):
            mapped_reason = "window_invalid"
        with self.state_lock:
            source_mode = self.mode if token == self.run_token else None
        if not self.transition(None, "idle", reason=mapped_reason, token=token):
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
        pos = self.observe_cursor_activity()
        rect = self.current_rect()
        if pos is None or not rect:
            return False
        left, top, right, bottom = rect
        return left - tolerance <= pos[0] < right + tolerance and top - tolerance <= pos[1] < bottom + tolerance

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

    def ensure_runtime(self, config):
        valid_path, path_reason = validate_ldplayer_executable(config.ldplayer_path, config.settings, require_attach=False)
        if not valid_path:
            self.ui(lambda r=path_reason: messagebox.showerror("雷电路径不合法", r))
            return False
        report = windows_runtime_report(config.ldplayer_path)
        if not report.get("ok"):
            self.log_exception("runtime.environment", RuntimeError("environment_not_ready"), report)
            self.ui(lambda r=report: messagebox.showerror("运行环境不符合要求", json.dumps(r, ensure_ascii=False, indent=2)))
            return False
        self.ensure_storage_runtime(config)
        if not self.window_manager or self.window_manager.executable_path != config.ldplayer_path or self.window_manager.settings != config.settings:
            self.window_manager = WindowManager(config.ldplayer_path, config.settings)
        if not self.executor or self.executor.window_manager is not self.window_manager or self.executor.settings != config.settings:
            self.executor = HumanMouseExecutor(self.window_manager, config.settings)
        return self.window_manager.launch_or_attach()

    def learning_mode(self):
        self.request_active_mode("learning")

    def training_mode(self):
        self.request_active_mode("training")

    def request_active_mode(self, target_mode):
        if self.required_import_error():
            self.show_import_error()
            return
        config = self.read_config()
        self.status_var.set("正在检查运行环境")
        if not self.ensure_runtime(config):
            self.status_var.set("运行环境不符合要求，未进入模式")
            return
        token, stop_event = self.begin_run("starting", reason="click_learning" if target_mode == "learning" else "click_training")
        if not token:
            self.status_var.set("请先终止当前模式，或等待当前模式结束")
            return
        if not self.minimize_panel_for_active_mode(config):
            self.finish_run(token, "控制面板最小化失败", 0.0, reason="minimize_failed")
            return
        self.update_progress(0.0)
        self.status_var.set("正在启动或连接雷电模拟器")
        self.mode_thread = threading.Thread(target=self.mode_job, args=(token, target_mode, config, stop_event), daemon=True)
        self.mode_thread.start()

    def minimize_panel_for_active_mode(self, config):
        def apply():
            try:
                self.iconify()
                self.update_idletasks()
                return self.state() == "iconic"
            except Exception:
                return False
        return bool(self.ui_sync(apply, config.settings.window_event_wait))

    def mode_job(self, token, mode, config, stop_event):
        try:
            if not self.ensure_runtime(config):
                if self.is_run_active(token):
                    self.ui(lambda: messagebox.showerror("未找到窗口", "没有找到雷电模拟器窗口，请确认路径正确或手动启动雷电模拟器。"))
                    self.finish_run(token, "未找到雷电模拟器", 0.0, reason="runtime_error")
                return
            if stop_event.is_set() or not self.is_run_active(token):
                self.finish_run(token, "当前模式已终止", 0.0, reason="user_stop")
                return
            self.window_manager.foreground()
            stop_event.wait(config.settings.window_event_wait)
            check = self.window_manager.check_window(force=True)
            if not check.ok:
                self.finish_run(token, f"雷电模拟器窗口异常：{check.reason}", 0.0, reason=f"window_{check.reason}")
                return
            if not self.ensure_cursor_inside_window(check.rect):
                self.finish_run(token, "无法确保鼠标位于雷电模拟器窗口内", 0.0, reason="window_invalid")
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

    def request_escape(self):
        with self.state_lock:
            if self.mode == "sleep":
                self.termination_reason = "esc"
                if self.active_session:
                    self.active_session.termination_reason = "esc"
                self.stop_event.set()
                self.ui(lambda: self.status_var.set("睡眠模式正在保存数据"))
                return
        self.stop_current_mode()

    def stop_current_mode(self):
        with self.state_lock:
            mode = self.mode
            token = self.run_token
            if mode not in ["starting", "learning", "training", "sleep", "migration"]:
                self.ui(lambda: self.status_var.set("当前没有正在运行的模式"))
                return
            self.stop_event.set()
            progress_now = self.progress_value
        if not self.transition(mode, "idle", reason="user_stop", token=token):
            return
        self.update_progress(self.idle_progress_value(mode, progress_now), force=True)
        self.ui(lambda: self.status_var.set("当前模式已终止"))
        self.ui(self.release_window_and_panel)

    def sleep_mode(self, restart_training=False):
        token, stop_event = self.begin_run("sleep")
        if not token:
            self.status_var.set("请先终止当前模式，或等待当前模式结束")
            return
        config = self.read_config()
        try:
            self.ensure_storage_runtime(config)
        except Exception as exc:
            self.log_exception("sleep.storage", exc, {"data_path": str(config.data_path)})
            self.ui(lambda e=str(exc): messagebox.showerror("睡眠模式数据环境异常", e))
            self.finish_run(token, "睡眠模式数据环境异常", 0.0, release=False, reason="runtime_error")
            return
        self.status_var.set("睡眠模式运行中")
        self.update_progress(0.0)
        self.mode_thread = threading.Thread(target=self.sleep_loop, args=(token, config, stop_event, restart_training), daemon=True)
        self.mode_thread.start()

    def sleep_progress_fields(self, started_perf, sleep_seconds, completed_steps, target_training_steps, compaction_progress):
        elapsed_ratio = clamp((time.perf_counter() - started_perf) / max(1.0, safe_float(sleep_seconds, 1.0)), 0.0, 1.0)
        training_ratio = clamp(safe_float(completed_steps, 0.0) / max(1.0, safe_float(target_training_steps, 1.0)), 0.0, 1.0)
        compact_ratio = clamp(compaction_progress, 0.0, 1.0)
        previous_ratio = clamp(getattr(self, "progress_value", 0.0) / 100.0, 0.0, 1.0)
        review_ratio = 1.0 if completed_steps > 0 or target_training_steps <= 0 else min(elapsed_ratio, 0.25)
        save_ratio = 0.0
        task_ratio = review_ratio * 0.2 + training_ratio * 0.55 + compact_ratio * 0.2 + save_ratio * 0.05
        floor_ratio = elapsed_ratio * 0.15
        return {"time": elapsed_ratio, "review": review_ratio, "training": training_ratio, "compaction": compact_ratio, "save": save_ratio, "overall": max(previous_ratio, task_ratio, floor_ratio)}

    def sleep_progress_percent(self, started_perf, sleep_seconds, completed_steps, target_training_steps, compaction_progress):
        return clamp(self.sleep_progress_fields(started_perf, sleep_seconds, completed_steps, target_training_steps, compaction_progress)["overall"] * 100.0, 0.0, 99.0)

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

    def sleep_completion_reached(self, completed, target_training_steps, recent_improvements, improvement_threshold, batch_confidence, confidence_target, compaction_complete):
        pending_state = safe_int(getattr(self.store, "pending_state_writes", 0), 0) if self.store else 0
        improvement_ready = bool(recent_improvements) and sum(recent_improvements) >= improvement_threshold * len(recent_improvements)
        return completed >= target_training_steps and improvement_ready and batch_confidence >= confidence_target and compaction_complete and pending_state == 0

    def sleep_loop(self, token, config, stop_event, restart_training=False):
        started = self.events.publish("sleep_started", seconds=config.sleep_seconds, data_path=str(config.data_path))
        completed = 0
        submitted = 0
        workers = max(1, config.settings.sleep_worker_count)
        queue_depth = max(workers, config.settings.sleep_queue_depth)
        batch_size = max(1, config.settings.sleep_batch_size)
        best_seen = None
        recent_improvements = deque(maxlen=max(1, min(max(workers, queue_depth), max(1, self.experience_pool.count() // batch_size if self.experience_pool else 1))))
        stale_batches = 0
        poor_limit = max(workers, queue_depth)
        target_training_steps = max(poor_limit, math.ceil(max(1, self.experience_pool.count() if self.experience_pool else 1) / batch_size), workers * max(1, config.settings.ui_metric_columns))
        improvement_threshold = 1.0 / max(100.0, target_training_steps * batch_size)
        confidence_target = clamp(1.0 - 1.0 / max(2.0, config.settings.nearest_top_k + batch_size), 0.5, 0.98)
        initial_compact = {"changed": False, "size_bytes": self.store.experience_pool_size_bytes() if self.store else 0, "target_bytes": max(1, int(config.experience_pool_gb * 1024 * 1024 * 1024))}
        compaction_progress = self.sleep_compaction_progress(initial_compact)
        compaction_complete = self.sleep_compaction_complete(initial_compact)
        poor_optimization = False
        time_limit_reached = False
        completed_success = False
        started_perf = time.perf_counter()
        sleep_deadline = started_perf + max(1, config.sleep_seconds)
        run_guard = lambda: should_stop_run(stop_event, sleep_deadline, self.should_stop_by_escape)
        self.ui(lambda: self.progress_label_var.set("睡眠评分复核中｜正在检查全部画面评分"))
        recheck_result = {"checked": 0, "rescored": 0, "missing": 0, "errors": 0, "image_missing": 0, "image_corrupt": 0, "hash_missing": 0, "unrecoverable": 0}
        try:
            with ScreenAnalyzer(config.settings.hash_size) as analyzer:
                recheck_result = self.experience_pool.recheck_screen_scores(self.store, analyzer, run_guard=run_guard)
            self.events.publish("sleep_screen_scores_rechecked", **recheck_result)
            self.store.merge_experience_records_by_id(copy.deepcopy(self.experience_pool.records))
        except Exception as exc:
            self.log_exception("sleep_score_recheck", exc, recheck_result)
        self.ui(lambda r=recheck_result: self.progress_label_var.set(f"睡眠训练进度｜评分复核 {r.get('checked', 0)} 条｜重评 {r.get('rescored', 0)} 条｜不可恢复 {r.get('unrecoverable', 0)} 条"))
        if self.store and recheck_result.get("unrecoverable", 0):
            self.store.log_error("sleep_score_recheck_unrecoverable", RuntimeError("unrecoverable_screen_records"), recheck_result)
        def train_once():
            if run_guard():
                return {"trained": 0, "best_score": 0.0, "avg_score": 0.0, "avg_confidence": 0.0}
            return self.experience_pool.sleep_training_step(batch_size, self.store.add_screen_score_total if self.store else None, run_guard=run_guard)
        def submit_next(executor, futures):
            nonlocal submitted
            if stop_event.is_set() or not self.is_run_active(token, "sleep"):
                return
            futures.add(executor.submit(train_once))
            submitted += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = set()
            for _ in range(queue_depth):
                submit_next(executor, futures)
            while not stop_event.is_set() and self.is_run_active(token, "sleep"):
                guarded = run_guard()
                if guarded:
                    with self.state_lock:
                        self.termination_reason = guarded
                        if self.active_session:
                            self.active_session.termination_reason = guarded
                    if guarded == "time_limit":
                        time_limit_reached = True
                        self.events.publish("sleep_time_limit_reached", seconds=config.sleep_seconds, started_sequence=started["sequence"])
                    stop_event.set()
                    break
                percent = self.sleep_progress_percent(started_perf, config.sleep_seconds, completed, target_training_steps, compaction_progress)
                self.update_progress(percent)
                if not futures:
                    submit_next(executor, futures)
                wait_left = max(0.0, sleep_deadline - time.perf_counter())
                done, futures = concurrent.futures.wait(futures, timeout=min(config.settings.sleep_event_wait, wait_left), return_when=concurrent.futures.FIRST_COMPLETED)
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
                        result = {"trained": 0, "best_score": 0.0, "avg_score": 0.0, "avg_confidence": 0.0}
                    completed += 1
                    trained += safe_int(result.get("trained", 0), 0)
                    best_score = max(best_score, safe_float(result.get("best_score", 0.0), 0.0))
                    avg_score += safe_float(result.get("avg_score", 0.0), 0.0)
                    avg_confidence += safe_float(result.get("avg_confidence", 0.0), 0.0)
                    submit_next(executor, futures)
                divisor = max(1, len(done))
                batch_score = avg_score / divisor
                batch_confidence = avg_confidence / divisor
                if trained > 0:
                    improvement_amount = best_score if best_seen is None else max(0.0, best_score - best_seen)
                    improved = best_seen is None or best_score > best_seen
                    recent_improvements.append(improvement_amount)
                    best_seen = best_score if best_seen is None else max(best_seen, best_score)
                    stale_batches = 0 if improved else stale_batches + len(done)
                    poor_optimization = completed >= poor_limit and stale_batches >= poor_limit and batch_confidence <= 1.0 / max(1, batch_size)
                    if poor_optimization:
                        stop_event.set()
                divisor = max(1, len(done))
                screen_score_total = self.store.state.get("screen_score_total", 0.0) if self.store else 0.0
                compact = {"changed": False, "size_bytes": self.store.experience_pool_size_bytes() if self.store else 0, "target_bytes": max(1, int(config.experience_pool_gb * 1024 * 1024 * 1024))}
                compaction_progress = self.sleep_compaction_progress(compact)
                compaction_complete = True
                self.events.publish("sleep_batch_completed", trained=trained, best_score=best_score, confidence=batch_confidence, completed_batches=completed)
                decision = {"reason": "sleep_prioritized_replay", "confidence": batch_confidence, "candidate_count": trained, "best_score": best_score, "completed_batches": completed, "target_training_steps": target_training_steps, "confidence_target": confidence_target, "workers": workers, "pool_compacted": compact.get("changed", False), "pool_removed": compact.get("removed", 0)}
                self.update_metrics(0.0, 50.0 + clamp(batch_confidence * 50.0, 0.0, 50.0), 0.0, batch_score, best_score, screen_score_total, decision)
                remaining_seconds = max(0.0, sleep_deadline - time.perf_counter())
                self.ui(lambda r=remaining_seconds, c=completed, t=target_training_steps, q=batch_confidence: self.progress_label_var.set(f"睡眠训练进度｜剩余 {r:.1f} 秒｜已训练批次 {c}/{t}｜当前置信度 {q * 100.0:.1f}%"))
                self.update_progress(self.sleep_progress_percent(started_perf, config.sleep_seconds, completed, target_training_steps, compaction_progress))
                if self.sleep_completion_reached(completed, target_training_steps, recent_improvements, improvement_threshold, batch_confidence, confidence_target, compaction_complete):
                    completed_success = True
                    self.events.publish("sleep_completion_reached", completed_batches=completed, target_training_steps=target_training_steps, confidence=batch_confidence, confidence_target=confidence_target)
                    break
                if poor_optimization:
                    break
            for future in futures:
                future.cancel()
        with self.state_lock:
            stopped_reason = self.termination_reason if self.termination_reason in ("esc", "user_stop") else None
        save_status = "completed" if completed_success else ("poor_optimization" if poor_optimization else ("time_limit" if time_limit_reached else (stopped_reason or "incomplete")))
        saved, save_error = self.save_sleep_data(config, save_status, run_guard=run_guard)
        if not saved:
            if self.is_run_active(token, "sleep"):
                self.finish_run(token, "保存失败：" + str(save_error), self.progress_value, release=False, reason="runtime_error")
                self.ui(lambda e=str(save_error): messagebox.showerror("保存失败", e))
            return
        final_reason = None
        if restart_training and self.is_run_active(token, "sleep") and save_status not in ("esc", "user_stop"):
            self.main_thread_events.put({"type": "restart_training", "reason": save_status, "config": config, "token": token, "created_at": now_text()})
            self.ui(self.process_main_thread_events)
            return
        if self.is_run_active(token, "sleep") and completed_success:
            final_reason = "completed"
            self.finish_run(token, "睡眠模式任务完成，数据已保存", 100.0, release=False, reason=final_reason)
        elif self.is_run_active(token, "sleep") and time_limit_reached:
            final_reason = "time_limit"
            self.finish_run(token, "睡眠模式到达时间上限，数据已保存", 100.0, release=False, reason=final_reason)
        elif self.is_run_active(token, "sleep") and poor_optimization:
            final_reason = "poor_optimization"
            self.finish_run(token, "AI模型优化效果差，已保存数据并退出睡眠模式", self.progress_value, release=False, reason=final_reason)
        elif self.is_run_active(token, "sleep"):
            final_reason = stopped_reason or "user_stop"
            self.finish_run(token, "睡眠模式已终止，数据已保存", 0.0, release=False, reason=final_reason)
        if restart_training and final_reason not in ("esc", "user_stop"):
            self.main_thread_events.put({"type": "restart_training", "reason": final_reason, "config": config, "token": token, "created_at": now_text()})
            self.ui(self.process_main_thread_events)

    def save_sleep_data(self, config, status, run_guard=None):
        if not self.store or not self.experience_pool:
            return True, None
        try:
            self.persistence_paused.set()
            if self.persistence_queue:
                self.persistence_queue.flush()
            self.store.flush_state(force=True)
            with self.experience_pool.lock:
                recheck_records = copy.deepcopy(self.experience_pool.records)
                self.store.merge_experience_records_by_id(recheck_records)
                compact = self.store.compact_experience_pool(config.experience_pool_gb, run_guard=run_guard)
                if compact.get("changed"):
                    self.events.publish("experience_pool_compaction_completed", removed=compact.get("removed", 0), size_bytes=compact.get("size_bytes", 0))
                    self.experience_pool = ExperiencePool(config.settings, self.store.load_experience(config.settings.experience_load_limit), self.store.load_latest_model_state(config.settings))
                    self.brain = ActionBrain(self.experience_pool, config.settings)
                records = copy.deepcopy(self.experience_pool.records)
                model = self.experience_pool.model
            self.store.save_ai_model_snapshot(records, config.settings, config.ai_model_limit, status, model, run_guard=run_guard)
            self.store.flush_state(force=True)
            self.events.publish("save_completed", kind="sleep_data", status=status)
            self.ui(lambda c=self.experience_pool.count(): self.pool_var.set(str(c)))
            return True, None
        except Exception as exc:
            self.log_exception("sleep_save", exc, {"status": status})
            return False, exc
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
            path = self.store.new_screen_path(mode)
            checksum = image_content_checksum(image)
            snapshot = ScreenSnapshot(path=path, relative_path=self.store.relative_path(path), hash_value=hash_value, captured_at=now_text(), perf_time=perf_time, elapsed=round(perf_time - session_start, 3), rect=tuple(rect), capture_latency_ms=round((captured_perf - perf_time) * 1000.0, 3), image_priority=priority, image_checksum=checksum)
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

    def write_record(self, mode, session_id, snapshot, action, event_name, decision=None, action_anchor_perf=None, after_snapshot=None, planned_action=None, failed_action=False, window_rect_changed=False, capture_latency_ms=None, execution_latency_ms=None, execution_error=None):
        if not self.store or not snapshot:
            return None
        before_novelty, batch = self.experience_pool.compute_screen_score(snapshot.hash_value, exact_checksum=getattr(snapshot, "image_checksum", ""))[:2]
        normalized = normalize_mouse_action(action, snapshot.rect) if action else None
        mouse_source = normalized.get("source") if normalized else "idle"
        human_score = self.experience_pool.human_score(normalized) if normalized else 50.0
        after_novelty = self.experience_pool.compute_screen_score(after_snapshot.hash_value, exact_checksum=getattr(after_snapshot, "image_checksum", ""))[0] if after_snapshot else before_novelty
        transition_reward = round(after_novelty - before_novelty, 2)
        scoring_novelty = after_novelty if normalized and after_snapshot else before_novelty
        reward_info = reward_breakdown(scoring_novelty, human_score if normalized else self.settings.score_default, self.settings)
        novelty_reward = reward_info["screen_primary_reward"]
        human_delta = reward_info["mouse_action_delta"]
        reward = reward_info["total_reward"]
        human_action_reward = max(0.0, human_delta)
        human_action_penalty = max(0.0, -human_delta)
        if failed_action:
            reward = round(-abs(max(1.0, human_action_penalty, 100.0 - human_score)), 2)
        screen_score_total = self.store.add_screen_score_total(reward)
        started_perf = safe_float(normalized.get("started_perf"), None) if normalized else None
        offset_source = started_perf if started_perf is not None else action_anchor_perf
        offset_ms = round((float(offset_source) - snapshot.perf_time) * 1000.0, 3) if offset_source is not None else None
        sims = [round(item["similarity"], 4) for item in batch]
        record_event = self.events.publish("record_ready", mode=mode, session_id=session_id, event_name=event_name)
        record = {"record_schema_version": 2, "reward_version": reward_info["reward_version"], "id": uuid.uuid4().hex, "event_sequence": record_event["sequence"], "session_id": session_id, "created_at": now_text(), "mode": mode, "event": event_name, "elapsed": snapshot.elapsed, "screen_path": snapshot.relative_path, "screen_hash": snapshot.hash_value.hex, "screen_hash_hex": snapshot.hash_value.hex, "screen_hash_int": snapshot.hash_value.value, "screen_hash_bits": snapshot.hash_value.bits, "screen_captured_at": snapshot.captured_at, "screen_perf": round(snapshot.perf_time, 6), "mouse_action": normalized, "planned_action": normalize_mouse_action(planned_action, snapshot.rect) if planned_action else None, "actual_action": None if failed_action else normalized, "execution_error": str(execution_error) if execution_error else None, "mouse_source": mouse_source, "screen_action_offset_ms": offset_ms, "nearest": [{"id": item["record"].get("id"), "similarity": round(item["similarity"], 4)} for item in batch], "nearest_summary": {"count": len(sims), "max_similarity": max(sims) if sims else 0.0, "avg_similarity": round(sum(sims) / len(sims), 4) if sims else 0.0}, "novelty": before_novelty, "screen_score": before_novelty, "score_version": 1, "score_basis": "nearest_screen_content_live", "score_checked_at": now_text(), "score_neighbors": [{"id": item["record"].get("id"), "similarity": round(item["similarity"], 4)} for item in batch], "before_screen": snapshot.relative_path, "after_screen": after_snapshot.relative_path if after_snapshot else snapshot.relative_path, "before_screen_hash": snapshot.hash_value.hex, "before_screen_score": before_novelty, "after_screen_hash": after_snapshot.hash_value.hex if after_snapshot else snapshot.hash_value.hex, "after_screen_hash_int": after_snapshot.hash_value.value if after_snapshot else snapshot.hash_value.value, "after_screen_hash_bits": after_snapshot.hash_value.bits if after_snapshot else snapshot.hash_value.bits, "after_screen_score": after_novelty, "before_novelty": before_novelty, "after_novelty": after_novelty, "transition_reward": transition_reward, "screen_observation_reward": novelty_reward, "screen_primary_reward": reward_info["screen_primary_reward"], "human_tie_break_reward": reward_info["human_tie_break_reward"], "reward_breakdown": {"screen_novelty": reward_info["screen_novelty"], "screen_reward": reward_info["screen_reward"], "human_similarity": reward_info["human_similarity"], "human_tiebreak": reward_info["human_tiebreak"], "screen_score_delta": reward_info["screen_score_delta"], "basis": reward_info["basis"]}, "reward_sort_key": reward_info["reward_sort_key"], "mouse_action_reward": human_action_reward, "mouse_action_penalty": human_action_penalty, "human_score": human_score, "total_reward": reward, "reward": reward, "novelty_reward": novelty_reward, "human_action_reward": human_action_reward, "human_action_penalty": human_action_penalty, "screen_score_delta": max(0.0, reward), "screen_score_settled": max(0.0, reward), "penalty_delta": max(0.0, -reward), "screen_score_total": screen_score_total, "client_rect": list(snapshot.rect), "failed_action": bool(failed_action), "window_rect_changed": bool(window_rect_changed), "image_checksum": getattr(snapshot, "image_checksum", ""), "after_image_checksum": getattr(after_snapshot, "image_checksum", "") if after_snapshot else getattr(snapshot, "image_checksum", ""), "image_dropped": bool(getattr(snapshot, "image_dropped", False)), "screen_file_expected": not bool(getattr(snapshot, "image_dropped", False)), "capture_latency_ms": capture_latency_ms if capture_latency_ms is not None else getattr(snapshot, "capture_latency_ms", None), "execution_latency_ms": execution_latency_ms, "termination_reason": None, "policy_snapshot": {"hash_size": self.settings.hash_size, "nearest_top_k": self.settings.nearest_top_k, "training_event_wait": self.settings.training_event_wait, "explore_min_rate": self.settings.explore_min_rate, "explore_max_rate": self.settings.explore_max_rate, "action_jitter": self.settings.action_jitter}}
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
                self.mark_snapshot_image_result(snapshot, False)
            else:
                self.mark_snapshot_image_result(snapshot, self.persistence_queue.enqueue_image(analyzer, image, snapshot.path, self.store, priority="low"))
        except Exception as exc:
            self.log_exception("learning_screen_event.save", exc, {"path": str(snapshot.path)})
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
        self.last_learning_event_perf = time.perf_counter()
        self.last_learning_event_hash = None
        with ScreenAnalyzer(config.settings.hash_size) as analyzer:
            self.write_record("learning", session_id, self.capture_snapshot(analyzer, "learning", session_id, start, priority="critical"), None, "mode_start")
            while not stop_event.is_set() and self.is_run_active(token, "learning"):
                if self.should_stop_by_escape():
                    termination_reason = "esc"
                    stop_event.set()
                    break
                now_perf = time.perf_counter()
                check = self.window_manager.check_window(force=True)
                if not check.ok:
                    termination_reason = "window_invalid"
                    stop_event.set()
                    self.ui(lambda r=check.reason: self.status_var.set(f"学习模式结束：雷电模拟器窗口异常：{r}"))
                    break
                if not self.cursor_inside_window():
                    termination_reason = "window_invalid"
                    stop_event.set()
                    self.ui(lambda: self.status_var.set("学习模式结束：鼠标位于雷电模拟器窗口外"))
                    break
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
                        self.mouse_recorder.wait(max(0.0, config.still_seconds - idle_seconds))
                else:
                    stop_event.wait(max(0.0, config.still_seconds - idle_seconds))
            self.write_record("learning", session_id, self.capture_snapshot(analyzer, "learning", session_id, start, priority="critical"), None, "mode_end")
        if self.is_run_active(token, "learning"):
            self.finish_run(token, "学习模式结束", 0.0, reason=termination_reason)
        else:
            self.release_window_and_panel()


    def best_zero_score_recovery_action(self, hash_value):
        batch = self.experience_pool.nearest(hash_value, limit=max(1, self.settings.nearest_top_k)) if self.experience_pool else []
        weighted = []
        for item in batch:
            record = item.get("record", {})
            action = record.get("mouse_action")
            if action:
                reward = safe_float(record.get("sleep_policy_reward", record.get("reward", 0.0)), 0.0)
                similarity = clamp(item.get("similarity", 0.0), 0.0, 1.0)
                weighted.append((max(0.01, reward + 100.0) * max(0.05, similarity), action))
        action = weighted_choice(weighted) if weighted else (self.experience_pool.best_global_action() if self.experience_pool else None)
        if action:
            return self.brain.mutate_action(action, 0.8)
        return self.brain.random_action(0.15)

    def recover_zero_screen_score(self, analyzer, session_id, start, stop_event, zero_score_started_at, current_score, config=None):
        stage_names = ("recapture", "verify_window", "refresh_index", "wait_render", "trusted_history", "bounded_random", "rescore")
        still_seconds = safe_float(getattr(config, "still_seconds", self.settings.generated_action_complete_wait), self.settings.generated_action_complete_wait)
        threshold = max(self.settings.training_event_wait, min(still_seconds, self.settings.generated_action_complete_wait * max(1, self.settings.training_fail_stop_count)))
        if not zero_score_started_at or time.perf_counter() - zero_score_started_at < threshold:
            return current_score, zero_score_started_at, False
        score = current_score
        stage_index = 0
        deadline = start + max(1, safe_int(getattr(config, "training_seconds", DEFAULT_TRAINING_SECONDS), DEFAULT_TRAINING_SECONDS))
        run_guard = lambda: should_stop_run(stop_event, deadline, self.should_stop_by_escape)
        while score <= 0.0 and not run_guard() and self.is_run_active(self.active_session.token if self.active_session else None, "training"):
            stage = stage_names[min(stage_index, len(stage_names) - 1)]
            self.events.publish("training_zero_score_recovery", stage=stage, elapsed=round(time.perf_counter() - zero_score_started_at, 3), screen_score=score)
            if stage == "verify_window":
                check = self.window_manager.check_window(force=True)
                if not check.ok:
                    self.termination_reason = "window_invalid"
                    stop_event.set()
                    break
            elif stage == "refresh_index":
                with self.experience_pool.lock:
                    self.experience_pool.rebuild_index_locked()
                    self.experience_pool.nearest_cache.clear()
            elif stage == "wait_render":
                stop_event.wait(max(self.settings.generated_action_complete_wait, self.settings.training_event_wait))
            snapshot = self.capture_snapshot(analyzer, "training", session_id, start, priority="critical")
            if not snapshot:
                self.termination_reason = "window_invalid"
                stop_event.set()
                break
            score = self.experience_pool.novelty(snapshot.hash_value)[0]
            self.events.publish("training_zero_score_recovery_recheck", stage=stage, screen_score=score)
            if score > 0.0:
                return score, None, True
            if stage in ("trusted_history", "bounded_random"):
                rect = snapshot.rect
                if stage == "trusted_history":
                    action = self.best_zero_score_recovery_action(snapshot.hash_value)
                else:
                    action = self.brain.random_action(0.35)
                if action and self.executor and not run_guard():
                    try:
                        decision = {"reason": "zero_score_recovery", "zero_score_recovery_stage": stage, "zero_score_factor": 1.0}
                        before_snapshot = snapshot
                        actual = self.executor.execute(action, rect, stop_event, lambda: bool(run_guard()))
                        after_snapshot = self.capture_snapshot(analyzer, "training", session_id, start, rect=rect, priority="critical") if not run_guard() else None
                        if actual:
                            record = self.write_record("training", session_id, before_snapshot, actual, "ai_mouse_zero_score_recovery", decision=decision, action_anchor_perf=actual.get("started_perf"), after_snapshot=after_snapshot, planned_action=action, execution_error=actual.get("execution_error"))
                            if record:
                                score = safe_float(record.get("after_screen_score", record.get("screen_score", score)), score)
                        elif before_snapshot:
                            self.write_record("training", session_id, before_snapshot, None, "ai_mouse_zero_score_recovery_failed", decision=decision, planned_action=action, failed_action=True, execution_error="empty_action_result")
                    except Exception as exc:
                        self.log_exception("zero_score_recovery_action", exc, {"stage": stage})
            stage_index += 1
            guarded = run_guard()
            if guarded:
                self.termination_reason = guarded
                stop_event.set()
        return score, zero_score_started_at, score > 0.0

    def training_loop(self, token, stop_event, config):
        session_id = uuid.uuid4().hex
        start = time.perf_counter()
        consecutive_failures = 0
        zero_score_events = 0
        zero_score_started_at = None
        last_training_error = None
        self.termination_reason = "completed"
        service = self.training_service
        with ScreenAnalyzer(config.settings.hash_size) as analyzer:
            self.write_record("training", session_id, self.capture_snapshot(analyzer, "training", session_id, start, priority="critical"), None, "mode_start")
            while not stop_event.is_set() and self.is_run_active(token, "training"):
                if service.should_stop(start, config, stop_event):
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
                zero_score_factor = clamp(zero_score_events / max(1, self.settings.training_fail_stop_count), 0.0, 1.0)
                action, decision = service.decide_action(snapshot, zero_score_factor=zero_score_factor)
                if decision is not None:
                    zero_stage = min(5, 1 + int(zero_score_factor * 5.0)) if zero_score_events > 0 else 0
                    decision["zero_score_streak"] = zero_score_events
                    decision["zero_score_strategy_stage"] = zero_stage
                    decision["zero_score_strategy"] = ["normal", "recapture_validate_window", "refresh_neighbors", "trusted_history_action", "bounded_random_exploration", "alternate_mouse_action"][zero_stage]
                if not action:
                    self.write_record("training", session_id, snapshot, None, "screen_event", decision=decision)
                    continue
                success, record = service.execute_and_record(analyzer, session_id, start, rect, snapshot, action, decision, stop_event)
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
                current_screen_score = safe_float(record.get("after_screen_score", record.get("screen_score", 0.0)), 0.0) if record else 0.0
                if current_screen_score <= 0.0:
                    zero_score_events += 1
                    if zero_score_started_at is None:
                        zero_score_started_at = time.perf_counter()
                    current_screen_score, zero_score_started_at, recovered = self.recover_zero_screen_score(analyzer, session_id, start, stop_event, zero_score_started_at, current_screen_score, config)
                    self.events.publish("training_zero_screen_score", streak=zero_score_events, zero_score_elapsed=round(time.perf_counter() - zero_score_started_at, 3) if zero_score_started_at else 0.0, strategy_stage=min(5, 1 + int(clamp(zero_score_events / max(1, self.settings.training_fail_stop_count), 0.0, 1.0) * 5.0)), screen_score=current_screen_score, recovered=bool(recovered))
                    if recovered:
                        zero_score_events = 0
                else:
                    zero_score_events = 0
                    zero_score_started_at = None
                delay = safe_float(record["mouse_action"].get("duration", 0.0), 0.0) if record and record.get("mouse_action") else 0.0
                deadline = time.perf_counter() + max(self.settings.min_action_delay_seconds, delay)
                while time.perf_counter() < deadline and not stop_event.is_set():
                    if self.should_stop_by_escape():
                        self.termination_reason = "esc"
                        stop_event.set()
                        break
                    stop_event.wait(min(self.settings.generated_action_complete_wait, deadline - time.perf_counter()))
            self.write_record("training", session_id, self.capture_snapshot(analyzer, "training", session_id, start, priority="critical"), None, "mode_end")
        if self.is_run_active(token, "training"):
            final_reason = self.termination_reason or ("esc" if self.should_stop_by_escape() else "user_stop")
            if final_reason == "time_limit":
                session = self.transition("training", "sleep", reason="time_limit", token=token)
                if session:
                    self.update_progress(0.0, force=True)
                    self.ui(self.update_mode_button_states)
                    self.ui(lambda: self.status_var.set("训练模式达到时间上限，进入睡眠模式"))
                    self.mode_thread = threading.Thread(target=self.sleep_loop, args=(session.token, config, session.stop_event, True), daemon=True)
                    self.mode_thread.start()
            else:
                self.finish_run(token, "训练模式已终止" if stop_event.is_set() else "训练模式结束", 0.0, reason=final_reason)
        else:
            self.release_window_and_panel()

    def close(self):
        with self.state_lock:
            self.stop_event.set()
            self.run_token += 1
            self.mode = "idle"
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
            self.persistence_queue.close()
        if self.store:
            self.store.flush_state(force=True)
        mode_thread = self.mode_thread
        if mode_thread and mode_thread.is_alive():
            mode_thread.join(timeout=self.settings.persistence_close_seconds)
        try:
            self.destroy()
        except Exception:
            pass

    def refresh_hardware_state(self):
        event_perf = time.perf_counter()
        full = read_hardware_state()
        if self.hardware_state:
            full["cpu_load"] = safe_float(full.get("cpu_load", self.hardware_state.get("cpu_load", 0.0)), self.hardware_state.get("cpu_load", 0.0))
            full["memory_free_ratio"] = safe_float(full.get("memory_free_ratio", self.hardware_state.get("memory_free_ratio", 0.0)), self.hardware_state.get("memory_free_ratio", 0.0))
        self.hardware_state = full
        self.hardware_last_full_refresh_perf = event_perf
        self.hardware_last_light_refresh_perf = event_perf
        return self.hardware_state

if __name__ == "__main__":
    if "--self-test" in sys.argv:
        run_self_test()
        sys.exit(0)
    prepare_startup_environment()
    app = ControlPanel()
    app.mainloop()
