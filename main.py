import copy
import ctypes
import json
import heapq
import math
import random
import queue
import subprocess
import sys
import os
import shutil
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, fields, replace
from typing import Optional
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
    editable_fields: tuple


AGENT_SPEC = AgentSpec(
    default_ldplayer_path=r"D:\LDPlayer9\dnplayer.exe",
    default_data_path=r"C:\Users\Administrator\Desktop\AAA",
    default_training_seconds=900,
    default_sleep_seconds=1800,
    default_still_seconds=10.0,
    editable_fields=("ldplayer_path", "data_path", "training_seconds", "sleep_seconds", "still_seconds")
)


def fail_and_exit(message):
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("启动失败", f"{message}\n\n点击确定后退出。")
        root.destroy()
    except Exception:
        pass
    sys.exit(1)



if sys.version_info < MIN_PYTHON_VERSION:
    fail_and_exit(f"需要 Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} 或更高版本，当前版本为 {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}。")

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
        fail_and_exit("依赖安装后仍无法导入，已停止重复安装：" + "；".join(f"{name}: {error}" for name, error in IMPORT_ERRORS.items()))
    try:
        pip_check = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True, timeout=30)
        write_startup_install_log([sys.executable, "-m", "pip", "--version"], result=pip_check)
        if pip_check.returncode != 0:
            detail = (pip_check.stderr or pip_check.stdout or "").strip()
            fail_and_exit(f"无法启动 pip，不能自动安装依赖。请检查 Python 环境。\n{detail}")
    except Exception as exc:
        write_startup_install_log([sys.executable, "-m", "pip", "--version"], error=exc)
        fail_and_exit(f"无法启动 pip，不能自动安装依赖。请检查 Python 环境或网络。\n{exc}")
    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--user", *missing]
    try:
        env = dict(os.environ)
        env[install_key] = "1"
        result = subprocess.run(command, capture_output=True, text=True, timeout=240, env=env)
        write_startup_install_log(command, result=result)
    except subprocess.TimeoutExpired as exc:
        write_startup_install_log(command, error=exc)
        detail = (exc.stderr or exc.stdout or "").strip()
        fail_and_exit(f"自动安装依赖超时（240秒）: {' '.join(missing)}\n请检查网络或手动执行: {' '.join(command)}\n{detail}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        install_log_path = Path(globals().get("DEFAULT_DATA_PATH", AGENT_SPEC.default_data_path)) / "startup_install.log"
        fail_and_exit(f"自动安装依赖失败: {' '.join(missing)}\n日志位置: {install_log_path}\n请手动执行: {' '.join(command)}\n{detail}")
    if "pywin32" in missing:
        postinstall = [sys.executable, "-m", "pywin32_postinstall", "-install"]
        try:
            postinstall_result = subprocess.run(postinstall, capture_output=True, text=True, timeout=120)
            write_startup_install_log(postinstall, result=postinstall_result)
            if postinstall_result.returncode != 0:
                detail = (postinstall_result.stderr or postinstall_result.stdout or "").strip()
                fail_and_exit(f"pywin32 安装后配置失败。请手动执行: {' '.join(postinstall)}\n{detail}")
        except Exception as exc:
            write_startup_install_log(postinstall, error=exc)
            fail_and_exit(f"pywin32 安装后配置失败。请手动执行: {' '.join(postinstall)}\n{exc}")
    failed = verify_installed_modules()
    if failed:
        fail_and_exit("自动安装依赖后仍无法导入：" + "；".join(failed))
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

if "--self-test" not in sys.argv:
    bootstrap_dependencies()

DEFAULT_LDPLAYER_PATH = AGENT_SPEC.default_ldplayer_path
DEFAULT_DATA_PATH = AGENT_SPEC.default_data_path
DEFAULT_TRAINING_SECONDS = AGENT_SPEC.default_training_seconds
DEFAULT_SLEEP_SECONDS = AGENT_SPEC.default_sleep_seconds
DEFAULT_STILL_SECONDS = AGENT_SPEC.default_still_seconds
MODE_NAMES = {"idle": "空闲", "starting": "准备中", "learning": "学习模式", "training": "训练模式", "sleep": "睡眠模式", "migration": "数据迁移"}
CONFIG_SCHEMA_VERSION = 1
USER_EDITABLE_STARTUP_FIELDS = ("ldplayer_path", "data_path")
USER_EDITABLE_RUNTIME_FIELDS = ("training_seconds", "sleep_seconds", "still_seconds")
USER_EDITABLE_FIELDS = AGENT_SPEC.editable_fields
TERMINATION_REASONS = ("window_invalid", "rect_changed", "empty_action", "executor_error", "time_limit", "esc", "still_timeout", "user_stop", "migration_error", "completed")
HUMAN_FEATURE_NAMES = ("duration", "direct", "bend", "points", "speed_mean", "speed_variance", "acceleration_change", "pauses", "hover_before", "drag_curvature", "double_click_interval")

RUNTIME_NUMBER_RULES = {
    "hash_size": ("screen_pixels", "pool_count", "capture_ms"),
    "nearest_top_k": ("pool_count", "cpu_count", "life_experience"),
    "nearest_candidate_limit": ("pool_count", "window_instability", "cpu_count", "gpu_count"),
    "hash_prefix_bits": ("pool_count", "screen_pixels"),
    "mouse_still_tick": ("capture_ms", "cpu_load"),
    "training_tick": ("capture_ms", "execution_ms", "cpu_load", "gpu_factor", "window_instability"),
    "sleep_tick": ("cpu_load", "cpu_count", "memory_free_ratio"),
    "key_debounce_seconds": ("cpu_load", "capture_ms"),
    "window_attach_seconds": ("cpu_load", "window_instability"),
    "window_poll_seconds": ("cpu_load", "gpu_factor", "window_instability"),
    "explore_max_rate": ("recent_success", "learning_similarity", "window_instability"),
    "explore_min_rate": ("recent_success", "learning_similarity", "pool_count"),
    "action_jitter": ("recent_success", "window_instability", "screen_pixels"),
    "softmax_temperature": ("pool_count", "recent_success", "life_experience"),
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
    "reward_total_min": ("life_experience", "recent_success"),
    "reward_total_max": ("life_experience", "recent_success"),
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
    "learning_screen_fps": ("capture_ms", "cpu_count", "gpu_count"),
    "learning_screen_similarity_threshold": ("pool_count", "learning_similarity"),
    "training_fail_stop_count": ("recent_success", "window_instability"),
    "ui_refresh_interval": ("capture_ms", "cpu_load", "gpu_factor"),
    "ui_progress_delta": ("window_instability", "cpu_load"),
    "persistence_poll_seconds": ("capture_ms", "persistence_latency", "cpu_load"),
    "persistence_close_seconds": ("execution_ms", "capture_ms", "cpu_load"),
    "async_queue_size": ("cpu_count", "gpu_count", "pool_count"),
    "global_action_heap_limit": ("pool_count", "cpu_count"),
    "local_action_heap_limit": ("pool_count", "window_instability"),
    "action_score_similarity_weight": ("nearest_similarity", "pool_count"),
    "action_score_reward_weight": ("recent_success", "life_experience"),
    "action_score_human_weight": ("recent_success", "learning_similarity"),
    "action_score_novelty_weight": ("window_instability", "pool_count"),
    "motion_score_magnitude_weight": ("recent_success", "screen_pixels"),
    "motion_score_continuity_weight": ("recent_success", "screen_pixels"),
    "human_feature_weights": ("recent_success", "human_feature_count")
}


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
    mouse_still_tick: float = 0.0
    training_tick: float = 0.0
    sleep_tick: float = 0.0
    key_debounce_seconds: float = 0.0
    window_attach_seconds: float = 0.0
    window_poll_seconds: float = 0.0
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
    generated_sleep_tick: float = 0.0
    generated_wait_tick: float = 0.0
    motion_steps_per_second: float = 0.0
    motion_curve_offset_min: float = 0.0
    motion_curve_offset_max: float = 0.0
    motion_first_control_min: float = 0.0
    motion_first_control_max: float = 0.0
    motion_second_control_min: float = 0.0
    motion_second_control_max: float = 0.0
    learning_screen_fps: float = 0.0
    learning_screen_similarity_threshold: float = 0.0
    training_fail_stop_count: int = 0
    ui_refresh_interval: float = 0.0
    ui_progress_delta: float = 0.0
    persistence_poll_seconds: float = 0.0
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

    def build(self, base_settings, rect, pool_count, life, hardware=None):
        capture_ms = self._avg(self.capture_latency_ms, 24.0)
        execution_ms = self._avg(self.execution_latency_ms, 140.0)
        window_instability = self._avg(self.window_change_flags, 0.0)
        recent_success = self._avg(self.outcome_flags, 1.0)
        similarity = self._avg(self.learning_similarity, 0.97)
        cpu_load = 0.0
        if hardware:
            cpu_load = safe_float(hardware.get("cpu_load", 0.0), 0.0)
        return derive_runtime_settings(base_settings=base_settings, rect=rect, pool_count=pool_count, capture_ms=capture_ms, cpu_load=cpu_load, execution_ms=execution_ms, window_instability=window_instability, recent_success=recent_success, life=life, learning_similarity=similarity, hardware=hardware)


@dataclass
class ModeSession:
    token: int
    mode: str
    started_at: float
    deadline: Optional[float]
    stop_event: threading.Event
    termination_reason: Optional[str] = None


@dataclass(frozen=True)
class Config:
    ldplayer_path: Path
    data_path: Path
    training_seconds: int
    sleep_seconds: int
    still_seconds: float
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


def enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def run_self_test():
    settings = derive_runtime_settings()
    assert len(settings.human_feature_weights) == len(HUMAN_FEATURE_NAMES)
    assert sum(settings.human_feature_weights) > 0.0
    factory = RuntimeNumberFactory(read_hardware_state(), (1280, 720), 8, 24.0, 140.0, 24.0, 1.0, 0.0, 0.97, 0.0)
    assert factory.count("training_fail_stop_count", 1, 99) >= 1
    assert "training_fail_stop_count" in factory.audit
    assert RUNTIME_NUMBER_RULES["training_tick"]
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
    assert details["reward_sort_key"][0] == details["screen_primary_reward"]
    assert "human_tie_break_reward" in details
    assert set(USER_EDITABLE_FIELDS) == {"ldplayer_path", "data_path", "training_seconds", "sleep_seconds", "still_seconds"}
    pool = ExperiencePool(settings)
    pool.add({"id": "t1", "mode": "learning", "mouse_action": {"type": "click", "start_rel": [0.5, 0.5]}, "reward": 12, "screen_hash_hex": "f", "screen_hash_bits": 4, "mouse_source": "user"})
    novelty, batch = pool.novelty(a)
    assert novelty >= 0.0
    values = {item.name: getattr(settings, item.name) for item in fields(Settings)}
    values["hash_prefix_bits"] = max(1, settings.hash_prefix_bits - 1)
    changed = Settings(**values)
    pool.apply_settings(changed)
    assert pool.index_settings.hash_prefix_bits == changed.hash_prefix_bits
    assert pool.nearest(a)
    brain = ActionBrain(pool, changed)
    _, decision = brain.choose(a, novelty, batch, 0.0)
    assert isinstance(decision, dict)
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        store = DataStore(folder)
        store.save_settings({"training_seconds": 1, "sleep_seconds": 2, "still_seconds": 3, "forbidden": 4})
        assert "forbidden" not in json.loads(store.settings_file.read_text(encoding="utf-8"))
        store.experience_file.write_text("{bad json}\n" + json.dumps({"id": "ok"}) + "\n", encoding="utf-8")
        loaded = store.load_experience()
        assert len(loaded) == 1 and loaded[0]["id"] == "ok"
        assert (store.root / "experience.bad.jsonl").exists()
    tiny = replace(settings, async_queue_size=1, persistence_poll_seconds=settings.sleep_tick, persistence_close_seconds=settings.sleep_tick)
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
        mouse_still_tick=max(0.001, safe_float(settings.mouse_still_tick, 0.001)),
        training_tick=max(0.001, safe_float(settings.training_tick, 0.001)),
        sleep_tick=max(0.001, safe_float(settings.sleep_tick, 0.001)),
        key_debounce_seconds=max(0.001, safe_float(settings.key_debounce_seconds, 0.001)),
        window_attach_seconds=max(0.001, safe_float(settings.window_attach_seconds, 0.001)),
        window_poll_seconds=max(0.001, safe_float(settings.window_poll_seconds, 0.001)),
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
        generated_sleep_tick=max(0.001, safe_float(settings.generated_sleep_tick, 0.001)),
        generated_wait_tick=max(0.001, safe_float(settings.generated_wait_tick, 0.001)),
        motion_steps_per_second=max(0.001, safe_float(settings.motion_steps_per_second, 0.001)),
        motion_curve_offset_min=clamp(settings.motion_curve_offset_min, 0.0, 1.0),
        motion_curve_offset_max=clamp(settings.motion_curve_offset_max, 0.0, 1.0),
        motion_first_control_min=clamp(settings.motion_first_control_min, 0.0, 1.0),
        motion_first_control_max=clamp(settings.motion_first_control_max, 0.0, 1.0),
        motion_second_control_min=clamp(settings.motion_second_control_min, 0.0, 1.0),
        motion_second_control_max=clamp(settings.motion_second_control_max, 0.0, 1.0),
        learning_screen_fps=max(0.001, safe_float(settings.learning_screen_fps, 0.001)),
        learning_screen_similarity_threshold=clamp(settings.learning_screen_similarity_threshold, 0.0, 1.0),
        training_fail_stop_count=max(1, safe_int(settings.training_fail_stop_count, 1)),
        ui_refresh_interval=max(0.001, safe_float(settings.ui_refresh_interval, 0.001)),
        ui_progress_delta=max(0.001, safe_float(settings.ui_progress_delta, 0.001)),
        persistence_poll_seconds=max(0.001, safe_float(settings.persistence_poll_seconds, 0.001)),
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


class RuntimeNumberFactory:
    def __init__(self, hardware, screen, pool_count, capture_ms, execution_ms, latency, success_rate, window_instability, learning_similarity, life):
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
        self.life = max(0.0, safe_float(life, 0.0))
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
        self.audit = {}

    def source_tuple(self, name):
        return runtime_number_rule(name)

    def remember(self, name, value):
        self.audit[name] = {"value": value, "sources": self.source_tuple(name)}
        return value

    def value(self, name):
        explore_base = 0.35 + (1.0 - self.success_rate) * 0.45 + self.learning_similarity * 0.1
        formulas = {
            "mouse_still_tick": 0.02 * self.capture_factor,
            "training_tick": 0.03 * self.capture_factor * self.cpu_factor * self.timing_factor * (0.85 + self.window_instability * 0.45) / self.gpu_factor,
            "sleep_tick": 0.08 * self.cpu_factor / self.core_factor,
            "key_debounce_seconds": 0.2 * self.cpu_factor,
            "window_attach_seconds": 20.0 + self.cpu_load * 0.6,
            "window_poll_seconds": 0.2 * self.cpu_factor / self.gpu_factor,
            "action_duration_min": (self.capture_ms + self.execution_ms * 0.15) / 1200.0,
            "action_duration_max": (self.capture_ms + self.execution_ms * 0.75) / 320.0,
            "random_click_duration_min": (self.capture_ms + self.execution_ms * 0.08) / 1500.0,
            "random_click_duration_max": (self.capture_ms + self.execution_ms * 0.35) / 700.0,
            "generated_click_hold_max": (self.capture_ms + self.execution_ms * 0.2) / 500.0,
            "ui_refresh_interval": (self.capture_ms / 1000.0) * self.cpu_factor / self.gpu_factor,
            "persistence_poll_seconds": (self.capture_ms + self.latency) / 240.0 * self.cpu_factor,
            "persistence_close_seconds": (self.execution_ms + self.capture_ms) / 90.0 * self.cpu_factor,
            "explore_max_rate": explore_base,
            "explore_min_rate": explore_base * 0.12,
            "action_jitter": 0.008 + (1.0 - self.success_rate) * 0.03 + self.window_instability * 0.012,
            "global_action_probability": 0.45 + 0.15 / max(1.0, self.record_factor),
            "random_action_min": 0.02 + self.window_instability * 0.03,
            "random_action_max": 0.98 - self.window_instability * 0.03,
            "motion_curve_offset_min": 0.04 + self.window_instability * 0.04,
            "motion_curve_offset_max": 0.2 + (1.0 - self.success_rate) * 0.22 + self.window_instability * 0.12,
            "motion_first_control_min": 0.12 + self.window_instability * 0.06,
            "motion_first_control_max": 0.46 + (1.0 - self.success_rate) * 0.12,
            "motion_second_control_min": 0.54 - (1.0 - self.success_rate) * 0.08,
            "motion_second_control_max": 0.9 - self.window_instability * 0.05,
            "learning_screen_similarity_threshold": 0.95 + min(0.04, self.record_factor * 0.0025 + self.learning_similarity * 0.01),
            "ui_progress_delta": 0.08 + self.window_instability * 0.28 + self.cpu_load / 500.0,
            "action_score_similarity_weight": 1.0,
            "action_score_reward_weight": 0.42 + self.success_rate * 0.18,
            "action_score_human_weight": 0.18 + (1.0 - self.success_rate) * 0.14,
            "action_score_novelty_weight": 0.04 + self.window_instability * 0.1,
            "motion_score_magnitude_weight": 0.65 + self.success_rate * 0.15,
            "motion_score_continuity_weight": 0.35 - self.success_rate * 0.15,
            "hash_size": round(8 + self.pixel_factor * 4 + self.record_factor * 0.8),
            "nearest_top_k": round(10 + self.record_factor * 10),
            "nearest_candidate_limit": round((14 + self.record_factor * 7 + self.window_instability * 8) * 64),
            "hash_prefix_bits": round(6 + self.record_factor),
            "human_profile_min_samples": round(12 + self.record_factor * 8),
            "human_profile_max_samples": round(2400 + self.record_factor * 900),
            "human_profile_keep_samples": round(1800 + self.record_factor * 800),
            "ui_width": round(self.width * 0.65),
            "ui_height": round(self.height * 0.72),
            "ui_min_width": round(self.width * 0.42),
            "ui_min_height": round(self.height * 0.45),
            "ui_padding": round(min(self.width, self.height) * 0.012),
            "ui_section_padding": round(min(self.width, self.height) * 0.008),
            "ui_metric_columns": round(self.width / 340.0),
            "ui_metric_min_column_width": round(self.width / 4.8),
            "experience_load_limit": round(12000 + self.record_factor * 8000),
            "training_fail_stop_count": round((1.0 - self.success_rate) * 16 + self.window_instability * 10 + 2),
            "async_queue_size": round((self.cpu_count + self.gpu_count + self.record_factor) * 48),
            "global_action_heap_limit": round((self.record_factor + self.core_factor * 4) * 256),
            "local_action_heap_limit": round((self.record_factor + self.window_instability + 1) * 128),
            "softmax_temperature": 12.0 + self.record_factor * 2.0,
            "reward_total_min": -10000.0 - self.life,
            "reward_total_max": 10000.0 + self.life,
            "motion_steps_per_second": (42.0 + self.width / 24.0) / max(0.6, self.cpu_factor),
            "learning_screen_fps": 1.0 / max(0.05, self.capture_ms / 1000.0) * clamp(self.core_factor * self.gpu_factor, 0.8, 2.4)
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


def derive_runtime_settings(base_settings=None, rect=None, pool_count=0, capture_ms=None, cpu_load=0.0, execution_ms=None, window_instability=0.0, recent_success=1.0, life=0.0, learning_similarity=0.97, hardware=None):
    source = base_settings or Settings()
    width, height = rect_size(rect) if rect else (safe_int(getattr(win32api, "GetSystemMetrics", lambda _: 1920)(0), 1920), safe_int(getattr(win32api, "GetSystemMetrics", lambda _: 1080)(1), 1080))
    hardware = dict(hardware or {})
    hardware["cpu_load"] = cpu_load
    factory = RuntimeNumberFactory(hardware, (width, height), pool_count, capture_ms, execution_ms, capture_ms, recent_success, window_instability, learning_similarity, life)
    values = {item.name: getattr(source, item.name) for item in fields(Settings)}
    values.update({
        "hash_size": factory.count("hash_size", 8, 24),
        "nearest_top_k": factory.count("nearest_top_k", 8, 256),
        "nearest_candidate_limit": factory.count("nearest_candidate_limit", 256, 20000),
        "hash_prefix_bits": factory.count("hash_prefix_bits", 4, 20),
        "mouse_still_tick": round(factory.seconds("mouse_still_tick", 0.01, 0.2), 4),
        "training_tick": round(factory.seconds("training_tick", 0.01, 0.9), 4),
        "sleep_tick": round(factory.seconds("sleep_tick", 0.05, 1.0), 4),
        "key_debounce_seconds": round(factory.seconds("key_debounce_seconds", 0.05, 1.0), 3),
        "window_attach_seconds": round(factory.seconds("window_attach_seconds", 5.0, 120.0), 2),
        "window_poll_seconds": round(factory.seconds("window_poll_seconds", 0.05, 1.0), 3),
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
        "action_duration_min": factory.seconds("action_duration_min", 0.05, 0.35),
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
        "learning_screen_fps": factory.scalar("learning_screen_fps", 0.5, 20.0),
        "learning_screen_similarity_threshold": factory.ratio("learning_screen_similarity_threshold", 0.9, 0.999),
        "training_fail_stop_count": factory.count("training_fail_stop_count", 2, 24),
        "ui_refresh_interval": round(factory.seconds("ui_refresh_interval", 0.01, 0.35), 4),
        "ui_progress_delta": factory.ratio("ui_progress_delta", 0.02, 1.0),
        "persistence_poll_seconds": round(factory.seconds("persistence_poll_seconds", 0.02, 1.0), 4),
        "persistence_close_seconds": round(factory.seconds("persistence_close_seconds", 0.2, 8.0), 4),
        "async_queue_size": factory.count("async_queue_size", 32, 8192),
        "global_action_heap_limit": factory.count("global_action_heap_limit", 128, 8192),
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
        "path_rel": normalize_path(rect, action.get("path", []), start_abs)
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


def hash_similarity(hash_a, hash_b):
    if not hash_a or not hash_b:
        return 0.0
    bits = min(hash_a.bits, hash_b.bits)
    if bits <= 0:
        return 0.0
    a_value = hash_a.value >> max(0, hash_a.bits - bits)
    b_value = hash_b.value >> max(0, hash_b.bits - bits)
    return clamp(1.0 - (a_value ^ b_value).bit_count() / bits, 0.0, 1.0)


def reward_breakdown(novelty, human_score, settings):
    screen_reward = round(clamp(novelty, 0.0, 100.0), 2)
    human_delta = round((clamp(human_score, 0.0, 100.0) - settings.score_default) * 2.0, 2)
    reward_span = max(1.0, abs(safe_float(settings.reward_total_max, 100.0) - safe_float(settings.reward_total_min, 0.0)))
    tie_breaker = reward_span / max(1.0, settings.global_action_heap_limit * settings.local_action_heap_limit)
    human_tiebreak = clamp(human_delta / 100.0, -1.0, 1.0) * tie_breaker
    total = round(clamp(screen_reward + human_tiebreak, settings.reward_total_min, settings.reward_total_max), 4)
    return {"screen_primary_reward": screen_reward, "human_tie_break_reward": round(human_tiebreak, 6), "mouse_action_delta": human_delta, "total_reward": total, "reward_sort_key": [screen_reward, round(human_tiebreak, 6)]}


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

class AppConfigStore:
    def __init__(self):
        base = Path(os.environ.get("APPDATA") or (Path.home() / ".config"))
        self.root = base / "AGI"
        self.settings_file = self.root / "startup.json"
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    def load_settings(self):
        if not self.settings_file.exists():
            self.save_settings({"ldplayer_path": AGENT_SPEC.default_ldplayer_path, "data_path": AGENT_SPEC.default_data_path})
        try:
            with self.settings_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict):
                return {}
            return {name: data[name] for name in USER_EDITABLE_STARTUP_FIELDS if name in data}
        except Exception:
            return {}

    def save_settings(self, settings):
        source = {name: settings.get(name) for name in USER_EDITABLE_STARTUP_FIELDS} if isinstance(settings, dict) else {}
        payload = {"schema_version": CONFIG_SCHEMA_VERSION, "ldplayer_path": str(source.get("ldplayer_path") or AGENT_SPEC.default_ldplayer_path), "data_path": str(source.get("data_path") or AGENT_SPEC.default_data_path)}
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.settings_file.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            temporary.replace(self.settings_file)


class DataStore:
    def __init__(self, root):
        self.root = Path(root)
        self.screen_dir = self.root / "screens"
        self.experience_file = self.root / "experience.jsonl"
        self.state_file = self.root / "state.json"
        self.settings_file = self.root / "settings.json"
        self.error_file = self.root / "errors.jsonl"
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.screen_dir.mkdir(parents=True, exist_ok=True)
        self.state = self.load_state()
        self.pending_state_writes = 0
        self.last_state_save_perf = time.perf_counter()

    @property
    def life(self):
        return safe_float(self.state.get("life_experience", 0.0), 0.0)

    def load_state(self):
        if not self.state_file.exists():
            return {"life_experience": 0.0, "penalty": 0.0}
        try:
            with self.state_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return {"life_experience": safe_float(data.get("life_experience", 0.0), 0.0), "penalty": safe_float(data.get("penalty", 0.0), 0.0)}
        except Exception:
            return {"life_experience": 0.0, "penalty": 0.0}

    def save_state(self):
        with self.lock:
            temporary = self.state_file.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(self.state, file, ensure_ascii=False, indent=2)
            temporary.replace(self.state_file)

    def add_life_experience(self, value):
        with self.lock:
            delta = safe_float(value, 0.0)
            current = safe_float(self.state.get("life_experience", 0.0), 0.0)
            penalty = safe_float(self.state.get("penalty", 0.0), 0.0)
            if delta >= 0.0:
                self.state["life_experience"] = round(current + delta, 2)
            else:
                self.state["life_experience"] = round(current, 2)
                self.state["penalty"] = round(penalty + abs(delta), 2)
            return self.state["life_experience"]

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

    def append_experience(self, record):
        with self.lock:
            with self.experience_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_settings(self):
        if not self.settings_file.exists():
            return {}
        try:
            with self.settings_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict):
                return {}
            return {name: data[name] for name in USER_EDITABLE_RUNTIME_FIELDS if name in data}
        except Exception:
            return {}

    def save_settings(self, settings):
        source = {name: settings.get(name) for name in USER_EDITABLE_RUNTIME_FIELDS} if isinstance(settings, dict) else {}
        payload = {"schema_version": CONFIG_SCHEMA_VERSION, "training_seconds": max(1, safe_int(source.get("training_seconds", AGENT_SPEC.default_training_seconds), AGENT_SPEC.default_training_seconds)), "sleep_seconds": max(1, safe_int(source.get("sleep_seconds", AGENT_SPEC.default_sleep_seconds), AGENT_SPEC.default_sleep_seconds)), "still_seconds": max(0.1, safe_float(source.get("still_seconds", AGENT_SPEC.default_still_seconds), AGENT_SPEC.default_still_seconds))}
        with self.lock:
            temporary = self.settings_file.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            temporary.replace(self.settings_file)

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
                    reward = safe_float(parsed.get("reward", parsed.get("total_reward", 0.0)), 0.0)
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
                time.sleep(self.settings.window_poll_seconds)
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
            if not force and cache.get("rect") == rect and now_perf - cache.get("basic_perf", 0.0) < self.settings.window_poll_seconds:
                return cache.get("check", WindowCheck(bool(cache.get("ok", False)), cache.get("reason", "cached"), rect))
            cache["rect"] = rect
            cache["basic_perf"] = now_perf
            if force or now_perf - cache.get("visibility_perf", 0.0) >= clamp(self.settings.window_poll_seconds, 0.2, 0.5):
                if not win32gui.IsWindowVisible(hwnd):
                    result = WindowCheck(False, "invisible", rect)
                    cache.update({"ok": result.ok, "reason": result.reason, "check": result})
                    return result
                if win32gui.IsIconic(hwnd):
                    result = WindowCheck(False, "minimized", rect)
                    cache.update({"ok": result.ok, "reason": result.reason, "check": result})
                    return result
                left, top, right, bottom = rect
                screen_w = max(1, safe_int(win32api.GetSystemMetrics(0), 1))
                screen_h = max(1, safe_int(win32api.GetSystemMetrics(1), 1))
                if left < 0 or top < 0 or right > screen_w or bottom > screen_h:
                    result = WindowCheck(False, "out_of_screen", rect)
                    cache.update({"ok": result.ok, "reason": result.reason, "check": result})
                    return result
                cache["visibility_perf"] = now_perf
            if not force and now_perf - cache.get("occlusion_perf", 0.0) < clamp(self.settings.window_poll_seconds * 2.0, 0.5, 1.0):
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
        if str(path).lower().endswith(".png"):
            compression = 1 if priority == "critical" else 3
            if settings is not None and settings.training_tick > settings.ui_refresh_interval:
                compression = 1
            save_kwargs = {"optimize": priority == "critical", "compress_level": int(clamp(compression, 0, 9))}
        image.save(path, **save_kwargs)

    def fingerprint(self, image):
        small = image.convert("L").resize((self.hash_size, self.hash_size), self.resample)
        pixels = list(small.getdata())
        average = sum(pixels) / max(1, len(pixels))
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


class ExperiencePool:
    INDEX_SETTING_NAMES = ("hash_prefix_bits", "nearest_candidate_limit")

    def __init__(self, settings, records=None):
        self.settings = settings
        self.index_settings = settings
        self.records = []
        self.hashes = []
        self.index = defaultdict(list)
        self.sorted_prefixes = []
        self.profile = HumanProfile(settings)
        self.lock = threading.RLock()
        self.action_cache = []
        self.prefix_neighbor_cache = OrderedDict()
        self.global_action_heap = []
        self.nearest_cache = OrderedDict()
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
            if rebuild:
                self.index_settings = settings
                self.rebuild_index_locked()
            self.nearest_cache.clear()

    def rebuild_index_locked(self):
        self.index = defaultdict(list)
        self.sorted_prefixes = []
        self.prefix_neighbor_cache = OrderedDict()
        self.index_version += 1
        for index, hash_value in enumerate(self.hashes):
            if hash_value:
                prefix = self._prefix(hash_value)
                bucket = self.index[prefix]
                if not bucket:
                    self.sorted_prefixes.append(prefix)
                bucket.append(index)

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

    def nearest(self, hash_value):
        if not hash_value:
            return []
        with self.lock:
            top_k = max(1, self.settings.nearest_top_k)
            cache_key = (self.index_version, hash_value.bits, hash_value.hex, self.index_settings.hash_prefix_bits, self.index_settings.nearest_candidate_limit, self.settings.nearest_top_k)
            cached = self.nearest_cache.get(cache_key)
            if cached is not None:
                self.nearest_cache.move_to_end(cache_key)
                return [copy.deepcopy(item) for item in cached]
            candidate_indexes = self.candidate_indices(hash_value)
            if len(self.records) > self.index_settings.nearest_candidate_limit:
                recent_limit = min(len(self.records), max(top_k, self.index_settings.nearest_candidate_limit // 4))
                candidate_indexes = list(dict.fromkeys(candidate_indexes + list(range(len(self.records) - recent_limit, len(self.records))) + [index for _, index in heapq.nlargest(min(len(self.global_action_heap), top_k), self.global_action_heap)]))
            snapshot = [(index, self.hashes[index], self.records[index]) for index in candidate_indexes if self.hashes[index]]
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

    def human_score(self, action):
        return self.profile.score(action)


class ActionBrain:
    def __init__(self, pool, settings):
        self.pool = pool
        self.settings = settings
        self.last_action = None
        self.last_decision = None

    def exploration_rate(self, novelty, life):
        action_count = len(self.pool.action_records())
        count_factor = 1.0 / math.sqrt(max(1.0, action_count))
        novelty_factor = clamp(novelty / 100.0, 0.0, 1.0)
        life_factor = 1.0 / math.sqrt(max(1.0, 1.0 + life / 200.0))
        rate = (0.12 + 0.28 * count_factor + 0.18 * novelty_factor) * life_factor
        return round(clamp(rate, self.settings.explore_min_rate, self.settings.explore_max_rate), 4)

    def score_candidate(self, item):
        record = item["record"]
        similarity = clamp(item.get("similarity", 0.0), 0.0, 1.0)
        reward = max(-80.0, safe_float(record.get("reward", 0.0), 0.0))
        human_score = clamp(record.get("human_score", 50.0), 0.0, 100.0)
        novelty = clamp(record.get("novelty", 50.0), 0.0, 100.0)
        source_bonus = 4.0 if record.get("mode") == "learning" else 0.0
        return similarity * 100.0 * self.settings.action_score_similarity_weight + reward * self.settings.action_score_reward_weight + human_score * self.settings.action_score_human_weight + novelty * self.settings.action_score_novelty_weight + source_bonus

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

    def fallback_action(self):
        learned = self.pool.best_global_action()
        if learned and random.random() < self.settings.global_action_probability:
            return self.mutate_action(learned, 1.8), "global_experience"
        return None, "observe_only"

    def choose(self, hash_value, novelty, batch, life):
        rate = self.exploration_rate(novelty, life)
        usable = []
        for item in batch:
            action = item["record"].get("mouse_action")
            if not action or safe_float(item["record"].get("reward", 0.0), 0.0) < -60.0:
                continue
            score = self.score_candidate(item)
            usable.append((math.exp(clamp(score, -100.0, 240.0) / self.settings.softmax_temperature), {"item": item, "score": score, "action": action}))
        if random.random() < rate or not usable:
            action, reason = self.fallback_action()
            decision = {"reason": reason, "exploration_rate": rate, "candidate_count": len(usable), "confidence": 0.0, "nearest_similarity": round(batch[0]["similarity"], 4) if batch else 0.0}
        else:
            chosen = weighted_choice(usable)
            item = chosen["item"]
            confidence = clamp(item.get("similarity", 0.0) * 0.65 + clamp(chosen.get("score", 0.0), 0.0, 200.0) / 200.0 * 0.35, 0.0, 1.0)
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
        self.current = None
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
            self.wake.clear()

    def active(self):
        return self.get_mode() == "learning"

    def capture_event(self, kind, x, y, extra=None, allow_current=False):
        if not self.active():
            return None
        rect = self.get_rect()
        if not rect:
            return None
        inside = point_inside(rect, x, y)
        if not inside and not allow_current:
            return None
        self.on_activity()
        event = {"type": kind, "t": time.perf_counter(), "x": int(x), "y": int(y), "inside": bool(inside)}
        if extra:
            event.update(extra)
        return event

    def push_start_marker(self, action_id, event, action_type):
        self.start_markers.append({"action_id": action_id, "action_type": action_type, "perf_time": event["t"], "created_at": now_text(), "x": event["x"], "y": event["y"]})
        self.wake.set()

    def on_move(self, x, y):
        with self.lock:
            has_current = bool(self.current)
        event = self.capture_event("move", x, y, allow_current=has_current)
        if not event:
            return
        with self.lock:
            if self.current:
                self.current["path"].append(event)

    def on_scroll(self, x, y, dx, dy):
        event = self.capture_event("scroll", x, y, {"dx": int(dx), "dy": int(dy)})
        if not event:
            return
        action_id = uuid.uuid4().hex
        action = {"action_id": action_id, "type": "scroll", "button": "scroll", "source": "user", "started_at": now_text(), "ended_at": now_text(), "started_perf": event["t"], "ended_perf": event["t"], "duration": 0.0, "start_abs": [int(x), int(y)], "end_abs": [int(x), int(y)], "path": [event], "scroll": [int(dx), int(dy)]}
        with self.lock:
            self.push_start_marker(action_id, event, "scroll")
            self.actions.append(action)
            self.wake.set()

    def on_click(self, x, y, button, pressed):
        with self.lock:
            has_current = bool(self.current)
        event = self.capture_event("press" if pressed else "release", x, y, {"button": str(button)}, allow_current=(not pressed and has_current))
        if not event:
            return
        with self.lock:
            if pressed:
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
            items = list(self.actions)
            self.actions.clear()
            if not self.start_markers:
                self.wake.clear()
            return items

    def wait(self, timeout):
        self.wake.wait(timeout)


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
            time.sleep(min(self.settings.generated_sleep_tick, max(0.0, deadline - time.perf_counter())))

    def move_smooth(self, start, end, duration, stop_event, should_stop):
        points = self.smooth_points(start, end, duration)
        actual = []
        delay = duration / max(1, len(points) - 1) if duration > 0.0 else 0.0
        base_t = time.perf_counter()
        for point in points:
            if stop_event.is_set() or should_stop():
                stop_event.set()
                break
            self.controller.position = point
            actual.append({"x": int(point[0]), "y": int(point[1]), "t": time.perf_counter() - base_t})
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
        actual_path = self.move_smooth(current, start_abs, approach_duration, stop_event, should_stop)
        if stop_event.is_set():
            return None
        started_at = now_text()
        action_t = time.perf_counter()
        pressed = False
        try:
            if action_type == "drag":
                self.controller.press(button)
                pressed = True
                actual_path.extend(self.move_smooth(start_abs, end_abs, main_duration, stop_event, should_stop))
            elif action_type == "scroll":
                scroll = action.get("scroll") or [0, 0]
                self.controller.scroll(int(scroll[0]), int(scroll[1]))
                actual_path.append({"x": int(start_abs[0]), "y": int(start_abs[1]), "t": time.perf_counter() - action_t})
            else:
                self.controller.press(button)
                pressed = True
                hold_floor = min(self.settings.random_click_duration_min, self.settings.random_click_duration_max)
                hold_ceiling = max(self.settings.random_click_duration_min, self.settings.random_click_duration_max)
                hold_duration = clamp(main_duration, hold_floor, hold_ceiling if hold_ceiling > 0.0 else self.settings.generated_click_hold_max)
                self.stoppable_sleep(clamp(hold_duration, 0.0, self.settings.generated_click_hold_max), stop_event, should_stop)
                actual_path.append({"x": int(end_abs[0]), "y": int(end_abs[1]), "t": time.perf_counter() - action_t})
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

    def poll_pressed(self):
        if not win32api:
            return False
        try:
            if win32api.GetAsyncKeyState(0x1B) & 0x8000:
                return self.trigger()
        except Exception:
            return False
        return False


class AsyncPersistenceQueue:
    def __init__(self, settings_or_maxsize):
        if isinstance(settings_or_maxsize, Settings):
            self.settings = settings_or_maxsize
            maxsize = settings_or_maxsize.async_queue_size
            self.poll_seconds = settings_or_maxsize.persistence_poll_seconds
            self.close_seconds = settings_or_maxsize.persistence_close_seconds
        else:
            self.settings = derive_runtime_settings()
            maxsize = settings_or_maxsize
            self.poll_seconds = self.settings.persistence_poll_seconds
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
        while not self.stop_event.is_set() or not self.jobs.empty():
            try:
                job = self.jobs.get(timeout=self.poll_seconds)
            except queue.Empty:
                continue
            try:
                if job.get("type") == "image" and job.get("image") is not None:
                    job["analyzer"].save_image(job["image"], job["path"], priority=job.get("priority", "normal"), settings=self.settings)
                elif job.get("type") == "record":
                    store = job["store"]
                    store.append_experience(job["record"])
                    store.flush_state(min_interval=self.settings.persistence_close_seconds, max_pending=max(1, self.settings.async_queue_size // max(1, self.settings.global_action_heap_limit // self.settings.local_action_heap_limit)))
            except Exception as exc:
                store = job.get("store")
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
        deadline = time.perf_counter() + self.close_seconds
        while self.jobs.unfinished_tasks and time.perf_counter() < deadline:
            time.sleep(min(self.poll_seconds, max(0.001, deadline - time.perf_counter())))
        if self.jobs.unfinished_tasks:
            self.drop_pending_images()
        while self.jobs.unfinished_tasks and time.perf_counter() < deadline + self.close_seconds:
            time.sleep(min(self.poll_seconds, max(0.001, deadline + self.close_seconds - time.perf_counter())))
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

    def update(self, novelty, human_score, screen_reward, action_reward, reward, life, decision=None):
        return self.panel.update_metrics(novelty, human_score, screen_reward, action_reward, reward, life, decision=decision)


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

    def prepare_tick(self, rect):
        panel = self.panel
        pool_count = panel.experience_pool.count() if panel.experience_pool else 0
        life_value = panel.store.life if panel.store else 0.0
        panel.hardware_state = panel.refresh_hardware_state()
        settings = panel.adaptive_policy.build(panel.settings, rect, pool_count, life_value, hardware=panel.hardware_state)
        panel.apply_runtime_settings(settings)
        return settings

    def observe_screen(self, analyzer, session_id, start, rect):
        return self.panel.capture_snapshot(analyzer, "training", session_id, start, rect=rect)

    def decide_action(self, snapshot):
        panel = self.panel
        novelty, batch = panel.experience_pool.novelty(snapshot.hash_value)
        life = panel.store.life if panel.store else 0.0
        return panel.brain.choose(snapshot.hash_value, novelty, batch, life)

    def should_stop(self, start, config, stop_event):
        panel = self.panel
        if panel.should_stop_by_escape():
            stop_event.set()
            return True
        elapsed = time.perf_counter() - start
        if elapsed >= config.training_seconds:
            return True
        if not panel.window_manager.window_ok():
            stop_event.set()
            panel.ui(lambda: panel.status_var.set("训练模式结束：雷电模拟器窗口异常"))
            return True
        panel.update_progress(clamp((config.training_seconds - elapsed) / config.training_seconds * 100.0, 0.0, 100.0))
        panel.ui(lambda: panel.progress_label_var.set("模式进度"))
        return False

    def execute_and_record(self, analyzer, session_id, start, rect, snapshot, action, decision, stop_event):
        panel = self.panel
        if action.get("end_rel") is None:
            action["end_rel"] = action.get("start_rel", [0.5, 0.5])
        latest_rect = panel.current_rect()
        if not latest_rect or latest_rect != rect:
            panel.write_record("training", session_id, snapshot, None, "ai_mouse_failed", decision=decision, planned_action=action, failed_action=True, window_rect_changed=True, execution_error="window_rect_changed")
            panel.adaptive_policy.observe_execution(success=False)
            return False, None
        actual = panel.executor.execute(action, rect, stop_event, panel.should_stop_by_escape)
        if not actual:
            panel.log_exception("training.execute", RuntimeError("empty_action_result"))
            panel.write_record("training", session_id, snapshot, None, "ai_mouse_failed", decision=decision, planned_action=action, failed_action=True, execution_error="empty_action_result")
            panel.adaptive_policy.observe_execution(success=False)
            return False, None
        if actual.get("execution_error"):
            panel.log_exception("training.execute", RuntimeError(actual.get("execution_error")), {"detail": actual, "decision": decision})
            panel.write_record("training", session_id, snapshot, None, "ai_mouse_failed", decision=decision, planned_action=action, failed_action=True, execution_error=actual.get("execution_error"))
            panel.adaptive_policy.observe_execution(success=False)
            return False, None
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
        self.adaptive_policy = AdaptivePolicy()
        initial_capture_ms = self.measure_capture_latency()
        self.hardware_state = read_hardware_state()
        self.settings = derive_runtime_settings(rect=self.screen_rect(), pool_count=0, capture_ms=initial_capture_ms, cpu_load=safe_float(self.hardware_state.get("cpu_load", 0.0), 0.0), hardware=self.hardware_state)
        self.adaptive_policy.observe_capture(initial_capture_ms)
        self.progress_value = 0.0
        self.last_progress_update_perf = 0.0
        self.last_metrics_update_perf = 0.0
        self.last_status_update_perf = 0.0
        self.last_metric_payload = None
        self.persistence_queue = AsyncPersistenceQueue(self.settings)
        self.persistence_paused = threading.Event()
        self.runtime_context = RuntimeContext(self)
        self.mode_controller = ModeController(self)
        self.migration_service = MigrationService(self)
        self.metrics_presenter = MetricsPresenter(self)
        self.learning_service = LearningService(self)
        self.training_service = TrainingService(self)
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
        self.escape_monitor = EscapeMonitor(self.stop_current_mode, self.settings.key_debounce_seconds)
        self.ldplayer_var = tk.StringVar(value=DEFAULT_LDPLAYER_PATH)
        self.data_var = tk.StringVar(value=DEFAULT_DATA_PATH)
        self.training_seconds_var = tk.StringVar(value=str(DEFAULT_TRAINING_SECONDS))
        self.sleep_seconds_var = tk.StringVar(value=str(DEFAULT_SLEEP_SECONDS))
        self.still_seconds_var = tk.StringVar(value=str(DEFAULT_STILL_SECONDS))
        self.mode_var = tk.StringVar(value=MODE_NAMES["idle"])
        self.status_var = tk.StringVar(value="等待开始")
        self.life_var = tk.StringVar(value="0")
        self.reward_var = tk.StringVar(value="0")
        self.screen_reward_var = tk.StringVar(value="0")
        self.action_reward_var = tk.StringVar(value="0")
        self.novelty_var = tk.StringVar(value="0%")
        self.human_var = tk.StringVar(value="0%")
        self.ai_var = tk.StringVar(value="未决策")
        self.pool_var = tk.StringVar(value="0")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="0%")
        self.last_learning_tick_perf = 0.0
        self.last_learning_tick_hash = None
        self.hardware_last_full_refresh_perf = 0.0
        self.hardware_last_light_refresh_perf = 0.0
        self.progress_label_var = tk.StringVar(value="进度")
        self.metric_items = []
        self.hint_label = None
        self.metrics_frame = None
        self.app_config_store = AppConfigStore()
        self.build_ui()
        self.load_persistent_settings()
        self.bind("<Configure>", self.on_window_resize)
        self.bind("<Escape>", lambda event: self.stop_current_mode())
        if self.required_import_error():
            self.after(200, self.show_import_error)
        else:
            self.mouse_recorder = MouseRecorder(self.current_mode, self.current_rect, self.mark_learning_activity)
            self.mouse_recorder.start()
        self.escape_monitor.start()
        if not pynput_keyboard:
            self.status_var.set("全局键盘监听不可用，已启用 Windows ESC 轮询兜底")
        self.protocol("WM_DELETE_WINDOW", self.close)

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
        canvas = tk.Canvas(root, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        container = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=container, anchor="nw")
        container.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(canvas_window, width=event.width))
        self.scroll_canvas = canvas
        container.bind("<Enter>", self.bind_mousewheel)
        container.bind("<Leave>", self.unbind_mousewheel)
        ttk.Label(container, text="雷电模拟器学习训练控制面板", style="Title.TLabel").pack(anchor="w")
        path_frame = ttk.LabelFrame(container, text="路径与时间", padding=self.settings.ui_section_padding)
        path_frame.pack(fill="x", pady=(16, 10))
        path_frame.columnconfigure(1, weight=1)
        ttk.Label(path_frame, text="雷电模拟器").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(path_frame, textvariable=self.ldplayer_var, justify="right", state="readonly").grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(path_frame, text="修改", command=self.choose_ldplayer).grid(row=0, column=2, padx=(8, 0), pady=6)
        ttk.Label(path_frame, text="数据存储").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(path_frame, textvariable=self.data_var, justify="right", state="readonly").grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Button(path_frame, text="修改", command=self.choose_data).grid(row=1, column=2, padx=(8, 0), pady=6)
        time_frame = ttk.Frame(path_frame)
        time_frame.grid(row=2, column=1, columnspan=2, sticky="w", pady=6)
        ttk.Label(path_frame, text="时间设置").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=6)
        for label, variable in (("训练/秒", self.training_seconds_var), ("睡眠/秒", self.sleep_seconds_var), ("静止/秒", self.still_seconds_var)):
            ttk.Label(time_frame, text=label).pack(side="left")
            ttk.Entry(time_frame, textvariable=variable, width=10).pack(side="left", padx=(6, 16))
        button_frame = ttk.Frame(container)
        button_frame.pack(fill="x", pady=(4, 12))
        self.button_frame = button_frame
        self.control_buttons = [
            ttk.Button(button_frame, text="学习模式", command=self.learning_mode),
            ttk.Button(button_frame, text="训练模式", command=self.training_mode),
            ttk.Button(button_frame, text="睡眠模式", command=self.sleep_mode),
            ttk.Button(button_frame, text="终止当前模式", command=self.stop_current_mode),
            ttk.Button(button_frame, text="退出", command=self.close)
        ]
        self.reflow_buttons()
        status_frame = ttk.LabelFrame(container, text="状态", padding=self.settings.ui_section_padding)
        status_frame.pack(fill="both", expand=True)
        self.metrics_frame = ttk.Frame(status_frame)
        self.metrics_frame.grid(row=0, column=0, sticky="nsew")
        status_frame.columnconfigure(0, weight=1)
        metrics = [("当前模式", self.mode_var), ("人生阅历", self.life_var), ("经验条数", self.pool_var), ("新颖度", self.novelty_var), ("真人评分", self.human_var), ("画面奖励", self.screen_reward_var), ("鼠标奖惩", self.action_reward_var), ("本次奖励", self.reward_var), ("AI决策", self.ai_var)]
        for title, variable in metrics:
            self.metric_items.append(self.create_metric(self.metrics_frame, title, variable))
        self.reflow_metrics()
        ttk.Label(status_frame, text="快捷键", style="CardTitle.TLabel").grid(row=1, column=0, sticky="w", pady=(18, 6), padx=(0, 12))
        hint = "ESC 终止当前学习、训练或睡眠。学习模式下鼠标静止超时会自动结束；鼠标移出客户区不会终止模式，客户区外的新动作会被忽略。截图与坐标均使用雷电客户区。"
        self.hint_label = ttk.Label(status_frame, text=hint, wraplength=max(320, self.settings.ui_width - 120), style="Hint.TLabel")
        self.hint_label.grid(row=2, column=0, sticky="ew", pady=6)
        progress_frame = ttk.LabelFrame(container, text=self.progress_label_var.get(), padding=self.settings.ui_section_padding)
        self.progress_label_var.trace_add("write", lambda *args: progress_frame.configure(text=self.progress_label_var.get()))
        progress_frame.pack(fill="x", pady=(12, 0))
        progress_frame.columnconfigure(0, weight=1)
        ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100).grid(row=0, column=0, sticky="ew")
        ttk.Label(progress_frame, textvariable=self.progress_text_var, width=8, anchor="e").grid(row=0, column=1, sticky="e", padx=(10, 0))
        ttk.Label(container, textvariable=self.status_var).pack(anchor="w", pady=(10, 0))

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

    def ui(self, func):
        try:
            self.after(0, func)
        except Exception as exc:
            self.log_exception("ui.dispatch", exc)

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

    def required_import_error(self):
        return {name: IMPORT_ERRORS[name] for name in tuple(REQUIRED_MODULES) if name in IMPORT_ERRORS}

    def show_import_error(self):
        missing = self.required_import_error()
        lines = [f"{name}: {error}" for name, error in missing.items()]
        messagebox.showerror("依赖异常", "依赖自动安装或加载失败。\n\n当前错误：\n" + "\n".join(lines))
        self.status_var.set("依赖缺失")

    def choose_ldplayer(self):
        path = filedialog.askopenfilename(title="选择 dnplayer.exe", filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if path:
            self.ldplayer_var.set(path)
            self.save_persistent_settings()

    def choose_data(self):
        path = filedialog.askdirectory(title="选择数据存储目录")
        if not path:
            return
        old_path = Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)
        new_path = Path(path)
        if old_path == new_path:
            self.data_var.set(str(new_path))
            self.save_persistent_settings()
            return
        token, stop_event = self.begin_run("migration")
        if not token:
            self.status_var.set("请先终止当前模式，或等待当前模式结束")
            return
        self.status_var.set("正在迁移数据")
        self.update_progress(0.0)
        values = {"ldplayer_path": self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH, "training_seconds": max(1, safe_int(self.training_seconds_var.get(), DEFAULT_TRAINING_SECONDS)), "sleep_seconds": max(1, safe_int(self.sleep_seconds_var.get(), DEFAULT_SLEEP_SECONDS)), "still_seconds": max(0.1, safe_float(self.still_seconds_var.get(), DEFAULT_STILL_SECONDS))}
        self.mode_thread = threading.Thread(target=self.migration_service.run, args=(token, old_path, new_path, stop_event, values), daemon=True)
        self.mode_thread.start()

    def migration_items(self, old_path):
        root = Path(old_path)
        items = []
        for name in ("screens", "experience.jsonl", "state.json", "settings.json", "errors.jsonl"):
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
                self.update_progress(clamp(copied / max(1, total) * 100.0, 0.0, 99.0))
        try:
            shutil.copystat(source, target)
        except Exception:
            pass
        return copied, True

    def migration_known_names(self):
        return {"screens", "experience.jsonl", "state.json", "settings.json", "errors.jsonl"}

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
        lines = 0
        experience = root / "experience.jsonl"
        if experience.exists():
            try:
                with experience.open("r", encoding="utf-8") as file:
                    for _ in file:
                        lines += 1
            except Exception:
                pass
        return {"screens": screen_count, "experience_lines": lines, "settings": (root / "settings.json").exists(), "state": (root / "state.json").exists()}

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
                DataStore(temp_root).save_settings({"training_seconds": values["training_seconds"], "sleep_seconds": values["sleep_seconds"], "still_seconds": values["still_seconds"]})
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
                self.experience_pool = ExperiencePool(self.settings, self.store.load_experience(self.settings.experience_load_limit))
                self.brain = ActionBrain(self.experience_pool, self.settings)
                self.update_progress(100.0)
                self.finish_run(token, reason, 100.0, release=False, reason=reason)
            elif self.is_run_active(token, "migration"):
                self.finish_run(token, reason, self.progress_value, release=False, reason=reason)
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
            DataStore(Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)).save_settings({"training_seconds": DEFAULT_TRAINING_SECONDS, "sleep_seconds": DEFAULT_SLEEP_SECONDS, "still_seconds": DEFAULT_STILL_SECONDS})
        self.training_seconds_var.set(str(max(1, safe_int(data_settings.get("training_seconds", self.training_seconds_var.get()), DEFAULT_TRAINING_SECONDS))))
        self.sleep_seconds_var.set(str(max(1, safe_int(data_settings.get("sleep_seconds", self.sleep_seconds_var.get()), DEFAULT_SLEEP_SECONDS))))
        self.still_seconds_var.set(str(max(0.1, safe_float(data_settings.get("still_seconds", self.still_seconds_var.get()), DEFAULT_STILL_SECONDS))))

    def save_persistent_settings(self):
        data_path = Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)
        self.app_config_store.save_settings({"ldplayer_path": self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH, "data_path": str(data_path)})
        DataStore(data_path).save_settings({"training_seconds": max(1, safe_int(self.training_seconds_var.get(), DEFAULT_TRAINING_SECONDS)), "sleep_seconds": max(1, safe_int(self.sleep_seconds_var.get(), DEFAULT_SLEEP_SECONDS)), "still_seconds": max(0.1, safe_float(self.still_seconds_var.get(), DEFAULT_STILL_SECONDS))})

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
        training_seconds = max(1, safe_int(self.training_seconds_var.get(), DEFAULT_TRAINING_SECONDS))
        sleep_seconds = max(1, safe_int(self.sleep_seconds_var.get(), DEFAULT_SLEEP_SECONDS))
        still_seconds = max(0.1, safe_float(self.still_seconds_var.get(), DEFAULT_STILL_SECONDS))
        self.training_seconds_var.set(str(training_seconds))
        self.sleep_seconds_var.set(str(sleep_seconds))
        self.still_seconds_var.set(str(still_seconds))
        self.save_persistent_settings()
        data_path = Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)
        self.hardware_state = read_hardware_state()
        self.hardware_last_full_refresh_perf = time.perf_counter()
        self.hardware_last_light_refresh_perf = self.hardware_last_full_refresh_perf
        cpu_load = safe_float(self.hardware_state.get("cpu_load", 0.0), 0.0)
        capture_ms = self.adaptive_policy._avg(self.adaptive_policy.capture_latency_ms, 0.0)
        if capture_ms <= 0.0:
            capture_ms = self.measure_capture_latency()
        life_value = self.store.life if self.store else 0.0
        settings = derive_runtime_settings(base_settings=self.settings, rect=self.current_rect() or self.screen_rect(), pool_count=self.experience_pool.count() if self.experience_pool else 0, capture_ms=capture_ms, cpu_load=cpu_load, execution_ms=self.adaptive_policy._avg(self.adaptive_policy.execution_latency_ms, 0.0), window_instability=self.adaptive_policy._avg(self.adaptive_policy.window_change_flags, 0.0), recent_success=self.adaptive_policy._avg(self.adaptive_policy.outcome_flags, 1.0), life=life_value, learning_similarity=self.adaptive_policy._avg(self.adaptive_policy.learning_similarity, 0.97), hardware=self.hardware_state)
        self.settings = settings
        self.escape_monitor.debounce_seconds = settings.key_debounce_seconds
        self.ui(lambda: self.minsize(settings.ui_min_width, settings.ui_min_height))
        return Config(Path(self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH), data_path, training_seconds, sleep_seconds, still_seconds, settings)

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
            if expected is not None and self.mode != expected:
                return None
            if target == "idle":
                if self.active_session:
                    self.active_session.termination_reason = reason
                self.termination_reason = reason
                self.mode = "idle"
                self.active_session = ModeSession(self.run_token, "idle", time.perf_counter(), None, self.stop_event, reason)
                session = self.active_session
            else:
                self.run_token += 1 if token is None else 0
                session_token = self.run_token if token is None else token
                stop_event = self.stop_event if token is not None else threading.Event()
                self.stop_event = stop_event
                self.mode = target
                session = ModeSession(session_token, target, time.perf_counter(), deadline, stop_event, reason)
                self.active_session = session
        self.set_mode_ui(target)
        return session

    def is_run_active(self, token, mode=None):
        with self.state_lock:
            return token == self.run_token and (mode is None or self.mode == mode)

    def begin_run(self, mode, deadline=None):
        session = self.transition("idle", mode, token=None, deadline=deadline)
        if not session:
            return None, None
        return session.token, session.stop_event

    def activate_run(self, token, mode):
        session = self.transition(None, mode, token=token)
        return bool(session)

    def finish_run(self, token, status, progress=0.0, release=True, reason=None):
        if not self.transition(None, "idle", reason=reason or status, token=token):
            return False
        self.update_progress(progress)
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
        return self.escape_monitor.poll_pressed()

    def release_window_and_panel(self):
        self.restore_panel()

    def ensure_runtime(self, config):
        reload_pool = False
        if not self.store or self.store.root != config.data_path:
            self.store = DataStore(config.data_path)
            reload_pool = True
        if reload_pool or self.experience_pool.settings != config.settings:
            self.experience_pool = ExperiencePool(config.settings, self.store.load_experience(config.settings.experience_load_limit))
            self.brain = ActionBrain(self.experience_pool, config.settings)
            self.ui(lambda: self.life_var.set(str(self.store.state.get("life_experience", 0.0))))
            self.ui(lambda: self.pool_var.set(str(self.experience_pool.count())))
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
        token, stop_event = self.begin_run("starting")
        if not token:
            self.status_var.set("请先终止当前模式，或等待当前模式结束")
            return
        config = self.read_config()
        self.update_progress(100.0)
        self.status_var.set("正在启动或连接雷电模拟器")
        self.mode_thread = threading.Thread(target=self.mode_job, args=(token, target_mode, config, stop_event), daemon=True)
        self.mode_thread.start()

    def mode_job(self, token, mode, config, stop_event):
        try:
            if not self.ensure_runtime(config):
                if self.is_run_active(token):
                    self.ui(lambda: messagebox.showerror("未找到窗口", "没有找到雷电模拟器窗口，请确认路径正确或手动启动雷电模拟器。"))
                    self.finish_run(token, "未找到雷电模拟器", 0.0)
                return
            if stop_event.is_set() or not self.is_run_active(token):
                self.finish_run(token, "当前模式已终止", 0.0)
                return
            self.ui(self.iconify)
            if not self.activate_run(token, mode):
                return
            if mode == "learning":
                if self.mouse_recorder:
                    self.mouse_recorder.clear()
                self.mark_learning_activity()
                self.ui(lambda: self.status_var.set("学习模式：记录客户区画面与用户鼠标，移出客户区不会终止"))
                self.learning_loop(token, stop_event, config)
            elif mode == "training":
                self.ui(lambda: self.status_var.set("训练模式：根据实时画面执行 AI 鼠标并记录"))
                self.training_loop(token, stop_event, config)
        except Exception as exc:
            message = str(exc)
            self.ui(lambda m=message: messagebox.showerror("运行失败", m))
            self.finish_run(token, "运行失败", 0.0)

    def stop_current_mode(self):
        with self.state_lock:
            mode = self.mode
            if mode not in ["starting", "learning", "training", "sleep", "migration"]:
                self.ui(lambda: self.status_var.set("当前没有正在运行的模式"))
                return
            self.stop_event.set()
            self.run_token += 1
            progress_now = self.progress_value
            self.mode = "idle"
            self.termination_reason = "user_stop"
            self.active_session = ModeSession(self.run_token, "idle", time.perf_counter(), None, self.stop_event, "user_stop")
        self.set_mode_ui("idle")
        self.update_progress(0.0 if mode == "sleep" else progress_now)
        self.ui(lambda: self.status_var.set("当前模式已终止"))
        self.ui(self.release_window_and_panel)

    def sleep_mode(self):
        token, stop_event = self.begin_run("sleep")
        if not token:
            self.status_var.set("请先终止当前模式，或等待当前模式结束")
            return
        config = self.read_config()
        self.status_var.set("睡眠模式运行中")
        self.update_progress(0.0)
        self.mode_thread = threading.Thread(target=self.sleep_loop, args=(token, config, stop_event), daemon=True)
        self.mode_thread.start()

    def sleep_loop(self, token, config, stop_event):
        start = time.perf_counter()
        self.ui(lambda: self.progress_label_var.set("模式进度"))
        while not stop_event.is_set() and self.is_run_active(token, "sleep"):
            if self.should_stop_by_escape():
                stop_event.set()
                break
            elapsed = time.perf_counter() - start
            percent = clamp(elapsed / config.sleep_seconds * 100.0, 0.0, 100.0)
            self.update_progress(percent)
            if percent >= 100.0:
                break
            time.sleep(config.settings.sleep_tick)
        if self.is_run_active(token, "sleep") and not stop_event.is_set():
            self.finish_run(token, "睡眠模式完成", 100.0, release=False)
        elif self.is_run_active(token, "sleep"):
            self.finish_run(token, "睡眠模式已终止", 0.0, release=False)

    def restore_panel(self):
        def apply():
            try:
                self.deiconify()
                self.lift()
            except Exception:
                pass
        self.ui(apply)

    def update_progress(self, percent):
        percent = round(clamp(percent, 0.0, 100.0), 1)
        now_perf = time.perf_counter()
        if abs(percent - self.progress_value) < self.settings.ui_progress_delta and now_perf - self.last_progress_update_perf < self.settings.ui_refresh_interval:
            return
        self.progress_value = percent
        self.last_progress_update_perf = now_perf
        def apply():
            self.progress_var.set(percent)
            self.progress_text_var.set(f"{percent:.1f}%")
        self.ui(apply)

    def update_metrics(self, novelty, human_score, screen_reward, action_reward, reward, life, decision=None):
        payload = (round(float(novelty), 2), round(float(human_score), 2), round(float(screen_reward), 2), round(float(action_reward), 2), round(float(reward), 2), round(float(life), 2), decision.get("reason") if decision else None, round(safe_float(decision.get("confidence", 0.0), 0.0), 3) if decision else None)
        now_perf = time.perf_counter()
        if payload == self.last_metric_payload and now_perf - self.last_metrics_update_perf < self.settings.ui_refresh_interval:
            return
        self.last_metric_payload = payload
        self.last_metrics_update_perf = now_perf
        def apply():
            self.novelty_var.set(f"{round(float(novelty), 2)}%")
            self.human_var.set(f"{round(float(human_score), 2)}%")
            self.screen_reward_var.set(str(round(float(screen_reward), 2)))
            self.action_reward_var.set(str(round(float(action_reward), 2)))
            self.reward_var.set(str(round(float(reward), 2)))
            self.life_var.set(str(round(float(life), 2)))
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
            snapshot = ScreenSnapshot(path=path, relative_path=self.store.relative_path(path), hash_value=hash_value, captured_at=now_text(), perf_time=perf_time, elapsed=round(perf_time - session_start, 3), rect=tuple(rect), capture_latency_ms=round((captured_perf - perf_time) * 1000.0, 3), image_priority=priority)
            return snapshot, image
        except Exception as exc:
            self.log_exception("capture_snapshot", exc, {"mode": mode, "rect": list(rect)})
            return None, None

    def mark_snapshot_image_result(self, snapshot, saved):
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
                if self.persistence_paused.is_set() and priority != "critical":
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
        before_novelty, batch = self.experience_pool.novelty(snapshot.hash_value)
        normalized = normalize_mouse_action(action, snapshot.rect) if action else None
        mouse_source = normalized.get("source") if normalized else "idle"
        human_score = self.experience_pool.human_score(normalized) if normalized else 50.0
        after_novelty = self.experience_pool.novelty(after_snapshot.hash_value)[0] if after_snapshot else before_novelty
        transition_reward = round(after_novelty - before_novelty, 2)
        reward_info = reward_breakdown(after_novelty, human_score if normalized else self.settings.score_default, self.settings)
        novelty_reward = reward_info["screen_primary_reward"]
        human_delta = reward_info["mouse_action_delta"]
        reward = reward_info["total_reward"]
        human_action_reward = max(0.0, human_delta)
        human_action_penalty = max(0.0, -human_delta)
        if failed_action:
            reward = round(-abs(max(1.0, human_action_penalty, 100.0 - human_score)), 2)
        life = self.store.add_life_experience(reward)
        started_perf = safe_float(normalized.get("started_perf"), None) if normalized else None
        offset_source = started_perf if started_perf is not None else action_anchor_perf
        offset_ms = round((float(offset_source) - snapshot.perf_time) * 1000.0, 3) if offset_source is not None else None
        sims = [round(item["similarity"], 4) for item in batch]
        record = {"record_schema_version": 2, "id": uuid.uuid4().hex, "session_id": session_id, "created_at": now_text(), "mode": mode, "event": event_name, "elapsed": snapshot.elapsed, "screen_path": snapshot.relative_path, "screen_hash": snapshot.hash_value.hex, "screen_hash_hex": snapshot.hash_value.hex, "screen_hash_int": snapshot.hash_value.value, "screen_hash_bits": snapshot.hash_value.bits, "screen_captured_at": snapshot.captured_at, "screen_perf": round(snapshot.perf_time, 6), "mouse_action": normalized, "planned_action": normalize_mouse_action(planned_action, snapshot.rect) if planned_action else None, "actual_action": None if failed_action else normalized, "execution_error": str(execution_error) if execution_error else None, "mouse_source": mouse_source, "screen_action_offset_ms": offset_ms, "nearest": [{"id": item["record"].get("id"), "similarity": round(item["similarity"], 4)} for item in batch], "nearest_summary": {"count": len(sims), "max_similarity": max(sims) if sims else 0.0, "avg_similarity": round(sum(sims) / len(sims), 4) if sims else 0.0}, "novelty": after_novelty, "before_screen": snapshot.relative_path if normalized else None, "after_screen": after_snapshot.relative_path if after_snapshot else snapshot.relative_path, "before_novelty": before_novelty, "after_novelty": after_novelty, "transition_reward": transition_reward, "screen_observation_reward": novelty_reward, "screen_primary_reward": reward_info["screen_primary_reward"], "human_tie_break_reward": reward_info["human_tie_break_reward"], "reward_sort_key": reward_info["reward_sort_key"], "mouse_action_reward": human_action_reward, "mouse_action_penalty": human_action_penalty, "human_score": human_score, "total_reward": reward, "reward": reward, "novelty_reward": novelty_reward, "human_action_reward": human_action_reward, "human_action_penalty": human_action_penalty, "life_experience_delta": max(0.0, reward), "penalty_delta": max(0.0, -reward), "life_experience": life, "client_rect": list(snapshot.rect), "failed_action": bool(failed_action), "window_rect_changed": bool(window_rect_changed), "image_dropped": bool(getattr(snapshot, "image_dropped", False)), "screen_file_expected": not bool(getattr(snapshot, "image_dropped", False)), "capture_latency_ms": capture_latency_ms if capture_latency_ms is not None else getattr(snapshot, "capture_latency_ms", None), "execution_latency_ms": execution_latency_ms, "termination_reason": None, "policy_snapshot": {"hash_size": self.settings.hash_size, "nearest_top_k": self.settings.nearest_top_k, "training_tick": self.settings.training_tick, "explore_min_rate": self.settings.explore_min_rate, "explore_max_rate": self.settings.explore_max_rate, "action_jitter": self.settings.action_jitter}}
        if decision:
            record["ai_decision"] = decision
        self.persistence_queue.enqueue_record(self.store, record)
        self.experience_pool.add(record)
        self.update_metrics(after_novelty, human_score, novelty_reward, human_delta, reward, life, decision)
        return record

    def maybe_learning_screen_tick(self, analyzer, session_id, start, now_perf, config):
        interval = 1.0 / max(0.5, self.settings.learning_screen_fps)
        if now_perf - self.last_learning_tick_perf < interval:
            return
        snapshot, image = self.capture_snapshot_image(analyzer, "learning", start, priority="low")
        if not snapshot:
            return
        if self.last_learning_tick_hash:
            similarity = hash_similarity(snapshot.hash_value, self.last_learning_tick_hash)
            self.adaptive_policy.observe_capture(0.0, similarity=similarity, window_rect_changed=False)
            if similarity >= self.settings.learning_screen_similarity_threshold:
                self.last_learning_tick_perf = now_perf
                return
        try:
            if self.persistence_paused.is_set():
                self.mark_snapshot_image_result(snapshot, False)
            else:
                self.mark_snapshot_image_result(snapshot, self.persistence_queue.enqueue_image(analyzer, image, snapshot.path, self.store, priority="low"))
        except Exception as exc:
            self.log_exception("learning_screen_tick.save", exc, {"path": str(snapshot.path)})
            return
        self.adaptive_policy.observe_capture(getattr(snapshot, "capture_latency_ms", 0.0))
        self.last_learning_tick_perf = now_perf
        self.last_learning_tick_hash = snapshot.hash_value
        self.write_record("learning", session_id, snapshot, None, "screen_tick")

    def learning_loop(self, token, stop_event, config):
        session_id = uuid.uuid4().hex
        start = time.perf_counter()
        pending_snapshots = {}
        self.mark_learning_activity()
        self.last_learning_tick_perf = time.perf_counter()
        self.last_learning_tick_hash = None
        with ScreenAnalyzer(config.settings.hash_size) as analyzer:
            self.write_record("learning", session_id, self.capture_snapshot(analyzer, "learning", session_id, start, priority="critical"), None, "mode_start")
            while not stop_event.is_set() and self.is_run_active(token, "learning"):
                if self.should_stop_by_escape():
                    stop_event.set()
                    break
                now_perf = time.perf_counter()
                if not self.window_manager.window_ok():
                    stop_event.set()
                    self.ui(lambda: self.status_var.set("学习模式结束：雷电模拟器窗口异常"))
                    break
                self.maybe_learning_screen_tick(analyzer, session_id, start, now_perf, config)
                idle_seconds = self.learning_idle_seconds()
                self.ui(lambda: self.progress_label_var.set("静止倒计时"))
                self.update_progress(clamp((config.still_seconds - idle_seconds) / config.still_seconds * 100.0, 0.0, 100.0))
                if idle_seconds >= config.still_seconds:
                    stop_event.set()
                    self.ui(lambda: self.status_var.set("学习模式结束：鼠标静止超时"))
                    break
                if self.mouse_recorder:
                    markers = self.mouse_recorder.pop_start_markers()
                    for marker in markers:
                        marker_snapshot = self.capture_snapshot(analyzer, "learning", session_id, start, priority="critical")
                        if marker_snapshot:
                            pending_snapshots[marker["action_id"]] = marker_snapshot
                    actions = self.mouse_recorder.pop_actions()
                    for action in actions:
                        action_snapshot = pending_snapshots.pop(action.get("action_id"), None) or self.capture_snapshot(analyzer, "learning", session_id, start, priority="critical")
                        after_snapshot = self.capture_snapshot(analyzer, "learning", session_id, start, priority="critical")
                        self.write_record("learning", session_id, action_snapshot, action, "user_mouse", action_anchor_perf=action.get("started_perf") or action.get("t0"), after_snapshot=after_snapshot, planned_action=action)
                    if not markers and not actions:
                        self.mouse_recorder.wait(config.settings.mouse_still_tick)
                else:
                    time.sleep(config.settings.mouse_still_tick)
            self.write_record("learning", session_id, self.capture_snapshot(analyzer, "learning", session_id, start, priority="critical"), None, "mode_end")
        if self.is_run_active(token, "learning"):
            self.finish_run(token, "学习模式结束", 0.0)
        else:
            self.release_window_and_panel()

    def training_loop(self, token, stop_event, config):
        session_id = uuid.uuid4().hex
        start = time.perf_counter()
        consecutive_failures = 0
        service = self.training_service
        with ScreenAnalyzer(config.settings.hash_size) as analyzer:
            self.write_record("training", session_id, self.capture_snapshot(analyzer, "training", session_id, start, priority="critical"), None, "mode_start")
            while not stop_event.is_set() and self.is_run_active(token, "training"):
                if service.should_stop(start, config, stop_event):
                    break
                rect = self.current_rect()
                service.prepare_tick(rect)
                if not rect:
                    time.sleep(self.settings.training_tick)
                    continue
                snapshot = service.observe_screen(analyzer, session_id, start, rect)
                if not snapshot:
                    time.sleep(self.settings.training_tick)
                    continue
                action, decision = service.decide_action(snapshot)
                if not action:
                    self.write_record("training", session_id, snapshot, None, "screen_tick", decision=decision)
                    time.sleep(self.settings.training_tick)
                    continue
                success, record = service.execute_and_record(analyzer, session_id, start, rect, snapshot, action, decision, stop_event)
                if not success:
                    consecutive_failures += 1
                    if consecutive_failures >= self.settings.training_fail_stop_count:
                        stop_event.set()
                        self.ui(lambda: self.status_var.set("训练模式结束：连续执行失败"))
                    continue
                consecutive_failures = 0
                delay = safe_float(record["mouse_action"].get("duration", 0.0), 0.0) if record and record.get("mouse_action") else 0.0
                deadline = time.perf_counter() + max(self.settings.min_action_delay_seconds, delay)
                while time.perf_counter() < deadline and not stop_event.is_set():
                    if self.should_stop_by_escape():
                        stop_event.set()
                        break
                    time.sleep(min(self.settings.generated_wait_tick, deadline - time.perf_counter()))
            self.write_record("training", session_id, self.capture_snapshot(analyzer, "training", session_id, start, priority="critical"), None, "mode_end")
        if self.is_run_active(token, "training") and not stop_event.is_set():
            self.finish_run(token, "训练模式结束", 0.0)
        elif self.is_run_active(token, "training"):
            self.finish_run(token, "训练模式已终止", self.progress_value)
        else:
            self.release_window_and_panel()

    def close(self):
        with self.state_lock:
            self.stop_event.set()
            self.run_token += 1
            self.mode = "idle"
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
        now_perf = time.perf_counter()
        if not self.hardware_state:
            self.hardware_state = read_hardware_state()
            self.hardware_last_full_refresh_perf = now_perf
            self.hardware_last_light_refresh_perf = now_perf
            return self.hardware_state
        if now_perf - self.hardware_last_light_refresh_perf >= 1.0:
            cpu_load = safe_float(psutil.cpu_percent(interval=0.0), self.hardware_state.get("cpu_load", 0.0)) if psutil else self.hardware_state.get("cpu_load", 0.0)
            memory = psutil.virtual_memory() if psutil else None
            memory_total = safe_float(getattr(memory, "total", 0.0), 0.0)
            memory_available = safe_float(getattr(memory, "available", 0.0), 0.0)
            memory_free_ratio = clamp(memory_available / memory_total if memory_total > 0 else self.hardware_state.get("memory_free_ratio", 0.0), 0.0, 1.0)
            self.hardware_state["cpu_load"] = clamp(cpu_load, 0.0, 100.0)
            self.hardware_state["memory_free_ratio"] = memory_free_ratio
            self.hardware_last_light_refresh_perf = now_perf
        if now_perf - self.hardware_last_full_refresh_perf >= 30.0:
            full = read_hardware_state()
            full["cpu_load"] = self.hardware_state.get("cpu_load", full.get("cpu_load", 0.0))
            full["memory_free_ratio"] = self.hardware_state.get("memory_free_ratio", full.get("memory_free_ratio", 0.0))
            self.hardware_state = full
            self.hardware_last_full_refresh_perf = now_perf
        return self.hardware_state

if __name__ == "__main__":
    if "--self-test" in sys.argv:
        run_self_test()
        sys.exit(0)
    app = ControlPanel()
    app.mainloop()
