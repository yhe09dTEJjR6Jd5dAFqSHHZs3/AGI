import copy
import ctypes
import json
import heapq
import math
import random
import subprocess
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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

DEFAULT_LDPLAYER_PATH = r"D:\LDPlayer9\dnplayer.exe"
DEFAULT_DATA_PATH = r"C:\Users\Administrator\Desktop\AAA"
DEFAULT_TRAINING_SECONDS = 900
DEFAULT_SLEEP_SECONDS = 1800
DEFAULT_STILL_SECONDS = 10
REQUIRED_MODULES = ("mss", "PIL", "psutil", "pywin32", "pynput.mouse")
MODE_NAMES = {"idle": "空闲", "starting": "准备中", "learning": "学习模式", "training": "训练模式", "sleep": "睡眠模式"}


@dataclass(frozen=True)
class Settings:
    hash_size: int = 16
    nearest_top_k: int = 96
    nearest_candidate_limit: int = 4096
    hash_prefix_bits: int = 12
    mouse_still_tick: float = 0.03
    training_tick: float = 0.05
    sleep_tick: float = 0.1
    key_debounce_seconds: float = 0.35
    window_attach_seconds: float = 45.0
    window_poll_seconds: float = 0.5
    min_action_delay_seconds: float = 0.03
    random_action_min: float = 0.08
    random_action_max: float = 0.92
    explore_min_rate: float = 0.04
    explore_max_rate: float = 0.55
    action_jitter: float = 0.018
    softmax_temperature: float = 16.0
    human_profile_min_samples: int = 24
    human_profile_max_samples: int = 5000
    human_profile_keep_samples: int = 4000
    window_title_keywords: tuple = ("ldplayer", "雷电", "leidian")
    ui_width: int = 940
    ui_height: int = 660
    ui_min_width: int = 880
    ui_min_height: int = 620
    ui_padding: int = 18
    ui_section_padding: int = 12
    ui_metric_columns: int = 5
    ui_metric_min_column_width: int = 180
    click_direct_threshold: float = 0.006
    drag_direct_threshold: float = 0.006
    drag_min_points: int = 4
    drag_bend_penalty_threshold: float = 3.2
    click_long_duration: float = 0.9
    reward_total_min: float = -100.0
    reward_total_max: float = 200.0
    experience_load_limit: int = 30000
    score_default: float = 65.0
    scroll_score_default: float = 72.0
    fallback_score_base: float = 62.0
    global_action_probability: float = 0.55
    random_click_duration_min: float = 0.07
    random_click_duration_max: float = 0.22
    action_duration_min: float = 0.05
    action_duration_max: float = 1.8
    generated_click_hold_max: float = 0.25
    generated_sleep_tick: float = 0.015
    generated_wait_tick: float = 0.02
    motion_steps_per_second: float = 60.0
    motion_curve_offset_min: float = 0.04
    motion_curve_offset_max: float = 0.16
    motion_first_control_min: float = 0.25
    motion_first_control_max: float = 0.45
    motion_second_control_min: float = 0.55
    motion_second_control_max: float = 0.75


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


def enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


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


def normalize_settings(settings):
    random_min = clamp(settings.random_action_min, 0.0, 1.0)
    random_max = clamp(settings.random_action_max, 0.0, 1.0)
    if random_min > random_max:
        random_min, random_max = random_max, random_min
    max_samples = max(10, safe_int(settings.human_profile_max_samples, 5000))
    keep_samples = clamp(settings.human_profile_keep_samples, 10, max_samples)
    return Settings(
        hash_size=max(4, safe_int(settings.hash_size, 16)),
        nearest_top_k=max(1, safe_int(settings.nearest_top_k, 96)),
        nearest_candidate_limit=max(1, safe_int(settings.nearest_candidate_limit, 4096)),
        hash_prefix_bits=max(1, safe_int(settings.hash_prefix_bits, 12)),
        mouse_still_tick=clamp(settings.mouse_still_tick, 0.01, 1.0),
        training_tick=clamp(settings.training_tick, 0.01, 5.0),
        sleep_tick=clamp(settings.sleep_tick, 0.05, 5.0),
        key_debounce_seconds=clamp(settings.key_debounce_seconds, 0.05, 5.0),
        window_attach_seconds=clamp(settings.window_attach_seconds, 1.0, 600.0),
        window_poll_seconds=clamp(settings.window_poll_seconds, 0.05, 10.0),
        min_action_delay_seconds=clamp(settings.min_action_delay_seconds, 0.0, 5.0),
        random_action_min=random_min,
        random_action_max=random_max,
        explore_min_rate=clamp(settings.explore_min_rate, 0.0, 1.0),
        explore_max_rate=clamp(settings.explore_max_rate, 0.0, 1.0),
        action_jitter=clamp(settings.action_jitter, 0.0, 0.5),
        softmax_temperature=max(0.1, safe_float(settings.softmax_temperature, 16.0)),
        human_profile_min_samples=max(1, safe_int(settings.human_profile_min_samples, 24)),
        human_profile_max_samples=max_samples,
        human_profile_keep_samples=int(keep_samples),
        window_title_keywords=tuple(settings.window_title_keywords) or ("ldplayer", "雷电", "leidian"),
        ui_width=max(1, safe_int(settings.ui_width, 940)),
        ui_height=max(1, safe_int(settings.ui_height, 660)),
        ui_min_width=max(1, safe_int(settings.ui_min_width, 880)),
        ui_min_height=max(1, safe_int(settings.ui_min_height, 620)),
        ui_padding=max(0, safe_int(settings.ui_padding, 18)),
        ui_section_padding=max(0, safe_int(settings.ui_section_padding, 12)),
        ui_metric_columns=max(1, safe_int(settings.ui_metric_columns, 5)),
        ui_metric_min_column_width=max(120, safe_int(settings.ui_metric_min_column_width, 180)),
        click_direct_threshold=clamp(settings.click_direct_threshold, 0.0, 1.0),
        drag_direct_threshold=clamp(settings.drag_direct_threshold, 0.0, 1.0),
        drag_min_points=max(1, safe_int(settings.drag_min_points, 4)),
        drag_bend_penalty_threshold=clamp(settings.drag_bend_penalty_threshold, 1.0, 20.0),
        click_long_duration=clamp(settings.click_long_duration, 0.0, 60.0),
        reward_total_min=clamp(settings.reward_total_min, -10000.0, 10000.0),
        reward_total_max=clamp(settings.reward_total_max, -10000.0, 10000.0),
        experience_load_limit=max(1000, safe_int(getattr(settings, "experience_load_limit", 30000), 30000)),
        score_default=clamp(settings.score_default, 0.0, 100.0),
        scroll_score_default=clamp(settings.scroll_score_default, 0.0, 100.0),
        fallback_score_base=clamp(settings.fallback_score_base, 0.0, 100.0),
        global_action_probability=clamp(settings.global_action_probability, 0.0, 1.0),
        random_click_duration_min=clamp(settings.random_click_duration_min, 0.0, 60.0),
        random_click_duration_max=clamp(settings.random_click_duration_max, 0.0, 60.0),
        action_duration_min=clamp(settings.action_duration_min, 0.0, 60.0),
        action_duration_max=clamp(settings.action_duration_max, 0.0, 60.0),
        generated_click_hold_max=clamp(settings.generated_click_hold_max, 0.0, 60.0),
        generated_sleep_tick=clamp(settings.generated_sleep_tick, 0.001, 1.0),
        generated_wait_tick=clamp(settings.generated_wait_tick, 0.001, 1.0),
        motion_steps_per_second=clamp(settings.motion_steps_per_second, 1.0, 1000.0),
        motion_curve_offset_min=clamp(settings.motion_curve_offset_min, 0.0, 1.0),
        motion_curve_offset_max=clamp(settings.motion_curve_offset_max, 0.0, 1.0),
        motion_first_control_min=clamp(settings.motion_first_control_min, 0.0, 1.0),
        motion_first_control_max=clamp(settings.motion_first_control_max, 0.0, 1.0),
        motion_second_control_min=clamp(settings.motion_second_control_min, 0.0, 1.0),
        motion_second_control_max=clamp(settings.motion_second_control_max, 0.0, 1.0)
    )


def settings_from_dict(data):
    if not isinstance(data, dict):
        return Settings()
    defaults = Settings()
    values = {}
    for item in fields(Settings):
        if item.name not in data:
            continue
        raw = data[item.name]
        default = getattr(defaults, item.name)
        try:
            if isinstance(default, bool):
                values[item.name] = bool(raw)
            elif isinstance(default, int) and not isinstance(default, bool):
                values[item.name] = int(float(raw))
            elif isinstance(default, float):
                values[item.name] = float(raw)
            elif isinstance(default, tuple):
                if isinstance(raw, str):
                    values[item.name] = tuple(part.strip() for part in raw.split(",") if part.strip())
                elif isinstance(raw, (list, tuple)):
                    values[item.name] = tuple(str(part).strip() for part in raw if str(part).strip())
            else:
                values[item.name] = raw
        except Exception:
            pass
    try:
        return normalize_settings(Settings(**values))
    except Exception:
        return Settings()


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


def reward_parts(novelty, human_score, settings):
    screen_reward = round(clamp(novelty, 0.0, 100.0), 2)
    action_reward = round((clamp(human_score, 0.0, 100.0) - 50.0) * 2.0, 2)
    total = round(clamp(screen_reward + action_reward, settings.reward_total_min, settings.reward_total_max), 2)
    return screen_reward, action_reward, total


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
    points = [[safe_float(item[0], 0.0), safe_float(item[1], 0.0)] for item in action.get("path_rel", []) if isinstance(item, (list, tuple)) and len(item) >= 2]
    if not points:
        points = [normalize_rel_point(start), normalize_rel_point(end, start)]
    direct = distance(points[0], points[-1])
    total = path_length(points)
    scroll = action.get("scroll") or [0, 0]
    return {
        "duration": max(0.0, safe_float(action.get("duration", 0.0), 0.0)),
        "direct": direct,
        "total": total,
        "bend": clamp(total / direct if direct > 1e-6 else 1.0, 1.0, 5.0),
        "points": len(points),
        "scroll_abs": abs(safe_int(scroll[0], 0)) + abs(safe_int(scroll[1], 0))
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
        if self.enough(action_type):
            with self.lock:
                stats = {name: tuple(values) for name, values in self.stats[action_type].items()}
            scores = [
                percentile_score(features.get("duration", 0.0), stats.get("duration", []), self.settings.score_default),
                percentile_score(features.get("direct", 0.0), stats.get("direct", []), self.settings.score_default),
                percentile_score(features.get("bend", 1.0), stats.get("bend", []), self.settings.score_default),
                percentile_score(features.get("points", 2.0), stats.get("points", []), self.settings.score_default)
            ]
            weights = [0.34, 0.26, 0.24, 0.16]
            return round(clamp(sum(score * weight for score, weight in zip(scores, weights)), 0.0, 100.0), 2)
        return self.fallback_score(action_type, features)

    def score_scroll(self, features):
        with self.lock:
            samples = list(self.stats["scroll"].get("scroll_abs", []))
        if len(samples) >= max(6, self.settings.human_profile_min_samples // 3):
            return percentile_score(features.get("scroll_abs", 0.0), samples, self.settings.scroll_score_default)
        amount = features.get("scroll_abs", 0.0)
        if amount <= 0.0:
            return 55.0
        return 78.0 if amount <= 8.0 else 68.0

    def fallback_score(self, action_type, features):
        duration = features.get("duration", 0.0)
        direct = features.get("direct", 0.0)
        bend = features.get("bend", 1.0)
        points = features.get("points", 0.0)
        score = self.settings.fallback_score_base
        score += clamp(math.log1p(duration * 10.0) * 10.0, 0.0, 18.0) if duration > 0.0 else -8.0
        if action_type == "click":
            score += 8.0 if direct <= self.settings.click_direct_threshold else -4.0
            if duration > self.settings.click_long_duration:
                score -= 8.0
        elif action_type == "drag":
            score += 10.0 if direct > self.settings.drag_direct_threshold else -8.0
            if points >= self.settings.drag_min_points:
                score += 8.0
            if bend > self.settings.drag_bend_penalty_threshold:
                score -= 5.0
        else:
            score += 4.0
        return round(clamp(score, 0.0, 100.0), 2)


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

    def load_state(self):
        if not self.state_file.exists():
            return {"life_experience": 0.0}
        try:
            with self.state_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return {"life_experience": safe_float(data.get("life_experience", 0.0), 0.0)}
        except Exception:
            return {"life_experience": 0.0}

    def load_settings(self):
        if not self.settings_file.exists():
            return Settings()
        try:
            with self.settings_file.open("r", encoding="utf-8") as file:
                return settings_from_dict(json.load(file))
        except Exception:
            return Settings()

    def save_state(self):
        with self.lock:
            temporary = self.state_file.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(self.state, file, ensure_ascii=False, indent=2)
            temporary.replace(self.state_file)

    def add_life_experience(self, value):
        with self.lock:
            current = safe_float(self.state.get("life_experience", 0.0), 0.0)
            self.state["life_experience"] = round(max(0.0, current + safe_float(value, 0.0)), 2)
            self.save_state()
            return self.state["life_experience"]

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

    def load_experience(self, limit=None):
        records = []
        if not self.experience_file.exists():
            return records
        tail = deque(maxlen=max(1, safe_int(limit, 0))) if limit else None
        with self.experience_file.open("r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if not text:
                    continue
                if tail is not None:
                    tail.append(text)
                else:
                    try:
                        records.append(json.loads(text))
                    except Exception:
                        pass
        if tail is None:
            return records
        for text in tail:
            try:
                records.append(json.loads(text))
            except Exception:
                pass
        return records

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


class WindowManager:
    def __init__(self, executable_path, settings):
        self.executable_path = Path(executable_path)
        self.settings = settings
        self.process = None
        self.hwnd = None
        self.lock = threading.RLock()

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
        with self.lock:
            hwnd = self.hwnd
        if not hwnd:
            return
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
        except Exception:
            pass
        self.foreground()

    def not_topmost(self):
        with self.lock:
            hwnd = self.hwnd
        if not hwnd:
            return
        try:
            win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        except Exception:
            pass


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

    def capture(self, rect, path):
        left, top, right, bottom = rect
        width, height = rect_size(rect)
        shot = self.sct.grab({"left": int(left), "top": int(top), "width": width, "height": height})
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        image.save(path)
        return image

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
    def __init__(self, settings, records=None):
        self.settings = settings
        self.records = []
        self.hashes = []
        self.index = defaultdict(list)
        self.sorted_prefixes = []
        self.profile = HumanProfile(settings)
        self.lock = threading.RLock()
        for record in records or []:
            self.add(record)

    def _prefix(self, hash_value):
        bits = min(max(1, self.settings.hash_prefix_bits), hash_value.bits)
        return hash_value.value >> max(0, hash_value.bits - bits)

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
                bucket.append(index)
            if record.get("mouse_action") and record.get("mode") == "learning" and record.get("mouse_source") == "user":
                self.profile.observe(record["mouse_action"])

    def count(self):
        with self.lock:
            return len(self.records)

    def action_records(self):
        with self.lock:
            return [record for record in self.records if record.get("mouse_action")]

    def candidate_indices(self, hash_value):
        with self.lock:
            if len(self.records) <= self.settings.nearest_candidate_limit:
                return [index for index, item in enumerate(self.hashes) if item]
            query_prefix = self._prefix(hash_value)
            result = []
            nearby = sorted(self.sorted_prefixes, key=lambda item: (item ^ query_prefix).bit_count())
            for prefix in nearby:
                result.extend(self.index[prefix])
                if len(result) >= self.settings.nearest_candidate_limit:
                    break
            if result:
                return result[:self.settings.nearest_candidate_limit]
            valid = [index for index, item in enumerate(self.hashes) if item]
            return random.sample(valid, self.settings.nearest_candidate_limit) if len(valid) > self.settings.nearest_candidate_limit else valid

    def nearest(self, hash_value):
        if not hash_value:
            return []
        with self.lock:
            top_k = max(1, self.settings.nearest_top_k)
            scored = []
            for index in self.candidate_indices(hash_value):
                other = self.hashes[index]
                if other:
                    similarity = hash_similarity(hash_value, other)
                    item = {"similarity": similarity, "record": self.records[index]}
                    if len(scored) < top_k:
                        heapq.heappush(scored, (similarity, item))
                    elif similarity > scored[0][0]:
                        heapq.heapreplace(scored, (similarity, item))
        return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)]

    def novelty(self, hash_value):
        batch = self.nearest(hash_value)
        if not batch:
            return 100.0, []
        return round(clamp((1.0 - batch[0]["similarity"]) * 100.0, 0.0, 100.0), 2), batch

    def best_global_action(self):
        weighted = []
        for record in self.action_records():
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
        return similarity * 100.0 + reward * 0.55 + human_score * 0.25 + novelty * 0.08 + source_bonus

    def mutate_point(self, point, scale):
        return [
            round(clamp(safe_float(point[0], 0.5) + random.uniform(-scale, scale), self.settings.random_action_min, self.settings.random_action_max), 6),
            round(clamp(safe_float(point[1], 0.5) + random.uniform(-scale, scale), self.settings.random_action_min, self.settings.random_action_max), 6)
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
        point = [round(random.uniform(self.settings.random_action_min, self.settings.random_action_max), 6), round(random.uniform(self.settings.random_action_min, self.settings.random_action_max), 6)]
        action = {"type": "click", "button": "Button.left", "source": "ai", "duration": round(random.uniform(self.settings.random_click_duration_min, self.settings.random_click_duration_max), 6), "start_rel": point, "end_rel": point, "path_rel": []}
        return action, "explore_random"

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
        self.last_action = copy.deepcopy(action)
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
            duration = clamp((distance_to_start + main_distance) / max(width, height), self.settings.action_duration_min, 1.2)
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
                self.stoppable_sleep(clamp(main_duration, 0.03, self.settings.generated_click_hold_max), stop_event, should_stop)
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


class ControlPanel(tk.Tk):
    def __init__(self):
        enable_dpi_awareness()
        super().__init__()
        self.settings = Settings()
        self.title("雷电模拟器学习训练控制面板")
        self.geometry(f"{self.settings.ui_width}x{self.settings.ui_height}")
        self.minsize(self.settings.ui_min_width, self.settings.ui_min_height)
        self.state_lock = threading.RLock()
        self.mode = "idle"
        self.run_token = 0
        self.stop_event = threading.Event()
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
        self.progress_label_var = tk.StringVar(value="进度")
        self.metric_items = []
        self.hint_label = None
        self.metrics_frame = None
        self.build_ui()
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
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 17, "bold"))
        style.configure("CardTitle.TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("Value.TLabel", font=("Microsoft YaHei UI", 14, "bold"))
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
        ttk.Label(container, text="雷电模拟器学习训练控制面板", style="Title.TLabel").pack(anchor="w")
        path_frame = ttk.LabelFrame(container, text="路径与时间", padding=self.settings.ui_section_padding)
        path_frame.pack(fill="x", pady=(16, 10))
        path_frame.columnconfigure(1, weight=1)
        ttk.Label(path_frame, text="雷电模拟器").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(path_frame, textvariable=self.ldplayer_var).grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(path_frame, text="修改", command=self.choose_ldplayer).grid(row=0, column=2, padx=(8, 0), pady=6)
        ttk.Label(path_frame, text="数据存储").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(path_frame, textvariable=self.data_var).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Button(path_frame, text="修改", command=self.choose_data).grid(row=1, column=2, padx=(8, 0), pady=6)
        time_frame = ttk.Frame(path_frame)
        time_frame.grid(row=2, column=1, columnspan=2, sticky="w", pady=6)
        ttk.Label(path_frame, text="时间设置").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=6)
        for label, variable in (("训练/秒", self.training_seconds_var), ("睡眠/秒", self.sleep_seconds_var), ("静止/秒", self.still_seconds_var)):
            ttk.Label(time_frame, text=label).pack(side="left")
            ttk.Entry(time_frame, textvariable=variable, width=10).pack(side="left", padx=(6, 16))
        button_frame = ttk.Frame(container)
        button_frame.pack(fill="x", pady=(4, 12))
        ttk.Button(button_frame, text="学习模式", command=self.learning_mode).pack(side="left", padx=(0, 8))
        ttk.Button(button_frame, text="训练模式", command=self.training_mode).pack(side="left", padx=8)
        ttk.Button(button_frame, text="睡眠模式", command=self.sleep_mode).pack(side="left", padx=8)
        ttk.Button(button_frame, text="终止当前模式", command=self.stop_current_mode).pack(side="left", padx=8)
        ttk.Button(button_frame, text="退出", command=self.close).pack(side="right")
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
        progress_frame = ttk.LabelFrame(container, textvariable=self.progress_label_var, padding=self.settings.ui_section_padding)
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
        preferred = max(1, self.settings.ui_metric_columns)
        columns = max(1, min(preferred, width // min_width if width >= min_width else 1))
        for column in range(preferred):
            self.metrics_frame.columnconfigure(column, weight=0)
        for column in range(columns):
            self.metrics_frame.columnconfigure(column, weight=1)
        for index, item in enumerate(self.metric_items):
            row = index // columns
            column = index % columns
            item.grid(row=row, column=column, sticky="ew", padx=(0, 12))

    def on_window_resize(self, _event):
        if self.hint_label:
            width = max(320, self.winfo_width() - self.settings.ui_padding * 4)
            self.hint_label.configure(wraplength=width)
        self.reflow_metrics()

    def ui(self, func):
        try:
            self.after(0, func)
        except Exception:
            pass

    def log_exception(self, where, error, context=None):
        if self.store:
            try:
                self.store.log_error(where, error, context)
            except Exception:
                pass

    def required_import_error(self):
        return {name: IMPORT_ERRORS[name] for name in REQUIRED_MODULES if name in IMPORT_ERRORS}

    def show_import_error(self):
        missing = self.required_import_error()
        lines = [f"{name}: {error}" for name, error in missing.items()]
        messagebox.showerror("缺少依赖", "请先安装依赖：pip install mss pillow pynput psutil pywin32\n\n当前错误：\n" + "\n".join(lines))
        self.status_var.set("依赖缺失")

    def choose_ldplayer(self):
        path = filedialog.askopenfilename(title="选择 dnplayer.exe", filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if path:
            self.ldplayer_var.set(path)

    def choose_data(self):
        path = filedialog.askdirectory(title="选择数据存储目录")
        if path:
            self.data_var.set(path)

    def read_config(self):
        training_seconds = max(1, safe_int(self.training_seconds_var.get(), DEFAULT_TRAINING_SECONDS))
        sleep_seconds = max(1, safe_int(self.sleep_seconds_var.get(), DEFAULT_SLEEP_SECONDS))
        still_seconds = max(0.1, safe_float(self.still_seconds_var.get(), DEFAULT_STILL_SECONDS))
        self.training_seconds_var.set(str(training_seconds))
        self.sleep_seconds_var.set(str(sleep_seconds))
        self.still_seconds_var.set(str(still_seconds))
        data_path = Path(self.data_var.get().strip() or DEFAULT_DATA_PATH)
        settings = normalize_settings(DataStore(data_path).load_settings())
        self.settings = settings
        self.escape_monitor.debounce_seconds = settings.key_debounce_seconds
        self.ui(lambda: self.minsize(settings.ui_min_width, settings.ui_min_height))
        return Config(Path(self.ldplayer_var.get().strip() or DEFAULT_LDPLAYER_PATH), data_path, training_seconds, sleep_seconds, still_seconds, settings)

    def current_mode(self):
        with self.state_lock:
            return self.mode

    def set_mode_ui(self, mode):
        self.ui(lambda m=mode: self.mode_var.set(MODE_NAMES.get(m, m)))

    def is_run_active(self, token, mode=None):
        with self.state_lock:
            return token == self.run_token and (mode is None or self.mode == mode)

    def begin_run(self, mode):
        with self.state_lock:
            if self.mode != "idle":
                return None, None
            self.run_token += 1
            token = self.run_token
            self.stop_event = threading.Event()
            self.mode = mode
            stop_event = self.stop_event
        self.set_mode_ui(mode)
        return token, stop_event

    def activate_run(self, token, mode):
        with self.state_lock:
            if token != self.run_token:
                return False
            self.mode = mode
        self.set_mode_ui(mode)
        return True

    def finish_run(self, token, status, progress=0.0, release=True):
        with self.state_lock:
            if token != self.run_token:
                return False
            self.mode = "idle"
        self.set_mode_ui("idle")
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
        if self.window_manager:
            self.window_manager.not_topmost()
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
            self.window_manager.topmost()
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
            if mode not in ["starting", "learning", "training", "sleep"]:
                self.ui(lambda: self.status_var.set("当前没有正在运行的模式"))
                return
            self.stop_event.set()
            self.run_token += 1
            progress_now = self.progress_var.get()
            self.mode = "idle"
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
        def apply():
            self.progress_var.set(percent)
            self.progress_text_var.set(f"{percent:.1f}%")
        self.ui(apply)

    def update_metrics(self, novelty, human_score, screen_reward, action_reward, reward, life, decision=None):
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

    def capture_snapshot(self, analyzer, mode, session_id, session_start, rect=None):
        if not self.store:
            return None
        rect = rect or self.current_rect()
        if not rect:
            return None
        path = self.store.new_screen_path(mode)
        perf_time = time.perf_counter()
        image = analyzer.capture(rect, path)
        hash_value = analyzer.fingerprint(image)
        return ScreenSnapshot(path=path, relative_path=self.store.relative_path(path), hash_value=hash_value, captured_at=now_text(), perf_time=perf_time, elapsed=round(perf_time - session_start, 3), rect=tuple(rect))

    def write_record(self, mode, session_id, snapshot, action, event_name, decision=None, action_anchor_perf=None, after_snapshot=None):
        if not self.store or not snapshot:
            return None
        before_novelty, batch = self.experience_pool.novelty(snapshot.hash_value)
        normalized = normalize_mouse_action(action, snapshot.rect) if action else None
        mouse_source = normalized.get("source") if normalized else None
        human_score = self.experience_pool.human_score(normalized) if normalized else 0.0
        after_novelty = self.experience_pool.novelty(after_snapshot.hash_value)[0] if after_snapshot else before_novelty
        transition_reward = round(after_novelty - before_novelty, 2) if normalized else 0.0
        screen_reward, action_reward, reward = reward_parts(after_novelty, human_score, self.settings) if normalized else (0.0, 0.0, 0.0)
        reward = round(clamp(reward + transition_reward, self.settings.reward_total_min, self.settings.reward_total_max), 2) if normalized else 0.0
        life = self.store.add_life_experience(reward) if normalized else self.store.state.get("life_experience", 0.0)
        started_perf = safe_float(normalized.get("started_perf"), None) if normalized else None
        offset_source = started_perf if started_perf is not None else action_anchor_perf
        offset_ms = round((float(offset_source) - snapshot.perf_time) * 1000.0, 3) if offset_source is not None else None
        record = {"id": uuid.uuid4().hex, "session_id": session_id, "created_at": now_text(), "mode": mode, "event": event_name, "elapsed": snapshot.elapsed, "screen_path": snapshot.relative_path, "screen_hash": snapshot.hash_value.hex, "screen_hash_hex": snapshot.hash_value.hex, "screen_hash_int": snapshot.hash_value.value, "screen_hash_bits": snapshot.hash_value.bits, "screen_captured_at": snapshot.captured_at, "screen_perf": round(snapshot.perf_time, 6), "mouse_action": normalized, "mouse_source": mouse_source, "screen_action_offset_ms": offset_ms, "nearest": [{"id": item["record"].get("id"), "similarity": round(item["similarity"], 4)} for item in batch], "novelty": after_novelty, "before_screen": snapshot.relative_path if normalized else None, "after_screen": after_snapshot.relative_path if after_snapshot else snapshot.relative_path, "before_novelty": before_novelty, "after_novelty": after_novelty, "transition_reward": transition_reward, "human_score": human_score, "total_reward": reward, "screen_reward": screen_reward, "action_reward": action_reward, "reward": reward, "life_experience": life, "client_rect": list(snapshot.rect)}
        if decision:
            record["ai_decision"] = decision
        self.store.append_experience(record)
        self.experience_pool.add(record)
        self.update_metrics(after_novelty, human_score, screen_reward, action_reward, reward, life, decision)
        return record

    def learning_loop(self, token, stop_event, config):
        session_id = uuid.uuid4().hex
        start = time.perf_counter()
        pending_snapshots = {}
        self.mark_learning_activity()
        with ScreenAnalyzer(config.settings.hash_size) as analyzer:
            self.write_record("learning", session_id, self.capture_snapshot(analyzer, "learning", session_id, start), None, "mode_start")
            while not stop_event.is_set() and self.is_run_active(token, "learning"):
                if self.should_stop_by_escape():
                    stop_event.set()
                    break
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
                        marker_snapshot = self.capture_snapshot(analyzer, "learning", session_id, start)
                        if marker_snapshot:
                            pending_snapshots[marker["action_id"]] = marker_snapshot
                    actions = self.mouse_recorder.pop_actions()
                    for action in actions:
                        action_snapshot = pending_snapshots.pop(action.get("action_id"), None) or self.capture_snapshot(analyzer, "learning", session_id, start)
                        after_snapshot = self.capture_snapshot(analyzer, "learning", session_id, start)
                        self.write_record("learning", session_id, action_snapshot, action, "user_mouse", action_anchor_perf=action.get("started_perf") or action.get("t0"), after_snapshot=after_snapshot)
                    if not markers and not actions:
                        self.mouse_recorder.wait(config.settings.mouse_still_tick)
                else:
                    time.sleep(config.settings.mouse_still_tick)
            self.write_record("learning", session_id, self.capture_snapshot(analyzer, "learning", session_id, start), None, "mode_end")
        if self.is_run_active(token, "learning"):
            self.finish_run(token, "学习模式结束", 0.0)
        else:
            self.release_window_and_panel()

    def training_loop(self, token, stop_event, config):
        session_id = uuid.uuid4().hex
        start = time.perf_counter()
        with ScreenAnalyzer(config.settings.hash_size) as analyzer:
            self.write_record("training", session_id, self.capture_snapshot(analyzer, "training", session_id, start), None, "mode_start")
            while not stop_event.is_set() and self.is_run_active(token, "training"):
                if self.should_stop_by_escape():
                    stop_event.set()
                    break
                elapsed = time.perf_counter() - start
                if elapsed >= config.training_seconds:
                    break
                self.update_progress(clamp((config.training_seconds - elapsed) / config.training_seconds * 100.0, 0.0, 100.0))
                self.ui(lambda: self.progress_label_var.set("模式进度"))
                rect = self.current_rect()
                if not rect:
                    time.sleep(config.settings.training_tick)
                    continue
                snapshot = self.capture_snapshot(analyzer, "training", session_id, start, rect=rect)
                if not snapshot:
                    time.sleep(config.settings.training_tick)
                    continue
                novelty, batch = self.experience_pool.novelty(snapshot.hash_value)
                life = safe_float(self.store.state.get("life_experience", 0.0), 0.0) if self.store else 0.0
                action, decision = self.brain.choose(snapshot.hash_value, novelty, batch, life)
                if action.get("end_rel") is None:
                    action["end_rel"] = action.get("start_rel", [0.5, 0.5])
                actual = self.executor.execute(action, rect, stop_event, self.should_stop_by_escape)
                if not actual:
                    self.log_exception("training.execute", RuntimeError("empty_action_result"))
                    continue
                if actual.get("execution_error"):
                    self.log_exception("training.execute", RuntimeError(actual.get("execution_error")), {"detail": actual, "decision": decision})
                    continue
                after_snapshot = self.capture_snapshot(analyzer, "training", session_id, start, rect=rect)
                record = self.write_record("training", session_id, snapshot, actual, "ai_mouse", decision=decision, action_anchor_perf=actual.get("started_perf"), after_snapshot=after_snapshot)
                delay = safe_float(record["mouse_action"].get("duration", 0.0), 0.0) if record and record.get("mouse_action") else 0.0
                deadline = time.perf_counter() + max(config.settings.min_action_delay_seconds, delay)
                while time.perf_counter() < deadline and not stop_event.is_set():
                    if self.should_stop_by_escape():
                        stop_event.set()
                        break
                    time.sleep(min(config.settings.generated_wait_tick, deadline - time.perf_counter()))
            self.write_record("training", session_id, self.capture_snapshot(analyzer, "training", session_id, start), None, "mode_end")
        if self.is_run_active(token, "training") and not stop_event.is_set():
            self.finish_run(token, "训练模式结束", 0.0)
        elif self.is_run_active(token, "training"):
            self.finish_run(token, "训练模式已终止", self.progress_var.get())
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
        if self.window_manager:
            self.window_manager.not_topmost()
        try:
            self.destroy()
        except Exception:
            pass

if __name__ == "__main__":
    app = ControlPanel()
    app.mainloop()
