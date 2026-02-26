import json
import logging
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
import traceback
import warnings
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =========================
# Win11 高分屏坐标修复（必须在 pyautogui 导入前）
# =========================
try:
    import ctypes

    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

# =========================
# 全局告警处理（按需求预防）
# =========================
warnings.filterwarnings("default")

# =========================
# 日志：禁止静默失败
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def global_exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logging.info("收到 KeyboardInterrupt，程序退出。")
        return
    logging.error("发生未捕获异常：")
    logging.error("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))


sys.excepthook = global_exception_handler


# =========================
# 运行参数与路径
# =========================
DESKTOP = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
AAA_DIR = DESKTOP / "AAA"
MODEL_FILE = AAA_DIR / "AI模型.json"
EXPERIENCE_DIR = AAA_DIR / "经验池"
FIREFOX_PATH = Path(r"E:\FirefoxPortable\FirefoxPortable.exe")
MAX_EXPERIENCE_BYTES = 20 * 1024 * 1024 * 1024

LOCAL_REGION_SIZE = 512
RECONCILE_INTERVAL_SECONDS = 300


@dataclass
class BrowserRect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


# 延迟导入，错误打印详细信息
try:
    import easyocr
    import keyboard
    import numpy as np
    import pyautogui
    import pygetwindow as gw
    import pytweening

    def setup_comtypes_cache() -> None:
        try:
            import shutil

            gen_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "LocalBrowserAI" / "comtypes_gen"
            if gen_dir.exists():
                shutil.rmtree(gen_dir, ignore_errors=True)
            gen_dir.mkdir(parents=True, exist_ok=True)
            os.environ["COMTYPES_CACHE"] = str(gen_dir)
        except Exception:
            logging.warning("设置 comtypes 缓存目录失败：\n%s", traceback.format_exc())

    setup_comtypes_cache()
    import uiautomation as automation
    from PIL import ImageGrab
except Exception:
    logging.error("依赖导入失败，详细错误如下：")
    logging.error(traceback.format_exc())
    raise

pyautogui.FAILSAFE = False


def refresh_screen_size() -> None:
    global SCREEN_WIDTH, SCREEN_HEIGHT
    try:
        size = pyautogui.size()
        SCREEN_WIDTH, SCREEN_HEIGHT = int(size.width), int(size.height)
    except Exception:
        SCREEN_WIDTH, SCREEN_HEIGHT = 2560, 1600
        logging.error("动态获取分辨率失败，回退为默认 2560x1600：\n%s", traceback.format_exc())


def ensure_package(module_name: str, pip_name: Optional[str] = None) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        target = pip_name or module_name
        logging.warning("依赖 %s 缺失，尝试自动安装：%s", module_name, target)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", target])
            __import__(module_name)
            logging.info("依赖安装成功：%s", target)
            return True
        except Exception:
            logging.error("自动安装依赖失败（%s）：\n%s", target, traceback.format_exc())
            return False

STOP_EVENT = threading.Event()
TRIM_LOCK = threading.Lock()
POOL_DB_LOCK = threading.Lock()

OCR_READER = None
POOL_DB_FILE = EXPERIENCE_DIR / "experience_index.sqlite3"
CURRENT_POOL_SIZE = 0
VLM_AVAILABLE = False
VLM_MODEL = None
VLM_PROCESSOR = None
LAST_RECONCILE_TS = 0.0
LAST_ACTIONS: List[Dict] = []
FIREFOX_WINDOW_CACHE = {"hwnd": None, "last_rect": None}
SCREEN_WIDTH, SCREEN_HEIGHT = 0, 0


def ensure_init_files() -> None:
    """初始化阶段：严格检查 AAA/AI模型/经验池，缺失即创建。"""
    try:
        AAA_DIR.mkdir(parents=True, exist_ok=True)
        EXPERIENCE_DIR.mkdir(parents=True, exist_ok=True)

        if not MODEL_FILE.exists():
            base_model = {
                "name": "LocalBrowserAI",
                "version": "3.4-vlm-uia-rag",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "policy": {
                    "auto_model_download_if_missing": True,
                    "allow_keyboard_keys_except_esc": True,
                    "only_mouse_inside_browser": True,
                },
                "knowledge": {
                    "goals": ["理解页面", "优先高价值交互", "记忆历史操作", "避免重复无效动作"],
                    "safe_actions": ["click", "scroll", "type", "wait", "hotkey", "search_new_tab"],
                },
            }
            MODEL_FILE.write_text(json.dumps(base_model, ensure_ascii=False, indent=2), encoding="utf-8")
            logging.info("已创建本地 AI 模型文件（支持缺失模型自动下载）：%s", MODEL_FILE)
        else:
            logging.info("检测到本地 AI 模型文件：%s", MODEL_FILE)

        logging.info("检测到经验池目录：%s", EXPERIENCE_DIR)
    except Exception:
        logging.error("初始化阶段失败，详细错误：\n%s", traceback.format_exc())
        raise


def init_ocr() -> None:
    global OCR_READER
    try:
        OCR_READER = easyocr.Reader(["ch_sim", "en"], gpu=True, verbose=False)
        logging.info("OCR 初始化完成（EasyOCR，中文+英文，GPU加速已开启）。")
    except Exception:
        logging.error("OCR 初始化失败，详细错误：\n%s", traceback.format_exc())
        raise


def init_vlm() -> None:
    """VLM 初始化：本地优先，缺失自动下载，失败时降级规则语义。"""
    global VLM_AVAILABLE, VLM_MODEL, VLM_PROCESSOR
    model_map = {
        "moondream2": "vikhyatk/moondream2",
        "phi3_vision": "microsoft/Phi-3-vision-128k-instruct",
    }
    model_dir_name = next((k for k in ["moondream2", "phi3_vision"] if (AAA_DIR / k).is_dir()), "moondream2")
    model_root = AAA_DIR / model_dir_name

    if not model_root.exists():
        if not ensure_package("huggingface_hub"):
            logging.warning("缺少 huggingface_hub，无法自动下载模型，使用规则语义回退。")
            return
        try:
            from huggingface_hub import snapshot_download

            logging.warning("未检测到本地模型目录 %s，尝试自动下载 %s ...", model_root, model_map[model_dir_name])
            snapshot_download(
                repo_id=model_map[model_dir_name],
                local_dir=str(model_root),
                resume_download=True,
                local_dir_use_symlinks=False,
            )
            logging.info("模型下载完成：%s", model_root)
        except Exception:
            logging.error("自动下载模型失败，降级到规则语义：\n%s", traceback.format_exc())
            return

    if not ensure_package("transformers"):
        logging.warning("缺少 transformers，无法加载 VLM，使用规则语义回退。")
        return

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor

        VLM_PROCESSOR = AutoProcessor.from_pretrained(str(model_root), trust_remote_code=True, local_files_only=True)
        VLM_MODEL = AutoModelForCausalLM.from_pretrained(
            str(model_root), trust_remote_code=True, local_files_only=True, device_map="auto"
        )
        VLM_AVAILABLE = True
        logging.info("VLM 初始化完成：%s", model_root)
    except Exception:
        logging.error("VLM 初始化失败，降级到规则语义：\n%s", traceback.format_exc())


def init_experience_pool_index() -> None:
    """初始化经验池索引，避免每次全量扫描经验池目录。"""
    global CURRENT_POOL_SIZE
    try:
        with sqlite3.connect(POOL_DB_FILE) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    created_ts REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_created_ts ON files(created_ts)")

            row = conn.execute("SELECT COUNT(1), COALESCE(SUM(size), 0) FROM files").fetchone()
            indexed_count = int(row[0]) if row else 0
            indexed_size = int(row[1]) if row else 0

            if indexed_count == 0:
                logging.info("首次构建经验池索引，扫描历史文件中...")
                records = []
                for p in EXPERIENCE_DIR.iterdir():
                    if not p.is_file():
                        continue
                    if not p.name.startswith("exp_"):
                        continue
                    if p.suffix.lower() not in {".jpg", ".json"}:
                        continue
                    stat = p.stat()
                    records.append((str(p), int(stat.st_size), float(stat.st_mtime)))

                if records:
                    conn.executemany(
                        "INSERT OR REPLACE INTO files(path, size, created_ts) VALUES (?, ?, ?)",
                        records,
                    )
                    conn.commit()
                    indexed_size = sum(size for _, size, _ in records)
                logging.info("经验池索引已构建，共 %s 条。", len(records))

            CURRENT_POOL_SIZE = indexed_size
            logging.info("经验池当前大小：%.2fGB", CURRENT_POOL_SIZE / (1024**3))
    except Exception:
        logging.error("经验池索引初始化失败：\n%s", traceback.format_exc())
        raise




def reconcile_experience_pool_index() -> None:
    """周期性校验经验池索引与磁盘，避免 DB 漂移。"""
    global CURRENT_POOL_SIZE, LAST_RECONCILE_TS
    try:
        disk_records = {}
        for p in EXPERIENCE_DIR.iterdir():
            if p.is_file() and p.name.startswith("exp_") and p.suffix.lower() in {".jpg", ".json"}:
                stat = p.stat()
                disk_records[str(p)] = (int(stat.st_size), float(stat.st_mtime))

        with POOL_DB_LOCK:
            with sqlite3.connect(POOL_DB_FILE) as conn:
                db_rows = conn.execute("SELECT path, size, created_ts FROM files").fetchall()
                db_map = {row[0]: (int(row[1]), float(row[2])) for row in db_rows}

                for path, (size, cts) in disk_records.items():
                    if path not in db_map or db_map[path][0] != size:
                        conn.execute(
                            "INSERT OR REPLACE INTO files(path, size, created_ts) VALUES (?, ?, ?)",
                            (path, size, cts),
                        )
                for path in set(db_map.keys()) - set(disk_records.keys()):
                    conn.execute("DELETE FROM files WHERE path = ?", (path,))
                conn.commit()

        CURRENT_POOL_SIZE = sum(v[0] for v in disk_records.values())
        LAST_RECONCILE_TS = time.time()
        logging.info("经验池索引已对齐磁盘：%d 文件，%.2fGB", len(disk_records), CURRENT_POOL_SIZE / (1024**3))
    except Exception:
        logging.error("经验池索引对齐失败：\n%s", traceback.format_exc())


def add_file_to_pool_index(file_path: Path) -> None:
    global CURRENT_POOL_SIZE
    try:
        stat = file_path.stat()
        size = int(stat.st_size)
        created_ts = float(stat.st_mtime)
        with POOL_DB_LOCK:
            with sqlite3.connect(POOL_DB_FILE) as conn:
                old = conn.execute("SELECT size FROM files WHERE path = ?", (str(file_path),)).fetchone()
                if old:
                    CURRENT_POOL_SIZE -= int(old[0])
                conn.execute(
                    "INSERT OR REPLACE INTO files(path, size, created_ts) VALUES (?, ?, ?)",
                    (str(file_path), size, created_ts),
                )
                conn.commit()
            CURRENT_POOL_SIZE += size
    except Exception:
        logging.error("经验池索引写入失败：%s\n%s", file_path, traceback.format_exc())


def trim_experience_pool() -> None:
    """经验池超过 20GB 时，删除最旧数据直到 <20GB（基于索引队列）。"""
    global CURRENT_POOL_SIZE
    try:
        if CURRENT_POOL_SIZE <= MAX_EXPERIENCE_BYTES:
            return

        logging.warning("经验池大小 %.2fGB > 20GB，开始删除最旧数据...", CURRENT_POOL_SIZE / (1024**3))
        with POOL_DB_LOCK:
            with sqlite3.connect(POOL_DB_FILE) as conn:
                while CURRENT_POOL_SIZE > MAX_EXPERIENCE_BYTES:
                    row = conn.execute("SELECT path, size FROM files ORDER BY created_ts ASC LIMIT 1").fetchone()
                    if not row:
                        break

                    file_path = Path(row[0])
                    file_size = int(row[1])
                    try:
                        if file_path.exists():
                            file_path.unlink()
                    except Exception:
                        logging.error("删除旧经验文件失败：%s\n%s", file_path, traceback.format_exc())

                    conn.execute("DELETE FROM files WHERE path = ?", (str(file_path),))
                    conn.commit()
                    CURRENT_POOL_SIZE -= file_size
                    logging.info("已删除旧经验：%s (%.2fMB)", file_path.name, file_size / (1024**2))

        logging.info("经验池清理完成，当前大小：%.2fGB", CURRENT_POOL_SIZE / (1024**3))
    except Exception:
        logging.error("经验池清理失败：\n%s", traceback.format_exc())


def launch_or_get_firefox_window(force_activate: bool = False):
    """获取 Firefox 窗口，不存在则尝试启动便携版；优先复用缓存 hwnd。"""
    try:
        cached_hwnd = FIREFOX_WINDOW_CACHE.get("hwnd")
        if cached_hwnd:
            for w in gw.getAllWindows():
                if getattr(w, "_hWnd", None) == cached_hwnd:
                    if w.isMinimized:
                        w.restore()
                    if force_activate:
                        w.activate()
                    return w

        wins = gw.getWindowsWithTitle("Mozilla Firefox")
        if not wins:
            if FIREFOX_PATH.exists():
                os.startfile(str(FIREFOX_PATH))
                logging.info("已尝试启动 Firefox Portable：%s", FIREFOX_PATH)
                time.sleep(4)
                wins = gw.getWindowsWithTitle("Mozilla Firefox")
            else:
                raise FileNotFoundError(f"Firefox 路径不存在：{FIREFOX_PATH}")

        if not wins:
            raise RuntimeError("未找到 Firefox 窗口，请确认浏览器已打开且标题含 Mozilla Firefox")

        win = wins[0]
        if win.isMinimized:
            win.restore()
        if force_activate:
            win.activate()
            time.sleep(0.2)
        FIREFOX_WINDOW_CACHE["hwnd"] = getattr(win, "_hWnd", None)
        return win
    except Exception:
        logging.error("浏览器获取失败：\n%s", traceback.format_exc())
        raise


def window_to_rect(win) -> BrowserRect:
    rect = BrowserRect(left=win.left, top=win.top, width=win.width, height=win.height)
    if rect.width <= 10 or rect.height <= 10:
        raise ValueError(f"浏览器窗口尺寸异常：{rect}")
    return rect


def get_safe_rect(rect: BrowserRect) -> Tuple[int, int, int, int]:
    left = max(0, rect.left)
    top = max(0, rect.top)
    right = min(SCREEN_WIDTH, rect.right)
    bottom = min(SCREEN_HEIGHT, rect.bottom)
    return left, top, right, bottom


def grab_browser_snapshot(rect: BrowserRect):
    safe_box = get_safe_rect(rect)
    if safe_box[2] <= safe_box[0] or safe_box[3] <= safe_box[1]:
        raise ValueError("浏览器完全在屏幕外或不可见，无法截图。")

    image = ImageGrab.grab(safe_box)
    sample = image.resize((96, 60)).convert("L")
    pixels = np.asarray(sample, dtype=np.float32).flatten()
    avg = float(np.mean(pixels))
    texture = float(np.mean(np.abs(np.diff(pixels)))) if len(pixels) > 2 else 0.0
    hist_vals, _ = np.histogram(pixels, bins=8, range=(0, 256))
    return image, {
        "brightness": avg,
        "texture": texture,
        "histogram_bins": hist_vals.astype(int).tolist(),
    }


def get_local_region(image, rect: BrowserRect):
    left, top, right, bottom = get_safe_rect(rect)
    abs_mouse_x, abs_mouse_y = pyautogui.position()
    local_left = max(left, abs_mouse_x - LOCAL_REGION_SIZE // 2)
    local_top = max(top, abs_mouse_y - LOCAL_REGION_SIZE // 2)
    local_right = min(right, local_left + LOCAL_REGION_SIZE)
    local_bottom = min(bottom, local_top + LOCAL_REGION_SIZE)
    if local_right <= local_left or local_bottom <= local_top:
        return image, (left, top)

    rel_box = (local_left - left, local_top - top, local_right - left, local_bottom - top)
    local_image = image.crop(rel_box)
    return local_image, (local_left, local_top)


def _normalize_rect(rect: BrowserRect, x: int, y: int) -> Tuple[float, float]:
    safe_left, safe_top, safe_right, safe_bottom = get_safe_rect(rect)
    w = max(1, safe_right - safe_left)
    h = max(1, safe_bottom - safe_top)
    return ((x - safe_left) / w, (y - safe_top) / h)


def get_clickable_elements_by_uia(rect: BrowserRect, hwnd: Optional[int]) -> List[Dict]:
    """使用窗口句柄抓取 UIA 根节点，感知阶段不移动鼠标。"""
    items: List[Dict] = []
    try:
        if not hwnd:
            return items
        focus_control = automation.ControlFromHandle(hwnd)
        if focus_control is None:
            return items

        seen = set()
        queue = [focus_control]
        max_nodes = 220
        while queue and len(seen) < max_nodes:
            ctrl = queue.pop(0)
            key = f"{ctrl.ControlTypeName}:{ctrl.Name}:{id(ctrl)}"
            if key in seen:
                continue
            seen.add(key)

            try:
                br = ctrl.BoundingRectangle
                if not br:
                    continue
                l, t, r, b = int(br.left), int(br.top), int(br.right), int(br.bottom)
                if r <= l or b <= t:
                    continue
                if l < rect.left or t < rect.top or r > rect.right or b > rect.bottom:
                    continue
                ctype = (ctrl.ControlTypeName or "").lower()
                if ctype in {"buttoncontrol", "hyperlinkcontrol", "editcontrol", "menuitemcontrol", "tabitemcontrol"}:
                    cx, cy = (l + r) // 2, (t + b) // 2
                    x_ratio, y_ratio = _normalize_rect(rect, cx, cy)
                    items.append({
                        "name": (ctrl.Name or "").strip(),
                        "type": ctype,
                        "x_ratio": round(x_ratio, 4),
                        "y_ratio": round(y_ratio, 4),
                        "rect": [l, t, r, b],
                    })

                children = ctrl.GetChildren()
                if children:
                    queue.extend(children[:50])
            except Exception:
                logging.error("UIA 子节点解析失败：\n%s", traceback.format_exc())

        return items
    except Exception:
        logging.error("UIA 提取可交互元素失败：\n%s", traceback.format_exc())
        return []


def run_ocr_text_detection(image) -> List[Dict]:
    if OCR_READER is None:
        return []
    try:
        results = OCR_READER.readtext(np.array(image))
        parsed: List[Dict] = []
        for box, text, confidence in results:
            if not text:
                continue
            parsed.append({"text": str(text), "confidence": float(confidence), "box": box})
        return parsed
    except Exception:
        logging.error("OCR 识别失败：\n%s", traceback.format_exc())
        return []


def summarize_scene_with_vlm(image, ocr_results: List[Dict], ui_elements: List[Dict]) -> Dict:
    """VLM理解场景：例如识别登录框、广告关闭按钮等。"""
    ocr_text = " ".join(x.get("text", "") for x in ocr_results).lower()
    labels = []

    if VLM_AVAILABLE and VLM_MODEL is not None and VLM_PROCESSOR is not None:
        try:
            prompt = (
                "请识别当前浏览器页面的主要场景，并给出简短标签，"
                "例如 login_form / ad_popup / article / search_result。只返回逗号分隔标签。"
            )
            inputs = VLM_PROCESSOR(text=prompt, images=image, return_tensors="pt")
            outputs = VLM_MODEL.generate(**inputs, max_new_tokens=24)
            text = VLM_PROCESSOR.batch_decode(outputs, skip_special_tokens=True)[0].lower()
            labels = [x.strip() for x in re.split(r"[,;\n]", text) if x.strip()]
        except Exception:
            logging.error("VLM 推理失败，回退到规则语义：\n%s", traceback.format_exc())

    if not labels:
        if any(k in ocr_text for k in ["登录", "sign in", "log in", "password", "账号", "邮箱"]):
            labels.append("login_form")
        if any(k in ocr_text for k in ["advertisement", "sponsored", "广告", "推广"]):
            labels.append("ad_popup")
        if any(u.get("name", "").lower() in {"close", "关闭", "x"} for u in ui_elements):
            labels.append("has_close_button")
        if not labels:
            labels.append("generic_page")

    return {
        "labels": list(dict.fromkeys(labels))[:6],
        "ocr_text": ocr_text[:600],
    }


def compute_embedding(features: Dict, ui_elements: List[Dict], ocr_results: List[Dict], scene: Dict) -> np.ndarray:
    hist = features.get("histogram_bins", [0] * 8)
    hist = np.asarray(hist, dtype=np.float32)
    hist = hist / max(1.0, np.sum(hist))

    ui_count = float(len(ui_elements))
    button_count = float(sum(1 for e in ui_elements if "button" in e.get("type", "")))
    text_count = float(len(ocr_results))
    scene_hash = float(sum(ord(ch) for lb in scene.get("labels", []) for ch in lb) % 997)

    vector = np.concatenate(
        [
            np.asarray([features.get("brightness", 0.0), features.get("texture", 0.0)], dtype=np.float32) / 255.0,
            np.asarray([ui_count, button_count, text_count, scene_hash / 997.0], dtype=np.float32) / 20.0,
            hist,
        ]
    )
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        vector = vector / norm
    return vector.astype(np.float32)


def list_recent_experiences(limit: int = 240) -> List[Dict]:
    files = sorted(EXPERIENCE_DIR.glob("exp_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    records: List[Dict] = []
    for p in files[:limit]:
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            logging.warning("读取经验文件失败：%s", p)
    return records


def retrieve_similar_experiences(current_embedding: np.ndarray, records: List[Dict], top_k: int = 12) -> List[Dict]:
    scored = []
    for rec in records:
        emb = rec.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        try:
            v = np.asarray(emb, dtype=np.float32)
            denom = float(np.linalg.norm(v) * np.linalg.norm(current_embedding))
            if denom <= 1e-9:
                continue
            score = float(np.dot(v, current_embedding) / denom)
            scored.append((score, rec))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


def page_changed(prev_hist: Optional[List[int]], curr_hist: List[int], threshold: float = 0.08) -> bool:
    if not prev_hist:
        return True
    a = np.asarray(prev_hist, dtype=np.float32)
    b = np.asarray(curr_hist, dtype=np.float32)
    a = a / max(1.0, float(a.sum()))
    b = b / max(1.0, float(b.sum()))
    return float(np.mean(np.abs(a - b))) > threshold


def pick_ocr_click_target(ocr_results: List[Dict], rect: BrowserRect) -> Optional[Dict]:
    keywords = ["确定", "下一步", "同意", "搜索", "关闭", "登录", "继续", "开始"]
    best = None
    best_score = -1.0
    for it in ocr_results:
        txt = str(it.get("text", ""))
        conf = float(it.get("confidence", it.get("conf", 0.0)))
        box = it.get("box", [])
        if len(box) < 4:
            continue
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        cx, cy = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
        x_ratio, y_ratio = _normalize_rect(rect, cx, cy)
        hit = any(k in txt for k in keywords)
        score = conf + (0.35 if hit else 0.0)
        if score > best_score and 0.02 < x_ratio < 0.98 and 0.02 < y_ratio < 0.98:
            best_score = score
            best = {"x_ratio": x_ratio, "y_ratio": y_ratio, "text": txt, "conf": conf, "keyword_hit": hit}
    return best


def penalize_repeated_action(action: Dict) -> Dict:
    if action.get("action") != "click":
        return action
    x, y = float(action.get("x_ratio", 0.5)), float(action.get("y_ratio", 0.5))
    now = time.time()
    repeat = 0
    for rec in LAST_ACTIONS[-8:]:
        if rec.get("action") == "click" and now - float(rec.get("ts", 0.0)) < 8.0:
            if abs(float(rec.get("x_ratio", 0)) - x) < 0.05 and abs(float(rec.get("y_ratio", 0)) - y) < 0.05:
                repeat += 1
    if repeat >= 2:
        action["x_ratio"] = max(0.08, min(0.92, x + random.uniform(-0.15, 0.15)))
        action["y_ratio"] = max(0.08, min(0.92, y + random.uniform(-0.15, 0.15)))
        action["reason"] = f"{action.get('reason','')}（避免重复点击抖动）"
    return action


def choose_action(
    features: Dict,
    similar_records: List[Dict],
    ui_elements: List[Dict],
    ocr_results: List[Dict],
    scene: Dict,
    rect: BrowserRect,
) -> Dict:
    labels = set(scene.get("labels", []))
    ocr_text = scene.get("ocr_text", "")

    if "ad_popup" in labels and ui_elements:
        close_candidates = [e for e in ui_elements if e.get("name", "").lower() in {"close", "关闭", "x"}]
        if close_candidates:
            c = random.choice(close_candidates)
            return {"action": "click", "x_ratio": c["x_ratio"], "y_ratio": c["y_ratio"], "reason": "检测到广告弹窗"}

    if "login_form" in labels:
        input_box = next((e for e in ui_elements if "editcontrol" in e.get("type", "")), None)
        if input_box:
            return {"action": "click", "x_ratio": input_box["x_ratio"], "y_ratio": input_box["y_ratio"], "reason": "登录框聚焦"}

    action_votes: Dict[str, float] = {"click": 0.0, "scroll": 0.0, "type": 0.0}
    for rec in similar_records:
        raw = rec.get("action", {})
        name = raw.get("action") if isinstance(raw, dict) else raw
        reward = float(rec.get("reward", 0.0))
        if isinstance(name, str) and name in action_votes:
            action_votes[name] += max(-0.2, min(1.5, reward + 0.4))

    if similar_records and max(action_votes.values()) > 0.5:
        best = max(action_votes, key=action_votes.get)
        if best == "click" and ui_elements:
            t = random.choice(ui_elements)
            return {"action": "click", "x_ratio": t["x_ratio"], "y_ratio": t["y_ratio"], "reason": "RAG 复用动作"}
        if best == "scroll":
            return {"action": "scroll", "amount": -320, "reason": "RAG 建议滚动"}

    if "captcha" in ocr_text or "验证码" in ocr_text:
        return {"action": "hotkey", "keys": ["f5"], "reason": "验证码页面刷新"}

    if ui_elements and random.random() < 0.82:
        target = random.choice(ui_elements)
        return {"action": "click", "x_ratio": target["x_ratio"], "y_ratio": target["y_ratio"], "reason": "UIA 元素点击"}

    ocr_target = pick_ocr_click_target(ocr_results, rect)
    if ocr_target:
        return {
            "action": "click",
            "x_ratio": round(float(ocr_target["x_ratio"]), 4),
            "y_ratio": round(float(ocr_target["y_ratio"]), 4),
            "reason": f"OCR 兜底点击:{ocr_target['text'][:12]}",
        }

    if features.get("brightness", 120) < 70:
        return {"action": "scroll", "amount": -280, "reason": "页面偏暗尝试滚动"}

    return {"action": "click", "x_ratio": round(random.uniform(0.2, 0.8), 3), "y_ratio": round(random.uniform(0.2, 0.8), 3), "reason": "默认探索"}


def sanitize_text_for_keyboard(text: str) -> str:
    return re.sub(r"escape|esc", "", text, flags=re.IGNORECASE)


def clamp_mouse_to_browser(rect: BrowserRect, x_ratio: float, y_ratio: float) -> Tuple[int, int]:
    left, top, right, bottom = get_safe_rect(rect)
    if right <= left or bottom <= top:
        raise ValueError("浏览器可见区域为空，无法执行鼠标操作。")

    visible_width = right - left
    visible_height = bottom - top

    x = left + int(visible_width * max(0.0, min(1.0, x_ratio)))
    y = top + int(visible_height * max(0.0, min(1.0, y_ratio)))

    x = max(left + 2, min(x, right - 2))
    y = max(top + 2, min(y, bottom - 2))
    return x, y


def human_like_move_to(start_x: int, start_y: int, end_x: int, end_y: int, jitter_px: float = 1.6) -> None:
    distance = math.hypot(end_x - start_x, end_y - start_y)
    duration = random.uniform(0.25, 0.6)
    steps = max(10, int(distance / 24))

    c1x = start_x + (end_x - start_x) * random.uniform(0.2, 0.4) + random.uniform(-40, 40)
    c1y = start_y + (end_y - start_y) * random.uniform(0.2, 0.4) + random.uniform(-30, 30)
    c2x = start_x + (end_x - start_x) * random.uniform(0.6, 0.85) + random.uniform(-40, 40)
    c2y = start_y + (end_y - start_y) * random.uniform(0.6, 0.85) + random.uniform(-30, 30)

    for i in range(1, steps + 1):
        t_raw = i / steps
        t = pytweening.easeInOutQuad(t_raw)
        omt = 1 - t
        x = omt**3 * start_x + 3 * omt**2 * t * c1x + 3 * omt * t**2 * c2x + t**3 * end_x
        y = omt**3 * start_y + 3 * omt**2 * t * c1y + 3 * omt * t**2 * c2y + t**3 * end_y
        if i < steps:
            x += random.uniform(-jitter_px, jitter_px)
            y += random.uniform(-jitter_px, jitter_px)
        pyautogui.moveTo(int(x), int(y), duration=max(0.0015, duration / steps), tween=pytweening.easeInOutQuad)


def execute_action(action: Dict, rect: BrowserRect) -> None:
    name = action.get("action", "wait")
    try:
        if name == "click":
            x, y = clamp_mouse_to_browser(rect, float(action.get("x_ratio", 0.5)), float(action.get("y_ratio", 0.5)))
            cur_x, cur_y = pyautogui.position()
            human_like_move_to(cur_x, cur_y, x, y)
            pyautogui.click()

        elif name == "scroll":
            center_x, center_y = clamp_mouse_to_browser(rect, 0.5, 0.5)
            cur_x, cur_y = pyautogui.position()
            human_like_move_to(cur_x, cur_y, center_x, center_y, jitter_px=1.2)
            pyautogui.scroll(int(action.get("amount", -300)))

        elif name == "type":
            text = sanitize_text_for_keyboard(str(action.get("text", "")))
            if text:
                pyautogui.write(text, interval=0.03)
            if bool(action.get("press_enter", False)):
                pyautogui.press("enter")

        elif name == "hotkey":
            keys = action.get("keys", [])
            if isinstance(keys, list) and keys:
                safe_keys = [str(k).lower() for k in keys if str(k).lower() not in ["esc", "escape"]]
                if safe_keys:
                    if len(safe_keys) == 1:
                        pyautogui.press(safe_keys[0])
                    else:
                        pyautogui.hotkey(*safe_keys)
                else:
                    logging.warning("hotkey 动作包含被禁止按键 ESC，已拦截：%s", keys)

            text = sanitize_text_for_keyboard(str(action.get("text", "")))
            if text:
                pyautogui.write(text, interval=0.03)
            if bool(action.get("press_enter", False)):
                pyautogui.press("enter")

        elif name == "search_new_tab":
            pyautogui.hotkey("ctrl", "t")
            sleep_interruptible(0.06)
            text = sanitize_text_for_keyboard(str(action.get("text", "")))
            if text:
                pyautogui.write(text, interval=0.03)
            pyautogui.press("enter")

        else:
            time.sleep(0.2)

    except Exception:
        logging.error("执行动作失败：%s\n%s", action, traceback.format_exc())


def estimate_reward(next_features: Dict, previous_features: Dict, action: Dict, changed: bool) -> float:
    reward = 0.0
    reward += min(0.6, abs(next_features.get("brightness", 0) - previous_features.get("brightness", 0)) / 255.0)
    reward += min(0.6, abs(next_features.get("texture", 0) - previous_features.get("texture", 0)) / 80.0)
    if action.get("action") == "click":
        reward += 0.15
    if not changed:
        reward -= 0.28
    return round(float(reward - 0.15), 4)


def save_experience(
    image,
    action: Dict,
    features: Dict,
    rect: BrowserRect,
    embedding: np.ndarray,
    scene: Dict,
    ui_elements: List[Dict],
    reward: float,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_file = EXPERIENCE_DIR / f"exp_{ts}.jpg"
    json_file = EXPERIENCE_DIR / f"exp_{ts}.json"

    record = {
        "time": ts,
        "screen": {"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT},
        "browser_rect": rect.__dict__,
        "action": action,
        "features": features,
        "scene": scene,
        "ui_elements": ui_elements[:80],
        "embedding": embedding.astype(float).tolist(),
        "reward": reward,
    }

    try:
        image.save(img_file, format="JPEG", quality=76)
        json_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        add_file_to_pool_index(img_file)
        add_file_to_pool_index(json_file)
        if CURRENT_POOL_SIZE > MAX_EXPERIENCE_BYTES:
            async_trim_experience_pool()
    except Exception:
        logging.error("保存经验失败：\n%s", traceback.format_exc())


def esc_pressed() -> bool:
    return STOP_EVENT.is_set()


def on_esc_press() -> None:
    if STOP_EVENT.is_set():
        return
    STOP_EVENT.set()
    logging.error("检测到 ESC，准备立即终止（优先安全退出，1秒后兜底硬退出）。")

    def force_exit() -> None:
        time.sleep(1.0)
        if STOP_EVENT.is_set():
            logging.error("主线程未在1秒内结束，执行兜底硬退出。")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

    threading.Thread(target=force_exit, daemon=True, name="esc-force-exit").start()


def setup_esc_kill_switch() -> None:
    try:
        keyboard.add_hotkey("esc", on_esc_press, suppress=False, trigger_on_release=False)
        logging.info("ESC 终止热键已启用（即时生效）。")
    except Exception:
        logging.error("ESC 热键注册失败：\n%s", traceback.format_exc())
        raise


def sleep_interruptible(seconds: float) -> None:
    end = time.time() + max(0.0, seconds)
    while not STOP_EVENT.is_set():
        remain = end - time.time()
        if remain <= 0:
            return
        time.sleep(min(0.03, remain))


def async_trim_experience_pool() -> None:
    def worker() -> None:
        if not TRIM_LOCK.acquire(blocking=False):
            logging.info("经验池清理任务正在进行，跳过本轮触发。")
            return
        try:
            trim_experience_pool()
        finally:
            TRIM_LOCK.release()

    thread = threading.Thread(target=worker, daemon=True, name="exp-trim-worker")
    thread.start()


def main() -> None:
    logging.info("=" * 64)
    logging.info("浏览器 AI 启动（单文件版：VLM + UIA + RAG + 自愈索引）")
    logging.info("按 ESC 终止程序")
    logging.info("=" * 64)

    refresh_screen_size()
    ensure_package("nvidia_smi", "nvidia-ml-py")
    ensure_init_files()
    init_experience_pool_index()
    reconcile_experience_pool_index()
    init_ocr()
    init_vlm()
    setup_esc_kill_switch()
    browser = launch_or_get_firefox_window(force_activate=True)

    loop_count = 0
    prev_features = {"brightness": 0.0, "texture": 0.0, "histogram_bins": [0] * 8}
    cached_ocr: List[Dict] = []

    while True:
        if esc_pressed():
            logging.info("检测到 ESC，程序终止。")
            break

        try:
            loop_start = time.perf_counter()
            refresh_screen_size()
            browser = launch_or_get_firefox_window(force_activate=False)
            rect = window_to_rect(browser)
            screenshot, features = grab_browser_snapshot(rect)
            local_region, _ = get_local_region(screenshot, rect)

            changed = page_changed(prev_features.get("histogram_bins"), features.get("histogram_bins", []), threshold=0.06)
            if changed or not cached_ocr:
                full_scan = screenshot.resize((max(320, screenshot.width // 2), max(220, screenshot.height // 2)))
                ocr_global = run_ocr_text_detection(full_scan)
                ocr_local = run_ocr_text_detection(local_region)
                cached_ocr = ocr_global + ocr_local
            ocr_results = cached_ocr

            hwnd = getattr(browser, "_hWnd", None)
            ui_elements = get_clickable_elements_by_uia(rect, hwnd)
            scene = summarize_scene_with_vlm(local_region, ocr_results, ui_elements)
            features["ocr_text_count"] = len(ocr_results)
            features["ui_count"] = len(ui_elements)
            features["scene_labels"] = scene.get("labels", [])

            embedding = compute_embedding(features, ui_elements, ocr_results, scene)
            experiences = list_recent_experiences(limit=240)
            similar_records = retrieve_similar_experiences(embedding, experiences, top_k=10)
            action = choose_action(features, similar_records, ui_elements, ocr_results, scene, rect)
            action = penalize_repeated_action(action)

            logging.info(
                "第 %s 轮: scene=%s, ui=%s, ocr=%s, changed=%s, action=%s",
                loop_count,
                scene.get("labels"),
                len(ui_elements),
                len(ocr_results),
                changed,
                action,
            )

            execute_action(action, rect)
            LAST_ACTIONS.append({
                "ts": time.time(),
                "action": action.get("action"),
                "x_ratio": action.get("x_ratio", 0.0),
                "y_ratio": action.get("y_ratio", 0.0),
            })
            if len(LAST_ACTIONS) > 64:
                del LAST_ACTIONS[:-64]

            reward = estimate_reward(features, prev_features, action, changed)
            save_experience(screenshot, action, features, rect, embedding, scene, ui_elements, reward)

            prev_features = features
            loop_count += 1
            if loop_count % 90 == 0:
                async_trim_experience_pool()
            if time.time() - LAST_RECONCILE_TS > RECONCILE_INTERVAL_SECONDS:
                reconcile_experience_pool_index()

            loop_spent = time.perf_counter() - loop_start
            target_sleep = max(0.05, 0.22 - min(0.12, loop_spent / 6.0))
            sleep_interruptible(target_sleep)

        except Exception:
            logging.error("主循环异常：\n%s", traceback.format_exc())
            sleep_interruptible(0.35)

    logging.info("程序已安全退出。")


if __name__ == "__main__":
    main()
