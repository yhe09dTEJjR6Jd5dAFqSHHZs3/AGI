import json
import logging
import os
import random
import re
import sys
import time
import traceback
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =========================
# 全局告警处理（按需求预防）
# =========================
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*pynvml package is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp\.GradScaler.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*torch\.utils\.checkpoint.*use_reentrant.*")

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

SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1600


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
    import keyboard
    import pyautogui
    import pygetwindow as gw
    from PIL import ImageGrab
except Exception:
    logging.error("依赖导入失败，详细错误如下：")
    logging.error(traceback.format_exc())
    raise


pyautogui.FAILSAFE = False


def ensure_init_files() -> None:
    """初始化阶段：严格检查 AAA/AI模型/经验池，缺失即创建（不联网下载模型）。"""
    try:
        AAA_DIR.mkdir(parents=True, exist_ok=True)
        EXPERIENCE_DIR.mkdir(parents=True, exist_ok=True)

        if not MODEL_FILE.exists():
            base_model = {
                "name": "LocalBrowserAI",
                "version": "2.0-offline",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "policy": {
                    "no_network_model_download": True,
                    "allow_keyboard_keys_except_esc": True,
                    "only_mouse_inside_browser": True,
                },
                "knowledge": {
                    "goals": ["探索页面", "优先点击可交互区域", "避免无效重复动作"],
                    "safe_actions": ["click", "scroll", "type", "wait"],
                },
            }
            MODEL_FILE.write_text(json.dumps(base_model, ensure_ascii=False, indent=2), encoding="utf-8")
            logging.info("已创建本地 AI 模型文件（离线，不联网下载）：%s", MODEL_FILE)
        else:
            logging.info("检测到本地 AI 模型文件：%s", MODEL_FILE)

        logging.info("检测到经验池目录：%s", EXPERIENCE_DIR)
    except Exception:
        logging.error("初始化阶段失败，详细错误：\n%s", traceback.format_exc())
        raise


def experience_size_bytes() -> int:
    total = 0
    for root, _, files in os.walk(EXPERIENCE_DIR):
        for f in files:
            p = Path(root) / f
            if p.is_file():
                total += p.stat().st_size
    return total


def trim_experience_pool() -> None:
    """经验池超过 20GB 时，删除最旧数据直到 <20GB。"""
    try:
        total = experience_size_bytes()
        if total <= MAX_EXPERIENCE_BYTES:
            return

        logging.warning("经验池大小 %.2fGB > 20GB，开始删除最旧数据...", total / (1024 ** 3))
        files: List[Path] = [p for p in EXPERIENCE_DIR.iterdir() if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime)

        for p in files:
            if total < MAX_EXPERIENCE_BYTES:
                break
            size = p.stat().st_size
            p.unlink(missing_ok=False)
            total -= size
            logging.info("已删除旧经验：%s (%.2fMB)", p.name, size / (1024 ** 2))

        logging.info("经验池清理完成，当前大小：%.2fGB", total / (1024 ** 3))
    except Exception:
        logging.error("经验池清理失败：\n%s", traceback.format_exc())


def launch_or_get_firefox_window():
    """获取 Firefox 窗口，不存在则尝试启动便携版。"""
    try:
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
        win.activate()
        time.sleep(0.2)
        return win
    except Exception:
        logging.error("浏览器获取失败：\n%s", traceback.format_exc())
        raise


def window_to_rect(win) -> BrowserRect:
    rect = BrowserRect(left=win.left, top=win.top, width=win.width, height=win.height)
    if rect.width <= 10 or rect.height <= 10:
        raise ValueError(f"浏览器窗口尺寸异常：{rect}")
    return rect


def grab_browser_snapshot(rect: BrowserRect):
    """抓取浏览器区域截图并提取轻量特征。"""
    image = ImageGrab.grab((rect.left, rect.top, rect.right, rect.bottom))
    sample = image.resize((96, 60)).convert("L")
    pixels = list(sample.getdata())
    avg = sum(pixels) / len(pixels)

    hist = Counter(int(p // 32) for p in pixels)
    texture = sum(abs(pixels[i] - pixels[i - 1]) for i in range(1, len(pixels))) / len(pixels)
    return image, {
        "brightness": avg,
        "texture": texture,
        "histogram_bins": dict(hist),
    }


def list_recent_experiences(limit: int = 120) -> List[Dict]:
    files = sorted(EXPERIENCE_DIR.glob("exp_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    records: List[Dict] = []
    for p in files[:limit]:
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            logging.warning("读取经验文件失败：%s", p)
    return records


def choose_action(features: Dict, recent_records: List[Dict]) -> Dict:
    """离线“模型”：结合视觉特征 + 经验池做策略决策。"""
    recent_actions = [r.get("action") for r in recent_records if isinstance(r, dict)]
    action_counts = Counter(a for a in recent_actions if a)

    brightness = features["brightness"]
    texture = features["texture"]

    # 避免陷入重复动作
    last_three = recent_actions[:3]
    repeated_scroll = len(last_three) == 3 and all(a == "scroll" for a in last_three)
    repeated_click = len(last_three) == 3 and all(a == "click" for a in last_three)

    if repeated_scroll:
        return {"action": "click", "x_ratio": round(random.uniform(0.35, 0.65), 3), "y_ratio": 0.18, "reason": "连续滚动后尝试激活顶部交互区"}
    if repeated_click:
        return {"action": "scroll", "amount": -420, "reason": "连续点击无效，切换滚动探索"}

    # 根据页面亮度/纹理做启发式判断
    if brightness < 60:
        return {"action": "scroll", "amount": -360, "reason": "页面偏暗，向下探索更多内容"}

    if texture > 28 and action_counts.get("click", 0) <= action_counts.get("scroll", 0):
        return {
            "action": "click",
            "x_ratio": round(random.uniform(0.2, 0.8), 3),
            "y_ratio": round(random.uniform(0.15, 0.75), 3),
            "reason": "页面信息密度较高，优先点击潜在交互元素",
        }

    # 低纹理页面：尝试输入搜索词（禁止 ESC）
    if texture < 15 and random.random() < 0.25:
        terms = ["最新消息", "教程", "AI", "帮助", "示例"]
        return {"action": "type", "text": random.choice(terms), "press_enter": True, "reason": "页面较空，尝试输入关键词"}

    if random.random() < 0.5:
        return {"action": "scroll", "amount": -random.randint(250, 520), "reason": "常规探索"}

    return {
        "action": "click",
        "x_ratio": round(random.uniform(0.15, 0.85), 3),
        "y_ratio": round(random.uniform(0.18, 0.82), 3),
        "reason": "默认交互探索",
    }


def sanitize_text_for_keyboard(text: str) -> str:
    # 严格防止 ESC：过滤 esc/escape（各种大小写）
    clean = re.sub(r"escape|esc", "", text, flags=re.IGNORECASE)
    return clean


def clamp_mouse_to_browser(rect: BrowserRect, x_ratio: float, y_ratio: float) -> Tuple[int, int]:
    x = rect.left + int(rect.width * max(0.0, min(1.0, x_ratio)))
    y = rect.top + int(rect.height * max(0.0, min(1.0, y_ratio)))

    x = max(rect.left + 2, min(x, rect.right - 2))
    y = max(rect.top + 2, min(y, rect.bottom - 2))
    return x, y


def execute_action(action: Dict, rect: BrowserRect) -> None:
    name = action.get("action", "wait")
    try:
        if name == "click":
            x, y = clamp_mouse_to_browser(rect, float(action.get("x_ratio", 0.5)), float(action.get("y_ratio", 0.5)))
            pyautogui.moveTo(x, y, duration=0.25)
            pyautogui.click()

        elif name == "scroll":
            center_x, center_y = clamp_mouse_to_browser(rect, 0.5, 0.5)
            pyautogui.moveTo(center_x, center_y, duration=0.15)
            pyautogui.scroll(int(action.get("amount", -320)))

        elif name == "type":
            text = sanitize_text_for_keyboard(str(action.get("text", "")))
            if text:
                pyautogui.write(text, interval=0.03)
            if bool(action.get("press_enter", False)):
                pyautogui.press("enter")

        else:
            time.sleep(0.5)

    except Exception:
        logging.error("执行动作失败：%s\n%s", action, traceback.format_exc())


def save_experience(image, action: Dict, features: Dict, rect: BrowserRect) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_file = EXPERIENCE_DIR / f"exp_{ts}.jpg"
    json_file = EXPERIENCE_DIR / f"exp_{ts}.json"

    record = {
        "time": ts,
        "screen": {"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT},
        "browser_rect": rect.__dict__,
        "action": action,
        "features": features,
    }

    try:
        image.save(img_file, format="JPEG", quality=78)
        json_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logging.error("保存经验失败：\n%s", traceback.format_exc())

    trim_experience_pool()


def esc_pressed() -> bool:
    try:
        return keyboard.is_pressed("esc")
    except Exception:
        logging.error("ESC 检测异常：\n%s", traceback.format_exc())
        return False


def main() -> None:
    logging.info("=" * 60)
    logging.info("浏览器 AI 启动（单文件离线版）")
    logging.info("按 ESC 终止程序")
    logging.info("=" * 60)

    ensure_init_files()
    browser = launch_or_get_firefox_window()

    loop_count = 0
    while True:
        if esc_pressed():
            logging.info("检测到 ESC，程序终止。")
            break

        try:
            browser = launch_or_get_firefox_window()
            rect = window_to_rect(browser)
            screenshot, features = grab_browser_snapshot(rect)

            experiences = list_recent_experiences(limit=160)
            action = choose_action(features, experiences)

            logging.info("第 %s 轮决策：%s", loop_count, action)
            execute_action(action, rect)
            save_experience(screenshot, action, features, rect)

            loop_count += 1
            time.sleep(0.8)

        except Exception:
            logging.error("主循环异常：\n%s", traceback.format_exc())
            time.sleep(1.2)

    logging.info("程序已安全退出。")


if __name__ == "__main__":
    main()
