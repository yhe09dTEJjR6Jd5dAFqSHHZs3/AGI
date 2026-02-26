import os
import sys
import time
import json
import glob
import logging
import traceback
import warnings
import subprocess
import importlib
from importlib import metadata
from datetime import datetime

# ==========================================
# 1. 消除特定的警告信息
# ==========================================
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*GradScaler.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_reentrant.*")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # 屏蔽非必要的 C++ 底层警告

# 配置详细日志输出
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

AUTO_FIX_DEPS = os.environ.get("AGI_AUTO_FIX_DEPS", "1") == "1"
MAX_REPAIR_RESTARTS = 2
REPAIR_RESTART_COUNT = int(os.environ.get("AGI_REPAIR_RESTART_COUNT", "0"))
FORCE_REPAIR_ON_START = os.environ.get("AGI_FORCE_REPAIR_ON_START", "0") == "1"
MIN_TRANSFORMERS = "4.45.0"
MIN_TOKENIZERS = "0.20.0"


def _pip_install(requirements, force_reinstall=False):
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + requirements
    if force_reinstall:
        cmd.extend(["--force-reinstall", "--no-cache-dir"])
    logging.warning(f"检测到依赖问题，尝试自动修复: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error("❌ 自动修复依赖失败。pip stdout/stderr 如下：")
        logging.error(result.stdout.strip())
        logging.error(result.stderr.strip())
        raise RuntimeError("自动修复依赖失败")
    logging.info("✅ 依赖库自动修复完成。")


def _restart_current_process(force_repair=False):
    if REPAIR_RESTART_COUNT >= MAX_REPAIR_RESTARTS:
        raise RuntimeError("依赖修复重启次数已达上限，请手动检查 Python 环境。")

    os.environ["AGI_REPAIR_RESTART_COUNT"] = str(REPAIR_RESTART_COUNT + 1)
    if force_repair:
        os.environ["AGI_FORCE_REPAIR_ON_START"] = "1"
    logging.warning("准备重启当前 Python 进程以应用依赖修复...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


def _repair_dependencies_before_import():
    if not FORCE_REPAIR_ON_START:
        return

    logging.info("检测到重启修复标记，先在全新进程中重装关键依赖...")
    _pip_install([
        "pip",
        f"transformers>={MIN_TRANSFORMERS}",
        f"tokenizers>={MIN_TOKENIZERS}",
        "accelerate",
        "qwen-vl-utils",
        "bitsandbytes",
        "torchvision",
    ], force_reinstall=True)
    os.environ["AGI_FORCE_REPAIR_ON_START"] = "0"
    _restart_current_process(force_repair=False)


_repair_dependencies_before_import()

# ==========================================
# 2. 全局异常处理，绝对避免静默失败
# ==========================================
def global_exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("❌ 程序发生未捕获的致命错误！详细信息如下：")
    logging.error("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    
sys.excepthook = global_exception_handler

# ==========================================
# 3. 依赖库严格自检与深度导入
# ==========================================
logging.info("正在执行依赖库深度检查...")
try:
    # 基础自动化库
    import pyautogui
    import pygetwindow as gw
    import keyboard
    from PIL import ImageGrab, Image
    import psutil

    # 核心 AI 库深度检查 (剥离 HF 延迟加载的模糊报错)
    import torch
    import torchvision
    import bitsandbytes # 4-bit 量化必须
    import accelerate
    from packaging import version

    def _get_installed_version(pkg_name):
        try:
            return metadata.version(pkg_name)
        except metadata.PackageNotFoundError:
            return None

    def _purge_modules(prefixes):
        removed = []
        for module_name in list(sys.modules.keys()):
            if any(module_name == p or module_name.startswith(f"{p}.") for p in prefixes):
                removed.append(module_name)
                sys.modules.pop(module_name, None)
        if removed:
            logging.info(f"已清理模块缓存: {', '.join(sorted(removed)[:8])} ... 共 {len(removed)} 个")

    def _cleanup_tokenizers_shadowing():
        """清理常见的 tokenizers/decoders 命名冲突与损坏缓存。"""
        import site

        candidates = []
        for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
            candidates.extend(glob.glob(os.path.join(site_dir, "decoders*")))
            candidates.extend(glob.glob(os.path.join(site_dir, "tokenizers*")))

        for item in sorted(set(candidates)):
            base = os.path.basename(item).lower()
            # 第三方 decoders 包会污染 `from tokenizers import decoders`
            should_remove = (
                base == "decoders" or base.startswith("decoders-")
                or base.startswith("tokenizers-") or base == "tokenizers"
            )
            if not should_remove:
                continue

            try:
                if os.path.isdir(item):
                    import shutil
                    shutil.rmtree(item, ignore_errors=False)
                else:
                    os.remove(item)
                logging.warning(f"已清理可能冲突/损坏的包缓存: {item}")
            except Exception as cleanup_error:
                logging.warning(f"清理冲突文件失败（可忽略，后续会强制重装）: {item} -> {cleanup_error}")

    def _import_transformers_stack():
        import transformers as _transformers

        if version.parse(_transformers.__version__) < version.parse(MIN_TRANSFORMERS):
            raise ImportError(
                f"transformers 版本过低 ({_transformers.__version__})，需要 >= {MIN_TRANSFORMERS}"
            )

        # 某些环境会出现 tokenizers/decoders ABI 不匹配导致 DecodeStream 丢失
        tokenizers_version = _get_installed_version("tokenizers")
        if tokenizers_version is None:
            raise ImportError("未检测到 tokenizers 包，请先安装。")
        if version.parse(tokenizers_version) < version.parse(MIN_TOKENIZERS):
            raise ImportError(
                f"tokenizers 版本过低 ({tokenizers_version})，需要 >= {MIN_TOKENIZERS}"
            )

        try:
            from tokenizers import decoders as _tokenizer_decoders
            if not hasattr(_tokenizer_decoders, "DecodeStream"):
                raise ImportError("tokenizers.decoders 缺少 DecodeStream，疑似安装损坏")
            decoders_file = getattr(_tokenizer_decoders, "__file__", "")
            if "site-packages" in decoders_file and "tokenizers" not in decoders_file.lower():
                raise ImportError(f"tokenizers.decoders 指向了异常路径: {decoders_file}")
        except Exception as tokenizer_error:
            raise ImportError(
                f"tokenizers 导入异常（常见于包损坏/版本冲突）: {tokenizer_error}"
            ) from tokenizer_error

        from transformers import Qwen2VLForConditionalGeneration as _Qwen2VLForConditionalGeneration
        from transformers import AutoProcessor as _AutoProcessor

        return _transformers, _Qwen2VLForConditionalGeneration, _AutoProcessor

    try:
        transformers, Qwen2VLForConditionalGeneration, AutoProcessor = _import_transformers_stack()
    except Exception as first_error:
        logging.error("首次导入 transformers/Qwen2-VL 失败，详细原因：")
        logging.error(traceback.format_exc())
        if not AUTO_FIX_DEPS:
            raise first_error
        _purge_modules(["tokenizers", "transformers", "decoders"])
        _cleanup_tokenizers_shadowing()
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "decoders"], capture_output=True, text=True)
        logging.warning("检测到可能的 tokenizers ABI/文件锁问题，将重启到干净进程后再重装依赖。")
        _restart_current_process(force_repair=True)

    from qwen_vl_utils import process_vision_info

    logging.info("✅ 所有依赖库检查通过！")

except ImportError as e:
    logging.error("❌ 依赖库导入失败！")
    logging.error(f"直接原因: {e}")
    logging.error("请确保执行了以下命令：")
    logging.error("pip install --upgrade transformers tokenizers accelerate qwen-vl-utils bitsandbytes torchvision")
    logging.error("完整报错堆栈如下：")
    traceback.print_exc()
    sys.exit(1)

# PyAutoGUI 安全设置
pyautogui.FAILSAFE = False

# ==========================================
# 4. 初始化参数与路径
# ==========================================
DESKTOP_PATH = os.path.join(os.environ['USERPROFILE'], 'Desktop')
AAA_DIR = os.path.join(DESKTOP_PATH, 'AAA')
MODEL_DIR = os.path.join(AAA_DIR, 'Model')
EXP_DIR = os.path.join(AAA_DIR, 'Experience Pool')
FIREFOX_PATH = r"E:\FirefoxPortable\FirefoxPortable.exe"
MAX_EXP_SIZE = 20 * 1024 * 1024 * 1024  # 20 GB

# ==========================================
# 5. 初始化阶段：检查与创建文件夹
# ==========================================
def init_environment():
    logging.info("开始初始化环境...")
    for directory in [AAA_DIR, MODEL_DIR, EXP_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"已自动生成缺失的文件夹: {directory}")
        else:
            logging.info(f"检查文件夹存在: {directory}")
            
# ==========================================
# 6. 经验池管理 (20GB 限制，丢弃旧数据)
# ==========================================
def get_dir_size(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def manage_experience_pool():
    size = get_dir_size(EXP_DIR)
    if size > MAX_EXP_SIZE:
        logging.info(f"⚠️ 经验池大小 ({size / 1024**3:.2f}GB) 超过 20GB，开始清理最旧数据...")
        files = []
        for f in os.listdir(EXP_DIR):
            fp = os.path.join(EXP_DIR, f)
            if os.path.isfile(fp):
                files.append((fp, os.path.getmtime(fp), os.path.getsize(fp)))
        
        # 按修改时间从旧到新排序
        files.sort(key=lambda x: x[1])
        
        for fp, _, fsize in files:
            try:
                os.remove(fp)
                size -= fsize
                if size < MAX_EXP_SIZE:
                    logging.info("✅ 清理完毕，经验池已恢复到 20GB 以下。")
                    break
            except Exception as e:
                logging.warning(f"无法删除文件 {fp}: {e}")

def save_experience(image, action_dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = os.path.join(EXP_DIR, f"exp_{timestamp}.jpg")
    json_path = os.path.join(EXP_DIR, f"exp_{timestamp}.json")
    
    # 保存压缩图片以节省空间
    image.save(img_path, format='JPEG', quality=80)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(action_dict, f, ensure_ascii=False, indent=2)
    manage_experience_pool()

# ==========================================
# 7. 浏览器获取与控制
# ==========================================
def launch_and_get_browser():
    logging.info("检查并启动 Firefox 浏览器...")
    firefox_windows = gw.getWindowsWithTitle('Mozilla Firefox')
    
    if not firefox_windows:
        if os.path.exists(FIREFOX_PATH):
            subprocess.Popen(FIREFOX_PATH)
            logging.info("已发送启动 Firefox 的指令，等待 5 秒窗口出现...")
            time.sleep(5)
            firefox_windows = gw.getWindowsWithTitle('Mozilla Firefox')
        else:
            raise FileNotFoundError(f"找不到指定的浏览器文件: {FIREFOX_PATH}")
    
    if not firefox_windows:
        raise RuntimeError("无法找到 Firefox 窗口，请确保它已正确启动且未被杀毒软件拦截。")
    
    win = firefox_windows[0]
    if win.isMinimized:
        win.restore()
    win.activate()
    return win

# ==========================================
# 8. AI 模型加载 (专为 4GB VRAM 优化)
# ==========================================
def load_ai_model():
    logging.info("正在加载 AI 模型 (Qwen2-VL-2B-Instruct)...")
    logging.info("⏳ 你的 GPU 显存为 4GB，正在启用 4-bit 极限压缩量化...")
    
    os.environ['HF_HOME'] = MODEL_DIR
    
    try:
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
        processor = AutoProcessor.from_pretrained(model_id)
        logging.info("✅ AI 模型加载成功！显存已驻留。")
        return model, processor
    except Exception as e:
        logging.error("❌ 模型加载失败！")
        logging.error(traceback.format_exc())
        raise e

# ==========================================
# 9. 核心 AI 决策逻辑
# ==========================================
def think_and_act(model, processor, browser_win):
    bbox = (browser_win.left, browser_win.top, browser_win.right, browser_win.bottom)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        logging.warning("浏览器窗口尺寸异常或已最小化，跳过此次操作。")
        time.sleep(1)
        return

    screenshot = ImageGrab.grab(bbox)
    
    # 动态缩小分辨率，防止推理时显存溢出 (4GB 显存极其紧张)
    max_dim = 800 
    screenshot.thumbnail((max_dim, max_dim))

    prompt_text = """你是一个生存在浏览器里的 AI 智能体。观察上面的网页截图。
请思考下一步应该做什么来探索网页。
你必须且只能输出一个 JSON 字典。
可用的 action:
1. "click": 点击。提供 "x_ratio" 和 "y_ratio" (0.0~1.0 相对坐标)。
2. "type": 输入。提供 "text" 字段。
3. "scroll": 滚动。提供 "amount" (负数向下，正数向上)。
4. "wait": 等待观察。

示例：
{"action": "click", "x_ratio": 0.5, "y_ratio": 0.2, "reason": "点击搜索框"}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": screenshot},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    try:
        # 预处理
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 推理 (限制 max_new_tokens 降低显存压力)
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    except Exception as e:
        logging.error(f"❌ AI 推理时爆显存或出错: {e}")
        logging.error(traceback.format_exc())
        return
    finally:
        # 极度暴力的显存清理，防止 4GB 显卡 OOM 崩溃
        if 'inputs' in locals(): del inputs
        if 'generated_ids' in locals(): del generated_ids
        torch.cuda.empty_cache()

    # 解析 JSON
    try:
        if "```json" in output_text:
            output_text = output_text.split("```json")[1].split("```")[0].strip()
        elif "```" in output_text:
            output_text = output_text.split("```")[1].split("```")[0].strip()

        action_dict = json.loads(output_text)
        logging.info(f"🧠 AI 决定执行: {action_dict}")
        
        execute_action(action_dict, browser_win)
        save_experience(screenshot, action_dict)
        
    except json.JSONDecodeError:
        logging.warning(f"⚠️ AI 输出了非 JSON 格式，跳过。原始输出: {output_text}")
    except Exception as e:
        logging.error(f"❌ 执行动作时出错: {e}")
        logging.error(traceback.format_exc())

def execute_action(action_dict, browser_win):
    action = action_dict.get("action")
    
    if action == "click":
        x_ratio = float(action_dict.get("x_ratio", 0.5))
        y_ratio = float(action_dict.get("y_ratio", 0.5))
        
        # 严格的坐标越界保护
        target_x = browser_win.left + int(browser_win.width * x_ratio)
        target_y = browser_win.top + int(browser_win.height * y_ratio)
        target_x = max(browser_win.left + 5, min(target_x, browser_win.right - 5))
        target_y = max(browser_win.top + 5, min(target_y, browser_win.bottom - 5))
        
        pyautogui.moveTo(target_x, target_y, duration=0.4)
        pyautogui.click()
        
    elif action == "type":
        text = action_dict.get("text", "")
        # 严格过滤 ESC 键
        text_safe = text.replace("esc", "").replace("Escape", "")
        pyautogui.write(text_safe, interval=0.05)
        if "\n" in text:
            pyautogui.press('enter')
            
    elif action == "scroll":
        amount = int(action_dict.get("amount", -300))
        pyautogui.moveTo(
            browser_win.left + browser_win.width // 2,
            browser_win.top + browser_win.height // 2
        )
        pyautogui.scroll(amount)
        
    elif action == "wait":
        time.sleep(2)
        
    else:
        logging.warning(f"⚠️ 遇到未知动作: {action}")

# ==========================================
# 10. 主程序生命周期
# ==========================================
def main():
    logging.info("="*55)
    logging.info(" 🤖 浏览器 AI 智能体 (4GB VRAM 优化版) - 启动序列 ")
    logging.info(" 🛡️  随时长按 [ESC] 键终止程序 ")
    logging.info("="*55)
    
    init_environment()
    browser_win = launch_and_get_browser()
    model, processor = load_ai_model()
    
    logging.info("🚀 系统就绪，AI 已接管控制权...")
    time.sleep(2)

    try:
        while True:
            # 监听全局退出指令
            if keyboard.is_pressed('esc'):
                logging.info("🛑 检测到用户按下 [ESC] 键，正在安全终止程序...")
                break
                
            try:
                browser_win = gw.getWindowsWithTitle('Mozilla Firefox')[0]
                browser_win.activate()
            except IndexError:
                logging.error("❌ 浏览器窗口被关闭！AI 失去宿主环境，即将退出。")
                break
                
            think_and_act(model, processor, browser_win)
            
            # 给 3050Ti 显卡一点“喘息”的时间，防止过热和驱动假死
            time.sleep(2)
            
    except Exception as e:
        logging.error("❌ 主循环发生未预期的崩溃！")
        logging.error(traceback.format_exc())
    finally:
        logging.info("🧹 正在释放显存并关闭程序...")
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        logging.info("✅ 程序已完全退出。经验池保存在 Desktop/AAA/Experience Pool。")

if __name__ == "__main__":
    main()
