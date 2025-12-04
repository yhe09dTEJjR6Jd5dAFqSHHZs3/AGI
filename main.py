import os
import sys
import time
import json
import uuid
import glob
import ctypes
import pickle
import random
import threading
import webbrowser
import warnings
import logging
import importlib
import importlib.util
import numpy as np
import cv2
import mss
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import pyautogui
from pynput import mouse, keyboard
from flask import Flask, request, jsonify

warnings.filterwarnings("ignore")
logging.getLogger('werkzeug').setLevel(logging.ERROR)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

def panic(msg):
    ctypes.windll.user32.MessageBoxW(0, str(msg), "Error", 0x10)
    os._exit(1)

nvml_module = None
if importlib.util.find_spec("nvidia_ml_py"):
    nvml_module = importlib.import_module("nvidia_ml_py")
elif importlib.util.find_spec("pynvml"):
    nvml_module = importlib.import_module("pynvml")
gpu_handle = None
if nvml_module:
    try:
        nvml_module.nvmlInit()
        gpu_handle = nvml_module.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        nvml_module = None
        gpu_handle = None

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
root_dir = os.path.join(desktop_path, "AAA")
exp_pool_dir = os.path.join(root_dir, "ExperiencePool")
model_dir = os.path.join(root_dir, "Models")

for d in [root_dir, exp_pool_dir, model_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

app = Flask(__name__)

class SharedState:
    def __init__(self):
        self.mode = "IDLE"
        self.prev_mode = "IDLE"
        self.is_running = True
        self.fps = 30
        self.seq_len = 10
        self.buffer = []
        self.mouse_state = {"pressed": False, "start_pos": None, "start_time": None}
        self.current_episode_id = None
        self.last_screen = None
        self.resource_stats = {"cpu": 0, "mem": 0, "gpu": 0, "vram": 0, "fps": 30, "seq": 10}
        self.optimization_progress = 0
        self.optimization_status = "就绪"
        self.lock = threading.Lock()
        self.esc_pressed = False
        self.window_handle = None
        self.is_boosting = False
        self.ai_button_down = False

state = SharedState()

class AIModel(nn.Module):
    def __init__(self):
        super(AIModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(32 * 5 * 5, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc_mouse_x = nn.Linear(128, 1)
        self.fc_mouse_y = nn.Linear(128, 1)
        self.fc_action = nn.Linear(128, 3)
        self.fc_result = nn.Linear(128, 3)

    def forward(self, x, hidden=None):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(batch_size * seq_len, -1)
        x = torch.relu(self.fc1(x))
        x = x.view(batch_size, seq_len, -1)
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]
        mx = torch.sigmoid(self.fc_mouse_x(out))
        my = torch.sigmoid(self.fc_mouse_y(out))
        act = torch.softmax(self.fc_action(out), dim=1)
        res = torch.softmax(self.fc_result(out), dim=1)
        return mx, my, act, res, hidden

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AIModel().to(device)
if os.path.exists(os.path.join(model_dir, "latest.pth")):
    try:
        model.load_state_dict(torch.load(os.path.join(model_dir, "latest.pth")))
    except: pass
optimizer = optim.Adam(model.parameters(), lr=0.001)

def get_window_handle():
    if importlib.util.find_spec("pygetwindow"):
        gw = importlib.import_module("pygetwindow")
        wins = gw.getWindowsWithTitle("AI AGENT HUD")
        for w in wins:
            if w.title:
                return int(w._hWnd)
    return ctypes.windll.user32.GetForegroundWindow()

def set_window_state(minimize=True):
    if state.window_handle:
        CMD_SHOW = 5
        CMD_MINIMIZE = 6
        cmd = CMD_MINIMIZE if minimize else CMD_SHOW
        ctypes.windll.user32.ShowWindow(state.window_handle, cmd)
        if not minimize:
            ctypes.windll.user32.SetForegroundWindow(state.window_handle)

def check_disk_space():
    try:
        total_size = 0
        files = []
        for f in glob.glob(os.path.join(exp_pool_dir, "*.pkl")):
            s = os.path.getsize(f)
            total_size += s
            files.append((f, os.path.getctime(f)))
        files.sort(key=lambda x: x[1])
        limit = 20 * 1024 * 1024 * 1024
        while total_size > limit and files:
            f_path, _ = files.pop(0)
            s = os.path.getsize(f_path)
            os.remove(f_path)
            total_size -= s
    except Exception as e:
        panic(f"Disk Check Failed: {e}")

def encode_image(img):
    ok, buf = cv2.imencode('.jpg', img)
    return buf.tobytes() if ok else b''

def decode_image(buf):
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def monitor_resources():
    while state.is_running:
        try:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            if nvml_module and gpu_handle:
                nv_info = nvml_module.nvmlDeviceGetUtilizationRates(gpu_handle)
                mem_info = nvml_module.nvmlDeviceGetMemoryInfo(gpu_handle)
                gpu = nv_info.gpu
                vram = (mem_info.used / mem_info.total) * 100
            else:
                gpu = 0
                vram = 0
            M = max(cpu, mem, gpu, vram)
            with state.lock:
                state.resource_stats = {"cpu": cpu, "mem": mem, "gpu": gpu, "vram": vram, "fps": state.fps, "seq": state.seq_len}
                if M > 90:
                    state.is_boosting = False
                    if state.fps > 1: state.fps -= 1
                    if state.seq_len > 1: state.seq_len -= 1
                if M < 10:
                    state.is_boosting = True
                if state.is_boosting:
                    if M < 40:
                        if state.fps < 100: state.fps += 1
                        if state.seq_len < 100: state.seq_len += 1
                    else:
                        state.is_boosting = False
        except Exception as e:
            panic(f"Monitor Failed: {e}")
        time.sleep(1)

def on_click(x, y, button, pressed):
    if button == mouse.Button.left:
        t = time.time()
        with state.lock:
            if pressed:
                state.mouse_state["pressed"] = True
                state.mouse_state["start_pos"] = (x, y)
                state.mouse_state["start_time"] = t
            else:
                if state.mouse_state["pressed"]:
                    data_point = {
                        "type": "click",
                        "start_pos": state.mouse_state["start_pos"],
                        "start_time": state.mouse_state["start_time"],
                        "end_pos": (x, y),
                        "end_time": t,
                        "screen_jpg": state.last_screen,
                        "source": "HUMAN" if state.mode == "LEARNING" else "AI",
                        "button": 0,
                        "episode": state.current_episode_id
                    }
                    if state.mode in ["LEARNING", "TRAINING", "PRACTICAL"]:
                        state.buffer.append(data_point)
                state.mouse_state["pressed"] = False

def on_press(key):
    if key == keyboard.Key.esc:
        with state.lock:
            if state.mode in ["LEARNING", "TRAINING", "PRACTICAL"]:
                state.esc_pressed = True

def listener_thread():
    with mouse.Listener(on_click=on_click) as m_listener, \
         keyboard.Listener(on_press=on_press) as k_listener:
        try:
            m_listener.join()
            k_listener.join()
        except Exception as e:
            panic(f"Input Listener Failed: {e}")

def save_buffer(result_label):
    fname = f"{int(time.time())}_{result_label}.pkl"
    fpath = os.path.join(exp_pool_dir, fname)
    try:
        payload = {
            "episode_id": state.current_episode_id or str(uuid.uuid4()),
            "result": result_label,
            "mode": state.prev_mode if state.prev_mode else state.mode,
            "steps": state.buffer
        }
        with open(fpath, 'wb') as f:
            pickle.dump(payload, f)
    except Exception as e:
        panic(f"Save Failed: {e}")
    state.current_episode_id = None
    check_disk_space()

def worker_thread():
    sct = mss.mss()
    screen_w, screen_h = pyautogui.size()
    hidden_state = None
    criterion = nn.CrossEntropyLoss()

    def align_episode(steps):
        ordered = sorted(steps, key=lambda s: s.get('timestamp', s.get('start_time', time.time())))
        frames = [f for f in ordered if f.get('type') == 'frame']
        clicks = [c for c in ordered if c.get('type') == 'click']
        frames.sort(key=lambda f: f.get('timestamp', time.time()))
        for f in frames:
            if 'action' not in f:
                f['action'] = 2 if f.get('button', 0) == 1 else 0
        for c in clicks:
            st = c.get('start_time', c.get('timestamp'))
            et = c.get('end_time', st)
            if st is None or et is None:
                continue
            rel = [(i, f) for i, f in enumerate(frames) if f.get('timestamp') is not None and f.get('timestamp') >= st and f.get('timestamp') <= et]
            if not rel:
                continue
            start_idx = min(rel, key=lambda p: abs(p[1].get('timestamp') - st))[0]
            end_idx = min(rel, key=lambda p: abs(p[1].get('timestamp') - et))[0]
            if start_idx <= end_idx:
                frames[start_idx]['action'] = 1
                frames[start_idx]['button'] = 1
                for j in range(start_idx + 1, end_idx):
                    frames[j]['action'] = 2
                    frames[j]['button'] = 1
                frames[end_idx]['action'] = 0
                frames[end_idx]['button'] = 0
            else:
                frames[end_idx]['action'] = 0
                frames[end_idx]['button'] = 0
                frames[start_idx]['action'] = 1
                frames[start_idx]['button'] = 1
        return frames

    while state.is_running:
        try:
            with state.lock:
                current_mode = state.mode
                esc = state.esc_pressed
            if current_mode not in ["TRAINING", "PRACTICAL"]:
                hidden_state = None
                with state.lock:
                    state.ai_button_down = False

            if esc:
                if current_mode == "PRACTICAL":
                    pyautogui.mouseUp()
                    with state.lock:
                        state.prev_mode = state.mode
                    save_buffer("INTERRUPTED")
                    with state.lock:
                        state.buffer = []
                        state.mode = "IDLE"
                        state.esc_pressed = False
                        state.ai_button_down = False
                    set_window_state(minimize=False)
                else:
                    with state.lock:
                        state.prev_mode = state.mode
                        state.mode = "PAUSED"
                        state.esc_pressed = False
                    set_window_state(minimize=False)
                continue

            if current_mode == "LEARNING":
                start_time = time.time()
                img = np.array(sct.grab(sct.monitors[1]))
                img_small = cv2.resize(img, (256, 160))
                img_bytes = encode_image(img_small)
                with state.lock:
                    state.last_screen = img_bytes
                    if not state.current_episode_id:
                        state.current_episode_id = str(uuid.uuid4())
                    x, y = pyautogui.position()
                    btn_flag = 1 if state.mouse_state["pressed"] else 0
                    state.buffer.append({
                        "type": "frame",
                        "timestamp": time.time(),
                        "pos": (x, y),
                        "button": btn_flag,
                        "press_time": state.mouse_state["start_time"] if btn_flag else None,
                        "screen_jpg": img_bytes,
                        "source": "HUMAN",
                        "episode": state.current_episode_id
                    })
                elapsed = time.time() - start_time
                sleep_time = max(0, (1.0 / state.fps) - elapsed)
                time.sleep(sleep_time)

            elif current_mode in ["TRAINING", "PRACTICAL"]:
                start_time = time.time()
                img = np.array(sct.grab(sct.monitors[1]))
                img_small = cv2.resize(img, (256, 160))
                img_bytes = encode_image(img_small)
                with state.lock:
                    state.last_screen = img_bytes
                    if not state.current_episode_id:
                        state.current_episode_id = str(uuid.uuid4())

                img_tensor = torch.FloatTensor(img_small).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device) / 255.0
                with torch.no_grad():
                    mx, my, act, res, hidden_state = model(img_tensor, hidden_state)

                mx = int(mx.item() * screen_w)
                my = int(my.item() * screen_h)
                action = torch.argmax(act).item()
                pred_result = torch.argmax(res).item()

                pyautogui.moveTo(mx, my)
                if action == 1:
                    pyautogui.click()
                    with state.lock:
                        state.ai_button_down = False
                elif action == 2:
                    pyautogui.mouseDown()
                    with state.lock:
                        state.ai_button_down = True
                else:
                    pyautogui.mouseUp()
                    with state.lock:
                        state.ai_button_down = False

                with state.lock:
                    btn_flag = 1 if state.ai_button_down else 0
                    state.buffer.append({
                        "type": "frame",
                        "timestamp": time.time(),
                        "pos": (mx, my),
                        "action": action,
                        "button": btn_flag,
                        "screen_jpg": img_bytes,
                        "source": "AI",
                        "episode": state.current_episode_id
                    })

                if current_mode == "PRACTICAL" and pred_result != 0:
                    res_map = {1: "WIN", 2: "LOSS"}
                    final_res = res_map.get(pred_result, "DRAW")
                    save_buffer(final_res)
                    with state.lock:
                        state.buffer = []
                        state.mode = "IDLE"
                    set_window_state(minimize=False)

                elapsed = time.time() - start_time
                sleep_time = max(0, (1.0 / state.fps) - elapsed)
                time.sleep(sleep_time)

            elif current_mode == "OPTIMIZING":
                files = glob.glob(os.path.join(exp_pool_dir, "*.pkl"))
                if not files:
                    with state.lock:
                        state.mode = "IDLE"
                        state.optimization_status = "无数据"
                    continue

                state.optimization_status = "优化中..."
                model.train()
                episodes = []
                for f in files[-10:]:
                    try:
                        with open(f, 'rb') as fb:
                            loaded = pickle.load(fb)
                            if isinstance(loaded, dict) and 'steps' in loaded:
                                episodes.append(loaded)
                            elif isinstance(loaded, list):
                                episodes.append({"episode_id": str(uuid.uuid4()), "result": "UNKNOWN", "steps": loaded})
                    except Exception:
                        pass

                if not episodes:
                    with state.lock:
                        state.mode = "IDLE"
                        state.optimization_status = "数据损坏"
                    continue

                sequences = []
                seq_len = max(1, state.seq_len)
                for ep in episodes:
                    steps = ep.get('steps', [])
                    aligned = align_episode(steps)
                    if len(aligned) < seq_len:
                        continue
                    for i in range(len(aligned) - seq_len + 1):
                        seg = aligned[i:i+seq_len]
                        if all(('screen_jpg' in step or 'screen' in step) for step in seg):
                            sequences.append((seg, ep.get('result', "UNKNOWN")))

                if not sequences:
                    with state.lock:
                        state.mode = "IDLE"
                        state.optimization_status = "数据不足"
                    continue

                if len(sequences) < 10:
                    sequences = sequences * max(1, 10 // len(sequences))

                batch_count = 100
                for i in range(batch_count):
                    optimizer.zero_grad()
                    seq_data, seq_res = random.choice(sequences)
                    frames = []
                    for step in seq_data:
                        img_data = step.get('screen_jpg') if step.get('screen_jpg') is not None else step.get('screen')
                        if img_data is None:
                            frames = []
                            break
                        img_arr = decode_image(img_data) if isinstance(img_data, (bytes, bytearray)) else img_data
                        if img_arr is None:
                            frames = []
                            break
                        frames.append(torch.FloatTensor(img_arr).permute(2, 0, 1))
                    if not frames:
                        continue
                    data_tensor = torch.stack(frames).unsqueeze(0).to(device) / 255.0
                    mx_pred, my_pred, act_pred, res_pred, _ = model(data_tensor)

                    target_step = seq_data[-1]
                    if 'pos' in target_step:
                        tx = target_step['pos'][0] / screen_w
                        ty = target_step['pos'][1] / screen_h
                    elif 'start_pos' in target_step:
                        tx = target_step['start_pos'][0] / screen_w
                        ty = target_step['start_pos'][1] / screen_h
                    else:
                        tx = 0.5
                        ty = 0.5

                    action_label = target_step.get('action', 2 if target_step.get('button', 0) == 1 else 0)
                    res_map = {"WIN": 1, "LOSS": 2}
                    res_label = res_map.get(seq_res, 0)

                    pos_loss = ((mx_pred - torch.tensor([[tx]], device=device)) ** 2 + (my_pred - torch.tensor([[ty]], device=device)) ** 2).mean()
                    act_loss = criterion(act_pred, torch.tensor([action_label], device=device))
                    res_loss = criterion(res_pred, torch.tensor([res_label], device=device))
                    loss = pos_loss + act_loss + res_loss
                    loss.backward()
                    optimizer.step()

                    with state.lock:
                        state.optimization_progress = int((i / batch_count) * 100)

                torch.save(model.state_dict(), os.path.join(model_dir, "latest.pth"))
                with state.lock:
                    state.mode = "IDLE"
                    state.optimization_status = "完成"
            else:
                time.sleep(0.1)
        except Exception as e:
            panic(f"Worker Failed: {e}")
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>AI AGENT HUD</title>
    <style>
        body { background-color: #0a0a12; color: #00f3ff; font-family: 'Segoe UI', sans-serif; margin: 0; overflow: hidden; user-select: none; }
        .container { display: flex; flex-direction: column; height: 100vh; padding: 20px; box-sizing: border-box; background: radial-gradient(circle at center, #1a1a2e 0%, #000 100%); }
        .header { border-bottom: 1px solid #00f3ff; padding-bottom: 15px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 5px 15px rgba(0, 243, 255, 0.1); }
        .title { font-size: 28px; letter-spacing: 4px; text-shadow: 0 0 15px #00f3ff; font-weight: 800; }
        .status { font-size: 18px; color: #fff; text-shadow: 0 0 5px #fff; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; flex-grow: 1; padding-top: 30px; }
        .panel { background: rgba(0, 20, 40, 0.6); border: 1px solid rgba(0, 243, 255, 0.3); padding: 20px; border-radius: 5px; position: relative; backdrop-filter: blur(5px); }
        .panel::before { content: ''; position: absolute; top: -1px; left: -1px; width: 10px; height: 10px; border-top: 2px solid #00f3ff; border-left: 2px solid #00f3ff; }
        .panel::after { content: ''; position: absolute; bottom: -1px; right: -1px; width: 10px; height: 10px; border-bottom: 2px solid #00f3ff; border-right: 2px solid #00f3ff; }
        .btn { background: rgba(0, 243, 255, 0.1); border: 1px solid #00f3ff; color: #00f3ff; padding: 15px; font-size: 16px; cursor: pointer; transition: 0.2s; margin-bottom: 15px; width: 100%; font-weight: bold; letter-spacing: 1px; text-transform: uppercase; }
        .btn:hover { background: #00f3ff; color: #000; box-shadow: 0 0 20px #00f3ff; }
        .btn:disabled { border-color: #444; color: #444; cursor: not-allowed; background: transparent; box-shadow: none; }
        .stat-row { display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 14px; align-items: center; }
        .bar-bg { background: #222; height: 8px; flex-grow: 1; margin: 0 15px; border-radius: 4px; overflow: hidden; }
        .bar-fill { background: #00f3ff; height: 100%; width: 0%; transition: width 0.3s; box-shadow: 0 0 8px #00f3ff; }
        #modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 999; justify-content: center; align-items: center; backdrop-filter: blur(8px); }
        .modal-box { border: 1px solid #00f3ff; padding: 40px; background: #000; text-align: center; box-shadow: 0 0 50px rgba(0, 243, 255, 0.3); width: 300px; }
        .modal-title { font-size: 24px; margin-bottom: 30px; color: #fff; letter-spacing: 2px; }
    </style>
</head>
<body>
    <div id="modal">
        <div class="modal-box">
            <div class="modal-title">任务暂停</div>
            <button class="btn" onclick="submitResult('WIN')">胜利</button>
            <button class="btn" onclick="submitResult('DRAW')">平局</button>
            <button class="btn" onclick="submitResult('LOSS')">失败</button>
            <button class="btn" onclick="submitResult('CONTINUE')">继续任务</button>
        </div>
    </div>
    <div class="container">
        <div class="header">
            <div class="title">NEURAL LINK</div>
            <div class="status" id="status-display">SYSTEM: STANDBY</div>
        </div>
        <div class="grid">
            <div class="panel">
                <h3 style="margin-top:0; border-bottom:1px solid #333; padding-bottom:10px;">RESOURCE MONITOR</h3>
                <div id="stats-container"></div>
                <div style="margin-top: 20px; border-top: 1px solid #333; padding-top: 15px;">
                    <div class="stat-row"><span>SAMPLING RATE</span> <span id="val-fps" style="color:#fff">30</span> Hz</div>
                    <div class="stat-row"><span>MEMORY DEPTH</span> <span id="val-seq" style="color:#fff">10</span> Frames</div>
                </div>
            </div>
            <div class="panel" style="display: flex; flex-direction: column; justify-content: center;">
                <button class="btn" id="btn-learn" onclick="post('/start_learn')">开始学习 [LEARN]</button>
                <button class="btn" id="btn-train" onclick="post('/start_train')">开始训练 [TRAIN]</button>
                <button class="btn" id="btn-prac" onclick="post('/start_practical')">开始实操 [AUTO]</button>
                <button class="btn" id="btn-opt" onclick="post('/optimize')">优化模型 [OPTIMIZE]</button>
                <div id="opt-ui" style="display:none; margin-top: 10px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px;"><span>OPTIMIZING...</span><span id="opt-text">0%</span></div>
                    <div class="bar-bg"><div id="opt-bar" class="bar-fill"></div></div>
                </div>
            </div>
        </div>
    </div>
    <script>
        function update() {
            fetch('/status').then(r => r.json()).then(d => {
                const m = {'IDLE':'待机', 'LEARNING':'学习中', 'TRAINING':'训练中', 'PRACTICAL':'实操中', 'OPTIMIZING':'优化中', 'PAUSED':'暂停'};
                document.getElementById('status-display').innerText = 'SYSTEM: ' + (m[d.mode] || d.mode);
                document.getElementById('val-fps').innerText = d.stats.fps;
                document.getElementById('val-seq').innerText = d.stats.seq;
                let h = '';
                ['cpu', 'mem', 'gpu', 'vram'].forEach(k => {
                    h += `<div class="stat-row"><span>${k.toUpperCase()}</span><div class="bar-bg"><div class="bar-fill" style="width:${d.stats[k]}%"></div></div><span>${Math.round(d.stats[k])}%</span></div>`;
                });
                document.getElementById('stats-container').innerHTML = h;
                const isOpt = d.mode === 'OPTIMIZING';
                document.getElementById('opt-ui').style.display = isOpt ? 'block' : 'none';
                if(isOpt) {
                    document.getElementById('opt-bar').style.width = d.opt_progress + '%';
                    document.getElementById('opt-text').innerText = d.opt_progress + '%';
                }
                document.getElementById('modal').style.display = (d.mode === 'PAUSED') ? 'flex' : 'none';
                ['btn-learn', 'btn-train', 'btn-prac', 'btn-opt'].forEach(b => document.getElementById(b).disabled = (d.mode !== 'IDLE'));
            }).catch(()=>{});
        }
        function post(u) { fetch(u, {method:'POST'}); }
        function submitResult(r) {
            fetch('/submit_result', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({result:r})});
        }
        setInterval(update, 1000);
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return HTML_TEMPLATE

@app.route('/status')
def status():
    with state.lock:
        return jsonify({"mode": state.mode, "stats": state.resource_stats, "opt_progress": state.optimization_progress})

@app.route('/start_learn', methods=['POST'])
def start_learn():
    with state.lock:
        if state.mode == "IDLE":
            state.mode = "LEARNING"
            state.buffer = []
            state.current_episode_id = str(uuid.uuid4())
            state.ai_button_down = False
            state.window_handle = get_window_handle()
            set_window_state(minimize=True)
    return jsonify({})

@app.route('/start_train', methods=['POST'])
def start_train():
    with state.lock:
        if state.mode == "IDLE":
            state.mode = "TRAINING"
            state.buffer = []
            state.current_episode_id = str(uuid.uuid4())
            state.ai_button_down = False
            state.window_handle = get_window_handle()
            set_window_state(minimize=True)
    return jsonify({})

@app.route('/start_practical', methods=['POST'])
def start_practical():
    with state.lock:
        if state.mode == "IDLE":
            state.mode = "PRACTICAL"
            state.buffer = []
            state.current_episode_id = str(uuid.uuid4())
            state.ai_button_down = False
            state.window_handle = get_window_handle()
            set_window_state(minimize=True)
    return jsonify({})

@app.route('/optimize', methods=['POST'])
def optimize():
    with state.lock:
        if state.mode == "IDLE":
            state.mode = "OPTIMIZING"
            state.optimization_progress = 0
    return jsonify({})

@app.route('/submit_result', methods=['POST'])
def submit_result():
    res = request.json.get('result')
    with state.lock:
        if res == "CONTINUE":
            state.mode = state.prev_mode
            set_window_state(minimize=True)
        else:
            save_buffer(res)
            state.buffer = []
            state.mode = "IDLE"
            set_window_state(minimize=False)
    return jsonify({})

if __name__ == "__main__":
    threading.Thread(target=monitor_resources, daemon=True).start()
    threading.Thread(target=listener_thread, daemon=True).start()
    threading.Thread(target=worker_thread, daemon=True).start()
    try:
        webbrowser.open("http://127.0.0.1:5000")
        app.run(port=5000, use_reloader=False)
    except Exception as e:
        panic(f"App Start Failed: {e}")
