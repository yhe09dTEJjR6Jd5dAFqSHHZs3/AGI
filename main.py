import os
import sys
import time
import json
import uuid
import glob
import shutil
import ctypes
import pickle
import random
import threading
import webbrowser
import warnings
import logging
import numpy as np
import cv2
import mss
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import pynvml
import pyautogui
from pynput import mouse, keyboard
from flask import Flask, request, jsonify

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

try:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    sys.exit(1)

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

state = SharedState()

class AIModel(nn.Module):
    def __init__(self):
        super(AIModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(32 * 63 * 99, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc_mouse_x = nn.Linear(128, 1)
        self.fc_mouse_y = nn.Linear(128, 1)
        self.fc_action = nn.Linear(128, 3)

    def forward(self, x, hidden=None):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(batch_size * seq_len, -1)
        x = torch.relu(self.fc1(x))
        x = x.view(batch_size, seq_len, -1)
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]
        mx = torch.sigmoid(self.fc_mouse_x(out))
        my = torch.sigmoid(self.fc_mouse_y(out))
        act = torch.softmax(self.fc_action(out), dim=1)
        return mx, my, act, hidden

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AIModel().to(device)
if os.path.exists(os.path.join(model_dir, "latest.pth")):
    model.load_state_dict(torch.load(os.path.join(model_dir, "latest.pth")))
optimizer = optim.Adam(model.parameters(), lr=0.001)

def get_window_handle():
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
        try:
            os.remove(f_path)
            total_size -= s
        except:
            pass

def monitor_resources():
    while state.is_running:
        try:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            nv_info = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu = nv_info.gpu
            vram = (mem_info.used / mem_info.total) * 100
            
            M = max(cpu, mem, gpu, vram)
            
            with state.lock:
                state.resource_stats = {"cpu": cpu, "mem": mem, "gpu": gpu, "vram": vram, "fps": state.fps, "seq": state.seq_len}
                
                if M > 90:
                    if state.fps > 1: state.fps -= 1
                    if state.seq_len > 1: state.seq_len -= 1
                elif M < 10:
                    M_low = max(cpu, mem, gpu, vram)
                    if M_low < 40:
                         if state.fps < 100: state.fps += 1
                         if state.seq_len < 100: state.seq_len += 1
                    
        except Exception as e:
            sys.exit(1)
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
                        "screen": state.last_screen
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
        except:
            sys.exit(1)

def worker_thread():
    sct = mss.mss()
    screen_w, screen_h = pyautogui.size()
    
    while state.is_running:
        try:
            with state.lock:
                current_mode = state.mode
                esc = state.esc_pressed
            
            if esc:
                with state.lock:
                    state.prev_mode = state.mode
                    state.mode = "PAUSED"
                    state.esc_pressed = False
                set_window_state(minimize=False)
                continue

            if current_mode == "LEARNING":
                start_time = time.time()
                img = np.array(sct.grab(sct.monitors[1]))
                img_small = cv2.resize(img, (260, 160)) 
                
                with state.lock:
                    state.last_screen = img_small
                    if not state.mouse_state["pressed"]:
                        x, y = pyautogui.position()
                        state.buffer.append({
                            "type": "hover",
                            "timestamp": time.time(),
                            "pos": (x, y),
                            "screen": img_small,
                            "source": "HUMAN"
                        })
                
                elapsed = time.time() - start_time
                sleep_time = max(0, (1.0 / state.fps) - elapsed)
                time.sleep(sleep_time)

            elif current_mode in ["TRAINING", "PRACTICAL"]:
                start_time = time.time()
                img = np.array(sct.grab(sct.monitors[1]))
                img_small = cv2.resize(img, (260, 160))
                
                with state.lock:
                    state.last_screen = img_small
                
                img_tensor = torch.FloatTensor(img_small).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device) / 255.0
                
                with torch.no_grad():
                    mx, my, act, _ = model(img_tensor)
                
                mx = int(mx.item() * screen_w)
                my = int(my.item() * screen_h)
                action = torch.argmax(act).item()
                
                pyautogui.moveTo(mx, my)
                if action == 1: 
                    pyautogui.click()
                elif action == 2:
                    pyautogui.mouseDown()
                else:
                    pyautogui.mouseUp()
                
                with state.lock:
                    state.buffer.append({
                        "type": "ai_action",
                        "timestamp": time.time(),
                        "pos": (mx, my),
                        "action": action,
                        "screen": img_small,
                        "source": "AI"
                    })
                
                if current_mode == "PRACTICAL":
                    pass

                elapsed = time.time() - start_time
                sleep_time = max(0, (1.0 / state.fps) - elapsed)
                time.sleep(sleep_time)

            elif current_mode == "OPTIMIZING":
                files = glob.glob(os.path.join(exp_pool_dir, "*.pkl"))
                if not files:
                    with state.lock:
                        state.mode = "IDLE"
                        state.optimization_status = "数据不足"
                    continue
                
                state.optimization_status = "训练中..."
                model.train()
                
                data_cache = []
                for f in files[-5:]:
                    try:
                        with open(f, 'rb') as fb:
                            data_cache.extend(pickle.load(fb))
                    except: pass
                
                if not data_cache:
                    with state.lock:
                        state.mode = "IDLE"
                        state.optimization_status = "数据无效"
                    continue

                if len(data_cache) < 50:
                    data_cache = data_cache * 50
                
                batch_count = 100
                for i in range(batch_count):
                    optimizer.zero_grad()
                    sample = random.choice(data_cache)
                    if 'screen' not in sample: continue
                    
                    img = torch.FloatTensor(sample['screen']).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device) / 255.0
                    mx, my, act, _ = model(img)
                    
                    if 'pos' in sample:
                        tx = sample['pos'][0] / screen_w
                        ty = sample['pos'][1] / screen_h
                    elif 'start_pos' in sample:
                        tx = sample['start_pos'][0] / screen_w
                        ty = sample['start_pos'][1] / screen_h
                    else:
                        tx, ty = 0.5, 0.5

                    loss = (mx - tx)**2 + (my - ty)**2
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

        except Exception:
            sys.exit(1)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>智能控制终端</title>
    <style>
        body { background-color: #050505; color: #00ffcc; font-family: 'Microsoft YaHei', sans-serif; margin: 0; overflow: hidden; }
        .container { display: flex; flex-direction: column; height: 100vh; padding: 20px; box-sizing: border-box; }
        .hud-header { border-bottom: 2px solid #00ffcc; padding-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
        .hud-title { font-size: 24px; letter-spacing: 2px; text-shadow: 0 0 10px #00ffcc; font-weight: bold; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; flex-grow: 1; padding-top: 20px; }
        .panel { border: 1px solid #003333; background: rgba(0, 20, 20, 0.8); padding: 15px; position: relative; }
        .panel::after { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 1px solid transparent; box-shadow: inset 0 0 20px rgba(0, 255, 204, 0.1); pointer-events: none; }
        .btn { background: transparent; border: 1px solid #00ffcc; color: #00ffcc; padding: 15px 30px; font-size: 18px; cursor: pointer; transition: 0.3s; margin: 5px; width: 100%; font-family: 'Microsoft YaHei'; font-weight: bold; }
        .btn:hover { background: #00ffcc; color: #000; box-shadow: 0 0 15px #00ffcc; }
        .btn:disabled { border-color: #333; color: #333; cursor: not-allowed; box-shadow: none; }
        .stat-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }
        .bar-container { background: #111; height: 10px; width: 60%; position: relative; display: flex; align-items: center; }
        .bar-fill { background: #00ffcc; height: 100%; width: 0%; transition: width 0.5s; box-shadow: 0 0 5px #00ffcc; }
        
        #modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 1000; justify-content: center; align-items: center; }
        .modal-content { border: 2px solid #00ffcc; padding: 40px; background: #000; text-align: center; box-shadow: 0 0 30px #00ffcc; width: 400px; }
        .modal-title { font-size: 30px; margin-bottom: 30px; color: #fff; }
    </style>
</head>
<body>
    <div id="modal">
        <div class="modal-content">
            <div class="modal-title">任务中断</div>
            <button class="btn" onclick="submitResult('WIN')">胜利</button>
            <button class="btn" onclick="submitResult('DRAW')">平局</button>
            <button class="btn" onclick="submitResult('LOSS')">失败</button>
            <button class="btn" onclick="submitResult('CONTINUE')">继续</button>
        </div>
    </div>

    <div class="container">
        <div class="hud-header">
            <div class="hud-title">智能控制终端</div>
            <div id="status-display">状态: 待机</div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <h3>系统监控</h3>
                <div id="stats-container"></div>
                <div style="margin-top: 20px;">
                    <div class="stat-row"><span>采样频率:</span> <span id="val-fps">30</span> Hz</div>
                    <div class="stat-row"><span>记忆深度:</span> <span id="val-seq">10</span> Frames</div>
                </div>
            </div>
            
            <div class="panel" style="display: flex; flex-direction: column; justify-content: center;">
                <button class="btn" id="btn-learn" onclick="startMode('learn')">开始学习</button>
                <button class="btn" id="btn-train" onclick="startMode('train')">开始训练</button>
                <button class="btn" id="btn-prac" onclick="startMode('practical')">开始实操</button>
                <button class="btn" id="btn-opt" onclick="startOpt()">优化模型</button>
                <div id="opt-progress-container" style="display:none; margin-top: 15px;">
                    <div>优化进度 <span id="opt-text">0%</span></div>
                    <div class="bar-container" style="width: 100%; margin-top: 5px;"><div id="opt-bar" class="bar-fill"></div></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateStats() {
            fetch('/status').then(r => r.json()).then(data => {
                const modeMap = {
                    'IDLE': '待机', 'LEARNING': '学习模式', 'TRAINING': '训练模式', 
                    'PRACTICAL': '实操模式', 'OPTIMIZING': '优化中', 'PAUSED': '暂停'
                };
                document.getElementById('status-display').innerText = '状态: ' + (modeMap[data.mode] || data.mode);
                document.getElementById('val-fps').innerText = data.stats.fps;
                document.getElementById('val-seq').innerText = data.stats.seq;
                
                let html = '';
                ['cpu', 'mem', 'gpu', 'vram'].forEach(k => {
                    html += `<div class="stat-row"><span>${k.toUpperCase()}</span><div class="bar-container"><div class="bar-fill" style="width:${data.stats[k]}%"></div></div><span>${Math.round(data.stats[k])}%</span></div>`;
                });
                document.getElementById('stats-container').innerHTML = html;

                if (data.mode === 'OPTIMIZING') {
                    document.getElementById('opt-progress-container').style.display = 'block';
                    document.getElementById('opt-bar').style.width = data.opt_progress + '%';
                    document.getElementById('opt-text').innerText = data.opt_progress + '%';
                } else {
                    document.getElementById('opt-progress-container').style.display = 'none';
                }

                if (data.mode === 'PAUSED') {
                    document.getElementById('modal').style.display = 'flex';
                } else {
                    document.getElementById('modal').style.display = 'none';
                }
                
                const btns = ['btn-learn', 'btn-train', 'btn-prac', 'btn-opt'];
                btns.forEach(b => document.getElementById(b).disabled = (data.mode !== 'IDLE'));
            }).catch(() => {});
        }

        function startMode(mode) {
            fetch('/start_' + mode, {method: 'POST'});
        }

        function startOpt() {
            fetch('/optimize', {method: 'POST'});
        }

        function submitResult(res) {
            fetch('/submit_result', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({result: res})
            });
        }

        setInterval(updateStats, 1000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/status')
def status():
    with state.lock:
        return jsonify({
            "mode": state.mode,
            "stats": state.resource_stats,
            "opt_progress": state.optimization_progress,
            "opt_status": state.optimization_status
        })

@app.route('/start_learn', methods=['POST'])
def start_learn():
    with state.lock:
        if state.mode == "IDLE":
            state.mode = "LEARNING"
            state.current_episode_id = str(uuid.uuid4())
            state.buffer = []
            state.window_handle = get_window_handle()
            set_window_state(minimize=True)
    return jsonify({"status": "ok"})

@app.route('/start_train', methods=['POST'])
def start_train():
    with state.lock:
        if state.mode == "IDLE":
            state.mode = "TRAINING"
            state.current_episode_id = str(uuid.uuid4())
            state.buffer = []
            state.window_handle = get_window_handle()
            set_window_state(minimize=True)
    return jsonify({"status": "ok"})

@app.route('/start_practical', methods=['POST'])
def start_practical():
    with state.lock:
        if state.mode == "IDLE":
            state.mode = "PRACTICAL"
            state.current_episode_id = str(uuid.uuid4())
            state.buffer = []
            state.window_handle = get_window_handle()
            set_window_state(minimize=True)
    return jsonify({"status": "ok"})

@app.route('/optimize', methods=['POST'])
def optimize():
    with state.lock:
        if state.mode == "IDLE":
            state.mode = "OPTIMIZING"
            state.optimization_progress = 0
            return jsonify({"status": "started"})
    return jsonify({"status": "busy"})

@app.route('/submit_result', methods=['POST'])
def submit_result():
    data = request.json
    res = data.get('result')
    
    with state.lock:
        if res == "CONTINUE":
            state.mode = state.prev_mode
            set_window_state(minimize=True)
        else:
            fname = f"{int(time.time())}_{res}.pkl"
            fpath = os.path.join(exp_pool_dir, fname)
            try:
                with open(fpath, 'wb') as f:
                    pickle.dump(state.buffer, f)
            except:
                pass
            
            check_disk_space()
            
            state.buffer = []
            state.mode = "IDLE"
            set_window_state(minimize=False)
            
    return jsonify({"status": "ok"})

def run_app():
    webbrowser.open("http://127.0.0.1:5000")
    app.run(port=5000, use_reloader=False)

if __name__ == "__main__":
    t_res = threading.Thread(target=monitor_resources, daemon=True)
    t_res.start()
    
    t_list = threading.Thread(target=listener_thread, daemon=True)
    t_list.start()
    
    t_work = threading.Thread(target=worker_thread, daemon=True)
    t_work.start()
    
    try:
        run_app()
    except:
        sys.exit(1)
