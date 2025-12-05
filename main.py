import sys
import os
import subprocess
import time
import threading
import queue
import datetime
import shutil
import ctypes
import random
import glob

def install_requirements():
    required = [
        "torch", "torchvision", "numpy", "opencv-python", "mss", 
        "pynput", "psutil", "nvidia-ml-py3", "pillow"
    ]
    for package in required:
        try:
            __import__(package.split("==")[0].replace("-", "_").replace("nvidia_ml_py3", "pynvml").replace("pillow", "PIL"))
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except Exception as e:
                print(f"Error installing {package}: {e}")

install_requirements()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import mss
import psutil
from pynput import mouse, keyboard
import pynvml

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
base_dir = os.path.join(desktop_path, "AAA")
data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "model")
temp_dir = os.path.join(base_dir, "temp")

for d in [base_dir, data_dir, model_dir, temp_dir]:
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except Exception as e:
            print(f"Error creating directory {d}: {e}")

MODE_LEARNING = "LEARNING"
MODE_SLEEP = "SLEEP"
MODE_TRAINING = "TRAINING"
current_mode = MODE_LEARNING
stop_training_flag = False

capture_freq = 10
seq_len = 5
screen_w, screen_h = 2560, 1600
target_w, target_h = 256, 160
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mouse_state = {
    "x": 0, "y": 0,
    "l_down": False, "l_down_ts": 0.0, "l_up_pos": (0,0), "l_up_ts": 0.0,
    "r_down": False, "r_down_ts": 0.0, "r_up_pos": (0,0), "r_up_ts": 0.0,
    "scroll": 0, "traj": []
}
mouse_lock = threading.Lock()

data_queue = queue.Queue()

class UniversalAI(nn.Module):
    def __init__(self):
        super(UniversalAI, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_input_dim = 64 * 17 * 29
        self.mouse_dim = 10 
        
        self.lstm = nn.LSTM(input_size=self.fc_input_dim + self.mouse_dim, hidden_size=512, batch_first=True)
        
        self.fc_action = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4) 
        )

    def forward(self, img, mouse_input, hidden=None):
        batch_size, seq, c, h, w = img.size()
        img_reshaped = img.view(batch_size * seq, c, h, w)
        feat = self.conv(img_reshaped)
        feat = feat.view(batch_size, seq, -1)
        
        combined = torch.cat((feat, mouse_input), dim=2)
        
        out, hidden = self.lstm(combined, hidden)
        
        action = self.fc_action(out)
        return action, hidden

def get_sys_usage():
    try:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        gpu = 0
        vram = 0
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu = util.gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram = (mem_info.used / mem_info.total) * 100
        except:
            pass
        return max(cpu, mem, gpu, vram)
    except Exception as e:
        print(f"Monitor Error: {e}")
        return 50

def resource_monitor():
    global capture_freq, seq_len
    while True:
        try:
            m = get_sys_usage()
            if m > 90:
                capture_freq = max(1, int(capture_freq * 0.9))
                seq_len = max(1, int(seq_len * 0.9))
            elif m < 10:
                capture_freq = min(100, int(capture_freq * 1.1) + 1)
                seq_len = min(100, int(seq_len * 1.1) + 1)
            elif m >= 40 and m <= 60:
                pass 
            elif m < 40 and capture_freq < 100: 
                 capture_freq = min(100, int(capture_freq * 1.05) + 1)
                 seq_len = min(100, int(seq_len * 1.05) + 1)

            time.sleep(1)
        except Exception as e:
            print(f"Resource Monitor Error: {e}")
            time.sleep(1)

def on_move(x, y):
    with mouse_lock:
        mouse_state["x"] = x
        mouse_state["y"] = y
        mouse_state["traj"].append((x, y))
        if len(mouse_state["traj"]) > 50:
            mouse_state["traj"].pop(0)

def on_click(x, y, button, pressed):
    with mouse_lock:
        t = time.time()
        if button == mouse.Button.left:
            mouse_state["l_down"] = pressed
            if pressed:
                mouse_state["l_down_ts"] = t
            else:
                mouse_state["l_up_pos"] = (x, y)
                mouse_state["l_up_ts"] = t
        elif button == mouse.Button.right:
            mouse_state["r_down"] = pressed
            if pressed:
                mouse_state["r_down_ts"] = t
            else:
                mouse_state["r_up_pos"] = (x, y)
                mouse_state["r_up_ts"] = t

def on_scroll(x, y, dx, dy):
    with mouse_lock:
        mouse_state["scroll"] = dy

def on_press_key(key):
    global current_mode, stop_training_flag
    if current_mode == MODE_TRAINING:
        stop_training_flag = True

def check_disk_space():
    try:
        total_size = 0
        files = []
        for f in glob.glob(os.path.join(data_dir, "*.npy")):
            size = os.path.getsize(f)
            total_size += size
            files.append((f, os.path.getmtime(f)))
        
        limit = 20 * 1024 * 1024 * 1024
        if total_size > limit:
            files.sort(key=lambda x: x[1])
            while total_size > limit and files:
                f, _ = files.pop(0)
                try:
                    s = os.path.getsize(f)
                    os.remove(f)
                    total_size -= s
                except:
                    pass
    except Exception as e:
        print(f"Disk clean error: {e}")

class GameDataset(Dataset):
    def __init__(self, file_list):
        self.data = []
        for f in file_list:
            try:
                arr = np.load(f, allow_pickle=True)
                self.data.append(arr)
            except:
                pass
        if self.data:
            self.data = np.concatenate(self.data, axis=0)

    def __len__(self):
        if len(self.data) == 0: return 0
        return len(self.data) - seq_len

    def __getitem__(self, idx):
        try:
            slice_data = self.data[idx : idx + seq_len]
            imgs = []
            m_ins = []
            labels = []
            
            for item in slice_data:
                img = item["screen"]
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                img = img.transpose(2, 0, 1) / 255.0
                imgs.append(img)
                
                mx = item["mouse_x"] / screen_w
                my = item["mouse_y"] / screen_h
                m_vec = [
                    mx, my,
                    1.0 if item["l_down"] else 0.0,
                    1.0 if item["r_down"] else 0.0,
                    item["scroll"],
                    0, 0, 0, 0, 0
                ]
                m_ins.append(m_vec)
            
            last_item = slice_data[-1]
            if idx + seq_len < len(self.data):
                next_item = self.data[idx + seq_len]
                target_x = next_item["mouse_x"] / screen_w
                target_y = next_item["mouse_y"] / screen_h
                
                target_l = 1.0 if next_item["l_down"] else 0.0
                target_r = 1.0 if next_item["r_down"] else 0.0
                labels = [target_x, target_y, target_l, target_r]
            else:
                labels = [last_item["mouse_x"]/screen_w, last_item["mouse_y"]/screen_h, 0.0, 0.0]

            return torch.FloatTensor(np.array(imgs)), torch.FloatTensor(np.array(m_ins)), torch.FloatTensor(np.array(labels))
        except Exception as e:
            print(f"Data load error: {e}")
            return torch.zeros((seq_len, 3, target_h, target_w)), torch.zeros((seq_len, 10)), torch.zeros(4)

def optimize_ai():
    print("Starting Optimization...")
    model_path = os.path.join(model_dir, "ai_model.pth")
    model = UniversalAI().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except:
            pass
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not files:
        print("No data to train.")
        return

    random.shuffle(files)
    batch_files = [files[i:i+5] for i in range(0, len(files), 5)]
    
    for bf in batch_files:
        dataset = GameDataset(bf)
        if len(dataset) == 0: continue
        loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)
        
        for imgs, mins, labels in loader:
            imgs = imgs.to(device)
            mins = mins.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            out, _ = model(imgs, mins)
            loss = criterion(out[:, -1, :], labels)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), model_path)
    print("Optimization Complete.")

def start_training_mode():
    global stop_training_flag, current_mode
    stop_training_flag = False
    
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    ctypes.windll.user32.ShowWindow(hwnd, 6)
    
    model_path = os.path.join(model_dir, "ai_model.pth")
    if not os.path.exists(model_path):
        print("No model found. Aborting training.")
        ctypes.windll.user32.ShowWindow(hwnd, 9) 
        return

    model = UniversalAI().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    mouse_ctrl = mouse.Controller()
    
    hidden = None
    input_buffer_img = []
    input_buffer_mouse = []
    
    sct = mss.mss()
    
    with torch.no_grad():
        while not stop_training_flag:
            start_time = time.time()
            
            try:
                img = np.array(sct.grab({"top": 0, "left": 0, "width": screen_w, "height": screen_h}))
                img_small = cv2.resize(img, (target_w, target_h))
                
                with mouse_lock:
                    curr_x, curr_y = mouse_state["x"], mouse_state["y"]
                    l_down = mouse_state["l_down"]
                    r_down = mouse_state["r_down"]
                    scroll = mouse_state["scroll"]

                mouse_state["scroll"] = 0 
                
                img_tensor = cv2.cvtColor(img_small, cv2.COLOR_BGRA2RGB)
                img_tensor = img_tensor.transpose(2, 0, 1) / 255.0
                
                m_vec = [
                    curr_x / screen_w, curr_y / screen_h,
                    1.0 if l_down else 0.0,
                    1.0 if r_down else 0.0,
                    scroll, 0, 0, 0, 0, 0
                ]
                
                input_buffer_img.append(img_tensor)
                input_buffer_mouse.append(m_vec)
                
                if len(input_buffer_img) > seq_len:
                    input_buffer_img.pop(0)
                    input_buffer_mouse.pop(0)
                
                if len(input_buffer_img) == seq_len:
                    t_imgs = torch.FloatTensor(np.array([input_buffer_img])).to(device)
                    t_mins = torch.FloatTensor(np.array([input_buffer_mouse])).to(device)
                    
                    out, hidden = model(t_imgs, t_mins, hidden)
                    
                    action = out[0, -1, :].cpu().numpy()
                    
                    pred_x = int(action[0] * screen_w)
                    pred_y = int(action[1] * screen_h)
                    pred_l = action[2]
                    pred_r = action[3]
                    
                    mouse_ctrl.position = (pred_x, pred_y)
                    
                    if pred_l > 0.5 and not l_down:
                        mouse_ctrl.press(mouse.Button.left)
                    elif pred_l <= 0.5 and l_down:
                        mouse_ctrl.release(mouse.Button.left)
                        
                    if pred_r > 0.5 and not r_down:
                        mouse_ctrl.press(mouse.Button.right)
                    elif pred_r <= 0.5 and r_down:
                        mouse_ctrl.release(mouse.Button.right)
                
                elapsed = time.time() - start_time
                wait = (1.0 / capture_freq) - elapsed
                if wait > 0:
                    time.sleep(wait)
                    
            except Exception as e:
                print(f"Training Runtime Error: {e}")
                break

    ctypes.windll.user32.ShowWindow(hwnd, 9)
    current_mode = MODE_LEARNING
    print("Exited Training Mode. Back to Learning.")

def record_data_loop():
    sct = mss.mss()
    buffer = []
    
    while True:
        if current_mode == MODE_LEARNING or current_mode == MODE_TRAINING:
            try:
                start_time = time.time()
                
                img = np.array(sct.grab({"top": 0, "left": 0, "width": screen_w, "height": screen_h}))
                img_small = cv2.resize(img, (target_w, target_h))
                
                with mouse_lock:
                    c_state = mouse_state.copy()
                    c_state["traj"] = list(mouse_state["traj"])
                    mouse_state["scroll"] = 0
                
                entry = {
                    "ts": start_time,
                    "screen": img_small,
                    "mouse_x": c_state["x"],
                    "mouse_y": c_state["y"],
                    "l_down": c_state["l_down"],
                    "l_down_ts": c_state["l_down_ts"],
                    "l_up_pos": c_state["l_up_pos"],
                    "l_up_ts": c_state["l_up_ts"],
                    "r_down": c_state["r_down"],
                    "r_down_ts": c_state["r_down_ts"],
                    "r_up_pos": c_state["r_up_pos"],
                    "r_up_ts": c_state["r_up_ts"],
                    "scroll": c_state["scroll"],
                    "traj": c_state["traj"],
                    "source": "AI" if current_mode == MODE_TRAINING else "Human"
                }
                
                buffer.append(entry)
                
                if len(buffer) >= 100:
                    fname = os.path.join(data_dir, f"{int(time.time()*1000)}.npy")
                    np.save(fname, buffer)
                    buffer = []
                    check_disk_space()

                elapsed = time.time() - start_time
                wait = (1.0 / capture_freq) - elapsed
                if wait > 0:
                    time.sleep(wait)
            except Exception as e:
                print(f"Recording Error: {e}")
        else:
            time.sleep(1)

def input_loop():
    global current_mode
    while True:
        cmd = input().strip()
        if cmd == "睡眠":
            if current_mode == MODE_LEARNING:
                current_mode = MODE_SLEEP
                optimize_ai()
                current_mode = MODE_LEARNING
                print("Back to Learning.")
        elif cmd == "训练":
            if current_mode == MODE_LEARNING:
                current_mode = MODE_TRAINING
                print("Entering Training Mode...")
                t_thread = threading.Thread(target=start_training_mode)
                t_thread.daemon = True
                t_thread.start()

if __name__ == "__main__":
    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
    mouse_listener.start()
    
    key_listener = keyboard.Listener(on_press=on_press_key)
    key_listener.start()
    
    t_res = threading.Thread(target=resource_monitor)
    t_res.daemon = True
    t_res.start()
    
    t_rec = threading.Thread(target=record_data_loop)
    t_rec.daemon = True
    t_rec.start()
    
    t_input = threading.Thread(target=input_loop)
    t_input.daemon = True
    t_input.start()
    
    print("System initialized. Mode: LEARNING")
    
    while True:
        time.sleep(10)
