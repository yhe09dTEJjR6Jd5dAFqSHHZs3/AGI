import sys
import os
import time
import threading
import pickle
import psutil
import math
import random
import warnings
import numpy as np
import cv2
import mss
import torch
import torch.nn as nn
import torch.optim as optim
from PyQt5 import QtWidgets, QtCore, QtGui
from pynput import keyboard
import pyautogui

warnings.filterwarnings("ignore")

pyautogui.FAILSAFE = False

DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
BASE_DIR = os.path.join(DESKTOP_PATH, "AAA")

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

global_running = True
global_optimizing = False
global_pause_recording = False

screen_width, screen_height = pyautogui.size()
center_x, center_y = screen_width // 2, screen_height // 2

current_fps = 30
current_scale = 0.5
max_vram_gb = 4.0
max_ram_gb = 16.0
max_buffer_gb = 10.0

lock = threading.Lock()

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ActionPredictor(nn.Module):
    def __init__(self):
        super(ActionPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64 + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    
    def forward(self, embedding, action):
        b, c, h, w = embedding.size()
        embedding_flat = embedding.view(b, -1)
        if embedding_flat.size(1) != 64:
            matcher = nn.Linear(embedding_flat.size(1), 64).to(embedding.device)
            embedding_flat = matcher(embedding_flat)
        x = torch.cat([embedding_flat, action], dim=1)
        return self.fc(x)

class ExperienceBuffer:
    def __init__(self):
        self.buffer = []
    
    def add(self, state_img, mouse_state, reward, td_error, screen_novelty):
        data = {
            "img": state_img.half(),
            "mouse": mouse_state,
            "reward": reward,
            "error": td_error,
            "novelty": screen_novelty
        }
        self.buffer.append(data)
        self.enforce_limit()

    def enforce_limit(self):
        if len(self.buffer) % 50 != 0:
            return
        
        try:
            sample_size = 0
            if len(self.buffer) > 0:
                s = self.buffer[0]
                img_size = s["img"].element_size() * s["img"].nelement()
                mouse_size = s["mouse"].element_size() * s["mouse"].nelement()
                sample_size = img_size + mouse_size + 64 
            
            total_size_gb = (len(self.buffer) * sample_size) / (1024**3)
            
            if total_size_gb > max_buffer_gb:
                self.buffer.sort(key=lambda x: x["reward"])
                cut_idx = int(len(self.buffer) * 0.2)
                self.buffer = self.buffer[cut_idx:]
        except:
            if len(self.buffer) > 5000:
                self.buffer = self.buffer[-4000:]

    def sample(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return self.buffer
        return random.sample(self.buffer, batch_size)

    def save(self):
        path = os.path.join(BASE_DIR, "experience_pool.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(self.buffer, f)
        except:
            pass

exp_buffer = ExperienceBuffer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ai_model = Autoencoder().to(device)
action_model = ActionPredictor().to(device)

def get_mouse_state():
    x_phys, y_phys = pyautogui.position()
    x = x_phys - center_x
    y = center_y - y_phys
    
    left_btn = 0
    right_btn = 0
    
    try:
        import win32api
        if win32api.GetKeyState(0x01) < 0: left_btn = 1
        if win32api.GetKeyState(0x02) < 0: right_btn = 1
    except:
        pass

    status = 0
    if left_btn and right_btn: status = 3
    elif left_btn: status = 1
    elif right_btn: status = 2
    
    return [x / screen_width, y / screen_height, status / 3.0, 1.0]

class ResourceMonitor(threading.Thread):
    def run(self):
        global current_fps, current_scale
        while global_running:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            
            gpu = 0
            vram = 0
            if torch.cuda.is_available():
                try:
                    vram = torch.cuda.memory_allocated() / (1024**3) / max_vram_gb * 100
                except: pass

            m_val = max(cpu, mem, gpu, vram)
            
            with lock:
                if m_val > 80:
                    current_fps = max(1, current_fps - 2)
                    current_scale = max(0.05, current_scale * 0.9)
                elif m_val < 60:
                    current_fps = min(120, current_fps + 2)
                    current_scale = min(1.0, current_scale * 1.05)
            
            time.sleep(1)

class AgentThread(threading.Thread):
    def run(self):
        global global_pause_recording
        sct = mss.mss()
        criterion = nn.MSELoss()

        while global_running:
            if global_optimizing or global_pause_recording:
                time.sleep(0.1)
                continue

            start_time = time.time()

            with lock:
                fps = current_fps
                scale = current_scale

            try:
                monitor = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
                img_np = np.array(sct.grab(monitor))
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)
                
                h, w = img_np.shape[:2]
                target_h = int(h * scale)
                target_w = int(w * scale)
                
                target_h = (target_h // 4) * 4
                target_w = (target_w // 4) * 4
                if target_h < 4: target_h = 4
                if target_w < 4: target_w = 4
                
                img_resized = cv2.resize(img_np, (target_w, target_h))
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)

                mouse_val = get_mouse_state()
                mouse_tensor = torch.tensor([mouse_val], dtype=torch.float32).to(device)

                with torch.no_grad():
                    recon = ai_model(img_tensor)
                    screen_novelty = criterion(recon, img_tensor).item()
                    
                    latent = ai_model.encoder(img_tensor)
                    pred_mouse = action_model(latent, mouse_tensor) 
                    
                action_novelty = torch.mean((pred_mouse - mouse_tensor)**2).item()
                
                survival_penalty = 0.0001
                reward = (screen_novelty * action_novelty) - survival_penalty
                
                td_error = screen_novelty + action_novelty

                exp_buffer.add(img_tensor.cpu(), mouse_tensor.cpu(), reward, td_error, screen_novelty)

                pred_np = pred_mouse.cpu().numpy()[0]
                dx_norm = float(pred_np[0] * 10)
                dy_norm = float(pred_np[1] * 10)
                dx = int(round(dx_norm))
                dy = int(round(dy_norm))
                if dx == 0 and abs(dx_norm) > 0.2: dx = 1 if dx_norm > 0 else -1
                if dy == 0 and abs(dy_norm) > 0.2: dy = 1 if dy_norm > 0 else -1
                if dx != 0 or dy != 0:
                    try:
                        pyautogui.moveRel(dx, -dy, _pause=False)
                    except: pass
            
            except Exception:
                pass

            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / fps) - elapsed)
            time.sleep(sleep_time)

class SciFiWindow(QtWidgets.QWidget):
    optimizer_signal = QtCore.pyqtSignal()
    finished_signal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.resize(600, 400)
        self.center()
        self.initUI()
        self.optimizer_signal.connect(self.run_optimization)
        self.finished_signal.connect(self.on_optimization_finished)

    def center(self):
        frame = self.frameGeometry()
        center_point = QtWidgets.QDesktopWidget().availableGeometry().center()
        frame.moveCenter(center_point)
        self.move(frame.topLeft())

    def initUI(self):
        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(20, 20, 20, 20)
        card = QtWidgets.QFrame()
        card.setObjectName("card")
        card_layout = QtWidgets.QVBoxLayout()
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(20)
        self.label = QtWidgets.QLabel("SYSTEM: ONLINE")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.info_label = QtWidgets.QLabel("AI AGENT RUNNING...")
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.addWidget(self.label)
        card_layout.addStretch()
        card_layout.addWidget(self.info_label)
        card_layout.addWidget(self.progress)
        card.setLayout(card_layout)
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 0)
        shadow.setColor(QtGui.QColor(0, 240, 255, 150))
        card.setGraphicsEffect(shadow)
        outer.addWidget(card)
        self.setLayout(outer)
        style = """
            #card {
                background-color: qradialgradient(cx:0.5, cy:0.5, fx:0.5, fy:0.5, radius:1.0, stop:0 #0a1224, stop:1 #050a14);
                border: 2px solid #00f0ff;
                border-radius: 16px;
            }
            QLabel {
                color: #00f0ff;
                font-family: Consolas;
                font-size: 18px;
                font-weight: bold;
                border: none;
            }
            QProgressBar {
                border: 1px solid #00f0ff;
                background-color: #0a0f1f;
                height: 26px;
                text-align: center;
                color: #00f0ff;
            }
            QProgressBar::chunk {
                background-color: #00f0ff;
            }
        """
        self.setStyleSheet(style)

    def trigger_optimization(self):
        self.label.setText("SYSTEM: OPTIMIZING")
        self.info_label.setText("TRAINING NEURAL NETWORK...")
        self.progress.setValue(0)
        self.optimizer_signal.emit()

    def run_optimization(self):
        t = threading.Thread(target=self._optimize_task)
        t.start()

    def _optimize_task(self):
        global global_optimizing
        samples = exp_buffer.sample(32)
        total_steps = 100
        
        optimizer = optim.Adam(list(ai_model.parameters()) + list(action_model.parameters()), lr=1e-3)
        loss_fn = nn.MSELoss()

        for i in range(total_steps):
            if not global_running: break
            
            if len(samples) > 0:
                for batch in samples:
                    img = batch["img"].float().to(device)
                    mse_mouse = batch["mouse"].float().to(device)
                    
                    optimizer.zero_grad()
                    recon = ai_model(img)
                    loss_ae = loss_fn(recon, img)
                    
                    latent = ai_model.encoder(img).detach()
                    pred_act = action_model(latent, mse_mouse)
                    loss_act = loss_fn(pred_act, mse_mouse) 
                    
                    loss = loss_ae + loss_act
                    loss.backward()
                    optimizer.step()

            progress_val = int((i + 1) / total_steps * 100)
            QtCore.QMetaObject.invokeMethod(self.progress, "setValue", QtCore.Q_ARG(int, progress_val))
            time.sleep(0.01)

        torch.save(ai_model.state_dict(), os.path.join(BASE_DIR, "ai_model.pth"))
        exp_buffer.save()
        self.finished_signal.emit()

    def on_optimization_finished(self):
        QtWidgets.QMessageBox.information(self, "SYSTEM", "OPTIMIZATION COMPLETE.\nDATA SAVED.")
        self.label.setText("SYSTEM: ONLINE")
        self.info_label.setText("AI AGENT RUNNING...")
        self.progress.setValue(0)
        global global_optimizing, global_pause_recording
        global_optimizing = False
        global_pause_recording = False

class InputHandler:
    def __init__(self, gui_window):
        self.gui = gui_window
        self.listener_k = keyboard.Listener(on_press=self.on_press)
        self.listener_k.start()

    def on_press(self, key):
        global global_running, global_optimizing, global_pause_recording
        if key == keyboard.Key.esc:
            global_running = False
            QtWidgets.QApplication.quit()
            return False
        
        if key == keyboard.Key.enter:
            if not global_optimizing:
                global_pause_recording = True
                global_optimizing = True
                self.gui.trigger_optimization()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SciFiWindow()
    window.show()

    monitor_thread = ResourceMonitor()
    monitor_thread.start()

    agent_thread = AgentThread()
    agent_thread.start()

    input_handler = InputHandler(window)

    sys.exit(app.exec_())
