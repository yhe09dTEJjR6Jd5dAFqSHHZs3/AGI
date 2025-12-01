import sys
import os
import time
import threading
import pickle
import math
import random
import warnings
import subprocess
import importlib
from collections import deque

warnings.filterwarnings("ignore")

def import_or_install(module_name, package_name=None):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pkg = package_name or module_name
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except Exception:
            sys.exit(1)
        return importlib.import_module(module_name)
    except Exception:
        sys.exit(1)

psutil = import_or_install("psutil")
np = import_or_install("numpy")
cv2 = import_or_install("cv2", "opencv-python")
mss = import_or_install("mss")
torch = import_or_install("torch")
torch.backends.cudnn.benchmark = True
nn = torch.nn
optim = torch.optim
F = torch.nn.functional
autocast = import_or_install("torch.cuda.amp").autocast
GradScaler = import_or_install("torch.cuda.amp").GradScaler
QtWidgets = import_or_install("PyQt5").QtWidgets
QtCore = import_or_install("PyQt5").QtCore
QtGui = import_or_install("PyQt5").QtGui
keyboard = import_or_install("pynput.keyboard")
pyautogui = import_or_install("pyautogui")
ctypes = import_or_install("ctypes")
platform = import_or_install("platform")

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
resolution_factor = 1.0
temporal_context_window = 4
max_vram_gb = 4.0
max_ram_gb = 16.0
max_buffer_gb = 10.0
standard_res = (128, 128)
latent_pool = 4

lock = threading.Lock()
mouse_left_down = False
mouse_right_down = False

def scaled_standard_res():
    with lock:
        factor = resolution_factor
    return max(1, int(standard_res[0] * factor)), max(1, int(standard_res[1] * factor))

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
        self.pool = nn.AdaptiveAvgPool2d((latent_pool, latent_pool))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return x

    def decode(self, latent, output_size):
        x = self.decoder(latent)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent, x.shape[2:])

class FuturePredictor(nn.Module):
    def __init__(self):
        super(FuturePredictor, self).__init__()
        embed_dim = 64 * latent_pool * latent_pool
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, aggregated):
        return self.fc(aggregated)

class ActionPredictor(nn.Module):
    def __init__(self):
        super(ActionPredictor, self).__init__()
        embed_dim = 64 * latent_pool * latent_pool
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, aggregated):
        return self.fc(aggregated)

class ExperienceBuffer:
    def __init__(self):
        self.buffer = []

    def add(self, state_img, mouse_state, reward, td_error, screen_novelty, latent_tensor):
        img = state_img.float()
        if img.dim() == 4:
            img = img[0]
        res_h, res_w = scaled_standard_res()
        img = F.interpolate(img.unsqueeze(0), size=(res_h, res_w), mode="bilinear", align_corners=False).squeeze(0).half()
        mouse = mouse_state.float()
        if mouse.dim() > 1:
            mouse = mouse[0]
        latent_store = latent_tensor.detach().cpu().half()
        data = {
            "img": img,
            "mouse": mouse,
            "reward": reward,
            "error": td_error,
            "novelty": screen_novelty,
            "latent": latent_store
        }
        self.buffer.append(data)
        self.enforce_limit()

    def _sample_bytes(self, entry):
        img_size = entry["img"].element_size() * entry["img"].nelement()
        mouse_size = entry["mouse"].element_size() * entry["mouse"].nelement()
        latent_size = entry["latent"].element_size() * entry["latent"].nelement()
        meta_size = 24
        return img_size + mouse_size + latent_size + meta_size

    def _buffer_size_gb(self):
        total_bytes = 0
        for item in self.buffer:
            total_bytes += self._sample_bytes(item)
        return total_bytes / (1024**3)

    def enforce_limit(self):
        total_gb = self._buffer_size_gb()
        while total_gb > max_buffer_gb and len(self.buffer) > 0:
            min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i]["reward"])
            removed = self.buffer.pop(min_idx)
            total_gb -= self._sample_bytes(removed) / (1024**3)

    def sample_sequences(self, batch_size, window):
        if len(self.buffer) <= window:
            return []
        sequences = []
        max_start = len(self.buffer) - window - 1
        for _ in range(batch_size):
            idx = random.randint(0, max_start)
            context_entries = self.buffer[idx:idx + window]
            target_entry = self.buffer[idx + window]
            sequences.append((context_entries, target_entry))
        return sequences

    def save(self):
        path = os.path.join(BASE_DIR, "experience_pool.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(self.buffer, f)
        except Exception:
            pass

exp_buffer = ExperienceBuffer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ai_model = Autoencoder().to(device)
future_model = FuturePredictor().to(device)
action_model = ActionPredictor().to(device)
scaler = GradScaler(enabled=device.type == "cuda")

def ensure_files():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    buffer_path = os.path.join(BASE_DIR, "experience_pool.pkl")
    model_path = os.path.join(BASE_DIR, "ai_model.pth")
    action_path = os.path.join(BASE_DIR, "action_model.pth")
    future_path = os.path.join(BASE_DIR, "future_model.pth")
    if not os.path.exists(buffer_path):
        with open(buffer_path, "wb") as f:
            pickle.dump([], f)
    if not os.path.exists(model_path):
        torch.save(ai_model.state_dict(), model_path)
    if not os.path.exists(action_path):
        torch.save(action_model.state_dict(), action_path)
    if not os.path.exists(future_path):
        torch.save(future_model.state_dict(), future_path)

def load_state():
    global exp_buffer
    buffer_path = os.path.join(BASE_DIR, "experience_pool.pkl")
    model_path = os.path.join(BASE_DIR, "ai_model.pth")
    action_path = os.path.join(BASE_DIR, "action_model.pth")
    future_path = os.path.join(BASE_DIR, "future_model.pth")
    if os.path.exists(buffer_path):
        try:
            with open(buffer_path, "rb") as f:
                exp_buffer.buffer = pickle.load(f)
        except Exception:
            exp_buffer.buffer = []
    if os.path.exists(model_path):
        try:
            ai_model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception:
            pass
    if os.path.exists(action_path):
        try:
            action_model.load_state_dict(torch.load(action_path, map_location=device))
        except Exception:
            pass
    if os.path.exists(future_path):
        try:
            future_model.load_state_dict(torch.load(future_path, map_location=device))
        except Exception:
            pass

def save_state():
    torch.save(ai_model.state_dict(), os.path.join(BASE_DIR, "ai_model.pth"))
    torch.save(action_model.state_dict(), os.path.join(BASE_DIR, "action_model.pth"))
    torch.save(future_model.state_dict(), os.path.join(BASE_DIR, "future_model.pth"))
    exp_buffer.save()

def get_mouse_state():
    x_phys, y_phys = pyautogui.position()
    x = x_phys - center_x
    y = center_y - y_phys
    left_btn = 0
    right_btn = 0
    try:
        win32api = import_or_install("win32api")
        if win32api.GetKeyState(0x01) < 0: left_btn = 1
        if win32api.GetKeyState(0x02) < 0: right_btn = 1
    except Exception:
        pass
    status = 0
    if left_btn and right_btn: status = 3
    elif left_btn: status = 1
    elif right_btn: status = 2
    return [x / screen_width, y / screen_height, status / 3.0, 1.0]

def apply_mouse_buttons(pred_status):
    global mouse_left_down, mouse_right_down
    status_idx = int(round(max(0.0, min(1.0, pred_status)) * 3))
    target_left = status_idx in (1, 3)
    target_right = status_idx in (2, 3)
    if target_left != mouse_left_down:
        try:
            if target_left:
                if platform.system().lower().startswith("win"):
                    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
                pyautogui.mouseDown(button="left")
            else:
                if platform.system().lower().startswith("win"):
                    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
                pyautogui.mouseUp(button="left")
            mouse_left_down = target_left
        except Exception:
            pass
    if target_right != mouse_right_down:
        try:
            if target_right:
                if platform.system().lower().startswith("win"):
                    ctypes.windll.user32.mouse_event(0x0008, 0, 0, 0, 0)
                pyautogui.mouseDown(button="right")
            else:
                if platform.system().lower().startswith("win"):
                    ctypes.windll.user32.mouse_event(0x0010, 0, 0, 0, 0)
                pyautogui.mouseUp(button="right")
            mouse_right_down = target_right
        except Exception:
            pass

def move_mouse(pred_abs_x, pred_abs_y, pred_status):
    current = get_mouse_state()
    dx = (pred_abs_x - current[0]) * screen_width
    dy = (pred_abs_y - current[1]) * screen_height
    if platform.system().lower().startswith("win"):
        try:
            ctypes.windll.user32.mouse_event(0x0001, int(dx), int(-dy), 0, 0)
        except Exception:
            pass
    else:
        try:
            pyautogui.moveRel(int(dx), int(-dy), _pause=False)
        except Exception:
            pass
    apply_mouse_buttons(pred_status)

def aggregate_latents(latents):
    embed_dim = 64 * latent_pool * latent_pool
    if len(latents) == 0:
        return torch.zeros(1, embed_dim * 2, device=device)
    stacked = torch.stack(latents).view(len(latents), -1)
    mean = stacked.mean(dim=0, keepdim=True)
    std = stacked.std(dim=0, keepdim=True) + 1e-6
    return torch.cat([mean, std], dim=1)

class ResourceMonitor(threading.Thread):
    def run(self):
        global current_fps, current_scale, resolution_factor, temporal_context_window
        prev_m = None
        while global_running:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            gpu = 0
            vram = 0
            if torch.cuda.is_available():
                try:
                    vram_use = torch.cuda.memory_allocated() / (1024**3)
                    vram = min(100.0, vram_use / max_vram_gb * 100)
                except Exception:
                    vram = 0
            m_val = max(cpu, mem, gpu, vram)
            delta = 0 if prev_m is None else m_val - prev_m
            prev_m = m_val
            high_load = m_val > 90
            with lock:
                if high_load or delta > 0.5:
                    current_fps = max(1, int(current_fps * 0.9))
                    current_scale = max(0.05, current_scale * 0.9)
                    resolution_factor = max(0.2, resolution_factor * 0.9)
                    temporal_context_window = max(2, temporal_context_window - 1)
                elif delta < -0.5 and m_val < 85:
                    current_fps = min(120, int(current_fps * 1.05) + 1)
                    current_scale = min(1.0, current_scale * 1.05)
                    resolution_factor = min(1.0, resolution_factor * 1.05)
                    temporal_context_window = min(12, temporal_context_window + 1)
            time.sleep(1)

class AgentThread(threading.Thread):
    def run(self):
        global global_pause_recording
        sct = mss.mss()
        criterion = nn.MSELoss()
        context_latents = deque(maxlen=12)
        while global_running:
            if global_optimizing or global_pause_recording:
                time.sleep(0.1)
                continue
            start_time = time.time()
            with lock:
                fps = current_fps
                scale = current_scale
                res_factor = resolution_factor
                window_len = temporal_context_window
            try:
                monitor = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
                img_np = np.array(sct.grab(monitor))
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)
                h, w = img_np.shape[:2]
                target_h = int(h * scale * res_factor)
                target_w = int(w * scale * res_factor)
                target_h = (target_h // 4) * 4
                target_w = (target_w // 4) * 4
                if target_h < 4: target_h = 4
                if target_w < 4: target_w = 4
                img_resized = cv2.resize(img_np, (target_w, target_h))
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    aggregated = aggregate_latents(list(context_latents)[-window_len:])
                    predicted_latent_vec = future_model(aggregated)
                    predicted_latent = predicted_latent_vec.view(1, 64, latent_pool, latent_pool)
                    predicted_img = ai_model.decode(predicted_latent, img_tensor.shape[2:])
                mouse_val = get_mouse_state()
                mouse_tensor = torch.tensor([mouse_val], dtype=torch.float32).to(device)
                with torch.no_grad():
                    latent = ai_model.encode(img_tensor)
                context_latents.append(latent.detach())
                with torch.no_grad():
                    pred_mouse_raw = action_model(aggregated)
                pred_mouse = torch.zeros_like(mouse_tensor)
                pred_mouse[:, 0] = torch.tanh(pred_mouse_raw[:, 0]) * 0.5
                pred_mouse[:, 1] = torch.tanh(pred_mouse_raw[:, 1]) * 0.5
                pred_mouse[:, 2] = torch.sigmoid(pred_mouse_raw[:, 2])
                pred_mouse[:, 3] = torch.ones_like(pred_mouse[:, 3])
                screen_novelty = criterion(predicted_img, img_tensor).item()
                action_novelty = torch.mean((pred_mouse - mouse_tensor) ** 2).item()
                survival_penalty = 0.0001
                reward = (screen_novelty * action_novelty) - survival_penalty
                td_error = screen_novelty + action_novelty
                exp_buffer.add(img_tensor.cpu(), mouse_tensor.cpu(), reward, td_error, screen_novelty, latent.cpu())
                pred_np = pred_mouse.cpu().numpy()[0]
                move_mouse(float(pred_np[0]), float(pred_np[1]), float(pred_np[2]))
            except Exception:
                pass
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / fps) - elapsed)
            time.sleep(sleep_time)

class SciFiWindow(QtWidgets.QWidget):
    optimizer_signal = QtCore.pyqtSignal()
    finished_signal = QtCore.pyqtSignal()
    status_signal = QtCore.pyqtSignal(str)
    info_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.resize(600, 400)
        self.center()
        self.initUI()
        self.optimizer_signal.connect(self.run_optimization)
        self.finished_signal.connect(self.on_optimization_finished)
        self.status_signal.connect(self.label.setText)
        self.info_signal.connect(self.info_label.setText)

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
        self.status_signal.emit("SYSTEM: OPTIMIZING")
        self.info_signal.emit("TRAINING NEURAL NETWORK...")
        QtCore.QMetaObject.invokeMethod(self.progress, "setValue", QtCore.Q_ARG(int, 0))
        self.optimizer_signal.emit()

    def run_optimization(self):
        t = threading.Thread(target=self._optimize_task)
        t.start()

    def _optimize_task(self):
        global global_optimizing, global_pause_recording
        samples = exp_buffer.sample_sequences(64, temporal_context_window)
        if len(samples) == 0:
            global_optimizing = False
            global_pause_recording = False
            self.finished_signal.emit()
            return
        total_steps = 100
        optimizer = optim.Adam(list(ai_model.parameters()) + list(action_model.parameters()) + list(future_model.parameters()), lr=1e-3)
        loss_fn = nn.MSELoss()
        for i in range(total_steps):
            if not global_running:
                break
            batch_indices = list(range(0, len(samples), 8))
            for start in batch_indices:
                batch = samples[start:start + 8]
                context_latent_list = []
                target_imgs = []
                target_mice = []
                target_latents = []
                for context_entries, target_entry in batch:
                    latents = [item["latent"].float() for item in context_entries]
                    context_latent_list.append(latents)
                    target_imgs.append(target_entry["img"].float())
                    target_mice.append(target_entry["mouse"].float())
                    target_latents.append(target_entry["latent"].float())
                optimizer.zero_grad()
                agg_batch = []
                for latents in context_latent_list:
                    agg = aggregate_latents([l.to(device) for l in latents])
                    agg_batch.append(agg)
                aggregated_tensor = torch.cat(agg_batch, dim=0)
                target_imgs_tensor = torch.stack(target_imgs).to(device)
                target_mice_tensor = torch.stack(target_mice).to(device)
                target_latents_tensor = torch.stack(target_latents).to(device)
                with autocast(enabled=device.type == "cuda"):
                    pred_latent_vec = future_model(aggregated_tensor)
                    pred_latent = pred_latent_vec.view(-1, 64, latent_pool, latent_pool)
                    pred_img = ai_model.decode(pred_latent, target_imgs_tensor.shape[2:])
                    pred_mouse_raw = action_model(aggregated_tensor)
                    pred_mouse = torch.zeros_like(target_mice_tensor)
                    pred_mouse[:, 0] = torch.tanh(pred_mouse_raw[:, 0]) * 0.5
                    pred_mouse[:, 1] = torch.tanh(pred_mouse_raw[:, 1]) * 0.5
                    pred_mouse[:, 2] = torch.sigmoid(pred_mouse_raw[:, 2])
                    pred_mouse[:, 3] = torch.ones_like(pred_mouse[:, 3])
                    recon = ai_model(target_imgs_tensor)
                    loss_screen = loss_fn(pred_img, target_imgs_tensor)
                    loss_mouse = loss_fn(pred_mouse, target_mice_tensor)
                    loss_ae = loss_fn(recon, target_imgs_tensor) + loss_fn(target_latents_tensor, ai_model.encode(target_imgs_tensor))
                    loss = loss_screen + loss_mouse + loss_ae
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(list(ai_model.parameters()) + list(action_model.parameters()) + list(future_model.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
            progress_val = int((i + 1) / total_steps * 100)
            QtCore.QMetaObject.invokeMethod(self.progress, "setValue", QtCore.Q_ARG(int, progress_val))
            time.sleep(0.01)
        save_state()
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
    ensure_files()
    load_state()
    global mouse_left_down, mouse_right_down
    init_state = get_mouse_state()
    init_status = int(round(init_state[2] * 3))
    mouse_left_down = init_status in (1, 3)
    mouse_right_down = init_status in (2, 3)
    window = SciFiWindow()
    window.show()
    monitor_thread = ResourceMonitor()
    monitor_thread.start()
    agent_thread = AgentThread()
    agent_thread.start()
    input_handler = InputHandler(window)
    sys.exit(app.exec_())
