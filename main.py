import sys
import os
import time
import random
import glob
import pickle
import psutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pyautogui
import traceback
from pynput import mouse, keyboard
import tkinter as tk
from tkinter import ttk

def main():
    try:
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        torch.backends.cudnn.benchmark = True
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        base_dir = os.path.join(desktop, "AAA")
        pool_dir = os.path.join(base_dir, "experience_pool")
        model_dir = os.path.join(base_dir, "ai_models")
        for d in [base_dir, pool_dir, model_dir]:
            os.makedirs(d, exist_ok=True)
        screen_w, screen_h = pyautogui.size()
        t_w = max(1, int(screen_w * 0.1))
        t_h = max(1, int(screen_h * 0.1))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            if props.total_memory > 4 * 1024 ** 3:
                fraction = (4 * 1024 ** 3) / float(props.total_memory)
                torch.cuda.set_per_process_memory_fraction(fraction, 0)
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Conv2d(3, 16, 3, 2, 1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 3, 2, 1),
                    nn.ReLU(),
                    nn.Flatten()
                )
                with torch.no_grad():
                    d = torch.zeros(1, 3, t_h, t_w)
                    o = self.enc(d).shape[1]
                self.lstm = nn.LSTM(o + 8, 256, batch_first=True)
                self.dec_h = max(1, t_h // 4)
                self.dec_w = max(1, t_w // 4)
                self.fc_s = nn.Linear(256, 32 * self.dec_h * self.dec_w)
                self.dec = nn.Sequential(
                    nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(16, 3, 3, 2, 1, 1),
                    nn.Sigmoid()
                )
                self.head_m = nn.Sequential(nn.Linear(256, 8), nn.Tanh())
            def forward(self, img, m):
                b, s, c, h, w = img.size()
                e = self.enc(img.view(b * s, c, h, w)).view(b, s, -1)
                o, _ = self.lstm(torch.cat([e, m], 2))
                l = o[:, -1, :]
                scr = self.dec(self.fc_s(l).view(-1, 32, self.dec_h, self.dec_w))
                if scr.shape[2] != t_h or scr.shape[3] != t_w:
                    scr = F.interpolate(scr, (t_h, t_w), mode="bilinear", align_corners=False)
                return scr, self.head_m(l)
        net = Net().to(device)
        opt = optim.Adam(net.parameters(), lr=1e-4)
        m_path = os.path.join(model_dir, "core.pth")
        if os.path.exists(m_path):
            c = torch.load(m_path, map_location=device)
            net.load_state_dict(c["n"])
            opt.load_state_dict(c["o"])
        else:
            torch.save({"n": net.state_dict(), "o": opt.state_dict()}, m_path)
        ctx = {
            "mode": "LEARNING",
            "last_act": time.time(),
            "act_start": 0.0,
            "stop": False,
            "m": {"x": 0.0, "y": 0.0, "dx": 0.0, "dy": 0.0, "l": 0.0, "r": 0.0, "dl": 0.0, "dr": 0.0}
        }
        def on_mov(x, y):
            dx = x - ctx["m"]["x"]
            dy = y - ctx["m"]["y"]
            ctx["m"]["dx"] = dx
            ctx["m"]["dy"] = dy
            ctx["m"]["x"] = x
            ctx["m"]["y"] = y
            ctx["last_act"] = time.time()
            if ctx["mode"] == "ACTIVE" and (abs(dx) > 2 or abs(dy) > 2):
                ctx["mode"] = "LEARNING"
        def on_clk(x, y, b, p):
            ctx["last_act"] = time.time()
            v = 1.0 if p else 0.0
            if b == mouse.Button.left:
                ctx["m"]["l"] = v
            elif b == mouse.Button.right:
                ctx["m"]["r"] = v
            l = ctx["m"]["l"]
            r = ctx["m"]["r"]
            ctx["m"]["dl"] = 1.0 if l > 0.5 and r > 0.5 else 0.0
            ctx["m"]["dr"] = 1.0 if l < 0.5 and r < 0.5 else 0.0
            if ctx["mode"] == "ACTIVE":
                ctx["mode"] = "LEARNING"
        def on_key(k):
            ctx["last_act"] = time.time()
            if k == keyboard.Key.esc:
                ctx["stop"] = True
                return False
            if ctx["mode"] == "ACTIVE":
                ctx["mode"] = "LEARNING"
        ml = mouse.Listener(on_move=on_mov, on_click=on_clk, on_scroll=lambda x, y, dx, dy: on_clk(x, y, mouse.Button.middle, True))
        kl = keyboard.Listener(on_press=on_key)
        ml.start()
        kl.start()
        freq = 10
        seq_len = 5
        buf_img = []
        buf_mouse = []
        last_pred = None
        prev_loss = None
        def optimize_model():
            root = tk.Tk()
            root.attributes("-topmost", True)
            root.geometry("360x90")
            root.overrideredirect(True)
            text_var = tk.StringVar()
            label = ttk.Label(root, textvariable=text_var)
            label.pack(pady=5)
            pb = ttk.Progressbar(root, length=320, mode="determinate", maximum=100)
            pb.pack(pady=5)
            text_var.set("优化中 0%")
            try:
                files = sorted(glob.glob(os.path.join(pool_dir, "*.pkl")))
                samples = []
                if files:
                    max_files = min(len(files), 200)
                    indices = np.random.choice(len(files), max_files, replace=False)
                    for index in indices:
                        with open(files[index], "rb") as f:
                            samples.append(pickle.load(f))
                if samples:
                    samples.sort(key=lambda x: x["t"])
                    while len(samples) <= seq_len + 1:
                        samples.extend(samples)
                    net.train()
                    total_epochs = 5
                    for ep in range(total_epochs):
                        if ctx["stop"]:
                            break
                        progress = int((ep + 1) * 100 / total_epochs)
                        pb["value"] = progress
                        text_var.set("优化中 " + str(progress) + "%")
                        root.update_idletasks()
                        root.update()
                        if ctx["stop"]:
                            break
                        max_start = len(samples) - seq_len - 1
                        if max_start <= 0:
                            start_index = 0
                        else:
                            start_index = random.randint(0, max_start)
                        s_data = samples[start_index:start_index + seq_len]
                        t_data = samples[start_index + seq_len]
                        seq_s = torch.tensor(np.array([x["s"] for x in s_data]), dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device) / 255.0
                        seq_m = torch.tensor(np.array([x["m"] for x in s_data]), dtype=torch.float32).unsqueeze(0).to(device)
                        target_s = torch.tensor(t_data["s"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                        target_m = torch.tensor(t_data["m"], dtype=torch.float32).unsqueeze(0).to(device)
                        opt.zero_grad()
                        pred_s, pred_m = net(seq_s, seq_m)
                        loss_s = nn.MSELoss()(pred_s, target_s)
                        loss_m = nn.MSELoss()(pred_m, target_m)
                        loss = loss_s + loss_m
                        loss.backward()
                        opt.step()
                net.eval()
                torch.save({"n": net.state_dict(), "o": opt.state_dict()}, m_path)
                all_files = sorted(glob.glob(os.path.join(pool_dir, "*.pkl")), key=os.path.getctime)
                size_bytes = 0
                for path in all_files:
                    size_bytes += os.path.getsize(path)
                limit_bytes = 20 * 1024 * 1024 * 1024
                index = 0
                while size_bytes > limit_bytes and index < len(all_files):
                    path = all_files[index]
                    file_size = os.path.getsize(path)
                    os.remove(path)
                    size_bytes -= file_size
                    index += 1
            finally:
                try:
                    root.destroy()
                except:
                    pass
        while not ctx["stop"]:
            loop_start = time.time()
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            vram = 0.0
            if torch.cuda.is_available():
                total_vram = float(torch.cuda.get_device_properties(0).total_memory)
                if total_vram > 0:
                    allocated = float(torch.cuda.memory_allocated(0))
                    vram = allocated / total_vram * 100.0
            load = max(cpu, mem, vram)
            if load > 61.8:
                if freq > 1:
                    freq -= 1
                if seq_len > 1:
                    seq_len -= 1
            elif load < 38.2:
                if freq < 100:
                    freq += 1
                if seq_len < 100:
                    seq_len += 1
            now = time.time()
            if ctx["mode"] == "LEARNING" and now - ctx["last_act"] > 10.0:
                ctx["mode"] = "ACTIVE"
                ctx["act_start"] = now
            if ctx["mode"] == "ACTIVE" and ctx["act_start"] > 0 and now - ctx["act_start"] > 60.0:
                ctx["mode"] = "SLEEP"
            if ctx["mode"] == "SLEEP":
                optimize_model()
                if ctx["stop"]:
                    break
                ctx["mode"] = "ACTIVE"
                ctx["act_start"] = time.time()
                buf_img = []
                buf_mouse = []
                last_pred = None
                prev_loss = None
            else:
                s_raw = np.array(pyautogui.screenshot())
                s_s = cv2.resize(cv2.cvtColor(s_raw, cv2.COLOR_RGB2BGR), (t_w, t_h))
                nx = float(screen_w) if screen_w > 0 else 1.0
                ny = float(screen_h) if screen_h > 0 else 1.0
                mx = ctx["m"]["x"] / nx
                my = ctx["m"]["y"] / ny
                mdx = ctx["m"]["dx"] / nx
                mdy = ctx["m"]["dy"] / ny
                mlp = float(ctx["m"]["l"])
                mrp = float(ctx["m"]["r"])
                both_down = float(ctx["m"]["dl"])
                both_up = float(ctx["m"]["dr"])
                m_vec = [mx, my, mdx, mdy, mlp, mrp, both_down, both_up]
                cur_t = torch.tensor(s_s, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                rew = 0.0
                if last_pred is not None:
                    diff = torch.mean(torch.abs(last_pred - cur_t)).item()
                    if prev_loss is None:
                        prev_loss = diff
                    else:
                        if diff < prev_loss:
                            rew = 1.0
                        elif diff > prev_loss:
                            rew = -1.0
                        prev_loss = diff
                pkt = {"t": time.time(), "r": rew, "s": s_s, "m": m_vec}
                file_path = os.path.join(pool_dir, f"{time.time_ns()}.pkl")
                with open(file_path, "wb") as f:
                    pickle.dump(pkt, f)
                buf_img.append(s_s)
                buf_mouse.append(m_vec)
                if len(buf_img) > seq_len:
                    buf_img.pop(0)
                    buf_mouse.pop(0)
                if len(buf_img) == seq_len:
                    seq_s = torch.tensor(np.array(buf_img), dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device) / 255.0
                    seq_m = torch.tensor(np.array(buf_mouse), dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        last_pred, pm = net(seq_s, seq_m)
                    if ctx["mode"] == "ACTIVE":
                        cx = pm[0, 0].item()
                        cy = pm[0, 1].item()
                        tx = max(0, min(screen_w - 1, int(cx * screen_w)))
                        ty = max(0, min(screen_h - 1, int(cy * screen_h)))
                        noise_x = random.gauss(0.0, 5.0)
                        noise_y = random.gauss(0.0, 5.0)
                        pyautogui.moveTo(tx + noise_x, ty + noise_y, duration=0.01)
                        lp = pm[0, 4].item()
                        rp = pm[0, 5].item()
                        if lp > 0.5:
                            pyautogui.mouseDown()
                            pyautogui.mouseUp()
                        if rp > 0.5:
                            pyautogui.click(button="right")
            elapsed = time.time() - loop_start
            if freq < 1:
                freq = 1
            sleep_time = 1.0 / float(freq)
            if elapsed < sleep_time:
                time.sleep(sleep_time - elapsed)
        ml.stop()
        kl.stop()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
