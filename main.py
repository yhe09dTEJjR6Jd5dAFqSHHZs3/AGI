import sys
import os
import time
import random
import glob
import pickle
import threading
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
        t_w, t_h = int(screen_w * 0.1), int(screen_h * 0.1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            props = torch.cuda.get_device_properties(0)
            if props.total_memory > 4 * 1024**3:
                fraction = (4 * 1024**3) / props.total_memory
                torch.cuda.set_per_process_memory_fraction(fraction, 0)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                    nn.Flatten()
                )
                with torch.no_grad():
                    d = torch.zeros(1, 3, t_h, t_w)
                    o = self.enc(d).shape[1]
                self.lstm = nn.LSTM(o + 8, 256, batch_first=True)
                self.dec_h, self.dec_w = t_h // 4, t_w // 4
                self.fc_s = nn.Linear(256, 32 * self.dec_h * self.dec_w)
                self.dec = nn.Sequential(
                    nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ReLU(),
                    nn.ConvTranspose2d(16, 3, 3, 2, 1, 1), nn.Sigmoid()
                )
                self.head_m = nn.Sequential(nn.Linear(256, 8), nn.Tanh())

            def forward(self, img, m):
                b, s, c, h, w = img.size()
                e = self.enc(img.view(b*s, c, h, w)).view(b, s, -1)
                o, _ = self.lstm(torch.cat([e, m], 2))
                l = o[:, -1, :]
                scr = self.dec(self.fc_s(l).view(-1, 32, self.dec_h, self.dec_w))
                if scr.shape[2:] != (t_h, t_w):
                    scr = F.interpolate(scr, (t_h, t_w), mode='bilinear')
                return scr, self.head_m(l)

        net = Net().to(device)
        opt = optim.Adam(net.parameters(), lr=1e-4)
        m_path = os.path.join(model_dir, "core.pth")
        
        if os.path.exists(m_path):
            try:
                c = torch.load(m_path, map_location=device)
                net.load_state_dict(c['n'])
                opt.load_state_dict(c['o'])
            except: pass

        ctx = {
            "mode": "LEARNING",
            "last_act": time.time(),
            "act_start": 0,
            "stop": False,
            "m": {"x":0,"y":0,"dx":0,"dy":0,"l":0,"r":0,"dl":0,"dr":0}
        }

        def on_mov(x, y):
            ctx["m"]["dx"] = x - ctx["m"]["x"]
            ctx["m"]["dy"] = y - ctx["m"]["y"]
            ctx["m"]["x"], ctx["m"]["y"] = x, y
            if ctx["mode"] == "ACTIVE" and (abs(ctx["m"]["dx"])>2 or abs(ctx["m"]["dy"])>2):
                ctx["mode"] = "LEARNING"
                ctx["last_act"] = time.time()
            elif ctx["mode"] == "LEARNING":
                ctx["last_act"] = time.time()

        def on_clk(x, y, b, p):
            ctx["last_act"] = time.time()
            v = 1 if p else 0
            if b == mouse.Button.left: ctx["m"]["l"] = v
            elif b == mouse.Button.right: ctx["m"]["r"] = v
            ctx["m"]["dl"] = 1 if (ctx["m"]["l"] and ctx["m"]["r"]) else 0
            ctx["m"]["dr"] = 1 if (not ctx["m"]["l"] and not ctx["m"]["r"]) else 0
            if ctx["mode"] == "ACTIVE": ctx["mode"] = "LEARNING"

        def on_key(k):
            ctx["last_act"] = time.time()
            if k == keyboard.Key.esc:
                ctx["stop"] = True
                return False
            if ctx["mode"] == "ACTIVE": ctx["mode"] = "LEARNING"

        ml = mouse.Listener(on_move=on_mov, on_click=on_clk, on_scroll=lambda x,y,dx,dy: on_clk(0,0,mouse.Button.middle,True))
        kl = keyboard.Listener(on_press=on_key)
        ml.start()
        kl.start()

        freq, seq_len = 10, 5
        buf_img, buf_mouse = [], []
        last_pred, prev_loss = None, 1.0

        while not ctx["stop"]:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            vram = 0
            if torch.cuda.is_available():
                try:
                    vram = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100
                except: pass
            load = max(cpu, mem, vram)
            
            if load > 61.8:
                freq = max(1, freq - 1)
                seq_len = max(1, seq_len - 1)
            elif load < 38.2:
                freq = min(100, freq + 1)
                seq_len = min(100, seq_len + 1)

            start_t = time.time()

            if ctx["mode"] == "SLEEP":
                try:
                    root = tk.Tk()
                    root.attributes("-topmost", True)
                    root.geometry("300x50")
                    root.overrideredirect(True)
                    pb = ttk.Progressbar(root, length=280, mode='determinate')
                    pb.pack(pady=10)
                    
                    files = sorted(glob.glob(os.path.join(pool_dir, "*.pkl")))
                    if not files:
                        time.sleep(1)
                    else:
                        samples = []
                        f_idx = np.random.choice(len(files), min(len(files), 200), replace=False)
                        for i in f_idx:
                            try:
                                with open(files[i], "rb") as f: samples.append(pickle.load(f))
                            except: pass
                        
                        if samples:
                            samples.sort(key=lambda x: x['t'])
                            while len(samples) <= seq_len + 1: samples.extend(samples)
                            
                            net.train()
                            for ep in range(5):
                                if ctx["stop"]: break
                                pb['value'] = (ep+1)*20
                                root.update()
                                
                                idx = random.randint(0, len(samples) - seq_len - 1)
                                s_data = samples[idx : idx + seq_len]
                                t_data = samples[idx + seq_len]
                                
                                si = torch.tensor(np.array([x['s'] for x in s_data]), dtype=torch.float32).permute(0,3,1,2).unsqueeze(0).to(device)/255.0
                                mi = torch.tensor(np.array([x['m'] for x in s_data]), dtype=torch.float32).unsqueeze(0).to(device)
                                st = torch.tensor(t_data['s'], dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)/255.0
                                mt = torch.tensor(t_data['m'], dtype=torch.float32).unsqueeze(0).to(device)
                                
                                opt.zero_grad()
                                ps, pm = net(si, mi)
                                loss = nn.MSELoss()(ps, st) + nn.MSELoss()(pm, mt)
                                loss.backward()
                                opt.step()
                            
                    torch.save({'n': net.state_dict(), 'o': opt.state_dict()}, m_path)
                    
                    all_f = sorted(glob.glob(os.path.join(pool_dir, "*.pkl")), key=os.path.getctime)
                    sz = sum(os.path.getsize(f) for f in all_f)
                    while sz > 20 * 1024**3 and all_f:
                        rm = all_f.pop(0)
                        try:
                            sz -= os.path.getsize(rm)
                            os.remove(rm)
                        except: pass
                            
                    root.destroy()
                except:
                    try: root.destroy()
                    except: pass
                    
                ctx["mode"] = "ACTIVE"
                ctx["act_start"] = time.time()
                buf_img, buf_mouse = [], []
                last_pred = None
            
            else:
                s_raw = np.array(pyautogui.screenshot())
                s_s = cv2.resize(cv2.cvtColor(s_raw, cv2.COLOR_RGB2BGR), (t_w, t_h))
                m_vec = [ctx["m"]["x"]/screen_w, ctx["m"]["y"]/screen_h, ctx["m"]["dx"]/screen_w, 
                         ctx["m"]["dy"]/screen_h, ctx["m"]["l"], ctx["m"]["r"], ctx["m"]["dl"], ctx["m"]["dr"]]
                
                cur_t = torch.tensor(s_s, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)/255.0
                
                rew = 0.0
                if last_pred is not None:
                    loss = torch.mean(torch.abs(last_pred - cur_t)).item()
                    rew = 1.0 if loss < prev_loss else -1.0
                    prev_loss = loss
                
                pkt = {"t": time.time(), "r": rew, "s": s_s, "m": m_vec}
                try:
                    with open(os.path.join(pool_dir, f"{time.time_ns()}.pkl"), "wb") as f:
                        pickle.dump(pkt, f)
                except: pass
                
                buf_img.append(s_s)
                buf_mouse.append(m_vec)
                if len(buf_img) > seq_len:
                    buf_img.pop(0)
                    buf_mouse.pop(0)
                
                if len(buf_img) == seq_len:
                    inp_s = torch.tensor(np.array(buf_img), dtype=torch.float32).permute(0,3,1,2).unsqueeze(0).to(device)/255.0
                    inp_m = torch.tensor(np.array(buf_mouse), dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        last_pred, pm = net(inp_s, inp_m)
                    
                    if ctx["mode"] == "ACTIVE":
                        tx, ty = pm[0,0].item()*screen_w, pm[0,1].item()*screen_h
                        nx, ny = random.gauss(0, 5), random.gauss(0, 5)
                        try:
                            pyautogui.moveTo(tx+nx, ty+ny, duration=0.01)
                            if pm[0,4].item() > 0.5: pyautogui.mouseDown(); pyautogui.mouseUp()
                            if pm[0,5].item() > 0.5: pyautogui.click(button='right')
                        except: pass
                        
                        if time.time() - ctx["act_start"] > 60:
                            ctx["mode"] = "SLEEP"
                
                if ctx["mode"] == "LEARNING" and time.time() - ctx["last_act"] > 10:
                    ctx["mode"] = "ACTIVE"
                    ctx["act_start"] = time.time()
            
            delta = time.time() - start_t
            if delta < 1.0/freq: time.sleep(1.0/freq - delta)

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
