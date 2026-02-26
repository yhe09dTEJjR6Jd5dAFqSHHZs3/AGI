仅用一个Python脚本，实现一个“活在浏览器里的AI”，越聪明越好。

出现错误时，在控制台输出详细信息，避免静默失败。

初始化阶段：检查文件是否存在：AI模型、经验池（在电脑桌面上的AAA文件夹内，若缺失，自动生成）

工作阶段：AI可以操控鼠标（在浏览器窗口范围内）和键盘（除了ESC）

结束阶段：若检测到用户按下ESC，终止程序。

操作系统：Windows 11
显示器分辨率：2560×1600
CPU：12th Gen Intel(R) Core(TM) i7-12650H
基准速度：2.30 GHz
插槽：1
内核：10
逻辑处理器：16
内存：15.8GB
GPU：NVIDIA GeForce RTX 3050 Ti
显存：4GB
浏览器：Mozilla Firefox, Portable Edition
默认文件路径：E:\FirefoxPortable\FirefoxPortable.exe

当经验池大小＞20GB时，丢弃最旧数据，直到经验池大小＜20GB。

需要预防的报错：

FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead.

FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.

UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True.

