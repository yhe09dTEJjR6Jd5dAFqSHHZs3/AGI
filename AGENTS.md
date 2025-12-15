仅用1个Python脚本实现一个“活在电脑里的人脑”，严禁使用注释或OCR。

窗口：
出现问题时，在窗口上显示详细信息，避免静默失败。
用户点击“睡眠”按钮后，如果此时程序正处于学习模式：暂停记录并保存相关数据，进入睡眠模式。
用户点击“早停”按钮后，如果此时程序正处于睡眠模式：停止优化，保存AI模型以及相关数据，在窗口上显示详细信息，然后进入学习模式。
用户点击“训练”按钮后，如果此时程序正处于学习模式：最小化窗口，进入训练模式。

初始化阶段：
检查文件是否存在：若缺失，自动生成。
AI模型、记录的数据，保存在电脑桌面上的AAA文件夹。

一切就绪后，进入学习模式。
学习模式：持续记录以下数据：时间戳、若干个视觉画面（256×160）、每个视觉画面在电脑屏幕上的位置、鼠标（滚轮、左键按下时的位置、左键按下时的时间戳、左键释放时的位置、左键释放时的时间戳、右键按下时的位置、右键按下时的时间戳、右键释放时的位置、右键释放时的时间戳、轨迹）、操作来源（人）、其他。

AI试图最小化总损失L。
L受到各种因素影响，包括但不限于：对来自人类的鼠标操作的预测误差、对视觉画面的预测误差、生存惩罚。

睡眠模式：
从经验池中挑选部分样本，优化AI。
用户可通过窗口上的两个进度条（百分比，两位小数）和提示文本，观察数据扫描和AI优化的进度。
优化完成后，进度条归零，保存AI模型以及相关数据，在窗口上显示详细信息，进入学习模式。

训练模式：
①持续记录以下数据：时间戳、若干个视觉画面（256×160）、每个视觉画面在电脑屏幕上的位置、鼠标（滚轮、左键按下时的位置、左键按下时的时间戳、左键释放时的位置、左键释放时的时间戳、右键按下时的位置、右键按下时的时间戳、右键释放时的位置、右键释放时的时间戳、轨迹）、操作来源（AI）、其他。
②AI在电脑屏幕上输出鼠标操作。

退出训练模式的唯一条件：用户按下ESC。
用户按下ESC后：暂停输出鼠标操作，保存相关数据，退出训练模式，窗口从最小化状态恢复到前台可见，进入学习模式。

视觉画面包括：整个电脑屏幕、鼠标周围区域、AI注意力所在区域。
AI的注意力有可能会在预测的下一目标点附近，也有可能会在帧差分最大的区域。

我的电脑：
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

当经验池大小＞20GB时，丢弃最旧数据，直到经验池大小＜20GB。

需要预防的报错：

FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead.

FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.

UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True.
