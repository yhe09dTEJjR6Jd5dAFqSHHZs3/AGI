仅用1个Python脚本实现AGI，严禁使用注释或OCR。

出现问题时，在控制台输出错误信息。

初始化阶段：
自动安装缺失库。
检查文件是否存在：若缺失，自动生成。
生成的所有文件，均存放于电脑桌面上的AAA文件夹及其子文件夹下（包括但不限于：经验池、AI模型文件）

一切就绪后，进入学习模式。
学习模式：持续记录以下数据到经验池：时间戳、电脑屏幕画面（256×160）、鼠标（滚轮、左键按下时的位置、左键按下时的时间戳、左键释放时的位置、左键释放时的时间戳、右键按下时的位置、右键按下时的时间戳、右键释放时的位置、右键释放时的时间戳、轨迹）、操作来源（人）

AI试图最小化总损失L。
L受到各种因素影响，包括但不限于：对来自人类的鼠标操作的预测误差、对电脑屏幕画面的预测误差、生存惩罚。

用户在控制台输入“睡眠”后，如果此时程序正处于学习模式：暂停记录并保存相关数据，进入睡眠模式。
睡眠模式：挑选经验池中部分数据（40%人类高质示范 + 30%近期高频数据 + 20%高惊讶度数据 + 10%随机历史存档），开始优化AI。
优化过程中，用户可通过带有提示文本的百分比进度条，观察优化进度。
用户按下任意键盘按键后，如果此时程序正处于睡眠模式：停止优化，保存AI模型以及相关数据，在控制台上输出详细信息，然后进入学习模式。
优化完成后，保存AI模型以及相关数据，在控制台上输出详细信息，然后进入学习模式。

用户在控制台输入“训练”后，如果此时程序正处于学习模式：最小化控制台窗口，进入训练模式。

训练模式：
①持续记录以下数据到经验池：时间戳、电脑屏幕画面（256×160）、鼠标（左键按下时的位置、左键按下时的时间戳、左键释放时的位置、左键释放时的时间戳、轨迹）、操作来源（AI）
②AI在电脑屏幕上输出鼠标操作（点按/长按/拖动）。

用户按下任意键盘按键后，如果此时程序正处于训练模式，暂停输出鼠标操作，保存相关数据，进入学习模式。

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

当经验池大小＞20GB时，删除最旧数据，直到经验池大小＜20GB。

需要预防的报错：

FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead.

FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.

UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True.
