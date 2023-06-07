# Gesture-Recognition
石头-剪刀-布手势实时识别（Pytorch and GUI）

## 1. 项目描述
本应用主要是一个简单的基于卷积神经网络（CNN）的手势识别，主要就是对石头-剪刀-布画面进行实时检测，基本上是可以实现实时检测，准确率也是可以的。

<img src="https://github.com/BubbleByteX/Gesture-Recognition/assets/115935683/b5f8be27-83a5-46be-bd29-0b91e67df8dd" alt="Image" width="600px" height="500px">

## 2. 项目需求
1. 考虑到除了石头-剪刀-布三种手势外，在不摆手势的时候，人脸便会出现在画面中。因此，本实验中添加了人脸的数据集，用于在不摆手势时，将检测结果检测为“人”。
2. 主要利用卷积神经网络进行检测（5层卷积），如果大家想要效果更好的话，其实是可以利用Yolov进行识别的。
3. 石头-剪刀-布的数据集图片均是 500 x 500 的大小，人脸数据集大小是 178 x 218，为了方便模型的训练，我们统一将图片大小resize成 500 x 500，如下：
```
transform = transforms.Compose([  
    transforms.Resize((500, 500)),  
    transforms.ToTensor(),         # 转换为张量  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化  
])
```
4. 为了方便实时检测，本项目设计了一个基于Opencv和PyQt5的GUI界面，支持实时画面来检测手势，实时画面直接将结果反馈在一个文本框中，并有一个退出按钮；按钮和文本里在画面右边，并将画面改成非镜像。

## 3. 数据集
1. [瑞士Dalle Molle研究所数据](https://github.com/alessandro-giusti/rock-paper-scissors)。该项目包含在各种场合收集到的剪刀石头布的几千张图片数据，D1 -> D7 包含不同人在不同日期获取的图片数据。本项目主要选用了D7数据集，D7数据集是于2018 年2月由米兰理工大学的硕士和博士生获得。
![image](https://github.com/BubbleByteX/Gesture-Recognition/assets/115935683/22bef801-7cb3-48b3-b03e-326a0501c345)

2. [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)。CelebA是一个大规模的名人人脸数据集，包含了大约20万张名人的面部图像。数据集中的人脸图像具有不同的姿势、表情和背景。
![image](https://github.com/BubbleByteX/Gesture-Recognition/assets/115935683/9b339488-40bc-42dd-9ed3-c2f59f684294)

3. 为了保持数据类别之间的平衡，我们将不同类别的数据分别只取了165-170张左右，防止在训练过程中因数据集不平衡给模型带来干扰。
4. 数据类别： 'paper'表示'布',  'people'表示'人脸',  'rock'表示'石头',  'scissor'表示'剪刀'。
```
{'paper': 0, 
 'people': 1, 
 'rock': 2, 
 'scissor': 3}
 ```
 
 数据集大家可以自己整理，也可以直接使用我整理好的[数据集](https://wwvv.lanzout.com/iRmjI0yiyjyf)。
 
 ![image](https://github.com/BubbleByteX/Gesture-Recognition/assets/115935683/e2aad1e4-a87f-48c4-946a-980288a2bf62)
 
## 4. 项目执行
右击```myGUI.py```文件，运行即可 或 在终端执行命令```python myGUI.py```。

![image](https://github.com/BubbleByteX/Gesture-Recognition/assets/115935683/00eaf0d6-7112-40df-b9c4-fb9cc6694686)
