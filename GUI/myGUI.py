import sys
import torch
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QHBoxLayout
from torchvision.transforms import transforms

# 加载训练好的模型
from models.cnn import ConvNet
model = ConvNet()
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

class GestureRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gesture Recognition")
        self.setGeometry(100, 100, 800, 600)

        self.videoLabel = QLabel(self)
        self.resultTextEdit = QTextEdit(self)
        self.exitButton = QPushButton("退出", self)

        layout = QHBoxLayout()
        layout.addWidget(self.videoLabel)

        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.exitButton)
        rightLayout.addWidget(self.resultTextEdit)

        layout.addLayout(rightLayout)

        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showFrame)
        self.timer.start(30)

        self.exitButton.clicked.connect(self.exit)

    def showFrame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)  # 非镜像显示
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape

            # 调整图像大小为模型需要的输入大小
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((500, 500)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            image = transform(rgb_image)
            image = image.unsqueeze(0)

            # 使用模型进行手势识别
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)

            gesture_label = ["布", "人", "石头", "剪刀"]
            result = gesture_label[predicted.item()]

            # 在实时画面上绘制识别结果
            cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # 在文本框中显示识别结果
            self.resultTextEdit.setText(result)

            # 显示处理后的图像
            q_image = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
            self.videoLabel.setPixmap(QPixmap.fromImage(q_image))

    def exit(self):
        self.capture.release()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GestureRecognitionApp()
    window.show()
    sys.exit(app.exec_())
