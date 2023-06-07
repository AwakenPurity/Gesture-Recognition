import cv2

img_path = 'C:/Users/10935/Desktop/Gesture Recognition/datasets/paper/paper1.jpg'

# 读取图片
image = cv2.imread(img_path)

# 获取图片大小
height, width, _ = image.shape

# 打印图片大小
print(f"图片大小：{width} x {height}")

# 图片大小：500 x 500