import cv2
import albumentations as albu
import matplotlib.pyplot as plt

# 读取原始图片
image = cv2.imread("data/images/0.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 定义仿射变换
affine_transform = albu.Affine(
)

# 应用仿射变换
transformed_image = affine_transform(image=image)['image']

# 显示原始图片和变换后的图片
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(transformed_image)
plt.title("Transformed Image")
plt.axis("off")

plt.show()
