import cv2
import numpy as np
import os

def image_augmentation(image, n):
    images = [image]
    filenames = [f"PE_{n:03d}.png"]

    # 水平翻轉
    flipped_horizontally = cv2.flip(image, 1)
    images.append(flipped_horizontally)
    filenames.append(f"PE_{n:03d}_flip_h.png")

    # 垂直翻轉
    flipped_vertically = cv2.flip(image, 0)
    images.append(flipped_vertically)
    filenames.append(f"PE_{n:03d}_flip_v.png")

    # 對角翻轉
    flipped_diagonally = cv2.flip(image, -1)
    images.append(flipped_diagonally)
    filenames.append(f"PE_{n:03d}_flip_diag.png")


   # 不同角度的旋轉
    angles = [30, 60, 90]
    for angle in angles:
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        images.append(rotated)
        filenames.append(f"PE_{n:03d}_rotated_{angle}.png")

    # 不同尺度的縮放
    scales = [0.8, 1.2]
    for scale in scales:
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        scaled = cv2.resize(image, (new_width, new_height))
        images.append(scaled)
        filenames.append(f"PE_{n:03d}_scaled_{scale}.png")

    # 隨機平移
    shift_x = np.random.randint(-50, 50)
    shift_y = np.random.randint(-50, 50)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated = cv2.warpAffine(image, M, (cols, rows))
    images.append(translated)
    filenames.append(f"PE_{n:03d}_translated_{shift_x}_{shift_y}.png")


    return images, filenames

folder_path = "mask"
n = 0
# 遍历文件夹中的所有图像文件
for filename in os.listdir(folder_path):
    # 检查文件类型是否为图像
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # 构建图像文件的完整路径
        file_path = os.path.join(folder_path, filename)
        # 使用OpenCV读取图像
        img = cv2.imread(file_path)

        r, g, b = 255, 255, 1

        low = np.array([b-50, g-50, r-50])
        high = np.array([b+50, g+50, r+50])
        # 建立遮罩
        mask = cv2.inRange(img, low, high)

        

        # 顯示遮罩結果
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        grey = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(grey,127,255,cv2.THRESH_BINARY)

        x_min = 86
        x_max = 885
        y_min = 89
        y_max = 645

        cropped_image = thresh1[y_min:y_max, x_min:x_max]
        imgs, filenames = image_augmentation(cropped_image, n)
        for i in range(len(imgs)):
            cv2.imwrite(f'mask_resize/{filenames[i]}',imgs[i])
        n = n + 1
