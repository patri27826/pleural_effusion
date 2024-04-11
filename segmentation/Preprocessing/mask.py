import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold

class MainWindow():
    def __init__(self):
        self.single_start()
        
    def image_augmentation(self, image, n):
        images = [image]
        filenames = [f"PE_{n}.png"]

        # 水平翻轉
        flipped_horizontally = cv2.flip(image, 1)
        images.append(flipped_horizontally)
        filenames.append(f"PE_{n}_flip_h.png")

        # 形變操作：輕度的向右彎曲
        rows, cols = image.shape
        M_right = cv2.getRotationMatrix2D((cols / 2, rows / 2), -15, 1)
        distorted_right = cv2.warpAffine(image, M_right, (cols, rows))
        images.append(distorted_right)
        filenames.append(f"PE_{n}_distorted_right.png")

        # 形變操作：輕度的向左彎曲
        M_left = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
        distorted_left = cv2.warpAffine(image, M_left, (cols, rows))
        images.append(distorted_left)
        filenames.append(f"PE_{n}_distorted_left.png")

        # # 加噪聲
        # noisy_image = image + 0.1 * image.std() * np.random.normal(size=image.shape)
        # images.append(noisy_image)
        # filenames.append(f"PE_{n}_noisy.png")

        # 裁剪和填充
        cropped = image[10:rows-100, 10:cols-100]
        images.append(cropped)
        filenames.append(f"PE_{n}_crop.png")
        
         # 對比度和亮度調整
        contrast = np.random.uniform(0.5, 1.5)
        brightness = np.random.randint(-30, 30)
        adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        images.append(adjusted_image)
        filenames.append(f"PE_{n}_adjusted.png")

        # # 形態學操作：膨脹
        # kernel = np.ones((5, 5), np.uint8)
        # dilated_image = cv2.dilate(image, kernel, iterations=1)
        # images.append(dilated_image)
        # filenames.append(f"PE_{n}_dilated.png")
        
        # # 形態學操作：侵蝕
        # eroded_image = cv2.erode(image, kernel, iterations=1)
        # images.append(eroded_image)
        # filenames.append(f"PE_{n}_eroded.png")

        # 椒鹽噪聲
        salt_pepper_image = self.salt_and_pepper_medical(image)
        images.append(salt_pepper_image)
        filenames.append(f"PE_{n}_salt_pepper.png")
        
        return images, filenames
        
    def salt_and_pepper_medical(self, image, amount=0.01, salt_prob=0.5):
        noisy_image = np.copy(image)
        num_salt = int(amount * image.size)
        num_salt_pixels = int(num_salt * salt_prob)
        num_pepper_pixels = num_salt - num_salt_pixels

        # 添加鹽噪聲（白色）
        coords_salt = [np.random.randint(0, i-1, num_salt_pixels) for i in image.shape]
        noisy_image[coords_salt[0], coords_salt[1]] = 255

        # 添加椒噪聲（黑色）
        coords_pepper = [np.random.randint(0, i-1, num_pepper_pixels) for i in image.shape]
        noisy_image[coords_pepper[0], coords_pepper[1]] = 0
        
        return noisy_image
    
    def image_cropping(self, img):
        r, g, b = 255, 255, 1
        low = np.array([b-50, g-50, r-50])
        high = np.array([b+50, g+50, r+50])
        # 建立遮罩
        mask = cv2.inRange(img, low, high)
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        grey = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(grey,127,255,cv2.THRESH_BINARY)

        x_min = 86
        x_max = 885
        y_min = 89
        y_max = 645

        cropped_image = thresh1[y_min:y_max, x_min:x_max]
        return cropped_image
        
    def start(self):
        image_folder = 'mask'  # Change this to your image folder path
        image_filenames = os.listdir(image_folder)
        image_labels = [filename.split('-')[2] for filename in image_filenames]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(image_filenames, image_labels, test_size=0.1, random_state=42)

        for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train), start=1):
            train_filenames = [image_filenames[i] for i in train_idx]
            valid_filenames = [image_filenames[i] for i in valid_idx]
            
            # Create a folder for the fold
            fold_dir = os.path.join('data', f'fold_{fold}')
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)

            # Create subfolders for images in train, valid, and test sets
            for dataset in ['train', 'valid', 'test']:
                dataset_dir = os.path.join(fold_dir, dataset, 'mask_resize')
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)

            # Copy training images
            for filename in train_filenames:
                image_path = os.path.join(image_folder, filename)
                index = filename.split('-')[0]
                img = cv2.imread(image_path)
                img = self.image_cropping(img)
                imgs, filenames = self.image_augmentation(img, index)

                # for i in range(len(imgs)):
                #     cv2.imwrite(os.path.join(fold_dir, 'train', 'mask_resize', filenames[i]), imgs[i])
                cv2.imwrite(os.path.join(fold_dir, 'train', 'mask_resize', filename), img)

            # Copy validation images
            for filename in valid_filenames:
                image_path = os.path.join(image_folder, filename)
                img = cv2.imread(image_path)
                img = self.image_cropping(img)
                cv2.imwrite(os.path.join(fold_dir, 'valid', 'mask_resize', filename), img)

            # Copy testing images
            for filename in X_test:
                image_path = os.path.join(image_folder, filename)
                img = cv2.imread(image_path)
                img = self.image_cropping(img)
                cv2.imwrite(os.path.join(fold_dir, 'test', 'mask_resize', filename), img)
                
    def single_start(self):
        image_folder = 'mask'  # Change this to your image folder path
        image_filenames = os.listdir(image_folder)
        
        n = 0
        for image_filename in image_filenames:
            image_name = image_filename.split('-')[0]
            filename = image_name + '.jpg'
            image_path = os.path.join(image_folder, image_filename)
            img = cv2.imread(image_path)
            img = self.image_cropping(img)
            cv2.imwrite(os.path.join('data', 'masks', f'{n}.jpg'), img)
            n = n + 1
    
if __name__ == '__main__':
    MainWindow()

                
