import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

class MainWindow():
    def __init__(self):

        #self.rename_file()
        self.start()

    def rename_file(self):
        name = []
        number = []
        df = pd.read_csv('D:/研究所/腮腺腫瘤超音波影像計畫/data/71019B報表/71019B_全院677位_2011_01_01_2021_10_25.csv',header=[0])
        print(df.head())
        #print(df['病人姓名'])
        #print(df['病歷號'])
        dict = {}
        for f in range(len(df['病人姓名'])):
            if df['病歷號'][f].is_integer() == True:
                df['病歷號'][f] = int(df['病歷號'][f])
            #print(df['病人姓名'][f], df['病歷號'][f])
            dict.update({df['病人姓名'][f] : df['病歷號'][f]})
        print(dict)
        
        filepath = r'D:/研究所/腮腺腫瘤超音波影像計畫/data/Echo/label/'
        #print(os.listdir(filepath))
        pictures = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
        print(pictures)
        for f in os.listdir(filepath):
            123

        '''
        for f in range(0, len(pictures)):
            old_file = os.path.join(filepath, pictures[f])
            print(old_file)
            new_file = os.path.join(filepath, 'label_' + str(f) + '.bmp')
            #os.rename(old_file, new_file)
            #img = cv2.imdecode(np.fromfile(new_file,dtype = np.uint8),-1)
            #print(new_file)
            #cv2.imshow('test', img)
            #cv2.waitKey(500)
            #cv2.destroyAllWindows()
        '''


    def mouse_handler(self, event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 標記點位置
            cv2.circle(data['img'], (x,y), 3, (0,0,255), 5, 16)
            #circle1 = plt.Circle((event.xdata,event.ydata), 10, color='green')

            # 改變顯示 window 的內容
            cv2.imshow("Image", data['img'])
            
            # 顯示 (x,y) 並儲存到 list中
            print("get points: (x, y) = ({}, {})".format(x, y))
            data['points'].append((x,y))

    def get_points(self, im):
        # 建立 data dict, img:存放圖片, points:存放點
        data = {}
        data['img'] = im.copy()
        data['points'] = []
        
        # 建立一個 window
        cv2.namedWindow("Image", 0)
        
        # 改變 window 成為適當圖片大小
        h, w, dim = im.shape
        print("Img height, width: ({}, {})".format(h, w))
        cv2.resizeWindow("Image", w, h)
            
        
        
        # 利用滑鼠回傳值，資料皆保存於 data dict中
        cv2.setMouseCallback("Image", self.mouse_handler, data)
        
        # 顯示圖片在 window 中
        cv2.imshow('Image',im)

        # 等待按下任意鍵，藉由 OpenCV 內建函數釋放資源
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        # 回傳點 list
        return data['points']
    
    def fold_start(self):
        image_folder = 'image'  # Change this to your image folder path
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
                dataset_dir = os.path.join(fold_dir, dataset, 'image_resize')
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)

            # Copy training images
            for filename in train_filenames:
                image_path = os.path.join(image_folder, filename)
                index = filename.split('-')[0]
                img = cv2.imread(image_path)
                img = self.denoising_image(img)
                img = self.image_enhancement(img)
                img = self.image_cropping(img)
                imgs, filenames = self.image_augmentation(img, index)
                
                # for i in range(len(imgs)):
                #     cv2.imwrite(os.path.join(fold_dir, 'train', 'image_resize', filenames[i]), imgs[i])
                cv2.imwrite(os.path.join(fold_dir, 'train', 'image_resize', filename), img)
            # Copy validation images
            for filename in valid_filenames:
                image_path = os.path.join(image_folder, filename)
                img = cv2.imread(image_path)
                img = self.denoising_image(img)
                img = self.image_enhancement(img)
                img = self.image_cropping(img)
                cv2.imwrite(os.path.join(fold_dir, 'valid', 'image_resize', filename), img)

            # Copy testing images
            for filename in X_test:
                image_path = os.path.join(image_folder, filename)
                img = cv2.imread(image_path)
                img = self.denoising_image(img)
                img = self.image_enhancement(img)
                img = self.image_cropping(img)
                cv2.imwrite(os.path.join(fold_dir, 'test', 'image_resize', filename), img)
        
    def start(self):
        n = 0
        # 迴圈讀取資料夾中的圖片
        for filename in os.listdir('image'):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp'):  
                image_path = os.path.join('image', filename)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            crop_img = self.image_cropping(img)
            
            # # denoise and enhancement
            denoise_img = self.denoising_image(crop_img)
            # # cv2.imwrite(f'{n}_denoise.bmp',denoise_img)
            enhance_img = self.image_enhancement(denoise_img)
            
            # resize_img = cv2.resize(enhance_img, (352, 352), interpolation=cv2.INTER_AREA)
            # cv2.imwrite(f'{n}_resize.jpg', resize_img)
            # cv2.imwrite(f'{n}_enhancement.bmp',enhance_img)
            # imgs, filenames = self.image_augmentation(enhance_img, n)

            cv2.imwrite(f'data/images/{n}.jpg',enhance_img)
            n = n + 1
        return
    
    def image_cropping(self, img):
        x = 630 
        y = 90 
        width = 30
        height = 30
        img[y:y+height, x:x+width] = (0, 0, 0)
        
        # remove unnecessary info
        mask = img.copy()      
        for y in range(0,mask.shape[1]):
            for x in range(0,mask.shape[0]):
                
                if mask[x,y][0] >= 250 and mask[x,y][1] >= 250 and mask[x,y][2] >= 250:
                    mask[x,y] = (255,255,255)
                elif mask[x,y][0] == 254 and mask[x,y][1] >= 254 and mask[x,y][2] == 254:
                    mask[x,y] = (255,255,255)
                elif mask[x,y][0] >= 200 and mask[x,y][1] >= 200 and mask[x,y][2] >= 210:
                    mask[x,y] = (255,255,255)
                elif mask[x,y][0] == 255 and mask[x,y][1] == 149 and mask[x,y][2] == 5:
                    mask[x,y] = (255,255,255)
                else:
                    mask[x,y] = (0,0,0)

        cv2.imwrite('mask.bmp',mask)
        mask = cv2.imread('mask.bmp',0)
        
        #鄰近消除法去除雜訊
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        dst1 = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
        cv2.imwrite('after_mask.bmp',dst1)
        
        # crop
        x_min = 86
        x_max = 885
        y_min = 89
        y_max = 645
        img = dst1[y_min:y_max, x_min:x_max]
        
        return img
    
    def image_augmentation(self, image, n):
        images = [image]
        filenames = [f"PE_{n}.png"]

        # 水平翻轉
        flipped_horizontally = cv2.flip(image, 1)
        images.append(flipped_horizontally)
        filenames.append(f"PE_{n}_flip_h.png")

        # 形變操作：輕度的向右彎曲
        rows, cols, _ = image.shape
        M_right = cv2.getRotationMatrix2D((cols / 2, rows / 2), -15, 1)
        distorted_right = cv2.warpAffine(image, M_right, (cols, rows))
        images.append(distorted_right)
        filenames.append(f"PE_{n}_distorted_right.png")

        # 形變操作：輕度的向左彎曲
        M_left = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
        distorted_left = cv2.warpAffine(image, M_left, (cols, rows))
        images.append(distorted_left)
        filenames.append(f"PE_{n}_distorted_left.png")

        # # 加雜訊
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

        # 椒鹽雜訊
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
        noisy_image[coords_salt[0], coords_salt[1], :] = [255, 255, 255]

        # 添加椒噪聲（黑色）
        coords_pepper = [np.random.randint(0, i-1, num_pepper_pixels) for i in image.shape]
        noisy_image[coords_pepper[0], coords_pepper[1], :] = [0, 0, 0]

        return noisy_image

    def image_enhancement(self,img):
         # Convert image to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Split LAB channels
        l, a, b = cv2.split(lab)

        # Apply adaptive histogram equalization to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_equalized = clahe.apply(l)

        # Merge LAB channels back
        lab_equalized = cv2.merge([l_equalized, a, b])

        # Convert back to BGR color space
        enhanced_image = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
        
        return enhanced_image

    def denoising_image(self, img):
        # return cv2.medianBlur(img, 3)
        return cv2.bilateralFilter(img, 3, 20, 75)
        # return cv2.blur(img, (3, 3))
        
    def resize_image(self,img, height=240, width=240):
        
        top, bottom, left, right = (0, 0, 0, 0)
        h, w, _ = img.shape
    
        # 找長邊
        longest_edge = max(h, w)
    
        # 算短邊
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass

        # RGB color
        BLACK = [0, 0, 0]

        # 填黑邊
        constant = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

        # resize大小
        return cv2.resize(constant, (height, width),  interpolation=cv2.INTER_NEAREST)
            
        
    def cut(self,img):
        
        #crop first time, leave black line in both side
        # img = img.crop((203, 68, 595, 475))
        img = img[68:475, 203:595]

        #crop second time, leave only ultrasound image
        left_x,left_y,right_x,right_y = 0,0,0,0
        # pixdata = img.load()
        flag = 0
        for x in range(img.shape[0]-1):
            if flag == 0:
                for y in range(0,img.shape[1]-1):
                    if img[x,y][0] >0 or img[x,y][1] >0 or img[x,y][2] >0 and flag == 0:
                        left_x = x
                        left_y = y
                        flag = 1
                        break
        flag = 0
        for y in range(img.shape[1]-1):
            if flag == 0:
                for x in range(img.shape[0]-1):
                    if img[x,y][0] >0 or img[x,y][1] >0 or img[x,y][2] >0 and flag == 0:
                        left_x = min(x,left_x)
                        left_y = min(y,left_y)
                        flag = 1
                        break
        flag = 0
        for x in range(img.shape[0]-1,0,-1):
            if flag == 0:
                for y in range(img.shape[1]-1,0,-1):
                    if img[x,y][0] >0 or img[x,y][1] >0 or img[x,y][2] >0 and flag == 0:
                        right_x = x
                        right_y = y
                        flag = 1
                        break
        flag = 0
        for y in range(img.shape[1]-1,0,-1):
            if flag == 0:
                for x in range(img.shape[0]-1,0,-1):
                    if img[x,y][0] >0 or img[x,y][1] >0 or img[x,y][2] >0 and flag == 0:
                        right_x = max(x,right_x)
                        right_y = max(y,right_y)
                        flag = 1
                        break
        print(left_x,left_y,right_x,right_y)
        img = img[left_x:right_x , left_y:right_y]
        return img

    def get_label_mask(self):
        filepath = r'C:/Users/Jeff/Desktop/tumour_project/code/unet_master/data/membrane/preprocessed/LABEL_INPAINT_TELEA'
        #print(os.listdir(filepath))
        pictures = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
        #print(pictures)
        for f in range(0, len(pictures)):
            imgfile = 'C:/Users/Jeff/Desktop/tumour_project/code/unet_master/data/membrane/preprocessed/LABEL_INPAINT_TELEA/LABEL_INPAINT_TELEA_' + str(f) + '.bmp'
            img = cv2.imread(imgfile)
            h, w, _ = img.shape

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Find Contour
            contours, hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # 需要搞一个list给cv2.drawContours()才行！！！！！
            c_max = []
            for i in range(len(contours)):
                cnt = contours[i]
                area = cv2.contourArea(cnt)

                # 处理掉小的轮廓区域，这个区域的大小自己定义。
                if(area < (h/10*w/10)):
                    c_min = []
                    c_min.append(cnt)
                    # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。
                    cv2.drawContours(img, c_min, -1, (0,0,0), thickness=-1)
                    continue
                c_max.append(cnt)
            
            #plt.figure(figsize=(8,8))
            #plt.fill(c_max)
            #plt.show()

            cv2.drawContours(img, c_max, -1, (255, 255, 255), thickness=-1)

            cv2.imwrite('C:/Users/Jeff/Desktop/tumour_project/code/unet_master/data/membrane/preprocessed/LABEL_MASK/MASK_' + str(f) + '.bmp', img)
            #cv2.imshow('mask',img)
            #cv2.waitKey(0)

    def fill_mask(self, filepath, f):

        #邊緣偵測，將白色pixel連起來填補邊框缺口
        # img = cv2.imread(filepath + 'out_' + f)
        # gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # edges = cv2.Canny(gray,80,210)
        # result = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel=(3,3),iterations=3)
        # # cv2.imshow('After Canny',gray)
        # # cv2.imshow('After Morphology Close',result)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        # cv2.imwrite(filepath + 'out_' + f,result)

        #邊緣偵測，將白色pixel連起來填補邊框缺口
        img = cv2.imread(filepath + 'out_' + f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        contours,heirarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img,contours,-1,(255,255,255),2)
        cv2.imwrite(filepath + 'out_' + f,img)
        # img = np.zeros((img.shape[0],img.shape[1])) # create a single channel 200x200 pixel black image 
        # cv2.polylines(img, contours, 3, (255,255,255),2)
        # cv2.imwrite(filepath + 'out_' + f,img)
        

        #填補圈選區域內為白色
        img = Image.open(filepath + 'out_' + f)
        img = img.convert('L')
        #img.show()
        pixdata = img.load()
        for y in range(0,img.size[1]-1):
            for x in range(0,img.size[0]-1):
                if pixdata[x,y] <= 100:
                    pixdata[x,y] = 0
                elif pixdata[x,y] >= 200:
                    pixdata[x,y] = 255
        for y in range(0,img.size[1]-1):
            tmp = 2
            for x in range(0,img.size[0]-1):
                
                #當前為黑，下一個為白
                if (pixdata[x,y] == 0) and (pixdata[x+1,y] == 255)  and tmp %2 != 0:
                    tmp += 1
                    pixdata[x,y] = 255
                    continue
                
                #當前為白，下一個為黑
                if (pixdata[x,y] == 255) and (pixdata[x+1,y] == 0):
                    
                    right = 0
                    #找是否為閉區間
                    for z in range(x+1,img.size[0]-1):
                        if pixdata[z,y] == 255:
                            right = 1
                            break
                    if right == 1:
                        tmp += 1
                    #pixdata[x,y] = (255,255,255,0)
                    continue
                
                #填補
                if tmp % 2 != 0:
                    pixdata[x,y] = 255
        #img.show()
        img.save(filepath + 'last_' + f)

if __name__ == "__main__":
    window = MainWindow()