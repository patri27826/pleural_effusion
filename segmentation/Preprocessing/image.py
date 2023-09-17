import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd

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
    
    def start(self):
        
        # 用滑鼠找指定pixel的location及RGB數值
        # read_file = '0001-Echo-1.jpg'
        # img2 = cv2.imread(read_file)
        # points  = self.get_points(img2)
        # print("\npoints list:")
        # print(points)
        
        # img = Image.open(read_file)
        # img = img.convert('RGB')
        # for i in points:
        #     print(img.getpixel((i[0],i[1])))
        
        # return



        
        # 讀取檔案

        # filepath = r'C:/Users/CCC/Desktop/tumour_project/data/Echo_0629/20220922_redo/mixed/img_origin'
        #print(os.listdir(filepath))
        # for f in os.listdir(filepath):
        #     print(f)
        
        # pictures = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
        #print(pictures)

        
        #for f in range(0, len(pictures)):
        # for f in os.listdir(filepath):
        #     img = cv2.imread(os.path.join(filepath,f))
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
        n = 0
        # 迴圈讀取資料夾中的圖片
        for filename in os.listdir('image'):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp'):  
                image_path = os.path.join('image', filename)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # remove the right top "T" 
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

            cv2.imwrite(f'{n}_mask.bmp',mask)
            mask = cv2.imread(f'{n}_mask.bmp',0)
            
            #鄰近消除法去除雜訊
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            dst1 = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
            cv2.imwrite(f'{n}_after_mask.bmp',dst1)
            
            # crop
            x_min = 86
            x_max = 885
            y_min = 89
            y_max = 645
            crop_img = dst1[y_min:y_max, x_min:x_max]
            
            # denoise and enhancement
            # denoise_img = self.denoising_image(crop_img)
            # cv2.imwrite(f'{n}_denoise.bmp',denoise_img)
            # enhance_img = self.image_enhancement(denoise_img)
            # cv2.imwrite(f'{n}_enhancement.bmp',enhance_img)
            imgs, filenames = self.image_augmentation(crop_img, n)
            for i in range(len(imgs)):
                cv2.imwrite(f'image_resize/{filenames[i]}',imgs[i])
            n = n + 1
        return
    
    def image_augmentation(self,image, n):
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
        adaptive_histogram = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
        
        # contrast_enhance = cv2.convertScaleAbs(adaptive_histogram, alpha=1.5, beta=0)
        
        # # Create sharpening kernel
        # kernel = np.array([[-1, -1, -1],
        #                 [-1,  9, -1],
        #                 [-1, -1, -1]])
        # sharpened = cv2.filter2D(contrast_enhance, -1, kernel)

        return adaptive_histogram

    def denoising_image(self, img):
        return cv2.medianBlur(img, 3)
        
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