import cv2
import numpy as np
import os
from tqdm import tqdm, trange

def read_path(file_pathname):
    #讀取圖片資料夾
    for filename in os.listdir(file_pathname):
        #讀取資料夾內部的名稱 filename  資料夾名稱
        # print('filename',filename)

        #設定 調低亮度的圖片資料夾名稱

        # print('Black_picture',Black_picture)

        #讀取資料夾內部的圖片 picname  圖片名稱
        for picname in tqdm(os.listdir(file_pathname + '/' + filename)):
        # print('picname',picname)

            try:
                # print("try")

                img = cv2.imread(file_pathname + '/' + filename + '/' + picname)
                # 讀取圖片

                image = np.power(img, 0.7)
                # 調低亮度


                path = "./L_pict/" + filename
                # print("path",path)
                # (資料夾為圖片名稱)  path = "D:\\Yichen\\CV\\project\\B_pict\\" + filename

                # 判斷資料夾是否存在
                if not os.path.exists(path):
                    # print("Not exists")

                    #建立資料夾
                    os.makedirs(path)
                else:
                    # print("exists")
                    pass

                # 將圖片寫入資料夾命名好的資料夾(Black_picture)內部
                cv2.imwrite(path + '/' + picname, image)
            except:
                print("fail to read ")
                continue

read_path("./pict")

