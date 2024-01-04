import cv2
import os
import numpy as np
from tqdm import tqdm

'''
img_size = 64
HOG_len = 1764
'''
img_size = 256
HOG_len = 34596
''''''

def load_data(dirpath):

    data = np.zeros((len(os.listdir(dirpath)), img_size, img_size, 3))
    i = 0
    for filename in tqdm(os.listdir(dirpath)):
        img = cv2.imread(dirpath + '/' + filename)
        img = cv2.resize(img, (img_size, img_size))
        data[i] = img
        i += 1
    print('(', dirpath, ')', ' data shape : ', data.shape)

    return data


def Raw_image(img):

    RI_img = np.reshape(img, (-1, 3))

    return RI_img


def Color_histogram(img):

    img = np.uint8(img)
    CH_B = cv2.calcHist([img], [0], None, [256], [0, 256])
    CH_G = cv2.calcHist([img], [1], None, [256], [0, 256])
    CH_R = cv2.calcHist([img], [2], None, [256], [0, 256])
    CH_img = np.array([CH_B, CH_G, CH_R])
    CH_img = np.reshape(CH_img, (3, -1)).T

    return CH_img


def Gabor_Filters(img):

    GF_img = np.zeros((5 * 8, img_size, img_size))
    img = np.mean(img, axis=2)
    ksize = 20
    sigma_range = np.arange(5,0,-1)
    theta_range = np.arange(0, np.pi, np.pi / 8)
    lambda_range = np.arange(2,12,2)
    gamma = 0
    for i in range(5):
        kernel = cv2.getGaborKernel((ksize, ksize), sigma_range[i], theta_range[i], lambda_range[i], gamma)
        GF_img[i] = cv2.filter2D(img, -1, kernel)
    #print(GF_img.shape)

    return GF_img


def Histogram_of_Oriented_Gradient(img):

    img = np.mean(img,axis=2)
    img = np.uint8(img)
    winSize = (img_size, img_size)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    HOG_img = hog.compute(img)
    #print(HOG_img.shape)

    return HOG_img


def Nearest_Neighbor_Classification(class_name, class_center, test_img):

    pre_class = class_name[0]
    pre_class_SAD = np.sum(abs(class_center[0] - test_img))
    #print(pre_class_SAD)
    for i in range(1, len(class_name)):
        if pre_class_SAD > np.sum(abs(class_center[i] - test_img)):
            pre_class = class_name[i]
            pre_class_SAD = np.sum(abs(class_center[i] - test_img))

    return pre_class,pre_class_SAD


def train_hw2_recognizer(train_dirpath):


    class_name = np.array([])

    RI_class_center = np.zeros((len(os.listdir(train_dirpath)), img_size * img_size, 3))
    CH_class_center = np.zeros((len(os.listdir(train_dirpath)), 256, 3))
    GF_class_center = np.zeros((len(os.listdir(train_dirpath)), 5 * 8, img_size, img_size))
    HOG_class_center = np.zeros((len(os.listdir(train_dirpath)), HOG_len))

    i = 0
    for classname in os.listdir(train_dirpath):
        class_name = np.append(class_name, classname)
        print('loading train data(' + classname + ')...')
        class_img = load_data(train_dirpath + '/' + classname)

        print('extracting train feature(' + classname + ')...')
        RI_sum = 0
        CH_sum = 0
        GF_sum = 0
        HOG_sum = 0
        for img in tqdm(class_img):
            RI_sum += Raw_image(img)
            CH_sum += Color_histogram(img)
            GF_sum += Gabor_Filters(img)
            HOG_sum += Histogram_of_Oriented_Gradient(img)
        RI_sum /= len(os.listdir(train_dirpath + '/' + classname))
        RI_class_center[i] = RI_sum
        CH_sum /= len(os.listdir(train_dirpath + '/' + classname))
        CH_class_center[i] = CH_sum
        GF_sum /= len(os.listdir(train_dirpath + '/' + classname))
        GF_class_center[i] = GF_sum
        HOG_sum /= len(os.listdir(train_dirpath + '/' + classname))
        HOG_class_center[i] = HOG_sum

        i += 1

    # print(class_name.shape)
    # print(RI_class_center.shape)
    # print(CH_class_center.shape)
    # print(GF_class_center.shape)
    # print(HOG_class_center.shape)

    RI_class_center.reshape(1,-1).tofile('./RI.yml', sep=',')
    CH_class_center.reshape(1,-1).tofile('./CH.yml', sep=',')
    GF_class_center.reshape(1,-1).tofile('./GF.yml', sep=',')
    HOG_class_center.reshape(1,-1).tofile('./HOG.yml', sep=',')
    print("hw2_recognizer trained and saved successfully.")


def hw2_recognizer(img, class_list=["cage","trump"], model="HOG"):

    img = cv2.resize(img, (img_size, img_size))
    if model == 'RI':
        img = Raw_image(img)
        class_center = np.fromfile('./' + model + '.yml', sep=',', dtype=float).reshape(len(class_list), img_size * img_size, 3)
    elif model == 'CH':
        img = Color_histogram(img)
        class_center = np.fromfile('./' + model + '.yml', sep=',', dtype=float).reshape(len(class_list), 256, 3)
    elif model == 'GF':
        img = Gabor_Filters(img)
        class_center = np.fromfile('./' + model + '.yml', sep=',', dtype=float).reshape(len(class_list), 5 * 8, img_size, img_size)
    else:
        img = Histogram_of_Oriented_Gradient(img)
        class_center = np.fromfile('./' + model + '.yml', sep=',', dtype=float).reshape(len(class_list), HOG_len)

    #print(class_center.shape)
    person_num = range(1, len(class_list) + 1)

    return Nearest_Neighbor_Classification(person_num, class_center, img)


if __name__ == "__main__":

    # train_dirpath = './test_train_pict'
    # train_hw2_recognizer(train_dirpath)
    img1 = cv2.imread('./test_train_pict/cage/105037753.jpg')
    img1_center = np.fromfile('./' + 'HOG' + '.yml', sep=',', dtype=float).reshape(2, HOG_len)
    img2 = cv2.imread('./test_train_pict/trump/462887438.jpg')
    img1 = cv2.imread('./test_train_pict/cage/2455911.jpg')
    img2 = cv2.imread('./test_train_pict/trump/460598264.jpg')
    img1 = cv2.imread('./test_face_pict/extra/cage1.jpg')
    img2 = cv2.imread('./test_face_pict/extra/trump1.jpg')

    img1 = cv2.resize(img1, (img_size, img_size))
    img2 = cv2.resize(img2, (img_size, img_size))

    print(img1_center.shape)

    print(np.sum(abs(img1_center[0] - Histogram_of_Oriented_Gradient(img1))))
    print(np.sum(abs(img1_center[1] - Histogram_of_Oriented_Gradient(img1))))

    print(np.sum(abs(img1_center[0] - Histogram_of_Oriented_Gradient(img2))))
    print(np.sum(abs(img1_center[1] - Histogram_of_Oriented_Gradient(img2))))

    print(np.sum(abs(Histogram_of_Oriented_Gradient(img1).reshape(1,-1).reshape(-1, HOG_len) - Histogram_of_Oriented_Gradient(img1))))
    print(np.sum(abs(Histogram_of_Oriented_Gradient(img1) - Histogram_of_Oriented_Gradient(img1))))

    print(Nearest_Neighbor_Classification([1,2], np.fromfile('./' + 'HOG' + '.yml', sep=',', dtype=float).reshape(2, HOG_len), Histogram_of_Oriented_Gradient(img1)))

    print(hw2_recognizer(img1))
    print(hw2_recognizer(img2))

    print(img1_center[0] - Histogram_of_Oriented_Gradient(img1).reshape(1,-1).reshape(-1, HOG_len))



