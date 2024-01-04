import cv2
import os
import numpy as np

def prepare_training_data(data_folder_path):
    faces = []
    labels = []

    person_num = range(1,len(os.listdir(data_folder_path))+1)
    i = 0

    for person_name in os.listdir(data_folder_path):
        person_path = os.path.join(data_folder_path, person_name)

        if not os.path.isdir(person_path):
            continue

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            face = cv2.resize(image, (256, 256))
            faces.append(face)
            labels.append(person_num[i])  # 修改這裡

        i += 1

    return faces, labels

def train_lbph_recognizer(data_folder, save_path="lbph_model_H.yml"):
    faces, labels = prepare_training_data(data_folder)
    labels = np.array(labels)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)

    recognizer.save(save_path)
    print("LBPH recognizer trained and saved successfully.")

def recognize_face(test_image, model_path="lbph_model_H.yml"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image = cv2.resize(test_image, (256, 256))

    label, confidence = recognizer.predict(test_image)
    # if label == 1 and confidence<60 :
    #     print(f"Predicted Label:Cage, Confidence: {confidence}")
    # elif label == 2 and confidence<60:
    #     print(f"Predicted Label:Jay_c, Confidence: {confidence}")
    # else:
    #     print(f"stranger")

    return label,confidence

if __name__ == "__main__":
    # Step 1: Prepare Training Data
    training_data_folder = './train_face_pict_H'  #訓練資料集路徑

    # Step 2: Train LBPH Recognizer
    train_lbph_recognizer(training_data_folder)

    # Step 3: Recognize Faces
    # test_image_path = "./3.jpg"  # 替換成實際的測試圖片路徑
    # test_image = cv2.imread(test_image_path)
    # recognize_face(test_image)
