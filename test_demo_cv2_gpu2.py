import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from ultralytics import YOLO
from easyocr import Reader
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)
def format(predicted_characters, bounding_rects):
    if not bounding_rects:
        print("Không có biển số xe được phát hiện")
        return ""
    if len(bounding_rects) == 8:  # Nếu chỉ có một dòng
        # Sắp xếp các ký tự theo tọa độ x của bounding box
        sorted_characters = sorted(zip(predicted_characters, bounding_rects), key=lambda x: x[1][0])
        license_plate = "".join([char[0] for char, _ in sorted_characters])
    else:
        # Phân chia các ký tự thành hai dòng
        first_line = []
        second_line = []
        mid_y = bounding_rects[0][1] + bounding_rects[0][3] / 2

        for character, coordinate in zip(predicted_characters, bounding_rects):
            if coordinate[1] < mid_y:
                first_line.append((character, coordinate[0]))
            else:
                second_line.append((character, coordinate[0]))

        # Sắp xếp các ký tự trong mỗi dòng theo tọa độ x của bounding box
        first_line = sorted(first_line, key=lambda ele: ele[1])
        second_line = sorted(second_line, key=lambda ele: ele[1])

        # Tạo biển số xe từ hai dòng ký tự đã phân loại
        license_plate = "".join([char[0] for char, _ in first_line]) + "".join([char[0] for char, _ in second_line])

    return license_plate
model = YOLO(r"D:\PyCharm-Project\pythonProject12\runs\detect\train_3\weights\best.pt" )
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
           "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
# my_model = load_model(r"D:\PyCharm-Project\PBL5-KTMT\model_gray_thresh_30_40.h5")
X, Y, W, H = None, None, None, None
img_crop=None
cap = cv2.VideoCapture(0)
reader = Reader(["en"], gpu=True)
while True:
    # try:
        img_crop= None
        ret, frame = cap.read()

        if not ret:
            print("Không thể đọc từ camera.")
            break
        frame = cv2.resize(frame, (800, 600))
        results = model.predict(frame, show= True, stream=True, device='0')
        if results:

            X, Y, W, H = None, None, None, None
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    X = box.xyxy[0][0]
                    Y = box.xyxy[0][1]
                    W = box.xywh[0][2]
                    H = box.xywh[0][3]
            if X != None and Y != None and W != None and H != None:
                img_crop = frame[int(Y)-2: int(Y)+int(H)+2, int(X)-2: int(X)+int(W)+2]

                # img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                # _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                text = reader.readtext(img_crop)
                print(type(text))
                print(text)
                print("Biểu số nhận dạng: ", text[0][1])
                cv2.rectangle(frame, (int(X), int(Y)), (int(X + W), int(Y + H)), (0, 0, 255), 1)
                cv2.putText(frame, text[0][1], (int(X), int(Y)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("acb", frame)
            cv2.waitKey()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # except Exception as e:
    #     print(f"Đã xảy ra lỗi: {e}")

cap.release()
cv2.destroyAllWindows()