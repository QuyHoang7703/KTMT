import cv2
import numpy as np
from easyocr import Reader
import torch
from ultralytics import YOLO
if torch.cuda.is_available():
    print("gpu works")
else:
    print("gpu do not work")
model = YOLO(r'D:\PyCharm-Project\pythonProject12\runs\detect\train_3\weights\best.pt')
cap = cv2.VideoCapture(0)
reader = Reader(['en'], gpu=True)
while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể đọc từ camera.")
        break

    frame = cv2.resize(frame, (800, 600))
    results = model(frame,  stream=True, device='0')
    # results = model(frame, stream=True)
    if results:
        x, y, w, h = None, None , None, None
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x =(box.xyxy[0, 0])
                y=(box.xyxy[0, 1])
                w=(box.xywh[0, 2])
                h=(box.xywh[0, 3])

        # for i in range(len(x)):
            if x !=None and y != None and w != None and h != None:
                crop_img = frame[int(y):int(y + h), int(x):int(x + w)]
                crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                _, crop_thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


                text = reader.readtext(crop_thresh)

                if text:
                    print("Biển số nhận dạng: ", text[0][1])
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
                    cv2.putText(frame, text[0][1], (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            else:
                print("Không thể đọc được biển số.")

        cv2.imshow("Content", frame)
    # cv2.imshow("Content", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
