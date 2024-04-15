import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

def format(predicted_characters, bounding_rects):
    if len(bounding_rects) == 8:  # Nếu chỉ có một dòng
        sorted_characters = sorted(zip(predicted_characters, bounding_rects), key=lambda x: x[1][0])
        license_plate = "".join([char[0] for char, _ in sorted_characters])
    else:
        first_line = []
        second_line = []
        mid_y = bounding_rects[0][1] + bounding_rects[0][3] / 2
        for character, coordinate in zip(predicted_characters, bounding_rects):
            if coordinate[1] < mid_y:
                first_line.append((character, coordinate[0]))
            else:
                second_line.append((character, coordinate[0]))
        first_line = sorted(first_line, key=lambda ele: ele[1])
        second_line = sorted(second_line, key=lambda ele: ele[1])
        license_plate = "".join([char[0] for char, _ in first_line]) + "-" + "".join([char[0] for char, _ in second_line])
    return license_plate

model = YOLO(r"D:\PyCharm-Project\pythonProject12\runs\detect\train_3\weights\best.pt")
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
my_model = load_model(r"D:\PyCharm-Project\PBL5-KTMT\model_gray_thresh_30_40.h5")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc từ camera.")
        break

    frame = cv2.resize(frame, (800, 600))
    results = model.predict(frame, stream=True, device='0')
    if results and any([result.boxes.cpu().numpy().size for result in results]):
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                X, Y, W, H = box[0], box[1], box[2], box[3]
                img_crop = frame[int(Y):int(Y+H), int(X):int(X+W)]
                img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

                candidates = []
                bounding_rects = []
                for label in range(1, num_labels):
                    x, y, w, h = cv2.boundingRect(labels == label)
                    if 0.1 < w/h < 1.0:  # simple aspect ratio check
                        character = binary[y:y+h, x:x+w]
                        character_resized = cv2.resize(character, (30, 40))
                        character_normalized = character_resized / 255.0
                        character_input = character_normalized.reshape(1, 30, 40, 1)
                        candidates.append(character_input)
                        bounding_rects.append((x, y, w, h))

                if candidates:
                    predicted_characters = [classes[np.argmax(my_model.predict(candidate))] for candidate in candidates]
                    predicted_plate = format(predicted_characters, bounding_rects)
                    print("Detected license plate:", predicted_plate)
                    cv2.putText(frame, predicted_plate, (int(X), int(Y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Camera View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
