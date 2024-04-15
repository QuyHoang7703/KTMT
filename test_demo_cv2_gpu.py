import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from ultralytics import YOLO

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
my_model = load_model(r"D:\PyCharm-Project\PBL5-KTMT\model_gray_thresh_30_40.h5")
X, Y, W, H = None, None, None, None
img_crop=None
cap = cv2.VideoCapture(0)
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

                img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                print("Num_labels:", num_labels)
                # Khởi tạo danh sách để lưu các ký tự ứng viên và bounding rectangles
                candidates = []
                bounding_rects = []

                # Lặp qua các nhãn từ 1 đến num_labels - 1 (loại bỏ nhãn của background)
                for label in range(1, num_labels):
                    # Tạo mask chứa các pixel có nhãn cùng là label
                    mask = np.zeros(binary.shape, dtype=np.uint8)
                    mask[labels == label] = 255 # Các các pixel cùng nhãn giá trị 255

                    # Tìm contours từ mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Lọc contours theo tiêu chí aspect ratio, solidity và height ratio
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)

                        aspect_ratio = w / float(h)
                        solidity = cv2.contourArea(contour) / float(w * h)
                        height_ratio = h / float(binary.shape[0])

                        # Kiểm tra các điều kiện để loại bỏ nhiễu như dâu "." hay "-"
                        if 0.1 < aspect_ratio < 1.0 and solidity > 0.1 and 0.2 < height_ratio < 2.0:
                            bounding_rects.append((x, y, w, h))

                            # Trích xuất ký tự
                            # character = binary[y-3: y + h+3, x-3:x + w+3]
                            character = binary[y : y + h , x :x + w ]
                            # Đảm bảo kích thước ảnh phù hợp với mô hình
                            if character.size > 0:
                                character_resized = cv2.resize(character, (30, 40), interpolation=cv2.INTER_AREA)
                                # Chuẩn hóa giá trị pixel về khoảng [0, 1]
                                character_normalized = character_resized / 255.0
                                # Mở rộng chiều dữ liệu để phù hợp với input_shape của mô hình (32, 32, 1)
                                character_input = np.expand_dims(character_normalized, axis=-1)
                                # Thêm ký tự đã chuẩn bị vào danh sách các ký tự
                                # candidates.append(character_input)
                                candidates.append(character_input)

                for rect in bounding_rects:
                    x, y, w, h = rect
                    cv2.rectangle(img_crop, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # n = len(candidates)
                # figure = plt.figure(figsize=(n, 1))
                #
                # # Vòng lặp để vẽ từng ký tự
                # for i, character in enumerate(candidates, 1):
                #     ax = figure.add_subplot(1, n, i)
                #     ax.imshow(character, cmap='gray')  # Hiển thị ảnh xám
                #     ax.axis('off')  # Tắt trục
                #
                # plt.show()

                # Dự đoán các ký tự từ danh sách các ký tự ứng viên
                predicted_characters = []
                for character_input in candidates:
                    prediction = my_model.predict(np.array([character_input]))
                    # Lấy chỉ số của lớp có xác suất cao nhất
                    predicted_index = np.argmax(prediction)
                    # Chuyển chỉ số thành ký tự dự đoán
                    predicted_character = classes[predicted_index]  # Sử dụng danh sách classes để ánh xạ chỉ số sang ký tự
                    # Thêm ký tự dự đoán vào danh sách
                    predicted_characters.append(predicted_character)
                bien_so_du_doan = format(predicted_characters, bounding_rects)

                print("Biển số xe được dự đoán:", bien_so_du_doan)

                cv2.rectangle(frame, (int(X), int(Y)), (int(X+W), int(Y+H)), (0, 0, 255, 2))
                cv2.putText(frame, bien_so_du_doan, (int(X), int(Y)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Segmentation", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # except Exception as e:
    #     print(f"Đã xảy ra lỗi: {e}")

cap.release()
cv2.destroyAllWindows()