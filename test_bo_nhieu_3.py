import cv2
import numpy as np
from skimage.filters import threshold_local
from ultralytics import YOLO
from tensorflow.keras import models
import matplotlib.pyplot as plt
def format(predicted_characters, bounding_rects):
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
def preprocess_image(image):
    # Convert to HSV color space
    V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]

    # Adaptive thresholding
    T = threshold_local(V, 25, offset=5, method="gaussian")
    thresh = (V > T).astype("uint8") * 255

    # Invert the image
    thresh = cv2.bitwise_not(thresh)

    # Connected component analysis to remove noise
    _, labels = cv2.connectedComponents(thresh)

    # Create a mask to filter out noise
    mask = np.zeros(thresh.shape, dtype="uint8")
    total_pixels = thresh.shape[0] * thresh.shape[1]
    lower = total_pixels // 120
    upper = total_pixels // 20

    # Filter out components based on size
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)

    # Bitwise AND operation to get the final preprocessed image
    result = cv2.bitwise_and(thresh, mask)

    return result


def find_and_sort_characters(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    cv2.imshow("binary", preprocessed_image)
    cv2.waitKey(0)

    # Find contours of the characters
    cnts, _ = cv2.findContours(preprocessed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # Clean the bounding boxes by removing noise
    mean_w = np.mean([box[2] for box in boundingBoxes])
    mean_h = np.mean([box[3] for box in boundingBoxes])
    threshold_w = mean_w * 1.5
    threshold_h = mean_h * 1.5
    cleaned_boundingBoxes = [box for box in boundingBoxes if box[2] < threshold_w and box[3] < threshold_h]

    # Classify characters into lines
    mean_y = np.mean([box[1] for box in cleaned_boundingBoxes])
    line1 = []
    line2 = []
    for box in cleaned_boundingBoxes:
        x, y, w, h = box
        if y > mean_y * 1.2:
            line2.append(box)
        else:
            line1.append(box)

    # Sort characters from left to right
    line1 = sorted(line1, key=lambda box: box[0])
    line2 = sorted(line2, key=lambda box: box[0])

    return line1, line2
def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), thickness=2):
    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
# Đường dẫn tới ảnh biển số xe đã cắt
model = YOLO(r"D:\PyCharm-Project\pythonProject12\runs\detect\train_3\weights\best.pt" )
img = cv2.imread(r"D:\PyCharm-Project\pythonProject12\test\c6.png")
results = model.predict(img)

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
           "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
# Đọc ảnh biển số đã cắt
X, Y, W, H = None, None, None, None
for result in results:
    boxes = result.boxes.numpy()
    for box in boxes:
        X = box.xyxy[0][0]
        Y = box.xyxy[0][1]
        W = box.xywh[0][2]
        H = box.xywh[0][3]
img_crop = img[int(Y)-2: int(Y)+int(H)+2, int(X)-2: int(X)+int(W)+2]
cv2.imshow("Crop_img", img_crop)
cv2.waitKey()
# Tiền xử lý ảnh biển số xe
line1, line2 = find_and_sort_characters(img_crop)

# Draw bounding boxes on the original image
image_with_boxes = img_crop.copy()
draw_bounding_boxes(image_with_boxes, line1)
draw_bounding_boxes(image_with_boxes, line2)

# Display the image with bounding boxes
cv2.imshow("Image with Bounding Boxes", image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()


def preprocess_character(character):
    # Thêm padding cho ảnh ký tự
    paddingY = (128 - character.shape[0]) // 2 if character.shape[0] < 128 else int(0.15 * character.shape[0])
    paddingX = (128 - character.shape[1]) // 2 if character.shape[1] < 128 else int(0.45 * character.shape[1])
    character = cv2.copyMakeBorder(character, paddingY, paddingY,
                                   paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)

    # Chuyển từ ảnh xám sang ảnh RGB
    character = cv2.cvtColor(character, cv2.COLOR_GRAY2RGB)

    # Resize ảnh về kích thước 128x128
    character = cv2.resize(character, (128, 128))

    # Chuẩn hóa ảnh
    character = character.astype("float") / 255.0

    return character






# binary = preprocess_image(img_crop)
# cv2.imshow("abc", binary)
# cv2.waitKey()
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
# print("Num_labels:", num_labels)
# # Khởi tạo danh sách để lưu các ký tự ứng viên và bounding rectangles
# candidates = []
# bounding_rects = []
#
# # Lặp qua các nhãn từ 1 đến num_labels - 1 (loại bỏ nhãn của background)
# for label in range(1, num_labels):
#     # Tạo mask chứa các pixel có nhãn cùng là label
#     mask = np.zeros(binary.shape, dtype=np.uint8)
#     mask[labels == label] = 255 # Các các pixel cùng nhãn giá trị 255
#
#     # Tìm contours từ mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Lọc contours theo tiêu chí aspect ratio, solidity và height ratio
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#
#         aspect_ratio = w / float(h)
#         solidity = cv2.contourArea(contour) / float(w * h)
#         height_ratio = h / float(binary.shape[0])
#
#         # Kiểm tra các điều kiện để loại bỏ nhiễu như dâu "." hay "-"
#         if 0.2 < aspect_ratio < 1.0 and solidity > 0.1 and 0.2 < height_ratio < 1.0:
#             bounding_rects.append((x, y, w, h))
#
#             # Trích xuất ký tự
#             character = binary[y-3: y + h+3, x-3:x + w+3]
#             if character.size != 0:
#                 # Đảm bảo kích thước ảnh phù hợp với mô hình
#                 character_resized = cv2.resize(character, (30, 40), interpolation=cv2.INTER_AREA)
#                 # Chuẩn hóa giá trị pixel về khoảng [0, 1]
#                 character_normalized = character_resized / 255.0
#                 # Mở rộng chiều dữ liệu để phù hợp với input_shape của mô hình (32, 32, 1)
#                 character_input = np.expand_dims(character_normalized, axis=-1)
#                 # Thêm ký tự đã chuẩn bị vào danh sách các ký tự
#                 # candidates.append(character_input)
#                 candidates.append(character_input)
#
# print("Số lượng bouding rectangle", len(bounding_rects))
# for rect in bounding_rects:
#     x, y, w, h = rect
#     cv2.rectangle(img_crop, (x, y), (x+w, y+h), (0, 0, 255), 2)
#
# # Load mô hình nhận dạng ký tự
# my_model = models.load_model(r"D:\PyCharm-Project\PBL5-KTMT\model_gray_thresh_30_40.h5")
#
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
# # Dự đoán các ký tự từ danh sách các ký tự ứng viên
# predicted_characters = []
# for character_input in candidates:
#     prediction = my_model.predict(np.array([character_input]))
#     # Lấy chỉ số của lớp có xác suất cao nhất
#     predicted_index = np.argmax(prediction)
#     # Chuyển chỉ số thành ký tự dự đoán
#     predicted_character = classes[predicted_index]  # Sử dụng danh sách classes để ánh xạ chỉ số sang ký tự
#     # Thêm ký tự dự đoán vào danh sách
#     predicted_characters.append(predicted_character)
#
# bien_so_du_doan = format(predicted_characters, bounding_rects)
#
# print("Biển số xe được dự đoán:", bien_so_du_doan)
#
# cv2.rectangle(img, (int(X), int(Y)), (int(X+W), int(Y+H)), (0, 0, 255, 2))
# cv2.putText(img, bien_so_du_doan, (int(X), int(Y)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
# cv2.imshow("Segmentation", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

