import os
import cv2
import numpy as np

# Đường dẫn đến thư mục chứa bộ dữ liệu
dataset_path = r"D:\CNN letter Dataset"

# Khởi tạo danh sách chứa ảnh và nhãn tương ứng
images = []
labels = []

classes = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'A': 10,
    'B': 11,
    'C': 12,
    'D': 13,
    'E': 14,
    'F': 15,
    'G': 16,
    'H': 17,
    'I': 18,
    'J': 19,
    'K': 20,
    'L': 21,
    'M': 22,
    'N': 23,
    'P': 24,
    'Q': 25,
    'R': 26,
    'S': 27,
    'T': 28,
    'U': 29,
    'V': 30,
    'W': 31,
    'X': 32,
    'Y': 33,
    'Z': 34,
}

# Duyệt qua các thư mục con trong thư mục dataset
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):  # Kiểm tra xem đối tượng có phải là thư mục hay không
        label = classes[folder_name]  # Sử dụng tên thư mục làm nhãn
        print(label)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            # Đọc ảnh
            image = cv2.imread(image_path)
            # Resize ảnh về kích thước 32x32
            image = cv2.resize(image, (30,40))
            # Chuyển ảnh sang ảnh grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # # Áp dụng ngưỡng để chuyển sang ảnh nhị phân
            _, binary_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Thêm ảnh và nhãn tương ứng vào danh sách
            images.append(binary_img)
            labels.append(label)

# Chuyển danh sách ảnh và nhãn sang numpy arrays
images = np.array(images)
labels = np.array(labels)

images = np.expand_dims(images, axis=-1)
# Kiểm tra kích thước của các arrays
print("Kích thước của mảng ảnh:", images.shape)
print("Kích thước của mảng nhãn:", labels.shape)

save_path = r"D:\PyCharm-Project\PBL5-KTMT"

# Lưu mảng ảnh và nhãn xuống đĩa
np.save(os.path.join(save_path, "images_gray_thresh_30_40.npy"), images)
np.save(os.path.join(save_path, "labels_gray_thresh_30_40.npy"), labels)