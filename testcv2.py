import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ref, frame = cap.read()
    if not ref:
        break

    # width = int(cap.get(3))
    # height = int(cap.get(4))
    #
    # img = np.zeros(frame.shape, np.uint8)
    # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    #
    # # Rotate the small frame once
    # rotated_small_frame = cv2.rotate(small_frame, cv2.ROTATE_180)
    #
    # # Assign the rotated frame to the corresponding positions in img
    # img[:height//2, :width//2] = small_frame
    # img[:height//2, width//2:] = rotated_small_frame
    # img[height//2:, :width//2] = rotated_small_frame
    # img[height//2: ,width//2:] = rotated_small_frame

    cv2.imshow("My_Camera", frame)
    if(cv2.waitKey(1)==ord("q")):
        break

cap.release()
cv2.destroyAllWindows()



