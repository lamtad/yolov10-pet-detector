import cv2
import numpy as np
from imutils.video import VideoStream
from yolodetect import YoloDetect

# Mở video stream (có thể thay đổi với video file nếu cần)
video = VideoStream(src=0).start()

# Danh sách các điểm người dùng chọn để tạo polygon
points = []

# Khởi tạo đối tượng YoloDetect
model = YoloDetect()

# Hàm xử lý sự kiện chuột trái để người dùng chọn điểm
def handle_left_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])

# Hàm vẽ polygon trên ảnh
def draw_polygon(frame, points):
    for point in points:
        frame = cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)
        
    frame = cv2.polylines(frame, [np.int32(points)], False, (255,0, 0), thickness=2)
    return frame

detect = False

while True:
    ret, frame = video.read()
    if not ret:  # Nếu không đọc được frame (kết thúc video)
            print("Video ended.")
            break
    frame = cv2.flip(frame, 1)
    # Vẽ polygon lên ảnh
    frame = draw_polygon(frame, points)
    if detect:
        frame, status = model.detect(frame=frame, points=points)
        print(f"Detection Status: {status}")

    # Xử lý phím bấm
    key = cv2.waitKey(1)
    if key == ord('q'):  # Nhấn 'q' để thoát
        break
    elif key == ord('d'):  # Nhấn 'd' để hoàn tất vẽ polygon và bắt đầu phát hiện
         # Kiểm tra nếu polygon có đủ điểm
        points.append(points[0])  # Đóng polygon
        detect = True


    # Gắn callback cho sự kiện chuột
    cv2.setMouseCallback('Pet Location', handle_left_click, points)

    # Hiển thị ảnh ra màn hình
    cv2.imshow("Pet Location", frame)

    
# Dừng video và đóng cửa sổ
video.stop()
cv2.destroyAllWindows()
