import cv2
import numpy as np
from yolodetect import YoloDetect

# Đọc video từ tệp, thay đổi đường dẫn video tùy theo yêu cầu
video_path = "/Users/taduylam/Workspace/python/yolo-pet-detector/demo-sample/demo_room.mp4" 
video = cv2.VideoCapture(video_path)

# Kiểm tra nếu video được mở thành công
if not video.isOpened():
    print("Error: Không thể mở video.")
    exit()

# Đọc kích thước khung hình từ video
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Danh sách các điểm người dùng chọn để tạo polygon
points = []

# Khởi tạo đối tượng YoloDetect
model = YoloDetect(frame_width=frame_width, frame_height=frame_height)

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
paused = False  # Biến kiểm tra trạng thái tạm dừng

while True:
    if not paused:
        ret, frame = video.read()
        if not ret:  # Nếu không đọc được frame (kết thúc video)
            print("Video ended.")
            break


    # Vẽ polygon lên ảnh
    frame = draw_polygon(frame, points)

    if detect:
        # Gọi hàm detect từ YoloDetect để phát hiện đối tượng và kiểm tra xem có nằm trong polygon không
        frame, status = model.detect(frame=frame, points=points)
        # print(f"Detection Status: {status}")

    # Xử lý phím bấm
    key = cv2.waitKey(1)
    if key == ord('q'):  # Nhấn 'q' để thoát
        break
    elif key == ord('d'):  # Nhấn 'd' để hoàn tất vẽ polygon và bắt đầu phát hiện

        points.append(points[0])  # Đóng polygon
        detect = True
    elif key == 32:  # Nhấn phím Space để tạm dừng video
        paused = not paused  # Chuyển trạng thái tạm dừng

    # Hiển thị ảnh ra màn hình
    cv2.imshow("Pet Location", frame)

    # Gắn callback cho sự kiện chuột
    cv2.setMouseCallback('Pet Location', handle_left_click, points)

# Dừng video và đóng cửa sổ
video.release()
cv2.destroyAllWindows()
