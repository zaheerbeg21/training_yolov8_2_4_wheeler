import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'videos')

import os

VIDEOS_DIR = '/content/train_yolov8/input_video/'  # Assuming this is the correct directory for input videos
video_path_in = '20231229_112130.mp4'  # Specify only the filename here

video_path_in = os.path.join(VIDEOS_DIR, video_path_in)  # Construct the full input path
video_path_out = os.path.splitext(video_path_in)[0] + '_out.mp4'  # Extract base name and add '_out.mp4'

print(video_path_out)  # This will now print '/content/train_yolov8/input_video/20231229_112130_out.mp4'


cap = cv2.VideoCapture(video_path_in)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = '/content/train_yolov8/local_env/epc_100_img_165/weights/best.pt'
# model_path = '/content/train_yolov8/local_env/epoc_100_img_805/best.pt'
# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
