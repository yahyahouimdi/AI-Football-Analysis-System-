from pathlib import Path

from ultralytics import YOLO

video_path = Path("input_video") / "video.mp4"
if not video_path.is_file():
    raise FileNotFoundError(f"Input video not found: {video_path.resolve()}")

model = YOLO("models/best.pt") # load a pretrained YOLOv8n model
results = model.predict(str(video_path), save=True)
print(results[0])
print('=================================')
for box in results[0].boxes:
    print(box)