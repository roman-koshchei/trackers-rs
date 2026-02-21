import json

import cv2

# import torch
from rfdetr import RFDETRMedium

VIDEO_PATH = "data/walk.mp4"
OUTPUT_PATH = "data/detections.json"
THRESHOLD = 0.5

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
model = RFDETRMedium(device="cuda")
model.optimize_for_inference()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

detections_list = []
frame_count = 0
MAX_FRAMES = 110000
LOG_INTERVAL = 20

while True:
    success, frame_bgr = cap.read()
    if not success or frame_count >= MAX_FRAMES:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = model.predict(frame_rgb, threshold=THRESHOLD)

    frame_dets = []
    for i in range(len(detections)):
        frame_dets.append(
            {
                "box": detections.xyxy[i].tolist(),
                "class_id": int(detections.class_id[i]),
                "score": float(detections.confidence[i]),
            }
        )

    detections_list.append(frame_dets)
    frame_count += 1

    if frame_count % LOG_INTERVAL == 0:
        print(f"Processed frame {frame_count}/{MAX_FRAMES}")

cap.release()

output = {
    "video_path": VIDEO_PATH,
    "model": "RFDETRMedium",
    "threshold": THRESHOLD,
    "total_frames": len(detections_list),
    "detections": detections_list,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Processed {len(detections_list)} frames")
print(f"Saved detections to {OUTPUT_PATH}")
