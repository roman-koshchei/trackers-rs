# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "trackers",
#   "supervision",
#   "numpy",
# ]
# ///

import json
import time
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

INPUT_PATH = "data/detections.json"
OUTPUT_PATH = "data/tracked.json"

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

detections_list = data["detections"]
total_frames = data["total_frames"]

tracker = ByteTrackTracker(track_activation_threshold=0.25)

tracked_results = []
update_times = []

for frame_idx, frame_detections in enumerate(detections_list):
    if frame_detections:
        xyxy = np.array([d["box"] for d in frame_detections], dtype=np.float32)
        class_id = np.array([d["class_id"] for d in frame_detections], dtype=int)
        confidence = np.array([d["score"] for d in frame_detections], dtype=float)
        detections = sv.Detections(xyxy=xyxy, class_id=class_id, confidence=confidence)
    else:
        detections = sv.Detections.empty()

    start_time = time.perf_counter()
    tracked = tracker.update(detections)
    end_time = time.perf_counter()
    update_times.append(end_time - start_time)

    frame_tracked = []
    for i in range(len(tracked)):
        frame_tracked.append(
            {
                "box": tracked.xyxy[i].tolist(),
                "tracker_id": int(tracked.tracker_id[i]),
            }
        )

    tracked_results.append(frame_tracked)

    if (frame_idx + 1) % 100 == 0:
        print(f"Frame {frame_idx + 1}/{total_frames}: {len(frame_tracked)} tracked objects")

avg_time = sum(update_times) / len(update_times)

output = {
    "source_file": INPUT_PATH,
    "tracker": "ByteTrack",
    "total_frames": total_frames,
    "avg_performance_ms": avg_time * 1000,
    "detections": tracked_results,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Processed {total_frames} frames")
print(f"Average tracker update time: {avg_time * 1000:.4f} ms")
print(f"Saved tracked results to {OUTPUT_PATH}")
