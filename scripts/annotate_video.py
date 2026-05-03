import argparse
import json

import cv2
import numpy as np
import supervision as sv


def main():
    parser = argparse.ArgumentParser(
        description="Annotate a video with tracked detections."
    )
    parser.add_argument(
        "--detections",
        default="data/detections.json",
        help="Path to detections JSON file (default: data/detections.json)",
    )
    parser.add_argument(
        "--tracked",
        default="data/tracked_rs.json",
        help="Path to tracked detections JSON file (default: data/tracked_rs.json)",
    )
    parser.add_argument(
        "--output",
        default="data/annotated_video_rs.mp4",
        help="Output video path (default: data/annotated_video_rs.mp4)",
    )
    args = parser.parse_args()

    with open(args.detections, "r") as f:
        detections_data = json.load(f)

    with open(args.tracked, "r") as f:
        tracked_data = json.load(f)

    video_path = detections_data["video_path"]
    tracked_list = tracked_data["detections"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)

    for frame_idx in range(total_frames):
        success, frame = cap.read()
        if not success:
            break

        if frame_idx < len(tracked_list):
            tracked_detections = tracked_list[frame_idx]

            if tracked_detections:
                xyxy = np.array(
                    [d["box"] for d in tracked_detections], dtype=np.float32
                )
                tracker_id = np.array(
                    [d["tracker_id"] for d in tracked_detections], dtype=int
                )
                labels = [f"ID: {tid}" for tid in tracker_id]

                detections = sv.Detections(
                    xyxy=xyxy,
                    tracker_id=tracker_id,
                )

                annotated_frame = box_annotator.annotate(frame, detections)
                annotated_frame = label_annotator.annotate(
                    annotated_frame, detections, labels=labels
                )
            else:
                annotated_frame = frame
        else:
            annotated_frame = frame

        writer.write(annotated_frame)

        if (frame_idx + 1) % 100 == 0:
            print(f"Frame {frame_idx + 1}/{total_frames} processed")

    cap.release()
    writer.release()

    print(f"Processed {total_frames} frames")
    print(f"Saved annotated video to {args.output}")


if __name__ == "__main__":
    main()
