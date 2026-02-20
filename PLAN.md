# Rust implementation of ByteTrack

There is Python implementation of ByteTrack that can be found inside of `./reference/trackers` package directory.

There is 3 scripts in this project to run Python implementation on ByteTrack tracker which generates:
- `detections.json` with array of detections from video
- `tracked.json` with tracked detection using ByteTrack

Initialize Rust package in this project directory and implement ByteTrack based on correct implementation in `./reference/trackers`. Run it on array of detections from `detections.json` like `run_rfdetr_video.py` does but use Rust implementation. Then compare `tracked.json` created by Python implementation and results from Rust implementation, if they are the same, then consider Rust implementation to be correct.
