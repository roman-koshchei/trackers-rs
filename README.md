# Object trackers implementation in Rust

The goal is to implement ByteTrack object tracker in Rust language based on Python implementation as a reference and point of correctness for tests.

Correctness test is important part of the project to ensure Rust implementation matches Python one.

As a reference we use next project which are added as git submodules for reference:

- [trackers](https://github.com/roboflow/trackers) for the ByteTrackTracker itself.
- [supervision](https://github.com/roboflow/supervision) for Detections struct on which `trackers` depend.
- [scipy](https://github.com/scipy/scipy) for Hungarian algo, which is used by ByteTrack.

## Initial testing setup

First we `uv run scripts/run_rfdetr.py` generic RF-DETR model to detect object on random town walk video I have found. Detections are saved in `data/detections.json` file for future runs of trackers on these detections and single point of comparision.
