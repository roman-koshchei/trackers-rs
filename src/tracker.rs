use crate::detection::{Detection, TrackedDetection};

pub struct ByteTrackTracker {}

impl ByteTrackTracker {
    pub fn new() -> Self {
        Self {}
    }

    pub fn update(&mut self, _detections: &[Detection]) -> Vec<TrackedDetection> {
        return Vec::new();
    }
}
