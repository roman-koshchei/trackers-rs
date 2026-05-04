pub mod detection;
pub mod hungarian;
pub mod tracker;

pub use detection::{Detection, TrackedDetection};
pub use tracker::ByteTrackTracker;
