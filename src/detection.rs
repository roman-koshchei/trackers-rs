use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    #[serde(rename = "box")]
    pub box_coords: [f32; 4],
    pub class_id: i32,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedDetection {
    #[serde(rename = "box")]
    pub box_coords: [f32; 4],
    pub tracker_id: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InputData {
    #[serde(default)]
    pub video_path: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub threshold: Option<f32>,
    pub total_frames: usize,
    pub detections: Vec<Vec<Detection>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputData {
    pub source_file: String,
    pub tracker: String,
    pub total_frames: usize,
    #[serde(default)]
    pub avg_performance_ms: Option<f64>,
    pub detections: Vec<Vec<TrackedDetection>>,
}
