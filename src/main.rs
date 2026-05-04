mod detection;
mod hungarian;
mod iou;
mod kalman;
mod tracker;

use anyhow::{Context, Result};
use detection::{InputData, OutputData};
use std::fs;
use tracker::ByteTrackTracker;

fn main() -> Result<()> {
    println!("Loading detections from data/detections.json...");

    let input_data_content = fs::read_to_string("data/detections.json")
        .with_context(|| "Failed to read detections file: data/detections.json")?;

    let input_data: InputData = serde_json::from_str(&input_data_content)
        .with_context(|| "Failed to parse detections JSON")?;

    println!("Total frames: {}", input_data.total_frames);
    println!("Processing frames with ByteTrack...");

    let mut tracker = ByteTrackTracker::new(
        30,   // lost_track_buffer
        30.0, // frame_rate
        0.25, // track_activation_threshold
        2,    // minimum_consecutive_frames
        0.1,  // minimum_iou_threshold
        0.6,  // high_conf_det_threshold
    );

    let mut tracked_results = Vec::new();
    let mut update_times = Vec::new();

    for (frame_idx, frame_detections) in input_data.detections.iter().enumerate() {
        let start = std::time::Instant::now();
        let tracked = tracker.update(frame_detections);
        let duration = start.elapsed();
        update_times.push(duration.as_secs_f64());

        let count = tracked.len();
        tracked_results.push(tracked);

        if (frame_idx + 1) % 100 == 0 {
            println!(
                "Frame {}/{}: {} tracked objects",
                frame_idx + 1,
                input_data.total_frames,
                count
            );
        }
    }

    let avg_time = if update_times.is_empty() {
        0.0
    } else {
        update_times.iter().sum::<f64>() / update_times.len() as f64 * 1000.0
    };

    let output_data = OutputData {
        source_file: "data/detections.json".to_string(),
        tracker: "ByteTrack".to_string(),
        total_frames: input_data.total_frames,
        avg_performance_ms: Some(avg_time),
        detections: tracked_results,
    };

    let output_json =
        serde_json::to_string_pretty(&output_data).context("Failed to serialize output JSON")?;

    fs::write("data/tracked_rs.json", &output_json).context("Failed to write tracked_rs.json")?;

    println!("Saved tracked results to data/tracked_rs.json");
    println!("Average tracker update time: {:.4} ms", avg_time);

    Ok(())
}
