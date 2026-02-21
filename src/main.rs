mod detection;
mod iou;
mod kalman;
mod tracker;
mod utils;

use anyhow::{Context, Result};
use detection::{InputData, OutputData};
use std::fs;
use tracker::ByteTrackTracker;

fn load_detections(path: &str) -> Result<InputData> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read detections file: {}", path))?;

    let data: InputData =
        serde_json::from_str(&content).with_context(|| "Failed to parse detections JSON")?;

    Ok(data)
}

fn load_tracked_json(path: &str) -> Result<OutputData> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read tracked file: {}", path))?;

    let data: OutputData =
        serde_json::from_str(&content).with_context(|| "Failed to parse tracked JSON")?;

    Ok(data)
}

fn compare_outputs(rust_output: &OutputData, python_output: &OutputData) -> Result<bool> {
    let mut all_match = true;

    if rust_output.total_frames != python_output.total_frames {
        println!(
            "Frame count mismatch: rust={}, python={}",
            rust_output.total_frames, python_output.total_frames
        );
        all_match = false;
    }

    if rust_output.detections.len() != python_output.detections.len() {
        println!(
            "Detections array length mismatch: rust={}, python={}",
            rust_output.detections.len(),
            python_output.detections.len()
        );
        all_match = false;
    }

    let tolerance = 1e-6;

    for (frame_idx, (rust_frame, python_frame)) in rust_output
        .detections
        .iter()
        .zip(python_output.detections.iter())
        .enumerate()
    {
        if rust_frame.len() != python_frame.len() {
            println!(
                "Frame {}: detection count mismatch: rust={}, python={}",
                frame_idx,
                rust_frame.len(),
                python_frame.len()
            );
            all_match = false;
            continue;
        }

        for (det_idx, (rust_det, python_det)) in
            rust_frame.iter().zip(python_frame.iter()).enumerate()
        {
            if rust_det.tracker_id != python_det.tracker_id {
                println!(
                    "Frame {}, Detection {}: tracker_id mismatch: rust={}, python={}",
                    frame_idx, det_idx, rust_det.tracker_id, python_det.tracker_id
                );
                all_match = false;
            }

            for (coord_idx, (&r_coord, &p_coord)) in rust_det
                .box_coords
                .iter()
                .zip(python_det.box_coords.iter())
                .enumerate()
            {
                if (r_coord - p_coord).abs() > tolerance {
                    println!(
                        "Frame {}, Detection {}, Coord {}: box mismatch: rust={}, python={}",
                        frame_idx, det_idx, coord_idx, r_coord, p_coord
                    );
                    all_match = false;
                }
            }
        }
    }

    Ok(all_match)
}

fn main() -> Result<()> {
    println!("Loading detections from data/detections.json...");

    let input_data = load_detections("data/detections.json")?;

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

    fs::write("data/tracked_rust.json", &output_json)
        .context("Failed to write tracked_rust.json")?;

    println!("Saved tracked results to data/tracked_rust.json");
    println!("Average tracker update time: {:.4} ms", avg_time);

    println!("Comparing with Python output from data/tracked.json...");

    let python_output = load_tracked_json("data/tracked.json")?;

    let matches = compare_outputs(&output_data, &python_output)?;

    if let Some(python_avg) = python_output.avg_performance_ms {
        let speedup = python_avg / avg_time;
        println!("Performance comparison:");
        println!("  Python avg: {:.4} ms", python_avg);
        println!("  Rust avg: {:.4} ms", avg_time);
        println!("  Rust is {:.2}x faster than Python", speedup);
    }

    if matches {
        println!("SUCCESS: Rust implementation matches Python output!");
    } else {
        println!("FAILURE: Rust implementation does NOT match Python output.");
    }

    Ok(())
}
