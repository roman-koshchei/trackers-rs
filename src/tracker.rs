use crate::detection::{Detection, TrackedDetection};
use crate::hungarian::linear_sum_assignment;
use crate::iou::compute_iou_batch;
use crate::kalman::KalmanBoxTracker;

struct DetGroup {
    boxes: Vec<[f32; 4]>,
    scores: Vec<f32>,
}

// Ported from ByteTrackTracker._get_associated_indices:
//   reference/trackers/trackers/core/bytetrack/tracker.py
//   Uses scipy.optimize.linear_sum_assignment with maximize=True, then filters by min_similarity_thresh.
//   The Rust version negates the similarity matrix (cost = -similarity) instead of passing maximize=True.
fn get_associated_indices(
    similarity_flat: &[f32],
    n_trackers: usize,
    n_detections: usize,
    min_similarity_thresh: f32,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let mut matched_indices = Vec::new();
    let mut unmatched_tracks: Vec<usize> = (0..n_trackers).collect();
    let mut unmatched_detections: Vec<usize> = (0..n_detections).collect();

    if n_trackers > 0 && n_detections > 0 {
        let flat_cost: Vec<f32> = similarity_flat.iter().map(|&v| -v).collect();

        let assignment = linear_sum_assignment(&flat_cost, n_trackers, n_detections);

        for (row, opt_col) in assignment.iter().enumerate() {
            if let Some(col) = opt_col {
                let iou = similarity_flat[row * n_detections + col];
                if iou >= min_similarity_thresh {
                    matched_indices.push((row, *col));
                    unmatched_tracks.retain(|&x| x != row);
                    unmatched_detections.retain(|&x| x != *col);
                }
            }
        }
    }

    (matched_indices, unmatched_tracks, unmatched_detections)
}

// Ported from ByteTrackTracker: reference/trackers/trackers/core/bytetrack/tracker.py
pub struct ByteTrackTracker {
    maximum_frames_without_update: i32,
    minimum_consecutive_frames: i32,
    minimum_iou_threshold: f32,
    track_activation_threshold: f32,
    high_conf_det_threshold: f32,
    next_tracker_id: i32,

    tracks: Vec<KalmanBoxTracker>,
    updated_detections: Vec<TrackedDetection>,
    predicted_boxes: Vec<[f32; 4]>,
}

impl ByteTrackTracker {
    pub fn new(
        lost_track_buffer: i32,
        frame_rate: f32,
        track_activation_threshold: f32,
        minimum_consecutive_frames: i32,
        minimum_iou_threshold: f32,
        high_conf_det_threshold: f32,
    ) -> Self {
        let maximum_frames_without_update = (frame_rate / 30.0 * lost_track_buffer as f32) as i32;

        Self {
            maximum_frames_without_update,
            minimum_consecutive_frames,
            minimum_iou_threshold,
            track_activation_threshold,
            high_conf_det_threshold,
            next_tracker_id: 0,
            tracks: Vec::new(),
            updated_detections: Vec::new(),
            predicted_boxes: Vec::new(),
        }
    }

    fn update_detections(
        tracks: &mut [KalmanBoxTracker],
        boxes: &[[f32; 4]],
        updated_detections: &mut Vec<TrackedDetection>,
        matched_indices: &[(usize, usize)],
        minimum_consecutive_frames: i32,
        next_tracker_id: &mut i32,
    ) {
        for &(track_idx, det_idx) in matched_indices {
            let bbox = &boxes[det_idx];
            tracks[track_idx].update(bbox);

            if tracks[track_idx].number_of_successful_updates >= minimum_consecutive_frames
                && tracks[track_idx].tracker_id == -1
            {
                tracks[track_idx].tracker_id = *next_tracker_id;
                *next_tracker_id += 1;
            }

            updated_detections.push(TrackedDetection {
                box_coords: *bbox,
                tracker_id: tracks[track_idx].tracker_id,
            });
        }
    }

    pub fn update(&mut self, detections: &[Detection]) -> Vec<TrackedDetection> {
        if self.tracks.is_empty() && detections.is_empty() {
            return Vec::new();
        }

        self.updated_detections.clear();

        for tracker in &mut self.tracks {
            tracker.predict();
        }

        let mut high_conf = DetGroup {
            boxes: Vec::new(),
            scores: Vec::new(),
        };
        let mut low_conf = DetGroup {
            boxes: Vec::new(),
            scores: Vec::new(),
        };

        for det in detections {
            if det.score >= self.high_conf_det_threshold {
                high_conf.boxes.push(det.box_coords);
                high_conf.scores.push(det.score);
            } else {
                low_conf.boxes.push(det.box_coords);
                low_conf.scores.push(det.score);
            }
        }

        self.predicted_boxes.clear();
        for t in &self.tracks {
            self.predicted_boxes.push(t.get_state_bbox());
        }

        let (matched_indices, unmatched_track_indices, unmatched_det_indices) =
            if !high_conf.boxes.is_empty() && !self.predicted_boxes.is_empty() {
                let n_tracks = self.predicted_boxes.len();
                let n_dets = high_conf.boxes.len();
                let iou_flat = compute_iou_batch(&self.predicted_boxes, &high_conf.boxes);
                get_associated_indices(&iou_flat, n_tracks, n_dets, self.minimum_iou_threshold)
            } else {
                (
                    Vec::new(),
                    (0..self.tracks.len()).collect(),
                    (0..high_conf.boxes.len()).collect(),
                )
            };

        Self::update_detections(
            &mut self.tracks,
            &high_conf.boxes,
            &mut self.updated_detections,
            &matched_indices,
            self.minimum_consecutive_frames,
            &mut self.next_tracker_id,
        );

        let remaining_predicted_boxes: Vec<[f32; 4]> = unmatched_track_indices
            .iter()
            .map(|&idx| self.predicted_boxes[idx])
            .collect();

        let matched_indices_adjusted;
        let unmatched_det_indices2 =
            if !low_conf.boxes.is_empty() && !remaining_predicted_boxes.is_empty() {
                let n_tracks = remaining_predicted_boxes.len();
                let n_dets = low_conf.boxes.len();
                let iou_flat = compute_iou_batch(&remaining_predicted_boxes, &low_conf.boxes);
                let (matched2, _, unmatched_dets2) =
                    get_associated_indices(&iou_flat, n_tracks, n_dets, self.minimum_iou_threshold);
                matched_indices_adjusted = matched2
                    .iter()
                    .map(|&(i, j)| (unmatched_track_indices[i], j))
                    .collect();
                unmatched_dets2
            } else {
                matched_indices_adjusted = Vec::new();
                (0..low_conf.boxes.len()).collect()
            };

        Self::update_detections(
            &mut self.tracks,
            &low_conf.boxes,
            &mut self.updated_detections,
            &matched_indices_adjusted,
            self.minimum_consecutive_frames,
            &mut self.next_tracker_id,
        );

        for &det_idx in &unmatched_det_indices2 {
            self.updated_detections.push(TrackedDetection {
                box_coords: low_conf.boxes[det_idx],
                tracker_id: -1,
            });
        }

        for &det_idx in &unmatched_det_indices {
            if high_conf.scores[det_idx] >= self.track_activation_threshold {
                let bbox = high_conf.boxes[det_idx];
                self.tracks.push(KalmanBoxTracker::new(&bbox));
                self.updated_detections.push(TrackedDetection {
                    box_coords: bbox,
                    tracker_id: -1,
                });
            }
        }

        let minimum_consecutive_frames = self.minimum_consecutive_frames;
        let maximum_frames_without_update = self.maximum_frames_without_update;
        self.tracks.retain(|t| {
            let is_mature = t.number_of_successful_updates >= minimum_consecutive_frames;
            let is_active = t.time_since_update == 0;
            t.time_since_update < maximum_frames_without_update && (is_mature || is_active)
        });

        std::mem::take(&mut self.updated_detections)
    }
}
