use crate::detection::{Detection, TrackedDetection};
use crate::iou::compute_iou_batch;
use crate::kalman::KalmanBoxTracker;
use crate::utils::{get_alive_trackers, get_associated_indices};

pub struct ByteTrackTracker {
    maximum_frames_without_update: i32,
    minimum_consecutive_frames: i32,
    minimum_iou_threshold: f32,
    track_activation_threshold: f32,
    high_conf_det_threshold: f32,
    tracks: Vec<KalmanBoxTracker>,
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
            tracks: Vec::new(),
        }
    }

    fn update_detections(
        tracks: &mut [KalmanBoxTracker],
        detections: &[Detection],
        updated_detections: &mut Vec<TrackedDetection>,
        matched_indices: &[(usize, usize)],
        minimum_consecutive_frames: i32,
    ) {
        for &(track_idx, det_idx) in matched_indices {
            let bbox = &detections[det_idx].box_coords;
            tracks[track_idx].update(bbox);

            if tracks[track_idx].number_of_successful_updates >= minimum_consecutive_frames
                && tracks[track_idx].tracker_id == -1
            {
                tracks[track_idx].tracker_id = KalmanBoxTracker::get_next_tracker_id();
            }

            updated_detections.push(TrackedDetection {
                box_coords: *bbox,
                tracker_id: tracks[track_idx].tracker_id,
            });
        }
    }

    fn split_detections(&self, detections: &[Detection]) -> (Vec<Detection>, Vec<Detection>) {
        let mut high_conf = Vec::new();
        let mut low_conf = Vec::new();

        for det in detections {
            if det.score >= self.high_conf_det_threshold {
                high_conf.push(det.clone());
            } else {
                low_conf.push(det.clone());
            }
        }

        (high_conf, low_conf)
    }

    fn spawn_new_trackers(
        &mut self,
        detections: &[Detection],
        detection_boxes: &[[f32; 4]],
        unmatched_detections: &[usize],
        updated_detections: &mut Vec<TrackedDetection>,
    ) {
        for &det_idx in unmatched_detections {
            if det_idx < detections.len() {
                let confidence = detections[det_idx].score;
                if confidence >= self.track_activation_threshold {
                    let new_tracker = KalmanBoxTracker::new(&detection_boxes[det_idx]);
                    self.tracks.push(new_tracker);

                    updated_detections.push(TrackedDetection {
                        box_coords: detection_boxes[det_idx],
                        tracker_id: -1,
                    });
                }
            }
        }
    }

    pub fn update(&mut self, detections: &[Detection]) -> Vec<TrackedDetection> {
        if self.tracks.is_empty() && detections.is_empty() {
            return Vec::new();
        }

        let mut updated_detections = Vec::new();

        for tracker in &mut self.tracks {
            tracker.predict();
        }

        let (high_conf_detections, low_conf_detections) = self.split_detections(detections);

        let high_conf_boxes: Vec<[f32; 4]> =
            high_conf_detections.iter().map(|d| d.box_coords).collect();

        let low_conf_boxes: Vec<[f32; 4]> =
            low_conf_detections.iter().map(|d| d.box_coords).collect();

        let predicted_boxes: Vec<[f32; 4]> =
            self.tracks.iter().map(|t| t.get_state_bbox()).collect();

        let (matched_indices, unmatched_track_indices, unmatched_det_indices) =
            if !high_conf_boxes.is_empty() && !predicted_boxes.is_empty() {
                let iou_matrix = compute_iou_batch(&predicted_boxes, &high_conf_boxes);
                get_associated_indices(&iou_matrix, self.minimum_iou_threshold)
            } else {
                (
                    Vec::new(),
                    (0..self.tracks.len()).collect(),
                    (0..high_conf_boxes.len()).collect(),
                )
            };

        Self::update_detections(
            &mut self.tracks,
            &high_conf_detections,
            &mut updated_detections,
            &matched_indices,
            self.minimum_consecutive_frames,
        );

        let remaining_track_indices: Vec<usize> = unmatched_track_indices;

        let (matched_indices2, _unmatched_track_indices2, unmatched_det_indices2) =
            if !low_conf_boxes.is_empty() && !remaining_track_indices.is_empty() {
                let remaining_predicted_boxes: Vec<[f32; 4]> = remaining_track_indices
                    .iter()
                    .map(|&idx| predicted_boxes[idx])
                    .collect();

                let iou_matrix = compute_iou_batch(&remaining_predicted_boxes, &low_conf_boxes);
                get_associated_indices(&iou_matrix, self.minimum_iou_threshold)
            } else {
                (
                    Vec::new(),
                    (0..remaining_track_indices.len()).collect(),
                    (0..low_conf_boxes.len()).collect(),
                )
            };

        let matched_indices2_adjusted: Vec<(usize, usize)> = matched_indices2
            .iter()
            .map(|&(i, j)| (remaining_track_indices[i], j))
            .collect::<Vec<_>>();

        Self::update_detections(
            &mut self.tracks,
            &low_conf_detections,
            &mut updated_detections,
            &matched_indices2_adjusted,
            self.minimum_consecutive_frames,
        );

        for det_idx in unmatched_det_indices2 {
            updated_detections.push(TrackedDetection {
                box_coords: low_conf_boxes[det_idx],
                tracker_id: -1,
            });
        }

        self.spawn_new_trackers(
            &high_conf_detections,
            &high_conf_boxes,
            &unmatched_det_indices,
            &mut updated_detections,
        );

        let alive_indices: Vec<usize> = get_alive_trackers(
            &self.tracks,
            self.minimum_consecutive_frames,
            self.maximum_frames_without_update,
        );

        self.tracks = alive_indices
            .into_iter()
            .map(|i| self.tracks[i].clone())
            .collect();

        updated_detections
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.tracks.clear();
        KalmanBoxTracker::reset_counter();
    }
}
