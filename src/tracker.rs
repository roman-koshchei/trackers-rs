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
    next_tracker_id: i32,

    // Buffers for memory reuse
    tracks: Vec<KalmanBoxTracker>,
    updated_detections: Vec<TrackedDetection>,
    high_conf_detections: Vec<Detection>,
    low_conf_detections: Vec<Detection>,
    high_conf_boxes: Vec<[f32; 4]>,
    low_conf_boxes: Vec<[f32; 4]>,
    predicted_boxes: Vec<[f32; 4]>,
    remaining_predicted_boxes: Vec<[f32; 4]>,
    matched_indices: Vec<(usize, usize)>,
    matched_indices_adjusted: Vec<(usize, usize)>,
    unmatched_track_indices: Vec<usize>,
    unmatched_det_indices: Vec<usize>,
    alive_indices: Vec<usize>,
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
            high_conf_detections: Vec::new(),
            low_conf_detections: Vec::new(),
            high_conf_boxes: Vec::new(),
            low_conf_boxes: Vec::new(),
            predicted_boxes: Vec::new(),
            remaining_predicted_boxes: Vec::new(),
            matched_indices: Vec::new(),
            matched_indices_adjusted: Vec::new(),
            unmatched_track_indices: Vec::new(),
            unmatched_det_indices: Vec::new(),
            alive_indices: Vec::new(),
        }
    }

    fn update_detections(
        tracks: &mut [KalmanBoxTracker],
        detections: &[Detection],
        updated_detections: &mut Vec<TrackedDetection>,
        matched_indices: &[(usize, usize)],
        minimum_consecutive_frames: i32,
        next_tracker_id: &mut i32,
    ) {
        for &(track_idx, det_idx) in matched_indices {
            let bbox = &detections[det_idx].box_coords;
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

    fn split_detections(&mut self, detections: &[Detection]) {
        self.high_conf_detections.clear();
        self.low_conf_detections.clear();

        for det in detections {
            if det.score >= self.high_conf_det_threshold {
                self.high_conf_detections.push(det.clone());
            } else {
                self.low_conf_detections.push(det.clone());
            }
        }
    }

    fn spawn_new_trackers(
        &mut self,
        detections: &[Detection],
        detection_boxes: &[[f32; 4]],
        unmatched_detections: &[usize],
    ) {
        for &det_idx in unmatched_detections {
            if det_idx < detections.len() {
                let confidence = detections[det_idx].score;
                if confidence >= self.track_activation_threshold {
                    let new_tracker = KalmanBoxTracker::new(&detection_boxes[det_idx]);
                    self.tracks.push(new_tracker);

                    self.updated_detections.push(TrackedDetection {
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

        self.updated_detections.clear();

        for tracker in &mut self.tracks {
            tracker.predict();
        }

        self.split_detections(detections);

        self.high_conf_boxes.clear();
        for d in &self.high_conf_detections {
            self.high_conf_boxes.push(d.box_coords);
        }

        self.low_conf_boxes.clear();
        for d in &self.low_conf_detections {
            self.low_conf_boxes.push(d.box_coords);
        }

        self.predicted_boxes.clear();
        for t in &self.tracks {
            self.predicted_boxes.push(t.get_state_bbox());
        }

        self.matched_indices.clear();
        self.unmatched_track_indices.clear();
        self.unmatched_det_indices.clear();

        if !self.high_conf_boxes.is_empty() && !self.predicted_boxes.is_empty() {
            let iou_matrix = compute_iou_batch(&self.predicted_boxes, &self.high_conf_boxes);
            let (matched, unmatched_tracks, unmatched_dets) =
                get_associated_indices(&iou_matrix, self.minimum_iou_threshold);
            self.matched_indices = matched;
            self.unmatched_track_indices = unmatched_tracks;
            self.unmatched_det_indices = unmatched_dets;
        } else {
            self.unmatched_track_indices.extend(0..self.tracks.len());
            self.unmatched_det_indices
                .extend(0..self.high_conf_boxes.len());
        }

        Self::update_detections(
            &mut self.tracks,
            &self.high_conf_detections,
            &mut self.updated_detections,
            &self.matched_indices,
            self.minimum_consecutive_frames,
            &mut self.next_tracker_id,
        );

        self.remaining_predicted_boxes.clear();
        for &idx in &self.unmatched_track_indices {
            self.remaining_predicted_boxes
                .push(self.predicted_boxes[idx]);
        }

        self.matched_indices_adjusted.clear();
        let unmatched_det_indices2 =
            if !self.low_conf_boxes.is_empty() && !self.remaining_predicted_boxes.is_empty() {
                let iou_matrix =
                    compute_iou_batch(&self.remaining_predicted_boxes, &self.low_conf_boxes);
                let (matched2, _, unmatched_dets2) =
                    get_associated_indices(&iou_matrix, self.minimum_iou_threshold);
                for &(i, j) in &matched2 {
                    self.matched_indices_adjusted
                        .push((self.unmatched_track_indices[i], j));
                }
                unmatched_dets2
            } else {
                (0..self.low_conf_boxes.len()).collect()
            };

        Self::update_detections(
            &mut self.tracks,
            &self.low_conf_detections,
            &mut self.updated_detections,
            &self.matched_indices_adjusted,
            self.minimum_consecutive_frames,
            &mut self.next_tracker_id,
        );

        for &det_idx in &unmatched_det_indices2 {
            self.updated_detections.push(TrackedDetection {
                box_coords: self.low_conf_boxes[det_idx],
                tracker_id: -1,
            });
        }

        let high_conf_dets = self.high_conf_detections.clone();
        let high_conf_bxs = self.high_conf_boxes.clone();
        let unmatched_dets = self.unmatched_det_indices.clone();
        self.spawn_new_trackers(&high_conf_dets, &high_conf_bxs, &unmatched_dets);

        self.alive_indices.clear();
        let alive = get_alive_trackers(
            &self.tracks,
            self.minimum_consecutive_frames,
            self.maximum_frames_without_update,
        );
        self.alive_indices.extend(alive);

        let new_tracks: Vec<KalmanBoxTracker> = self
            .alive_indices
            .iter()
            .map(|&i| self.tracks[i].clone())
            .collect();
        self.tracks = new_tracks;

        std::mem::take(&mut self.updated_detections)
    }
}
