pub fn compute_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x1_inter = box1[0].max(box2[0]);
    let y1_inter = box1[1].max(box2[1]);
    let x2_inter = box1[2].min(box2[2]);
    let y2_inter = box1[3].min(box2[3]);

    if x2_inter <= x1_inter || y2_inter <= y1_inter {
        return 0.0;
    }

    let inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter);
    let box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    let union_area = box1_area + box2_area - inter_area;

    if union_area <= 0.0 {
        return 0.0;
    }

    inter_area / union_area
}

// Ported from _get_iou_matrix: reference/trackers/trackers/core/sort/utils.py
//   which calls sv.box_iou_batch from reference/supervision/supervision/detection/utils/iou_and_nms.py
pub fn compute_iou_batch(
    predicted_boxes: &[[f32; 4]],
    detection_boxes: &[[f32; 4]],
) -> Vec<f32> {
    let n_rows = predicted_boxes.len();
    let n_cols = detection_boxes.len();

    if n_rows == 0 || n_cols == 0 {
        return vec![0.0f32; n_rows * n_cols];
    }

    let mut flat = vec![0.0f32; n_rows * n_cols];

    for (i, pred_box) in predicted_boxes.iter().enumerate() {
        for (j, det_box) in detection_boxes.iter().enumerate() {
            flat[i * n_cols + j] = compute_iou(pred_box, det_box);
        }
    }

    flat
}
