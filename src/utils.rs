use super::kalman::KalmanBoxTracker;

pub fn get_alive_trackers(
    trackers: &[KalmanBoxTracker],
    minimum_consecutive_frames: i32,
    maximum_frames_without_update: i32,
) -> Vec<usize> {
    let mut alive_indices = Vec::new();

    for (idx, tracker) in trackers.iter().enumerate() {
        let is_mature = tracker.number_of_successful_updates >= minimum_consecutive_frames;
        let is_active = tracker.time_since_update == 0;

        if tracker.time_since_update < maximum_frames_without_update && (is_mature || is_active) {
            alive_indices.push(idx);
        }
    }

    alive_indices
}

pub fn linear_sum_assignment(
    cost_matrix: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> Vec<Option<usize>> {
    if n_rows == 0 || n_cols == 0 {
        return vec![None; n_rows];
    }

    let mut nr = n_rows;
    let mut nc = n_cols;
    let mut transposed = false;

    let mut cost_owned: Vec<f32>;
    let cost: &[f32];

    if nc < nr {
        cost_owned = vec![0.0f32; nr * nc];
        for i in 0..nr {
            for j in 0..nc {
                cost_owned[j * nr + i] = cost_matrix[i * nc + j];
            }
        }
        std::mem::swap(&mut nr, &mut nc);
        transposed = true;
        cost = &cost_owned;
    } else {
        cost = cost_matrix;
    }

    let mut u = vec![0.0f32; nr];
    let mut v = vec![0.0f32; nc];
    let mut shortest_path_costs = vec![f32::INFINITY; nc];
    let mut path = vec![usize::MAX; nc];
    let mut col4row = vec![usize::MAX; nr];
    let mut row4col = vec![usize::MAX; nc];
    let mut sr = vec![false; nr];
    let mut sc = vec![false; nc];
    let mut remaining = vec![0usize; nc];

    for cur_row in 0..nr {
        let mut min_val = 0.0f32;

        let mut num_remaining = nc;
        #[allow(clippy::needless_range_loop)]
        for it in 0..nc {
            remaining[it] = nc - it - 1;
        }

        sr.fill(false);
        sc.fill(false);
        shortest_path_costs.fill(f32::INFINITY);

        let mut sink = usize::MAX;
        let mut i = cur_row;

        while sink == usize::MAX {
            let mut index = usize::MAX;
            let mut lowest = f32::INFINITY;
            sr[i] = true;

            #[allow(clippy::needless_range_loop)]
            for it in 0..num_remaining {
                let j = remaining[it];
                let r = min_val + cost[i * nc + j] - u[i] - v[j];

                if r < shortest_path_costs[j] {
                    path[j] = i;
                    shortest_path_costs[j] = r;
                }

                if shortest_path_costs[j] < lowest
                    || (shortest_path_costs[j] == lowest && row4col[j] == usize::MAX)
                {
                    lowest = shortest_path_costs[j];
                    index = it;
                }
            }

            min_val = lowest;
            if min_val == f32::INFINITY {
                break;
            }

            let j = remaining[index];
            if row4col[j] == usize::MAX {
                sink = j;
            } else {
                i = row4col[j];
            }

            sc[j] = true;
            remaining[index] = remaining[num_remaining - 1];
            num_remaining -= 1;
        }

        if sink == usize::MAX {
            break;
        }

        u[cur_row] += min_val;
        for ii in 0..nr {
            if sr[ii] && ii != cur_row && col4row[ii] != usize::MAX {
                u[ii] += min_val - shortest_path_costs[col4row[ii]];
            }
        }

        for j in 0..nc {
            if sc[j] {
                v[j] -= min_val - shortest_path_costs[j];
            }
        }

        let mut j = sink;
        loop {
            let ii = path[j];
            row4col[j] = ii;
            j = std::mem::replace(&mut col4row[ii], j);
            if ii == cur_row {
                break;
            }
        }
    }

    let mut assignment;
    if transposed {
        let mut row_assignments: Vec<Option<usize>> = vec![None; n_rows];
        for col in 0..nc {
            let row = row4col[col];
            if row != usize::MAX {
                row_assignments[col] = Some(row);
            }
        }
        assignment = row_assignments;
    } else {
        assignment = vec![None; n_rows];
        for i in 0..nr {
            if col4row[i] != usize::MAX {
                assignment[i] = Some(col4row[i]);
            }
        }
    }

    assignment
}

pub fn get_associated_indices(
    similarity_matrix: &nalgebra::DMatrix<f32>,
    min_similarity_thresh: f32,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let n_trackers = similarity_matrix.nrows();
    let n_detections = similarity_matrix.ncols();

    let mut matched_indices = Vec::new();
    let mut unmatched_tracks: Vec<usize> = (0..n_trackers).collect();
    let mut unmatched_detections: Vec<usize> = (0..n_detections).collect();

    if n_trackers > 0 && n_detections > 0 {
        let cost_matrix = -similarity_matrix;

        let mut flat_cost = vec![0.0f32; n_trackers * n_detections];
        for i in 0..n_trackers {
            for j in 0..n_detections {
                flat_cost[i * n_detections + j] = cost_matrix[(i, j)];
            }
        }

        let assignment = linear_sum_assignment(&flat_cost, n_trackers, n_detections);

        for (row, opt_col) in assignment.iter().enumerate() {
            if let Some(col) = opt_col {
                let iou = similarity_matrix[(row, *col)];
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
