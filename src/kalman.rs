use nalgebra::{DVector, OMatrix, OVector, U4, U8};

#[derive(Clone)]
pub struct KalmanBoxTracker {
    pub tracker_id: i32,
    pub time_since_update: i32,
    pub number_of_successful_updates: i32,
    state: OVector<f32, U8>,
    f: OMatrix<f32, U8, U8>,
    h: OMatrix<f32, U4, U8>,
    q: OMatrix<f32, U8, U8>,
    r: OMatrix<f32, U4, U4>,
    p: OMatrix<f32, U8, U8>,
}

impl KalmanBoxTracker {
    pub fn new(bbox: &[f32; 4]) -> Self {
        let mut state = OVector::<f32, U8>::zeros();
        state[0] = bbox[0];
        state[1] = bbox[1];
        state[2] = bbox[2];
        state[3] = bbox[3];

        let mut f = OMatrix::<f32, U8, U8>::identity();
        for i in 0..4 {
            f[(i, i + 4)] = 1.0;
        }

        let h = OMatrix::<f32, U4, U8>::identity();
        let q = OMatrix::<f32, U8, U8>::identity() * 0.01;
        let r = OMatrix::<f32, U4, U4>::identity() * 0.1;
        let p = OMatrix::<f32, U8, U8>::identity();

        Self {
            tracker_id: -1,
            time_since_update: 0,
            number_of_successful_updates: 1,
            state,
            f,
            h,
            q,
            r,
            p,
        }
    }

    pub fn predict(&mut self) {
        self.state = self.f * self.state;
        self.p = self.f * self.p * self.f.transpose() + self.q;
        self.time_since_update += 1;
    }

    pub fn update(&mut self, bbox: &[f32; 4]) {
        self.time_since_update = 0;
        self.number_of_successful_updates += 1;

        let measurement = DVector::<f32>::from_row_slice(bbox);

        let s = self.h * self.p * self.h.transpose() + self.r;
        let s_inv = s.try_inverse().expect("Failed to invert S matrix");
        let k = self.p * self.h.transpose() * s_inv;

        let y = measurement - (self.h * self.state);
        self.state += k * y;

        let i = OMatrix::<f32, U8, U8>::identity();
        self.p = (i - k * self.h) * self.p;
    }

    pub fn get_state_bbox(&self) -> [f32; 4] {
        [self.state[0], self.state[1], self.state[2], self.state[3]]
    }
}
