#[derive(Clone, Debug)]
pub struct TensorRing<const N: usize> {
    buffer: [f64; N],
    idx: usize,
    pub count: usize,
}

impl<const N: usize> TensorRing<N> {
    pub fn new() -> Self {
        Self {
            buffer: [0.0; N],
            idx: 0,
            count: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, val: f64) {
        self.buffer[self.idx] = val;
        self.idx = (self.idx + 1) % N;
        if self.count < N {
            self.count += 1;
        }
    }

    #[inline(always)]
    pub fn get_current(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        self.buffer[(self.idx + N - 1) % N]
    }

    #[inline(always)]
    pub fn velocity(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        let current = self.buffer[(self.idx + N - 1) % N];
        let prev = self.buffer[(self.idx + N - 2) % N];
        current - prev
    }

    #[inline(always)]
    pub fn acceleration(&self) -> f64 {
        if self.count < 3 { return 0.0; }
        let current = self.buffer[(self.idx + N - 1) % N];
        let prev1 = self.buffer[(self.idx + N - 2) % N];
        let prev2 = self.buffer[(self.idx + N - 3) % N];
        
        let v1 = current - prev1;
        let v2 = prev1 - prev2;
        v1 - v2
    }
    
    #[inline(always)]
    pub fn jerk(&self) -> f64 {
        if self.count < 4 { return 0.0; }
        let current = self.buffer[(self.idx + N - 1) % N];
        let p1 = self.buffer[(self.idx + N - 2) % N];
        let p2 = self.buffer[(self.idx + N - 3) % N];
        let p3 = self.buffer[(self.idx + N - 4) % N];
        
        let v1 = current - p1;
        let v2 = p1 - p2;
        let v3 = p2 - p3;
        
        let a1 = v1 - v2;
        let a2 = v2 - v3;
        
        a1 - a2
    }
}
