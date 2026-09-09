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
        if self.count == 0 {
            return 0.0;
        }
        self.buffer[(self.idx + N - 1) % N]
    }

    #[inline(always)]
    pub fn velocity(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let current = self.buffer[(self.idx + N - 1) % N];
        let prev = self.buffer[(self.idx + N - 2) % N];
        current - prev
    }

    #[inline(always)]
    pub fn acceleration(&self) -> f64 {
        if self.count < 3 {
            return 0.0;
        }
        let current = self.buffer[(self.idx + N - 1) % N];
        let prev1 = self.buffer[(self.idx + N - 2) % N];
        let prev2 = self.buffer[(self.idx + N - 3) % N];

        let v1 = current - prev1;
        let v2 = prev1 - prev2;
        v1 - v2
    }

    #[inline(always)]
    pub fn jerk(&self) -> f64 {
        if self.count < 4 {
            return 0.0;
        }
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

impl<const N: usize> Default for TensorRing<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_ring_kinematics_derivatives() {
        let mut ring = TensorRing::<10>::new();
        assert_eq!(ring.get_current(), 0.0);
        assert_eq!(ring.velocity(), 0.0);
        assert_eq!(ring.acceleration(), 0.0);
        assert_eq!(ring.jerk(), 0.0);

        ring.push(10.0);
        assert_eq!(ring.get_current(), 10.0);

        ring.push(15.0); // v = 5.0
        assert_eq!(ring.velocity(), 5.0);

        ring.push(22.0); // v1 = 7.0, v2 = 5.0 => a = 2.0
        assert_eq!(ring.acceleration(), 2.0);

        ring.push(31.0); // v1 = 9.0, v2 = 7.0, v3 = 5.0 => a1 = 2.0, a2 = 2.0 => jerk = 0.0
        assert_eq!(ring.jerk(), 0.0);

        ring.push(43.0); // v1 = 12.0, v2 = 9.0, v3 = 7.0 => a1 = 3.0, a2 = 2.0 => jerk = 1.0
        assert_eq!(ring.jerk(), 1.0);
    }

    #[test]
    fn test_tensor_ring_wrap_around() {
        let mut ring = TensorRing::<3>::new();
        ring.push(1.0);
        ring.push(2.0);
        ring.push(3.0);
        ring.push(4.0);

        assert_eq!(ring.count, 3);
        assert_eq!(ring.get_current(), 4.0);
        assert_eq!(ring.velocity(), 1.0);
    }
}
