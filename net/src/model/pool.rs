use ndarray::{s, Array4};

#[derive(Debug)]
pub struct MaxPool2D {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    mask: Option<Array4<bool>>,
}

impl MaxPool2D {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            mask: None,
        }
    }

    //define the padding function
    pub fn pad_input(&self, input: &Array4<f32>) -> Array4<f32> {
        if self.padding == 0 {
            input.clone()
        } else {
            let (b, c, h, w) = (
                input.shape()[0],
                input.shape()[1],
                input.shape()[2],
                input.shape()[3],
            );
            let new_h = 2 * self.padding + h;
            let new_w = 2 * self.padding + w;

            let mut padded = Array4::<f32>::zeros((b, c, new_h, new_w));
            padded
                .slice_mut(s![
                    ..,
                    ..,
                    self.padding..self.padding + h,
                    self.padding..self.padding + w
                ])
                .assign(input);

            padded
        }
    }

    //forward pass with padding
    pub fn forward(&mut self, input: &Array4<f32>) -> Array4<f32> {
        let (batch, channels, height, width) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );

        let out_h = (height - self.kernel_size) / self.stride + 1;
        let out_w = (width - self.kernel_size) / self.stride + 1;

        let mut output = Array4::<f32>::zeros((batch, channels, out_h, out_w));
        let mut mask = Array4::<bool>::default(input.raw_dim());

        for b in 0..batch {
            for c in 0..channels {
                for i in 0..out_h {
                    for j in 0..out_w {
                        let h_start = i * self.stride;
                        let w_start = j * self.stride;

                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = (0, 0);

                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let h = h_start + kh;
                                let w = w_start + kw;
                                let val = input[[b, c, h, w]];
                                if val > max_val {
                                    max_val = val;
                                    max_idx = (h, w);
                                }
                            }
                        }

                        output[[b, c, i, j]] = max_val;
                        mask[[b, c, max_idx.0, max_idx.1]] = true;
                    }
                }
            }
        }

        self.mask = Some(mask);
        output
    }

    pub fn backward(&self, grad_output: &Array4<f32>) -> Array4<f32> {
        let mask = self
            .mask
            .as_ref()
            .expect("MaxPool2D: forward must be called before backward");
        let mut grad_input = Array4::<f32>::zeros(mask.raw_dim());

        let (batch, channels, out_h, out_w) = (
            grad_output.shape()[0],
            grad_output.shape()[1],
            grad_output.shape()[2],
            grad_output.shape()[3],
        );

        for b in 0..batch {
            for c in 0..channels {
                for i in 0..out_h {
                    for j in 0..out_w {
                        let h_start = i * self.stride;
                        let w_start = j * self.stride;

                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let h = h_start + kh;
                                let w = w_start + kw;

                                if h < grad_input.shape()[2] && w < grad_input.shape()[3] {
                                    if mask[[b, c, h, w]] {
                                        grad_input[[b, c, h, w]] = grad_output[[b, c, i, j]];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        grad_input
    }
}
