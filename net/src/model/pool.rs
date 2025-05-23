use ndarray::{Array4, s};

#[derive(Debug)]
pub struct MaxPool2D {
    pub kernel_size: usize,
    pub stride: usize, 
    pub padding: usize
}

impl MaxPool2D {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self { 
            kernel_size, 
            stride, 
            padding
        }
    }

    //define the padding function
    pub fn pad_input(&self, input: &Array4<f32>) -> Array4<f32> {
        if self.padding == 0{
            input.clone()
        } else {
            let (b, c, h, w) = (
                input.shape()[0],
                input.shape()[1],
                input.shape()[2],
                input.shape()[3]
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
    pub fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        let input = self.pad_input(input);
        let (batch, channels, h_in, w_in) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );

        let out_h = (h_in - self.kernel_size) / self.stride + 1;
        let out_w = (w_in - self.kernel_size) / self.stride + 1;

        let mut output = Array4::<f32>::zeros((batch, channels, out_h, out_w));

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;

                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let ih = oh * self.stride + kh;
                                let iw = ow * self.stride + kw;

                                max_val = max_val.max(input[[b, c, ih, iw]]);
                            }
                        }

                        output[[b, c, oh, ow]] = max_val;
                    }
                }
            }
        }

        output
    }
}