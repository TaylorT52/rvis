use tensor::tensor::Tensor4;
use tensor::storage::naive_cpu::NaiveCpu;

pub struct ReluLayer<const D0: usize, const D1: usize, const D3: usize, const D4: usize>
where
    [(); D0 * (D1 * (D3 * D4))]:,
{
    input: Option<Tensor4<f32, D0, D1, D3, D4, NaiveCpu>>,
}

impl<const D0: usize, const D1: usize, const D3: usize, const D4: usize>
    ReluLayer<D0, D1, D3, D4>
where
    [(); D0 * (D1 * (D3 * D4))]:,
{
    pub fn new() -> Self {
        Self { input: None }
    }

    //forward pass--uses backend relu on tensor
    pub fn forward(
        &mut self,
        input: Tensor4<f32, D0, D1, D3, D4, NaiveCpu>,
    ) -> Tensor4<f32, D0, D1, D3, D4, NaiveCpu> {
        //does max (0, x) (.relu())
        self.input = Some(input);
        self.input.clone().unwrap().relu()
    }

    //backward pass--uses backend relu backprop on tensor
    pub fn backward(
        &self,
        grad_output: Tensor4<f32, D0, D1, D3, D4, NaiveCpu>,
    ) -> Tensor4<f32, D0, D1, D3, D4, NaiveCpu> {
        let input = self
            .input
            .as_ref()
            .expect("ReLU forward must be called before backward");

        Tensor4::relu_backward(input, &grad_output)
    }
}
