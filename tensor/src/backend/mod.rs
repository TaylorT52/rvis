use crate::tensor::Tensor;

mod unoptimized;

pub trait Backend {
    // mul, add, sub

    fn mul<T, N>(a: TensorND<T, N, Self>, b: TensorND<T, N, Self>) -> TensorND<T, N, Self>;
}