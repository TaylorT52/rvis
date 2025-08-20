use layers::ReLU;
use tensor::tensor::Tensor3;
use tensor::storage::metal_gpu::MetalGpu;

#[test]
fn test_relu_creation() {
    //test that relu can be created
    let relu = ReLU::new();
    assert!(true);
}