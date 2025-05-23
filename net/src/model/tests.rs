// Inside net/src/model/conv.rs or pool.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_layer_shape() {
        let conv = Conv2D::new(3, 8, 3, 1, 1, InitScheme::HeNormal);
        let input = ndarray::Array4::<f32>::zeros((1, 3, 32, 32));
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[1, 8, 32, 32]);
    }

    #[test]
    fn test_softmax_normalizes_to_one() {
        let softmax = Softmax::new(1);

        let input = arr4(&[[[[3.0]], [[1.0]], [[0.2]]]]);
        let output = softmax.forward(&input);

        let sum: f32 = output.index_axis(Axis(1), 0)[[0, 0]]
            + output.index_axis(Axis(1), 1)[[0, 0]]
            + output.index_axis(Axis(1), 2)[[0, 0]];

        assert!((sum - 1.0).abs() < 1e-5, "softmax outputs do not sum to 1, got {}", sum);
    }

    #[test]
    fn test_relu_activation() {
        let relu = ReLU::new();
        let input = arr4(&[[[[-1.0, 0.0], [2.5, -3.3]]]]);
        let output = relu.forward(&input);
        let expected = arr4(&[[[[0.0, 0.0], [2.5, 0.0]]]]);

        assert_eq!(output, expected);
    }

    #[test]
    fn test_simple_cnn_forward_shape() {
        let model = SimpleCNN::new();
        let input = Array4::<f32>::zeros((1, 3, 32, 32)); 
        let output = model.forward(&input);

        println!("Output shape: {:?}", output.shape());
        assert_eq!(output.shape(), &[1, 16, 8, 8]);
    }
}
