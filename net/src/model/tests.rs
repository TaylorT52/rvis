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
}
