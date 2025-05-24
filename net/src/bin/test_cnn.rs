use net::model::SimpleCNN;
use net::data::load_cifar10_bin_batch;

fn main() {
    let model = SimpleCNN::new();

    let (x, y) = load_cifar10_bin_batch("cifar-10-batches-bin/data_batch_1.bin", 8);

    let out = model.forward(&x);

    for (i, (logits, target)) in out
        .axis_iter(ndarray::Axis(0))
        .zip(y.axis_iter(ndarray::Axis(0)))
        .enumerate()
    {
        let pred = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(usize::MAX);

        let label = target
            .iter()
            .position(|x| (*x - 1.0).abs() < 1e-5)
            .unwrap_or(usize::MAX);

        println!("Sample {}: predicted {}, label {}", i, pred, label);
    }
}
