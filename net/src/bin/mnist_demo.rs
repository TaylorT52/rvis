use ndarray::Axis;
use net::data::load_mnist_idx_batch;
use net::model::SimpleCNN;

fn main() {
    //init the mnist model
    let mut model = SimpleCNN::new();

    //load and train on 10,000 images
    let (x_train, y_train) = load_mnist_idx_batch(
        "mnist/train-images-idx3-ubyte",
        "mnist/train-labels-idx1-ubyte",
        10_000,
    );

    println!("\n--- training ---");
    for epoch in 1..=50 {
        let loss = model.train_batch(&x_train, &y_train, 0.001);
        println!("epoch {epoch}: loss = {loss:.4}");
    }

    //test on 2,000 images (not trained on)
    let (x_test, y_test) = load_mnist_idx_batch(
        "mnist/t10k-images-idx3-ubyte",
        "mnist/t10k-labels-idx1-ubyte",
        2_000,
    );
    let probs = model.forward(&x_test);

    println!("\n--- testing ---");
    let mut correct = 0;
    for (preds, target) in probs.axis_iter(Axis(0)).zip(y_test.axis_iter(Axis(0))) {
        // preds shape: [10, 1, 1]
        let preds_flat = preds.into_shape((10,)).unwrap().to_owned();

        let pred_label = preds_flat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let true_label = target
            .iter()
            .position(|&v| (v - 1.0).abs() < 1e-6)
            .unwrap();

        if pred_label == true_label {
            correct += 1;
        }
    }

    let acc = 100.0 * correct as f32 / x_test.len_of(Axis(0)) as f32;
    println!("test acc: {acc:.2}%");
}