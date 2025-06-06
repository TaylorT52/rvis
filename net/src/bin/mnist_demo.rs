use ndarray::{Array2, Array4, Axis};
use net::data::load_mnist_idx_batch;
use net::model::MnistCNN;

fn main() {
    let mut model = MnistCNN::new();

    let (x_train, y_train) = load_mnist_idx_batch(
        "mnist/train-images-idx3-ubyte",
        "mnist/train-labels-idx1-ubyte",
        10_000,
    );

    println!("\n--- training ---");
    let batch_size = 128;
    let num_batches = x_train.shape()[0] / batch_size;
    let mut learning_rate = 0.0005; 
    let momentum = 0.9; 
    let mut epochs = 10;

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;

        //shuffle data
        let mut indices: Vec<usize> = (0..x_train.shape()[0]).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        for i in 0..num_batches {
            let start = i * batch_size;
            let end = start + batch_size;

            //mini-batch
            let x_batch = x_train.select(Axis(0), &indices[start..end]);
            let y_batch = y_train.select(Axis(0), &indices[start..end]);

            let loss = model.train_batch(&x_batch, &y_batch, learning_rate);
            total_loss += loss;
        }

        // Cosine learning rate decay
        learning_rate = 0.0005 * (1.0 + (epoch as f32 * std::f32::consts::PI / 50.0).cos()) / 2.0;

        let avg_loss = total_loss / num_batches as f32;
        println!("epoch {epoch}: loss = {avg_loss:.4}, lr = {learning_rate:.6}");
    }

    //test
    let (x_test, y_test) = load_mnist_idx_batch(
        "mnist/t10k-images-idx3-ubyte",
        "mnist/t10k-labels-idx1-ubyte",
        2_000,
    );
    let (probs, _) = model.forward(&x_test);

    println!("\n--- testing ---");
    let mut correct = 0;
    for (preds, target) in probs.axis_iter(Axis(0)).zip(y_test.axis_iter(Axis(0))) {
        let preds_flat = preds.into_shape((10,)).unwrap().to_owned();

        let pred_label = preds_flat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let true_label = target.iter().position(|&v| (v - 1.0).abs() < 1e-6).unwrap();

        if pred_label == true_label {
            correct += 1;
        }
    }

    let acc = 100.0 * correct as f32 / x_test.len_of(Axis(0)) as f32;
    println!("test acc: {acc:.2}%");
}
