use ndarray::Axis;
use net::data::load_cifar10_bin_batch;
use net::model::SimpleCNN;

fn main() {
    let mut model = SimpleCNN::new();

    //train on 200 samples, batch1
    let (x_train, y_train) = load_cifar10_bin_batch("cifar-10-batches-bin/data_batch_1.bin", 200);

    println!("\n--- training ---");
    for i in 0..5 {
        println!("Epoch {}", i + 1);
        let loss = model.train_batch(&x_train, &y_train, 0.001);
        println!("  loss = {:.4}", loss);
    }

    //teste on 100 samples batch 2
    let (x_test, y_test) = load_cifar10_bin_batch("cifar-10-batches-bin/data_batch_2.bin", 100);
    let out = model.forward(&x_test);

    println!("\n--- testing ---");

    let mut correct = 0;
    for (i, (logits, target)) in out
        .axis_iter(Axis(0))
        .zip(y_test.axis_iter(Axis(0)))
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

        if pred == label {
            correct += 1;
        }

        println!(
            "sample {:3}: predicted {:2}, label {:2}  {}",
            i,
            pred,
            label,
            if pred == label { "✓" } else { "✗" }
        );
    }

    let accuracy = correct as f32 / x_test.len_of(Axis(0)) as f32 * 100.0;
    println!("\ntest acc: {:.2}%", accuracy);
}
