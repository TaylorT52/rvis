use ndarray::{Array4, Array2};
use std::fs::File;
use std::io::{BufReader, Read};

//load data from CIFAR 10 dataset
pub fn load_cifar10_bin_batch(path: &str, batch_size: usize) -> (Array4<f32>, Array2<f32>) {
    const IMAGE_SIZE: usize = 32 * 32 * 3;
    const ENTRY_SIZE: usize = 1 + IMAGE_SIZE;

    let mut file = BufReader::new(File::open(path).expect("Failed to open CIFAR batch"));
    let mut buffer = vec![0u8; batch_size * ENTRY_SIZE];
    file.read_exact(&mut buffer).expect("Read failed");

    let mut images = Array4::<f32>::zeros((batch_size, 3, 32, 32));
    let mut labels = Array2::<f32>::zeros((batch_size, 10));

    for i in 0..batch_size {
        let offset = i * ENTRY_SIZE;

        let label = buffer[offset] as usize;
        assert!(
            label < 10,
            "bad label {} at sample {}",
            label, i
        );
        labels[(i, label)] = 1.0;

        let image_flat = &buffer[offset + 1..offset + ENTRY_SIZE];
        for c in 0..3 {
            let channel = &image_flat[c * 1024..(c + 1) * 1024];
            for j in 0..1024 {
                let h = j / 32;
                let w = j % 32;
                images[(i, c, h, w)] = channel[j] as f32 / 255.0;
            }
        }
    }

    (images, labels)
}
