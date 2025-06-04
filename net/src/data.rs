use ndarray::{Array4, Array2};
use std::fs::File;
use std::io::{BufReader, Read};

//load data from MNIST dataset
pub fn load_mnist_idx_batch(images_path: &str, labels_path: &str, batch_size: usize) -> (Array4<f32>, Array2<f32>) {
    // Read images
    let mut images_file = BufReader::new(File::open(images_path).expect("Failed to open MNIST images"));
    let mut magic = [0u8; 4];
    images_file.read_exact(&mut magic).expect("Failed to read magic number");
    let magic = u32::from_be_bytes(magic);
    assert_eq!(magic, 2051, "Invalid MNIST image file magic number");

    let mut num_images = [0u8; 4];
    images_file.read_exact(&mut num_images).expect("Failed to read number of images");
    let num_images = u32::from_be_bytes(num_images) as usize;
    assert!(batch_size <= num_images, "Batch size larger than available images");

    let mut rows = [0u8; 4];
    images_file.read_exact(&mut rows).expect("Failed to read number of rows");
    let rows = u32::from_be_bytes(rows) as usize;

    let mut cols = [0u8; 4];
    images_file.read_exact(&mut cols).expect("Failed to read number of columns");
    let cols = u32::from_be_bytes(cols) as usize;

    // Read labels
    let mut labels_file = BufReader::new(File::open(labels_path).expect("Failed to open MNIST labels"));
    let mut magic = [0u8; 4];
    labels_file.read_exact(&mut magic).expect("Failed to read magic number");
    let magic = u32::from_be_bytes(magic);
    assert_eq!(magic, 2049, "Invalid MNIST label file magic number");

    let mut num_labels = [0u8; 4];
    labels_file.read_exact(&mut num_labels).expect("Failed to read number of labels");
    let num_labels = u32::from_be_bytes(num_labels) as usize;
    assert_eq!(num_labels, num_images, "Number of labels doesn't match number of images");

    // Read the actual data
    let mut images = Array4::<f32>::zeros((batch_size, 1, rows, cols));
    let mut labels = Array2::<f32>::zeros((batch_size, 10));

    let mut image_buffer = vec![0u8; rows * cols];
    let mut label_buffer = vec![0u8; 1];

    for i in 0..batch_size {
        // Read image
        images_file.read_exact(&mut image_buffer).expect("Failed to read image");
        for j in 0..(rows * cols) {
            let h = j / cols;
            let w = j % cols;
            images[(i, 0, h, w)] = image_buffer[j] as f32 / 255.0;
        }

        // Read label
        labels_file.read_exact(&mut label_buffer).expect("Failed to read label");
        let label = label_buffer[0] as usize;
        assert!(label < 10, "Invalid label {} at sample {}", label, i);
        labels[(i, label)] = 1.0;
    }

    (images, labels)
}

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
