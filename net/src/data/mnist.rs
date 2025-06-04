// net/data/mnist.rs
use byteorder::{BigEndian, ReadBytesExt};
use ndarray::{Array1, Array2, Array4};
use std::fs::File;
use std::io::{BufReader, Read};

pub fn load_mnist_idx_batch(
    img_path: &str,
    lbl_path: &str,
    n: usize,
) -> (Array4<f32>, Array2<f32>) {
    let mut img_file = BufReader::new(File::open(img_path).expect("image file"));
    let magic = img_file.read_u32::<BigEndian>().unwrap();
    assert_eq!(magic, 0x0000_0803, "bad image magic");

    let num_images = img_file.read_u32::<BigEndian>().unwrap() as usize;
    let rows = img_file.read_u32::<BigEndian>().unwrap() as usize;
    let cols = img_file.read_u32::<BigEndian>().unwrap() as usize;
    assert!(rows == 28 && cols == 28);

    let take = n.min(num_images);
    let mut buf = vec![0u8; take * rows * cols];
    img_file.read_exact(&mut buf).unwrap();

    let mut images = Array4::<f32>::zeros((take, 1, 28, 28));
    for (i, px) in buf.iter().enumerate() {
        let img = i / (rows * cols);
        let rem = i % (rows * cols);
        let r = rem / cols;
        let c = rem % cols;
        images[[img, 0, r, c]] = *px as f32 / 255.0;
    }

    let mut lbl_file = BufReader::new(File::open(lbl_path).expect("label file"));
    let magic = lbl_file.read_u32::<BigEndian>().unwrap();
    assert_eq!(magic, 0x0000_0801, "bad label magic");
    let num_labels = lbl_file.read_u32::<BigEndian>().unwrap() as usize;

    let take = take.min(num_labels);
    let mut lbl_buf = vec![0u8; take];
    lbl_file.read_exact(&mut lbl_buf).unwrap();

    let mut labels = Array2::<f32>::zeros((take, 10));
    for (i, &digit) in lbl_buf.iter().enumerate() {
        labels[[i, digit as usize]] = 1.0;
    }

    (images, labels)
}
