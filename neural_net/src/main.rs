mod layers;

use mnist::{Mnist, MnistBuilder};
use tensor::storage::naive_cpu::NaiveCpu;
use tensor::tensor::Tensor4;

fn main() {
    // Run ./download_mnist.sh to download the MNIST dataset.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        // .download_and_extract()
        .finalize();

    let trn_img: Vec<f32> = trn_img.into_iter().map(|x| x as f32 / 255.0).collect();
    let train_data = Tensor4::<f32, 50_000, 1, 28, 28, NaiveCpu>::new_from_slice(&trn_img);
    println!("{}", trn_lbl.len());

    let tst_img: Vec<f32> = tst_img.into_iter().map(|x| x as f32 / 255.0).collect();
    let test_images = Tensor4::<f32, 10_000, 1, 28, 28, NaiveCpu>::new_from_slice(&tst_img);



    println!("Hello, world!");
}
