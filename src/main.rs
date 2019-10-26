extern crate ndarray;
extern crate mnist;
extern crate image;

use mnist::{MnistBuilder, Mnist};
use ndarray::{array, Array1};

pub fn sigmoid(x: Array1<f64>) -> Array1<f64> {
    x.map(|&el| 1. / (1. + (-el).exp()))
}

pub fn step_function(x: Array1<f64>) -> Array1<f64> {
    x.map(|&el| if el > 0. { 1. } else { 0. })
}

pub fn relu(x: Array1<f64>) -> Array1<f64> {
    x.map(|&el| if el > 0. { el } else { 0. })
}

pub fn softmax(x: Array1<f64>) -> Array1<f64> {
    let c = x.fold(0. / 0., |m, v| v.max(m));
    let sum = x.map(|&el| (el - c).exp()).sum();

    x.map(|&el| (el - c).exp() / sum)
}

fn main() {
    // dim()で要素が分かる
    // let a = array![[1., 2., 3.], [4., 5., 6.]];
    // let b = array![[1., 2.], [3., 4.], [5., 6.]];

    // println!("{:?}", a.dot(&b));

    // let a = array![1., 2., 3.];
    // println!("{:?}", sigmoid(a));

    // let a = array![1010., 1000., 990.];
    // println!("{:?}", softmax(a));

    let (trn_size, rows, cols) = (50_000, 28, 28);

    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // Get the label of the first digit.
    let first_label = trn_lbl[0];
    println!("The first digit is a {}.", first_label);

    println!("train image num: {}, test image num: {}", trn_img.len(), tst_img.len());
    println!("test image num: {}", tst_img.len() / 28 / 28);

    let mut buffer = &trn_img[0..28*28];

    image::save_buffer("image.png", buffer, 28, 28, image::Gray(8)).unwrap();
}
