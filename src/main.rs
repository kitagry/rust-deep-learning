extern crate ndarray;

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
    let a = array![[1., 2., 3.], [4., 5., 6.]];
    let b = array![[1., 2.], [3., 4.], [5., 6.]];

    println!("{:?}", a.dot(&b));

    let a = array![1., 2., 3.];
    println!("{:?}", sigmoid(a));

    let a = array![1010., 1000., 990.];
    println!("{:?}", softmax(a));
}
