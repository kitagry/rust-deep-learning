extern crate image;
extern crate mnist;
extern crate ndarray;

use mnist::{Mnist, MnistBuilder};
use ndarray::{Axis, array, rcarr1, Array1};
use std::ops::{Mul, Sub};
use rand::thread_rng;
use rand::prelude::*;
use std::vec::Vec;

// それぞれの評価関数
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

// それぞれの損失関数
pub fn mean_squared_error(y: Array1<f64>, t: Array1<f64>) -> f64 {
  y.sub(&t).map(|&el| el * el).sum() * 0.5
}

pub fn cross_entropy_error(y: Array1<f64>, t: Array1<f64>) -> f64 {
  let delta = 1e-7;
  -t.mul(&y.map(|&el| (el + delta).ln())).sum()
}

fn choice(a: usize, size: i32) -> Vec<usize> {
  let mut rng = thread_rng();
  let mut nums: Vec<usize> = (0..a).collect();
  nums.shuffle(&mut rng);
  nums[0..size as usize].to_vec()
}

fn main() {
  let (trn_size, tst_size, rows, cols) = (50_000, 10_000, 28, 28);

  // Deconstruct the returned Mnist struct.
  let Mnist {
    trn_img,
    trn_lbl,
    tst_img,
    tst_lbl,
    ..
  } = MnistBuilder::new()
    .label_format_one_hot()
    .training_set_length(trn_size as u32)
    .validation_set_length(10_000)
    .test_set_length(tst_size as u32)
    .finalize();

  let trn_img = rcarr1(&trn_img).reshape((trn_size, rows * cols));
  let trn_lbl = rcarr1(&trn_lbl).reshape((trn_size, 10));
  let tst_img = rcarr1(&tst_img).reshape((tst_size, rows * cols));
  let tst_lbl = rcarr1(&tst_lbl).reshape((tst_size, 10));

  let t = array![0., 0., 1., 0., 0., 0., 0., 0., 0., 0.];
  let y = array![0.1, 0.05, 0.6, 0., 0.05, 0.1, 0., 0.1, 0., 0.];
  println!("{}", mean_squared_error(y, t));

  let choices = choice(trn_size as usize, 10);
  println!("{:?}", trn_img.select(Axis(0), &choices));
}
