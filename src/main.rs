use plotlib::page::Page;
use plotlib::line::{Line, Style};
use plotlib::style::Line as LineStyle;
use plotlib::view::ContinuousView;

pub fn sigmoid(x: Vec<f64>) -> Vec<f64> {
    x.into_iter().map(|el| 1. / (1. + (-el).exp())).collect()
}

pub fn step_function(x: Vec<f64>) -> Vec<f64> {
    x.into_iter().map(|el| if el > 0. { 1. } else { 0. }).collect()
}

fn main() {
    let x = (-500..500).map(|n| n as f64 / 100.).collect::<Vec<f64>>();
    let y = step_function(x.clone());

    let data = x.into_iter().enumerate().map(|(i, xd)| (xd, y[i])).collect::<Vec<(f64, f64)>>();

    let l1 = Line::new(&data).style(
        Style::new().colour("#DD3355"),
    );

    let v = ContinuousView::new()
        .add(&l1)
        .x_range(-5.5, 5.5)
        .y_range(-0.5, 1.5);

    Page::single(&v).save("step_function.svg").unwrap();

    let x = (-500..500).map(|n| n as f64 / 100.).collect::<Vec<f64>>();
    let y = sigmoid(x.clone());

    let data = x.into_iter().enumerate().map(|(i, xd)| (xd, y[i])).collect::<Vec<(f64, f64)>>();

    let l1 = Line::new(&data).style(
        Style::new().colour("#DD3355"),
    );

    let v = ContinuousView::new()
        .add(&l1)
        .x_range(-5.5, 5.5)
        .y_range(-0.5, 1.5);

    Page::single(&v).save("sigmoid.svg").unwrap();
}
