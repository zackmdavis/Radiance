#![allow(dead_code)]

use std::rc::Rc;

use super::Tensor;

trait Optimizer {
    fn step(&self);
    fn unset_gradients(&self);
}

struct StochasticGradientDescentOptimizer {
    parameters: Vec<Rc<Tensor>>,
    learning_rate: f32,
    step_count: usize,
}

impl StochasticGradientDescentOptimizer {
    fn new(parameters: Vec<Rc<Tensor>>, learning_rate: f32) -> Self {
        Self {
            parameters,
            learning_rate,
            step_count: 0,
        }
    }

    fn step(&self) {
        // ...

        // Wait, how do I perform gradient updates on immutable array?
        // I think Tensor.array is going to need to be a wrapped in a RefCell
    }

    fn unset_gradients(&self) {
        for parameter in &self.parameters {
            parameter.unset_gradient();
        }
    }
}
