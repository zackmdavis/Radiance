use std::rc::Rc;

use ndarray::prelude::*;

use super::Tensor;

pub trait Optimizer {
    fn step(&mut self);
    fn step_count(&self) -> usize;
    fn unset_gradients(&self);
}

pub struct AdaptiveMomentEstimationOptimizer {
    parameters: Vec<Rc<Tensor>>,
    learning_rate: f32,
    first_moment_estimates: Vec<ArrayD<f32>>,
    second_moment_estimates: Vec<ArrayD<f32>>,
    first_moment_estimate_decay: f32,
    second_moment_estimate_decay: f32,
    ε: f32,
    step_count: usize,
}

impl AdaptiveMomentEstimationOptimizer {
    pub fn new(
        parameters: Vec<Rc<Tensor>>,
        learning_rate: f32,
        first_moment_estimate_decay: f32,
        second_moment_estimate_decay: f32,
        ε: f32,
    ) -> Self {
        let mut first_moment_estimates = Vec::new();
        let mut second_moment_estimates = Vec::new();
        for parameter in &parameters {
            first_moment_estimates.push(Array::zeros(parameter.borrow_array().shape()));
            second_moment_estimates.push(Array::zeros(parameter.borrow_array().shape()));
        }

        Self {
            parameters,
            learning_rate,
            first_moment_estimates,
            second_moment_estimates,
            first_moment_estimate_decay,
            second_moment_estimate_decay,
            ε,
            step_count: 0,
        }
    }
}

impl Optimizer for AdaptiveMomentEstimationOptimizer {
    fn step(&mut self) {
        for (i, parameter) in self.parameters.iter().enumerate() {
            let mut array = parameter.array.borrow_mut();
            let and_some_gradient = parameter.gradient.borrow();
            let some_and_gradient = and_some_gradient.as_ref();
            let gradient = some_and_gradient.expect("gradient should exist");

            self.first_moment_estimates[i] = self.first_moment_estimate_decay
                * &self.first_moment_estimates[i]
                + (1. - self.first_moment_estimate_decay) * gradient;
            self.second_moment_estimates[i] = self.second_moment_estimate_decay
                * &self.second_moment_estimates[i]
                + (1. - self.second_moment_estimate_decay) * (gradient * gradient);

            let corrected_first_moments = self.first_moment_estimates[i].clone()
                / (1.
                    - self
                        .first_moment_estimate_decay
                        .powi((self.step_count + 1) as i32));
            let corrected_second_moments = self.second_moment_estimates[i].clone()
                / (1.
                    - self
                        .second_moment_estimate_decay
                        .powi((self.step_count + 1) as i32));

            *array = &*array
                - self.learning_rate * &corrected_first_moments
                    / (corrected_second_moments.sqrt() + self.ε);
        }

        self.step_count += 1;
    }

    fn step_count(&self) -> usize {
        self.step_count
    }

    fn unset_gradients(&self) {
        for parameter in &self.parameters {
            parameter.unset_gradient();
        }
    }
}

pub struct StochasticGradientDescentOptimizer {
    parameters: Vec<Rc<Tensor>>,
    learning_rate: f32,
    step_count: usize,
}

impl StochasticGradientDescentOptimizer {
    pub fn new(parameters: Vec<Rc<Tensor>>, learning_rate: f32) -> Self {
        Self {
            parameters,
            learning_rate,
            step_count: 0,
        }
    }
}

impl Optimizer for StochasticGradientDescentOptimizer {
    fn step(&mut self) {
        for parameter in &self.parameters {
            let mut array = parameter.array.borrow_mut();
            // Compiler refuses to do one-shot conversion from
            // &Option<ArrayD<f32>> to Option<&ArrayD<f32>> to &ArrayD<f32>
            let and_some_gradient = parameter.gradient.borrow();
            let some_and_gradient = and_some_gradient.as_ref();
            let gradient = some_and_gradient.expect("gradient should exist");
            *array -= &(self.learning_rate * gradient);
        }
        self.step_count += 1;
    }

    fn step_count(&self) -> usize {
        self.step_count
    }

    fn unset_gradients(&self) {
        for parameter in &self.parameters {
            parameter.unset_gradient();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    use crate::core::operations::{Exponentiation, Operation};
    use crate::core::{backprop, TensorBuilder};

    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sgd_optimization_step() {
        // Test written by Claude Sonnet 3.5
        let tensor1 = Rc::new(
            TensorBuilder::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn())
                .requires_gradient(true)
                .gradient(array![[0.1, 0.2], [0.3, 0.4]].into_dyn())
                .build(),
        );

        let tensor2 = Rc::new(
            TensorBuilder::new(array![5.0, 6.0].into_dyn())
                .requires_gradient(true)
                .gradient(array![0.5, 0.6].into_dyn())
                .build(),
        );

        // Create optimizer
        let parameters = vec![tensor1.clone(), tensor2.clone()];
        let learning_rate = 0.1;
        let mut optimizer = StochasticGradientDescentOptimizer::new(parameters, learning_rate);

        // Perform optimization step
        optimizer.step();

        // Check updated values
        {
            let array1 = tensor1.array.borrow();
            assert_eq!(*array1, array![[0.99, 1.98], [2.97, 3.96]].into_dyn());
        }
        {
            let array2 = tensor2.array.borrow();
            assert_eq!(*array2, array![4.95, 5.94].into_dyn());
        }

        // Check gradients still exist
        assert!(tensor1.gradient.borrow().is_some());
        assert!(tensor2.gradient.borrow().is_some());

        // Unset gradients
        optimizer.unset_gradients();

        // Check gradients are unset
        assert!(tensor1.gradient.borrow().is_none());
        assert!(tensor2.gradient.borrow().is_none());
    }

    #[test]
    fn test_early_adam_steps() {
        // golden values from PyTorch—
        //
        // import torch
        // torch.set_printoptions(precision=8)

        // my_tensor = torch.tensor([[-2.0, -1.0], [1.0, 2.0]], requires_grad=True)
        // optimizer = torch.optim.Adam([my_tensor], lr=0.05, betas=(0.9, 0.999), eps=1e-8)

        // for step in range(3):
        //     print("step", step)
        //     exp_my_tensor = torch.exp(my_tensor)
        //     exp_my_tensor.backward(torch.ones_like(my_tensor))
        //     print("gradient", my_tensor.grad)
        //     optimizer.step()
        //     optimizer.zero_grad()
        //     print("my_tensor", my_tensor)
        let my_tensor = Rc::new(
            TensorBuilder::new(array![[-2.0, -1.0], [1.0, 2.0]].into_dyn())
                .requires_gradient(true)
                .build(),
        );
        let parameters = vec![my_tensor.clone()];
        let learning_rate = 0.05;
        let β1 = 0.9;
        let β2 = 0.999;
        let ε = 1e-8;
        let mut optimizer =
            AdaptiveMomentEstimationOptimizer::new(parameters, learning_rate, β1, β2, ε);

        let out_1 = Exponentiation {}.forward(vec![my_tensor.clone()]);
        backprop(out_1);

        assert_abs_diff_eq!(
            my_tensor
                .gradient
                .borrow()
                .clone()
                .expect("gradient exists"),
            array![[0.13533528, 0.36787945], [2.71828175, 7.38905621]].into_dyn()
        );

        optimizer.step();
        optimizer.unset_gradients();

        assert_abs_diff_eq!(
            *my_tensor.borrow_array(),
            array![[-2.04999995, -1.04999995], [0.94999999, 1.95000005]].into_dyn()
        );

        let out_2 = Exponentiation {}.forward(vec![my_tensor.clone()]);
        backprop(out_2);

        assert_abs_diff_eq!(
            my_tensor
                .gradient
                .borrow()
                .clone()
                .expect("gradient exists"),
            array![[0.12873492, 0.34993777], [2.58570957, 7.02868795]].into_dyn()
        );

        optimizer.step();
        optimizer.unset_gradients();

        assert_abs_diff_eq!(
            *my_tensor.borrow_array(),
            array![[-2.09991932, -1.09991920], [0.90008074, 1.90008080]].into_dyn(),
            epsilon = 1e-6
        );

        let out_3 = Exponentiation {}.forward(vec![my_tensor.clone()]);
        backprop(out_3);

        assert_abs_diff_eq!(
            my_tensor
                .gradient
                .borrow()
                .clone()
                .expect("gradient exists"),
            array![[0.12246631, 0.33289799], [2.45980167, 6.68643475]].into_dyn(),
            epsilon = 1e-6
        );

        optimizer.step();
        optimizer.unset_gradients();

        assert_abs_diff_eq!(
            *my_tensor.borrow_array(),
            array![[-2.14970469, -1.14970446], [0.85029542, 1.85029554]].into_dyn(),
            epsilon = 1e-6
        );
    }
}
