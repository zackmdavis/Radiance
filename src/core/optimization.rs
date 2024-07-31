use std::rc::Rc;

use super::Tensor;

pub trait Optimizer {
    fn step(&self);
    fn unset_gradients(&self);
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

    pub fn step(&mut self) {
        for parameter in &self.parameters {
            let mut array = parameter.array.borrow_mut();
            // Compiler refuses to one-shot conversion from
            // &Option<ArrayD<f32>> to Option<&ArrayD<f32>> to &ArrayD<f32>
            let and_some_gradient = parameter.gradient.borrow();
            let some_and_gradient = and_some_gradient.as_ref();
            let gradient = some_and_gradient.expect("gradient should exist");
            *array -= &(self.learning_rate * gradient);
        }
        self.step_count += 1;
        if self.step_count % 10 == 0 {
            println!("optimization step {}!", self.step_count);
        }
    }

    pub fn unset_gradients(&self) {
        for parameter in &self.parameters {
            parameter.unset_gradient();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TensorBuilder;
    use ndarray::array;

    #[test]
    fn test_optimization_step() {
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
}
