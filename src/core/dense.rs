#![allow(dead_code)]

use std::rc::Rc;

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use super::{Tensor, TensorBuilder};
use crate::core::operations::{
    Addition, LeakyRectifiedLinearUnit, MatrixMultiplication, Operation,
};

pub(super) struct Linear {
    #[allow(dead_code)]
    identifier: String,
    // TODO: `pub` is dubious, but needed for the demo to still work
    pub weights: Rc<Tensor>,
    pub biases: Rc<Tensor>,
}

fn generate_test_weight(i: usize, j: usize, in_dimensionality: usize) -> f32 {
    // contributed by Claude Sonnet 3.5
    let limit = (1.0 / (in_dimensionality as f32)).sqrt();
    let pseudo_random = ((i * 31 + j * 37) % 1009) as f32 / 1009.0;
    2.0 * limit * pseudo_random - limit
}

impl Linear {
    pub fn from_weights(identifier: &str, weights: ArrayD<f32>, biases: ArrayD<f32>) -> Linear {
        Linear {
            identifier: identifier.to_owned(),
            weights: Rc::new(
                TensorBuilder::new(weights)
                    .requires_gradient(true)
                    .identifier(format!("{}_weights", identifier))
                    .build(),
            ),
            biases: Rc::new(
                TensorBuilder::new(biases)
                    .requires_gradient(true)
                    .identifier(format!("{}_biases", identifier))
                    .build(),
            ),
        }
    }

    pub fn new(identifier: &str, in_dimensionality: usize, out_dimensionality: usize) -> Linear {
        let k = 1. / (in_dimensionality as f32);
        let weights = Array::random(
            (out_dimensionality, in_dimensionality),
            Uniform::new(-k.sqrt(), k.sqrt()),
        )
        .into_dyn();
        let biases =
            Array::random((out_dimensionality, 1), Uniform::new(-k.sqrt(), k.sqrt())).into_dyn();

        println!(
            "creating Linear layer with weights shape {:?} and biases shape {:?}",
            (out_dimensionality, in_dimensionality),
            (out_dimensionality, 1)
        );
        Self::from_weights(identifier, weights, biases)
    }

    pub fn new_with_test_weights(
        identifier: &str,
        in_dimensionality: usize,
        out_dimensionality: usize,
    ) -> Linear {
        // thanks to Claude for the `from_shape_fn`/`generate_test_weight` design
        let weights = Array::from_shape_fn((out_dimensionality, in_dimensionality), |(i, j)| {
            generate_test_weight(i, j, in_dimensionality)
        })
        .into_dyn();
        let biases = Array::from_shape_fn((out_dimensionality, 1), |(i, j)| {
            generate_test_weight(i, j, in_dimensionality)
        })
        .into_dyn();
        Self::from_weights(identifier, weights, biases)
    }

    pub fn forward(&self, input: Rc<Tensor>) -> Rc<Tensor> {
        let product = MatrixMultiplication {}.forward(vec![self.weights.clone(), input]);
        let sum = Addition {}.forward(vec![product, self.biases.clone()]);
        sum
    }
}

struct MultiLayerPerceptron {
    identifier: String,
    layers: Vec<Linear>,
}

impl MultiLayerPerceptron {
    fn new(identifier: &str, layer_dimensionalities: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        for (i, window) in layer_dimensionalities.windows(2).enumerate() {
            let &[in_dimensionality, out_dimensionality] = window else {
                panic!("impossible")
            };
            layers.push(Linear::new(
                &format!("{}_layer_{}", identifier, i),
                in_dimensionality,
                out_dimensionality,
            ));
        }
        Self {
            identifier: identifier.to_owned(),
            layers,
        }
    }

    fn forward(&self, input: Rc<Tensor>) -> Rc<Tensor> {
        let mut x = input;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i < self.layers.len() - 1 {
                x = LeakyRectifiedLinearUnit::new(0.1).forward(vec![x]);
            }
        }
        x
    }

    fn parameters(&self) -> Vec<Rc<Tensor>> {
        let mut parameters = Vec::new();
        for layer in &self.layers {
            parameters.push(layer.weights.clone());
            parameters.push(layer.biases.clone());
        }
        parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_forward() {
        // Written by Claude Sonnet 3.5 (with human edits to satisfy borrow
        // checker and fix dimensionality)

        // Initialize Linear layer with known weights and biases
        let weights = array![[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]].into_dyn();
        let biases = array![[0.1], [-0.2], [0.3]].into_dyn();
        let linear = Linear::from_weights("test_linear", weights, biases);

        // Create input tensor
        let input_data = array![[1.0], [2.0]].into_dyn();
        let input = Rc::new(
            TensorBuilder::new(input_data)
                .requires_gradient(true)
                .identifier("input".to_owned())
                .build(),
        );

        // Perform forward pass
        let output = linear.forward(input);
        let output_array = output.array.borrow();
        let answer = output_array.as_slice().unwrap();

        // [1.0 * 0.1 + 2.0 * -0.2 + 0.1, 1.0 * 0.3 + 2.0 * 0.4 - 0.2, 1.0 * -0.5 + 2.0 * 0.6 + 0.3]
        let expected_output = array![-0.2, 0.9, 1.0];
        let expected_answer = expected_output.as_slice().unwrap();

        // Compare outputs
        assert_relative_eq!(answer, expected_answer, epsilon = 1e-5);
    }
}
