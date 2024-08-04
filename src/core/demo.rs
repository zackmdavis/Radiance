use std::cell::RefCell;
use std::rc::Rc;

use rand::Rng;

use ndarray::prelude::*;

use super::operations::{LeakyRectifiedLinearUnit, Operation, Reshape, SquaredError};
use super::{backprop, Tensor, TensorBuilder};
use crate::core::dense::Linear;

use super::optimization::StochasticGradientDescentOptimizer;

struct MyLittlePerceptron {
    layers: Vec<Linear>,
    activation_cache: RefCell<Option<Vec<Rc<Tensor>>>>,
}

impl MyLittlePerceptron {
    fn new(layer_dimensionalities: Vec<usize>, test_weights: bool) -> Self {
        let mut layers = Vec::new();
        for (i, window) in layer_dimensionalities.windows(2).enumerate() {
            let &[in_dimensionality, out_dimensionality] = window else {
                panic!("impossible")
            };
            println!(
                "initializing layer {} of dimensionality {}→{}",
                i, in_dimensionality, out_dimensionality
            );
            if test_weights {
                layers.push(Linear::new_with_test_weights(
                    &format!("Layer{}", i),
                    in_dimensionality,
                    out_dimensionality,
                ));
            } else {
                layers.push(Linear::new(
                    &format!("Layer{}", i),
                    in_dimensionality,
                    out_dimensionality,
                ));
            }
        }
        Self {
            layers,
            activation_cache: RefCell::new(None),
        }
    }

    fn forward(&self, input: Rc<Tensor>) -> Rc<Tensor> {
        let mut x = input;
        let mut activation_cache = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i < self.layers.len() - 1 {
                x = LeakyRectifiedLinearUnit::new(0.1).forward(vec![x]);
            }
            activation_cache.push(x.clone());
        }

        // set the activation cache
        *self.activation_cache.borrow_mut() = Some(activation_cache);

        // needs a reshape because we end up with a [1, 1] but the loss
        // function is going to expect a [1]
        x = Reshape::new(vec![1]).forward(vec![x]);
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

fn generate_xor_data(sample_count: usize) -> Vec<(Vec<f32>, f32)> {
    // contributed by Claude Sonnet 3.5
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(sample_count);

    for _ in 0..sample_count {
        let x1 = rng.gen_bool(0.5) as u8 as f32;
        let x2 = rng.gen_bool(0.5) as u8 as f32;
        let y = (x1 != x2) as u8 as f32;
        data.push((vec![x1, x2], y));
    }

    data
}

fn train_mlp() -> MyLittlePerceptron {
    let network = MyLittlePerceptron::new(vec![2, 4, 1], false);
    let mut optimizer = StochasticGradientDescentOptimizer::new(network.parameters(), 0.05);
    let training_data = generate_xor_data(2000);
    for (raw_input, raw_expected_output) in training_data {
        let input_array = Array::from_shape_vec((2, 1), raw_input.to_vec())
            .expect("shape is correct")
            .into_dyn();
        let input = Rc::new(TensorBuilder::new(input_array).build());

        let expected_output_array = Array::from_shape_vec((1,), vec![raw_expected_output])
            .expect("shape is correct")
            .into_dyn();
        let expected_output = Rc::new(TensorBuilder::new(expected_output_array).build());
        let output = network.forward(input.clone());
        let loss = SquaredError {}.forward(vec![expected_output, output]);
        backprop(loss);
        optimizer.step();
        optimizer.unset_gradients();
    }
    network
}

pub fn demo() {
    let network = train_mlp();
    println!("network trained!");
    for raw_input in [[0., 0.], [0., 1.], [1., 0.], [1., 1.]] {
        let array_form = Array::from_shape_vec((2, 1), raw_input.to_vec())
            .expect("shape is correct")
            .into_dyn();
        let input = TensorBuilder::new(array_form).build();
        let output = network.forward(Rc::new(input));
        println!("output for {:?}: {:?}", raw_input, output.array);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_xor_data_generation() {
        // Test by Claude Sonnet 3.5
        let data = generate_xor_data(1000);

        for (inputs, output) in data {
            assert!(inputs.len() == 2, "Input should have 2 elements");
            assert!(
                inputs.iter().all(|&x| x == 0.0 || x == 1.0),
                "Inputs should be 0.0 or 1.0"
            );
            assert!(
                output == 0.0 || output == 1.0,
                "Output should be 0.0 or 1.0"
            );

            // Check XOR logic
            let expected = (inputs[0] != inputs[1]) as u8 as f32;
            assert_eq!(output, expected, "XOR output should be correct");
        }
    }

    #[test]
    fn test_xor_logic() {
        // Test by Claude Sonnet 3.5
        assert_eq!((0.0_f32 != 0.0_f32) as u8 as f32, 0.0);
        assert_eq!((0.0_f32 != 1.0_f32) as u8 as f32, 1.0);
        assert_eq!((1.0_f32 != 0.0_f32) as u8 as f32, 1.0);
        assert_eq!((1.0_f32 != 1.0_f32) as u8 as f32, 0.0);
    }

    #[test]
    fn test_weights_activations_gradients_etc() {
        // If we define the same network architecture in PyTorch, we should get
        // the same results modulo floating point error.
        let network = MyLittlePerceptron::new(vec![2, 4, 1], true);

        // magic numbers should be from the parallel PyTorch implementation (whose
        // parallel test-weights function should have the same outputs)

        let layer_0_weights_before = network.layers[0]
            .weights
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");

        let expected_layer_0_weights_before = Array::from_shape_vec(
            (4, 2),
            vec![
                -0.7071, -0.6552, -0.6637, -0.6118, -0.6202, -0.5683, -0.5768, -0.5249,
            ],
        )
        .unwrap();
        assert_abs_diff_eq!(
            layer_0_weights_before,
            expected_layer_0_weights_before,
            epsilon = 0.0001
        );

        let layer_0_biases_before = network.layers[0]
            .biases
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");

        let expected_layer_0_biases_before =
            Array::from_shape_vec((4, 1), vec![-0.7071, -0.6637, -0.6202, -0.5768]).unwrap();

        assert_abs_diff_eq!(
            layer_0_biases_before,
            expected_layer_0_biases_before,
            epsilon = 0.0001
        );

        let layer_1_weights_before = network.layers[1]
            .weights
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");

        // danger: PyTorch version had shape [4], not [1, 4]
        let expected_layer_1_weights_before =
            Array::from_shape_vec((1, 4), vec![-0.5000f32, -0.4633, -0.4267, -0.3900]).unwrap();

        let layer_1_biases_before = network.layers[1]
            .biases
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");

        let expected_layer_1_biases_before =
            Array::from_shape_vec((1, 1), vec![-0.5000f32]).unwrap();

        assert_abs_diff_eq!(
            layer_1_weights_before,
            expected_layer_1_weights_before,
            epsilon = 0.0001
        );
        assert_abs_diff_eq!(
            layer_1_biases_before,
            expected_layer_1_biases_before,
            epsilon = 0.0001
        );

        let mut optimizer = StochasticGradientDescentOptimizer::new(network.parameters(), 0.05);
        let raw_input = Array::from_shape_vec((2, 1), vec![1., 1.]).unwrap();

        let input = Rc::new(TensorBuilder::new(raw_input.clone().into_dyn()).build());
        let expected_output = Rc::new(
            TensorBuilder::new(Array::from_shape_vec((1,), vec![1.]).unwrap().into_dyn()).build(),
        );

        let output = network.forward(input.clone());
        assert_abs_diff_eq!(output.array.borrow()[0], -0.1641, epsilon = 0.0001);

        let layer_0_activations = network
            .activation_cache
            .borrow()
            .clone()
            .expect("activations cached")[0]
            .array
            .borrow()
            .clone();
        let expected_layer_0_activations =
            Array::from_shape_vec((4, 1), vec![-0.2069, -0.1939, -0.1809, -0.1678])
                .unwrap()
                .into_dyn();
        assert_abs_diff_eq!(
            layer_0_activations,
            expected_layer_0_activations,
            epsilon = 0.0001
        );

        let layer_1_activations = network
            .activation_cache
            .borrow()
            .clone()
            .expect("activations cached")[1]
            .array
            .borrow()
            .clone();
        let expected_layer_1_activations = Array::from_shape_vec((1, 1), vec![-0.1641])
            .unwrap()
            .into_dyn();
        assert_abs_diff_eq!(
            layer_1_activations,
            expected_layer_1_activations,
            epsilon = 0.0001
        );

        let loss = SquaredError {}.forward(vec![expected_output, output]);
        assert_abs_diff_eq!(loss.array.borrow()[0], 1.3550, epsilon = 0.0001);

        backprop(loss);

        let layer_1_weight_gradients = network.layers[1]
            .weights
            .gradient
            .borrow()
            .clone()
            .expect("gradient should be set")
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");

        let expected_layer_1_weight_gradients =
            Array::from_shape_vec((1, 4), vec![0.4818, 0.4514, 0.4211, 0.3908]).unwrap();

        assert_abs_diff_eq!(
            layer_1_weight_gradients,
            expected_layer_1_weight_gradients,
            epsilon = 0.0001
        );

        let layer_1_bias_gradients = network.layers[1]
            .biases
            .gradient
            .borrow()
            .clone()
            .expect("gradient should be set")
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");

        let expected_layer_1_bias_gradients = Array::from_shape_vec((1, 1), vec![-2.3281]).unwrap();
        assert_abs_diff_eq!(
            layer_1_bias_gradients,
            expected_layer_1_bias_gradients,
            epsilon = 0.0001
        );

        optimizer.step();

        // TODO further testing
    }
}
