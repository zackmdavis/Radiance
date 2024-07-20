use std::rc::Rc;

use rand::Rng;

use ndarray::prelude::*;

use super::operations::{Operation, RectifiedLinearUnit, Reshape, SquaredError};
use super::{backprop, Linear, Tensor, TensorBuilder};

use super::optimization::StochasticGradientDescentOptimizer;

struct MyLittlePerceptron {
    layers: Vec<Linear>,
}

impl MyLittlePerceptron {
    fn new(layer_dimensionalities: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        for (i, window) in layer_dimensionalities.windows(2).enumerate() {
            let &[in_dimensionality, out_dimensionality] = window else {
                panic!("impossible")
            };
            println!(
                "initializing layer {} of dimensionality {}→{}",
                i, in_dimensionality, out_dimensionality
            );
            layers.push(Linear::new(
                &format!("Layer{}", i),
                in_dimensionality,
                out_dimensionality,
            ));
        }
        Self { layers }
    }

    fn forward(&self, input: Rc<Tensor>) -> Rc<Tensor> {
        let mut x = input;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i < self.layers.len() - 1 {
                x = RectifiedLinearUnit {}.forward(vec![x]);
            }
        }
        // needs a reshape becuase we end up with a [1, 1] but the loss
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
    let network = MyLittlePerceptron::new(vec![2, 4, 1]);
    let mut optimizer = StochasticGradientDescentOptimizer::new(network.parameters(), 0.1);
    let training_data = generate_xor_data(1000);
    for (raw_input, raw_expected_output) in training_data {
        let input_array = Array::from_shape_vec((2, 1), raw_input.to_vec())
            .expect("shape is correct")
            .into_dyn();
        let input = Rc::new(TensorBuilder::new(input_array).build());

        let expected_output_array = Array::from_shape_vec((1,), vec![raw_expected_output])
            .expect("shape is correct")
            .into_dyn();
        let expected_output = Rc::new(TensorBuilder::new(expected_output_array).build());
        let output = network.forward(input);
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
}
