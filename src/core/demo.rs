#![allow(dead_code)]

use std::rc::Rc;

use rand::Rng;

use super::{Linear, Tensor};
use super::operations::{Operation, RectifiedLinearUnit};

#[allow(unused_imports)]
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
        x
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


fn train_mlp() {
    let _network = MyLittlePerceptron::new(vec![2, 4, 1]);
    // TODO ...
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
            assert!(inputs.iter().all(|&x| x == 0.0 || x == 1.0), "Inputs should be 0.0 or 1.0");
            assert!(output == 0.0 || output == 1.0, "Output should be 0.0 or 1.0");

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
