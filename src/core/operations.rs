use std::rc::Rc;

use ndarray::prelude::*;

use super::{Origin, Tensor, TensorBuilder};

pub(super) struct Addition {}

pub(super) trait Operation {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor>;
    fn backward(
        &self,
        _out_gradient: &ArrayD<f32>,
        // TODO CONSISTENCY: why did I say `inputs` going forwards, but `args`
        // going backwards?
        _args: Vec<Rc<Tensor>>,
        _arg_index: usize,
    ) -> ArrayD<f32>;
}

impl Operation for Addition {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        assert!(inputs.len() == 2, "binary operation expected");
        // clone has dubious performance implications?
        let array = inputs[0].array.borrow().clone() + inputs[1].array.borrow().clone();
        let origin = Origin {
            operation: Box::new(Addition {}),
            parents: vec![inputs[0].clone(), inputs[1].clone()],
        };
        Rc::new(TensorBuilder::new(array).origin(origin).build())
    }
    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        _args: Vec<Rc<Tensor>>,
        _arg_index: usize,
    ) -> ArrayD<f32> {
        // Addition just passes the gradient through to both branches.
        out_gradient.clone()
    }
}

pub(super) struct Multiplication {}

impl Operation for Multiplication {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        assert!(inputs.len() == 2, "binary operation expected");
        let array = inputs[0].array.borrow().clone() * inputs[1].array.borrow().clone(); // dubious performance &c.
        let origin = Origin {
            operation: Box::new(Multiplication {}),
            parents: vec![inputs[0].clone(), inputs[1].clone()],
        };
        Rc::new(TensorBuilder::new(array).origin(origin).build())
    }
    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        args: Vec<Rc<Tensor>>,
        arg_index: usize,
    ) -> ArrayD<f32> {
        let other_arg_index = match arg_index {
            0 => 1,
            1 => 0,
            _ => panic!("binary operation expected"),
        };
        // d/dx(xy) = y
        out_gradient * args[other_arg_index].array.borrow().clone() // dubious perf &c.
    }
}

pub(super) struct MatrixMultiplication {}

impl Operation for MatrixMultiplication {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        assert!(inputs.len() == 2, "binary operation expected");
        let a = inputs[0]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");
        let b = inputs[1]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");
        let array = a.dot(&b).into_dyn();
        let origin = Origin {
            operation: Box::new(MatrixMultiplication {}),
            parents: vec![inputs[0].clone(), inputs[1].clone()],
        };
        Rc::new(TensorBuilder::new(array).origin(origin).build())
    }

    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        args: Vec<Rc<Tensor>>,
        arg_index: usize,
    ) -> ArrayD<f32> {
        let out_gradient = out_gradient
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("out gradient is two-dimensional");
        // matrix multiplication is not commutative; separate cases for
        // out_gradient @ B^T and A^T @ out_gradient
        match arg_index {
            0 => {
                let other = args[1]
                    .array
                    .borrow()
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .expect("arg1 is two-dimensional");
                let other_transpose = other.t();
                out_gradient.dot(&other_transpose).into_dyn()
            }
            1 => {
                let other = args[0]
                    .array
                    .borrow()
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .expect("arg0 is two-dimensional");
                let other_transpose = other.t();
                other_transpose.dot(&out_gradient).into_dyn()
            }
            _ => panic!("binary operation expected"),
        }
    }
}

pub(super) struct RectifiedLinearUnit {}

impl Operation for RectifiedLinearUnit {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        assert!(inputs.len() == 1, "unary operation expected");
        let array = inputs[0]
            .array
            .borrow()
            .map(|&x| if x > 0. { x } else { 0. });
        let origin = Origin {
            operation: Box::new(RectifiedLinearUnit {}),
            parents: vec![inputs[0].clone()],
        };
        Rc::new(TensorBuilder::new(array).origin(origin).build())
    }

    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        args: Vec<Rc<Tensor>>,
        _arg_index: usize,
    ) -> ArrayD<f32> {
        let mut gradient = Array::zeros(args[0].array.borrow().shape()).into_dyn();
        azip!((g in &mut gradient, o in out_gradient, a in &*args[0].array.borrow()) if a > &0. { *g += o });
        gradient
    }
}

pub(super) struct LeakyRectifiedLinearUnit {
    leak: f32,
}

impl LeakyRectifiedLinearUnit {
    pub fn new(leak: f32) -> Self {
        Self { leak }
    }
}

impl Operation for LeakyRectifiedLinearUnit {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        assert!(inputs.len() == 1, "unary operation expected");
        let array = inputs[0]
            .array
            .borrow()
            .map(|&x| if x > 0. { x } else { self.leak * x });
        let origin = Origin {
            operation: Box::new(LeakyRectifiedLinearUnit { leak: self.leak }),
            parents: vec![inputs[0].clone()],
        };
        Rc::new(TensorBuilder::new(array).origin(origin).build())
    }

    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        args: Vec<Rc<Tensor>>,
        _arg_index: usize,
    ) -> ArrayD<f32> {
        let mut gradient = Array::zeros(args[0].array.borrow().shape()).into_dyn();
        azip!(
            (g in &mut gradient, o in out_gradient, a in &*args[0].array.borrow())
              if a > &0. {
                  *g += o
              } else {
                  *g += self.leak * o
              }
        );
        gradient
    }
}

pub(super) struct Reshape {
    new_shape: Vec<usize>,
}

impl Reshape {
    pub fn new(new_shape: Vec<usize>) -> Self {
        Self { new_shape }
    }
}

impl Operation for Reshape {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        assert!(inputs.len() == 1, "unary operation expected");
        let array = inputs[0]
            .array
            .borrow()
            .clone()
            .into_shape(self.new_shape.clone())
            .expect("input must match");
        let origin = Origin {
            operation: Box::new(Reshape {
                new_shape: self.new_shape.clone(),
            }),
            parents: inputs.clone(),
        };
        Rc::new(TensorBuilder::new(array).origin(origin).build())
    }

    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        args: Vec<Rc<Tensor>>,
        arg_index: usize,
    ) -> ArrayD<f32> {
        assert!(arg_index == 0);
        out_gradient
            .clone()
            .into_shape(args[0].array.borrow().shape())
            .expect("input shape should match")
    }
}

pub(super) struct SquaredError {}

impl Operation for SquaredError {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        assert!(inputs.len() == 2, "binary operation expected");
        let prediction = inputs[0].array.borrow()[0];
        let target = inputs[1].array.borrow()[0];
        let squared_error = (target - prediction).powf(2.0);
        let origin = Origin {
            operation: Box::new(SquaredError {}),
            parents: inputs.clone(),
        };
        Rc::new(
            TensorBuilder::new(array![squared_error].into_dyn())
                .origin(origin)
                .build(),
        )
    }

    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        args: Vec<Rc<Tensor>>,
        arg_index: usize,
    ) -> ArrayD<f32> {
        // d/dx (y − x)² = 2(y − x) · d/dx(y − x) = 2(y − x) · −1 = −2(y − x)
        // d/dy (y − x)² = 2(y − x) · d/dy(y − x) = 2(y − x) · 1 = 2(y − x)
        let prediction = args[0].array.borrow()[0];
        let target = args[1].array.borrow()[0];
        let ddp = 2. * (target - prediction);
        let local_gradient = match arg_index {
            0 => -ddp,
            1 => ddp,
            _ => panic!("binary operation expected"),
        };
        out_gradient * array![local_gradient].into_dyn()
    }
}

struct SoftmaxCrossEntropy {}

fn softmax(x: Array1<f32>) -> Array1<f32> {
    let exp_x = x.iter().map(|x_i| x_i.exp()).collect::<Array1<f32>>();
    let scale: f32 = exp_x.iter().sum();
    let softmaxed = exp_x.iter().map(|x_i| x_i / scale).collect::<Array1<f32>>();
    softmaxed
}

impl Operation for SoftmaxCrossEntropy {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        assert!(inputs.len() == 2, "binary operation expected");
        let prediction = inputs[0]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix1>()
            .expect("one-dimensional");
        let target = inputs[1]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix1>()
            .expect("one-dimensional");

        // − Σ_i y_i · log softmax(x)_i
        let softmaxed = softmax(prediction);
        let loss: f32 = target.iter().zip(softmaxed).map(|(t, s)| -t * s.ln()).sum();

        let origin = Origin {
            operation: Box::new(SoftmaxCrossEntropy {}),
            parents: inputs.clone(),
        };
        Rc::new(
            TensorBuilder::new(array![loss].into_dyn())
                .origin(origin)
                .build(),
        )
    }

    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        args: Vec<Rc<Tensor>>,
        _arg_index: usize,
    ) -> ArrayD<f32> {
        assert!(args.len() == 2, "binary operation expected");
        // But we're ignoring `_arg_index` because we only care about the
        // gradient of the predictions.
        let prediction = args[0]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix1>()
            .expect("one-dimensional");
        let target = args[1]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix1>()
            .expect("one-dimensional");
        let softmaxed = softmax(prediction);
        // Loss is − Σ_i y_i · log softmax(x)_i
        // = − Σ_i y_i · log(exp(x_i)/Σ_j exp(x_j))
        // = − Σ_i y_i · (log(exp(x_i)) − log Σ_j exp(x_j))
        // = − Σ_i y_i · (x_i − log Σ_j exp(x_j))
        // so dL/dx_i = −y_i + (Σ_i y_i) · 1/(Σ_j exp(x_j)) · exp(x_j).
        // But (Σ_i y_i) is unity (because it's a probability distribution),
        // and the last two factors are themselves softmax
        //
        // The fact that this turns out to be so tidy is the motivation for
        // combining a softmax activation with a cross entropy-loss as one
        // operation.
        out_gradient * target.iter().zip(softmaxed).map(|(t, s)| s - t).collect::<Array1<f32>>().into_dyn()
    }
}

// TODO: token-embedding, positional-embedding, layernorm ...

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backprop;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_addition_forward() {
        let a = TensorBuilder::new(array![1.].into_dyn()).build();
        let b = TensorBuilder::new(array![2.].into_dyn()).build();
        let c = Addition {}.forward(vec![Rc::new(a), Rc::new(b)]);
        assert_eq!(*c.array.borrow(), array![3.].into_dyn());
    }

    #[test]
    fn test_multiplication_forward() {
        let a = TensorBuilder::new(array![2.].into_dyn()).build();
        let b = TensorBuilder::new(array![3.].into_dyn()).build();
        let c = Multiplication {}.forward(vec![Rc::new(a), Rc::new(b)]);
        assert_eq!(*c.array.borrow(), array![6.].into_dyn());
    }

    #[test]
    fn test_addition_backward() {
        let a = TensorBuilder::new(array![1.].into_dyn()).build();
        let b = TensorBuilder::new(array![2.].into_dyn()).build();
        let args = vec![Rc::new(a), Rc::new(b)];
        let out_gradient = array![1.].into_dyn();
        assert_eq!(
            Addition {}.backward(&out_gradient, args.clone(), 0),
            out_gradient
        );
        assert_eq!(
            Addition {}.backward(&out_gradient, args.clone(), 1),
            out_gradient
        );
    }

    #[test]
    fn test_multiplication_backward() {
        let a = TensorBuilder::new(array![2.].into_dyn()).build();
        let b = TensorBuilder::new(array![3.].into_dyn()).build();
        let args = vec![Rc::new(a), Rc::new(b)];
        let out_gradient = array![1.].into_dyn();
        assert_eq!(
            Multiplication {}.backward(&out_gradient, args.clone(), 0),
            array![3.].into_dyn()
        );
        assert_eq!(
            Multiplication {}.backward(&out_gradient, args.clone(), 1),
            array![2.].into_dyn()
        );
    }

    #[test]
    fn test_matrix_multiplication() {
        // Test written by Claude 3.5 Sonnet
        let a = Rc::new(
            TensorBuilder::new(array![[1., 2.], [3., 4.]].into_dyn())
                .identifier("a".to_string())
                .requires_gradient(true)
                .build(),
        );
        let b = Rc::new(
            TensorBuilder::new(array![[5., 6.], [7., 8.]].into_dyn())
                .identifier("b".to_string())
                .requires_gradient(true)
                .build(),
        );

        // Perform forward pass
        let matmul = MatrixMultiplication {};
        let result = matmul.forward(vec![a.clone(), b.clone()]);

        // Check forward pass result
        assert_eq!(
            *result.array.borrow(),
            array![[19., 22.], [43., 50.]].into_dyn()
        );

        // Perform backward pass
        let out_gradient = array![[1., 1.], [1., 1.]].into_dyn();

        let grad_a = matmul.backward(&out_gradient, vec![a.clone(), b.clone()], 0);
        let grad_b = matmul.backward(&out_gradient, vec![a.clone(), b.clone()], 1);

        // Check backward pass results
        assert_eq!(grad_a, array![[11., 15.], [11., 15.]].into_dyn());
        assert_eq!(grad_b, array![[4., 4.], [6., 6.]].into_dyn());

        // Test full backpropagation
        backprop(result);

        // Check gradients after backpropagation
        assert_eq!(
            *a.gradient.borrow().as_ref().unwrap(),
            array![[11., 15.], [11., 15.]].into_dyn()
        );
        assert_eq!(
            *b.gradient.borrow().as_ref().unwrap(),
            array![[4., 4.], [6., 6.]].into_dyn()
        );
    }

    #[test]
    fn test_matrix_multiplication_non_square() {
        // Test mostly written by Claude 3.5 Sonnet (expectations revised against PyTorch)
        let a = Rc::new(
            TensorBuilder::new(array![[1., 2., 3.], [4., 5., 6.]].into_dyn())
                .identifier("a".to_string())
                .requires_gradient(true)
                .build(),
        );
        let b = Rc::new(
            TensorBuilder::new(array![[7., 8.], [9., 10.], [11., 12.]].into_dyn())
                .identifier("b".to_string())
                .requires_gradient(true)
                .build(),
        );

        // Perform forward pass
        let matmul = MatrixMultiplication {};
        let result = matmul.forward(vec![a.clone(), b.clone()]);

        // Check forward pass result
        assert_eq!(
            *result.array.borrow(),
            array![[58., 64.], [139., 154.]].into_dyn()
        );

        // Test full backpropagation
        backprop(result);

        // Check gradients after backpropagation
        assert_eq!(
            *a.gradient.borrow().as_ref().unwrap(),
            array![[15., 19., 23.], [15., 19., 23.]].into_dyn()
        );
        assert_eq!(
            *b.gradient.borrow().as_ref().unwrap(),
            array![[5., 5.], [7., 7.], [9., 9.]].into_dyn()
        );
    }

    #[test]
    fn test_rectified_linear_unit() {
        // Test written by Claude 3.5 Sonnet
        let input = Rc::new(
            TensorBuilder::new(array![-2.0, -1.0, 0.0, 1.0, 2.0].into_dyn())
                .identifier("input".to_string())
                .requires_gradient(true)
                .build(),
        );

        // Perform forward pass
        let relu = RectifiedLinearUnit {};
        let result = relu.forward(vec![input.clone()]);

        // Check forward pass result
        assert_eq!(
            *result.array.borrow(),
            array![0.0, 0.0, 0.0, 1.0, 2.0].into_dyn()
        );

        // Test full backpropagation
        backprop(result);

        // Check gradients after backpropagation
        assert_eq!(
            *input.gradient.borrow().as_ref().unwrap(),
            array![0.0, 0.0, 0.0, 1.0, 1.0].into_dyn()
        );

        // Additional test with different input
        let input2 = Rc::new(
            TensorBuilder::new(array![-1.0, 0.0, 1.0, 2.0, 3.0].into_dyn())
                .identifier("input2".to_string())
                .requires_gradient(true)
                .build(),
        );

        let result2 = relu.forward(vec![input2.clone()]);
        assert_eq!(
            *result2.array.borrow(),
            array![0.0, 0.0, 1.0, 2.0, 3.0].into_dyn()
        );
    }

    #[test]
    fn test_leaky_rectified_linear_unit() {
        // Test written by Claude Sonnet 3.5
        let leak = 0.01;
        let lrelu = LeakyRectifiedLinearUnit::new(leak);

        // Test case 1: Standard input
        let input = Rc::new(
            TensorBuilder::new(array![-2.0, -1.0, 0.0, 1.0, 2.0].into_dyn())
                .identifier("input".to_string())
                .requires_gradient(true)
                .build(),
        );

        // Perform forward pass
        let result = lrelu.forward(vec![input.clone()]);

        // Check forward pass result
        assert_eq!(
            *result.array.borrow(),
            array![-0.02, -0.01, 0.0, 1.0, 2.0].into_dyn()
        );

        // Test full backpropagation
        backprop(result);

        // Check gradients after backpropagation
        assert_eq!(
            *input.gradient.borrow().as_ref().unwrap(),
            array![0.01, 0.01, 0.01, 1.0, 1.0].into_dyn()
        );

        // Test case 2: Different input
        let input2 = Rc::new(
            TensorBuilder::new(array![-1.0, 0.0, 1.0, 2.0, 3.0].into_dyn())
                .identifier("input2".to_string())
                .requires_gradient(true)
                .build(),
        );

        let result2 = lrelu.forward(vec![input2.clone()]);

        assert_eq!(
            *result2.array.borrow(),
            array![-0.01, 0.0, 1.0, 2.0, 3.0].into_dyn()
        );

        // Test case 3: Edge cases
        let input3 = Rc::new(
            TensorBuilder::new(array![-0.001, 0.001, 1e-8, -1e-8].into_dyn())
                .identifier("input3".to_string())
                .requires_gradient(true)
                .build(),
        );

        let result3 = lrelu.forward(vec![input3.clone()]);

        assert_abs_diff_eq!(
            *result3.array.borrow(),
            array![-0.00001, 0.001, 1e-8, -1e-10].into_dyn()
        );

        // Test case 4: Different leak factor
        let lrelu2 = LeakyRectifiedLinearUnit::new(0.1);
        let input4 = Rc::new(
            TensorBuilder::new(array![-1.0, 1.0].into_dyn())
                .identifier("input4".to_string())
                .requires_gradient(true)
                .build(),
        );

        let result4 = lrelu2.forward(vec![input4.clone()]);

        assert_eq!(*result4.array.borrow(), array![-0.1, 1.0].into_dyn());
    }

    #[test]
    fn test_squared_error() {
        // Test written by Claude Sonnet 3.5
        let prediction = Rc::new(
            TensorBuilder::new(array![0.7].into_dyn())
                .identifier("prediction".to_string())
                .requires_gradient(true)
                .build(),
        );
        let target = Rc::new(
            TensorBuilder::new(array![1.0].into_dyn())
                .identifier("target".to_string())
                .requires_gradient(true)
                .build(),
        );

        let squared_error = SquaredError {};
        let result = squared_error.forward(vec![prediction.clone(), target.clone()]);

        // Check forward pass result
        assert!((result.array.borrow()[0] - 0.09).abs() < 1e-6);

        // Test backward pass
        backprop(result);

        // Check gradients after backpropagation
        let prediction_gradient = prediction.gradient.borrow();
        let target_gradient = target.gradient.borrow();

        assert!((prediction_gradient.as_ref().unwrap()[0] + 0.6).abs() < 1e-6);
        assert!((target_gradient.as_ref().unwrap()[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_cross_entropy() {
        // Test drafted by Claude Sonnet 3.5, human-edited for correctness

        // Create a prediction tensor (logits)
        let prediction = Rc::new(
            TensorBuilder::new(array![2.0, 1.0, 0.1].into_dyn())
                .identifier("prediction".to_string())
                .requires_gradient(true)
                .build(),
        );

        // Create a target tensor (true probabilities)
        let target = Rc::new(
            TensorBuilder::new(array![1.0, 0.0, 0.0].into_dyn())
                .identifier("target".to_string())
                .requires_gradient(false)
                .build(),
        );

        let softmax_cross_entropy = SoftmaxCrossEntropy {};
        let result = softmax_cross_entropy.forward(vec![prediction.clone(), target.clone()]);

        assert_abs_diff_eq!(result.array.borrow()[0], 0.4170, epsilon = 0.0001);

        // Test backward pass
        backprop(result);

        // Check gradients after backpropagation
        let prediction_gradient = prediction.gradient.borrow();
        let expected_gradients = array![-0.3410, 0.2424, 0.0986]; // from PyTorch—
        // import torch
        // criterion = torch.nn.CrossEntropyLoss()
        // prediction = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
        // target = torch.tensor([1., 0., 0.], requires_grad=True)
        // loss = criterion(prediction, target)
        // loss.backward()
        //
        // # In [2]: prediction.grad
        // # Out[2]: tensor([-0.3410,  0.2424,  0.0986])

        for (&actual, &expected) in prediction_gradient
            .as_ref()
            .unwrap()
            .iter()
            .zip(expected_gradients.iter())
        {
            assert_abs_diff_eq!(actual, expected, epsilon = 0.0001)
        }
        // The target doesn't require gradients, so we don't check its gradient
    }
}
