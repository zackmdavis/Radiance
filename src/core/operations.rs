use std::rc::Rc;

use ndarray;
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
            .into_shape_with_order(self.new_shape.clone())
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
            .into_shape_with_order(args[0].array.borrow().shape())
            .expect("input shape should match")
    }
}

pub(super) struct Transpose {}

impl Operation for Transpose {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        assert!(inputs.len() == 1, "unary operation expected");
        let array = inputs[0].array.borrow().clone();
        let origin = Origin {
            operation: Box::new(Transpose {}),
            parents: inputs.clone(),
        };
        Rc::new(
            TensorBuilder::new(array.t().to_owned())
                .origin(origin)
                .build(),
        )
    }

    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        _args: Vec<Rc<Tensor>>,
        _arg_index: usize,
    ) -> ArrayD<f32> {
        out_gradient.t().to_owned()
    }
}

pub(super) struct Concatenate {}

impl Operation for Concatenate {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        let arrays = inputs
            .iter()
            .map(|t| t.array.borrow().clone())
            .collect::<Vec<_>>();
        let array_views = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
        let catted =
            ndarray::concatenate(Axis(1), array_views.as_slice()).expect("shapes should cat");
        let origin = Origin {
            operation: Box::new(Concatenate {}),
            parents: inputs.clone(),
        };
        Rc::new(TensorBuilder::new(catted.into_dyn()).origin(origin).build())
    }

    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        args: Vec<Rc<Tensor>>,
        arg_index: usize,
    ) -> ArrayD<f32> {
        let widths = args
            .iter()
            .map(|t| t.array.borrow().shape()[1])
            .collect::<Vec<_>>();
        let columns_before: usize = widths[..arg_index].iter().sum();
        out_gradient
            .slice(s![.., columns_before..columns_before + widths[arg_index]])
            .to_owned()
            .into_dyn()
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

fn softmax(x: Array1<f32>) -> Array1<f32> {
    // TODO: shift the maximum element to zero for numerical stability?
    // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    // #computing-softmax-and-numerical-stability
    let exp_x = x.iter().map(|x_i| x_i.exp()).collect::<Array1<f32>>();
    let scale: f32 = exp_x.iter().sum();
    let softmaxed = exp_x.iter().map(|x_i| x_i / scale).collect::<Array1<f32>>();
    softmaxed
}

fn δ(i: usize, j: usize) -> f32 {
    if i == j {
        1.
    } else {
        0.
    }
}

pub(super) struct Softmax {}

impl Operation for Softmax {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        assert!(inputs.len() == 1, "unary operation expected");
        let softmaxed = softmax(
            inputs[0]
                .array
                .borrow()
                .clone()
                .into_dimensionality::<Ix1>()
                .expect("one-dimensional"),
        );
        let origin = Origin {
            operation: Box::new(Softmax {}),
            parents: inputs.clone(),
        };
        Rc::new(
            TensorBuilder::new(softmaxed.into_dyn())
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
        let softmaxed = softmax(
            args[0]
                .array
                .borrow()
                .clone()
                .into_dimensionality::<Ix1>()
                .expect("one-dimensional"),
        );
        let out_gradient = out_gradient
            .clone()
            .into_dimensionality::<Ix1>()
            .expect("one-dimensional");
        let n = softmaxed.len();
        // The entries of the derivative matrix dS_i/dx_j are
        //
        // softmax(x)_i ·(δ_ij − softmax(x)_j)
        //
        // which tells us how the softmax outputs varies with its inputs.
        let d = Array2::from_shape_fn((n, n), |(i, j)| softmaxed[i] * (δ(i, j) - softmaxed[j]));
        // The out-gradient dL/dS_i tells us how loss varies with the softmax outputs.
        // So the product dS_i/dx_j · dL/dS_i = dL/dx_j tells
        // us how loss varies with the inputs.
        d.dot(&out_gradient).into_dyn()
    }
}

pub(super) struct SoftmaxRows {}

impl Operation for SoftmaxRows {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        assert!(inputs.len() == 1, "unary operation expected");
        let x = inputs[0]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");
        let mut softmaxed = Array2::zeros((0, x.shape()[1]));
        for row in x.rows() {
            let softmaxed_row = softmax(row.to_owned());
            softmaxed
                .push_row((&softmaxed_row).into())
                .expect("row should fit");
        }
        let origin = Origin {
            operation: Box::new(SoftmaxRows {}),
            parents: inputs.clone(),
        };
        Rc::new(
            TensorBuilder::new(softmaxed.into_dyn())
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
        let x = args[0]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");
        // XXX: redoing the forward fn inside of the backward fn has dubious
        // performance implications
        let mut softmaxed = Array2::zeros((0, x.shape()[1]));
        for row in x.rows() {
            let softmaxed_row = softmax(row.to_owned());
            softmaxed
                .push_row((&softmaxed_row).into())
                .expect("row should fit");
        }

        let out_gradient = out_gradient
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");

        let n = softmaxed.shape()[1];

        // As with `Softmax`, forming the derivative matrix dS_i/dx_j =
        // softmax(x)_i ·(δ_ij − softmax(x)_j) and multiplying it with the
        // out-gradient ... but row by row.

        let mut gradient = Array2::zeros((0, n));
        for i in 0..softmaxed.shape()[0] {
            // You'd expect this to be `for (softmaxed_row, out_gradient_row) in
            // softmaxed.rows().zip(out_gradient.rows())`, but the iterator trait
            // bound is missing, so, fine, we'll index
            let softmaxed_row = softmaxed.row(i);
            let out_gradient_row = out_gradient.row(i);

            let d = Array2::from_shape_fn((n, n), |(i, j)| {
                softmaxed_row[i] * (δ(i, j) - softmaxed_row[j])
            });
            let gradient_row = d.dot(&out_gradient_row);
            gradient
                .push_row((&gradient_row).into())
                .expect("row should fit");
        }

        gradient.into_dyn()
    }
}

pub(super) struct SoftmaxCrossEntropy {}

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
        out_gradient
            * target
                .iter()
                .zip(softmaxed)
                .map(|(t, s)| s - t)
                .collect::<Array1<f32>>()
                .into_dyn()
    }
}

pub(super) struct Mask {}

impl Operation for Mask {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        assert!(inputs.len() == 2, "binary operation expected");
        let array = inputs[0]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");
        let mask = inputs[1]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");
        let masked = Array2::from_shape_fn(array.raw_dim(), |(i, j)| {
            if mask[[i, j]] == 0. {
                f32::NEG_INFINITY
            } else {
                array[[i, j]]
            }
        });
        let origin = Origin {
            operation: Box::new(Mask {}),
            parents: inputs.clone(),
        };
        Rc::new(TensorBuilder::new(masked.into_dyn()).origin(origin).build())
    }

    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        args: Vec<Rc<Tensor>>,
        _arg_index: usize,
    ) -> ArrayD<f32> {
        // We can ignore `_arg_index` because we only care about the gradient
        // of the input being masked, not the mask itself?
        //
        // ... that actually seems kind of sketchy (backprop is still going to
        // call it with _arg_index=1, the answer will be presumably wrong, and
        // we're just betting that nothing important depends on that tensor
        let array = args[0]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");
        let mask = args[1]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");
        Array2::from_shape_fn(array.raw_dim(), |(i, j)| {
            if mask[[i, j]] == 0. {
                0.
            } else {
                out_gradient[[i, j]]
            }
        })
        .into_dyn()
    }
}

// `Lookup` Operation in embedding.rs

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
                .requires_gradient(true)
                .build(),
        );
        let b = Rc::new(
            TensorBuilder::new(array![[5., 6.], [7., 8.]].into_dyn())
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
                .requires_gradient(true)
                .build(),
        );
        let b = Rc::new(
            TensorBuilder::new(array![[7., 8.], [9., 10.], [11., 12.]].into_dyn())
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
                .requires_gradient(true)
                .build(),
        );

        let result4 = lrelu2.forward(vec![input4.clone()]);

        assert_eq!(*result4.array.borrow(), array![-0.1, 1.0].into_dyn());
    }

    #[test]
    fn test_transpose() {
        // Test written by Claude Sonnet 3.5

        // Create a 2x3 input matrix
        let input = Rc::new(
            TensorBuilder::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn())
                .requires_gradient(true)
                .build(),
        );

        let transpose = Transpose {};
        let result = transpose.forward(vec![input.clone()]);

        // Check forward pass result
        let expected_result = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
        assert_eq!(result.array.borrow().shape(), &[3, 2]);
        for (actual, expected) in result.array.borrow().iter().zip(expected_result.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }

        // Create a non-uniform gradient for backpropagation
        // This allows us to test if the gradient elements are correctly transposed,
        // not just if the shape is correct
        let out_gradient = Rc::new(
            TensorBuilder::new(array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]].into_dyn())
                .requires_gradient(false)
                .build(),
        );

        // Multiply result by out_gradient to create a scalar result
        let scalar_result = Multiplication {}.forward(vec![result, out_gradient]);

        // Test backward pass
        backprop(scalar_result);

        // Check gradients after backpropagation
        let input_gradient = input.gradient.borrow();
        let expected_gradient = array![[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]];

        assert_eq!(input_gradient.as_ref().unwrap().shape(), &[2, 3]);
        for (actual, expected) in input_gradient
            .as_ref()
            .unwrap()
            .iter()
            .zip(expected_gradient.iter())
        {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_concatenate() {
        // artisanally human-written, real vintage 2019 stuff here
        let a = array![[1., 2.], [3., 4.]];
        let b = array![[4., 5.], [6., 7.]];
        let ta = Rc::new(
            TensorBuilder::new(a.into_dyn())
                .requires_gradient(true)
                .build(),
        );
        let tb = Rc::new(
            TensorBuilder::new(b.into_dyn())
                .requires_gradient(true)
                .build(),
        );
        let result = Concatenate {}.forward(vec![ta, tb]);
        let result_array = result.array.borrow();
        assert_eq!(
            *result_array,
            array![[1.0, 2.0, 4.0, 5.0], [3.0, 4.0, 6.0, 7.0]].into_dyn()
        );
        let out_gradient = array![[1., 2., 3., 4.], [5., 6., 7., 8.]].into_dyn();
        let back_0 = Concatenate {}.backward(
            &out_gradient,
            result.origin.as_ref().unwrap().parents.clone(),
            0,
        );
        let back_1 = Concatenate {}.backward(
            &out_gradient,
            result.origin.as_ref().unwrap().parents.clone(),
            1,
        );
        assert_eq!(back_0, array![[1., 2.], [5., 6.]].into_dyn());
        assert_eq!(back_1, array![[3., 4.], [7., 8.]].into_dyn());
    }

    #[test]
    fn test_squared_error() {
        // Test written by Claude Sonnet 3.5
        let prediction = Rc::new(
            TensorBuilder::new(array![0.7].into_dyn())
                .identifier("prediction")
                .requires_gradient(true)
                .build(),
        );
        let target = Rc::new(
            TensorBuilder::new(array![1.0].into_dyn())
                .identifier("target")
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
                .identifier("prediction")
                .requires_gradient(true)
                .build(),
        );

        // Create a target tensor (true probabilities)
        let target = Rc::new(
            TensorBuilder::new(array![1.0, 0.0, 0.0].into_dyn())
                .identifier("target")
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
        // from PyTorch—
        //
        // import torch
        // criterion = torch.nn.CrossEntropyLoss()
        // prediction = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
        // target = torch.tensor([1., 0., 0.], requires_grad=True)
        // loss = criterion(prediction, target)
        // loss.backward()
        //
        // # In [2]: prediction.grad
        // # Out[2]: tensor([-0.3410,  0.2424,  0.0986])
        let expected_gradients = array![-0.3410, 0.2424, 0.0986];

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

    #[test]
    fn test_softmax() {
        // Test written by Claude Sonnet 3.5, with human edits for correctness
        // and clarity

        // Create an input tensor
        let input = Rc::new(
            TensorBuilder::new(array![2.0, 1.0, 0.1].into_dyn())
                .identifier("input")
                .requires_gradient(true)
                .build(),
        );

        let softmax = Softmax {};

        // Test forward pass
        let result = softmax.forward(vec![input.clone()]);
        let expected_output = array![0.6590, 0.2424, 0.0986];
        for (&actual, &expected) in result.array.borrow().iter().zip(expected_output.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 0.0001);
        }

        // Test backward pass
        // Send an extreme gradient backwards to get a stronger signal
        let out_gradient = array![10000.0, 1000.0, 1.0].into_dyn();
        let input_gradient = softmax.backward(&out_gradient, vec![input.clone()], 0);

        // from PyTorch—
        //
        // import torch
        // activation = torch.nn.Softmax(dim=0)
        // x = torch.tensor([2., 1., 0.1], requires_grad=True)
        // output = activation(x)
        // output.backward(gradient=torch.tensor([10000., 1000., 1.]))

        // In [2]: x.grad
        // Out[2]: tensor([ 2087.3577, -1414.0009,  -673.3572])
        let expected_gradient = array![2087.3577, -1414.0009, -673.3572];

        for (&actual, &expected) in input_gradient.iter().zip(expected_gradient.iter()) {
            // TODO: can we tolerate a lower ɛ after making the numerical
            // stability improvement to softmax?
            assert_abs_diff_eq!(actual, expected, epsilon = 0.001);
        }
    }

    #[test]
    fn test_softmax_rows() {
        // Test written by Claude Sonnet 3.5, with human feedback and edits

        // Create an input tensor with multiple rows
        let input = Rc::new(
            TensorBuilder::new(array![[2.0, 1.0, 0.1], [1.0, 2.0, 3.0]].into_dyn())
                .identifier("input")
                .requires_gradient(true)
                .build(),
        );

        let softmax_rows = SoftmaxRows {};

        // Test forward pass
        let result = softmax_rows.forward(vec![input.clone()]);

        let expected_output = array![[0.6590, 0.2424, 0.0986], [0.0900, 0.2447, 0.6652]];

        assert_eq!(result.array.borrow().shape(), &[2, 3]);
        for (actual_row, expected_row) in result
            .array
            .borrow()
            .rows()
            .into_iter()
            .zip(expected_output.rows())
        {
            for (&actual, &expected) in actual_row.iter().zip(expected_row.iter()) {
                assert_abs_diff_eq!(actual, expected, epsilon = 0.0001);
            }
        }

        // Test backward pass
        // Using extreme values to get a stronger signal
        let out_gradient = array![[10000.0, 1000.0, 1.0], [1.0, 100.0, 10000.0]].into_dyn();

        // Directly call backward
        let input_gradient = softmax_rows.backward(&out_gradient, vec![input.clone()], 0);

        // from PyTorch—
        //
        // import torch
        // activation = torch.nn.Softmax(dim=1)
        // x = torch.tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 3.0]], requires_grad=True)
        // output = activation(x)
        // output.backward(gradient=torch.tensor([[10000.0, 1000.0, 1.0], [1.0, 100.0, 10000.0]]))
        // x.grad
        let expected_gradient = array![
            [2087.3577, -1414.0009, -673.3572],
            [-601.0416, -1609.5725, 2210.6138]
        ];

        assert_eq!(input_gradient.shape(), &[2, 3]);
        for (actual_row, expected_row) in input_gradient
            .rows()
            .into_iter()
            .zip(expected_gradient.rows())
        {
            for (&actual, &expected) in actual_row.iter().zip(expected_row.iter()) {
                assert_abs_diff_eq!(actual, expected, epsilon = 0.001);
            }
        }
    }
    #[test]
    fn test_mask() {
        // Test written by Claude Sonnet 3.5

        let input = Rc::new(
            TensorBuilder::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn())
                .identifier("input")
                .requires_gradient(true)
                .build(),
        );

        // Create a mask tensor
        let mask = Rc::new(
            TensorBuilder::new(array![[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]].into_dyn())
                .identifier("mask")
                .requires_gradient(false)
                .build(),
        );

        let mask_op = Mask {};

        // Test forward pass
        let result = mask_op.forward(vec![input.clone(), mask.clone()]);
        let expected_output = array![[1.0, 2.0, f32::NEG_INFINITY], [4.0, f32::NEG_INFINITY, 6.0]];
        for (&actual, &expected) in result.array.borrow().iter().zip(expected_output.iter()) {
            if expected.is_finite() {
                assert_abs_diff_eq!(actual, expected, epsilon = 0.0001);
            } else {
                assert!(!actual.is_finite());
            }
        }

        // Test backward pass
        let out_gradient = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]].into_dyn();
        let input_gradient = mask_op.backward(&out_gradient, vec![input.clone(), mask.clone()], 0);
        let expected_gradient = array![[0.1, 0.2, 0.0], [0.4, 0.0, 0.6]];
        for (&actual, &expected) in input_gradient.iter().zip(expected_gradient.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 0.0001);
        }
    }
}
