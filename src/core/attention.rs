#![allow(dead_code, unused_imports, unused_variables)]

use std::rc::Rc;

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

use super::operations::{
    Mask, MatrixMultiplication, Multiplication, Operation, SoftmaxRows, Transpose,
};
use super::{Origin, Tensor, TensorBuilder};

pub struct AttentionHead {
    identifier: String,
    query_weights: Rc<Tensor>,
    // TODO—add biases
    // {key,query,value,output}_biases: Rc<Tensor>,
    key_weights: Rc<Tensor>,
    value_weights: Rc<Tensor>,
}

impl AttentionHead {
    pub fn new(
        identifier: &str,
        embedding_dimensionality: usize,
        attention_dimensionality: usize,
    ) -> Self {
        let query_weights = Rc::new(
            TensorBuilder::new(
                Array::random(
                    (embedding_dimensionality, attention_dimensionality),
                    Normal::new(0., 0.02).unwrap(),
                )
                .into_dyn(),
            )
            .requires_gradient(true)
            .build(),
        );
        // TODO— add biases
        //
        // let {key,query,value,output}_biases = Rc::new(
        //     TensorBuilder::new(Array::zeros((attention_dimensionality,)).into_dyn())
        //         .requires_gradient(true)
        //         .build(),
        // );
        let key_weights = Rc::new(
            TensorBuilder::new(
                Array::random(
                    (embedding_dimensionality, attention_dimensionality),
                    Normal::new(0., 0.02).unwrap(),
                )
                .into_dyn(),
            )
            .requires_gradient(true)
            .build(),
        );
        let value_weights = Rc::new(
            TensorBuilder::new(
                Array::random(
                    (embedding_dimensionality, attention_dimensionality),
                    Normal::new(0., 0.02).unwrap(),
                )
                .into_dyn(),
            )
            .requires_gradient(true)
            .build(),
        );
        let output_weights = Rc::new(
            TensorBuilder::new(
                Array::random(
                    (attention_dimensionality, embedding_dimensionality),
                    Normal::new(0., 0.02).unwrap(),
                )
                .into_dyn(),
            )
            .requires_gradient(true)
            .build(),
        );
        Self {
            identifier: identifier.to_owned(),
            query_weights,
            key_weights,
            value_weights,
        }
    }

    pub fn embedding_dimensionality(&self) -> usize {
        self.key_weights // without loss of generality
            .array
            .borrow()
            .shape()[0]
    }

    pub fn attention_dimensionality(&self) -> usize {
        self.key_weights.array.borrow().shape()[1]
    }

    pub fn forward(&self, x: Rc<Tensor>) -> Rc<Tensor> {
        // Input is shape (sequence_length, embedding_dimensionality).
        let n = x.array.borrow().shape()[0];
        assert!(n >= 1, "need at least one token");

        // Queries, keys, and values are each shape (embedding_dimensionality, attention_dimensionality).
        // So XW_q (respectively _k, _v) are shape (sequence_length, attention_dimensionality)
        let q = MatrixMultiplication {}.forward(vec![x.clone(), self.query_weights.clone()]);
        let k = MatrixMultiplication {}.forward(vec![x.clone(), self.key_weights.clone()]);
        let v = MatrixMultiplication {}.forward(vec![x.clone(), self.value_weights.clone()]);

        let k_t = Transpose {}.forward(vec![k]);
        let qk_t = MatrixMultiplication {}.forward(vec![q, k_t]);
        let scale_factor = 1. / (self.attention_dimensionality() as f32).sqrt();
        let scale =
            Rc::new(TensorBuilder::new((scale_factor * Array::ones((n, n))).into_dyn()).build());
        let scaled_qk_t = Multiplication {}.forward(vec![scale, qk_t]);

        // `tril` chokes on 1×1: https://github.com/rust-ndarray/ndarray/issues/1415
        let raw_mask = if n == 1 {
            Array::ones((n, n))
        } else {
            Array::ones((n, n)).tril(0)
        };

        let mask = Rc::new(TensorBuilder::new(raw_mask.into_dyn()).build());
        let masked = Mask {}.forward(vec![scaled_qk_t, mask]);
        let softmaxed = SoftmaxRows {}.forward(vec![masked]);
        let h = MatrixMultiplication {}.forward(vec![softmaxed, v]);
        h
    }
}

pub struct AttentionMultiHead {
    heads: Vec<AttentionHead>,
    output_weights: Rc<Tensor>,
    // TODO: MLP
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensionalities() {
        let head = AttentionHead::new("my_first_attention_head", 64, 16);
        assert_eq!(head.embedding_dimensionality(), 64);
        assert_eq!(head.attention_dimensionality(), 16);
    }

    #[test]
    fn test_input() {
        let head = AttentionHead::new("test_attention_head", 8, 4);
        let input = Rc::new(
            TensorBuilder::new(
                array![
                    [1., 2., 3., 4., 5., 6., 7., 8.],
                    [9., 8., 7., 6., 5., 4., 3., 2.]
                ]
                .into_dyn(),
            )
            .build(),
        );
        head.forward(input);
    }

    #[test]
    fn test_single_token_input() {
        let head = AttentionHead::new("single_token_test_attention_head", 8, 4);
        let input = Rc::new(
            TensorBuilder::new(array![[1., 2., 3., 4., 5., 6., 7., 8.]].into_dyn()).build(),
        );
        head.forward(input);
    }

    #[test]
    fn test_tiny_input() {
        let head = AttentionHead::new("single_token_test_attention_head", 8, 4);
        let input = Rc::new(
            TensorBuilder::new(array![[1e-10, 2e-9, 3e-3, 0., 0., 0., 0., 0.]].into_dyn()).build(),
        );
        head.forward(input);
    }

    #[test]
    fn test_causal_masking() {
        let head = AttentionHead::new("test_attention_head", 8, 4);
        let input_1 = Rc::new(
            TensorBuilder::new(
                array![
                    [1., 2., 3., 4., 5., 6., 7., 8.],
                    [9., 8., 7., 6., 5., 4., 3., 2.]
                ]
                .into_dyn(),
            )
            .build(),
        );
        let input_2 = Rc::new(
            TensorBuilder::new(
                array![
                    [1., 2., 3., 4., 5., 6., 7., 8.],
                    // second token is different
                    [5., 0., 5., 0., 5., 0., 5., 0.]
                ]
                .into_dyn(),
            )
            .build(),
        );
        let raw_output_1 = head.forward(input_1);
        let raw_output_2 = head.forward(input_2);

        let output_1 = raw_output_1.array.borrow();
        let output_2 = raw_output_2.array.borrow();

        // The causal mask prevents the first token's output from depending on
        // subsequent tokens. (QK^T will be different, but all the particular
        // entries that make it different get masked out.)

        let first_token_output_1 = output_1.slice(s![0, ..]);
        let first_token_output_2 = output_2.slice(s![0, ..]);
        assert_eq!(first_token_output_1, first_token_output_2);

        let second_token_output_1 = output_1.slice(s![1, ..]);
        let second_token_output_2 = output_2.slice(s![1, ..]);
        assert_ne!(second_token_output_1, second_token_output_2);
    }
}
