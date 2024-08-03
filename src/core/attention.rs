#![allow(dead_code, unused_imports, unused_variables)]

use std::rc::Rc;

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

use super::operations::{Mask, MatrixMultiplication, Multiplication, Operation, Softmax};
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

        // Queries, keys, and values are each shape (embedding_dimensionality, attention_dimensionality).
        // So XW_q (respectively _k, _v) are shape (sequence_length, attention_dimensionality)
        let q = MatrixMultiplication {}.forward(vec![x.clone(), self.query_weights.clone()]);
        let k = MatrixMultiplication {}.forward(vec![x.clone(), self.key_weights.clone()]);
        let v = MatrixMultiplication {}.forward(vec![x.clone(), self.value_weights.clone()]);

        let k_t = /* Transpose.forward(vec![k]); */ k; // TODO: need to write `Transpose` operation
        let qk_t = MatrixMultiplication {}.forward(vec![q, k_t]);
        let scale_factor = 1. / (self.attention_dimensionality() as f32).sqrt();
        let scale =
            Rc::new(TensorBuilder::new((scale_factor * Array::ones((n, n))).into_dyn()).build());
        let scaled_qk_t = Multiplication {}.forward(vec![scale, qk_t]);
        // TODO: need a mask
        // https://github.com/rust-ndarray/ndarray/pull/1386 added `triu`, but
        // there hasn't been an ndarray release in two years
        let mask = Rc::new(TensorBuilder::new(Array::ones((n, n)).into_dyn()).build()); // .tril(0);
        let masked = Mask {}.forward(vec![scaled_qk_t, mask]);
        // TODO: this probably needs to be SoftmaxRows?
        let softmaxed = Softmax {}.forward(vec![masked]);
        let h = MatrixMultiplication {}.forward(vec![softmaxed, v]);
        h
    }
}

pub struct AttentionMultiHead {
    heads: Vec<AttentionHead>,
    output_weights: Rc<Tensor>,
    // TODO: MLP
}
