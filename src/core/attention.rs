#![allow(dead_code, unused_imports, unused_variables)]

use std::rc::Rc;

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

use super::{Origin, Tensor, TensorBuilder};

pub struct AttentionHead {
    identifier: String,
    query_weights: Rc<Tensor>,
    // TODO—add biases
    // {key,query,value,output}_biases: Rc<Tensor>,
    key_weights: Rc<Tensor>,
    value_weights: Rc<Tensor>,
    output_weights: Rc<Tensor>,
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
            output_weights,
        }
    }

    pub fn forward(&self, input: Rc<Tensor>) /* -> Rc<Tensor> */ {
        // Input is shape (sequence_length, embedding_dimensionality).
        // Queries, keys, and values are each shape (embedding_dimensionality, attention_dimensionality).
        // So x⃗W_q + b_q (respectively _k, _v) are shape (sequence_length, attention_dimensionality)
    }
}
