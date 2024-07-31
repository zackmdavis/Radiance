#![allow(dead_code, unused_imports, unused_variables)]

use std::rc::Rc;

use ndarray::prelude::*;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

use super::{Tensor, TensorBuilder, Origin};
use super::operations::Operation;

pub struct TokenEmbedding {
    identifier: String,
    weights: Rc<Tensor>,
}

impl TokenEmbedding {
    pub fn new(identifier: &str, vocabulary_size: usize, embedding_dimensionality: usize) -> Self {
        let weights = Array::random(
            (vocabulary_size, embedding_dimensionality),
            StandardNormal {},
        ).into_dyn();
        Self {
            identifier: identifier.to_owned(),
            weights: Rc::new(
                TensorBuilder::new(weights)
                    .requires_gradient(true)
                    .identifier(format!("{}_weights", identifier))
                    .build(),
            ),
        }
    }

    pub fn dimensionality(&self) -> usize {
        self.weights.array.borrow().shape()[1]
    }
}

// Should TokenEmbedding be an Operation??—actually ... probably not!!
// Claude suggests `Lookup` as the internal operation for the embedding to use.

impl Operation for TokenEmbedding {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        let sequence = inputs[0].array.borrow();
        let weights = self.weights.array.borrow().clone().into_dimensionality::<Ix2>()
            .expect("two-dimensional");
        let mut representation = Array2::zeros((0, self.dimensionality()));
        for token in sequence.iter() {
            // token IDs are morally usize integers, but for now, my Tensors only support f32
            representation.push_row(weights.row(*token as usize)).expect("row should fit");
        }
        // If we don't separate operations from components, then
        // `origin.operation` would have to contain a box of the entire
        // embedding component, which seems wrong? Instead, `Lookup` could take
        // the token embedding weight matrix as a parameter.
        //
        // let origin = Origin {
        //     operation: Box::new(TokenEmbedding {}),
        //     parents: inputs.clone()
        // };
        Rc::new(TensorBuilder::new(representation.into_dyn())
                // .origin(origin)
                .build())
    }

    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        args: Vec<Rc<Tensor>>,
        arg_index: usize,
    ) -> ArrayD<f32> {
        array![0.0f32].into_dyn()
    }
}
