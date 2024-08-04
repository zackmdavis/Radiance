#![allow(dead_code)]

use std::rc::Rc;

use crate::core::attention::AttentionLayer;
use crate::core::embedding::{sequence_positional_encoding, TokenEmbedding};
use crate::core::Tensor;

use crate::core::operations::{Addition, Operation};

pub struct SmallLanguageModel {
    token_embedding: TokenEmbedding,
    attention_layers: Vec<AttentionLayer>,
}

impl SmallLanguageModel {
    pub fn parameters(&self) -> Vec<Rc<Tensor>> {
        let mut parameters = Vec::new();
        parameters.extend(self.token_embedding.parameters());
        for attention_layer in &self.attention_layers {
            parameters.extend(attention_layer.parameters());
        }
        parameters
    }

    pub fn forward(&self, token_sequence: Rc<Tensor>) -> Rc<Tensor> {
        let sequence_length = token_sequence.borrow_array().shape()[0];
        let mut x = self.token_embedding.embed(token_sequence);
        x = Addition {}.forward(vec![
            x,
            sequence_positional_encoding(sequence_length, self.token_embedding.dimensionality()),
        ]);
        for attention_layer in &self.attention_layers {
            x = attention_layer.forward(x);
        }
        let logits = self.token_embedding.unembed(x);
        logits
    }
}
