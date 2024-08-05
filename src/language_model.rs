#![allow(dead_code)]

use std::fs;
use std::rc::Rc;

use ndarray::prelude::*;

use crate::core::attention::AttentionLayer;
use crate::core::embedding::{sequence_positional_encoding, TokenEmbedding, TokenVocabulary};
use crate::core::{Tensor, TensorBuilder};

use crate::core::operations::{Addition, Operation};

use crate::core::optimization::StochasticGradientDescentOptimizer;

pub struct SmallLanguageModel {
    token_embedding: TokenEmbedding,
    attention_layers: Vec<AttentionLayer>,
}

impl SmallLanguageModel {
    pub fn new(
        identifier: &str,
        vocabulary_size: usize,
        embedding_dimensionality: usize,
        attention_dimensionality: usize,
        head_count: usize,
        layer_count: usize,
    ) -> Self {
        let token_embedding = TokenEmbedding::new(
            &format!("{}_token_embedding", identifier),
            vocabulary_size,
            embedding_dimensionality,
        );
        let mut attention_layers = Vec::new();
        for layer_no in 0..layer_count {
            attention_layers.push(AttentionLayer::new(
                &format!("{}_layer_{}", identifier, layer_no),
                head_count,
                embedding_dimensionality,
                attention_dimensionality,
            ));
        }
        Self {
            token_embedding,
            attention_layers,
        }
    }

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

pub fn train_slm() -> SmallLanguageModel {
    let network = SmallLanguageModel::new("my_language_model", 128, 64, 16, 4, 2);
    let mut _optimizer = StochasticGradientDescentOptimizer::new(network.parameters(), 0.01);
    let token_vocabulary = TokenVocabulary::default();

    let training_megastring = fs::read_to_string("training_data.txt").expect("file slurped");
    let training_tokenstream = token_vocabulary.tokenize(training_megastring);

    for context_window in training_tokenstream.windows(64) {
        // We shift the input sequence by one (padding the beginning with a
        // start-of-sequence token, so that each position can predict its own
        // next token.
        let mut input_vec = vec![0.0]; // start-of-sequence token
        input_vec.extend(&context_window[..63]);
        let target_vec = context_window.to_vec();
        println!("input: {:?} {}", input_vec, input_vec.len());
        println!("target: {:?} {}", target_vec, target_vec.len());
        let _input = Rc::new(
            TensorBuilder::new(
                Array2::from_shape_vec((1, 64), input_vec)
                    .expect("array should build")
                    .into_dyn(),
            )
            .build(),
        );
        let _target = Rc::new(
            TensorBuilder::new(
                Array2::from_shape_vec((1, 64), target_vec)
                    .expect("array should build")
                    .into_dyn(),
            )
            .build(),
        );

        // ... and my `SoftmaxCrossEntropy` needs to be rewritten to operate on
        // a sequence of distributions (one for each position) rather than a
        // single distribution

        // TODO continue ...
    }

    network
}
