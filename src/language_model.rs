#![allow(dead_code)]

use std::fs;
use std::rc::Rc;

use ndarray::prelude::*;

use crate::core::attention::AttentionLayer;
use crate::core::embedding::{sequence_positional_encoding, TokenEmbedding, TokenVocabulary};
use crate::core::{backprop, Tensor, TensorBuilder};

use crate::core::operations::{Addition, Operation, SoftmaxCrossEntropy};

use crate::core::optimization::StochasticGradientDescentOptimizer;

pub struct SmallLanguageModelConfiguration {
    vocabulary_size: usize,
    context_window_size: usize,
    embedding_dimensionality: usize,
    attention_dimensionality: usize,
    head_count: usize,
    layer_count: usize,
}

impl Default for SmallLanguageModelConfiguration {
    fn default() -> Self {
        Self {
            vocabulary_size: TokenVocabulary::default().size(),
            context_window_size: 100,
            embedding_dimensionality: 64,
            attention_dimensionality: 16,
            head_count: 4,
            layer_count: 2,
        }
    }
}

pub struct SmallLanguageModel {
    configuration: SmallLanguageModelConfiguration,
    token_embedding: TokenEmbedding,
    attention_layers: Vec<AttentionLayer>,
}

impl SmallLanguageModel {
    pub fn new(identifier: &str, configuration: SmallLanguageModelConfiguration) -> Self {
        let token_embedding = TokenEmbedding::new(
            &format!("{}_token_embedding", identifier),
            configuration.vocabulary_size,
            configuration.embedding_dimensionality,
        );
        let mut attention_layers = Vec::new();
        for layer_no in 0..configuration.layer_count {
            attention_layers.push(AttentionLayer::new(
                &format!("{}_layer_{}", identifier, layer_no),
                configuration.head_count,
                configuration.embedding_dimensionality,
                configuration.attention_dimensionality,
            ));
        }
        Self {
            configuration,
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

    pub fn forward(&self, input: Rc<Tensor>) -> Rc<Tensor> {
        let sequence_length = input.borrow_array().shape()[0];
        let mut x = self.token_embedding.embed(input);
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
    let network = SmallLanguageModel::new(
        "my_language_model",
        SmallLanguageModelConfiguration::default(),
    );
    let mut optimizer = StochasticGradientDescentOptimizer::new(network.parameters(), 0.01);
    let token_vocabulary = TokenVocabulary::default();

    let training_megastring = fs::read_to_string("training_data.txt").expect("file slurped");
    let training_tokenstream = token_vocabulary.tokenize(training_megastring);

    for context_window in training_tokenstream.windows(network.configuration.context_window_size) {
        // We shift the input sequence by one (padding the beginning with a
        // start-of-sequence token, so that each position can predict its own
        // next token.
        let mut input_vec = vec![0.0]; // start-of-sequence token
        input_vec.extend(&context_window[..network.configuration.context_window_size - 1]);
        let target_vec = context_window.to_vec();

        // Input is a one-dimensional array of token IDs.
        let input = Rc::new(
            TensorBuilder::new(
                Array1::from_shape_vec((network.configuration.context_window_size,), input_vec)
                    .expect("array should build")
                    .into_dyn(),
            )
            .build(),
        );

        let logits = network.forward(input);

        // Targets, like, logits, is a (context_window, vocabulary_size) matrix.
        let targets = Rc::new(
            TensorBuilder::new(
                Array2::from_shape_fn(
                    (
                        network.configuration.context_window_size,
                        token_vocabulary.size(),
                    ),
                    |(i, j)| {
                        if target_vec[i] == j as f32 {
                            1.
                        } else {
                            0.
                        }
                    },
                )
                .into_dyn(),
            )
            .build(),
        );
        let loss = SoftmaxCrossEntropy {}.forward(vec![logits, targets]);
        backprop(loss);
        optimizer.step();
        optimizer.unset_gradients();
    }
    network
}
