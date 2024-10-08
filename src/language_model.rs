use std::fs;
use std::rc::Rc;
use std::time;

use chrono;
use ndarray::prelude::*;
use rand::prelude::*;

use rand_distr::weighted_alias::WeightedAliasIndex;

use crate::core::attention::AttentionLayer;
use crate::core::embedding::{sequence_positional_encoding, TokenEmbedding, TokenVocabulary};
use crate::core::operations::{softmax, Addition, Operation, SoftmaxCrossEntropy};
use crate::core::optimization::{AdaptiveMomentEstimationOptimizer, Optimizer};
use crate::core::serialization::serialize;
use crate::core::{backprop, Parameterized, Tensor, TensorBuilder};

pub struct SmallLanguageModelConfiguration {
    pub token_vocabulary: TokenVocabulary,
    pub context_window_size: usize,
    pub embedding_dimensionality: usize,
    pub head_count: usize,
    pub layer_count: usize,
}

impl Default for SmallLanguageModelConfiguration {
    fn default() -> Self {
        Self {
            token_vocabulary: TokenVocabulary::default(),
            context_window_size: 256,
            embedding_dimensionality: 128,
            head_count: 4,
            layer_count: 4,
        }
    }
}

pub struct SmallLanguageModel {
    identifier: String,
    configuration: SmallLanguageModelConfiguration,
    token_embedding: TokenEmbedding,
    attention_layers: Vec<AttentionLayer>,
}

impl Parameterized for SmallLanguageModel {
    fn identifier(&self) -> &str {
        &self.identifier
    }

    fn parameters(&self) -> Vec<Rc<Tensor>> {
        let mut parameters = Vec::new();
        parameters.extend(self.token_embedding.parameters());
        for attention_layer in &self.attention_layers {
            parameters.extend(attention_layer.parameters());
        }
        parameters
    }
}

impl SmallLanguageModel {
    pub fn new(identifier: &str, configuration: SmallLanguageModelConfiguration) -> Self {
        let token_embedding = TokenEmbedding::new(
            &format!("{}_token_embedding", identifier),
            configuration.token_vocabulary.size(),
            configuration.embedding_dimensionality,
        );
        let mut attention_layers = Vec::new();
        for layer_no in 0..configuration.layer_count {
            attention_layers.push(AttentionLayer::new(
                &format!("{}_layer_{}", identifier, layer_no),
                configuration.head_count,
                configuration.embedding_dimensionality,
                configuration.embedding_dimensionality / configuration.head_count,
            ));
        }
        Self {
            identifier: identifier.to_owned(),
            configuration,
            token_embedding,
            attention_layers,
        }
    }

    pub fn configuration(&self) -> &SmallLanguageModelConfiguration {
        &self.configuration
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

pub fn sample_next_token(token_vocabulary: &TokenVocabulary, logits: Rc<Tensor>) -> char {
    let n = logits.borrow_array().shape()[0];
    let logit_matrix = logits
        .borrow_array()
        .clone()
        .into_dimensionality::<Ix2>()
        .expect("two-dimensional");
    let next_token_logits = logit_matrix.row(n - 1);
    let next_token_distribution =
        WeightedAliasIndex::new(softmax(next_token_logits.to_owned()).to_vec()).unwrap();
    let mut rng = thread_rng();
    let next_token_id = next_token_distribution.sample(&mut rng);
    let next_token = token_vocabulary
        .id_to_token
        .get(&(next_token_id as u8))
        .unwrap();
    *next_token
}

pub fn sample_text(network: &SmallLanguageModel, prompt: Vec<f32>) -> String {
    let mut raw_context = prompt;
    let mut text = Vec::new();
    for _ in 0..network.configuration.context_window_size {
        let input = Rc::new(
            TensorBuilder::new(
                Array::from_shape_vec((raw_context.len(),), raw_context.clone())
                    .expect("array should build")
                    .into_dyn(),
            )
            .build(),
        );
        let logits = network.forward(input);
        let next_token = sample_next_token(&network.configuration.token_vocabulary, logits);
        text.push(next_token);
        raw_context.push(
            *network
                .configuration
                .token_vocabulary
                .token_to_id
                .get(&next_token)
                .unwrap() as f32,
        );
    }
    text.iter().collect()
}

pub fn train_slm(network: SmallLanguageModel, max_steps: Option<usize>) -> SmallLanguageModel {
    let mut optimizer =
        AdaptiveMomentEstimationOptimizer::new(network.parameters(), 0.0004, 0.9, 0.999, 1e-8);

    let training_megastring = fs::read_to_string("training_data.txt").expect("file slurped");
    let training_tokenstream = network
        .configuration
        .token_vocabulary
        .tokenize(&training_megastring);

    let start_time = time::Instant::now();
    let mut last_status_update = time::Instant::now();
    let mut last_checkpoint = time::Instant::now();

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
                        network.configuration.token_vocabulary.size(),
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
        let loss_value = loss.item();
        backprop(loss);
        optimizer.step();
        optimizer.unset_gradients();

        if last_status_update.elapsed() > time::Duration::from_secs(60 * 10) {
            println!(
                "{}: after {}s, {} steps, loss: {}",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                start_time.elapsed().as_secs(),
                optimizer.step_count(),
                loss_value
            );
            println!("sample: {:?}", sample_text(&network, vec![0.0]));
            last_status_update = time::Instant::now();
        }

        if last_checkpoint.elapsed() > time::Duration::from_secs(60 * 30) {
            serialize(&network, &format!("{}", optimizer.step_count()))
                .expect("network should write");
            last_checkpoint = time::Instant::now();
        }

        if let Some(step_limit) = max_steps {
            if optimizer.step_count() > step_limit {
                break;
            }
        }
    }
    network
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_output() {
        let network = SmallLanguageModel::new(
            "my_language_model",
            SmallLanguageModelConfiguration::default(),
        );
        let output = network.forward(Rc::new(
            TensorBuilder::new(array![0.0, 1.0].into_dyn()).build(),
        ));
        assert_eq!(output.borrow_array().shape(), &[2, 97]);
    }
}
