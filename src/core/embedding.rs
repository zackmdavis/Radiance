use std::collections::HashMap;
use std::rc::Rc;

use ndarray::prelude::*;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

use super::operations::{MatrixMultiplication, Operation, Transpose};
use super::{Origin, Tensor, TensorBuilder};

pub struct TokenVocabulary {
    // TODO: getter methods (that don't return Option) instead of pub HashMap
    pub token_to_id: HashMap<char, u8>,
    #[allow(dead_code)]
    pub id_to_token: HashMap<u8, char>,
}

impl TokenVocabulary {
    pub fn new(tokens: Vec<char>) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        for (i, token) in tokens.iter().enumerate() {
            id_to_token.insert(i as u8, *token);
            token_to_id.insert(*token, i as u8);
        }
        TokenVocabulary {
            token_to_id,
            id_to_token,
        }
    }

    pub fn size(&self) -> usize {
        self.token_to_id.len()
    }

    pub fn tokenize(&self, text: String) -> Vec<f32> {
        let mut token_ids = Vec::new();
        for c in text.chars() {
            match self.token_to_id.get(&c) {
                Some(id) => token_ids.push(*id as u8 as f32),
                None => {}
            }
        }
        token_ids
    }
}

impl Default for TokenVocabulary {
    fn default() -> Self {
        let standard_vocabulary = vec![
            '▶', // start of sequence
            '\n', ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b',
            'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '—',
        ];
        Self::new(standard_vocabulary)
    }
}

#[allow(dead_code)]
fn single_positional_encoding(position: usize, embedding_dimensionality: usize) -> Vec<f32> {
    let mut vector = Vec::new();
    let base: f32 = 30.;
    for i in 0..embedding_dimensionality {
        if i % 2 == 0 {
            vector.push(
                ((position as f32) / base.powf(i as f32 / embedding_dimensionality as f32)).sin(),
            )
        } else {
            vector.push(
                ((position as f32) / base.powf((i - 1) as f32 / embedding_dimensionality as f32))
                    .cos(),
            )
        }
    }
    vector
}

#[allow(dead_code)]
pub fn sequence_positional_encoding(length: usize, embedding_dimensionality: usize) -> Rc<Tensor> {
    let mut position_embedding = Array2::zeros((0, embedding_dimensionality));
    for position in 0..length {
        position_embedding
            .push_row(
                Array::from_vec(single_positional_encoding(
                    position,
                    embedding_dimensionality,
                ))
                .view(),
            )
            .expect("row should fit");
    }
    Rc::new(
        TensorBuilder::new(position_embedding.into_dyn())
            .identifier(&format!(
                "positional_embedding_{}",
                embedding_dimensionality
            ))
            .requires_gradient(false)
            .build(),
    )
}

pub struct TokenEmbedding {
    #[allow(dead_code)]
    identifier: String,
    weights: Rc<Tensor>,
}

#[allow(dead_code)]
impl TokenEmbedding {
    pub fn new(identifier: &str, vocabulary_size: usize, embedding_dimensionality: usize) -> Self {
        let weights = Array::random(
            // The rows index over vocabulary tokens
            (vocabulary_size, embedding_dimensionality),
            StandardNormal {},
        )
        .into_dyn();
        Self {
            identifier: identifier.to_owned(),
            weights: Rc::new(
                TensorBuilder::new(weights)
                    .requires_gradient(true)
                    .identifier(&format!("{}_weights", identifier))
                    .build(),
            ),
        }
    }

    pub fn parameters(&self) -> Vec<Rc<Tensor>> {
        vec![self.weights.clone()]
    }

    pub fn dimensionality(&self) -> usize {
        self.weights.array.borrow().shape()[1]
    }

    pub fn embed(&self, input: Rc<Tensor>) -> Rc<Tensor> {
        Lookup {}.forward(vec![self.weights.clone(), input])
    }

    pub fn unembed(&self, x: Rc<Tensor>) -> Rc<Tensor> {
        let u = Transpose {}.forward(vec![self.weights.clone()]);
        MatrixMultiplication {}.forward(vec![x, u])
    }
}

pub(super) struct Lookup {}

impl Operation for Lookup {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        // The weights being the first arg is a distinguished convention,
        // because you could imagine passing in multiple sequences in a future
        // version? (The alternative would be multiple sequences in an
        // additional batch dimension of the one tensor.)
        let embedding_matrix = inputs[0]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");
        let sequence = inputs[1]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix1>()
            .expect("one-dimensional");
        let mut representation = Array2::zeros((0, embedding_matrix.shape()[1]));
        for token in sequence {
            // positions and token IDs are morally usize integers, but for now,
            // my Tensors only support f32
            representation
                .push_row(embedding_matrix.row(token as usize))
                .expect("row should fit");
        }
        let origin = Origin {
            operation: Box::new(Lookup {}),
            parents: inputs.clone(),
        };
        Rc::new(
            TensorBuilder::new(representation.into_dyn())
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
        // The gradients on the output get "un-plucked" back into the embedding matrix.
        let embedding_matrix = args[0]
            .array
            .borrow()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");
        let sequence = args[1].array.borrow();
        let mut gradient = Array2::zeros(embedding_matrix.raw_dim());
        let out_gradient = out_gradient
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("two-dimensional");
        for (i, token) in sequence.iter().enumerate() {
            let mut token_embedding = gradient.row_mut(*token as usize);
            for (j, component) in out_gradient.row(i).iter().enumerate() {
                token_embedding[j] += component;
            }
        }
        gradient.into_dyn()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_token_embedding_and_lookup() {
        // Test written by Claude Sonnet 3.5 (with some human feedback)

        // Create a small token embedding
        let vocab_size = 5;
        let embedding_dim = 3;
        let embedding = TokenEmbedding::new("test", vocab_size, embedding_dim);

        // Create a sequence of tokens with repeats
        let sequence = array![0.0, 2.0, 1.0, 4.0, 2.0, 0.0];
        let sequence_tensor = Rc::new(TensorBuilder::new(sequence.into_dyn()).build());

        // Perform the forward pass using TokenEmbedding's forward method
        let output = embedding.embed(sequence_tensor.clone());

        // Check the output shape
        assert_eq!(output.array.borrow().shape(), &[6, 3]);

        // Perform the backward pass
        let out_gradient = Array2::ones((6, 3)).into_dyn();
        let lookup = Lookup {};
        let inputs = vec![embedding.weights.clone(), sequence_tensor];
        let gradient = lookup.backward(&out_gradient, inputs, 0);

        // Check the gradient shape
        assert_eq!(gradient.shape(), &[5, 3]);

        // Check that the gradient is correct
        let expected_gradient = {
            let mut grad = Array2::zeros((5, 3));
            grad.row_mut(0).fill(2.0); // Token 0 appears twice
            grad.row_mut(1).fill(1.0); // Token 1 appears once
            grad.row_mut(2).fill(2.0); // Token 2 appears twice
            grad.row_mut(3).fill(0.0); // Token 3 doesn't appear
            grad.row_mut(4).fill(1.0); // Token 4 appears once
            grad
        };

        for (a, b) in gradient.iter().zip(expected_gradient.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-5);
        }
    }
}
