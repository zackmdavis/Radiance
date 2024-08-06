#![allow(mixed_script_confusables)]

mod core;
mod language_model;

fn main() {
    println!("Hello neural network world!");
    let network = language_model::SmallLanguageModel::new(
        "my_language_model",
        language_model::SmallLanguageModelConfiguration::default(),
    );
    for parameter in network.parameters() {
        println!("{:?}", parameter.identifier());
    }
    let token_vocabulary = core::embedding::TokenVocabulary::default();
    println!(
        "sample at initialization: {}",
        language_model::sample_text(&network, &token_vocabulary)
    );

    language_model::train_slm(network);
}
