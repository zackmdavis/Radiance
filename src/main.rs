#![allow(mixed_script_confusables)]

use std::env;

use crate::core::serialization::deserialize;
use crate::core::Parameterized;

mod core;
mod language_model;

fn main() {
    println!("Hello Radiance world!");
    let args = env::args().collect::<Vec<_>>();

    let network = language_model::SmallLanguageModel::new(
        "my_language_model",
        language_model::SmallLanguageModelConfiguration::default(),
    );

    println!("parameter count: {}", network.parameter_count());

    if args.len() > 1 {
        let filename = &args[1];
        println!("loading weights from {:?}", filename);
        deserialize(&network, filename).expect("network should deserialize");
    }

    println!(
        "sample at initialization: {}",
        language_model::sample_text(&network)
    );

    language_model::train_slm(network);
}
