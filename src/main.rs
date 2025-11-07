#![allow(mixed_script_confusables)]

use std::env;
use std::fs;
use std::io;
use std::io::Write;
use std::path::Path;

use crate::core::serialization::deserialize;
use crate::core::tokenization;
use crate::core::Parameterized;

mod core;
mod language_model;

fn main() {
    println!("Hello Radiance world!");

    env_logger::init();

    let args = env::args().collect::<Vec<_>>();

    let instructions = "pass `--build-vocabulary [vocabulary_size]`, `--train`, `--continue-training [NPY file]`, or `--chat [NPY file]`";

    let token_vocabulary = if Path::new("vocabulary.json").exists() {
        tokenization::TokenVocabulary::deserialize("vocabulary.json").expect("vocabulary should load")
    } else {
        tokenization::TokenVocabulary::default()
    };

    let network = language_model::SmallLanguageModel::new(
        "my_language_model",
        language_model::SmallLanguageModelConfiguration::default(),
        token_vocabulary,
    );
    println!("parameter count: {}", network.parameter_count());

    if args.len() <= 1 {
        println!("{}", instructions);
        return;
    }

    match args[1].as_str() {
        "--build-vocabulary" => {
            let token_training_megastring =
                fs::read_to_string("training_data/token_training_data.txt").expect("file slurped");
            let vocabulary_size = args.get(2).map(|n| n.parse().expect("arg should be int")).expect("arg should be supplied");
            let token_vocabulary =
                tokenization::TokenVocabulary::new_from_corpus(token_training_megastring, vocabulary_size);
            token_vocabulary
                .serialize("vocabulary.json")
                .expect("vocabulary should serialize");
        }
        "--train" => {
            println!(
                "sample at initialization: {}",
                language_model::sample_text(&network, vec![0.0])
            );
            language_model::train_slm(
                network,
                args.get(2).map(|n| n.parse().expect("arg should be int")),
            );
        }
        "--continue-training" => {
            let filename = &args[2];
            println!("loading weights from {:?}", filename);
            deserialize(&network, filename).expect("network should deserialize");
            println!(
                "sample at initialization: {}",
                language_model::sample_text(&network, vec![0.0])
            );
            language_model::train_slm(network, None);
        }
        "--chat" => {
            let filename = &args[2];
            println!("loading weights from {:?}", filename);
            deserialize(&network, filename).expect("network should deserialize");
            let mut prompt = String::with_capacity(250);
            loop {
                print!(">>> ");
                io::stdout().flush().expect("stdout should flush");
                io::stdin()
                    .read_line(&mut prompt)
                    .expect("line should read");
                println!(
                    "{}",
                    language_model::sample_text(
                        &network,
                        network.token_vocabulary.token_id_ize(&prompt)
                    )
                );
                prompt.clear();
            }
        }
        _ => {
            println!("{}", instructions);
        }
    }
}
