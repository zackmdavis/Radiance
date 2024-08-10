#![allow(mixed_script_confusables)]

use std::env;
use std::io;
use std::io::Write;

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

    let instructions =
        "pass `--train`, `--continue-training [NPY weight file]`, or `--chat [NPY weight file]`";

    if args.len() <= 1 {
        println!("{}", instructions);
        return;
    }

    match args[1].as_str() {
        "--train" => {
            println!(
                "sample at initialization: {}",
                language_model::sample_text(&network, vec![0.0])
            );
            language_model::train_slm(network);
        }
        "--continue-training" => {
            let filename = &args[2];
            println!("loading weights from {:?}", filename);
            deserialize(&network, filename).expect("network should deserialize");
            println!(
                "sample at initialization: {}",
                language_model::sample_text(&network, vec![0.0])
            );
            language_model::train_slm(network);
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
                        network.configuration().token_vocabulary.tokenize(&prompt)
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
