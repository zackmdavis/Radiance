#![allow(mixed_script_confusables)]

use std::env;
use std::fs;
// use std::io;
// use std::io::Write;

// use crate::core::serialization::deserialize;
use crate::core::tokenization;
use crate::core::Parameterized;

mod core;
mod language_model;

fn main() {
    println!("Hello Radiance world!");

    env_logger::init();

    let args = env::args().collect::<Vec<_>>();

    let instructions = "pass `--train`, `--continue-training [NPY file]`, or `--chat [NPY file]`";

    if args.len() <= 1 {
        println!("{}", instructions);
        return;
    }

    match args[1].as_str() {
        "--train" => {
            let token_training_megastring =
                fs::read_to_string("training_data/token_training_data.txt").expect("file slurped");
            let token_vocabulary =
                tokenization::TokenVocabulary::new_from_corpus(token_training_megastring, 200);
            let network = language_model::SmallLanguageModel::new(
                "my_language_model",
                language_model::SmallLanguageModelConfiguration::default(),
                token_vocabulary,
            );
            println!("parameter count: {}", network.parameter_count());
            println!(
                "sample at initialization: {}",
                language_model::sample_text(&network, vec![0.0])
            );
            language_model::train_slm(
                network,
                args.get(2).map(|n| n.parse().expect("arg should be int")),
            );
        }
        // TODO—redo serialization to include tokenization
        //
        // "--continue-training" => {
        //     let filename = &args[2];
        //     println!("loading weights from {:?}", filename);
        //     deserialize(&network, filename).expect("network should deserialize");
        //     println!(
        //         "sample at initialization: {}",
        //         language_model::sample_text(&network, vec![0.0])
        //     );
        //     language_model::train_slm(network, None);
        // }
        // "--chat" => {
        //     let filename = &args[2];
        //     println!("loading weights from {:?}", filename);
        //     deserialize(&network, filename).expect("network should deserialize");
        //     let mut prompt = String::with_capacity(250);
        //     loop {
        //         print!(">>> ");
        //         io::stdout().flush().expect("stdout should flush");
        //         io::stdin()
        //             .read_line(&mut prompt)
        //             .expect("line should read");
        //         println!(
        //             "{}",
        //             language_model::sample_text(
        //                 &network,
        //                 network.configuration().token_vocabulary.token_id_ize(&prompt)
        //             )
        //         );
        //         prompt.clear();
        //     }
        // }
        _ => {
            println!("{}", instructions);
        }
    }
}
