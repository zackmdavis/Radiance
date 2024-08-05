#![allow(mixed_script_confusables)]

mod core;
mod language_model;

fn main() {
    println!("Hello neural network world!");
    let network = language_model::train_slm();
    for parameter in network.parameters() {
        println!("{:?}", parameter.identifier());
    }
}
