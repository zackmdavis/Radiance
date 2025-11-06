#![allow(dead_code)]

use std::error::Error;
use std::fs::File;

use log::info;

use ndarray_npy::{NpzReader, NpzWriter};
use serde_json;

use super::Parameterized;

pub fn serialize(parameterized: &dyn Parameterized, specifier: &str) -> Result<(), Box<dyn Error>> {
    let filename = format!("{}-{}.npz", parameterized.identifier(), specifier);
    info!("serializing to {} ...", filename);
    let mut npz_writer = NpzWriter::new(File::create(filename)?);
    for parameter in parameterized.parameters() {
        npz_writer
            .add_array(parameter.identifier(), &*parameter.borrow_array())
            .expect("array should write");
    }
    npz_writer.finish()?;
    Ok(())
}

pub fn deserialize(
    parameterized: &dyn Parameterized,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let mut npz_reader = NpzReader::new(File::open(filename)?)?;
    for parameter in parameterized.parameters() {
        println!("loading weights for {:?}", parameter.identifier());
        *parameter.borrow_array_mut() = npz_reader.by_name(parameter.identifier())?;
    }
    Ok(())
}

use super::tokenization::{TokenVocabulary, STANDARD_VOCABULARY};
use std::collections::HashMap;

impl TokenVocabulary {
    // serialization/deserialization code courtesy of Claude Sonnet 4.5

    pub fn serialize(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &self.merge_rules)?;
        info!(
            "serialized vocabulary with {} merge rules to {}",
            self.merge_rules.len(),
            path
        );
        Ok(())
    }

    pub fn deserialize(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let merge_rules: Vec<(String, String)> = serde_json::from_reader(file)?;

        // Reconstruct hashmaps from STANDARD_VOCABULARY + merge_rules
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (id, token) in STANDARD_VOCABULARY.iter().enumerate() {
            token_to_id.insert(token.to_string(), id as u16);
            id_to_token.insert(id as u16, token.to_string());
        }

        for (rule_id, merge) in merge_rules.iter().enumerate() {
            let token = merge.0.clone() + &merge.1;
            token_to_id.insert(
                token.to_owned(),
                (rule_id + STANDARD_VOCABULARY.len()) as u16,
            );
            id_to_token.insert(
                (rule_id + STANDARD_VOCABULARY.len()) as u16,
                token.to_owned(),
            );
        }

        info!(
            "deserialized vocabulary with {} merge rules from {}",
            merge_rules.len(),
            path
        );
        Ok(TokenVocabulary {
            token_to_id,
            id_to_token,
            merge_rules,
        })
    }
}
