#![allow(dead_code)]

use std::error::Error;
use std::fs::File;

use ndarray_npy::{NpzReader, NpzWriter};

use super::Parameterized;

pub fn serialize(parameterized: &dyn Parameterized, specifier: &str) -> Result<(), Box<dyn Error>> {
    let mut npz_writer =
        NpzWriter::new(File::create(format!("{}-{}.npz", parameterized.identifier(), specifier))?);
    for parameter in parameterized.parameters() {
        println!("serializing weights for {:?}", parameter.identifier());
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
