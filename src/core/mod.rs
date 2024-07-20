use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Mutex;

use lazy_static::lazy_static;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use topological_sort::TopologicalSort;

pub mod demo;
pub mod operations;
pub mod optimization;

use self::operations::{Addition, MatrixMultiplication, Operation};

lazy_static! {
    static ref COUNTER: Mutex<u64> = Mutex::new(0);
}

fn generate_sequential_tensor_id() -> String {
    let mut num = COUNTER.lock().unwrap();
    *num += 1;
    format!("Tensor{}", num)
}

pub struct Tensor {
    identifier: String,
    array: RefCell<ArrayD<f32>>,
    requires_gradient: bool,
    gradient: RefCell<Option<ArrayD<f32>>>,
    origin: Option<Origin>,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("identifier", &self.identifier)
            .field("array", &self.array)
            .field("gradient", &self.gradient)
            .finish()
    }
}

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identifier.hash(state);
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.identifier == other.identifier
    }
}

impl Eq for Tensor {}

impl Tensor {
    fn unset_gradient(&self) {
        *self.gradient.borrow_mut() = None
    }
}

pub struct TensorBuilder {
    array: ArrayD<f32>,
    identifier: Option<String>,
    requires_gradient: bool,
    gradient: RefCell<Option<ArrayD<f32>>>,
    origin: Option<Origin>,
}

impl TensorBuilder {
    pub fn new(array: ArrayD<f32>) -> TensorBuilder {
        TensorBuilder {
            array,
            identifier: None,
            requires_gradient: true, // shouldn't this be false?
            gradient: RefCell::new(None),
            origin: None,
        }
    }

    pub fn identifier(mut self, identifier: String) -> TensorBuilder {
        self.identifier = Some(identifier);
        self
    }

    pub fn requires_gradient(mut self, requires: bool) -> TensorBuilder {
        self.requires_gradient = requires;
        self
    }

    #[allow(dead_code)]
    pub fn gradient(self, gradient: ArrayD<f32>) -> TensorBuilder {
        *self.gradient.borrow_mut() = Some(gradient);
        self
    }

    fn origin(mut self, origin: Origin) -> TensorBuilder {
        self.origin = Some(origin);
        self
    }

    pub fn build(self) -> Tensor {
        Tensor {
            array: RefCell::new(self.array),
            identifier: match self.identifier {
                Some(identifier) => identifier,
                None => generate_sequential_tensor_id(),
            },
            requires_gradient: self.requires_gradient,
            gradient: self.gradient,
            origin: self.origin,
        }
    }
}

impl From<Vec<f32>> for Tensor {
    fn from(value: Vec<f32>) -> Tensor {
        TensorBuilder::new(Array::from_vec(value).into_dyn()).build()
    }
}

impl From<f32> for Tensor {
    fn from(value: f32) -> Tensor {
        TensorBuilder::new(Array::from_vec(vec![value]).into_dyn()).build()
    }
}

struct Origin {
    operation: Box<dyn Operation>,
    parents: Vec<Rc<Tensor>>,
}

fn register_parents(sorter: &mut TopologicalSort<Rc<Tensor>>, child: Rc<Tensor>) {
    if let Some(origin) = &child.origin {
        for parent in &origin.parents {
            sorter.add_dependency(parent.clone(), child.clone());
            register_parents(sorter, parent.clone());
        }
    }
}

fn sorted_computation_graph(end: Rc<Tensor>) -> Vec<Rc<Tensor>> {
    let mut sorter = TopologicalSort::new();
    register_parents(&mut sorter, end);
    let mut sorted = sorter.collect::<Vec<_>>();
    // We actually want reverse-topological order
    sorted.reverse();
    sorted
}

fn backprop(culmination: Rc<Tensor>) {
    let mut gradients = HashMap::<String, ArrayD<f32>>::new();
    gradients.insert(
        culmination.identifier.clone(),
        Array::ones(culmination.array.borrow().shape()).into_dyn(),
    );
    for node in sorted_computation_graph(culmination) {
        let gradient = gradients
            .remove(&node.identifier)
            .expect("gradient should be stored");

        // If it requires gradient, set it from the map
        if node.requires_gradient {
            *node.gradient.borrow_mut() = Some(gradient.clone());
        }

        // If it has an origin, use the backward function to accumulate
        // gradients for the parents in our map
        if let Some(origin) = &node.origin {
            let out_gradient = gradient;
            for (i, parent) in origin.parents.iter().enumerate() {
                let contribution =
                    origin
                        .operation
                        .backward(&out_gradient, origin.parents.clone(), i);
                match gradients.get_mut(&parent.identifier) {
                    Some(gradient) => {
                        *gradient = &*gradient + contribution;
                    }
                    None => {
                        gradients.insert(parent.identifier.clone(), contribution);
                    }
                }
            }
        }
    }
}

struct Linear {
    #[allow(dead_code)]
    identifier: String,
    weights: Rc<Tensor>,
    biases: Rc<Tensor>,
}

impl Linear {
    fn new(identifier: &str, in_dimensionality: usize, out_dimensionality: usize) -> Linear {
        let k = 1. / (in_dimensionality as f32);
        println!(
            "creating Linear layer with weights shape {:?} and biases shape {:?}",
            (out_dimensionality, in_dimensionality),
            (out_dimensionality, 1)
        );
        Linear {
            identifier: identifier.to_owned(),
            weights: Rc::new(
                TensorBuilder::new(
                    Array::random(
                        (out_dimensionality, in_dimensionality),
                        Uniform::new(-k.sqrt(), k.sqrt()),
                    )
                    .into_dyn(),
                )
                .requires_gradient(true)
                .identifier(format!("{}_weights", identifier))
                .build(),
            ),
            biases: Rc::new(
                TensorBuilder::new(
                    Array::random((out_dimensionality, 1), Uniform::new(-k.sqrt(), k.sqrt()))
                        .into_dyn(),
                )
                .requires_gradient(true)
                .identifier(format!("{}_biases", identifier))
                .build(),
            ),
        }
    }

    fn forward(&self, input: Rc<Tensor>) -> Rc<Tensor> {
        let product = MatrixMultiplication {}.forward(vec![self.weights.clone(), input]);
        let sum = Addition {}.forward(vec![product, self.biases.clone()]);
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::operations::Multiplication;
    #[test]
    fn test_backprop() {
        // Test written by Claude 3.5 Sonnet
        let a = Rc::new(
            TensorBuilder::new(array![2.0].into_dyn())
                .identifier("a".to_string())
                .requires_gradient(true)
                .build(),
        );
        let b = Rc::new(
            TensorBuilder::new(array![3.0].into_dyn())
                .identifier("b".to_string())
                .requires_gradient(true)
                .build(),
        );
        let c = Rc::new(
            TensorBuilder::new(array![4.0].into_dyn())
                .identifier("c".to_string())
                .requires_gradient(true)
                .build(),
        );

        let mul = Multiplication {}.forward(vec![a.clone(), b.clone()]);
        let result = Addition {}.forward(vec![mul, c.clone()]);

        backprop(result);

        assert_eq!(
            *a.gradient.borrow().as_ref().unwrap(),
            array![3.0].into_dyn()
        );
        assert_eq!(
            *b.gradient.borrow().as_ref().unwrap(),
            array![2.0].into_dyn()
        );
        assert_eq!(
            *c.gradient.borrow().as_ref().unwrap(),
            array![1.0].into_dyn()
        );
    }

    #[test]
    fn test_backprop_with_reuse() {
        // Test written by Claude 3.5 Sonnet
        let a = Rc::new(
            TensorBuilder::new(array![2.0].into_dyn())
                .identifier("a".to_string())
                .requires_gradient(true)
                .build(),
        );
        let b = Rc::new(
            TensorBuilder::new(array![3.0].into_dyn())
                .identifier("b".to_string())
                .requires_gradient(true)
                .build(),
        );

        // Compute (a * b) + (a + b)
        let mul = Multiplication {}.forward(vec![a.clone(), b.clone()]);
        let add = Addition {}.forward(vec![a.clone(), b.clone()]);
        let result = Addition {}.forward(vec![mul, add]);

        backprop(result);

        // Gradient for 'a':
        // From (a * b): derivative is b = 3
        // From (a + b): derivative is 1
        // Total: 3 + 1 = 4
        assert_eq!(
            *a.gradient.borrow().as_ref().unwrap(),
            array![4.0].into_dyn()
        );

        // Gradient for 'b':
        // From (a * b): derivative is a = 2
        // From (a + b): derivative is 1
        // Total: 2 + 1 = 3
        assert_eq!(
            *b.gradient.borrow().as_ref().unwrap(),
            array![3.0].into_dyn()
        );
    }
}
