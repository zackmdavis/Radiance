use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Mutex;

use lazy_static::lazy_static;
use ndarray::prelude::*;

use topological_sort::TopologicalSort;

pub mod attention;
pub mod demo;
pub mod dense;
pub mod embedding;
pub mod operations;
pub mod optimization;

use self::operations::Operation;

lazy_static! {
    static ref COUNTER: Mutex<u64> = Mutex::new(0);
}

fn generate_sequential_tensor_id() -> String {
    let mut num = COUNTER.lock().unwrap();
    *num += 1;
    format!("Tensor{}", num)
}

// TODO SOMEDAY: genericize to Tensor<T> ... ArrayD<T> to support integer arrays?
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
    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn borrow_array(&self) -> Ref<ArrayD<f32>> {
        self.array.borrow()
    }

    pub fn unset_gradient(&self) {
        *self.gradient.borrow_mut() = None
    }

    pub fn item(&self) -> f32 {
        assert_eq!(self.borrow_array().shape(), &[1,]);
        self.borrow_array()[0]
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
            requires_gradient: false,
            gradient: RefCell::new(None),
            origin: None,
        }
    }

    pub fn identifier(mut self, identifier: &str) -> TensorBuilder {
        self.identifier = Some(identifier.to_owned());
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

#[derive(Debug)]
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

pub fn backprop(culmination: Rc<Tensor>) {
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
                if contribution.is_any_nan() {
                    println!(
                        "contribution for {:?} contains NaN, operation of origin: {:?}",
                        &parent.identifier, origin.operation
                    );
                    assert!(false);
                }
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

pub trait Parameterized {
    fn parameters(&self) -> Vec<Rc<Tensor>>;

    fn parameter_count(&self) -> usize {
        let mut scalar_parameter_count = 0;
        for tensor_parameter in self.parameters() {
            scalar_parameter_count += tensor_parameter
                .borrow_array()
                .shape()
                .iter()
                .product::<usize>();
        }
        scalar_parameter_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::operations::{Addition, Multiplication};

    #[test]
    fn test_backprop() {
        // Test written by Claude 3.5 Sonnet
        let a = Rc::new(
            TensorBuilder::new(array![2.0].into_dyn())
                .identifier("a")
                .requires_gradient(true)
                .build(),
        );
        let b = Rc::new(
            TensorBuilder::new(array![3.0].into_dyn())
                .identifier("b")
                .requires_gradient(true)
                .build(),
        );
        let c = Rc::new(
            TensorBuilder::new(array![4.0].into_dyn())
                .identifier("c")
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
                .identifier("a")
                .requires_gradient(true)
                .build(),
        );
        let b = Rc::new(
            TensorBuilder::new(array![3.0].into_dyn())
                .identifier("b")
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
