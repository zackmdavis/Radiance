use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Mutex;

use lazy_static::lazy_static;
use ndarray::prelude::*;

use topological_sort::TopologicalSort;

lazy_static! {
    static ref COUNTER: Mutex<u64> = Mutex::new(0);
}

fn generate_sequential_tensor_id() -> String {
    let mut num = COUNTER.lock().unwrap();
    *num += 1;
    format!("Tensor{}", num)
}

struct Tensor {
    identifier: String,
    array: ArrayD<f32>,
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

struct TensorBuilder {
    array: ArrayD<f32>,
    identifier: Option<String>,
    requires_gradient: bool,
    origin: Option<Origin>,
}

impl TensorBuilder {
    fn new(array: ArrayD<f32>) -> TensorBuilder {
        TensorBuilder {
            array,
            identifier: None,
            requires_gradient: true,
            origin: None,
        }
    }

    fn identifier(mut self, identifier: String) -> TensorBuilder {
        self.identifier = Some(identifier);
        self
    }

    fn requires_gradient(mut self, requires: bool) -> TensorBuilder {
        self.requires_gradient = requires;
        self
    }

    fn origin(mut self, origin: Origin) -> TensorBuilder {
        self.origin = Some(origin);
        self
    }

    fn build(self) -> Tensor {
        Tensor {
            array: self.array,
            identifier: match self.identifier {
                Some(identifier) => identifier,
                None => generate_sequential_tensor_id(),
            },
            requires_gradient: self.requires_gradient,
            gradient: RefCell::new(None),
            origin: self.origin,
        }
    }
}

struct Origin {
    operation: Box<dyn Operation>,
    parents: Vec<Rc<Tensor>>,
}

trait Operation {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor>;
    fn backward(
        &self,
        _out_gradient: &ArrayD<f32>,
        _args: Vec<Rc<Tensor>>,
        _arg_index: usize,
    ) -> ArrayD<f32>;
}

struct Addition {}

impl Operation for Addition {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        // clone has dubious performance implications?
        let array = inputs[0].array.clone() + inputs[1].array.clone();
        let origin = Origin {
            operation: Box::new(Addition {}),
            parents: vec![inputs[0].clone(), inputs[1].clone()],
        };
        Rc::new(TensorBuilder::new(array).origin(origin).build())
    }
    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        _args: Vec<Rc<Tensor>>,
        _arg_index: usize,
    ) -> ArrayD<f32> {
        // Addition just passes the gradient through to both branches.
        out_gradient.clone()
    }
}

struct Multiplication {}

impl Operation for Multiplication {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        let array = inputs[0].array.clone() * inputs[1].array.clone(); // dubious performance &c.
        let origin = Origin {
            operation: Box::new(Multiplication {}),
            parents: vec![inputs[0].clone(), inputs[1].clone()],
        };
        Rc::new(TensorBuilder::new(array).origin(origin).build())
    }
    fn backward(
        &self,
        out_gradient: &ArrayD<f32>,
        args: Vec<Rc<Tensor>>,
        arg_index: usize,
    ) -> ArrayD<f32> {
        let other_arg_index = match arg_index {
            0 => 1,
            1 => 0,
            _ => panic!("binary operation expected"),
        };
        // d/dx(xy) = y
        out_gradient * args[other_arg_index].array.clone() // dubious perf &c.
    }
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
    gradients.insert(culmination.identifier.clone(), array![1.].into_dyn());
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

#[test]
fn test_addition_forward() {
    let a = TensorBuilder::new(array![1.].into_dyn()).build();
    let b = TensorBuilder::new(array![2.].into_dyn()).build();
    let c = Addition {}.forward(vec![Rc::new(a), Rc::new(b)]);
    assert_eq!(c.array, array![3.].into_dyn());
}

#[test]
fn test_multiplication_forward() {
    let a = TensorBuilder::new(array![2.].into_dyn()).build();
    let b = TensorBuilder::new(array![3.].into_dyn()).build();
    let c = Multiplication {}.forward(vec![Rc::new(a), Rc::new(b)]);
    assert_eq!(c.array, array![6.].into_dyn());
}

#[test]
fn test_addition_backward() {
    let a = TensorBuilder::new(array![1.].into_dyn()).build();
    let b = TensorBuilder::new(array![2.].into_dyn()).build();
    let args = vec![Rc::new(a), Rc::new(b)];
    let out_gradient = array![1.].into_dyn();
    assert_eq!(
        Addition {}.backward(&out_gradient, args.clone(), 0),
        out_gradient
    );
    assert_eq!(
        Addition {}.backward(&out_gradient, args.clone(), 1),
        out_gradient
    );
}

#[test]
fn test_multiplication_backward() {
    let a = TensorBuilder::new(array![2.].into_dyn()).build();
    let b = TensorBuilder::new(array![3.].into_dyn()).build();
    let args = vec![Rc::new(a), Rc::new(b)];
    let out_gradient = array![1.].into_dyn();
    assert_eq!(
        Multiplication {}.backward(&out_gradient, args.clone(), 0),
        array![3.].into_dyn()
    );
    assert_eq!(
        Multiplication {}.backward(&out_gradient, args.clone(), 1),
        array![2.].into_dyn()
    );
}

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
