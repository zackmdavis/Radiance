use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use std::sync::Mutex;

use lazy_static::lazy_static;
use ndarray::prelude::*;

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
    #[allow(dead_code)]
    origin: Option<Origin>,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("identifier", &self.identifier)
            .field("array", &self.array)
            .field("requires_gradient", &self.requires_gradient)
            .field("gradient", &self.gradient)
            .finish()
    }
}

struct TensorBuilder {
    array: ArrayD<f32>,
    identifier: Option<String>,
    requires_gradient: bool,
    gradient: Option<ArrayD<f32>>,
    origin: Option<Origin>,
}

#[allow(dead_code)]
impl TensorBuilder {
    fn new(array: ArrayD<f32>) -> TensorBuilder {
        TensorBuilder {
            array,
            identifier: None,
            requires_gradient: true,
            gradient: None,
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

    fn gradient(mut self, gradient: ArrayD<f32>) -> TensorBuilder {
        self.gradient = Some(gradient);
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
            gradient: RefCell::new(self.gradient),
            origin: self.origin,
        }
    }
}

struct Origin {
    #[allow(dead_code)]
    operation: Box<dyn Operation>,
    #[allow(dead_code)]
    parents: Vec<Rc<Tensor>>,
}

trait Operation {
    fn forward(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor>;
    fn backward(
        &self,
        _out_gradient: ArrayD<f32>,
        _args: Vec<Rc<Tensor>>,
        _arg_number: usize,
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
        _out_gradient: ArrayD<f32>,
        _args: Vec<Rc<Tensor>>,
        _arg_number: usize,
    ) -> ArrayD<f32> {
        array![1.].into_dyn() // TODO
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
        _out_gradient: ArrayD<f32>,
        _args: Vec<Rc<Tensor>>,
        _arg_number: usize,
    ) -> ArrayD<f32> {
        array![1.].into_dyn() // TODO
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
