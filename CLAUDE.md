# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Radiance is an educational neural networks library and autodifferentiation framework written in Rust. It implements a from-scratch autograd system with computational graph tracking and backpropagation.

## Core Architecture

### Autodifferentiation System (src/core/mod.rs)

The foundation is a tensor-based autodifferentiation framework:

- **Tensor**: Core data structure wrapping `ArrayD<f32>` from ndarray with gradient tracking
- **TensorBuilder**: Fluent API for constructing tensors with optional identifiers, gradients, and origin tracking
- **Origin**: Records the operation and parent tensors that produced each tensor
- **Computational Graph**: Built implicitly during forward pass via `Origin` structs
- **Backpropagation**: `backprop(culmination)` performs reverse-mode autodiff by:
  1. Topologically sorting the computation graph via `sorted_computation_graph()`
  2. Traversing in reverse-topological order
  3. Calling each operation's `backward()` method to accumulate gradients

Key design decisions:
- Uses `Rc<Tensor>` for reference-counted sharing of tensors in the graph
- `RefCell` for interior mutability of arrays and gradients
- Tensors identified by unique strings for tracking in gradient map
- No automatic broadcasting (explicitly disallowed with error message about "moral law")

### Operations (src/core/operations.rs)

All operations implement the `Operation` trait with:
- `forward(inputs)`: Compute output and record `Origin`
- `backward(out_gradient, args, arg_index)`: Compute gradient contribution for specified input

Includes: Addition, Multiplication, MatrixMultiplication, Transpose, Concatenate, Reshape, Exponentiation, LeakyReLU, Softmax variants, Masking, Normalization, and SoftmaxCrossEntropy loss.

### Neural Network Components

- **Linear** (src/core/dense.rs): Fully connected layer with Xavier-style initialization
- **MultiLayerPerceptron**: Stacked linear layers with LeakyReLU activations
- **TokenEmbedding** (src/core/embedding.rs): Learnable token embeddings with tied weights for "unembedding" (projecting back to vocabulary space)
- **AttentionHead** (src/core/attention.rs): Single attention head with Q, K, V projections and causal masking
- **AttentionLayer**: Multi-head attention with residual connections, layer normalization, and MLP blocks

### Language Model (src/language_model.rs)

**SmallLanguageModel**: Decoder-only transformer with:
- Token embeddings + sinusoidal positional encoding
- Stack of attention layers
- Trained on character-level next-token prediction with SoftmaxCrossEntropy loss
- Uses AdaptiveMomentEstimationOptimizer (Adam) with typical hyperparameters

### Optimization (src/core/optimization.rs)

**AdaptiveMomentEstimationOptimizer**: Adam optimizer tracking first and second moments per parameter.

### Serialization (src/core/serialization.rs)

Uses ndarray-npy format for saving/loading model weights. Serializes all parameters from a `Parameterized` object to `.npy` files with structured naming.

## Common Commands

### Building
```bash
cargo build --release  # Build optimized binary
cargo build            # Build debug binary
```

### Testing
```bash
cargo test                           # Run all tests
cargo test test_backprop            # Run specific test
cargo test --no-fail-fast           # Run all tests even if some fail
```

### Running

Training requires a `training_data.txt` file in the project root (gitignored).

```bash
# Start new training run
cargo run --release -- --train

# Start new training with step limit
cargo run --release -- --train 10000

# Continue training from checkpoint
cargo run --release -- --continue-training path/to/checkpoint.npy

# Interactive chat mode with trained model
cargo run --release -- --chat path/to/checkpoint.npy
```

The model automatically:
- Logs status every 10 minutes (loss and sample text)
- Saves checkpoints every 30 minutes to `{step_count}.npy`
- Uses environment logger (set `RUST_LOG=trace` for verbose logging)

### Profiling

The project includes flamegraph artifacts suggesting performance profiling has been done:
```bash
# Profiling with perf (Linux)
cargo build --release
perf record --call-graph dwarf ./target/release/radiance --train 1000
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

Note: `profile.release` in Cargo.toml sets `debug = true` to enable symbols in release builds.

## Development Notes

### Parameter Management

The `Parameterized` trait provides:
- `identifier()`: Get component name
- `parameters()`: Collect all learnable tensors
- `parameter_count()`: Count total scalar parameters

All neural network components implement this trait for unified parameter access.

### Training Data Format

Character-level tokenization with a fixed vocabulary (see `TokenVocabulary::default()` in src/core/embedding.rs). The vocabulary includes a special `▶` start-of-sequence token (ID 0) and printable ASCII characters.

### Comparison Scripts

Python scripts in the root directory compare Radiance implementations against PyTorch:
- `torch_comparison.py`: General operations
- `adam_comparison.py`: Adam optimizer behavior
- `linear_comparison.py`: Linear layer outputs
- `layernorm_example.py`: Layer normalization

These are useful for validating correctness during development.

### Current TODOs (from README)

Big items:
- Big training run
- Add biases to attention QKV projections

Small items:
- Address input vs args inconsistent naming in backward pass
- Ignored arguments in backward pass handling
- Reduce Origin struct boilerplate
