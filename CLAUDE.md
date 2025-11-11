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
- Trained on next-token prediction with SoftmaxCrossEntropy loss
- Uses AdaptiveMomentEstimationOptimizer (Adam) with typical hyperparameters
- Current production vocab: ~1200 tokens (BPE-trained)

**Current default config (~900K parameters):**
- context_window: 256
- embedding_dim: 128
- heads: 4
- layers: 4

**Recommended scaling for impressive results (50M parameters):**
- context_window: 512
- embedding_dim: 768
- heads: 12
- layers: 12

This is roughly GPT-2 small territory and should show qualitatively better coherence and reasoning. Requires multi-core CPU (8+ cores recommended) and adequate training data to avoid overfitting.

### Optimization (src/core/optimization.rs)

**AdaptiveMomentEstimationOptimizer**: Adam optimizer tracking first and second moments per parameter.

### Serialization (src/core/serialization.rs)

Uses ndarray-npy format for saving/loading model weights. Serializes all parameters from a `Parameterized` object to `.npy` files with structured naming.

## Common Commands

### Building
```bash
cargo build --release                # Build optimized binary (pure Rust)
cargo build --release --features blas  # Build with BLAS acceleration
cargo build                          # Build debug binary
```

**BLAS Support**: The optional `blas` feature enables hardware-accelerated linear algebra via OpenBLAS. This provides:
- ~15% speedup on single-core systems (from vectorization/cache optimization)
- Significant speedup (2-4x+) on multi-core systems via parallel matrix operations

To use BLAS on Ubuntu/Debian:
```bash
sudo apt-get install libopenblas-dev liblapack-dev
```

If linking fails, create `.cargo/config.toml`:
```toml
[target.x86_64-unknown-linux-gnu]
rustflags = ["-L", "/usr/lib/x86_64-linux-gnu", "-l", "openblas"]
```

OpenBLAS automatically uses all available CPU cores. For single-core systems, BLAS benefits are minimal.

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

### Tokenization (src/core/tokenization.rs)

**TokenVocabulary**: BPE (Byte Pair Encoding) tokenizer with:
- Base vocabulary of 97 characters (`STANDARD_VOCABULARY`): start-of-sequence token `▶` (ID 0) plus printable ASCII
- `new_from_corpus(text, vocab_size)`: Learns merge rules from training data by iteratively finding and merging the most frequent character bigrams
- `tokenize(text)`: Applies learned merge rules to segment text into subword tokens
- `token_id_ize(text)`: Converts text to token IDs for model input
- `Default` implementation uses only the base 97-character vocabulary (character-level fallback)
- Current production vocabulary: ~1200 tokens (trained on corpus)

### Comparison Scripts

Python scripts in the root directory compare Radiance implementations against PyTorch:
- `torch_comparison.py`: General operations
- `adam_comparison.py`: Adam optimizer behavior
- `linear_comparison.py`: Linear layer outputs
- `layernorm_example.py`: Layer normalization

These are useful for validating correctness during development.

### Performance Considerations

**Training Speed**:
- Single-core without BLAS: ~1.8 steps/sec (900K param model)
- Single-core with BLAS: ~2.1 steps/sec (15% improvement)
- Multi-core (8 cores) with BLAS: Expected 15-20+ steps/sec due to parallel matrix operations

**Memory Requirements** (approximate):
- 900K params: <1 GB total (model + gradients + optimizer + activations)
- 50M params: 3-5 GB total
- 100M params: 6-10 GB total

Single-sequence training (no batching) keeps memory usage low. CPU speed is the primary bottleneck.

**Timing Accuracy Note**: The status logging includes time spent on text sampling, which can be non-trivial. Reported steps/sec in logs is slightly deflated. For accurate training-only timing, exclude sampling time from measurements.

### Current TODOs (from README)

Big items:
- Big training run (50M+ params on multi-core hardware)
- Add biases to attention QKV projections

Small items:
- Address input vs args inconsistent naming in backward pass
- Ignored arguments in backward pass handling
- Reduce Origin struct boilerplate
