# Radiance

Radiance is an educational neural networks library and autodifferentiation framework!

### TODO before big training run

#### craft

 * tokenize
 * curate training data
 * SoftmaxCrossEntropy log–sum–exp trick
 * BLAS
 
#### delegate to Claude?

 * attention QKV biases (see bd1e3e6aa85 for context) 
 * ignored arguments in backward pass are sketchy
 * reduce Origin boilerplate (more complicated than it looks; `Self {}` is not a legal constructor)
