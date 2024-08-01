# Radiance

Radiance is an educational neural networks library and autodifferentiation framework!

### TODO next

### big

 * build scaled dot-product attention from MatrixMultiplication, Multiplication, Mask, Softmax Operations??
   * https://paperswithcode.com/method/scaled

#### small

 * address TensorBuilder identifier infelicity
 * address `requires_gradient` dubious default infelicity
 * address input _vs._ args inconsistent naming infelicity
 * ignored arguments in backward pass are sketchy
 * numerically stabler softmax
 * reduce Origin boilerplate (more complicated than it looks; `Self {}` is not a legal constructor)
