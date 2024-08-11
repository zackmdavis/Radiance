# Radiance

Radiance is an educational neural networks library and autodifferentiation framework!

### TODO next

### big

 * port `Normalize` to `NormalizeRows`
 * LayerNorm
 * assess Adam vs. SGD situation; test Adam implementation against PyTorch
 * attention QKV biases
 * assess training speed impact of new elements (LayerNorm, bias, Adam)
 * choose bigger training hyperparams?
 * big training run!!

#### small

 * address input _vs._ args inconsistent naming infelicity
 * ignored arguments in backward pass are sketchy
 * reduce Origin boilerplate (more complicated than it looks; `Self {}` is not a legal constructor)
