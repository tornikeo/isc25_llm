# About llama 3.1:8b

Output shape: [4096, 128256]

Attention:
  Layers: 32
  Attention heads: 32
  context_length: 131072
  Model Dimension: 4096
  FFN Dimension: 14336
  vocab_size: 128256
  Activation: SwiGLU
  Key/Value Heads: 8
  Peak Learning Rate: 3e-4
  Positional Embeddings: RoPE(theta = 500,000)

Standard dense transformer model. Not mixture-of-experts.

Grouped query-attention (GQA). 8 key-value heads. <- Gotta learn this one. Ainslie et al. 2023.
Attention mask that prevents self-attention between different documents in same seq

RoPE base frequency hyperparameter to 500,000. Xiong et al. (2023), for context lengths up to 32,768



# Data

Web data -> plain text. Markdown markers excluded too.

# Inference
Inference done at fp8

# How model size was determined

First, create validation, isoFLOP curves, and identify the minima of each isoFLOP parabola.

Second, that minima is a compute-optimal model. 

Then, assume power law relation: Compute budget C, number of training tokens N(C)

N(C) = A C^alpha

Fitting the (alpha, A), they get (.53, .29). So:

N*(C) = .29 C**.53

N*(C) = 1.04636e13

C = 6 * P * N, where C is budget, P is N-model params, and N is training tokens

## How did llama become 405B?

Recipe.

First, make IsoFLOP curves. This is a curve, where FLOPS are equal, but model size and training token combinations vary. It is know that this relationship is as follows

$$
ComputeFlops = 6 * Parameters * Tokens
$$

Therefore, training 1B parameter model, on 1T tokens requires $6 * 1e9 * 1e12 = 6e21 FLOPs$. Plot each of the model's validation loss, and form IsoFlops curves. You will find the curves to be concave. Locate each center and hold onto that data.

That data now holds: 

- ComputeFlops
- Model size
- Training tokens
- best validation loss at that ComputeFlops

Next, we want to make a relationship between ComputeFlops and Tokens. We use above data to fit this function:

Tokens(ComputeFlops) = A * ComputeFlops ** alpha

Fitting least squares gives us A = 0.299, and alpha = 0.537.

We know our budget of ComputeFlops to be 3.8e25 . Therefore, Tokens(ComputeFlops) = 0.299 * 3.8e25 ** 0.537 = 1.62935223e13 tokens.

What is the model size? We get the model size from $Parameters = ComputeFlops / (6 * Tokens)$. This is $ 3.8e25 / (6 * 1.62935223e13 ) = 3.8870253e11$. 

This suggests training a 388 B (in paper 402B, because they don't give more digits of 0.537... fitted parameter alpha, and 402B is appartently sensitive to this) parameter model. But they round this up to 405B, possibly due to better fitting it to GPUs?

**How to calculate Model Flop Utilization?** Chowdhery et al. 2023. 

# Inference 

H100 has native fp8 support. Meta used that for training. FP8 quantization was applied to most matrix multiplications inside model. mat muls were 50% of inference time. 

Attention layers are not quantized. Feedforward network layers are quantized. 

First and last Transformer layers are not quantized.

**High-perplexity tokens such as dates can lead to large activation values.** To address this issue, we upper bound the dynamic scaling factors to 1200.

Row-wise quantization works better than tensor-wise. 