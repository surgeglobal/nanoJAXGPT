# nanoJAXGPT

A JAX/Equinox rewrite of [nanoGPT](https://github.com/karpathy/nanoGPT) that prioritizes pedagogy and modern ML frameworks. This implementation reimagines the original repository using JAX's powerful array computation capabilities while maintaining Pytorch-like simplicity through Equinox. The code is clean and pedagogical: `model.py` is a ~300-line GPT model definition with Equinox modules, and `train.py` implements training with JAX's powerful transformations.

> 📖 **Want to understand every line?** Check out our complete tutorial: [nanoJAXGPT: A pedagogical introduction to JAX/Equinox](https://huggingface.co/blog/sachithgunasekara/nanojaxgpt)

## What's cool about this?

- **JAX Native**: Leverages JAX's powerful features like `jit`, `grad`, and `vmap` for efficient training
- **Equinox Integration**: Maintains PyTorch-like simplicity while getting all of JAX's benefits
- **SwiGLU Activation**: Enhanced activation function replacing GELU in the original
- **Educational First**: Clear, documented code structure for learning JAX and Transformers

## Setup

First, clone the repository:

```bash
git clone https://github.com/surgeglobal/nanoJAXGPT
cd nanoJAXGPT
```

Then install the dependencies using the provided requirements.txt:

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies:

- JAX and Equinox for core model implementation
- Transformers for loading GPT-2 checkpoints
- Datasets for data processing
- Tiktoken for BPE tokenization
- Wandb for experiment tracking
- Additional utilities (numpy, tqdm)

For GPU support, make sure you have CUDA installed and install the CUDA-enabled version of JAX:

```bash
# For CUDA 12
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 11
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

To verify your installation:

```python
import jax
print("Available devices:", jax.devices())  # Should show your GPU(s) if available
```

Now you're ready to start training! Check out the [Quick start](#quick-start) section below.

## Quick Start

If you're new to JAX or just want to get your feet wet, let's start a simple training example.

> 📖 **Want to train on your custom datasets?** You can create custom scripts for your preferred datasets as we have done for the tinystories dataset.

**I have a GPU/TPU**. Excellent! JAX shines on accelerators. Let's train using the config in [config/train_shakespeare_char.py](config/train_shakespeare_char.py):

This project is set up to run locally by default, but you can easily switch to running on a remote GPU provided by [modal.com](https://modal.com).

<blockquote>⚠️ Before running the code (either locally or on the cloud), you need to be logged in to **modal.com**. This is because our script will try to attach a _Modal_ volume (similar to an S3 bucket, but free) regardless of the environment it is executed in.
<br /><br />
NOTE: This is something that comes along with _Modal_ and we are trying to remove this functionality when run on local (**PRs are welcome**)! </blockquote>

#### Steps

1. **Modal Setup**:  
   Before executing the code, you need to be logged in to Modal. To do this:

   ```bash
   modal setup
   ```

   This will prompt you to log in using your GitHub account. Follow the instructions on the terminal to authorize access. For more information, visit the [Modal.com Getting Started Guide](https://modal.com/docs/guide) for details.

2. **Run Locally or Remotely**:
   By default, running the following command will execute the training script on your local machine:

   ```bash
   modal run app.py
   ```

   If you prefer to run on a remote GPU using Modal's serverless GPUs (e.g., an A100 GPU), add the `--run-on-remote` option:

   ```bash
   modal run app.py --run-on-remote true
   ```

3. **Configuration File**:  
   Currently, the code doesn't directly accept a configuration file. However, you can modify parameters directly in the training script (`executables/train.py`) to customize the training process.

Make sure you are logged into [modal.com](https://modal.com) before executing the code by running `modal setup`.

**I only have a CPU**. No problem! JAX works great on CPU too. However, you may be limited to smaller parameter models due to memory/computation constraints. You may try a scaled-down version of the model by setting the following values in th `train.py` file.

```python
block_size=64
batch_size=12
n_layer=4
n_head=4
n_embd=128
max_iters=2000
lr_decay_iters=2000
```

## Sampling

> ⚠️ The sampling logic is available in the `sample.py` file in the root directory. However, it only supports remote sampling at the moment. While we intend to modify this to support both environments (again, PRs are welcome!), you may use them use that logic to setup your own sampling file.

<!-- ## training at scale

For serious deep learning professionals interested in training larger models, the repository supports multi-device training through JAX's excellent parallelism primitives. To train on multiple devices:

```bash
$ python train.py config/train_gpt2.py --devices=8
```

JAX automatically handles device parallelism - no DDP or FSDP needed! The code has been tested on configurations up to 8xA100 40GB GPUs. -->

## Model Architecture

The model implements a modern GPT architecture with a clean hierarchical structure:

```
GPT (eqx.Module)
├── TransformerLayer
│   ├── Embedding (wte)
│   ├── Embedding (wpe)
│   └── Block
│       ├── CausalSelfAttention
│       │   └── Linear
│       └── MLP
│           └── SwiGLU
```

Key components:

- `GPT`: The main model class containing the transformer and language model head
- `TransformerLayer`: Core component managing token/positional embeddings and transformer blocks
- `Block`: Structural component combining attention and MLP modules
- `CausalSelfAttention`: Implementation of masked self-attention with projection layers
- `MLP`: Feedforward network using projection layers and SwiGLU activation
- `SwiGLU`: Activation function with learnable parameters replacing traditional GELU

All components are implemented as Equinox modules, providing a PyTorch-like interface while leveraging JAX's functional programming model and computational benefits. This architecture maintains the core ideas of GPT while introducing modern improvements like SwiGLU activation and a cleaner organizational structure.

## Efficiency Notes

The code leverages several JAX optimizations:

- JIT compilation of training and inference loops
- Vectorization through `vmap`
- Automatic gradient computation with `grad`
- Efficient device parallelism
- PyTree-based parameter management through Equinox

## Todos

- Add model parallel training (DDP).
- When running locally, avoid trying to create a volume on _Modal_.
- Allow passing in a configuration file to overwrite default training values. Currently, you have to set them up in the `train.py` file.
- Modifying `sample.py` to allow sampling from either remote or local environment. Currently on remote is supported.

## Troubleshooting

The most common issues relate to JAX device placement and JIT compilation. If you're running into errors:

1. Ensure JAX can see your accelerators with `jax.devices()`
2. Try running without JIT using `--compile=False`
3. Check your JAX/CUDA version compatibility

For more context on transformers and language modeling, check out Andrej Karpathy's [Zero To Hero series](https://karpathy.ai/zero-to-hero.html).

For questions and discussions about this JAX implementation, feel free to open an issue!

## Acknowledgements

- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) for the original implementation
- We are also grateful for **Anh Tong** whose _Equinox_ version of _nanoGPT_ was a source of inspiration for our unique rewrite. We recommend referring to his version of nanoGPT as well here: **https://github.com/anh-tong/nanoGPT-equinox**
- The JAX team for an amazing framework
- The Equinox team for making JAX feel like PyTorch
- The Modal team for their effort in making serverless GPU usage accessible and affordable. Most importantly, for providing a free $30 credit for each workspace in your account.
