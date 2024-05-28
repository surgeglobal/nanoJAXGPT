"""
Sample from a trained model
"""

import modal
from modal import Image

# Initialize modal app
app = modal.App()

# Setup volume for storing model weights
volume = modal.Volume.from_name("pretraining-gpt2-matmul-prec")
MODEL_DIR = "/vol"

image = (
    Image.from_registry("thr3a/cuda12.1-torch")
    .pip_install("jax[cuda12]", "jaxlib", "cloudpickle", "tqdm", "equinox", "python-dotenv", "optax", "numpy", "tiktoken", gpu="A100")
)


@app.function(
    gpu="A100",
    timeout=86400, # Allow one day timout period
    image=image,
    mounts=[
        modal.Mount.from_local_dir(
            "./config",
            remote_path="/root/config"
        ),
        modal.Mount.from_local_dir(
            "./data",
            remote_path="/root/data"),
        modal.Mount.from_local_python_packages("helpers", "executables"),
    ],
    volumes={
        MODEL_DIR: volume
    },
)
def sample():
    import os
    import subprocess
    import cloudpickle
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import modal
    import tiktoken
    from executables.model import GPTConfig, GPT
    from functools import partial
    from tqdm import tqdm

    # -----------------------------------------------------------------------------
    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir = '/vol/models/gpt2/shakespeare' # ignored if init_from is not 'resume'

    start = """\n""" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"

    num_samples = 1 # number of samples to draw
    max_new_tokens = 100 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    # -----------------------------------------------------------------------------

    key = jax.random.PRNGKey(seed)
    jax.default_matmul_precision = "tensorfloat32"

    exec(open('config/train_shakespeare_char.py').read())

    checkpoint_params_file = os.path.join(out_dir, "params.pkl")
    checkpoint_file = os.path.join(out_dir, "model.eqx")

    with open(checkpoint_params_file, 'rb') as f:
        checkpoint_params = cloudpickle.load(f)

    gptconf = GPTConfig(**checkpoint_params['model_args'])
    model = GPT(gptconf, key=key)

    model = eqx.tree_deserialise_leaves(checkpoint_file, model)

    model = eqx.nn.inference_mode(model)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if 'config' in checkpoint_params and 'dataset' in checkpoint_params['config']: # older checkpoints might not have these...
        subprocess.run(["python", "prepare.py"], cwd=f'data/{checkpoint_params["config"]["dataset"]}')

        meta_path = os.path.join('data', checkpoint_params['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = cloudpickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i.item()] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    start_ids = encode(start)
    x = jnp.array(start_ids, dtype=jnp.int32)[None]

    def generate(model: GPT, token, key: jax.random.PRNGKey):
        generate_fn = partial(
            model.generate,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temperature
        )
        if token.shape[0] == 1:
            generated = generate_fn(token[0], key=key)
        else:
            key = jax.random.split(key, token.shape[0])
            generated = jax.vmap(generate_fn)(token, key=key)
        
        return decode(generated)

    for k in tqdm(range(num_samples), desc="samples"):
        sampling_key = jax.random.fold_in(key, k)

        generated = generate(model, x, sampling_key)
        print(generated)
        print('---------------')


@app.local_entrypoint()
def main():
    sample.remote()