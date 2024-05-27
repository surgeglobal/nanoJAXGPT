import modal
from modal import Image, Volume

# Initialize modal app
app = modal.App()

# Setup volume for storing model weights
volume = modal.Volume.from_name("pretraining-gpt2", create_if_missing=True)
MODEL_DIR = "/vol"

image = (
    Image.from_registry("thr3a/cuda12.1-torch")
    .pip_install("jax[cuda12]", "jaxlib", "equinox", "optax", "tqdm", "python-dotenv", "numpy", "tiktoken", "wandb", gpu="A100")
)


@app.function(
    gpu="A100",
    timeout=86400, # Allow one day timout period
    image=image,
    mounts=[
        modal.Mount.from_local_file(
            "./data/shakespeare_char/prepare.py",
            remote_path="/root/data/shakespeare_char/prepare.py"),
        modal.Mount.from_local_python_packages("helpers", "executables"),
    ],
    volumes={
        MODEL_DIR: volume
    },
    _allow_background_volume_commits=True
)
def run_train_on_modal():
    import subprocess
    import os
    subprocess.run(["python", "prepare.py"], cwd="data/shakespeare_char/")

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    from executables.train import train
    exec(open('config/train_shakespeare_char.py').read()) # overrides from command line or config file
    train()


@app.local_entrypoint()
def main():
    run_train_on_modal.remote()
