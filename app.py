import modal
from modal import Image

# Initialize modal app
app = modal.App()

# Setup volume for storing model weights
volume = modal.Volume.from_name("pretraining-gpt2-tinystories", create_if_missing=True)
MODEL_DIR = "/vol"

GPU = 'A100'

image = (
    Image.from_registry("thr3a/cuda12.1-torch")
    .pip_install("jax[cuda12]", "jaxlib", "equinox", "datasets", "optax", "tqdm", "python-dotenv", "numpy", "tiktoken", "wandb", gpu=GPU)
)


@app.function(
    gpu=GPU,
    timeout=86400, # Allow one day timout period
    image=image,
    mounts=[
        modal.Mount.from_local_dir(
            "./data/tinystories",
            remote_path="/root/data/tinystories"),
        modal.Mount.from_local_dir(
            "./config",
            remote_path="/root/config"
        ),
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
    subprocess.run(["python3", "prepare.py"], cwd="data/tinystories/")

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    from executables.train import train
    train()


@app.local_entrypoint()
def main(
    run_on_remote: bool = False
):
    if run_on_remote:
        run_train_on_modal.remote()
    else:
        run_train_on_modal.local()
