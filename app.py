import modal
from modal import Image

app = modal.App()

s3_bucket_name = "pre-training"  # Bucket name not ARN.
s3_access_credentials = modal.Secret.from_dict({
    "AWS_ACCESS_KEY_ID": "jumvfkoev3sdchkpvcvs7trbvr6q",
    "AWS_SECRET_ACCESS_KEY": "jz5wyefsre7lv3b6s4cm3c3ltqmefuh3kuw354rm7a7q2jntmnejq",
})

image = (
    Image.from_registry("thr3a/cuda12.1-torch")
    .pip_install("jax[cuda12]", "jaxlib", "equinox", "optax", "tqdm", "python-dotenv", "numpy", "tiktoken", gpu="A100")
)


@app.function(
    gpu="A100",
    image=image,
    mounts=[
        modal.Mount.from_local_file(
            "./data/shakespeare/prepare.py",
            remote_path="/root/data/shakespeare/prepare.py"),
        modal.Mount.from_local_python_packages("helpers", "executables")
    ],
    volumes={
        "/bucket": modal.CloudBucketMount(s3_bucket_name, bucket_endpoint_url="https://gateway.storjshare.io",
                                          secret=s3_access_credentials),
    }
)
def run_train_on_modal():
    import subprocess
    import os
    subprocess.run(["python", "prepare.py"], cwd="data/shakespeare/")

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    from executables.train import train
    train()


@app.local_entrypoint()
def main():
    run_train_on_modal.remote()
