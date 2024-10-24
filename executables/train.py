"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import cloudpickle
import random
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx

from executables.model import GPTConfig, GPT

from dotenv import load_dotenv

jax.config.update("jax_enable_x64", True)
jax.default_matmul_precision = "tensorfloat32"

# -----------------------------------------------------------------------------
# loading .env config

load_dotenv()

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O

# Get the current date and time
now = datetime.now()

# Format it as a string
timestamp = now.strftime("%Y%m%d_%H%M%S")

# Setup volume for storing model weights
if os.getenv("MODAL_ENVIRONMENT") == "remote":
    out_dir = "/vol"
else:
    out_dir = "./vol"


directories = [name for name in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, name))]
if len(directories) == 1:
    resume_dir = os.path.join(out_dir, directories[0])
else:
    print("There is more than one directory or no directories in the specified directory.")
    exit()

eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'resume'  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True  # disabled by default
wandb_project = 'tinystories'
wandb_run_name = 'gpt2'  # 'run' + str(time.time())
# data
dataset = 'tinystories'
batch_size = 16
block_size = 1024  ### If block size is different to 1024, changes need to be made in GPT.crop_block_size() method
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
# dtype = 'bfloat16' if jax.default_backend() != 'cpu' else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'float32'
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('config/train_shakespeare_char.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# DDP has not been implemented in this train script. We are open for contributions.
random_seed = 1337
seed_offset = 0

tokens_per_iter = batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)

np.random.seed(random_seed + seed_offset)
key = jax.random.PRNGKey(random_seed + seed_offset)
train_key, gpt_key = jax.random.split(key)

# poor man's data loader
data_dir = os.path.join('data', dataset)


def get_batch(split: str):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'validation.bin'), dtype=np.uint16, mode='r')

    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = jnp.stack([jnp.array(data[i:i + block_size], dtype=jnp.int64) for i in ix])
    y = jnp.stack([jnp.array(data[i + 1:i + 1 + block_size], dtype=jnp.int64) for i in ix])

    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = cloudpickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout
)  # start with model_args from command line


def convert_model_to_dtype():
    global model
    def convert_pytree_to_dtype(pytree, dtype):
        def _convert(leaf):
            if eqx.is_array(leaf):
                return leaf.astype(dtype)
            else:
                return leaf
    
        return jax.tree_util.tree_map(_convert, pytree)
    
    
    if dtype == 'bfloat16':
        model = convert_pytree_to_dtype(model, jnp.bfloat16)
    elif dtype == 'float16':
        model = convert_pytree_to_dtype(model, jnp.float16)
    elif dtype == 'float32':
        model = convert_pytree_to_dtype(model, jnp.float32)


if init_from == 'resume':
    print(f"Resuming training from {resume_dir}")
    # resume training from a checkpoint.
    checkpoint_params_file = os.path.join(resume_dir, "params.pkl")
    checkpoint_file = os.path.join(resume_dir, "model.eqx")
    with open(checkpoint_params_file, 'rb') as f:
        checkpoint_params = cloudpickle.load(f)
    gptconf = checkpoint_params['model_args']
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    gptconf.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
    model = GPT(gptconf, key=gpt_key)
    convert_model_to_dtype()
    model = eqx.tree_deserialise_leaves(checkpoint_file, model)
    iter_num = checkpoint_params['iter_num']
    best_val_loss = checkpoint_params['val_loss']
    learning_rate = checkpoint_params['learning_rate']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
else:
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, gpt_key)
    convert_model_to_dtype()


# # crop down the model block size if desired, using model surgery ## NOT IMPLEMENTED YET
# if block_size < model.config.block_size:
#     model.crop_block_size(block_size)
#     model_args['block_size'] = block_size  # so that the checkpoint will have the right value


# optimizer
# learning rate decay scheduler (cosine with warmup)
lr_scheduler = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=learning_rate,
    warmup_steps=warmup_iters if init_from == 'scratch' else 0,
    decay_steps=lr_decay_iters - iter_num,
    end_value=min_lr,
)
optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=lr_scheduler, b1=beta1, b2=beta2)

checkpoint_params = None  # free up memory


@eqx.filter_jit
def compute_loss(model, x, y):
    logits = jax.vmap(model, in_axes=(0, None))(x, True)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, # B, T, C
        labels=y, # B, T
    )

    return jnp.mean(loss)


@eqx.filter_jit
def make_step(
        model,
        optimizer_state,
        x,
        y
):
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, x, y)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, model)
    model = eqx.apply_updates(model, updates)
    # print(model.transformer.h[0].mlp.c_fc.weight, model.transformer.h[0].mlp.c_fc.weight.mean())
    return model, optimizer_state, loss


def estimate_loss(model):
    out = {}
    model = eqx.nn.inference_mode(model)
    for split in ['train', 'val']:
        losses = jnp.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            loss = compute_loss(model, jax.lax.stop_gradient(x), y)
            losses = losses.at[k].set(loss.item())
        out[split] = jnp.mean(losses)
    return out


# logging
if wandb_log:
    import wandb

    wandb.login(key='d35eb3616c9549c90972c0d35b1efcc3b6af528f')
    run = wandb.init(project=wandb_project, name=f"{wandb_run_name}-{timestamp}", config=config)


def train():
    global model
    global best_val_loss
    global optimizer_state_file
    global iter_num
    
    # Initialize optimizer state with filtered model
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))
    if init_from == 'resume':
        optimizer_state.hyperparams['learning_rate'] = jnp.array(learning_rate, dtype=jnp.float32)    

    t0 = time.time()
    for local_iter_num in range(iter_num, max_iters):
        x, y = get_batch("train")

        # evaluate the loss on train/val sets and write checkpoints
        if (local_iter_num % eval_interval) == 0 or (local_iter_num == max_iters - 1):
            losses = estimate_loss(model)
            lr = optimizer_state.hyperparams['learning_rate'].item()

            print(f"step {local_iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.4e}")
            print(f"best val loss {best_val_loss}")

            if wandb_log:
                wandb.log({
                    "iter": local_iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                })

            # Checkpointing
            checkpoint_params = {
                "model_args": gptconf,
                "iter_num": local_iter_num,
                "val_loss": losses["val"],
                "learning_rate": lr,
                "config": config,
            }
            print(checkpoint_params)
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if local_iter_num > 0:
                    os.makedirs(out_dir, exist_ok=True)
                    checkpoint_file = os.path.join(out_dir, 'model.eqx')
                    checkpoint_params_file = os.path.join(out_dir, 'params.pkl')
                    # optimizer_state_file = os.path.join(out_dir, "optimizer_state.eqx")

                    eqx.tree_serialise_leaves(checkpoint_file, model)
                    # eqx.tree_serialise_leaves(optimizer_state_file, optimizer_state.inner_state)
                    
                    with open(checkpoint_params_file, "wb") as f:
                        cloudpickle.dump(checkpoint_params, f)
                    print(f"save best checkpoint to {out_dir}")

            if always_save_checkpoint:
                # Save checkpoint to resume training
                latest_checkpoint_dir = os.path.join(out_dir, f"checkpoint-{local_iter_num}")
                os.makedirs(latest_checkpoint_dir, exist_ok=True)
                latest_checkpoint_file = os.path.join(latest_checkpoint_dir, 'model.eqx')
                latest_checkpoint_params_file = os.path.join(latest_checkpoint_dir, 'params.pkl')
    
                eqx.tree_serialise_leaves(latest_checkpoint_file, model)
                
                with open(latest_checkpoint_params_file, "wb") as f:
                    cloudpickle.dump(checkpoint_params, f)
                print(f"save checkpoint to {latest_checkpoint_dir}")
            

        if local_iter_num == 0 and eval_only:
            break

        # do a training step
        model, optimizer_state, loss = make_step(model, optimizer_state, x, y)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if local_iter_num % log_interval == 0:
            print(f"iter {local_iter_num} \t loss: {loss.item():.4f} \t time {dt*1000:.2f}ms")
            if wandb_log:
                wandb.log({"train_iter": local_iter_num, "train_loss": loss.item()})
