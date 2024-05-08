"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import jax
from jax import lax
import jax.numpy as jnp
import jax.nn as nn
import jax.tree_util as jtu
import equinox as eqx
import optax

from helpers import init

from typing import Callable

class SwiGLU(eqx.Module):
    """
    Implementation of the SwiGLU activation function in the paper by Noam Shazeer at Google

    References:
        GLU Variants Improve Transformer paper  : https://arxiv.org/abs/2002.05202
        Aziz et al. Paper Summaries             : https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
    """

    W: jax.Array
    V: jax.Array
    b: jax.Array
    c: jax.Array

    def __init__(self, dim_in, dim_out, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        self.W = jax.random.normal(k1, (dim_in, dim_out))
        self.V = jax.random.normal(k2, (dim_in, dim_out))
        self.b = jax.random.normal(k3, (dim_out,))
        self.c = jax.random.normal(k4, (dim_out,))

    def __call__(self, x):
        return nn.swish(jnp.dot(x, self.W) + self.b) * (jnp.dot(x, self.V) + self.c)

class CausalSelfAttention(eqx.Module):
    c_attn: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    attn_dropout: eqx.nn.Dropout
    resid_dropout: eqx.nn.Dropout

    def __init__(self, config, key):
        assert config.n_embd % config.n_head == 0

        # PRNGKey
        lkey1, lkey2 = jax.random.split(key, 2)

        # key, query, value projections for all heads, but in a batch
        self.c_attn = eqx.nn.Linear(config.n_embd, 3 * config.n_embd, use_bias=config.bias, key=lkey1)
        # output projection
        self.c_proj = eqx.nn.Linear(config.n_embd, config.n_embd, use_bias=config.bias, key=lkey2)
        # regularization
        self.attn_dropout = eqx.nn.Dropout(config.dropout)
        self.resid_dropout = eqx.nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # Has been made a buffer by using lax.stop_gradient whenever it is used.
        # Immutability calls for reshape, plus there is no view for jnp (or numpy) arrays.
        self.bias = jnp.tril(jnp.ones(config.block_size, config.block_size)).reshape(1, 1, config.block_size,
                                                                                     config.block_size)

    def __call__(self, x):
        T, C = jnp.shape(x)  # sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = jnp.split(self.c_attn(x), self.n_embd, axis=2)
        # Immutability calls for reshape, plus there is no view for jnp (or numpy) arrays.
        k = jnp.swapaxes(k.reshape(T, self.n_head, C // self.n_head), 0, 1)  # (B, nh, T, hs)
        q = jnp.swapaxes(q.reshape(T, self.n_head, C // self.n_head), 0, 1)  # (B, nh, T, hs)
        v = jnp.swapaxes(v.reshape(T, self.n_head, C // self.n_head), 0, 1)  # (B, nh, T, hs)

        # manual implementation of attention
        att = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / math.sqrt(jnp.shape(k)[-1])
        # Note: Added the stop_gradient just to be safe, I see no update rule acting on the bias inside the
        # forward pass.
        att = jnp.where(lax.stop_gradient(self.bias[:, :, :T, :T]) == 0, float('-inf'), att)
        att = jax.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = jnp.matmul(att, v)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # Reshaping with Immutability creates a new copy
        y = jnp.swapaxes(y, 1, 2).reshape(T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(eqx.Module):
    c_fc: eqx.nn.Linear
    swiglu: SwiGLU
    c_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, config, key):
        lkey1, lkey2, skey = jax.random.split(key, 3)

        self.c_fc = eqx.nn.Linear(config.n_embd, 4 * config.n_embd, use_bias=config.bias, key=lkey1)
        self.swiglu = SwiGLU(4 * config.n_embd, 4 * config.n_embd, skey)
        self.c_proj = eqx.nn.Linear(4 * config.n_embd, config.n_embd, use_bias=config.bias, key=lkey2)
        self.dropout = eqx.nn.Dropout(config.dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(eqx.Module):
    ln_1: eqx.nn.LayerNorm
    ln_2: eqx.nn.LayerNorm

    def __init__(self, config, key):
        ckey, mkey = jax.random.split(key, 2)

        self.ln_1 = eqx.nn.LayerNorm(config.n_embd, use_bias=config.bias)
        self.attn = CausalSelfAttention(config, ckey)
        self.ln_2 = eqx.nn.LayerNorm(config.n_embd, use_bias=config.bias)
        self.mlp = MLP(config, mkey)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(eqx.Module):

    def __init__(self, config, key):
        ekey1, ekey2, lmhkey, key4 = jax.random.split(key, 4)

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = dict(
            wte     = eqx.nn.Embedding(config.vocab_size, config.n_embd, key=ekey1),
            wpe     = eqx.nn.Embedding(config.block_size, config.n_embd, key=ekey2),
            drop    = eqx.nn.Dropout(config.dropout),
            h       = [Block(config, key) for _ in range(config.n_layer)],
            ln_f    = eqx.nn.LayerNorm(config.n_embd, bias=config.bias),
        )
        self.lm_head = eqx.nn.Linear(config.n_embd, config.vocab_size, use_bias=False, key=lmhkey)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer["wte"].weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        model = self._apply(self._init_weights, key)
        eqx.apply_updates(self, model)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for path, p in jax.tree_util.tree_flatten_with_path(self)[0]:
            pn = ''

            for index in range(len(path)):
                if isinstance(path[index], jax._src.tree_util.DictKey):
                    pn += '.' + path[index].key
                else:
                    pn += '.' + path[index].name

            if pn.endswith('c_proj.weight'):
                init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer), key=key4)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(jnp.prod(p.shape) for p in jax.tree_util.tree_leaves(self))
        if non_embedding:
            n_params -= jnp.prod(self.transformer["wpe"].weight.shape)
        return n_params
    
    def _apply(self, fun, key):
        return fun(self, key)

    @staticmethod
    def _init_weights(model: eqx.Module, key: jax.random.PRNGKey):
        def init_layer(model, is_layer: Callable, mean: float, std: float):
            get_weights = lambda m: [x.weight
                                     for x in jax.tree_util.tree_leaves(m, is_leaf=is_layer)
                                     if is_layer(x)]
            weights = get_weights(model)

            new_weights = [init.normal_(weight, mean=mean, std=std, key=subkey)
                           for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]

            return eqx.tree_at(get_weights, model, new_weights)

        def init_linear(model):
            is_linear = lambda x: isinstance(x, eqx.nn.Linear)

            model = init_layer(model, is_linear, mean=0.0, std=0.2)

            get_biases = lambda m: [x.bias
                                    for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                                    if is_linear(x) and x.bias is not None]
            biases = get_biases(model)

            new_biases = [init.zeros_(bias) for bias in biases]

            return eqx.tree_at(get_biases, model, new_biases)

        def init_embedding(model):
            is_embedding = lambda x: isinstance(x, eqx.nn.Embedding)

            return init_layer(model, is_embedding, mean=0.0, std=0.2)

        initialized_model = init_linear(model)
        initialized_model = init_embedding(initialized_model)

        eqx.apply_updates(model, initialized_model)

    def __call__(self, idx, targets=None):
        b, t = idx.shape # TODO Check what idx is sent
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = jnp.expand_dims(jnp.arange(0, t, dtype=jnp.int64), 0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer["wte"](idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer["wpe"](pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer["drop"](tok_emb + pos_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = optax.softmax_cross_entropy(logits.reshape((-1, logits.shape[-1])), targets.reshape(-1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer["wpe"].weight = self.transformer["wpe"].weight[:block_size]
        for block in self.transformer["h"]:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, key, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config, key)
        # TODO: Complete this module from here onwards
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (eqx.nn.Linear,)
        blacklist_weight_modules = (eqx.nn.LayerNorm, eqx.nn.LayerNorm, eqx.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        idx = jax.lax.stop_gradient(idx)

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = lax.top_k(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # sample from the distribution
            idx_next = jax.random.categorical(jax.random.PRNGKey(0), logits, shape=(1, ))
            # append sampled index to the running sequence and continue
            idx = jnp.concatenate((idx, idx_next), axis=1)

        return idx
