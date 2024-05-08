from typing import List, Callable
import jax
import equinox as eqx
import helpers.init as init

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

    return initialized_model

def initialize_eqx_module(Module, config, key):
    key1, key2 = jax.random.split(key, 2)

    inst = Module(config, key)
    new_inst = _init_weights(inst, key1)

    return new_inst
