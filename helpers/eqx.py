import jax
import equinox as eqx

from typing import Callable

def named_parameters(model: eqx.Module):
    out = []

    for path, p in jax.tree_util.tree_flatten_with_path(eqx.filter(model, eqx.is_array))[0]:
        pn = ''

        for index in range(len(path)):
            if isinstance(path[index], str):  # Check if path[index] is a string
                pn += '.' + path[index]
            else:
                pn += str(path[index])

        out.append((pn[1:], p))
    
    return out


def find_sub_tree(model: eqx.Module, sub_tree_name: str, filter_fn: Callable = None):
    out = []
    for path, p in jax.tree_util.tree_flatten_with_path(model, is_leaf=filter_fn)[0]:
        pn = ''
    
        for index in range(len(path)):
            if isinstance(path[index], jax._src.tree_util.DictKey):
                pn += '.' + path[index].key
            else:
                pn += str(path[index])
    
        if filter_fn:
            if filter_fn(p) and pn.endswith(sub_tree_name):
                out.append(p)
        elif pn.endswith(sub_tree_name):
            out.append(p)
    
    return out
