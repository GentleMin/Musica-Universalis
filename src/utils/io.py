# -*- coding: utf-8 -*-


import os, h5py
import numpy as np


def tree_h5(gp: h5py.Group | h5py.Dataset, dep: int | None = None):

    gp_name = gp.name.split('/')[-1]
    gp_list = list()

    if isinstance(gp, h5py.Dataset):
        gp_list.append((gp_name, gp.shape))
        return gp_list

    gp_list.append((gp_name, len(gp.keys())))
    if dep is None or dep > 0:
        dep_next = None if dep is None else dep - 1
        gp_children = [tree_h5(gp[child], dep_next) for child in gp]
        gp_list.append(gp_children)
    return gp_list


def tree_h5_file(file: str, dep: int | None = None):

    with h5py.File(file, 'r') as fp:
        tree = tree_h5(fp, dep=dep)
    tree[0] = (file, tree[0][1])
    return tree



INDENT = '    '
HANGER = '|   '
PT_MID = '├── '
PT_END = '└── '

def tree_str(tree: list, prefix: str = '', is_last: bool = False):

    item_str = PT_END if is_last else PT_MID

    if len(tree) == 1:
        node = tree[0]
        if isinstance(node[1], int):
            node_str = '{} with {} subgroups'.format(node[0], node[1])
        else:
            node_str = '{} with shape {}'.format(node[0], node[1])
        o_str = prefix + item_str + node_str + '\n'
        return o_str
    
    o_str = prefix + item_str + tree[0][0] + '\n'
    indent = INDENT if is_last else HANGER
    prefix_next = prefix + indent
    for i_child in range(len(tree[1])):
        child = tree[1][i_child]
        is_last_next = ((i_child + 1) == len(tree[1]))
        o_str += tree_str(child, prefix_next, is_last_next)
    
    return o_str
    
