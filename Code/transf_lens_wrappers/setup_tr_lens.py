def import_tr_lens():
    import copy
    import dataclasses
    import itertools
    import os
    import random
    from functools import partial
    from pathlib import Path
    from typing import List, Optional, Union

    import datasets
    import einops
    import numpy as np
    import plotly.express as px
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import tqdm.auto as tqdm
    import transformer_lens
    import transformer_lens.utils as utils
    from fancy_einsum import einsum
    from IPython.display import HTML
    from torch.utils.data import DataLoader
    from transformer_lens import (
        ActivationCache,
        FactoredMatrix,
        HookedTransformer,
        HookedTransformerConfig,
    )
    from transformer_lens.hook_points import (  # Hooking utilities
        HookedRootModule,
        HookPoint,
    )
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    return


def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs
    ).show(renderer)


def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x": xaxis, "y": yaxis}, **kwargs).show(
        renderer
    )


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    ).show(renderer)
