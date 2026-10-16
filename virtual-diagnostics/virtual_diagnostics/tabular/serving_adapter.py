"""Serving-only adaptations for the frozen all-numerical, fixed-schema dataset."""

import torch
import sdm.processing as sp


def recipe_without_constant_drop(model, x):
    # The vendor DropConstantColumns assumes 2-D tables within each member.
    # All tasks here share x, and all 88 features vary in the locked context.
    assert x.ndim == 2 and x.shape[-1] == 88
    assert torch.isfinite(x).all() and (x != x[:1]).any(0).all()
    recipe = model.default_recipe()

    def replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, sp.DropConstantColumns):
                module._modules[name] = sp.EnsembleProcessor.as_processor(sp.Identity())
            else:
                replace(child)

    replace(recipe.features)
    return recipe
