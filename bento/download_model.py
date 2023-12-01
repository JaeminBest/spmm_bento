import bentoml
from calc_property import calculate_property
from custom_model import get_model
import torch
import numpy as np


model = get_model("./Pretrain/checkpoint_SPMM_20m.ckpt", device='cuda')

print(model(mode='p2s', p2s_prop=calculate_property('COc1cccc(NC(=O)CN(C)C(=O)COC(=O)c2cc(c3cccs3)nc3ccccc23)c1').numpy(), p2s_mask=np.zeros(53),
            stochastic=True, k=2, n_sample=5))
print("=" * 100)
print(model(mode='s2p', s2p_smiles='COc1cccc(NC(=O)CN(C)C(=O)COC(=O)c2cc(c3cccs3)nc3ccccc23)c1'))

print("save model start")
bentoml.pytorch.save_model(
    "spmm",   # Model name in the local Model Store
    model,  # Model instance being saved
)
