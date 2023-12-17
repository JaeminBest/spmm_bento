from typing import Dict, List
import bentoml
from calc_property import calculate_property
from custom_model import get_model
import torch
import numpy as np


model = get_model("./checkpoint_SPMM.ckpt", device='cuda')

print(model(mode='p2s', p2s_prop=calculate_property('COc1cccc(NC(=O)CN(C)C(=O)COC(=O)c2cc(c3cccs3)nc3ccccc23)c1').numpy(), p2s_mask=np.zeros(53),
            stochastic=True, k=2, n_sample=5))
print("=" * 100)
print(model(mode='s2p', s2p_smiles='COc1cccc(NC(=O)CN(C)C(=O)COC(=O)c2cc(c3cccs3)nc3ccccc23)c1'))

print("save model start")
bentoml.pytorch.save_model(
    "spmm",   # Model name in the local Model Store
    model,  # Model instance being saved
)

def get_pv_and_mask(property_name: List[str], value: List[int]) -> Dict[str,np.ndarray]:
    p2i = dict()
    with open('./property_name.txt', 'r') as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        p2i[l.strip()] = i
    pv = np.zeros(53)
    mask = np.ones(53)
    assert len(property_name) == len(value)
    for i in range(len(property_name)):
        pv[p2i[property_name[i]]] = value[i]
        mask[p2i[property_name[i]]] = 0
    return dict(pv=pv, mask=mask)

saved_model = bentoml.picklable_model.save_model(
    'get_pv_and_mask',
    get_pv_and_mask,
    signatures={"__call__": {"batchable": False}}
)