import bentoml
import numpy as np
from bentoml.io import Text, NumpyNdarray, Multipart

spmm_model = bentoml.models.get("spmm:latest").to_runner()

svc = bentoml.Service(
    name="spmm", runners=[spmm_model]
)


@svc.api(input=Text(), output=NumpyNdarray())
async def SMILES_to_PV(smiles: str) -> np.ndarray:
    pv_numpy = await spmm_model.async_run(mode='s2p', s2p_smiles=smiles)
    return pv_numpy


@svc.api(input=Multipart(pv=NumpyNdarray(), mask=NumpyNdarray()), output=Text())
async def PV_to_SMILES(pv: np.ndarray, mask: np.ndarray) -> str:
    smiles_list = await spmm_model.async_run(mode='p2s', p2s_prop=pv, p2s_mask=mask, stochastic=True, k=2, n_sample=5)
    return ', '.join(smiles_list)
