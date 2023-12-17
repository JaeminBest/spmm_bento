import io
from typing import List, Optional
import bentoml
import numpy as np
from bentoml.io import Text, NumpyNdarray, JSON
from pydantic import BaseModel

spmm_model = bentoml.models.get("spmm:latest").to_runner()
get_pv_and_mask = bentoml.models.get("get_pv_and_mask:latest").to_runner()

svc = bentoml.Service(
    name="spmm", runners=[spmm_model,get_pv_and_mask]
)


@svc.api(input=Text(), output=NumpyNdarray())
async def SMILES_to_PV(smiles: str) -> np.ndarray:
    pv_numpy = await spmm_model.async_run(mode='s2p', s2p_smiles=smiles)
    return pv_numpy

class PV2SmilesDto(BaseModel):
    property_names: List[str]
    value: List[float]
    num_samples: Optional[int] = 5

@svc.api(input=JSON(pydantic_model=PV2SmilesDto), output=Text())
async def PV_to_SMILES(dto:PV2SmilesDto) -> str:
    property_name = dto.property_names
    value = dto.value
    d = await get_pv_and_mask.async_run(property_name, value)
    pv = d["pv"]
    mask = d["mask"]
    smiles_list = await spmm_model.async_run(mode='p2s', p2s_prop=pv, p2s_mask=mask, stochastic=True, k=2, n_sample=dto.num_samples)
    return ', '.join(smiles_list)
