# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Annotated, Literal

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm.engine.protocol import EngineClient

router = APIRouter()


class PrefillFairnessRequest(BaseModel):
    """Complete replacement configuration for prefill fairness."""

    fairness_engine: Literal["compute_share", "micro_slicing"] | None = None
    prefill_compute_share: Annotated[float | None, Field(gt=0.0, lt=1.0)] = None
    max_num_prefill_tokens_per_step: Annotated[int, Field(ge=0)] = 0
    max_num_partial_prefills: Annotated[int, Field(ge=0)] = 0
    decode_prefill_min_decode_steps: Annotated[int, Field(ge=0)] = 0
    decode_prefill_max_wait_ms: Annotated[int, Field(ge=0)] = 0


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/prefill_fairness")
async def get_prefill_fairness(raw_request: Request):
    """Return the active fairness engine and all tuning parameters."""
    config = await engine_client(raw_request).get_prefill_fairness()
    return JSONResponse(content=config)


@router.post("/prefill_fairness")
async def set_prefill_fairness(raw_request: Request, config: PrefillFairnessRequest):
    """Switch policy at idle without reloading weights or clearing caches."""
    result = await engine_client(raw_request).set_prefill_fairness(config.model_dump())
    status_code = 200
    if not result["applied"]:
        status_code = 409 if result["reason"] == "busy" else 422
    return JSONResponse(content=result, status_code=status_code)


def attach_router(app: FastAPI):
    app.include_router(router)
