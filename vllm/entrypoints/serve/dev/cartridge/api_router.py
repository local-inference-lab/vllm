# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _require_cartridge_engine(request: Request) -> EngineClient:
    """Return the engine client if it supports EXL3 cartridge hot-swap.

    Cartridge hot-swap (drain, swap, resume) is implemented on the V1
    ``AsyncLLM`` engine only; it is not part of the generic ``EngineClient``
    protocol.
    """
    engine = engine_client(request)
    if not hasattr(engine, "load_exl3_cartridge"):
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
            detail="EXL3 cartridge hot-swap requires the V1 AsyncLLM engine",
        )
    return engine


@router.post("/load_exl3_cartridge")
async def load_exl3_cartridge(raw_request: Request) -> JSONResponse:
    """Drain in-flight requests, load an EXL3 MSRT cartridge, and resume.

    On failure the compressed base graphs are restored automatically before
    generation resumes; if the restore itself fails the engine shuts down.
    """
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {e}",
        ) from e
    adapter_path = body.get("adapter_path")
    if not adapter_path:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'adapter_path' in request body",
        )

    engine = _require_cartridge_engine(raw_request)
    try:
        updated_layers = await engine.load_exl3_cartridge(adapter_path)
    except (ValueError, RuntimeError) as err:
        return JSONResponse(
            content={"error": str(err)},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    return JSONResponse(
        content={"status": "loaded", "updated_layers": updated_layers},
        status_code=HTTPStatus.OK.value,
    )


@router.post("/deactivate_exl3_cartridge")
async def deactivate_exl3_cartridge(raw_request: Request) -> JSONResponse:
    """Drain in-flight requests, release the active cartridge, and resume.

    A no-op (returns zero updated layers per worker) if no cartridge is
    currently active.
    """
    engine = _require_cartridge_engine(raw_request)
    updated_layers = await engine.deactivate_exl3_cartridge()
    return JSONResponse(
        content={"status": "deactivated", "updated_layers": updated_layers},
        status_code=HTTPStatus.OK.value,
    )


@router.get("/exl3_cartridge_status")
async def exl3_cartridge_status(raw_request: Request) -> JSONResponse:
    """Return whether each worker currently has an active cartridge."""
    engine = _require_cartridge_engine(raw_request)
    active = await engine.collective_rpc("has_exl3_cartridge")
    return JSONResponse(content={"active": list(active)})


def attach_router(app: FastAPI):
    app.include_router(router)
