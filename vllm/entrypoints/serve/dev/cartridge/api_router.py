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
    """Return the engine client attached to the application.

    Args:
        request: Incoming request whose application owns the engine client.

    Returns:
        The application's engine client.
    """
    return request.app.state.engine_client


def _require_cartridge_engine(request: Request) -> EngineClient:
    """Return the engine client if it supports EXL3 cartridge hot-swap.

    Cartridge hot-swap (drain, swap, resume) is implemented on the V1
    ``AsyncLLM`` engine only; it is not part of the generic ``EngineClient``
    protocol.

    Args:
        request: Incoming request whose application owns the engine client.

    Returns:
        The cartridge-capable engine client.

    Raises:
        HTTPException: If the engine does not support cartridge hot-swap.
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

    Args:
        raw_request: Request containing an ``adapter_path`` JSON field.

    Returns:
        A response containing the load status and per-worker layer counts.

    Raises:
        HTTPException: If the JSON body is malformed, ``adapter_path`` is
            missing, or the engine does not support cartridge hot-swap.
    """
    try:
        body = await raw_request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("Invalid JSON for EXL3 cartridge load")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Invalid JSON request body",
        ) from e
    adapter_path = body.get("adapter_path") if isinstance(body, dict) else None
    if not isinstance(adapter_path, str) or not adapter_path:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'adapter_path' in request body",
        )

    engine = _require_cartridge_engine(raw_request)
    try:
        updated_layers = await engine.load_exl3_cartridge(adapter_path)
    except (ValueError, RuntimeError, TimeoutError):
        logger.exception("EXL3 cartridge validation failed")
        return JSONResponse(
            content={"error": "EXL3 cartridge validation failed"},
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

    Args:
        raw_request: Incoming deactivation request.

    Returns:
        A response containing the deactivation status and per-worker counts.

    Raises:
        HTTPException: If the engine does not support cartridge hot-swap.
    """
    engine = _require_cartridge_engine(raw_request)
    updated_layers = await engine.deactivate_exl3_cartridge()
    return JSONResponse(
        content={"status": "deactivated", "updated_layers": updated_layers},
        status_code=HTTPStatus.OK.value,
    )


@router.get("/exl3_cartridge_status")
async def exl3_cartridge_status(raw_request: Request) -> JSONResponse:
    """Return whether each worker currently has an active cartridge.

    Args:
        raw_request: Incoming status request.

    Returns:
        A response containing each worker's active-cartridge state.

    Raises:
        HTTPException: If the engine does not support cartridge hot-swap.
    """
    engine = _require_cartridge_engine(raw_request)
    active = await engine.collective_rpc("has_exl3_cartridge")
    return JSONResponse(content={"active": list(active)})


def attach_router(app: FastAPI) -> None:
    """Attach the EXL3 cartridge development routes to an application.

    Args:
        app: FastAPI application to receive the routes.

    Returns:
        None.
    """
    app.include_router(router)
