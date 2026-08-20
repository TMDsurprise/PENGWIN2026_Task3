"""Grand Challenge invoke API for the deterministic e27 conservative model."""

from __future__ import annotations

import threading
import traceback

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from process import init_models, run


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
models = init_models()
invoke_lock = threading.Lock()


@app.get("/health")
def health():
    return Response(status_code=200)


@app.post("/invoke")
def invoke():
    try:
        with invoke_lock:
            run(models)
        return Response(status_code=201)
    except Exception as error:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": type(error).__name__, "message": str(error)},
        )
