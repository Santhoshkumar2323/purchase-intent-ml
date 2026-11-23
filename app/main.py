from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.deps import load_artifacts
from app.routes import explain, health, predict
from src.errors import PipelineError
from src.logs import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts, app.state.load_error = None, None
    try:
        app.state.artifacts = load_artifacts()
    except (FileNotFoundError, KeyError, ValueError, PipelineError) as err:
        app.state.load_error = str(err)
        log.error("Starting without a model: %s", err)  
    yield


app = FastAPI(title="Purchase Intent API", version="1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)