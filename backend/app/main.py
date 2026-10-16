from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import CORS_ORIGINS
from app.core.database import init_db
from app.features.face_landmark_vf50 import RUN_MODE as FACE_RUN_MODE
from app.features.face_landmark_vf50 import api as face_api
from app.features.face_landmark_vf50.services.run import execute_face_run
from app.routers import datasets, projects, runs, temporal, warnings
from app.features.human_pose.services.run import execute_run as execute_human_pose_run
from app.shared.jobs import register_run_handler

app = FastAPI(title="Landmark QA API", version="0.1.0")

for human_mode in ("K1", "K2", "K3", "K1_fallback"):
    register_run_handler(human_mode, execute_human_pose_run)
register_run_handler(FACE_RUN_MODE, execute_face_run)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(projects.router)
app.include_router(datasets.router)
app.include_router(runs.router)
app.include_router(warnings.router)
app.include_router(temporal.router)
app.include_router(face_api.router)
