from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_generation import router as generation_router
from app.api.routes_jobs import router as jobs_router
from app.api.routes_prompt import router as prompt_router
from app.api.routes_vlm import router as vlm_router


app = FastAPI(title="Visual Prompt Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prompt_router)
app.include_router(generation_router)
app.include_router(jobs_router)
app.include_router(vlm_router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
