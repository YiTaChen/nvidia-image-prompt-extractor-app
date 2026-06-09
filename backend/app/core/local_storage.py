import base64
from pathlib import Path

from PIL import Image

from app.core.config import PROJECT_ROOT
from app.core.image_io import image_to_png_bytes, load_normalized_image


STORAGE_ROOT = PROJECT_ROOT / "backend" / "app" / "storage" / "jobs"


def job_dir(job_id: str) -> Path:
    return STORAGE_ROOT / job_id


def save_original_image(job_id: str, content: bytes) -> Path:
    image = load_normalized_image(content)
    path = job_dir(job_id) / "original.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_to_png_bytes(image))
    return path


def save_generated_image(job_id: str, iteration: int, image_base64: str, mime_type: str) -> Path:
    extension = "jpg" if mime_type == "image/jpeg" else "png"
    path = job_dir(job_id) / "generated" / f"{iteration:03d}.{extension}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(image_base64))
    return path


def path_for_response(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))
