import base64
from io import BytesIO
from pathlib import Path

from PIL import Image


ALLOWED_IMAGE_TYPES = {"jpeg", "png", "webp"}
MIME_BY_TYPE = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


class ImageValidationError(ValueError):
    pass


def detect_image_type(content: bytes) -> str:
    try:
        with Image.open(BytesIO(content)) as image:
            image_type = (image.format or "").lower()
    except Exception as exc:
        raise ImageValidationError("Unsupported image format. Use jpg, png, or webp.") from exc
    if image_type == "jpg":
        image_type = "jpeg"
    if image_type not in ALLOWED_IMAGE_TYPES:
        raise ImageValidationError("Unsupported image format. Use jpg, png, or webp.")
    return image_type


def load_normalized_image(content: bytes, max_side: int = 1600) -> Image.Image:
    detect_image_type(content)
    with Image.open(BytesIO(content)) as image:
        normalized = image.convert("RGB")
        normalized.thumbnail((max_side, max_side))
        return normalized.copy()


def image_to_png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def bytes_to_data_url(content: bytes) -> str:
    image_type = detect_image_type(content)
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{MIME_BY_TYPE[image_type]};base64,{encoded}"


def image_to_png_data_url(image: Image.Image) -> str:
    png_bytes = image_to_png_bytes(image)
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path
