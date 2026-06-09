from io import BytesIO

from PIL import Image, ImageChops, ImageStat

from app.models.schemas import SimilarityScore


def compare_images(original: Image.Image, generated: Image.Image) -> SimilarityScore:
    original_rgb = _normalized_rgb(original)
    generated_rgb = _normalized_rgb(generated)
    histogram_score = _histogram_similarity(original_rgb, generated_rgb)
    average_hash_score = _average_hash_similarity(original_rgb, generated_rgb)
    final_score = round((0.6 * histogram_score) + (0.4 * average_hash_score), 2)
    return SimilarityScore(
        final_score=final_score,
        histogram_score=round(histogram_score, 2),
        average_hash_score=round(average_hash_score, 2),
    )


def image_from_base64_bytes(content: bytes) -> Image.Image:
    with Image.open(BytesIO(content)) as image:
        return image.convert("RGB")


def _normalized_rgb(image: Image.Image) -> Image.Image:
    normalized = image.convert("RGB")
    normalized.thumbnail((512, 512))
    canvas = Image.new("RGB", (512, 512), (255, 255, 255))
    x = (512 - normalized.width) // 2
    y = (512 - normalized.height) // 2
    canvas.paste(normalized, (x, y))
    return canvas


def _histogram_similarity(original: Image.Image, generated: Image.Image) -> float:
    original_hist = original.histogram()
    generated_hist = generated.histogram()
    distance = sum(abs(a - b) for a, b in zip(original_hist, generated_hist))
    max_distance = original.width * original.height * 2 * 3
    return max(0.0, 100.0 * (1.0 - (distance / max_distance)))


def _average_hash_similarity(original: Image.Image, generated: Image.Image) -> float:
    original_hash = _average_hash(original)
    generated_hash = _average_hash(generated)
    same_bits = sum(1 for a, b in zip(original_hash, generated_hash) if a == b)
    return 100.0 * same_bits / len(original_hash)


def _average_hash(image: Image.Image) -> tuple[int, ...]:
    grayscale = image.convert("L").resize((8, 8))
    values = list(grayscale.getdata())
    avg = ImageStat.Stat(grayscale).mean[0]
    return tuple(1 if value >= avg else 0 for value in values)
