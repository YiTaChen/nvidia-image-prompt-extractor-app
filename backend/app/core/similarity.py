from io import BytesIO

from PIL import Image, ImageChops, ImageFilter, ImageStat

from app.models.schemas import SimilarityScore


def compare_images(original: Image.Image, generated: Image.Image) -> SimilarityScore:
    original_rgb = _normalized_rgb(original)
    generated_rgb = _normalized_rgb(generated)
    histogram_score = _histogram_similarity(original_rgb, generated_rgb)
    average_hash_score = _average_hash_similarity(original_rgb, generated_rgb)
    original_subject = _subject_region(original_rgb)
    generated_subject = _subject_region(generated_rgb)
    subject_histogram_score = _histogram_similarity(original_subject, generated_subject)
    subject_hash_score = _average_hash_similarity(original_subject, generated_subject)
    subject_layout_score = _block_color_layout_similarity(original_subject, generated_subject)
    edge_layout_score = _edge_layout_similarity(original_subject, generated_subject)
    critical_detail_score = _critical_detail_similarity(_person_core_region(original_rgb), _person_core_region(generated_rgb))
    final_score = round(
        (0.10 * histogram_score)
        + (0.05 * average_hash_score)
        + (0.10 * subject_histogram_score)
        + (0.05 * subject_hash_score)
        + (0.10 * subject_layout_score)
        + (0.05 * edge_layout_score)
        + (0.55 * critical_detail_score),
        2,
    )
    return SimilarityScore(
        final_score=final_score,
        histogram_score=round(histogram_score, 2),
        average_hash_score=round(average_hash_score, 2),
        subject_histogram_score=round(subject_histogram_score, 2),
        subject_hash_score=round(subject_hash_score, 2),
        subject_layout_score=round(subject_layout_score, 2),
        edge_layout_score=round(edge_layout_score, 2),
        critical_detail_score=round(critical_detail_score, 2),
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


def _subject_region(image: Image.Image) -> Image.Image:
    width, height = image.size
    return image.crop(
        (
            int(width * 0.16),
            int(height * 0.05),
            int(width * 0.84),
            int(height * 0.98),
        )
    )


def _person_core_region(image: Image.Image) -> Image.Image:
    width, height = image.size
    return image.crop(
        (
            int(width * 0.25),
            int(height * 0.12),
            int(width * 0.75),
            int(height * 0.96),
        )
    )


def _block_color_layout_similarity(original: Image.Image, generated: Image.Image, grid_size: int = 8) -> float:
    original_small = original.resize((grid_size, grid_size), Image.Resampling.BILINEAR)
    generated_small = generated.resize((grid_size, grid_size), Image.Resampling.BILINEAR)
    distances = []
    for original_pixel, generated_pixel in zip(original_small.getdata(), generated_small.getdata()):
        distance = sum(abs(a - b) for a, b in zip(original_pixel, generated_pixel)) / 3
        distances.append(distance)
    mean_distance = sum(distances) / len(distances)
    return max(0.0, 100.0 * (1.0 - (mean_distance / 255.0)))


def _critical_detail_similarity(original: Image.Image, generated: Image.Image, grid_size: int = 8) -> float:
    original_small = original.resize((grid_size, grid_size), Image.Resampling.BILINEAR)
    generated_small = generated.resize((grid_size, grid_size), Image.Resampling.BILINEAR)
    block_scores = []
    for original_pixel, generated_pixel in zip(original_small.getdata(), generated_small.getdata()):
        distance = sum(abs(a - b) for a, b in zip(original_pixel, generated_pixel)) / 3
        block_scores.append(max(0.0, 100.0 * (1.0 - (distance / 255.0))))
    block_scores.sort()
    critical_count = max(1, len(block_scores) // 8)
    return sum(block_scores[:critical_count]) / critical_count


def _edge_layout_similarity(original: Image.Image, generated: Image.Image) -> float:
    original_edges = original.convert("L").resize((64, 64)).filter(ImageFilter.FIND_EDGES)
    generated_edges = generated.convert("L").resize((64, 64)).filter(ImageFilter.FIND_EDGES)
    diff = ImageChops.difference(original_edges, generated_edges)
    mean_distance = ImageStat.Stat(diff).mean[0]
    return max(0.0, 100.0 * (1.0 - (mean_distance / 255.0)))
