import base64
from dataclasses import dataclass

from PIL import Image

from app.core.image_io import bytes_to_data_url, load_normalized_image
from app.core.similarity import compare_images, image_from_base64_bytes
from app.models.schemas import ImageGenerationRequest, RefinementAttempt, RefinementResult


@dataclass
class RefinementSettings:
    threshold: float
    max_iterations: int
    width: int = 1024
    height: int = 1024


def run_refinement_loop(
    original_content: bytes,
    settings: RefinementSettings,
    vision_client,
    image_client,
) -> RefinementResult:
    original_image = load_normalized_image(original_content)
    prompt_result = vision_client.generate_initial_prompt(bytes_to_data_url(original_content))
    attempts: list[RefinementAttempt] = []

    for iteration in range(1, settings.max_iterations + 1):
        generated = image_client.generate_image(
            ImageGenerationRequest(
                prompt=prompt_result.prompt,
                negative_prompt=prompt_result.negative_prompt,
                width=settings.width,
                height=settings.height,
            )
        )
        generated_image = _image_from_generation_result(generated.image_base64)
        score = compare_images(original_image, generated_image)
        attempts.append(
            RefinementAttempt(
                iteration=iteration,
                prompt=prompt_result.prompt,
                negative_prompt=prompt_result.negative_prompt,
                generated_image_base64=generated.image_base64,
                generated_image_mime_type=generated.mime_type,
                score=score,
            )
        )
        if score.final_score >= settings.threshold:
            break
        prompt_result = _refine_prompt(vision_client, original_content, generated_image, prompt_result, score)

    best_attempt = max(attempts, key=lambda attempt: attempt.score.final_score)
    return RefinementResult(
        reached_threshold=best_attempt.score.final_score >= settings.threshold,
        threshold=settings.threshold,
        max_iterations=settings.max_iterations,
        best_score=best_attempt.score.final_score,
        final_prompt=best_attempt.prompt,
        attempts=attempts,
    )


def _image_from_generation_result(image_base64: str) -> Image.Image:
    return image_from_base64_bytes(base64.b64decode(image_base64))


def _refine_prompt(vision_client, original_content: bytes, generated_image: Image.Image, prompt_result, score):
    if hasattr(vision_client, "refine_prompt"):
        refined = vision_client.refine_prompt(
            original_image_data_url=bytes_to_data_url(original_content),
            generated_image=generated_image,
            previous_prompt=prompt_result.prompt,
            previous_negative_prompt=prompt_result.negative_prompt,
            similarity_report=score,
        )
        if _same_prompt(refined.prompt, prompt_result.prompt):
            refined.prompt = _append_foreground_fidelity_instruction(refined.prompt, score)
            refined.negative_prompt = _append_foreground_negative_prompt(refined.negative_prompt)
        return refined
    prompt_result.prompt = (
        f"{prompt_result.prompt}. Improve visual similarity to the source image. "
        f"Previous similarity score was {score.final_score:.1f}."
    )
    return prompt_result


def _same_prompt(next_prompt: str, previous_prompt: str) -> bool:
    return " ".join(next_prompt.lower().split()) == " ".join(previous_prompt.lower().split())


def _append_foreground_fidelity_instruction(prompt: str, score) -> str:
    return (
        f"{prompt} Add a foreground-person fidelity correction: explicitly match the original "
        "people's visible skin tone or visually apparent ethnicity, hair color and style, "
        "clothing pieces and colors, hand placement, walking/standing pose, facial expression, "
        "body spacing, and interaction before optimizing the background. "
        f"The previous critical foreground detail score was {score.critical_detail_score:.1f}."
    )


def _append_foreground_negative_prompt(negative_prompt: str) -> str:
    foreground_negative = (
        "wrong visible skin tone, wrong visually apparent ethnicity, wrong hair color, "
        "wrong hair style, wrong clothing, wrong pose, wrong hand placement, wrong facial expression"
    )
    if not negative_prompt:
        return foreground_negative
    if foreground_negative in negative_prompt:
        return negative_prompt
    return f"{negative_prompt}, {foreground_negative}"
