import json

import requests
from PIL import Image

from app.core.image_io import image_to_png_data_url
from app.core.config import Settings
from app.models.schemas import PromptExtractionResult, SimilarityScore


INITIAL_SYSTEM_PROMPT = (
    "You are an expert image prompt engineer. Analyze the uploaded image and produce "
    "a precise prompt for a text-to-image generation model. Preserve subject, "
    "composition, style, lighting, color palette, camera details, and important "
    "visual details. If foreground people are visible, prioritize them over the "
    "background: describe the number of people, visible skin tone or visually apparent "
    "ethnicity without guessing identity, hair color, hair style, clothing pieces and "
    "colors, pose, body orientation, facial expression, interaction, and relative "
    "position. These foreground-person details must appear in the prompt text itself, "
    "not only in analysis. Put the human-subject checklist before background details. "
    "Return strict JSON only with prompt, negative_prompt, and analysis. The analysis "
    "object should include a human_subjects key when people are present."
)

REFINEMENT_SYSTEM_PROMPT = (
    "You are an expert image prompt engineer improving a text-to-image prompt through "
    "visual comparison. You will receive the original image, the last generated image, "
    "the previous prompt, and a similarity report. Rewrite the prompt to make the next "
    "generated image more visually similar to the original. Treat foreground people "
    "as the highest priority: visible skin tone or visually apparent ethnicity, hair "
    "color and style, clothing, pose, facial expression, hand placement, body spacing, "
    "and interaction must match before background similarity is considered successful. "
    "The rewritten prompt text itself must explicitly include corrected foreground-person "
    "details before any background description. "
    "Return strict JSON only with prompt, negative_prompt, and analysis."
)


class NvidiaVisionClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_initial_prompt(self, image_data_url: str) -> PromptExtractionResult:
        if not self.settings.has_nvidia_key:
            raise RuntimeError("NVIDIA_API_KEY is required for real NVIDIA calls.")

        response = requests.post(
            f"{self.settings.nvidia_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.nvidia_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.nvidia_vlm_model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": INITIAL_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Create a detailed image-generation prompt for this image. "
                                    "If people are visible, the prompt must explicitly describe "
                                    "their visible skin tone or visually apparent ethnicity, hair "
                                    "color/style, clothing, pose, expression, and interaction. "
                                    "Return JSON only."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    },
                ],
            },
            timeout=90,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_prompt_result(content)

    def refine_prompt(
        self,
        original_image_data_url: str,
        generated_image: Image.Image,
        previous_prompt: str,
        previous_negative_prompt: str,
        similarity_report: SimilarityScore,
    ) -> PromptExtractionResult:
        if not self.settings.has_nvidia_key:
            raise RuntimeError("NVIDIA_API_KEY is required for real NVIDIA calls.")

        response = requests.post(
            f"{self.settings.nvidia_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.nvidia_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.nvidia_vlm_model,
                "temperature": 0.25,
                "messages": [
                    {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _build_refinement_instruction(
                                    previous_prompt=previous_prompt,
                                    previous_negative_prompt=previous_negative_prompt,
                                    similarity_report=similarity_report,
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": original_image_data_url}},
                            {"type": "image_url", "image_url": {"url": image_to_png_data_url(generated_image)}},
                        ],
                    },
                ],
            },
            timeout=90,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_prompt_result(content)


def _build_refinement_instruction(
    previous_prompt: str,
    previous_negative_prompt: str,
    similarity_report: SimilarityScore,
) -> str:
    return (
        "The last generated image did not meet the similarity threshold.\n\n"
        f"Previous prompt:\n{previous_prompt}\n\n"
        f"Previous negative prompt:\n{previous_negative_prompt or '(none)'}\n\n"
        "Similarity report:\n"
        f"- final_score: {similarity_report.final_score}\n"
        f"- histogram_score: {similarity_report.histogram_score}\n"
        f"- average_hash_score: {similarity_report.average_hash_score}\n\n"
        f"- subject_histogram_score: {similarity_report.subject_histogram_score}\n"
        f"- subject_hash_score: {similarity_report.subject_hash_score}\n"
        f"- subject_layout_score: {similarity_report.subject_layout_score}\n"
        f"- edge_layout_score: {similarity_report.edge_layout_score}\n"
        f"- critical_detail_score: {similarity_report.critical_detail_score}\n\n"
        "Compare the original image and the generated image. Background similarity alone "
        "is not enough. Identify whether the foreground people differ in visible skin tone, "
        "visually apparent ethnicity, hair color, hair style, clothing pieces/colors, pose, "
        "gesture, body orientation, facial expression, and interaction. If any of these "
        "human-subject details are wrong, rewrite the prompt with explicit constraints for "
        "each person and add negative prompt terms for the wrong attributes. Avoid returning "
        "the previous prompt verbatim."
    )


def _parse_prompt_result(content: str) -> PromptExtractionResult:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = {"prompt": cleaned, "negative_prompt": "", "analysis": {}}
    if isinstance(payload.get("analysis"), str):
        payload["analysis"] = {"summary": payload["analysis"]}
    if not isinstance(payload.get("analysis", {}), dict):
        payload["analysis"] = {}
    return PromptExtractionResult.model_validate(payload)
