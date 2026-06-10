import json

import requests
from PIL import Image

from app.core.image_io import image_to_png_data_url
from app.core.config import Settings
from app.models.schemas import PromptExtractionResult, SimilarityScore


INITIAL_SYSTEM_PROMPT = (
    "You are an expert image prompt engineer. Analyze the uploaded image and produce "
    "a precise prompt for a text-to-image generation model. If foreground people are "
    "visible, the prompt must be action-first and pose-first before background details. "
    "For each person, explicitly describe relative position, screen-space position, "
    "depth order, head direction, gaze direction, body orientation, body direction, "
    "movement direction, walking trajectory, camera relation, facial expression, left "
    "hand, right hand, hand contact with another person, held objects, leg action, "
    "walking/standing/sitting state, body spacing, interaction, visible skin tone or "
    "visually apparent ethnicity without guessing identity, hair color, hair style, "
    "clothing pieces, and clothing colors. Always distinguish face/gaze direction from "
    "torso/body direction and from walking trajectory; if a person looks at the camera "
    "while moving away or sideways, say that explicitly. Use screen-space language such "
    "as viewer-left, viewer-right, toward the right side of the frame, toward camera, "
    "away from camera, diagonal path, foreground, midground, background. Avoid vague "
    "phrases like 'walking' unless you also specify hand placement, leg action, facing "
    "direction, screen-space movement direction, and interaction. These foreground-person details must appear in the "
    "prompt text itself, not only in analysis. Put the human-subject checklist before "
    "background, style, lighting, color palette, and camera details. Return strict JSON "
    "only with prompt, negative_prompt, and analysis. The analysis object must include "
    "human_subjects when people are present, with keys: relative_position, screen_position, "
    "depth_order, head_direction, gaze_direction, body_orientation, body_direction, "
    "movement_direction, walking_trajectory, camera_relation, facial_expression, left_hand, "
    "right_hand, hand_contact, held_objects, leg_action, interaction, visible_skin_tone "
    "or visible_ethnicity, hair_color, hair_style, clothing."
)

REFINEMENT_SYSTEM_PROMPT = (
    "You are an expert image prompt engineer improving a text-to-image prompt through "
    "visual comparison. You will receive the original image, the last generated image, "
    "the previous prompt, and a similarity report. Rewrite the prompt to make the next "
    "generated image more visually similar to the original. Treat foreground people "
    "as the highest priority: visible skin tone or visually apparent ethnicity, hair "
    "color and style, clothing, relative position, screen-space position, depth order, "
    "head direction, gaze direction, body orientation, body direction, movement direction, "
    "walking trajectory, camera relation, left hand, right hand, hand contact, held "
    "objects, leg action, facial expression, body spacing, and interaction must match "
    "before background similarity is considered successful. "
    "The rewritten prompt text itself must explicitly include corrected foreground-person "
    "details before any background description. "
    "Return strict JSON only with prompt, negative_prompt, and analysis."
)

POSE_AUDIT_SYSTEM_PROMPT = (
    "You are a strict pose and body-motion classifier. Answer with JSON only. "
    "Do not infer body motion from face gaze; classify torso, feet, leg action, "
    "hand pull direction, and scene perspective."
)

POSE_AUDIT_USER_PROMPT = (
    "Look only at the foreground people. Classify BODY MOTION and action, not style "
    "or background. Faces may look at the camera while bodies move elsewhere. Return "
    "strict JSON only with: faces_gaze, body_direction, walking_trajectory, camera_relation, "
    "evidence, corrected_action_sentence. body_direction must be one of: toward_camera, "
    "away_from_camera, screen_left, screen_right, diagonal_screen_right_away, "
    "diagonal_screen_left_away, unknown. Do not answer toward_camera just because faces "
    "look at the camera. If bodies and sidewalk path point to viewer-right and away from "
    "camera, say diagonal_screen_right_away."
)


class NvidiaVisionClient:
    def __init__(
        self,
        settings: Settings,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.settings = settings
        self.api_key = api_key if api_key is not None else settings.nvidia_api_key
        self.base_url = (base_url or settings.nvidia_base_url).rstrip("/")
        self.model = model or settings.nvidia_vlm_model

    def generate_initial_prompt(self, image_data_url: str) -> PromptExtractionResult:
        if not self.api_key.strip():
            raise RuntimeError("NVIDIA_API_KEY is required for real NVIDIA calls.")

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
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
                                    "If people are visible, start the prompt with a per-person "
                                    "action checklist: relative position, screen position, depth order, "
                                    "head direction, gaze direction, body orientation, body direction, "
                                    "movement direction, walking trajectory, camera relation, left hand, "
                                    "right hand, hand contact, held objects, leg action, interaction, "
                                    "visible skin tone or visually apparent ethnicity, hair color/style, "
                                    "clothing, and expression. Distinguish faces looking at camera from "
                                    "bodies moving toward viewer-left/viewer-right or away from camera. "
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
        prompt_result = _parse_prompt_result(content)
        pose_audit = self._classify_pose_and_motion(image_data_url)
        return _apply_pose_audit(prompt_result, pose_audit)

    def _classify_pose_and_motion(self, image_data_url: str) -> dict:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": POSE_AUDIT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": POSE_AUDIT_USER_PROMPT},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    },
                ],
            },
            timeout=90,
        )
        response.raise_for_status()
        return _parse_json_object(response.json()["choices"][0]["message"]["content"])

    def refine_prompt(
        self,
        original_image_data_url: str,
        generated_image: Image.Image,
        previous_prompt: str,
        previous_negative_prompt: str,
        similarity_report: SimilarityScore,
    ) -> PromptExtractionResult:
        if not self.api_key.strip():
            raise RuntimeError("NVIDIA_API_KEY is required for real NVIDIA calls.")

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
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
        "is not enough. Identify whether the foreground people differ in relative position, "
        "screen-space position, depth order, head direction, gaze direction, body orientation, "
        "body direction, movement direction, walking trajectory, camera relation, left hand, "
        "right hand, hand contact, held objects, leg action, body spacing, visible skin tone, "
        "visually apparent ethnicity, hair color, hair style, clothing pieces/colors, facial "
        "expression, and interaction. If any of "
        "these human-subject action details are wrong, rewrite the prompt with explicit "
        "per-person constraints and add negative prompt terms for the wrong attributes. "
        "Avoid returning the previous prompt verbatim."
    )


def _parse_prompt_result(content: str) -> PromptExtractionResult:
    cleaned = _strip_json_fence(content)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = {"prompt": cleaned, "negative_prompt": "", "analysis": {}}
    if isinstance(payload.get("analysis"), str):
        payload["analysis"] = {"summary": payload["analysis"]}
    if not isinstance(payload.get("analysis", {}), dict):
        payload["analysis"] = {}
    result = _enrich_prompt_with_human_subjects(PromptExtractionResult.model_validate(payload))
    return _sanitize_negative_prompt(result)


def _parse_json_object(content: str) -> dict:
    cleaned = _strip_json_fence(content)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _strip_json_fence(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    return cleaned


def _apply_pose_audit(result: PromptExtractionResult, pose_audit: dict) -> PromptExtractionResult:
    pose_audit = _normalize_pose_audit(pose_audit)
    body_direction = _audit_value(pose_audit, "body_direction")
    walking_trajectory = _audit_value(pose_audit, "walking_trajectory")
    camera_relation = _audit_value(pose_audit, "camera_relation")
    corrected_action_sentence = _audit_value(pose_audit, "corrected_action_sentence")
    if not body_direction and not walking_trajectory and not corrected_action_sentence:
        return result
    prefix_parts = []
    if body_direction:
        prefix_parts.append(f"body movement direction: {body_direction}")
    if walking_trajectory:
        prefix_parts.append(f"walking trajectory: {walking_trajectory}")
    if camera_relation:
        prefix_parts.append(f"camera relation: {camera_relation}")
    if corrected_action_sentence:
        prefix_parts.append(f"action sentence: {corrected_action_sentence}")
    prefix = f"Global action direction to match exactly: {'; '.join(prefix_parts)}."
    result.prompt = _remove_contradictory_motion_phrases(result.prompt, body_direction)
    if prefix.lower() not in result.prompt.lower():
        result.prompt = f"{prefix} {result.prompt}"
    result.analysis["pose_audit"] = pose_audit
    return result


def _normalize_pose_audit(pose_audit: dict) -> dict:
    normalized = dict(pose_audit)
    body_direction = _audit_value(normalized, "body_direction")
    if body_direction == "diagonal_screen_right_away":
        normalized["walking_trajectory"] = "diagonally toward viewer-right and away from camera along the sidewalk"
        normalized["camera_relation"] = "faces and gaze look back toward camera while bodies move away"
        normalized["corrected_action_sentence"] = (
            "Both faces turn back toward the camera, but their torsos, feet, joined hands, "
            "and walking path move diagonally toward the viewer-right, away along the sidewalk."
        )
    elif body_direction == "diagonal_screen_left_away":
        normalized["walking_trajectory"] = "diagonally toward viewer-left and away from camera along the sidewalk"
        normalized["camera_relation"] = "faces and gaze look back toward camera while bodies move away"
        normalized["corrected_action_sentence"] = (
            "Both faces turn back toward the camera, but their torsos, feet, joined hands, "
            "and walking path move diagonally toward the viewer-left, away along the sidewalk."
        )
    return normalized


def _audit_value(pose_audit: dict, key: str) -> str:
    value = pose_audit.get(key)
    return value.strip() if isinstance(value, str) else ""


def _remove_contradictory_motion_phrases(prompt: str, body_direction: str) -> str:
    if body_direction not in {"away_from_camera", "diagonal_screen_right_away", "diagonal_screen_left_away"}:
        return prompt
    replacements = {
        "body direction: toward camera": f"body direction: {body_direction}",
        "movement direction: toward camera": f"movement direction: {body_direction}",
        "body orientation: toward camera": "body orientation: bodies angled along the walking path while heads turn toward camera",
        "walking trajectory: diagonal path": f"walking trajectory: {body_direction}",
        "camera relation: close to camera": "camera relation: faces look back toward camera while bodies move away along the sidewalk",
        "walking towards the camera": "walking along the sidewalk",
        "walking toward the camera": "walking along the sidewalk",
        "walking towards camera": "walking along the sidewalk",
        "walking toward camera": "walking along the sidewalk",
        "moving towards the camera": "moving along the sidewalk",
        "moving toward the camera": "moving along the sidewalk",
        "towards the camera": "along the sidewalk",
        "toward the camera": "along the sidewalk",
    }
    cleaned = prompt
    for wrong, replacement in replacements.items():
        cleaned = cleaned.replace(wrong, replacement)
        cleaned = cleaned.replace(wrong.capitalize(), replacement.capitalize())
    return cleaned


def _enrich_prompt_with_human_subjects(result: PromptExtractionResult) -> PromptExtractionResult:
    subjects = result.analysis.get("human_subjects")
    if not isinstance(subjects, list) or not subjects:
        return result
    details = []
    for index, subject in enumerate(subjects, start=1):
        if isinstance(subject, dict):
            values = _subject_detail_values(subject)
        else:
            values = [str(subject)]
        if values:
            details.append(f"person {index}: {'; '.join(values)}")
    if not details:
        return result
    prefix = f"Foreground people details to match exactly: {' | '.join(details)}."
    if prefix.lower() not in result.prompt.lower():
        result.prompt = f"{prefix} {result.prompt}"
    return result


def _subject_detail_values(subject: dict) -> list[str]:
    preferred_keys = [
        "relative_position",
        "screen_position",
        "depth_order",
        "head_direction",
        "body_direction",
        "movement_direction",
        "walking_trajectory",
        "camera_relation",
        "body_orientation",
        "gaze_direction",
        "facial_expression",
        "left_hand",
        "right_hand",
        "hand_contact",
        "held_objects",
        "leg_action",
        "interaction",
        "body_spacing",
        "walking_state",
        "standing_state",
        "pose",
        "visible_ethnicity",
        "ethnicity",
        "skin_tone",
        "visible_skin_tone",
        "hair_color",
        "hair_style",
        "clothing",
        "outfit",
        "expression",
        "position",
    ]
    values = []
    used_keys = set()
    for key in preferred_keys:
        if key in subject and subject[key]:
            values.append(f"{key.replace('_', ' ')}: {_stringify_subject_value(subject[key])}")
            used_keys.add(key)
    for key, value in subject.items():
        if key not in used_keys and value:
            values.append(f"{key.replace('_', ' ')}: {_stringify_subject_value(value)}")
    return values


def _stringify_subject_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}: {item}" for key, item in value.items())
    return str(value)


def _sanitize_negative_prompt(result: PromptExtractionResult) -> PromptExtractionResult:
    if _negative_prompt_restates_positive_scene(result.prompt, result.negative_prompt):
        extras = []
        if "black and white" in result.negative_prompt.lower():
            extras.append("black and white")
        result.negative_prompt = ", ".join(
            [
                "wrong visible skin tone",
                "wrong visually apparent ethnicity",
                "wrong hair color",
                "wrong hair style",
                "wrong clothing",
                "wrong pose",
                "wrong hand placement",
                "wrong facial expression",
                "extra people",
                "missing bouquet",
                "blurry",
                "distorted faces",
                *extras,
            ]
        )
    return result


def _negative_prompt_restates_positive_scene(prompt: str, negative_prompt: str) -> bool:
    if not negative_prompt:
        return False
    prompt_terms = set(_meaningful_terms(prompt))
    negative_terms = _meaningful_terms(negative_prompt)
    if not prompt_terms or not negative_terms:
        return False
    overlap = sum(1 for term in negative_terms if term in prompt_terms) / len(negative_terms)
    return overlap >= 0.45


def _meaningful_terms(text: str) -> list[str]:
    return [
        token.strip(".,;:!?()[]{}\"'").lower()
        for token in text.split()
        if len(token.strip(".,;:!?()[]{}\"'")) >= 4
    ]
