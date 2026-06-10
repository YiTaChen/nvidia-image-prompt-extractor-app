from PIL import Image

from pathlib import Path

from app.clients.mock_clients import MockImageGenerationClient, MockVisionClient
from app.core.refinement_loop import RefinementSettings, _refine_prompt, run_refinement_loop
from app.models.schemas import ImageGenerationResult, PromptExtractionResult, SimilarityScore


FIXTURE = Path(__file__).parent / "fixtures" / "sample.png"


def test_refinement_loop_returns_attempt_history():
    result = run_refinement_loop(
        original_content=FIXTURE.read_bytes(),
        settings=RefinementSettings(threshold=0, max_iterations=3, width=256, height=256),
        vision_client=MockVisionClient(),
        image_client=MockImageGenerationClient(),
    )

    assert result.reached_threshold is True
    assert len(result.attempts) == 1
    assert result.final_prompt


def test_refinement_loop_respects_max_iterations():
    result = run_refinement_loop(
        original_content=FIXTURE.read_bytes(),
        settings=RefinementSettings(threshold=101, max_iterations=3, width=256, height=256),
        vision_client=MockVisionClient(),
        image_client=MockImageGenerationClient(),
    )

    assert result.reached_threshold is False
    assert len(result.attempts) == 3


def test_refine_prompt_adds_foreground_instruction_when_vlm_returns_same_prompt():
    class SamePromptVisionClient:
        def refine_prompt(
            self,
            original_image_data_url,
            generated_image,
            previous_prompt,
            previous_negative_prompt,
            similarity_report,
        ):
            return PromptExtractionResult(
                prompt=previous_prompt,
                negative_prompt=previous_negative_prompt,
                analysis={"summary": "no change"},
            )

    previous = PromptExtractionResult(prompt="A couple on a city sidewalk.", negative_prompt="", analysis={})
    updated = _refine_prompt(
        SamePromptVisionClient(),
        FIXTURE.read_bytes(),
        Image.new("RGB", (8, 8), (255, 255, 255)),
        previous,
        SimilarityScore(final_score=55, histogram_score=90, average_hash_score=80, critical_detail_score=45),
    )

    assert updated.prompt != previous.prompt
    assert "foreground-person fidelity" in updated.prompt
    assert "hair color" in updated.prompt


def test_refine_prompt_rejects_degraded_short_prompt():
    class ShortPromptVisionClient:
        def refine_prompt(
            self,
            original_image_data_url,
            generated_image,
            previous_prompt,
            previous_negative_prompt,
            similarity_report,
        ):
            return PromptExtractionResult(
                prompt="A couple walking on a sidewalk.",
                negative_prompt=previous_negative_prompt,
                analysis={"summary": "too short"},
            )

    previous = PromptExtractionResult(
        prompt=(
            "Global action direction to match exactly: body movement direction: diagonal_screen_right_away; "
            "walking trajectory: diagonally toward viewer-right and away from camera along the sidewalk; "
            "camera relation: faces and gaze look back toward camera while bodies move away. "
            "Foreground people details to match exactly: woman viewer-left holding bouquet in left hand, "
            "right hand holding man's hand, long dark hair, brown suit; man viewer-right, short dark hair, "
            "light beige suit, both smiling, brick building and black gate background."
        ),
        negative_prompt="wrong pose",
        analysis={},
    )

    updated = _refine_prompt(
        ShortPromptVisionClient(),
        FIXTURE.read_bytes(),
        Image.new("RGB", (8, 8), (255, 255, 255)),
        previous,
        SimilarityScore(final_score=55, histogram_score=90, average_hash_score=80, critical_detail_score=45),
    )

    assert updated.prompt != "A couple walking on a sidewalk."
    assert "diagonal_screen_right_away" in updated.prompt
    assert "viewer-right and away from camera" in updated.prompt
    assert "foreground-person fidelity" in updated.prompt


def test_refinement_loop_refines_from_best_attempt_when_latest_gets_worse(monkeypatch):
    class ScoredImageClient:
        def __init__(self):
            self.prompts = []

        def generate_image(self, request):
            self.prompts.append(request.prompt)
            image = Image.new("RGB", (256, 256), (len(self.prompts) * 20, 0, 0))
            import base64
            import io

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return ImageGenerationResult(
                image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
                mime_type="image/png",
                model="test",
            )

    class TrackingVisionClient:
        def __init__(self):
            self.refined_from = []

        def generate_initial_prompt(self, image_data_url):
            return PromptExtractionResult(prompt="best prompt diagonal_screen_right_away viewer-right", negative_prompt="")

        def refine_prompt(
            self,
            original_image_data_url,
            generated_image,
            previous_prompt,
            previous_negative_prompt,
            similarity_report,
        ):
            self.refined_from.append(previous_prompt)
            return PromptExtractionResult(prompt=f"{previous_prompt} refined", negative_prompt=previous_negative_prompt)

    image_client = ScoredImageClient()
    vision_client = TrackingVisionClient()
    scores = [
        SimilarityScore(final_score=70, histogram_score=70, average_hash_score=70),
        SimilarityScore(final_score=40, histogram_score=40, average_hash_score=40),
        SimilarityScore(final_score=35, histogram_score=35, average_hash_score=35),
    ]

    monkeypatch.setattr("app.core.refinement_loop.compare_images", lambda original, generated: scores.pop(0))

    run_refinement_loop(
        original_content=FIXTURE.read_bytes(),
        settings=RefinementSettings(threshold=101, max_iterations=3, width=256, height=256),
        vision_client=vision_client,
        image_client=image_client,
    )

    assert vision_client.refined_from[0] == "best prompt diagonal_screen_right_away viewer-right"
    assert vision_client.refined_from[1] == "best prompt diagonal_screen_right_away viewer-right"
