from PIL import Image

from pathlib import Path

from app.clients.mock_clients import MockImageGenerationClient, MockVisionClient
from app.core.refinement_loop import RefinementSettings, _refine_prompt, run_refinement_loop
from app.models.schemas import PromptExtractionResult, SimilarityScore


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
