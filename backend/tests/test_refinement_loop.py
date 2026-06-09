from pathlib import Path

from app.clients.mock_clients import MockImageGenerationClient, MockVisionClient
from app.core.refinement_loop import RefinementSettings, run_refinement_loop


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
