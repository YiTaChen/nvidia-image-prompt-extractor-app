from app.clients.nvidia_vlm_client import _parse_prompt_result


def test_parse_prompt_result_accepts_string_analysis():
    result = _parse_prompt_result(
        '{"prompt":"a prompt","negative_prompt":"blur","analysis":"short analysis text"}'
    )

    assert result.prompt == "a prompt"
    assert result.analysis == {"summary": "short analysis text"}


def test_parse_prompt_result_accepts_structured_human_subject_analysis():
    result = _parse_prompt_result(
        """
        {
          "prompt": "a prompt",
          "negative_prompt": "blur",
          "analysis": {
            "human_subjects": [
              {"hair_color": "black", "clothing": "beige suit", "pose": "walking hand in hand"}
            ]
          }
        }
        """
    )

    assert result.analysis["human_subjects"][0]["hair_color"] == "black"
