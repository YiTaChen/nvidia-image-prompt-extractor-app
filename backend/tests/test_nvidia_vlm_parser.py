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


def test_parse_prompt_result_enriches_prompt_with_structured_human_subjects():
    result = _parse_prompt_result(
        """
        {
          "prompt": "A couple walking on a city sidewalk.",
          "negative_prompt": "wrong people",
          "analysis": {
            "human_subjects": [
              {
                "visible_ethnicity": "East Asian",
                "hair_color": "long black hair",
                "clothing": "brown suit",
                "pose": "holding a white bouquet and walking hand in hand"
              },
              {
                "visible_ethnicity": "East Asian",
                "hair_color": "short black hair",
                "clothing": "beige suit",
                "pose": "walking hand in hand"
              }
            ]
          }
        }
        """
    )

    assert result.prompt.startswith("Foreground people details to match exactly:")
    assert "East Asian" in result.prompt
    assert "long black hair" in result.prompt
    assert "beige suit" in result.prompt


def test_parse_prompt_result_enriches_prompt_with_specific_action_fields():
    result = _parse_prompt_result(
        """
        {
          "prompt": "A couple walking on a city sidewalk.",
          "negative_prompt": "wrong people",
          "analysis": {
            "human_subjects": [
              {
                "body_orientation": "front-facing torso",
                "head_direction": "head turned back toward camera",
                "body_direction": "torso angled toward screen-right",
                "movement_direction": "moving toward screen-right",
                "walking_trajectory": "walking diagonally away from the camera along the sidewalk",
                "camera_relation": "faces look back at camera while bodies move away",
                "gaze_direction": "looking directly at camera",
                "left_hand": "holding white bouquet low at waist",
                "right_hand": "holding partner's left hand",
                "leg_action": "mid-stride walking toward camera",
                "hand_contact": "hands joined between both bodies",
                "held_objects": "white bouquet in woman's left hand",
                "relative_position": "woman on viewer-left"
              }
            ]
          }
        }
        """
    )

    assert "body orientation: front-facing torso" in result.prompt
    assert "head direction: head turned back toward camera" in result.prompt
    assert "body direction: torso angled toward screen-right" in result.prompt
    assert "movement direction: moving toward screen-right" in result.prompt
    assert "walking trajectory: walking diagonally away from the camera along the sidewalk" in result.prompt
    assert "camera relation: faces look back at camera while bodies move away" in result.prompt
    assert "gaze direction: looking directly at camera" in result.prompt
    assert "left hand: holding white bouquet low at waist" in result.prompt
    assert "right hand: holding partner's left hand" in result.prompt
    assert "leg action: mid-stride walking toward camera" in result.prompt
    assert "hand contact: hands joined between both bodies" in result.prompt


def test_parse_prompt_result_replaces_negative_prompt_that_restates_positive_scene():
    result = _parse_prompt_result(
        """
        {
          "prompt": "A couple with dark hair walking hand in hand on a city sidewalk.",
          "negative_prompt": "A couple with dark hair walking hand in hand on a city sidewalk, black and white.",
          "analysis": {}
        }
        """
    )

    assert not result.negative_prompt.startswith("A couple with dark hair")
    assert "wrong hair color" in result.negative_prompt
    assert "black and white" in result.negative_prompt
