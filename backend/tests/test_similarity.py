from PIL import Image
from PIL import ImageDraw

from app.core.similarity import compare_images


def test_identical_images_score_high():
    image = Image.new("RGB", (64, 64), (20, 120, 70))

    score = compare_images(image, image)

    assert score.final_score == 100


def test_different_images_score_lower():
    green = Image.new("RGB", (64, 64), (20, 120, 70))
    red = Image.new("RGB", (64, 64), (220, 40, 40))

    score = compare_images(green, red)

    assert score.final_score < 100


def test_same_background_with_wrong_people_scores_much_lower_than_matching_people():
    original = _street_scene(
        left_hair=(20, 20, 20),
        right_hair=(30, 24, 18),
        left_clothes=(230, 210, 170),
        right_clothes=(235, 215, 185),
        left_pose="holding_hands",
        right_pose="holding_hands",
    )
    matching = _street_scene(
        left_hair=(20, 20, 20),
        right_hair=(30, 24, 18),
        left_clothes=(230, 210, 170),
        right_clothes=(235, 215, 185),
        left_pose="holding_hands",
        right_pose="holding_hands",
    )
    wrong_people_same_background = _street_scene(
        left_hair=(230, 220, 80),
        right_hair=(220, 40, 40),
        left_clothes=(25, 45, 130),
        right_clothes=(40, 150, 70),
        left_pose="arms_up",
        right_pose="arms_down",
    )

    matching_score = compare_images(original, matching)
    wrong_score = compare_images(original, wrong_people_same_background)

    assert matching_score.final_score == 100
    assert wrong_score.final_score < 80
    assert matching_score.final_score - wrong_score.final_score > 20


def _street_scene(
    left_hair: tuple[int, int, int],
    right_hair: tuple[int, int, int],
    left_clothes: tuple[int, int, int],
    right_clothes: tuple[int, int, int],
    left_pose: str,
    right_pose: str,
) -> Image.Image:
    image = Image.new("RGB", (512, 512), (205, 188, 165))
    draw = ImageDraw.Draw(image)
    for x in range(0, 512, 48):
        draw.line((x, 0, x, 330), fill=(155, 120, 100), width=2)
    for y in range(0, 330, 36):
        draw.line((0, y, 512, y), fill=(155, 120, 100), width=2)
    draw.rectangle((0, 330, 512, 512), fill=(120, 118, 112))
    draw.rectangle((340, 60, 455, 118), fill=(175, 25, 30))
    draw.rectangle((48, 135, 464, 158), fill=(40, 42, 45))

    _draw_person(draw, 205, 278, left_hair, left_clothes, left_pose, flip=False)
    _draw_person(draw, 300, 278, right_hair, right_clothes, right_pose, flip=True)
    return image


def _draw_person(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    hair: tuple[int, int, int],
    clothes: tuple[int, int, int],
    pose: str,
    flip: bool,
) -> None:
    skin = (205, 150, 115)
    draw.ellipse((x - 20, y - 88, x + 20, y - 48), fill=skin)
    draw.pieslice((x - 24, y - 96, x + 24, y - 56), start=180, end=360, fill=hair)
    draw.rounded_rectangle((x - 30, y - 48, x + 30, y + 58), radius=12, fill=clothes)
    draw.line((x - 16, y + 58, x - 30, y + 125), fill=(35, 35, 38), width=10)
    draw.line((x + 16, y + 58, x + 30, y + 125), fill=(35, 35, 38), width=10)
    if pose == "holding_hands":
        hand_x = x + (52 if not flip else -52)
        draw.line((x + (24 if not flip else -24), y - 22, hand_x, y - 2), fill=skin, width=9)
    elif pose == "arms_up":
        draw.line((x - 25, y - 28, x - 55, y - 92), fill=skin, width=9)
        draw.line((x + 25, y - 28, x + 55, y - 92), fill=skin, width=9)
    else:
        draw.line((x - 25, y - 22, x - 52, y + 38), fill=skin, width=9)
        draw.line((x + 25, y - 22, x + 52, y + 38), fill=skin, width=9)
