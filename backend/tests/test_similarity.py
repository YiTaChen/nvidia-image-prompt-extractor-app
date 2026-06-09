from PIL import Image

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
