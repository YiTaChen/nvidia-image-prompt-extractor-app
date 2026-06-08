from pathlib import Path

from app.core.image_io import bytes_to_data_url, load_normalized_image


FIXTURE = Path(__file__).parent / "fixtures" / "sample.png"


def test_load_normalized_image_from_fixture():
    image = load_normalized_image(FIXTURE.read_bytes())

    assert image.mode == "RGB"
    assert max(image.size) <= 1600


def test_bytes_to_data_url_uses_detected_mime():
    data_url = bytes_to_data_url(FIXTURE.read_bytes())

    assert data_url.startswith("data:image/png;base64,")
