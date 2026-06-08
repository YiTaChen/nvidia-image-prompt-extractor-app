import base64
from io import BytesIO

from PIL import Image, ImageDraw

from app.models.schemas import ImageGenerationRequest, ImageGenerationResult, PromptExtractionResult


class MockVisionClient:
    def generate_initial_prompt(self, image_data_url: str) -> PromptExtractionResult:
        return PromptExtractionResult(
            prompt=(
                "A detailed, faithful recreation of the uploaded image, preserving "
                "the main subject, composition, lighting, color palette, and texture."
            ),
            negative_prompt="blurry, distorted, low detail, incorrect composition",
            analysis={
                "subject": "uploaded image subject",
                "composition": "match the original framing",
                "lighting": "match original lighting",
                "style": "photorealistic",
            },
        )


class MockImageGenerationClient:
    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        image = Image.new("RGB", (request.width, request.height), (238, 241, 245))
        draw = ImageDraw.Draw(image)
        draw.rectangle((32, 32, request.width - 32, request.height - 32), outline=(43, 76, 126), width=8)
        draw.text((56, 56), request.prompt[:160], fill=(20, 27, 38))
        output = BytesIO()
        image.save(output, format="PNG")
        return ImageGenerationResult(
            image_base64=base64.b64encode(output.getvalue()).decode("ascii"),
            mime_type="image/png",
            model="mock-image-generator",
        )
