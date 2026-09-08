"""Reuse canonical RGB PNG payloads without changing model-visible pixels."""

import base64
from io import BytesIO

from PIL import Image


def native_image_data(messages, rgb_image):
    """Reuse plain RGB PNGs; retain RGB re-encoding for other image formats.

    Inspect the source header, not the processor output: RGBA, palettes and
    metadata may affect a server's decoding, so those keep the old path.
    """
    urls = [
        item["image_url"]["url"]
        for message in messages
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if item.get("type") == "image_url"
    ]
    if len(urls) != 1:
        raise ValueError("native Pacman request requires exactly one image")
    prefix = "data:image/png;base64,"
    if not urls[0].startswith(prefix):
        raise ValueError("native Pacman vision request requires an inline PNG")
    encoded = urls[0][len(prefix):]
    with Image.open(BytesIO(base64.b64decode(encoded))) as source:
        if source.format == "PNG" and source.mode == "RGB" and not source.info:
            return [encoded]
    # Preserve the previous normalization for non-canonical inputs.
    from areal.utils.image import image2base64

    return image2base64(rgb_image)
