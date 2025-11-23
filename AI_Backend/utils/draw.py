from PIL import Image, ImageDraw, ImageFont

def draw_boxes(pil_img, detections, box_color=(255,0,0), width=3):
    """
    pil_img: PIL.Image (RGB)
    detections: list of {"label", "confidence", "bbox":[x1,y1,x2,y2]}
    """
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = d.get("label", "obj")
        conf = d.get("confidence", 0.0)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=width)
        text = f"{label} {conf:.2f}"
        text_size = draw.textsize(text, font=font)
        draw.rectangle([x1, max(0, y1 - text_size[1] - 4), x1 + text_size[0] + 4, y1], fill=box_color)
        draw.text((x1 + 2, max(0, y1 - text_size[1] - 2)), text, fill=(255,255,255), font=font)
    return pil_img
