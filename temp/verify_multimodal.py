import requests, io, json
from PIL import Image, ImageDraw, ImageFont

def test(text, label):
    img = Image.new('RGB', (600, 300), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((20, 60), "FIRST LEVEL", fill='white')
    d.text((20, 130), text, fill='white')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    r = requests.post('http://127.0.0.1:8000/predict', files={'file': ('t.jpg', buf, 'image/jpeg')})
    data = r.json()
    print(f"\n=== {label} ===")
    print(f"VERDICT:   {data.get('classification')}")
    print(f"CONFIDENCE: {data.get('confidence')}")
    print(f"VISUAL:    {data.get('visual_analysis')}")
    print(f"TEXT:      {data.get('text_analysis')}")
    print(f"OCR TEXT:  {data.get('extracted_text', '')[:80]}")

test(
    "Your current abilities might seem limited and disappointing but dont worry; you'll soon be pleased enough",
    "SEXUALIZED MEME (FIRST LEVEL)"
)
test(
    "I am black and I was the president of the USA, you loser",
    "RACIST MEME (Obama)"
)
test(
    "Have a great day, everyone!",
    "SAFE MEME"
)
