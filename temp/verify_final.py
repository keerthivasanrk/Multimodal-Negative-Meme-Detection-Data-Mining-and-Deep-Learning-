import requests, io
from PIL import Image, ImageDraw

def test(text):
    img_buf = io.BytesIO()
    img = Image.new('RGB', (800, 200), color='white')
    d = ImageDraw.Draw(img)
    d.text((10, 80), text, fill='black')
    img.save(img_buf, format='JPEG')
    img_buf.seek(0)
    
    r = requests.post('http://127.0.0.1:8000/predict', files={'file': ('t.jpg', img_buf, 'image/jpeg')})
    print(f"TEXT: {text[:40]}...")
    print(f"RESULT: {r.json()['classification']}")
    print("-" * 20)

test('TAM BLACK AND I WAS THE PRESIDENT OF THE USA; YOU LOOSER')
test('Me explaining the deep lore of JRR Tolkein | The prostitute I am paying')
test('Puppies playing in the park')
