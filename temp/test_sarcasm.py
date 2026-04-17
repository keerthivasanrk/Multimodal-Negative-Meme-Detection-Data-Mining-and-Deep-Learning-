import requests, io
from PIL import Image, ImageDraw

def test(text):
    img = Image.new('RGB', (600, 200), color='white')
    d = ImageDraw.Draw(img)
    d.text((10, 80), text, fill='black')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    try:
        resp = requests.post('http://127.0.0.1:8000/predict', files={'file': ('t.jpg', buf, 'image/jpeg')}, timeout=10)
        data = resp.json()
        print(f'Text: {text}')
        print('Result:', data.get('classification'), data.get('text_analysis', {}).get('reason'))
    except Exception as e:
        print(f"Failed for text '{text}': {e}")
    print('-'*40)

print("Starting tests...")
test('Look at all this vibrant diversity')
test('These immigrants are mostly peaceful')
test('You are a sick man for thinking that')
print("Finished.")
