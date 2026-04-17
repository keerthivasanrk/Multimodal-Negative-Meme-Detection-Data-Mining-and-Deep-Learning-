"""
Batch accuracy test for the multimodal meme moderation system.

Datasets:
  - HATEFUL: C:/Users/keert/Downloads/datasets/TRAINING  (10001 images, all hateful)
  - SAFE:    C:/Users/keert/Downloads/meme/good meme     (5 images, all safe)

Sends images to the live API at http://127.0.0.1:8000/predict
and measures classification accuracy.
"""
import os
import random
import requests
import time
from pathlib import Path

API_URL = "http://127.0.0.1:8000/predict"
HATEFUL_DIR = r"C:\Users\keert\Downloads\datasets\TRAINING"
SAFE_DIR    = r"C:\Users\keert\Downloads\meme\good meme"

# Sample sizes (keep small to run in reasonable time)
N_HATEFUL = 30
N_SAFE    = 5   # use all safe memes

def get_images(folder, n=None, extensions={".jpg",".jpeg",".png",".webp"}):
    files = [f for f in Path(folder).iterdir()
             if f.is_file() and f.suffix.lower() in extensions]
    if n and len(files) > n:
        files = random.sample(files, n)
    return files

def predict(image_path):
    with open(image_path, "rb") as f:
        resp = requests.post(API_URL, files={"file": (image_path.name, f, "image/jpeg")}, timeout=60)
    return resp.json()

def run_test():
    random.seed(42)
    hateful_imgs = get_images(HATEFUL_DIR, N_HATEFUL)
    safe_imgs    = get_images(SAFE_DIR)

    results = {"HATEFUL": {"correct": 0, "wrong": 0, "details": []},
               "SAFE":    {"correct": 0, "wrong": 0, "details": []}}

    print(f"\nTesting {len(hateful_imgs)} hateful + {len(safe_imgs)} safe memes...\n")

    # ---- Test hateful images ----
    print("=== HATEFUL MEMES (expected: HATEFUL) ===")
    for img in hateful_imgs:
        try:
            r = predict(img)
            label = r.get("classification", "ERROR")
            conf  = r.get("confidence", 0)
            correct = label == "HATEFUL"
            results["HATEFUL"]["correct" if correct else "wrong"] += 1
            status = "✓" if correct else "✗"
            vis = r.get("visual_analysis", {})
            txt = r.get("text_analysis", {})
            detail = {
                "file": img.name, "predicted": label, "confidence": conf,
                "visual": vis.get("harmful"), "text": txt.get("harmful"),
                "text_reason": txt.get("reason","")
            }
            results["HATEFUL"]["details"].append(detail)
            print(f"  {status} {img.name:20s}  →  {label:8s}  conf={conf:.2f}  vis={str(vis.get('harmful')):5s}  txt={str(txt.get('harmful')):5s}  [{txt.get('reason','')}]")
        except Exception as e:
            print(f"  ERROR {img.name}: {e}")
        time.sleep(0.1)

    # ---- Test safe images ----
    print("\n=== SAFE MEMES (expected: SAFE) ===")
    for img in safe_imgs:
        try:
            r = predict(img)
            label = r.get("classification", "ERROR")
            conf  = r.get("confidence", 0)
            correct = label == "SAFE"
            results["SAFE"]["correct" if correct else "wrong"] += 1
            status = "✓" if correct else "✗"
            vis = r.get("visual_analysis", {})
            txt = r.get("text_analysis", {})
            detail = {
                "file": img.name, "predicted": label, "confidence": conf,
                "visual": vis.get("harmful"), "text": txt.get("harmful"),
                "text_reason": txt.get("reason","")
            }
            results["SAFE"]["details"].append(detail)
            print(f"  {status} {img.name:20s}  →  {label:8s}  conf={conf:.2f}  vis={str(vis.get('harmful')):5s}  txt={str(txt.get('harmful')):5s}  [{txt.get('reason','')}]")
        except Exception as e:
            print(f"  ERROR {img.name}: {e}")
        time.sleep(0.1)

    # ---- Summary ----
    h_total = results["HATEFUL"]["correct"] + results["HATEFUL"]["wrong"]
    s_total = results["SAFE"]["correct"] + results["SAFE"]["wrong"]
    h_acc = results["HATEFUL"]["correct"] / h_total * 100 if h_total else 0
    s_acc = results["SAFE"]["correct"] / s_total * 100 if s_total else 0
    overall_acc = (results["HATEFUL"]["correct"] + results["SAFE"]["correct"]) / (h_total + s_total) * 100

    print("\n" + "="*60)
    print("ACCURACY REPORT")
    print("="*60)
    print(f"  Hateful memes:  {results['HATEFUL']['correct']}/{h_total} correct  ({h_acc:.1f}%)")
    print(f"  Safe memes:     {results['SAFE']['correct']}/{s_total} correct  ({s_acc:.1f}%)")
    print(f"  Overall:        {overall_acc:.1f}%")
    print("="*60)

    # Misclassified
    misclassified = [d for d in results["HATEFUL"]["details"] if d["predicted"] != "HATEFUL"] + \
                    [d for d in results["SAFE"]["details"] if d["predicted"] != "SAFE"]
    if misclassified:
        print("\nMISSED CASES:")
        for m in misclassified:
            print(f"  {m['file']:25s} predicted={m['predicted']}  visual={m['visual']}  text={m['text']}  reason={m['text_reason']}")

if __name__ == "__main__":
    run_test()
