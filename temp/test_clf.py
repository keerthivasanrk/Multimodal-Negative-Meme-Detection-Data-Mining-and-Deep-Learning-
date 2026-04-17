from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
clf = pipeline("text-classification", model="Hate-speech-CNERG/dehatebert-mono-english", device=device)

tests = [
    "I am black and I was the president of the USA, you looser",
    "Me explaining the deep lore of JRR Tolkein. The prostitute I am paying to keep me company during COVID quarantine",
    "Have a nice day everyone!",
    "All immigrants should go back to their country",
]
for t in tests:
    r = clf(t)[0]
    label = r["label"]
    score = r["score"]
    print("Label:", label, " Score:", round(score, 4), "  Text:", t[:70])
