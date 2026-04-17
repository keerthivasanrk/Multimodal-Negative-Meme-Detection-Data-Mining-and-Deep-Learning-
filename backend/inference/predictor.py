import torch
import open_clip
from PIL import Image
from transformers import pipeline

from backend.ocr.ocr_engine import extract_text

# ---------------------------------------------------------------
# MULTIMODAL CONTENT MODERATION
#
# Uses TWO analysis layers:
#   1. VISUAL  — CLIP (ViT-B-32, already cached) for zero-shot
#                image classification against harmful categories.
#                Catches: sexual imagery, suggestive content,
#                         hate symbols, violence.
#   2. TEXT    — DehATEBERT + keyword/innuendo lexicon.
#                Catches: hate speech, sarcasm, double-meaning,
#                         sexual innuendo, slurs, insults.
#
# Final decision: HATEFUL if EITHER layer flags harmful content.
# ---------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_DEVICE = 0 if torch.cuda.is_available() else -1

print("[Predictor] Loading CLIP visual analysis model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model = clip_model.to(DEVICE)
clip_model.eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Zero-shot visual moderation prompts
# CLIP measures similarity between image and each text description
HARMFUL_VISUAL_PROMPTS = [
    "a sexually explicit or suggestive image",
    "a photo of a person in revealing or provocative clothing",
    "an image with hateful or racist symbols or messaging",
    "a violent or graphic image",
    "a meme sexualizing or targeting a person",
    "an adult-only or NSFW image",
    "an image making fun of someone based on their race or gender",
]
SAFE_VISUAL_PROMPTS = [
    "a safe, family-friendly image",
    "a normal everyday photo with no harmful content",
    "a funny but completely harmless meme",
    "an informative or educational image",
]

print("[Predictor] Loading TEXT hate speech model...")
hate_classifier = pipeline(
    "text-classification",
    model="Hate-speech-CNERG/dehatebert-mono-english",
    device=HF_DEVICE,
    truncation=True,
    max_length=512
)

# ---- Text-based keyword lexicons ----

SLURS_AND_INSULTS = {
    "looser", "loser", "stupid", "idiot", "dumb", "moron", "retard",
    "prostitute", "whore", "slut", "bitch", "hoe", "thot",
    "slave", "monkey", "subhuman", "filth", "garbage", "waste", "vermin",
    "rapist", "pervert", "creep", "sicko", "sicko", "predator",
}

# Explicit sexual / vulgar words — always flag regardless of context
EXPLICIT_WORDS = {
    "ass", "asses", "asshole",
    "tit", "tits", "tittys", "titties", "boob", "boobs", "boobies",
    "dick", "cock", "cocks", "penis", "vagina", "pussy",
    "fuck", "fucking", "fucker", "fucked",
    "shit", "shitty",
    "cunt", "cumshot", "cum",
    "naked", "nude", "nudes", "nsfw",
    "boner", "erection", "orgasm", "horny",
    "porn", "porno", "pornographic",
    "masturbat", "jerk off", "jack off",
    "sex", "sexy", "sexual",
}

SEXUAL_INNUENDO = [
    "pleased enough", "come again", "hard enough", "satisfy",
    "on your knees", "bang", "doing it", "get some", "put it in",
    "do it", "feels good", "laid", "get laid", "your lips",
    "hot and ready", "fill you up", "going down", "spread your",
    "pleasure", "climax", "finish", "tight", "ride me", "ride it",
    "prefer ass", "prefer tit", "like tit", "like ass",
    "sick man", "you're a sick", "you are a sick",
    "pull out", "deeper", "swallow", "spit or", "bend over",
    "sneaky link", "smash", "pass or smash", "smash or pass",
    "giving head", "give head", "good girl", "daddy",
]

SARCASM_HATE_PATTERNS = [
    "only good", "all of them should", "they are all",
    "go back to", "not one of us", "those people",
    "you people", "your kind", "their kind",
    "typical", "of course they", "well well well", 
    "dindu", "despite making up", "color me surprised",
    "another one", "usual suspects", "we all know",
    "religion of peace", "mostly peaceful", "vibrant diversity",
    "enrichment", "cultural enrichment"
]

IDENTITY_TERMS = {
    "black", "white", "asian", "chinese", "indian", "muslim", "jew", "jewish",
    "gay", "lesbian", "trans", "immigrant", "refugee", "mexican", "hispanic",
    "woman", "women", "female", "girl",
}


def analyze_image_visually(image_path: str):
    """
    Uses CLIP zero-shot classification to assess whether the image
    contains harmful visual content (sexual, violent, hateful, etc.)
    
    Returns: (is_harmful: bool, confidence: float, top_harmful_category: str)
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = clip_preprocess(image).unsqueeze(0).to(DEVICE)

        all_prompts = HARMFUL_VISUAL_PROMPTS + SAFE_VISUAL_PROMPTS
        text_tokens = clip_tokenizer(all_prompts).to(DEVICE)

        with torch.no_grad():
            img_feats = clip_model.encode_image(image_tensor)
            txt_feats = clip_model.encode_text(text_tokens)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            # Temperature-scaled cosine similarity then softmax
            similarity = (100.0 * img_feats @ txt_feats.T).softmax(dim=-1)

        similarities = similarity[0].cpu().tolist()

        n_harmful = len(HARMFUL_VISUAL_PROMPTS)
        harmful_scores = similarities[:n_harmful]
        safe_scores = similarities[n_harmful:]

        harmful_total = sum(harmful_scores)
        safe_total = sum(safe_scores)

        # Pick the single most-triggered harmful category
        top_idx = harmful_scores.index(max(harmful_scores))
        top_category = HARMFUL_VISUAL_PROMPTS[top_idx]

        is_harmful = harmful_total > safe_total
        return is_harmful, round(harmful_total, 4), top_category

    except Exception as e:
        print(f"[Visual Analysis] Error: {e}")
        return False, 0.0, "error"


def analyze_text(text: str):
    """
    Multi-layer text harm detection:
      1. Hate speech model (DehATEBERT)
      2. Slurs / insults keyword check
      3. Sexual innuendo / double-meaning phrases
      4. Sarcasm-disguised hate patterns
    
    Returns: (is_harmful: bool, confidence: float, reason: str)
    """
    lower = text.lower()

    # ---- Check 1: EXPLICIT WORDS (highest priority — always flag) ----
    matched_explicit = [w for w in EXPLICIT_WORDS if w in lower]
    if matched_explicit:
        return True, 0.85, f"explicit_content: {matched_explicit[0]}"

    # ---- Check 2: DehATEBERT hate speech model ----
    result = hate_classifier(text)[0]
    label = result["label"]
    score = result["score"]
    is_hate_model = label.lower() in ("hate", "hateful", "offensive")
    hate_conf = score if is_hate_model else (1.0 - score)

    # ---- Check 3: Keyword signals ----
    has_slur = any(kw in lower for kw in SLURS_AND_INSULTS)
    matched_innuendo = [p for p in SEXUAL_INNUENDO if p in lower]
    has_innuendo = bool(matched_innuendo)
    has_sarcasm_hate = any(p in lower for p in SARCASM_HATE_PATTERNS)
    has_identity = any(t in lower for t in IDENTITY_TERMS)

    # ---- Combine signals (hyper-sensitive) ----
    # 1. Immediate flag for keyword hits (sarcasm, innuendo, slurs) with very high confidence
    if has_innuendo:
        return True, max(0.85, hate_conf), f"sexual_innuendo: {matched_innuendo[0]}"

    if has_slur:
        # Find which slur matched for the reason string
        matched_slurs = [kw for kw in SLURS_AND_INSULTS if kw in lower]
        return True, max(0.85, hate_conf), f"offensive_language: {matched_slurs[0]}"

    if has_sarcasm_hate:
        return True, max(0.85, hate_conf), "sarcasm_or_targeted_speech"

    # 2. DehATEBERT hate speech model
    if is_hate_model:
        return True, round(hate_conf, 4), "hate_speech"

    # 3. Targeted speech: if an identity term is present and the model thinks there's ANY hate risk
    if has_identity and hate_conf > 0.05:
        return True, round(0.5 + hate_conf, 4), "potentially_targeted_hate"

    return False, round(1.0 - hate_conf, 4), "safe"



def predict(image_path: str):
    """
    Full multimodal content moderation pipeline.
    
    Analyzes BOTH the image visuals AND extracted text to detect:
     - Hate speech, racist content
     - Sexual / NSFW imagery
     - Sarcasm and double meaning
     - Offensive language and slurs
     - Targeted harassment
    """

    # ---- OCR ----
    extracted_text = extract_text(image_path)
    # Fix common OCR artefacts
    cleaned_text = (
        extracted_text
        .replace("TAM ", "I AM ")
        .replace("OFTHE", "OF THE")
        .strip()
    )
    text_to_classify = cleaned_text if cleaned_text else "no text"

    # ---- Visual Analysis ----
    is_visually_harmful, visual_conf, visual_category = analyze_image_visually(image_path)

    # ---- Text Analysis ----
    is_text_harmful, text_conf, text_reason = analyze_text(text_to_classify)

    # ---- Combined Decision ----
    # Flag as HATEFUL if EITHER the image OR text is harmful
    is_hateful = is_visually_harmful or is_text_harmful
    confidence = round(max(visual_conf, text_conf), 4)

    if not is_hateful:
        # Confident safe: invert so confidence represents "how safe"
        classification = "SAFE"
        confidence = round(1.0 - confidence, 4)
    else:
        classification = "HATEFUL"

    return {
        "classification": classification,
        "confidence": confidence,
        "extracted_text": extracted_text,
        "visual_analysis": {
            "harmful": is_visually_harmful,
            "score": visual_conf,
            "category": visual_category,
        },
        "text_analysis": {
            "harmful": is_text_harmful,
            "score": text_conf,
            "reason": text_reason,
        },
    }