📌 Multimodal Negative Meme Detection

Data Mining & Deep Learning Project

A multimodal deep learning system that detects hateful or negative memes by analyzing both:

🖼 Image content (visual features)

📝 Embedded text inside memes (via OCR)


The system uses CLIP + BERT + Neural Fusion to perform binary classification (Hateful / Non-Hateful).


---------------------------------------------------------------------------------


🚀 Project Overview

Memes often combine images and text to convey harmful or hateful messages. Traditional text-only or image-only models fail to capture this cross-modal context.

This project implements a multimodal fusion architecture that:

1. Extracts visual features using CLIP


2. Extracts textual features using BERT


3. Uses OCR (EasyOCR) to read meme text


4. Fuses both embeddings for final classification




---------------------------------------------------------------------------------


🧠 Architecture

+------------------+
            |     Input Meme   |
            +------------------+
                     |
        +------------+------------+
        |                         |
   Image Encoder              OCR Engine
   (CLIP - ViT-B/32)          (EasyOCR)
        |                         |
   512-d vector              Extracted Text
                                   |
                             Text Encoder
                             (BERT Base)
                                   |
                              768-d vector
        \_______   ______/
                            \ /
                      Fusion Network
                 (Fully Connected Layer)
                            |
                         Sigmoid
                            |
                    HATE / NON-HATE

---------------------------------------------------------------------------------


🛠 Tech Stack

PyTorch

OpenCLIP (ViT-B/32)

HuggingFace Transformers (BERT-base-uncased)

EasyOCR

FastAPI

Uvicorn



---------------------------------------------------------------------------------


📂 Project Structure

backend/
│
├── api/               # FastAPI server
├── models/            # Vision, Text, Fusion models
├── training/          # Model training scripts
├── inference/         # Prediction logic
├── ocr/               # OCR engine
│
weights/               # Saved model weights (ignored in Git)
datasets/              # Training dataset (ignored in Git)
│
README.md
requirements.txt
.gitignore


---------------------------------------------------------------------------------


📊 Model Details

Component	Model Used	Output Dim

Vision Encoder	CLIP ViT-B/32	512
Text Encoder	BERT Base Uncased	768
Fusion Layer	Fully Connected (Linear)	1 (binary)


Final classification uses:

Sigmoid → Probability → Threshold 0.5


---------------------------------------------------------------------------------


📈 Training

Loss Function: BCEWithLogitsLoss

Optimizer: Adam

Epochs: 8+

Device: CPU (GPU supported)

Dataset: Labeled meme dataset with hate annotations


Example Training Result:

Final Training Loss: ~0.24
Binary Classification Output


---------------------------------------------------------------------------------


🧪 How To Run

1️⃣ Install dependencies

pip install -r requirements.txt

2️⃣ Start FastAPI server

uvicorn backend.api.app:app

3️⃣ Open Swagger UI

http://127.0.0.1:8000/docs

Upload a meme image and get prediction.


---------------------------------------------------------------------------------


🔍 Sample Output

{
  "classification": "HATEFUL",
  "confidence": 0.8732,
  "extracted_text": "ROSES ARE RED, VIOLETS ARE BLUE..."
}


---------------------------------------------------------------------------------


⚠️ Limitations

OCR errors may introduce noisy text.

Model performance depends on dataset balance.

Limited to English language memes.

Fusion network is shallow (can be improved with deeper layers).



---------------------------------------------------------------------------------


🔮 Future Improvements

Add validation tracking & F1-score evaluation

Improve fusion network depth

Train with combined clean + OCR text

Deploy via Docker

Add frontend interface

Optimize using GPU training



---------------------------------------------------------------------------------

🎯 Key Learnings

Multimodal learning improves meme moderation performance.

Fusion models require architecture consistency between training and inference.

OCR noise significantly affects NLP model quality.

Proper model-weight alignment is critical in deployment.



---------------------------------------------------------------------------------


👤 Author

Keerthivasan R
Multimodal Deep Learning Project
Data Mining & Artificial Intelligence

---------------------------------------------------------------------------------

output:

<img width="1920" height="1080" alt="Screenshot 2026-02-26 223312" src="https://github.com/user-attachments/assets/9d1dd48e-2980-4765-9da6-53d111dbcef6" />
