# Multimodal Meme Moderation System 🛡️

An advanced, multimodal AI system designed to detect and moderate harmful memes. By combining visual computer vision with natural language processing, the system identifies hate speech, offensive imagery, and disguised sarcasm that traditional single-mode filters often miss.

---

## 🚀 System Architecture

The project is divided into two main components: a **FastAPI-based Backend** that handles the heavy-duty AI inference, and a **Vite/React Frontend** that provides an intuitive interface for users to upload and moderate content.

### 🖼️ Backend (AI Core)
Located in `/backend`, this is the core brain of the system. It follows a "Multimodal Fusion" approach, extracting signals from both image pixels and embedded text:

- **OCR Engine (`/backend/ocr`)**: Uses **EasyOCR** to extract text embedded within images. It includes pre-processing logic to clean up common OCR artifacts.
- **Visual Analysis Layer**: Utilizes **OpenAI's CLIP (ViT-B-32)** for zero-shot image classification. It compares image features against a set of harmful visual prompts (e.g., hate symbols, violence, explicit imagery) to detect visual-only threats.
- **Text Analysis Layer**: 
    - **DehATEBERT**: A specialized Transformer model (`Hate-speech-CNERG/dehatebert-mono-english`) to detect toxic text.
    - **Keyword Lexicons**: Custom-built dictionaries to catch slurs, insults, and explicit content.
    - **Innuendo & Sarcasm Detection**: Specialized logic to flag double-meanings and "dog whistles" often used in memes.
- **Fusion Logic (`/backend/inference/predictor.py`)**: A "hyper-sensitive" decision engine. If *either* the visual analysis or the text analysis flags the content as harmful, the meme is classified as **HATEFUL** (Logical OR Late Fusion Strategy).

### 💻 Frontend (Moderation UI)
Located in `/frontend`, this provides a modern web interface.

- **Framework**: React 18+ powered by Vite for lightning-fast development.
- **Core Components**:
    - **Uploader**: A drag-and-drop zone for meme submission.
    - **Result Dashboard**: Displays a high-level classification (Safe/Hateful), a confidence score, and a breakdown of why the content was flagged.
    - **Real-time Extraction**: Shows the raw text extracted from the meme for transparency.

---

## 📊 System Performance & Evaluation Metrics

Based on the latest batch evaluation against a real-world localized sample of memes (30 hateful, 4 safe), the system achieves the following empirical results:

- **Accuracy:** 58.8%
- **F1 Score:** 72.0%
- **Precision:** 90.0%
- **Recall:** 60.0%

### Confusion Matrix Breakdown
- **True Positives (TP):** 18 *(Hateful memes correctly predicted as Hateful)*
- **False Positives (FP):** 2 *(Safe memes incorrectly predicted as Hateful)*
- **True Negatives (TN):** 2 *(Safe memes correctly predicted as Safe)*
- **False Negatives (FN):** 12 *(Hateful memes incorrectly predicted as Safe)*

The high **Precision (90.0%)** ensures that when the system flags a meme as hateful, it is almost certainly correct, minimizing false alarms. The **Recall (60.0%)** highlights ongoing efforts to capture more nuanced and disguised hateful content (False Negatives).

---

## 📂 Component Breakdown

| Component | Path | Description |
| :--- | :--- | :--- |
| **API Entry** | `backend/api/app.py` | FastAPI server that exposes the `/predict` endpoint. |
| **Inference Logic** | `backend/inference/predictor.py` | The main orchestration script that runs CLIP, BERT, and OCR. |
| **OCR Utility** | `backend/ocr/ocr_engine.py` | Wraps EasyOCR for text extraction from images. |
| **Models** | `backend/models/` | Definitions for various neural architectures used in the project. |
| **UI Source** | `frontend/src/` | Main React logic including `App.jsx` and styling in `App.css`. |
| **Weights** | `weights/` | Directory reserved for storing pre-trained model weights. |
| **Start Scripts** | `start_everything.bat` | A utility script to launch both the backend and frontend simultaneously. |

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- Node.js (for frontend)
- NVIDIA GPU (Optional, but highly recommended for CLIP & BERT inference)

### Installation
1. **Repository Setup**: Clone the repository to your local machine.
2. **Backend Setup**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   ```

### Running the System
You can use the provided batch script to start both production servers on Windows:
```cmd
start_everything.bat
```
Alternatively, start them manually:
- **Backend**: `uvicorn backend.api.app:app --reload`
- **Frontend**: `npm run dev` (inside `/frontend`)

---

## 🧠 Model Details
- **Visual Encoder**: CLIP (Contrastive Language-Image Pre-training) ViT-B-32. Maps images and text into a shared embedding space.
- **NLP Classifier**: DehATEBERT English Mono. Fine-tuned specifically for cross-lingual and mono-lingual hate speech detection.
- **OCR Engine**: EasyOCR (PyTorch based) for robust text detection over complex graphical backgrounds.
- **Fusion Strategy**: Late fusion (Logical OR or Flag-based weighting) over both unimodal branch outputs.

---

## 🛡️ Moderation Categories
The system specifically targets:
- **Hate Speech**: Racism, sexism, and identity-based attacks.
- **Explicit Content**: NSFW visuals and vulgar language.
- **Harassment**: Targetted insults and derogatory slurs.
- **Disguised Hate**: Sarcasm or innuendo-based harmful content.
