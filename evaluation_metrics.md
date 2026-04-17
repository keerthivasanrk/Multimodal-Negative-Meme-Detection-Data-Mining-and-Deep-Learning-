# Meme Moderation System - Evaluation Metrics

Based on the batch evaluation results against a localized sample (30 hateful, 4 safe), the system performance is as follows:

## Overall Scores
- **Accuracy Score:** 58.8% (20 correct predictions out of 34 total)
- **F1 Score:** 72.0%

## Confusion Matrix Breakdown
- **True Positives (TP):** 18 *(Hateful memes correctly predicted as Hateful)*
- **False Positives (FP):** 2 *(Safe memes incorrectly predicted as Hateful)*
- **True Negatives (TN):** 2 *(Safe memes correctly predicted as Safe)*
- **False Negatives (FN):** 12 *(Hateful memes incorrectly predicted as Safe)*

## Detailed Metrics

**1. Precision (90.0%)**
Precision measures how many of the "Hateful" predictions were actually Hateful.
`Precision = TP / (TP + FP) = 18 / (18 + 2) = 18 / 20 = 0.90`

**2. Recall (60.0%)**
Recall measures how many of the actual Hateful memes the system successfully caught.
`Recall = TP / (TP + FN) = 18 / (18 + 12) = 18 / 30 = 0.60`

**3. F1 Score (72.0%)**
The F1 score is the harmonic mean of Precision and Recall.
`F1 = 2 * (Precision * Recall) / (Precision + Recall) = 2 * (0.90 * 0.60) / (0.90 + 0.60) = 1.08 / 1.50 = 0.720`
