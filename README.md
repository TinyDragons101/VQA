# Vietnamese VQA Dataset Generation Pipeline

This project is a comprehensive pipeline for generating high-quality, culturally-aware Visual Question Answering (VQA) datasets in Vietnamese. It leverages state-of-the-art Vision-Language Models (VLMs) and Large Language Models (LLMs) to automate the creation of image captions, questions, and ground-truth answers from web-crawled articles and images.

## 🚀 Overview

The pipeline consists of 5 sequential steps, designed to ensure data quality, factual accuracy, and linguistic consistency (strictly Vietnamese, avoiding Chinese characters and English).

### 📋 Pipeline Steps

1.  **Step 1: Image Captioning (`step_1_generate_caption.py`)**
    *   Generates detailed, context-aware captions for images using VLMs like **InternVL2** or **Qwen-VL**.
    *   Integrates image content with article titles and original captions for better context.
    *   Categorizes images into domains (Architecture, Culture, Art, etc.).

2.  **Step 2: Question Generation (`step_2_generate_question.py` / `_gemini.py`)**
    *   Creates diverse VQA pairs based on the generated captions and article context.
    *   Supports both local model inference and **Google Gemini API**.
    *   Follows specific domain-based guidelines for "Ground Truth" extraction.

3.  **Step 3: Answer Generation (`step_3_generating_answer.py` / `_gemini.py`)**
    *   Produces accurate answers for the generated questions.
    *   Uses the full article content to ensure the answers are factually grounded.

4.  **Step 4: Verification (`step_4_verify_answer.py`)**
    *   Audits the QA pairs for consistency and factual correctness.
    *   Filters out low-quality or hallucinated content.

5.  **Step 5: Difficulty Assessment (`step_5_define_difficulty.py`)**
    *   Categorizes the complexity of each VQA pair for better dataset balancing.

## 🛠️ Project Structure

*   `prompt_templates/`: Jinja2 templates for structured and consistent prompting across models.
*   `gemini.py`, `internvl.py`: Wrapper classes for model integration (API vs. Local).
*   `clean_data_*.py`: Utility scripts for data preprocessing, cleaning, and splitting.
*   `convert_jsonl_to_json.py`, `merge_jsons.py`: Format conversion tools.

## ⚙️ Installation

```bash
pip install torch torchvision transformers accelerate jinja2 google-generativeai
```

*Note: For local VLM inference, ensure you have a compatible GPU and the necessary weights for InternVL2 or Qwen-VL.*

## 📖 Usage

Each step is designed as a standalone script that processes a JSON/JSONL input and produces a refined output.

```bash
# Example: Step 1 Captioning
python step_1_generate_caption.py --input data.json --output captions.json

# Example: Step 2 Question Generation (Gemini)
python step_2_generate_question_gemini.py --input captions.json --output questions.json
```

## ⚖️ Guidelines

*   **Language:** All generated content must be in natural Vietnamese.
*   **Constraints:** No Chinese characters (Hán) or English words in the final QA pairs.
*   **Groundedness:** All questions and answers must be derived from the provided image or text context.

---
*Developed as part of a Thesis on Visual Question Answering.*


Format of database.json:
{
  "article_id_using_16hex": // f8097c7d27a8aac6 {
    "url": "...",
    "date": "...", // 2021-07-15T02:46:59Z
    "title": "...",
    "images": [
      {
        "image_id": "16hex",
        "caption": "...",
        "author": "..."
      }
    ],
    "content": "..."
  }, 
  "article_id_using_16hex2": {
    ...
  }
}


Format of image_caption.json
{
  "image_id1": {
    "article_id": ...,
    "title": ...,
    "category": ...,
    "original_caption": ...,
    "generated_caption": ...
  },
  "image_id2": {
    ...
  }
}


Format of image_questions.json
{
  "image_id1": [
    "question1",
    "question2",
    ...
  ],
  "image_id2": [
    "question3",
    "questions4"
  ],
  ...
}


Format of image_vqa.json
{
  "image_id": [
    [
        "question1",
        "answer1"
    ],
    [
        "question2",
        "answer2"
    ],
    ...
  ]
}

Format of image_vqa_with_difficult.json
{
  "image_id": [
    [
        "question1",
        "answer1",
        "3"
    ],
    [
        "question2",
        "answer2",
        "4"
    ],
    ...
  ]
}