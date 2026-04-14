# Gemini Project Instructions: Vietnamese VQA Thesis

These instructions are foundational for Gemini CLI when working on this Vietnamese Visual Question Answering (VQA) project.

## 📌 Project Context
- **Objective:** Developing a robust pipeline to generate high-quality Vietnamese VQA datasets.
- **Focus:** Cultural, historical, and architectural content of Vietnam.
- **Key Models:** Gemini (API), InternVL2, Qwen-VL (Local).

## 🛠️ Engineering Standards

### 🐍 Coding Style
- **Python 3.x:** Adhere to clean, modular, and well-documented Python code.
- **Model Integration:** Follow the existing patterns in `gemini.py` and `internvl.py` for adding or modifying model support.
- **Prompts:** Use Jinja2 templates from the `prompt_templates/` directory for all model interactions.
- **Data Formats:** Primarily JSON and JSONL for datasets.

### 🖋️ Linguistic Constraints (CRITICAL)
- **Primary Language:** Vietnamese.
- **Strict Prohibition:** NO Chinese characters (Hán) or English words in the final generated captions, questions, or answers.
- **Style:** Professional, journalistic, and culturally informed.

## 🏗️ Workflow Specifics

### Pipeline Maintenance
- When modifying any of the `step_X_*.py` scripts, ensure the data flow (input/output schemas) remains consistent.
- Always check and update related Jinja2 templates when prompt logic changes.

### Data Cleaning & Preprocessing
- Use `clean_data_*.py` scripts for bulk operations.
- Maintain data integrity when splitting or merging datasets.

### Testing & Validation
- For any code changes, verify with sample data to ensure the generation logic doesn't introduce hallucinations.
- Validate that output files are correctly formatted JSON/JSONL.

## 🤖 AI Assistant (Gemini) Specifics
- **Prompt Engineering:** Assist in refining Jinja2 templates to improve output quality (e.g., better reasoning, stricter constraint adherence).
- **Inference Issues:** Help debug CUDA/GPU-related issues for local model inference or API-related failures.
- **Batch Processing:** Support the creation of bash scripts or task runners to automate the full 5-step pipeline.

---
*Follow these rules strictly to ensure the thesis dataset is high-quality and culturally accurate.*
