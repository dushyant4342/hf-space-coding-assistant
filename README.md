---
title: Chat with DeepSeek Coder 1.3B
emoji: 💬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.25.0" # Or your specific Gradio version from requirements.txt
python_version: "3.10" # Or "3.9", "3.11" - ensure it matches your development env
app_file: app.py
pinned: false
# If you had secrets for API keys, you would list them here:
# secrets:
#   - YOUR_SECRET_NAME
---

# Chat with DeepSeek Coder 1.3B Instruct

Welcome to this interactive chat application powered by the `deepseek-ai/deepseek-coder-1.3b-instruct` model! This application is built using Python, Gradio, and the Hugging Face `transformers` library.

**Live Demo:** [https://huggingface.co/spaces/Dushyant4342/RAG-PDFChat](https://huggingface.co/spaces/Dushyant4342/RAG-PDFChat) (Replace with your actual Space URL if different)

## Features

* **Interactive Chat Interface:** Uses Gradio's `ChatInterface` for a user-friendly experience.
* **Powered by DeepSeek Coder 1.3B:** Leverages the `deepseek-ai/deepseek-coder-1.3b-instruct` model, known for its strong coding and instruction-following capabilities.
* **Conditional Quantization:**
    * If a CUDA-enabled GPU is detected in the Space environment, the model is loaded with 8-bit quantization using `bitsandbytes` for reduced memory footprint and potentially faster inference.
    * If running on CPU, the model is loaded in its native precision (typically FP32) to ensure stability and avoid potential `bitsandbytes` compatibility issues on CPU.
* **Dynamic Device Detection:** Automatically detects and uses available hardware (GPU/CPU).

## How It Works

The `app.py` script performs the following steps:

1.  **Loads Dependencies:** Imports necessary libraries like `gradio`, `torch`, and `transformers`.
2.  **Model & Tokenizer Loading:**
    * Downloads and loads the `deepseek-ai/deepseek-coder-1.3b-instruct` model and its tokenizer.
    * Applies 8-bit quantization if a GPU is available.
3.  **Chat Function:** A Python function `generate_chat_response` processes user input, formats it according to the DeepSeek Coder Instruct prompt style, and generates a response using the loaded model.
4.  **Gradio UI:** A `gr.ChatInterface` is created to provide the web-based chat UI.

## Running the Space

* **Model Loading Time:** Please be patient when the Space starts or after a rebuild. Downloading and loading the `deepseek-coder-1.3b-instruct` model (approx. 2.5GB) can take a few minutes, especially on the first run.
* **Performance:**
    * On GPU-enabled Spaces, responses should be relatively quick due to quantization and GPU acceleration.
    * On CPU-only Spaces, the model runs in full precision, which requires more RAM. Response times will be slower compared to GPU.

## Files in this Repository

* **`app.py`**: The main Python script containing the Gradio application logic and model interaction.
* **`requirements.txt`**: Lists all the Python dependencies required to run the application.
* **`README.md`**: This file, providing information about the Space and its configuration.

## Potential Improvements / Future Work

* Implement more sophisticated prompt engineering for varied tasks.
* Add options to tweak generation parameters (temperature, top_k, top_p) via the UI.
* Explore streaming responses for a more interactive feel.
* Integrate with RAG (Retrieval Augmented Generation) for chatting with documents (as the Space name "RAG-PDFChat" suggests this might be an ultimate goal).

---

Feel free to clone this Space, experiment with the code, or try different models!


