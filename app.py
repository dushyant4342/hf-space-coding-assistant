import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import time

print(f"[{time.time()}] SCRIPT START: DeepSeek Coder 1.3B Chat (Conditional Quantization). PID: {os.getpid()}")

# --- Configuration ---
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{time.time()}] Using device: {DEVICE}")
print(f"[{time.time()}] PyTorch version: {torch.__version__}")

# --- Load Model and Tokenizer ---
model = None
tokenizer = None
model_load_error = None

try:
    print(f"[{time.time()}] Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"[{time.time()}] Tokenizer loaded. Vocab size: {tokenizer.vocab_size if tokenizer else 'N/A'}")

    if tokenizer and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[{time.time()}] Set pad_token to eos_token: {tokenizer.pad_token}")

    print(f"[{time.time()}] Attempting to load model {MODEL_NAME}...")

    if DEVICE == "cuda":
        print(f"[{time.time()}] Configuring 8-bit quantization for GPU...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto", # Let accelerate handle device mapping for GPU
            trust_remote_code=True
        )
        print(f"[{time.time()}] Model {MODEL_NAME} loaded with 8-bit quantization on GPU.")
    else: # CPU
        print(f"[{time.time()}] Loading model {MODEL_NAME} on CPU without bitsandbytes quantization.")
        # When on CPU, load without quantization_config to avoid bitsandbytes issues.
        # This will use more RAM but is more stable if bitsandbytes CPU support is problematic.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32, # Use float32 for CPU for broader compatibility
            trust_remote_code=True,
            low_cpu_mem_usage=True # Helpful for larger models on CPU
        )
        # Explicitly move to CPU if not already (low_cpu_mem_usage might handle parts of this)
        model.to(DEVICE) 
        print(f"[{time.time()}] Model {MODEL_NAME} loaded on CPU (FP32 precision).")

    model.eval() 
    # print(f"[{time.time()}] Model footprint: {model.get_memory_footprint()}") # Useful if available

except Exception as e:
    model_load_error = str(e)
    print(f"[{time.time()}] CRITICAL ERROR loading model or tokenizer: {e}")
    import traceback
    traceback.print_exc()


# --- Chat Function (remains the same as your previous version) ---
def generate_chat_response(message, history):
    print(f"[{time.time()}] generate_chat_response called. Message: '{message}'")

    if model_load_error or not model or not tokenizer:
        error_msg = f"Model not loaded. Error: {model_load_error if model_load_error else 'Unknown reason.'}"
        print(f"[{time.time()}] {error_msg}")
        return error_msg

    prompt_parts = []
    for user_msg, bot_msg in history:
        prompt_parts.append(f"### Instruction:\n{user_msg}\n### Response:\n{bot_msg}")
    prompt_parts.append(f"### Instruction:\n{message}\n### Response:")
    prompt = "\n".join(prompt_parts)

    try:
        print(f"[{time.time()}] Encoding prompt for model (length: {len(prompt)} chars)...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
        
        # Move inputs to the model's device if not using device_map="auto" or if it's explicitly CPU
        if DEVICE == "cpu": # Or check model.device directly
            inputs = inputs.to(model.device) 
        # If device_map="auto" was used (GPU case), inputs are often handled by accelerate

        print(f"[{time.time()}] Generating response... Input token length: {inputs['input_ids'].shape[1]}")

        with torch.no_grad(): 
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=200,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True, 
                top_p=0.95,
                top_k=50,
                temperature=0.7
            )
        
        response_text = tokenizer.decode(output_sequences[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        response_text = response_text.strip() 
        
        print(f"[{time.time()}] Raw generated text: '{response_text}'")
        if not response_text:
            response_text = "I'm not sure how to respond to that right now."
        return response_text
    except Exception as e:
        print(f"[{time.time()}] Error during text generation: {e}")
        import traceback
        traceback.print_exc()
        return f"Sorry, I encountered an error while generating a response: {e}"

# --- Gradio Interface (remains the same) ---
if __name__ == "__main__":
    print(f"[{time.time()}] MAIN: Building Gradio interface (DeepSeek Coder - Conditional Quantization)...")
    interface_title = f"Chat with LLM (deepseek-coder-1.3B)"
    interface_description = f"""
    This app runs **{MODEL_NAME}** directly in this Space.
    Model loading might take a few minutes. Running on: **{DEVICE.upper()}**.
    Quantization is attempted on GPU, bypassed on CPU to avoid `bitsandbytes` issues.
    """
    if model_load_error:
        interface_description += f"\n\n<h3 style='color:red;'>MODEL LOADING FAILED: {model_load_error}</h3>"
    elif not model or not tokenizer:
        interface_description += "\n\n<h3 style='color:orange;'>Warning: Model or tokenizer not available. Chat may not function.</h3>"

    chat_interface = gr.ChatInterface(
        fn=generate_chat_response,
        title=interface_title,
        description=interface_description,
        examples=[["Hello, what can you do?"], ["Write a python function to calculate factorial."]],
        cache_examples=False,
    )
    print(f"[{time.time()}] MAIN: Attempting to launch Gradio app...")
    try:
        chat_interface.queue().launch(debug=True) 
        print(f"[{time.time()}] MAIN: Gradio app launch() called. Monitor logs for 'Application startup complete'.")
    except Exception as e:
        print(f"[{time.time()}] FATAL ERROR during launch: {e}")
        with open("launch_error.txt", "w") as f_err: 
            f_err.write(f"Error during launch: {str(e)}\n")
print(f"[{time.time()}] SCRIPT END: DeepSeek Coder app.py (Conditional Quantization) has finished.")
