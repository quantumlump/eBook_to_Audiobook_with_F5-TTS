# /app/preload_models.py

import os
from cached_path import cached_path
# Import necessary components from f5_tts if needed for loading
# from f5_tts.model import DiT
# Assuming load_vocoder itself handles caching/downloading its needs
from f5_tts.infer.utils_infer import load_vocoder

import torch
import gc

# Import transformers components needed for preloading
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

# Determine device (can be 'cpu' for preloading, saves GPU memory during build)
device = "cpu"
print(f"Preloading models on device: {device}")

# --- Preload Main F5-TTS Model (Keep Existing Logic) ---
print("Preloading F5-TTS model...")
try:
    # Just caching the path is enough, no need to load the full model here
    f5_ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
    print(f"F5-TTS model path cached: {f5_ckpt_path}")
except Exception as e:
    print(f"Error preloading/caching F5-TTS model: {e}")

# --- Preload Vocoder (Keep Existing Logic) ---
# Assuming load_vocoder uses hf_hub_download or similar as seen in its definition
print("Preloading Vocoder...")
try:
    # Call load_vocoder to trigger its internal download/cache mechanism
    # Pass device='cpu' to save GPU memory during build if supported by the function
    vocoder = load_vocoder(device=device)
    print("Vocoder preloaded successfully.")
    del vocoder # Free memory
except Exception as e:
    print(f"Error preloading Vocoder: {e}")


# --- Preload Transcription (ASR) Model ---
# Correct model identifier found in utils_infer.py -> initialize_asr_pipeline
ASR_MODEL_ID = "openai/whisper-large-v3-turbo"  # <<< CORRECTED IDENTIFIER
# -------------------------------------------------

print(f"Attempting to preload ASR model: {ASR_MODEL_ID}...")
try:
    # Although the code uses `pipeline`, preloading the processor and model
    # components individually is usually sufficient to download the necessary files.
    print(f"Preloading ASR Processor for {ASR_MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
    print(f"Preloading ASR Model for {ASR_MODEL_ID}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(ASR_MODEL_ID)
    print(f"ASR processor and model files for {ASR_MODEL_ID} should be cached.")
    del processor, model # Free memory

    # --- Optional Alternative: Preload using pipeline directly ---
    # This might be slightly more robust as it exactly matches the runtime usage pattern.
    # Uncomment this block and comment out the Processor/Model block above if you prefer.
    # print(f"Attempting to preload ASR model via pipeline: {ASR_MODEL_ID}...")
    # pipe = pipeline("automatic-speech-recognition", model=ASR_MODEL_ID, device=device)
    # print(f"ASR pipeline for {ASR_MODEL_ID} seems preloaded/cached.")
    # del pipe
    # --- End Optional Alternative ---

except ImportError:
    print(f"Could not preload ASR model: 'transformers' library might be missing or failed to import.")
except Exception as e:
    print(f"Error preloading ASR model ({ASR_MODEL_ID}): {e}")
# --- END OF ASR SECTION ---


# Cleanup
gc.collect()
# No need for torch.cuda.empty_cache() if using device='cpu'

print("Model preloading script finished.")