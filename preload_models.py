# preload_models.py
print("--- Starting Model Preloading ---")

# Import necessary functions/classes that trigger downloads
# We mainly need to ensure the files are downloaded via cached_path or huggingface_hub
from cached_path import cached_path
from f5_tts.infer.utils_infer import load_vocoder # To trigger Vocos download via its use of cached_path

# Define the model URLs used by the main app
VOCOS_MODEL_ID = "charactr/vocos-mel-24khz" # As seen in logs
F5TTS_MODEL_FILE = "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors" # As seen in logs

# --- Preload Vocos ---
print(f"Attempting to preload Vocos ({VOCOS_MODEL_ID})...")
try:
    # Calling the function that uses it is often enough if it uses cached_path
    # Or directly try caching its components if needed (less common)
    # Let's try calling load_vocoder, assuming it handles its own caching check
    _ = load_vocoder() # This function likely triggers the download internally
    print("Vocos preload function called.")
except Exception as e:
    print(f"Could not preload Vocos via load_vocoder: {e}")
    # As a fallback, try directly caching a known file from its repo if necessary
    # print("Trying direct cache for Vocos config...")
    # try:
    #     config_path = str(cached_path(f"hf://{VOCOS_MODEL_ID}/config.yaml"))
    #     print(f"Vocos config cached: {config_path}")
    # except Exception as e2:
    #     print(f"Direct cache for Vocos config failed: {e2}")


# --- Preload F5-TTS Model File ---
print(f"Attempting to preload F5-TTS model ({F5TTS_MODEL_FILE})...")
try:
    model_path = str(cached_path(F5TTS_MODEL_FILE))
    print(f"F5-TTS model cached: {model_path}")
except Exception as e:
    print(f"Could not preload F5-TTS model: {e}")


# You could add others if needed, e.g., tokenizers, other configs, vocab files, etc.
# print("Attempting to preload F5-TTS vocab...")
# try:
#    vocab_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/vocab.txt")) # Example, check actual path if needed
#    print(f"F5-TTS vocab cached: {vocab_path}")
# except Exception as e:
#    print(f"Could not preload F5-TTS vocab: {e}")

print("--- Model Preloading Script Finished ---")