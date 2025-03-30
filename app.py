# -*- coding: utf-8 -*-
import os
import re
import tempfile
import subprocess
import csv
import gc # Import garbage collector
from collections import OrderedDict
# from importlib.resources import files # No longer needed?
import shutil # For removing directories

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
# from transformers import AutoModelForCausalLM, AutoTokenizer # Not directly used here

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import nltk
try:
    # Check if 'punkt' resource is available
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' resource found.")
except LookupError:
    print("NLTK 'punkt' resource not found. Attempting download...")
    try:
        # Ensure NLTK_DATA is set if running in an env where it might not be obvious
        nltk_data_path = os.environ.get('NLTK_DATA', None)
        if nltk_data_path:
            print(f"Using NLTK_DATA path: {nltk_data_path}")
            # Ensure the specific download directory exists if specified
            if ":" in nltk_data_path: # Handle multiple paths if present
                nltk_data_path = nltk_data_path.split(":")[0]
            os.makedirs(os.path.join(nltk_data_path, 'tokenizers'), exist_ok=True)
            nltk.download('punkt', quiet=False, download_dir=nltk_data_path)
        else:
            nltk.download('punkt', quiet=False) # Download explicitly if missing, show output
        # Verify download
        nltk.data.find('tokenizers/punkt')
        print("'punkt' resource downloaded successfully.")
    except Exception as download_e:
        print(f"Failed to download NLTK 'punkt' resource: {download_e}. Text splitting may fall back to paragraphs.")


from nltk.tokenize import sent_tokenize
# from pydub import AudioSegment # No longer needed for stitching
import magic
from mutagen.id3 import ID3, APIC, error, TIT2, TPE1 # <<< Added TIT2 (Title) and TPE1 (Artist/Author)

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
)

import torch
import pkg_resources # Added for version checking

# Determine the available device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False

DEFAULT_TTS_MODEL = "F5-TTS"

# --- Constants ---
MAX_CHUNK_LENGTH = 750 # Default chunk length, can be overridden by UI
OUTPUT_DIR = os.path.join("Working_files", "Book")
TEMP_CONVERT_DIR = os.path.join("Working_files", "temp_converted")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_CONVERT_DIR, exist_ok=True)


# GPU Decorator
def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    return func

# Load models (globally)
print("Loading Vocoder...")
vocoder = load_vocoder()
print("Vocoder loaded.")

def load_f5tts(ckpt_path=None):
    if ckpt_path is None:
        ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
    model_cfg = {
        "dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2,
        "text_dim": 512, "conv_layers": 4
    }
    print(f"Loading F5-TTS model from {ckpt_path}...")
    model = load_model(DiT, model_cfg, ckpt_path)
    model.eval()
    model = model.to(device)
    print(f"F5-TTS Model loaded on {device}.")
    return model

F5TTS_ema_model = load_f5tts()

# --- Utility Functions (Extraction, Conversion, etc.) ---

def extract_metadata_and_cover(ebook_path):
    """Extract cover image from the eBook. Returns path or None."""
    print(f"Attempting to extract cover from: {ebook_path}")
    cover_path = None # Initialize cover_path
    try:
        # Create temp file within the persistent temp_converted dir to avoid potential OS restrictions on /tmp
        temp_cover_dir = TEMP_CONVERT_DIR
        ensure_directory(temp_cover_dir)
        # Use NamedTemporaryFile but ensure it's created in our accessible directory
        with tempfile.NamedTemporaryFile(dir=temp_cover_dir, suffix=".jpg", delete=False) as tmp_cover:
            cover_path = tmp_cover.name
        print(f"Temporary cover path: {cover_path}")

        command = ['ebook-meta', ebook_path, '--get-cover', cover_path]
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=False, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0 and os.path.exists(cover_path) and os.path.getsize(cover_path) > 0:
            print(f"Cover successfully extracted to {cover_path}")
            return cover_path
        else:
            print(f"Failed to extract cover. RC: {result.returncode}, Stderr: {result.stderr}, Stdout: {result.stdout}")
            if cover_path and os.path.exists(cover_path): os.remove(cover_path)
            return None
    except FileNotFoundError:
        print(f"Error: 'ebook-meta' command not found. Is Calibre installed and in PATH?")
        if cover_path and os.path.exists(cover_path):
             try: os.remove(cover_path)
             except OSError: pass
        return None
    except Exception as e:
        print(f"Error extracting eBook cover: {e}")
        if cover_path and os.path.exists(cover_path):
            try: os.remove(cover_path)
            except OSError: pass
        return None

def embed_metadata_into_mp3(mp3_path, cover_image_path, title, author):
    """Embed cover image, title, and author into the MP3 file's metadata."""
    if not mp3_path or not os.path.exists(mp3_path):
        print("MP3 path invalid/missing. Skipping metadata embedding.")
        return
    print(f"Attempting to embed metadata into {mp3_path}")
    try:
        audio = ID3(mp3_path)
    except error as e:
        print(f"Error loading ID3 tags from {mp3_path}, creating new. Error: {e}")
        try:
            audio = ID3()
        except Exception as create_e:
            print(f"Failed to create new ID3 tags for {mp3_path}. Error: {create_e}")
            return

    # Embed Cover Image
    if cover_image_path and os.path.exists(cover_image_path):
        try:
            audio.delall("APIC") # Remove existing cover art
            with open(cover_image_path, 'rb') as img:
                image_data = img.read()
            mime_type = magic.from_buffer(image_data, mime=True)
            if mime_type not in ['image/jpeg', 'image/png']:
                print(f"Warning: Cover MIME type {mime_type}, expected jpeg/png.")
            audio.add(APIC(encoding=3, mime=mime_type, type=3, desc='Front cover', data=image_data))
            print(f"Successfully prepared cover image for embedding.")
        except error as e:
            print(f"Mutagen ID3 Error preparing cover: {e}")
        except Exception as e:
            print(f"Failed to prepare cover image for embedding: {e}")
    else:
        print("Cover image path invalid/missing or file doesn't exist. Skipping cover embedding.")

    # Embed Title
    if title:
        audio.delall("TIT2") # Remove existing title
        audio.add(TIT2(encoding=3, text=title))
        print(f"Successfully prepared title '{title}' for embedding.")
    else:
        print("Title is empty. Skipping title embedding.")

    # Embed Author
    if author:
        audio.delall("TPE1") # Remove existing artist/author
        audio.add(TPE1(encoding=3, text=author))
        print(f"Successfully prepared author '{author}' for embedding.")
    else:
        print("Author is empty. Skipping author embedding.")

    # Save changes
    try:
        audio.save(mp3_path, v2_version=3)
        print(f"Successfully saved metadata to {mp3_path}")
    except error as e:
         print(f"Mutagen ID3 Error saving metadata: {e}")
    except Exception as e:
        print(f"Failed to save metadata to MP3: {e}")

def extract_text_title_author_from_epub(epub_path):
    """Extract full text, title, and author from an EPUB file in reading order."""
    try:
        book = epub.read_epub(epub_path)
        print(f"EPUB '{os.path.basename(epub_path)}' successfully read.")
    except Exception as e:
        raise RuntimeError(f"Failed to read EPUB file '{epub_path}': {e}")

    text_content = []
    title = "Untitled Audiobook"
    author = "Unknown Author"

    try:
        metadata_title = book.get_metadata('DC', 'title')
        if metadata_title: title = metadata_title[0][0]
        else: title = os.path.splitext(os.path.basename(epub_path))[0]
        print(f"Using title: {title}")
    except Exception as e:
        print(f"Could not get title metadata ({e}). Using filename as fallback.")
        title = os.path.splitext(os.path.basename(epub_path))[0]

    try:
        metadata_creator = book.get_metadata('DC', 'creator')
        if metadata_creator: author = ", ".join([creator[0] for creator in metadata_creator])
        else:
             author_meta = book.get_metadata('OPF', 'creator')
             if author_meta: author = author_meta[0][0]
             else: print("Author metadata ('DC:creator') not found. Keeping default.")
        print(f"Using author: {author}")
    except Exception as e:
        print(f"Could not get author metadata ({e}). Keeping default.")

    items_processed = 0
    spine_ids = [item[0] for item in book.spine] if book.spine else []
    ordered_items = []
    if spine_ids:
        item_map = {item.id: item for item in book.get_items()}
        for item_id in spine_ids:
            if item_id in item_map and item_map[item_id].get_type() == ITEM_DOCUMENT:
                ordered_items.append(item_map[item_id])
        for item in book.get_items_of_type(ITEM_DOCUMENT):
             if item.id not in spine_ids: ordered_items.append(item)
        print(f"Processing {len(ordered_items)} items based on spine order.")
    else:
        ordered_items = list(book.get_items_of_type(ITEM_DOCUMENT))
        print("Warning: EPUB spine information missing or empty. Processing items in default order.")

    for item in ordered_items:
            try:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                for script_or_style in soup(["script", "style", "head", "title"]): script_or_style.decompose()
                paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div'])
                item_text_parts = []
                for tag in paragraphs:
                    text_part = tag.get_text(separator=' ', strip=True)
                    text_part = re.sub(r'\s+', ' ', text_part).strip()
                    if text_part: item_text_parts.append(text_part)
                text = '\n\n'.join(item_text_parts)
                text = re.sub(r'[ \t]{2,}', ' ', text)
                text = re.sub(r'\n{3,}', '\n\n', text)
                if text:
                    text_content.append(text)
                    items_processed += 1
            except Exception as e:
                print(f"Warning: Error parsing item {item.get_id()} (Href: {item.get_name()}): {e}")

    if not text_content: raise ValueError(f"No text could be extracted from EPUB: {epub_path}")
    full_text = '\n\n'.join(text_content)
    print(f"Extracted {len(full_text)} chars from {items_processed} documents.")
    return full_text, title, author

def convert_to_epub(input_path, output_dir):
    """Convert an ebook to EPUB format using Calibre's ebook-convert."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    sanitized_base = sanitize_filename(base_name)
    output_path = os.path.join(output_dir, f"{sanitized_base}.epub")
    ensure_directory(output_dir)
    try:
        print(f"Converting '{input_path}' to EPUB '{output_path}'...")
        command = ['ebook-convert', input_path, output_path, '--enable-heuristics', '--keep-ligatures', '--input-encoding=utf-8']
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=False, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            error_message = f"Output EPUB not found or empty after conversion. RC: {result.returncode}. Error: {result.stderr or result.stdout}"
            print(error_message); raise RuntimeError(error_message)
        if result.returncode != 0: print(f"Warning: ebook-convert finished with return code {result.returncode}, but output file exists. Stderr: {result.stderr}, Stdout: {result.stdout}")
        print(f"Successfully converted to '{output_path}'.")
        return output_path
    except FileNotFoundError: raise RuntimeError(f"Error: 'ebook-convert' command not found. Is Calibre installed and in PATH?")
    except Exception as e: raise RuntimeError(f"eBook conversion failed: {e}")

def detect_file_type(file_path):
    """Detect the MIME type of a file."""
    if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
    try:
        mime_type = magic.Magic(mime=True).from_file(file_path)
        if mime_type is None:
             print(f"Warning: python-magic returned None for {file_path}. Attempting fallback detection.")
             ext = os.path.splitext(file_path)[1].lower()
             if ext == '.epub': return 'application/epub+zip'
             if ext == '.mobi': return 'application/x-mobipocket-ebook'
             if ext == '.pdf': return 'application/pdf'
             if ext == '.txt': return 'text/plain'
             if ext == '.html': return 'text/html'
             if ext == '.azw3': return 'application/vnd.amazon.ebook'
             if ext == '.fb2': return 'application/x-fictionbook+xml'
             return None
        return mime_type
    except Exception as e:
        print(f"Warning: Error detecting file type for {file_path}: {e}")
        return None

def ensure_directory(directory_path):
    """Ensure that a directory exists."""
    if not directory_path: raise ValueError("Directory path cannot be empty.")
    try: os.makedirs(directory_path, exist_ok=True)
    except OSError as e: raise RuntimeError(f"Error creating directory '{directory_path}': {e}")
    except Exception as e: raise RuntimeError(f"Unexpected error creating directory '{directory_path}': {e}")

def sanitize_filename(filename):
    """Sanitize a filename by removing invalid characters and replacing spaces."""
    if not filename: return "default_filename"
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    sanitized = re.sub(r'\s+', "_", sanitized)
    sanitized = sanitized.strip('_ ')
    return sanitized if sanitized else "sanitized_filename"

def show_converted_audiobooks():
    """List all converted audiobook files. Returns empty list if none found or on error."""
    if not os.path.exists(OUTPUT_DIR): print(f"Output directory '{OUTPUT_DIR}' does not exist."); return []
    try:
        files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('.mp3') and os.path.isfile(os.path.join(OUTPUT_DIR, f))]
        if not files: print(f"No MP3 files found in '{OUTPUT_DIR}'."); return []
        files.sort(key=os.path.getmtime, reverse=True)
        return files
    except Exception as e:
        print(f"Error listing audiobooks in '{OUTPUT_DIR}': {e}")
        return []

def split_text_into_chunks(text, max_length=MAX_CHUNK_LENGTH):
    """Splits text into chunks suitable for TTS, respecting sentence boundaries if possible."""
    chunks = []
    current_chunk = ""
    print(f"Splitting text with max_length = {max_length}")

    try:
        tokenizer_path = 'tokenizers/punkt/english.pickle'
        sent_tokenizer = nltk.data.load(tokenizer_path)
        sentences = sent_tokenizer.tokenize(text)
        print(f"NLTK tokenized into {len(sentences)} sentences.")
    except Exception as e:
        print(f"NLTK sentence tokenization failed: {e}. Falling back to paragraph/line split.")
        sentences = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not sentences: sentences = [p.strip() for p in text.split('\n') if p.strip()]
        if not sentences and len(text) > max_length:
             print(f"Text has no clear paragraphs/lines. Using fixed-length split of {max_length} chars as final fallback.")
             sentences = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        elif not sentences and text: sentences = [text]; print("Text is short and no splits found. Treating as single chunk.")
        print(f"Split into {len(sentences)} segments using fallback method.")

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        if len(sentence) > max_length:
            print(f"Warning: Sentence exceeds max_length ({len(sentence)} > {max_length}). Splitting mid-sentence.")
            if current_chunk: chunks.append(current_chunk); current_chunk = ""
            for i in range(0, len(sentence), max_length): chunks.append(sentence[i:i+max_length])
        elif len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += (" " + sentence) if current_chunk else sentence
        else:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk)
    chunks = [c for c in chunks if c]

    print(f"Split text into {len(chunks)} non-empty chunks.")
    if chunks:
        lengths = [len(c) for c in chunks]; print(f"Chunk lengths: Min={min(lengths)}, Max={max(lengths)}, Avg={sum(lengths)/len(lengths):.2f}")
    return chunks

@gpu_decorator
def infer_chunk(ref_audio_orig, ref_text, text_chunk, cross_fade_duration, speed, show_info):
    """ Perform inference for a single text chunk. """
    try:
        effective_show_info = show_info if callable(show_info) else (lambda *args, **kwargs: None)
        ref_audio, processed_ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=effective_show_info)
        if not ref_text and not processed_ref_text: raise RuntimeError("Automatic transcription of reference audio failed or produced empty text. Cannot proceed with TTS.")
        elif not ref_text and processed_ref_text: print("Automatic transcription successful.")
    except Exception as e:
        import traceback; print(f"Error during reference audio/text preprocessing: {e}\n{traceback.format_exc()}"); raise RuntimeError(f"Preprocessing failed: {e}")
    if not text_chunk or text_chunk.isspace(): print("Skipping empty or whitespace-only text chunk."); return None
    try:
        with torch.no_grad():
            final_wave, final_sample_rate, _ = infer_process(
                ref_audio, processed_ref_text, text_chunk, F5TTS_ema_model, vocoder,
                cross_fade_duration=cross_fade_duration, speed=speed, show_info=effective_show_info,
            )
        if final_wave is None or final_sample_rate is None: print("Warning: TTS inference returned None for wave or sample rate."); return None
        if isinstance(final_wave, torch.Tensor) and final_wave.numel() == 0: print("Warning: TTS inference returned an empty tensor."); return None
        if isinstance(final_wave, np.ndarray) and final_wave.size == 0: print("Warning: TTS inference returned an empty numpy array."); return None
        print(f"Generated chunk: {len(final_wave) if hasattr(final_wave, '__len__') else 'N/A'} samples @ {final_sample_rate} Hz.")
        return (final_sample_rate, final_wave)
    except Exception as e:
        import traceback; print(f"Error during TTS inference (infer_process): {e}\n{traceback.format_exc()}"); return None

# --- Main Processing Logic ---

def process_ebook_to_audio(ref_audio_input, ref_text_input, ebook_path, cross_fade_duration, speed, max_chunk_length, mp3_bitrate, progress=gr.Progress(track_tqdm=True)):
    """Processes a single eBook to an audiobook using chunking."""
    temp_dir = None; temp_chunk_files = []; final_mp3_path = None; converted_epub_path = None; extracted_cover_path = None
    sample_rate = 24000; ebook_title = "Untitled"; ebook_author = "Unknown Author"
    try:
        if not ebook_path or not os.path.exists(ebook_path): yield None, f"Error: Input file not found or path is invalid: {ebook_path}"; return
        progress(0, desc=f"Starting: {os.path.basename(ebook_path)}"); original_input_path = ebook_path
        file_type = detect_file_type(ebook_path); print(f"Detected file type: {file_type}")
        if file_type != 'application/epub+zip':
            progress(0.05, desc="Converting to EPUB...")
            try: converted_epub_path = convert_to_epub(ebook_path, TEMP_CONVERT_DIR); epub_path_to_process = converted_epub_path; print(f"Using converted EPUB: {epub_path_to_process}")
            except Exception as e: yield None, f"Error: Conversion to EPUB failed: {e}"; return
        else: epub_path_to_process = ebook_path; print("Input is already EPUB. No conversion needed.")
        progress(0.1, desc="Extracting text/metadata...")
        try:
            gen_text, ebook_title, ebook_author = extract_text_title_author_from_epub(epub_path_to_process)
            extracted_cover_path = extract_metadata_and_cover(epub_path_to_process)
            print(f"Extracted Title: '{ebook_title}', Author: '{ebook_author}'")
            print(f"Cover path: {extracted_cover_path}" if extracted_cover_path else "No cover found or extracted.")
        except ValueError as ve: yield None, f"Error: Failed to extract text content: {ve}"; return
        except Exception as e: import traceback; yield None, f"Error: Failed to extract content/metadata: {e}\n{traceback.format_exc()}"; return
        ref_text = ref_text_input; print("Using provided reference text." if ref_text else "Reference text is empty. Automatic transcription will be attempted.")
        sanitized_title = sanitize_filename(ebook_title); sanitized_author = sanitize_filename(ebook_author)
        if not sanitized_title: sanitized_title = f"audiobook_{os.path.splitext(os.path.basename(original_input_path))[0]}"
        base_filename = f"{sanitized_title}_by_{sanitized_author}" if sanitized_author and sanitized_author.lower() != "unknown_author" else sanitized_title
        final_mp3_path = os.path.join(OUTPUT_DIR, f"{base_filename}.mp3"); ensure_directory(os.path.dirname(final_mp3_path)); print(f"Planned output path: {final_mp3_path}")
        progress(0.2, desc="Splitting text..."); text_chunks = split_text_into_chunks(gen_text, max_length=max_chunk_length)
        total_chunks = len(text_chunks);
        if total_chunks == 0: yield None, "Error: Text splitting resulted in zero chunks."; return
        temp_dir = tempfile.mkdtemp(prefix="audio_chunks_"); print(f"Temporary directory for audio chunks: {temp_dir}")
        successful_chunks = 0; first_chunk_processed = False; dummy_show_info = lambda *args, **kwargs: None
        for i, chunk in enumerate(text_chunks):
            chunk_start_progress = 0.25 + (i / total_chunks) * 0.5; progress(chunk_start_progress, desc=f"Generating audio: Chunk {i+1}/{total_chunks}")
            print(f"\nProcessing Chunk {i+1}/{total_chunks} (Length: {len(chunk)})")
            clean_chunk = chunk.strip(); clean_chunk = re.sub(r'\s+', ' ', clean_chunk)
            if not clean_chunk: print("Skipping empty or whitespace-only chunk."); continue
            audio_out_chunk_data = None
            try: audio_out_chunk_data = infer_chunk(ref_audio_input, ref_text, clean_chunk, cross_fade_duration, speed, show_info=dummy_show_info)
            except RuntimeError as infer_err: print(f"Chunk {i+1} Error (from infer_chunk): {infer_err}")
            except Exception as generic_infer_err: print(f"Chunk {i+1} Unexpected Error during inference call: {generic_infer_err}")
            if audio_out_chunk_data:
                chunk_sample_rate, wave = audio_out_chunk_data
                if wave is not None and chunk_sample_rate is not None and hasattr(wave, 'size') and wave.size > 0:
                    chunk_filename = os.path.join(temp_dir, f"chunk_{i:05d}.wav")
                    try:
                        if isinstance(wave, torch.Tensor): wave = wave.squeeze().cpu().numpy()
                        if wave.ndim > 1: wave = np.mean(wave, axis=1)
                        if wave.dtype!=np.float32 and wave.dtype!=np.int16: wave = wave.astype(np.float32) if np.issubdtype(wave.dtype, np.floating) and np.max(np.abs(wave))<=1.0 else wave.astype(np.int16) if np.issubdtype(wave.dtype, np.integer) else wave.astype(np.float32)
                        if not first_chunk_processed: sample_rate = chunk_sample_rate; print(f"Audio sample rate set to {sample_rate} Hz (from first chunk)."); first_chunk_processed = True
                        elif chunk_sample_rate != sample_rate: print(f"Warning: Sample rate mismatch! Chunk {i+1} has SR={chunk_sample_rate}, expected {sample_rate}.")
                        sf.write(chunk_filename, wave, chunk_sample_rate); temp_chunk_files.append(chunk_filename); successful_chunks += 1; print(f"Saved chunk {i+1} to {chunk_filename}")
                    except Exception as e: print(f"Error saving chunk {i+1} WAV: {e}")
                else: print(f"Warning: Inference for chunk {i+1} produced invalid or empty audio data.")
            else: print(f"Warning: Inference failed or was skipped for chunk {i+1}.")
            del audio_out_chunk_data, wave; gc.collect();
            if device == torch.device("cuda"): torch.cuda.empty_cache()
        if not temp_chunk_files: yield None, "Error: No audio chunks were successfully generated."; return
        print(f"\nFinished generating audio chunks. Successful: {successful_chunks}/{total_chunks}.")
        progress(0.8, desc="Stitching audio..."); print(f"Concatenating audio chunks using ffmpeg (Bitrate: {mp3_bitrate})...")
        list_file_path = os.path.join(temp_dir, "mylist.txt")
        try:
            with open(list_file_path, 'w', encoding='utf-8') as f:
                for chunk_file_path in temp_chunk_files: escaped_path = chunk_file_path.replace("'", "'\\''"); normalized_path = escaped_path.replace(os.sep, '/'); f.write(f"file '{normalized_path}'\n")
            print(f"Generated ffmpeg file list: {list_file_path}")
        except Exception as e: yield None, f"Error creating ffmpeg file list: {e}"; return
        ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file_path, '-vn', '-c:a', 'libmp3lame', '-b:a', str(mp3_bitrate), '-ar', str(sample_rate), '-ac', '1', final_mp3_path]
        print(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
        try:
            proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
            if proc.returncode != 0: error_message = f"ffmpeg concatenation error (RC={proc.returncode}):\n{proc.stderr or proc.stdout}"; print(error_message); yield None, error_message; return
            else: print("ffmpeg concatenation successful.")
        except FileNotFoundError: yield None, "Error: 'ffmpeg' command not found. Is ffmpeg installed and in PATH?"; return
        except Exception as e: yield None, f"Error running ffmpeg: {e}"; return
        progress(0.95, desc="Adding metadata...");
        if os.path.exists(final_mp3_path): embed_metadata_into_mp3(final_mp3_path, extracted_cover_path, ebook_title, ebook_author)
        else: print(f"Warning: Final MP3 file '{final_mp3_path}' not found after ffmpeg process. Skipping metadata embedding.")
        progress(1, desc=f"Completed: {os.path.basename(final_mp3_path)}"); print(f"Successfully created audiobook: {final_mp3_path}"); yield final_mp3_path, None
    except Exception as e: import traceback; error_msg = f"Unexpected error processing {os.path.basename(ebook_path)}: {e}\n{traceback.format_exc()}"; print(f"Fatal Error: {error_msg}"); yield None, error_msg
    finally:
        print("Cleaning up temporary files...");
        if temp_dir and os.path.exists(temp_dir):
            try: shutil.rmtree(temp_dir); print(f"Removed temp dir: {temp_dir}")
            except Exception as e: print(f"Warn: Could not remove temp dir {temp_dir}: {e}")
        if converted_epub_path and os.path.exists(converted_epub_path):
            try: os.remove(converted_epub_path); print(f"Removed temp EPUB: {converted_epub_path}")
            except Exception as e: print(f"Warn: Could not remove temp EPUB {converted_epub_path}: {e}")
        if extracted_cover_path and os.path.exists(extracted_cover_path) and TEMP_CONVERT_DIR in extracted_cover_path:
            try: os.remove(extracted_cover_path); print(f"Removed temp cover: {extracted_cover_path}")
            except Exception as e: print(f"Warn: Could not remove temp cover {extracted_cover_path}: {e}")
        gc.collect();
        if device == torch.device("cuda"): torch.cuda.empty_cache(); print("Cleanup finished.")

def batch_process_ebooks(ref_audio_input, ref_text_input, gen_file_inputs, cross_fade_duration, speed, max_chunk_length, mp3_bitrate, progress=gr.Progress(track_tqdm=True)):
    """Handles batch processing of eBooks for Gradio, yielding results."""
    if not gen_file_inputs: gr.Warning("No eBook files provided."); yield None, show_converted_audiobooks(); return
    if not ref_audio_input and not ref_text_input: gr.Error("Reference Audio required if Reference Text not provided."); yield None, show_converted_audiobooks(); return
    elif not ref_audio_input and ref_text_input: print("Proceeding with reference text only."); gr.Warning("Reference Audio missing. TTS may be suboptimal.")
    processed_paths = []; last_successful_output_path = None; errors = []; ebook_paths = []
    if isinstance(gen_file_inputs, list): ebook_paths = [f.name for f in gen_file_inputs if hasattr(f, 'name') and f.name]
    elif hasattr(gen_file_inputs, 'name') and gen_file_inputs.name: ebook_paths = [gen_file_inputs.name]
    if not ebook_paths: gr.Warning("No valid eBook file paths found."); yield None, show_converted_audiobooks(); return
    num_ebooks = len(ebook_paths); print(f"Starting batch processing for {num_ebooks} eBook(s)..."); print(f"Settings: Speed={speed}, CrossFade={cross_fade_duration}, ChunkLen={max_chunk_length}, Bitrate={mp3_bitrate}")
    for idx, ebook_path in enumerate(ebook_paths):
        print(f"\n--- Processing eBook {idx+1}/{num_ebooks}: {os.path.basename(ebook_path)} ---")
        try:
            ebook_processor_gen = process_ebook_to_audio(ref_audio_input, ref_text_input, ebook_path, cross_fade_duration, speed, int(max_chunk_length), mp3_bitrate, progress)
            final_path, error_msg = next(ebook_processor_gen)
            if error_msg: error_details = f"'{os.path.basename(ebook_path)}': Failed - {error_msg}"; print(f"ERROR: {error_details}"); errors.append(error_details); yield last_successful_output_path, show_converted_audiobooks()
            elif final_path and os.path.exists(final_path): success_message = f"'{os.path.basename(ebook_path)}': OK -> '{os.path.basename(final_path)}'"; print(f"SUCCESS: {success_message}"); processed_paths.append(final_path); last_successful_output_path = final_path; yield last_successful_output_path, show_converted_audiobooks()
            else: unknown_error = f"'{os.path.basename(ebook_path)}': Finished, but no output file generated."; print(f"WARNING: {unknown_error}"); errors.append(unknown_error); yield last_successful_output_path, show_converted_audiobooks()
        except StopIteration: stop_iter_warn = f"'{os.path.basename(ebook_path)}': Processing generator finished unexpectedly."; print(f"WARNING: {stop_iter_warn}"); errors.append(stop_iter_warn); yield last_successful_output_path, show_converted_audiobooks()
        except Exception as e: import traceback; batch_loop_error = f"Batch loop error for '{os.path.basename(ebook_path)}': {e}\n{traceback.format_exc()}"; print(f"ERROR: {batch_loop_error}"); errors.append(f"'{os.path.basename(ebook_path)}': Critical batch error - {e}"); yield last_successful_output_path, show_converted_audiobooks()
    print("\n--- Batch Processing Complete ---"); final_audiobook_list = show_converted_audiobooks()
    if errors: error_summary = "Batch processing finished with errors:\n" + "\n".join(errors); gr.Warning(error_summary); print("\nSummary of Errors:"); [print(f"- {e}") for e in errors]
    print(f"\nSuccessful conversions: {len(processed_paths)}, Errors/warnings: {len(errors)}")
    yield last_successful_output_path, final_audiobook_list

# --- Gradio App Definition ---

DEFAULT_REF_AUDIO_PATH = "/app/default_voice.mp3"
DEFAULT_REF_TEXT = "For thirty-six years I was the confidential secretary of the Roman statesman Cicero. At first this was exciting, then astonishing, then arduous, and finally extremely dangerous."

def clear_ref_text_on_audio_change(audio_filepath):
    if audio_filepath and os.path.exists(audio_filepath): print(f"Ref audio changed ({os.path.basename(audio_filepath)}). Clearing ref text."); return ""
    else: print("Ref audio cleared. Ref text cleared."); return ""

# <<< UPDATED create_gradio_app with button moved >>>
def create_gradio_app():
    """Create and configure the Gradio application."""
    try: theme = gr.themes.Soft(); print("Soft theme loaded.")
    except AttributeError: print("Warning: Soft theme not found, using Default."); theme = gr.themes.Default()
    default_audio_exists = os.path.exists(DEFAULT_REF_AUDIO_PATH)
    if not default_audio_exists: print(f"Warning: Default ref audio not found: {DEFAULT_REF_AUDIO_PATH}")
    available_bitrates = ["128k", "192k", "256k", "320k"]; default_bitrate = "320k"; default_chunk_length = MAX_CHUNK_LENGTH

    with gr.Blocks(theme=theme, title="eBook-to-Audiobook") as app:
        gr.Markdown("# 📚➡️🎧 eBook to Audiobook Converter (F5-TTS)")
        gr.Markdown("Upload eBooks and a voice sample (<15s) to generate an audiobook.")

        with gr.Row():
            # --- Left Column: Inputs & Settings ---
            with gr.Column(scale=1, min_width=350):
                gr.Markdown("## 1. Provide Reference Voice")
                ref_audio_input = gr.Audio(label="Upload Voice Sample (<15s) or Record", sources=["upload", "microphone"], type="filepath", value=DEFAULT_REF_AUDIO_PATH if default_audio_exists else None)
                gr.Markdown("## 2. Upload eBooks")
                gen_file_input = gr.Files(label="Upload eBook File(s)", file_types=['.epub', '.mobi', '.pdf', '.txt', '.html', '.azw3', '.fb2'], file_count="multiple")
                gr.Markdown("## 3. Configure Settings")
                ref_text_input = gr.Textbox(label="Reference Text (Optional - Leave Blank for Auto-Transcription)", lines=3, placeholder="Enter EXACT transcript or leave blank...", value=DEFAULT_REF_TEXT if default_audio_exists else "")
                speed_slider = gr.Slider(label="Speech Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.05)
                cross_fade_duration_slider = gr.Slider(label="Chunk Cross-Fade (Seconds)", minimum=0.0, maximum=0.5, value=0.0, step=0.01)
                max_chunk_length_input = gr.Slider(label="Max Text Chunk Length (Characters)", minimum=100, maximum=1500, value=default_chunk_length, step=50)
                mp3_bitrate_input = gr.Dropdown(label="Output MP3 Bitrate (Quality)", choices=available_bitrates, value=default_bitrate, interactive=True)
                # --- Generate button is NOT defined here anymore ---

            # --- Right Column: Outputs & Actions ---
            with gr.Column(scale=2, min_width=400):
                gr.Markdown("## 4. Play & Download")
                player = gr.Audio(label="Listen to the Latest Audiobook", interactive=False)
                audiobooks_output = gr.Files(label="Completed Audiobooks (Download Links)", interactive=False, file_count="multiple")
                show_audiobooks_btn = gr.Button("Refresh Audiobook List", variant="secondary")
                # <<< Generate Button moved here, below Refresh button >>>
                generate_btn = gr.Button("Generate Audiobook(s)", variant="primary", scale=2) # scale might need adjustment here

        # --- Event Listeners ---
        ref_audio_input.change(fn=clear_ref_text_on_audio_change, inputs=[ref_audio_input], outputs=[ref_text_input], queue=False)
        generate_btn.click(
            fn=batch_process_ebooks,
            inputs=[ref_audio_input, ref_text_input, gen_file_input, cross_fade_duration_slider, speed_slider, max_chunk_length_input, mp3_bitrate_input],
            outputs=[player, audiobooks_output],
        )
        show_audiobooks_btn.click(fn=show_converted_audiobooks, inputs=[], outputs=[audiobooks_output], queue=False)
        app.load(fn=show_converted_audiobooks, inputs=None, outputs=[audiobooks_output], queue=False)

    return app

# --- Main Execution ---

@click.command()
@click.option("--port", "-p", default=7860, type=int, help="Port number.")
@click.option("--host", "-H", default="0.0.0.0", help="Host address (0.0.0.0 for Docker).")
@click.option("--share", "-s", default=False, is_flag=True, help="Share via Gradio link.")
@click.option("--debug", default=False, is_flag=True, help="Gradio debug mode.")
def main(port, host, share, debug):
    """Main entry point to launch the Gradio app."""
    print("--- Pre-run Checks ---")
    if not os.path.exists(DEFAULT_REF_AUDIO_PATH): print(f"WARN: Default ref audio missing: {DEFAULT_REF_AUDIO_PATH}")
    else: print(f"Default ref audio found: '{DEFAULT_REF_AUDIO_PATH}'")
    try: result=subprocess.run(['ffmpeg','-version'],check=True,capture_output=True,text=True); print(f"ffmpeg found. ({result.stdout.splitlines()[0]})")
    except Exception as e: print(f"ERROR: ffmpeg check failed: {e}. Audio stitching will likely fail.")
    try: calibre_path="/root/calibre/ebook-convert"; cmd=[calibre_path if os.path.exists(calibre_path) else 'ebook-convert', '--version']; result=subprocess.run(cmd,check=True,capture_output=True,text=True); print(f"ebook-convert found. ({result.stdout.strip()})")
    except Exception as e: print(f"ERROR: ebook-convert check failed: {e}. Conversion for non-EPUB formats will fail.")
    try: nltk.data.find('tokenizers/punkt'); print("NLTK 'punkt' data found.")
    except LookupError: print("WARN: NLTK 'punkt' data not found. Sentence splitting may be less accurate.")
    print("--- End Pre-run Checks ---")
    app = create_gradio_app()
    print(f"\nStarting Gradio app on http://{host}:{port}")
    if share: print("Gradio share link will be created.")
    if debug: print("Gradio debug mode enabled.")
    try: gradio_version = pkg_resources.get_distribution("gradio").version; print(f"Gradio Version: {gradio_version}")
    except Exception: print("Could not determine Gradio version.")
    app.queue().launch( server_name=host, server_port=port, share=share, debug=debug )

if __name__ == "__main__":
    import sys
    print("--- System/Library Information ---")
    print("Python Version:", sys.version.split('\n')[0])
    try: print("Torch Version:", torch.__version__)
    except Exception: print("Torch not found.")
    try: print("Torchaudio Version:", torchaudio.__version__)
    except Exception: print("Torchaudio not found.")
    try: print("Transformers Version:", pkg_resources.get_distribution("transformers").version)
    except Exception: print("Transformers not found.")
    try: print("NLTK Version:", nltk.__version__)
    except Exception: print("NLTK not found.")
    try: print("Gradio Version:", pkg_resources.get_distribution("gradio").version)
    except Exception: print("Gradio not found.")
    print(f"Using device: {device}")
    print("---------------------------------")
    if not USING_SPACES: main()
    else: print("Launching Gradio app in Spaces mode..."); app = create_gradio_app(); app.queue().launch(debug=True)