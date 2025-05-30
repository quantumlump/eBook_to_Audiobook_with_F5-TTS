import os
import re
import tempfile
import subprocess
import csv
from collections import OrderedDict
from importlib.resources import files

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
import magic
from mutagen.id3 import ID3, APIC, TIT2, TPE1, TALB, error

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
)

import torch  # Added missing import

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

# GPU Decorator
def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    return func

# Load models
vocoder = load_vocoder()

def load_f5tts(ckpt_path=None):
    if ckpt_path is None:
        ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
    model_cfg = {
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "conv_layers": 4
    }
    model = load_model(DiT, model_cfg, ckpt_path)
    model.eval()  # Ensure the model is in evaluation mode
    model = model.to(device)  # Move model to the selected device
    print(f"Model loaded on {device}.")
    return model

F5TTS_ema_model = load_f5tts()

chat_model_state = None
chat_tokenizer_state = None

@gpu_decorator
def generate_response(messages, model, tokenizer):
    """Generate a response using the provided model and tokenizer with full precision."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Increase max_new_tokens to a much larger number to avoid truncation
    # Previously: max_new_tokens=1024
    max_new_tokens = 1000000  # Large number to allow full generation

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
        )

    if not generated_ids:
        raise ValueError("No generated IDs returned by the model.")

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    if not generated_ids or not generated_ids[0]:
        raise ValueError("Generated IDs are empty after processing.")

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def extract_metadata_and_cover(ebook_path):
    """Extract cover image from the eBook."""
    try:
        cover_path = os.path.splitext(ebook_path)[0] + '.jpg'
        subprocess.run(['ebook-meta', ebook_path, '--get-cover', cover_path], check=True)
        if os.path.exists(cover_path):
            return cover_path
    except Exception as e:
        print(f"Error extracting eBook cover: {e}")
    return None

def embed_metadata_into_mp3(mp3_path, cover_image_path, title, author, album_title=None):
    """Embed cover image, title, author, and album into the MP3 file's metadata."""
    try:
        audio = ID3(mp3_path)
    except error: # If no ID3 tag exists, create one
        audio = ID3()
    except Exception as e:
        print(f"Error loading ID3 tags for {mp3_path}: {e}. Creating new tags.")
        audio = ID3() # Fallback to creating new tags

    # Add Title
    audio.delall("TIT2") # Remove existing title tags
    audio.add(TIT2(encoding=3, text=title))
    print(f"Set TIT2 (Title) to: {title}")

    # Add Author/Artist
    audio.delall("TPE1") # Remove existing artist tags
    audio.add(TPE1(encoding=3, text=author))
    print(f"Set TPE1 (Author/Artist) to: {author}")

    # Add Album (often used for Book Title in audiobook players)
    # If a specific album_title is provided, use it. Otherwise, use the main title.
    album_to_set = album_title if album_title else title
    audio.delall("TALB") # Remove existing album tags
    audio.add(TALB(encoding=3, text=album_to_set))
    print(f"Set TALB (Album) to: {album_to_set}")

    # Embed Cover Image (existing logic)
    if cover_image_path and os.path.exists(cover_image_path):
        audio.delall("APIC") # Remove existing cover art
        try:
            with open(cover_image_path, 'rb') as img:
                audio.add(APIC(
                    encoding=3,          # 3 is for UTF-8
                    mime='image/jpeg',   # or 'image/png'
                    type=3,              # 3 means front cover
                    desc='Front cover',
                    data=img.read()
                ))
            print(f"Embedded cover image into {mp3_path}")
        except Exception as e:
            print(f"Failed to embed cover image into MP3: {e}")
    else:
        print(f"No cover image provided or found at '{cover_image_path}'. Skipping cover embedding.")
        audio.delall("APIC") # Ensure no old APIC tag remains if new cover is not set

    try:
        # Save with ID3v2.3 for broad compatibility
        audio.save(mp3_path, v2_version=3)
        print(f"Successfully saved metadata to {mp3_path}")
    except Exception as e:
        print(f"Failed to save MP3 metadata: {e}")

def extract_text_and_title_from_epub(epub_path):
    """Extract full text, title, and author from an EPUB file in reading order."""
    try:
        book = epub.read_epub(epub_path)
        print(f"EPUB '{epub_path}' successfully read.")
    except Exception as e:
        raise RuntimeError(f"Failed to read EPUB file: {e}")

    text_content = []
    title = None
    author = None # Initialize author

    try:
        # Extract title
        title_metadata = book.get_metadata('DC', 'title')
        if title_metadata:
            title = title_metadata[0][0]
            print(f"Extracted title: {title}")
        else:
            title = os.path.splitext(os.path.basename(epub_path))[0]
            print(f"No title in metadata. Using filename: {title}")

        # Extract author
        author_metadata = book.get_metadata('DC', 'creator')
        if author_metadata:
            # Assuming the first creator is the primary author
            author = author_metadata[0][0]
            print(f"Extracted author: {author}")
        else:
            author = "Unknown Author" # Default if no author found
            print(f"No author in metadata. Using '{author}'.")

    except Exception as e: # Catch potential errors during metadata extraction
        print(f"Error extracting metadata: {e}")
        if title is None: # Ensure title has a fallback
            title = os.path.splitext(os.path.basename(epub_path))[0]
            print(f"Using filename as title due to error: {title}")
        if author is None: # Ensure author has a fallback
            author = "Unknown Author"
            print(f"Using '{author}' due to error in metadata extraction.")


    # Iterate over the book's spine in reading order
    for spine_item in book.spine:
        item = book.get_item_with_id(spine_item[0])
        if item and item.get_type() == ITEM_DOCUMENT:
            try:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                if text:
                    text_content.append(text)
            except Exception as e:
                print(f"Error parsing document item {item.get_id()}: {e}")

    full_text = ' '.join(text_content)

    if not full_text:
        raise ValueError("No text found in EPUB file.")

    print(f"Extracted {len(full_text)} characters from EPUB.")
    return full_text, title, author # Return author

def convert_to_epub(input_path, output_path):
    """Convert an ebook to EPUB format using Calibre."""
    try:
        ensure_directory(os.path.dirname(output_path))
        subprocess.run(['ebook-convert', input_path, output_path], check=True)
        print(f"Converted {input_path} to EPUB.")
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error converting eBook: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during conversion: {e}")

def detect_file_type(file_path):
    """Detect the MIME type of a file."""
    try:
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)
    except Exception as e:
        raise RuntimeError(f"Error detecting file type: {e}")

def ensure_directory(directory_path):
    """Ensure that a directory exists."""
    try:
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Error creating directory {directory_path}: {e}")

def sanitize_filename(filename):
    """Sanitize a filename by removing invalid characters."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    return sanitized.replace(" ", "_")

def show_converted_audiobooks():
    """List all converted audiobook files."""
    output_dir = os.path.join("Working_files", "Book")
    if not os.path.exists(output_dir):
        return ["No audiobooks found."]

    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(('.mp3', '.m4b'))]
    if not files:
        return ["No audiobooks found."]

    return files

@gpu_decorator
def infer(ref_audio_orig, ref_text, gen_text, cross_fade_duration=0.0, speed=1, show_info=gr.Info, progress=gr.Progress(),
          progress_start_fraction=0.0, progress_end_fraction=1.0, ebook_idx=0, num_ebooks=1): # Added ebook_idx, num_ebooks
    """Perform inference to generate audio from text without truncation."""
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)
    except Exception as e:
        raise RuntimeError(f"Error in preprocessing reference audio and text: {e}")

    if not gen_text.strip():
        raise ValueError("Generated text is empty. Please provide valid text content.")

    try:
        with torch.no_grad():
            final_wave, final_sample_rate, _ = infer_process(
                ref_audio,
                ref_text,
                gen_text,
                F5TTS_ema_model,
                vocoder,
                cross_fade_duration=cross_fade_duration,
                speed=speed,
                show_info=show_info,
                progress=progress,
                progress_start_fraction=progress_start_fraction,
                progress_end_fraction=progress_end_fraction,
                ebook_idx=ebook_idx,        # Pass down
                num_ebooks=num_ebooks       # Pass down
            )
    except Exception as e:
        raise RuntimeError(f"Error during inference process: {e}")

    print(f"Generated audio length: {len(final_wave)} samples at {final_sample_rate} Hz.")
    return (final_sample_rate, final_wave), ref_text

@gpu_decorator
def basic_tts(ref_audio_input, ref_text_input, gen_file_input, cross_fade_duration, speed, progress=gr.Progress()):
    try:
        processed_audiobooks = []
        num_ebooks = len(gen_file_input)

        # Define relative durations (fractions) for stages within ONE ebook's processing cycle.
        # These should sum to 1.0 for one ebook.
        ebook_frac = {
            "init_detect_convert": 0.005,  # 2% for initial setup, detection, and potential conversion
            "extract_text":        0.005,  # 2% for text extraction
            "infer":               0.89,  # 86% for the main TTS inference (bulk of the time)
            "stitch":              0.03,  # 3% for audio stitching
            "mp3_meta":            0.07,  # 7% for MP3 conversion and metadata embedding
            # Total: 0.02 + 0.02 + 0.86 + 0.03 + 0.07 = 1.00
        }

        for idx, ebook_file_data in enumerate(gen_file_input):
            current_ebook_base_progress = idx / float(num_ebooks)
            
            # This variable will track the progress offset *within the current ebook's allocated slot*
            # It represents how much of the current ebook's 1/num_ebooks fraction is completed by prior stages.
            progress_offset_within_ebook = 0.0

            original_ebook_path = ebook_file_data.name
            if not os.path.exists(original_ebook_path):
                progress((idx + 1) / float(num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: Skipped (File Not Found: {os.path.basename(original_ebook_path)})")
                continue 

            epub_path_for_extraction = original_ebook_path
            temp_epub_created = False

            # --- Stage: File detection and conversion ---
            desc_suffix = "Detecting file type..."
            # Show progress at the very start of this stage for this ebook
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), 
                     desc=f"Ebook {idx+1}/{num_ebooks}: {desc_suffix}")
            
            file_type = detect_file_type(original_ebook_path)
            if file_type != 'application/epub+zip':
                desc_suffix = "Converting to EPUB..."
                progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), 
                         desc=f"Ebook {idx+1}/{num_ebooks}: {desc_suffix}")
                sanitized_base = sanitize_filename(os.path.splitext(os.path.basename(original_ebook_path))[0])
                temp_epub_dir = os.path.join("Working_files", "temp_converted")
                ensure_directory(temp_epub_dir)
                temp_epub = os.path.join(temp_epub_dir, f"{sanitized_base}.epub")
                try:
                    convert_to_epub(original_ebook_path, temp_epub)
                    epub_path_for_extraction = temp_epub
                    temp_epub_created = True
                except Exception as e:
                    print(f"Error converting {original_ebook_path} to EPUB: {e}. Skipping.")
                    progress((idx + 1) / float(num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: Skipped (Conversion Error)")
                    continue 
            
            # Update progress offset after this stage
            progress_offset_within_ebook += ebook_frac["init_detect_convert"]
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), 
                     desc=f"Ebook {idx+1}/{num_ebooks}: File prepped.")

            # --- Stage: Extracting text ---
            desc_suffix = "Extracting text..."
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), 
                     desc=f"Ebook {idx+1}/{num_ebooks}: {desc_suffix}")
            
            try:
                gen_text, ebook_title, ebook_author = extract_text_and_title_from_epub(epub_path_for_extraction)
                cover_image = extract_metadata_and_cover(epub_path_for_extraction) 
            except Exception as e:
                print(f"Error extracting text/metadata from {epub_path_for_extraction}: {e}. Skipping.")
                progress((idx + 1) / float(num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: Skipped (Text Extraction Error)")
                if temp_epub_created and os.path.exists(epub_path_for_extraction): os.remove(epub_path_for_extraction)
                continue

            ref_text = ref_text_input or ""
            
            # Update progress offset after this stage
            progress_offset_within_ebook += ebook_frac["extract_text"]
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), 
                     desc=f"Ebook {idx+1}/{num_ebooks}: Text extracted.")

            # --- Stage: Inference ---
            # `overall_infer_start_frac` is the progress point where inference begins.
            # It's the base progress for this ebook + what's consumed by prior stages within this ebook's slot.
            overall_infer_start_frac = current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks)
            
            # `overall_infer_end_frac` is where inference is expected to end.
            overall_infer_end_frac = overall_infer_start_frac + (ebook_frac["infer"] / num_ebooks)
            
            # This message appears just before calling infer().
            # The `utils_infer.py` will then immediately show its "0.0%" message for chunk processing.
            progress(overall_infer_start_frac, desc=f"Ebook {idx+1}/{num_ebooks}: Preparing audio synthesis...")

            try:
                audio_out, _ = infer(
                    ref_audio_input,
                    ref_text,
                    gen_text,
                    cross_fade_duration,
                    speed,
                    show_info=gr.Info,
                    progress=progress,
                    ebook_idx=idx,
                    num_ebooks=num_ebooks,
                    progress_start_fraction=overall_infer_start_frac, # Pass the calculated start
                    progress_end_fraction=overall_infer_end_frac     # Pass the calculated end
                )
            except Exception as e:
                print(f"Error during TTS inference for {ebook_title}: {e}. Skipping.")
                progress((idx + 1) / float(num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: Skipped (Inference Error)")
                if temp_epub_created and os.path.exists(epub_path_for_extraction): os.remove(epub_path_for_extraction)
                if cover_image and os.path.exists(cover_image): os.remove(cover_image)
                continue
            
            # Update progress offset after inference.
            # The `infer` function itself will drive progress between overall_infer_start_frac and overall_infer_end_frac.
            # So, after it finishes, progress_offset_within_ebook should be at the end of inference's allocation.
            progress_offset_within_ebook += ebook_frac["infer"]
            # progress() call here is implicitly handled by the end of the infer() function.

            # --- Stage: Stitching ---
            desc_suffix = "Stitching audio..."
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), 
                     desc=f"Ebook {idx+1}/{num_ebooks}: {desc_suffix}")
            
            sample_rate, wave = audio_out
            if not wave.any(): # Check if wave is empty
                print(f"Warning: Generated audio wave is empty for {ebook_title}. Skipping.")
                progress((idx + 1) / float(num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: Skipped (Empty Audio)")
                if temp_epub_created and os.path.exists(epub_path_for_extraction): os.remove(epub_path_for_extraction)
                if cover_image and os.path.exists(cover_image): os.remove(cover_image)
                continue

            temp_audio_dir = os.path.join("Working_files", "temp_audio")
            ensure_directory(temp_audio_dir)
            tmp_wav_path = "" # Initialize
            try:
                with tempfile.NamedTemporaryFile(dir=temp_audio_dir, delete=False, suffix=".wav") as tmp_wav:
                    sf.write(tmp_wav.name, wave, sample_rate)
                    tmp_wav_path = tmp_wav.name
            except Exception as e:
                print(f"Error writing temporary WAV for {ebook_title}: {e}. Skipping.")
                progress((idx + 1) / float(num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: Skipped (WAV Error)")
                if temp_epub_created and os.path.exists(epub_path_for_extraction): os.remove(epub_path_for_extraction)
                if cover_image and os.path.exists(cover_image): os.remove(cover_image)
                continue
            
            # Update progress offset
            progress_offset_within_ebook += ebook_frac["stitch"]
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), 
                     desc=f"Ebook {idx+1}/{num_ebooks}: Audio stitched.")

            # --- Stage: MP3 Conversion & Metadata ---
            desc_suffix = "Converting to MP3 & adding metadata..."
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), 
                     desc=f"Ebook {idx+1}/{num_ebooks}: {desc_suffix}")

            sanitized_title = sanitize_filename(ebook_title) or f"audiobook_{idx}"
            final_mp3_dir = os.path.join("Working_files", "Book")
            ensure_directory(final_mp3_dir)
            tmp_mp3_path = os.path.join(final_mp3_dir, f"{sanitized_title}.mp3")

            try:
                audio = AudioSegment.from_wav(tmp_wav_path)
                audio.export(tmp_mp3_path, format="mp3", bitrate="320k", parameters=["-q:a", "0"]) # Consider "192k" for smaller files
                embed_metadata_into_mp3(tmp_mp3_path, cover_image, ebook_title, ebook_author, album_title=ebook_title)
            except Exception as e:
                print(f"Error during MP3 conversion/metadata for {ebook_title}: {e}. Skipping.")
                progress((idx + 1) / float(num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: Skipped (MP3/Meta Error)")
                if os.path.exists(tmp_wav_path): os.remove(tmp_wav_path)
                if temp_epub_created and os.path.exists(epub_path_for_extraction): os.remove(epub_path_for_extraction)
                if cover_image and os.path.exists(cover_image): os.remove(cover_image)
                continue

            # Update progress offset
            progress_offset_within_ebook += ebook_frac["mp3_meta"]
            # Final progress update for this ebook should reach the end of its allocated slot
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), 
                     desc=f"Ebook {idx+1}/{num_ebooks}: MP3 created.")

            # Cleanup
            if os.path.exists(tmp_wav_path):
                try: os.remove(tmp_wav_path)
                except OSError as e: print(f"Error removing temp WAV {tmp_wav_path}: {e}")
            if cover_image and os.path.exists(cover_image):
                try: os.remove(cover_image)
                except OSError as e: print(f"Error removing temp cover {cover_image}: {e}")
            if temp_epub_created and os.path.exists(epub_path_for_extraction):
                try: os.remove(epub_path_for_extraction)
                except OSError as e: print(f"Error removing temp EPUB {epub_path_for_extraction}: {e}")

            processed_audiobooks.append(tmp_mp3_path)
            # Ensure progress reaches the end for this ebook; progress_offset_within_ebook should be ~1.0 here.
            # If ebook_frac sums to 1.0, then current_ebook_base_progress + (1.0 / num_ebooks) is (idx+1)/num_ebooks
            final_ebook_progress = (idx + 1) / float(num_ebooks)
            progress(final_ebook_progress, desc=f"Ebook {idx+1}/{num_ebooks}: Completed.")
            yield tmp_mp3_path, processed_audiobooks
        
        # Final progress update if all ebooks processed successfully
        if num_ebooks > 0 and not processed_audiobooks and idx == num_ebooks -1 : # All skipped
             progress(1.0, desc="All eBooks skipped or failed processing.")
        elif processed_audiobooks : # At least one processed
             progress(1.0, desc=f"All {num_ebooks} eBook(s) processing finished.")


    except Exception as e:
        print(f"An error occurred in basic_tts: {e}")
        import traceback
        traceback.print_exc()
        if progress and hasattr(progress, '__call__'): 
            try:
                progress_val_on_error = 0.0
                if 'idx' in locals() and 'num_ebooks' in locals() and num_ebooks > 0:
                     progress_val_on_error = (idx / float(num_ebooks)) 
                current_progress_desc = f"Error processing"
                if 'idx' in locals() and 'num_ebooks' in locals():
                    current_progress_desc = f"Ebook {idx+1}/{num_ebooks}: Error"
                progress(progress_val_on_error, desc=f"{current_progress_desc} - {str(e)[:100]}. Check logs.") # Truncate error
            except Exception as pe: 
                print(f"Error updating progress during exception handling: {pe}")
        raise gr.Error(f"An error occurred: {str(e)}")

DEFAULT_REF_AUDIO_PATH = "/app/default_voice.mp3" 
DEFAULT_REF_TEXT = "For thirty-six years I was the confidential secretary of the Roman statesman Cicero. At first this was exciting, then astonishing, then arduous, and finally extremely dangerous."


def create_gradio_app():
    """Create and configure the Gradio application."""
    with gr.Blocks(theme=gr.themes.Ocean()) as app: # Use gr.themes.Ocean if available, else None
        gr.Markdown("# eBook to Audiobook with F5-TTS!")

        ref_audio_input = gr.Audio(
            label="Upload Voice File (<15 sec) or Record with Mic Icon (Ensure Natural Phrasing, Trim Silence)",
            type="filepath",
            value=DEFAULT_REF_AUDIO_PATH
        )

        gen_file_input = gr.Files(
            label="Upload eBook or Multiple for Batch Processing (epub, mobi, pdf, txt, html)",
            file_types=[".epub", ".mobi", ".pdf", ".txt", ".html"],
            # Assuming 'filepath' was intended here based on previous context,
            # but if it's causing issues or not what you want, adjust as needed.
            # Gradio's gr.Files often works with a list of file paths or temp files.
            # type="filepath", # This might be redundant if file_count="multiple" handles it
            file_count="multiple",
        )

        # Buttons are now stacked vertically. full_width removed for compatibility.
        generate_btn = gr.Button("Start", variant="primary")
        show_audiobooks_btn = gr.Button("Show All Completed Audiobooks", variant="secondary")

        audiobooks_output = gr.Files(label="Converted Audiobooks (Download Links)")
        player = gr.Audio(label="Play Latest Converted Audiobook", interactive=False)

        with gr.Accordion("Advanced Settings", open=False):
            ref_text_input = gr.Textbox(
                label="Reference Text (Leave Blank for Automatic Transcription)",
                lines=2,
                value=DEFAULT_REF_TEXT
            )
            speed_slider = gr.Slider(
                label="Speech Speed (Adjusting Can Cause Artifacts)",
                minimum=0.3,
                maximum=2.0,
                value=1.0,
                step=0.1,
            )
            cross_fade_duration_slider = gr.Slider(
                label="Cross-Fade Duration (Between Generated Audio Chunks)",
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.01,
            )

        generate_btn.click(
            basic_tts,
            inputs=[
                ref_audio_input,
                ref_text_input,
                gen_file_input,
                cross_fade_duration_slider,
                speed_slider,
            ],
            outputs=[player, audiobooks_output],
            
        )

        show_audiobooks_btn.click(
            show_converted_audiobooks,
            inputs=[],
            outputs=[audiobooks_output],
        )

    return app

@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
def main(port, host, share, api):
    """Main entry point to launch the Gradio app."""
    app = create_gradio_app()
    print("Starting app...")
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=port or 7860,
        share=share,
        show_api=api,
        debug=True
    )

if __name__ == "__main__":
    import sys
    print("Arguments passed to Python:", sys.argv)
    if not USING_SPACES:
        main()
    else:
        app = create_gradio_app()
        app.queue().launch(debug=True)
