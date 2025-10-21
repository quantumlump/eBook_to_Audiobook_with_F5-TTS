import os
import re
import gc
import tempfile
import subprocess
import csv
import time
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

import torch

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

class EbookProgressUpdater:
    """
    A wrapper for the Gradio progress object that provides a clear and
    intuitive progress display for both single and multi-book processing.
    """
    def __init__(self, progress, num_total_chunks, ebook_idx, num_ebooks, start_time):
        self._progress = progress
        self.num_total_chunks = num_total_chunks
        self.ebook_idx = ebook_idx
        self.num_ebooks = num_ebooks
        self.start_time = start_time
        self.current_chunk_idx = 0

    def set_chunk_index(self, i):
        """Sets the index of the current chunk being processed."""
        self.current_chunk_idx = i

    def __call__(self, value, desc=None):
        """
        This method is called to update the progress bar.
        `value` controls the overall progress bar and the final percentage display.
        `desc` contains chunk-specific details from the underlying process.
        """
        # --- Percentage Calculation for the Current Book ---
        chunk_percent = 0.0
        if desc:
            match = re.search(r'(\d+\.?\d*)%', desc)
            if match:
                try: chunk_percent = float(match.group(1))
                except ValueError: chunk_percent = 0.0

        book_progress_fraction = (self.current_chunk_idx + (chunk_percent / 100.0)) / self.num_total_chunks
        book_progress_percent = book_progress_fraction * 100

        # --- Time Calculations ---
        elapsed_seconds = time.time() - self.start_time
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))

        etr_str = "Calculating..."
        if elapsed_seconds > 2 and book_progress_fraction > 0.001:
            try:
                total_estimated_time = elapsed_seconds / book_progress_fraction
                etr_seconds = max(0, total_estimated_time - elapsed_seconds)
                etr_str = time.strftime('%H:%M:%S', time.gmtime(etr_seconds))
            except ZeroDivisionError:
                 etr_str = "Calculating..."

        # --- Construct the Final, Clearer Description String ---
        # The redundant percentage at the end is added by Gradio automatically.
        # This version removes the manual "Overall Progress" from the start
        # to avoid duplication. The percentage at the end now serves as the overall indicator.
        if self.num_ebooks > 1:
            final_desc = (
                f"Current Book: {self.ebook_idx + 1}/{self.num_ebooks} ({book_progress_percent:.1f}%) | "
                f"Chunk {self.current_chunk_idx + 1}/{self.num_total_chunks} | "
                f"Elapsed: {elapsed_str} | ETR: {etr_str}"
            )
        else:
            final_desc = (
                f"Chunk {self.current_chunk_idx + 1}/{self.num_total_chunks} | "
                f"Elapsed: {elapsed_str} | ETR: {etr_str}"
            )

        # Update the Gradio progress bar. The `value` argument controls both the
        # visual bar and the percentage that Gradio appends to `desc`.
        self._progress(value, desc=final_desc)


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
    model.eval()
    model = model.to(device)
    print(f"Model loaded on {device}.")
    return model

F5TTS_ema_model = load_f5tts()

chat_model_state = None
chat_tokenizer_state = None

@gpu_decorator
def generate_response(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    max_new_tokens = 1000000
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
    try:
        cover_path = os.path.splitext(ebook_path)[0] + '.jpg'
        subprocess.run(['ebook-meta', ebook_path, '--get-cover', cover_path], check=True)
        if os.path.exists(cover_path):
            return cover_path
    except Exception as e:
        print(f"Error extracting eBook cover: {e}")
    return None

def embed_metadata_into_mp3(mp3_path, cover_image_path, title, author, album_title=None):
    try:
        audio = ID3(mp3_path)
    except error:
        audio = ID3()
    except Exception as e:
        print(f"Error loading ID3 tags for {mp3_path}: {e}. Creating new tags.")
        audio = ID3()

    audio.delall("TIT2")
    audio.add(TIT2(encoding=3, text=title))
    print(f"Set TIT2 (Title) to: {title}")

    audio.delall("TPE1")
    audio.add(TPE1(encoding=3, text=author))
    print(f"Set TPE1 (Author/Artist) to: {author}")

    album_to_set = album_title if album_title else title
    audio.delall("TALB")
    audio.add(TALB(encoding=3, text=album_to_set))
    print(f"Set TALB (Album) to: {album_to_set}")

    if cover_image_path and os.path.exists(cover_image_path):
        audio.delall("APIC")
        try:
            with open(cover_image_path, 'rb') as img:
                audio.add(APIC(
                    encoding=3, mime='image/jpeg', type=3,
                    desc='Front cover', data=img.read()
                ))
            print(f"Embedded cover image into {mp3_path}")
        except Exception as e:
            print(f"Failed to embed cover image into MP3: {e}")
    else:
        print(f"No cover image provided or found at '{cover_image_path}'. Skipping cover embedding.")
        audio.delall("APIC")

    try:
        audio.save(mp3_path, v2_version=3)
        print(f"Successfully saved metadata to {mp3_path}")
    except Exception as e:
        print(f"Failed to save MP3 metadata: {e}")

def extract_text_and_title_from_epub(epub_path):
    try:
        book = epub.read_epub(epub_path)
        print(f"EPUB '{epub_path}' successfully read.")
    except Exception as e:
        raise RuntimeError(f"Failed to read EPUB file: {e}")

    text_content = []
    title = None
    author = None

    try:
        title_metadata = book.get_metadata('DC', 'title')
        if title_metadata:
            title = title_metadata[0][0]
            print(f"Extracted title: {title}")
        else:
            title = os.path.splitext(os.path.basename(epub_path))[0]
            print(f"No title in metadata. Using filename: {title}")

        author_metadata = book.get_metadata('DC', 'creator')
        if author_metadata:
            author = author_metadata[0][0]
            print(f"Extracted author: {author}")
        else:
            author = "Unknown Author"
            print(f"No author in metadata. Using '{author}'.")
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        if title is None:
            title = os.path.splitext(os.path.basename(epub_path))[0]
            print(f"Using filename as title due to error: {title}")
        if author is None:
            author = "Unknown Author"
            print(f"Using '{author}' due to error in metadata extraction.")

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
    return full_text, title, author

def convert_to_epub(input_path, output_path):
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
    try:
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)
    except Exception as e:
        raise RuntimeError(f"Error detecting file type: {e}")

def ensure_directory(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Error creating directory {directory_path}: {e}")

def sanitize_filename(filename):
    # CORRECTED: Added apostrophe to the list of characters to remove
    sanitized = re.sub(r'[\\/*?:"<>|\']', "", filename)
    return sanitized.replace(" ", "_")

def show_converted_audiobooks():
    output_dir = os.path.join("Working_files", "Book")
    if not os.path.exists(output_dir):
        return ["No audiobooks found."]
    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(('.mp3', '.m4b'))]
    if not files:
        return ["No audiobooks found."]
    return files

@gpu_decorator
def infer(ref_audio_orig, ref_text, gen_text, cross_fade_duration=0.0, speed=1, show_info=gr.Info, progress=gr.Progress(),
          progress_start_fraction=0.0, progress_end_fraction=1.0, ebook_idx=0, num_ebooks=1):
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)
    except Exception as e:
        raise RuntimeError(f"Error in preprocessing reference audio and text: {e}")
    if not gen_text.strip():
        raise ValueError("Generated text is empty. Please provide valid text content.")
    try:
        with torch.no_grad():
            final_wave, final_sample_rate, _ = infer_process(
                ref_audio, ref_text, gen_text, F5TTS_ema_model, vocoder,
                cross_fade_duration=cross_fade_duration, speed=speed,
                show_info=show_info, progress=progress,
                progress_start_fraction=progress_start_fraction,
                progress_end_fraction=progress_end_fraction,
                ebook_idx=ebook_idx, num_ebooks=num_ebooks
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
        ebook_frac = {"init_detect_convert": 0.005, "extract_text": 0.005, "infer": 0.89, "mp3_meta": 0.10}

        for idx, ebook_file_data in enumerate(gen_file_input):
            current_ebook_base_progress = idx / float(num_ebooks)
            progress_offset_within_ebook = 0.0
            original_ebook_path = ebook_file_data.name
            if not os.path.exists(original_ebook_path):
                progress((idx + 1) / float(num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: Skipped (File Not Found)")
                continue

            # --- Stage: File detection and conversion ---
            desc_suffix = "Detecting file type..."
            progress(current_ebook_base_progress, desc=f"Ebook {idx+1}/{num_ebooks}: {desc_suffix}")
            epub_path_for_extraction, temp_epub_created = (original_ebook_path, False)
            file_type = detect_file_type(original_ebook_path)
            if file_type != 'application/epub+zip':
                desc_suffix = "Converting to EPUB..."
                progress(current_ebook_base_progress, desc=f"Ebook {idx+1}/{num_ebooks}: {desc_suffix}")
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
            progress_offset_within_ebook += ebook_frac["init_detect_convert"]
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: File prepped.")

            # --- Stage: Extracting text ---
            desc_suffix = "Extracting text..."
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: {desc_suffix}")
            try:
                gen_text, ebook_title, ebook_author = extract_text_and_title_from_epub(epub_path_for_extraction)
                cover_image = extract_metadata_and_cover(epub_path_for_extraction)
            except Exception as e:
                print(f"Error extracting text/metadata from {epub_path_for_extraction}: {e}. Skipping.")
                progress((idx + 1) / float(num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: Skipped (Text Extraction Error)")
                if temp_epub_created and os.path.exists(epub_path_for_extraction): os.remove(epub_path_for_extraction)
                continue
            ref_text = ref_text_input or ""
            progress_offset_within_ebook += ebook_frac["extract_text"]
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: Text extracted.")

            # --- Stage: Inference and Chunk-based Audio Generation ---
            overall_infer_start_frac = current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks)
            temp_chunks_dir = os.path.join("Working_files", "temp_audio_chunks", sanitize_filename(ebook_title))
            ensure_directory(temp_chunks_dir)
            chunk_file_paths = []

            print("Tokenizing text into sentences for stable chunking...")
            sentences = sent_tokenize(gen_text)
            SENTENCES_PER_CHUNK = 75
            text_super_chunks = [' '.join(sentences[i:i + SENTENCES_PER_CHUNK]) for i in range(0, len(sentences), SENTENCES_PER_CHUNK)]
            num_super_chunks = len(text_super_chunks)

            if num_super_chunks == 0:
                print(f"Error: No text chunks could be created from {ebook_title}. Skipping.")
                continue

            print(f"Book text split into {num_super_chunks} sentence-based super-chunks for processing.")

            ebook_start_time = time.time()
            progress_updater = EbookProgressUpdater(
                progress=progress,
                num_total_chunks=num_super_chunks,
                ebook_idx=idx,
                num_ebooks=num_ebooks,
                start_time=ebook_start_time
            )

            try:
                for i, text_chunk in enumerate(text_super_chunks):
                    progress_updater.set_chunk_index(i)

                    chunk_progress_start = overall_infer_start_frac + (i / num_super_chunks) * (ebook_frac["infer"] / num_ebooks)
                    chunk_progress_end = overall_infer_start_frac + ((i + 1) / num_super_chunks) * (ebook_frac["infer"] / num_ebooks)

                    progress_updater(chunk_progress_start)
                    
                    # CORRECTED: Added try...finally for robust memory cleanup
                    wave_chunk = None # Initialize to None
                    try:
                        audio_out_chunk, _ = infer(
                            ref_audio_input, ref_text, text_chunk,
                            cross_fade_duration, speed, show_info=gr.Info,
                            progress=progress_updater,
                            ebook_idx=idx, num_ebooks=num_ebooks,
                            progress_start_fraction=chunk_progress_start,
                            progress_end_fraction=chunk_progress_end
                        )
                        sample_rate_chunk, wave_chunk = audio_out_chunk
                    finally:
                        # This cleanup now runs even if `infer` throws an error
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    if wave_chunk is not None and wave_chunk.any():
                        chunk_path = os.path.join(temp_chunks_dir, f"chunk_{i:04d}.wav")
                        sf.write(chunk_path, wave_chunk, sample_rate_chunk)
                        chunk_file_paths.append(chunk_path)
                    else:
                        print(f"Warning: Empty audio returned for super-chunk {i+1}. Skipping this part.")
            except Exception as e:
                print(f"Error during TTS inference loop for {ebook_title}: {e}")
                continue

            if not chunk_file_paths:
                print(f"Error: No audio chunks were generated for {ebook_title}. Skipping book.")
                continue

            progress_offset_within_ebook += ebook_frac["infer"]
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: Audio synthesized.")

            # --- Stage: MP3 Conversion & Metadata using FFmpeg Concat ---
            desc_suffix = "Finalizing MP3 & adding metadata..."
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: {desc_suffix}")
            sanitized_title = sanitize_filename(ebook_title) or f"audiobook_{idx}"
            final_mp3_dir = os.path.join("Working_files", "Book")
            final_mp3_path = os.path.join(final_mp3_dir, f"{sanitized_title}.mp3")

            # CORRECTED: Entire FFmpeg block replaced for robustness
            try:
                concat_list_path = os.path.join(temp_chunks_dir, "concat_list.txt")
                with open(concat_list_path, 'w') as f:
                    for path in chunk_file_paths:
                        # Use relative paths for robustness
                        chunk_filename = os.path.basename(path)
                        f.write(f"file '{chunk_filename}'\n")

                # Ensure the final output directory exists before running FFmpeg
                ensure_directory(final_mp3_dir)

                # Command uses relative paths, so it must be run from the chunks directory
                ffmpeg_command = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'concat_list.txt',
                    '-b:a', '192k', '-y', os.path.abspath(final_mp3_path) # Use absolute path for the output file
                ]

                print(f"Starting FFmpeg concatenation for {len(chunk_file_paths)} chunks...")
                print(f"Executing command: {' '.join(ffmpeg_command)} in directory {temp_chunks_dir}")

                # Use Popen for better process control and logging on long tasks
                process = subprocess.Popen(
                    ffmpeg_command,
                    cwd=temp_chunks_dir,  # CRITICAL: Execute FFmpeg in the chunks directory
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8'
                )

                # Stream stdout and stderr to print progress and errors in real-time
                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    print("--- FFmpeg Error ---")
                    print("Return Code:", process.returncode)
                    print("\n--- FFmpeg STDOUT ---")
                    print(stdout)
                    print("\n--- FFmpeg STDERR ---")
                    print(stderr)
                    raise gr.Error(f"FFmpeg failed for {ebook_title}. Check console for detailed logs.")
                else:
                    print("FFmpeg concatenation successful.")
                    embed_metadata_into_mp3(final_mp3_path, cover_image, ebook_title, ebook_author, album_title=ebook_title)

            except Exception as e:
                print(f"An error occurred during the FFmpeg/metadata stage for {ebook_title}: {e}")
                continue # Skip to the next book
            finally:
                import shutil
                if os.path.exists(temp_chunks_dir):
                    print(f"Cleaning up temporary directory: {temp_chunks_dir}")
                    shutil.rmtree(temp_chunks_dir)


            progress_offset_within_ebook += ebook_frac["mp3_meta"]
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: MP3 created.")

            # --- Final Cleanup for this book ---
            if cover_image and os.path.exists(cover_image):
                try: os.remove(cover_image)
                except OSError as e: print(f"Error removing temp cover {cover_image}: {e}")
            if temp_epub_created and os.path.exists(epub_path_for_extraction):
                try: os.remove(epub_path_for_extraction)
                except OSError as e: print(f"Error removing temp EPUB {epub_path_for_extraction}: {e}")

            processed_audiobooks.append(final_mp3_path)
            final_ebook_progress = (idx + 1) / float(num_ebooks)
            progress(final_ebook_progress, desc=f"Ebook {idx+1}/{num_ebooks}: Completed.")
            yield final_mp3_path, processed_audiobooks

        if num_ebooks > 0 and not processed_audiobooks and locals().get('idx') == num_ebooks - 1:
            progress(1.0, desc="All eBooks skipped or failed processing.")
        elif processed_audiobooks:
            progress(1.0, desc=f"All {num_ebooks} eBook(s) processing finished.")

    except Exception as e:
        print(f"An unhandled error occurred in basic_tts: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"An error occurred: {str(e)}")

DEFAULT_REF_AUDIO_PATH = "/app/default_voice.mp3"
DEFAULT_REF_TEXT = "For thirty-six years I was the confidential secretary of the Roman statesman Cicero. At first this was exciting, then astonishing, then arduous, and finally extremely dangerous."

def create_gradio_app():
    """Create and configure the Gradio application."""
    with gr.Blocks(theme=gr.themes.Ocean()) as app:
        gr.Markdown("# eBook to Audiobook with F5-TTS!")
        ref_audio_input = gr.Audio(
            label="Upload Voice File (<15 sec) or Record with Mic Icon (Ensure Natural Phrasing, Trim Silence)",
            type="filepath", value=DEFAULT_REF_AUDIO_PATH
        )
        gen_file_input = gr.Files(
            label="Upload eBook or Multiple for Batch Processing (epub, mobi, pdf, txt, html)",
            file_types=[".epub", ".mobi", ".pdf", ".txt", ".html"],
            file_count="multiple",
        )
        generate_btn = gr.Button("Start", variant="primary")
        show_audiobooks_btn = gr.Button("Show All Completed Audiobooks", variant="secondary")
        audiobooks_output = gr.Files(label="Converted Audiobooks (Download Links)")
        player = gr.Audio(label="Play Latest Converted Audiobook", interactive=False)
        with gr.Accordion("Advanced Settings", open=False):
            ref_text_input = gr.Textbox(
                label="Reference Text (Leave Blank for Automatic Transcription)",
                lines=2, value=DEFAULT_REF_TEXT
            )
            speed_slider = gr.Slider(
                label="Speech Speed (Adjusting Can Cause Artifacts)",
                minimum=0.3, maximum=2.0, value=1.0, step=0.1,
            )
            cross_fade_duration_slider = gr.Slider(
                label="Cross-Fade Duration (Between Generated Audio Chunks)",
                minimum=0.0, maximum=1.0, value=0.0, step=0.01,
            )
        generate_btn.click(
            basic_tts,
            inputs=[ref_audio_input, ref_text_input, gen_file_input, cross_fade_duration_slider, speed_slider],
            outputs=[player, audiobooks_output],
        )
        show_audiobooks_btn.click(
            show_converted_audiobooks, inputs=[], outputs=[audiobooks_output],
        )
    return app

@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option("--share", "-s", default=False, is_flag=True, help="Share the app via Gradio share link")
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
def main(port, host, share, api):
    """Main entry point to launch the Gradio app."""
    app = create_gradio_app()
    print("Starting app...")
    app.queue().launch(
        server_name="0.0.0.0", server_port=port or 7860,
        share=share, show_api=api, debug=True
    )

if __name__ == "__main__":
    import sys
    print("Arguments passed to Python:", sys.argv)
    if not USING_SPACES:
        main()
    else:
        app = create_gradio_app()
        app.queue().launch(debug=True)
