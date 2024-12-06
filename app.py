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
from mutagen.id3 import ID3, APIC, error

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

def embed_cover_into_mp3(mp3_path, cover_image_path):
    """Embed a cover image into the MP3 file's metadata."""
    try:
        audio = ID3(mp3_path)
    except error:
        audio = ID3()

    audio.delall("APIC")

    try:
        with open(cover_image_path, 'rb') as img:
            audio.add(APIC(
                encoding=3,          
                mime='image/jpeg',   
                type=3,              
                desc='Front cover',  
                data=img.read()
            ))
        audio.save(mp3_path, v2_version=3)
        print(f"Embedded cover image into {mp3_path}")
    except Exception as e:
        print(f"Failed to embed cover image into MP3: {e}")

def extract_text_and_title_from_epub(epub_path):
    """Extract full text and title from an EPUB file in reading order."""
    try:
        book = epub.read_epub(epub_path)
        print(f"EPUB '{epub_path}' successfully read.")
    except Exception as e:
        raise RuntimeError(f"Failed to read EPUB file: {e}")

    text_content = []
    title = None

    try:
        metadata = book.get_metadata('DC', 'title')
        if metadata:
            title = metadata[0][0]
            print(f"Extracted title: {title}")
        else:
            title = os.path.splitext(os.path.basename(epub_path))[0]
            print(f"No title in metadata. Using filename: {title}")
    except Exception:
        title = os.path.splitext(os.path.basename(epub_path))[0]
        print(f"Using filename as title: {title}")

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
    return full_text, title

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
def infer(ref_audio_orig, ref_text, gen_text, cross_fade_duration=0.0, speed=1, show_info=gr.Info, progress=gr.Progress()):
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
            )
    except Exception as e:
        raise RuntimeError(f"Error during inference process: {e}")

    # Log the length of the generated audio for debugging
    print(f"Generated audio length: {len(final_wave)} samples at {final_sample_rate} Hz.")

    return (final_sample_rate, final_wave), ref_text

@gpu_decorator
def basic_tts(ref_audio_input, ref_text_input, gen_file_input, cross_fade_duration, speed, progress=gr.Progress()):
    """Main function to convert eBooks to audiobooks with full text processing."""
    try:
        processed_audiobooks = []
        num_ebooks = len(gen_file_input)
        for idx, ebook in enumerate(gen_file_input):
            progress(0, desc=f"Processing ebook {idx+1}/{num_ebooks}")
            epub_path = ebook
            if not os.path.exists(epub_path):
                raise FileNotFoundError(f"File not found: {epub_path}")

            file_type = detect_file_type(epub_path)
            if file_type != 'application/epub+zip':
                sanitized_base = sanitize_filename(os.path.splitext(os.path.basename(epub_path))[0])
                temp_epub = os.path.join("Working_files", "temp_converted", f"{sanitized_base}.epub")
                convert_to_epub(ebook, temp_epub)
                epub_path = temp_epub

            progress(0.1, desc="Extracting text and title from EPUB")
            gen_text, ebook_title = extract_text_and_title_from_epub(epub_path)
            cover_image = extract_metadata_and_cover(epub_path)

            ref_text = ref_text_input or ""

            progress(0.2, desc="Starting inference")
            audio_out, _ = infer(
                ref_audio_input,
                ref_text,
                gen_text,
                cross_fade_duration,
                speed,
                progress=progress,
            )

            progress(0.8, desc="Stitching audio files")
            sample_rate, wave = audio_out
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                sf.write(tmp_wav.name, wave, sample_rate)
                tmp_wav_path = tmp_wav.name

            progress(0.9, desc="Converting to MP3")
            sanitized_title = sanitize_filename(ebook_title) or f"audiobook_{int(tempfile._get_default_tempdir())}"
            tmp_mp3_path = os.path.join("Working_files", "Book", f"{sanitized_title}.mp3")
            ensure_directory(os.path.dirname(tmp_mp3_path))

            audio = AudioSegment.from_wav(tmp_wav_path)
            audio.export(
                tmp_mp3_path,
                format="mp3",
                bitrate="320k",
                parameters=["-q:a", "0"]  
            )

            if cover_image:
                embed_cover_into_mp3(tmp_mp3_path, cover_image)

            os.remove(tmp_wav_path)
            if cover_image and os.path.exists(cover_image):
                os.remove(cover_image)

            processed_audiobooks.append(tmp_mp3_path)
            progress(1, desc=f"Completed processing ebook {idx+1}/{num_ebooks}")

            yield tmp_mp3_path, processed_audiobooks

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

def create_gradio_app():
    """Create and configure the Gradio application."""
    with gr.Blocks(theme=gr.themes.Ocean()) as app:
        gr.Markdown("# eBook to Audiobook with F5-TTS!")

        ref_audio_input = gr.Audio(
            label="Upload Voice File (<15 sec) or Record with Mic Icon (Ensure Natural Phrasing, Trim Silence)",
            type="filepath"
        )

        gen_file_input = gr.Files(
            label="Upload eBook or Multiple for Batch Processing (epub, mobi, pdf, txt, html)",
            file_types=[".epub", ".mobi", ".pdf", ".txt", ".html"],
            type="filepath",
            file_count="multiple",
        )

        with gr.Row():
            generate_btn = gr.Button("Start", variant="primary")
            show_audiobooks_btn = gr.Button("Show All Completed Audiobooks", variant="secondary")

        audiobooks_output = gr.Files(label="Converted Audiobooks (Download Links)")
        player = gr.Audio(label="Play Latest Converted Audiobook", interactive=False)

        with gr.Accordion("Advanced Settings", open=False):
            ref_text_input = gr.Textbox(
                label="Reference Text (Leave Blank for Automatic Transcription)",
                lines=2,
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
            show_progress=True,
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
