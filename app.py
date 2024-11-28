<<<<<<< HEAD
import os
import re
import tempfile
from collections import OrderedDict
from importlib.resources import files

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    # Removed save_spectrogram as it's no longer needed
)

# Corrected imports for EPUB processing
from ebooklib import epub, ITEM_DOCUMENT  # Updated import for ITEM_DOCUMENT
from bs4 import BeautifulSoup
import subprocess
import csv
import nltk
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
import magic  # For file type detection

DEFAULT_TTS_MODEL = "F5-TTS"

# Load models

vocoder = load_vocoder()


def load_f5tts(ckpt_path=str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))):
    F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


F5TTS_ema_model = load_f5tts()

chat_model_state = None
chat_tokenizer_state = None


@gpu_decorator
def generate_response(messages, model, tokenizer):
    """Generate response using Qwen"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        input_features=model_inputs.input_features,  # Updated from **model_inputs
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    # Check if generated_ids is not empty
    if not generated_ids:
        raise ValueError("No generated IDs returned by the model.")

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Ensure there is at least one generated_id
    if not generated_ids or not generated_ids[0]:
        raise ValueError("Generated IDs are empty after processing.")

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


@gpu_decorator
def infer(
    ref_audio_orig, ref_text, gen_text, remove_silence, cross_fade_duration=0.15, speed=1, show_info=gr.Info
):
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)
    except Exception as e:
        raise RuntimeError(f"Error in preprocessing reference audio and text: {e}")

    # Always use F5-TTS
    ema_model = F5TTS_ema_model

    # Check if gen_text is not empty
    if not gen_text.strip():
        raise ValueError("Generated text is empty. Please provide a valid EPUB file with text content.")

    try:
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            ref_text,
            gen_text,
            ema_model,
            vocoder,
            cross_fade_duration=cross_fade_duration,
            speed=speed,
            show_info=show_info,
            progress=gr.Progress(),
        )
    except IndexError as e:
        raise RuntimeError(f"Error during inference process: {e}")
    except Exception as e:
        raise RuntimeError(f"Error during inference process: {e}")

    # Remove silence
    if remove_silence:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, final_wave, final_sample_rate)
                remove_silence_for_generated_wav(f.name)
                final_wave, _ = torchaudio.load(f.name)
            final_wave = final_wave.squeeze().cpu().numpy()
        except Exception as e:
            raise RuntimeError(f"Error removing silence: {e}")

    # Removed spectrogram saving code
    # Spectrogram generation is no longer needed

    return (final_sample_rate, final_wave), ref_text


def extract_text_and_title_from_epub(epub_path):
    """
    Extracts and concatenates text from all chapters of an EPUB file and retrieves the title.
    """
    try:
        book = epub.read_epub(epub_path)
        print(f"EPUB '{epub_path}' successfully read.")
    except Exception as e:
        raise RuntimeError(f"Failed to read EPUB file: {e}")

    text_content = []
    title = None

    # Extract title from metadata
    try:
        metadata = book.get_metadata('DC', 'title')
        if metadata:
            title = metadata[0][0]
            print(f"Extracted title from metadata: {title}")
        else:
            # Fallback: use file name without extension
            title = os.path.splitext(os.path.basename(epub_path))[0]
            print(f"No title found in metadata. Using file name as title: {title}")
    except Exception as e:
        print(f"Error extracting title from metadata: {e}")
        title = os.path.splitext(os.path.basename(epub_path))[0]
        print(f"Using file name as title: {title}")

    document_count = 0
    text_item_count = 0

    # Enhanced debugging: list all items and their types
    print("Listing all items in the EPUB:")
    for item in book.get_items():
        item_type = item.get_type()
        # Replace get_media_type() with media_type attribute
        media_type = item.media_type
        print(f"Item ID: {item.get_id()}, Type: {item_type}, Media Type: {media_type}")
        if item_type == ITEM_DOCUMENT:  # Updated to use ITEM_DOCUMENT directly
            document_count += 1
            try:
                # Use BeautifulSoup to parse HTML content
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                # Extract text and append
                text = soup.get_text(separator=' ', strip=True)
                if text:  # Ensure text is not empty
                    text_content.append(text)
                    text_item_count += 1
                else:
                    print(f"Warning: No text found in document item {item.get_id()}.")
            except Exception as e:
                print(f"Error parsing document item {item.get_id()}: {e}")

    print(f"Total document items found: {document_count}")
    print(f"Total text items extracted: {text_item_count}")

    # Join all extracted texts into a single string
    full_text = ' '.join(text_content)

    if not full_text:
        raise ValueError("No text found in EPUB file.")

    print(f"Total characters extracted from EPUB: {len(full_text)}")

    return full_text, title


def convert_to_epub(input_path, output_path):
    """
    Converts various ebook formats to EPUB using Calibre's ebook-convert.
    Ensures the output directory exists before conversion.
    """
    try:
        output_dir = os.path.dirname(output_path)
        ensure_directory(output_dir)
        subprocess.run(['ebook-convert', input_path, output_path], check=True)
        print(f"Converted {input_path} to EPUB successfully.")
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"An error occurred while converting the eBook: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during conversion: {e}")


def detect_file_type(file_path):
    """
    Detects the MIME type of a file.
    """
    try:
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        return file_type
    except Exception as e:
        raise RuntimeError(f"Error detecting file type: {e}")


def ensure_directory(directory_path):
    """
    Ensures that a directory exists; if not, creates it.
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Ensured directory exists: {directory_path}")
    except Exception as e:
        raise RuntimeError(f"Error creating directory {directory_path}: {e}")


def create_chapter_labeled_book(ebook_file_path):
    # Function to ensure the existence of a directory
    ensure_directory(os.path.join(".", 'Working_files', 'Book'))

    def save_chapters_as_text(epub_path):
        # Create the directory if it doesn't exist
        directory = os.path.join(".", "Working_files", "temp_ebook")
        ensure_directory(directory)

        previous_chapter_text = ''
        previous_filename = ''
        chapter_counter = 0

        # Iterate through the items in the EPUB file
        for item in epub.read_epub(epub_path).get_items():
            if item.get_type() == ITEM_DOCUMENT:
                # Use BeautifulSoup to parse HTML content
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text()

                # Check if the text is not empty
                if text.strip():
                    if len(text) < 2300 and previous_filename:
                        # Append text to the previous chapter if it's short
                        with open(previous_filename, 'a', encoding='utf-8') as file:
                            file.write('\n' + text)
                    else:
                        # Create a new chapter file and increment the counter
                        previous_filename = os.path.join(directory, f"chapter_{chapter_counter}.txt")
                        chapter_counter += 1
                        with open(previous_filename, 'w', encoding='utf-8') as file:
                            file.write(text)
                            print(f"Saved chapter: {previous_filename}")

    # Example usage
    input_ebook = ebook_file_path  # Replace with your eBook file path
    output_epub = os.path.join(".", "Working_files", "temp.epub")

    if os.path.exists(output_epub):
        os.remove(output_epub)
        print(f"File {output_epub} has been removed.")
    else:
        print(f"The file {output_epub} does not exist.")

    if convert_to_epub(input_ebook, output_epub):
        save_chapters_as_text(output_epub)

    # Download the necessary NLTK data (if not already present)
    # nltk.download('punkt')

    def process_chapter_files(folder_path, output_csv):
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header row
            writer.writerow(['Text', 'Start Location', 'End Location', 'Is Quote', 'Speaker', 'Chapter'])

            # Process each chapter file
            chapter_files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
            for filename in chapter_files:
                if filename.startswith('chapter_') and filename.endswith('.txt'):
                    chapter_number = int(filename.split('_')[1].split('.')[0])
                    file_path = os.path.join(folder_path, filename)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            text = file.read()
                            # Insert "NEWCHAPTERABC" at the beginning of each chapter's text
                            if text:
                                text = "NEWCHAPTERABC" + text
                            sentences = nltk.tokenize.sent_tokenize(text)
                            for sentence in sentences:
                                start_location = text.find(sentence)
                                end_location = start_location + len(sentence)
                                writer.writerow([sentence, start_location, end_location, 'True', 'Narrator', chapter_number])
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")

    # Example usage
    folder_path = os.path.join(".", "Working_files", "temp_ebook")
    output_csv = os.path.join(".", "Working_files", "Book", "Other_book.csv")

    process_chapter_files(folder_path, output_csv)

    def sort_key(filename):
        """Extract chapter number for sorting."""
        match = re.search(r'chapter_(\d+)\.txt', filename)
        return int(match.group(1)) if match else 0

    def combine_chapters(input_folder, output_file):
        # Create the output folder if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # List all txt files and sort them by chapter number
        files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        sorted_files = sorted(files, key=sort_key)

        with open(output_file, 'w', encoding='utf-8') as outfile:  # Specify UTF-8 encoding here
            for i, filename in enumerate(sorted_files):
                with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:  # And here
                    outfile.write(infile.read())
                    # Add the marker unless it's the last file
                    if i < len(sorted_files) - 1:
                        outfile.write("\nNEWCHAPTERABC\n")

    # Paths
    input_folder = os.path.join(".", 'Working_files', 'temp_ebook')
    output_file = os.path.join(".", 'Working_files', 'Book', 'Chapter_Book.txt')

    # Combine the chapters
    combine_chapters(input_folder, output_file)

    ensure_directory(os.path.join(".", "Working_files", "Book"))


def combine_wav_files(input_directory, output_directory, file_name):
    # Ensure that the output directory exists, create it if necessary
    os.makedirs(output_directory, exist_ok=True)

    # Specify the output file path
    output_file_path = os.path.join(output_directory, file_name)

    # Initialize an empty audio segment
    combined_audio = AudioSegment.empty()

    # Get a list of all .wav files in the specified input directory and sort them
    input_file_paths = sorted(
        [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(".wav")],
        key=lambda f: int(''.join(filter(str.isdigit, f)))
    )

    # Sequentially append each file to the combined_audio
    for input_file_path in input_file_paths:
        audio_segment = AudioSegment.from_wav(input_file_path)
        combined_audio += audio_segment

    # Export the combined audio to the output file path
    combined_audio.export(output_file_path, format='wav')

    print(f"Combined audio saved to {output_file_path}")


# Function to split long strings into parts
def split_long_sentence(sentence, max_length=249, max_pauses=10):
    """
    Splits a sentence into parts based on length or number of pauses without recursion.

    :param sentence: The sentence to split.
    :param max_length: Maximum allowed length of a sentence.
    :param max_pauses: Maximum allowed number of pauses in a sentence.
    :return: A list of sentence parts that meet the criteria.
    """
    parts = []
    while len(sentence) > max_length or sentence.count(',') + sentence.count(';') + sentence.count('.') > max_pauses:
        possible_splits = [i for i, char in enumerate(sentence) if char in ',;.' and i < max_length]
        if possible_splits:
            # Find the best place to split the sentence, preferring the last possible split to keep parts longer
            split_at = possible_splits[-1] + 1
        else:
            # If no punctuation to split on within max_length, split at max_length
            split_at = max_length

        # Split the sentence and add the first part to the list
        parts.append(sentence[:split_at].strip())
        sentence = sentence[split_at:].strip()

    # Add the remaining part of the sentence
    parts.append(sentence)
    return parts


# Utility function to sanitize filenames
def sanitize_filename(filename):
    """
    Sanitize the filename by removing or replacing invalid characters.

    :param filename: Original filename.
    :return: Sanitized filename.
    """
    # Remove any characters that are not valid in filenames
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    # Optionally, replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    return sanitized


# Function to list converted audiobooks
def show_converted_audiobooks():
    """
    Lists all converted audiobook MP3 files in the Working_files/Book directory.

    :return: List of file paths to be displayed as downloadable files.
    """
    output_dir = os.path.join("Working_files", "Book")
    if not os.path.exists(output_dir):
        return ["No audiobooks found."]

    # List all MP3 files
    files = [f for f in os.listdir(output_dir) if f.endswith('.mp3')]

    if not files:
        return ["No audiobooks found."]

    # Create full file paths
    file_paths = [os.path.join(output_dir, f) for f in files]

    return file_paths


# Apply the Ocean theme by passing theme=gr.themes.Ocean() to gr.Blocks
with gr.Blocks(theme=gr.themes.Ocean()) as app:
    gr.Markdown(
        """
# eBook to Audiobook with F5-TTS!
"""
    )

    gr.Markdown("")

    ref_audio_input = gr.Audio(label="Upload Voice File (<15 sec) or Record with Mic Icon", type="filepath")

    gen_file_input = gr.File(
        label="eBook File",
        file_types=[".epub", ".mobi", ".pdf", ".txt", ".html"],
        type="filepath",  # Ensures the file path is passed
        interactive=True
    )

    generate_btn = gr.Button("Start", variant="primary")

    with gr.Accordion("Advanced Settings", open=False):
        ref_text_input = gr.Textbox(
            label="Reference Text",
            lines=2,
        )
        gr.Markdown("Leave blank to automatically transcribe the reference audio. If you enter text it will override automatic transcription.")

        remove_silence = gr.Checkbox(
            label="Remove Silences",
            value=False,
        )
        gr.Markdown("The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.")

        speed_slider = gr.Slider(
            label="Speed",
            minimum=0.3,
            maximum=2.0,
            value=1.0,
            step=0.1,
        )
        gr.Markdown("Adjust the speed of the audio.")

        cross_fade_duration_slider = gr.Slider(
            label="Cross-Fade Duration (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
        )
        gr.Markdown("Set the duration of the cross-fade between audio clips.")

    audio_output = gr.Audio(label="Generated Audiobook (Download button to the right of this message when finished ->)")
    # Removed spectrogram_output and error_output as per request

    # ==================== New Components for Showing Audiobooks ====================
    # Button to show converted audiobooks
    show_audiobooks_btn = gr.Button("Show Converted Audiobooks", variant="secondary")

    # Output component to display the list of audiobooks
    audiobooks_output = gr.Files(label="Converted Audiobooks")

    # ==================== End of New Components ====================

    @gpu_decorator
    def basic_tts(
        ref_audio_input,
        ref_text_input,
        gen_file_input,
        remove_silence,
        cross_fade_duration_slider,
        speed_slider,
    ):
        try:
            print("Starting basic_tts function...")
            # Initialize generated text and title
            gen_text = ""
            ebook_title = "audiobook"

            epub_path = None

            # Check if a file is uploaded
            if gen_file_input is not None:
                # Depending on Gradio version, gen_file_input can be a dict or a file path
                if isinstance(gen_file_input, dict):
                    epub_path = gen_file_input['name']
                else:
                    epub_path = gen_file_input
                print(f"Uploaded file path: {epub_path}")
                if not os.path.exists(epub_path):
                    raise FileNotFoundError(f"Uploaded file does not exist at path: {epub_path}")

                # Detect file type
                file_type = detect_file_type(epub_path)
                print(f"Detected file type: {file_type}")

                # Determine if conversion is needed
                if file_type != 'application/epub+zip':
                    # Convert to EPUB
                    temp_epub = os.path.join("Working_files", "temp_converted.epub")
                    convert_to_epub(epub_path, temp_epub)
                    epub_path = temp_epub
                    print(f"File converted to EPUB: {epub_path}")
                else:
                    print("File is already in EPUB format. No conversion needed.")

                # Extract text and title from EPUB
                gen_text, ebook_title = extract_text_and_title_from_epub(epub_path)
                print("Text extracted from EPUB successfully.")
            else:
                raise ValueError("No eBook file uploaded.")

            # Use reference text if provided
            if ref_text_input:
                ref_text = ref_text_input
                print("Using provided reference text.")
            else:
                ref_text = ""  # The infer function will handle automatic transcription
                print("No reference text provided. Will use automatic transcription if supported.")

            # Perform inference
            audio_out, ref_text_out = infer(
                ref_audio_input,
                ref_text,
                gen_text,
                remove_silence,
                cross_fade_duration_slider,
                speed_slider,
            )
            print("Inference completed successfully.")

            # Extract sample rate and waveform
            sample_rate, wave = audio_out

            # Save the waveform to a temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                sf.write(tmp_wav.name, wave, sample_rate)
                tmp_wav_path = tmp_wav.name
                print(f"Temporary WAV file created at: {tmp_wav_path}")

            # Convert WAV to MP3 with variable bitrate of 256kbps
            # Sanitize the ebook title to create a valid filename
            sanitized_title = sanitize_filename(ebook_title)
            if not sanitized_title:
                sanitized_title = "audiobook"

            tmp_mp3_path = os.path.join("Working_files", "Book", f"{sanitized_title}.mp3")
            ensure_directory(os.path.dirname(tmp_mp3_path))

            audio = AudioSegment.from_wav(tmp_wav_path)
            audio.export(tmp_mp3_path, format="mp3", bitrate="256k")
            print(f"Converted MP3 file created at: {tmp_mp3_path}")

            # Remove the temporary WAV file
            os.remove(tmp_wav_path)
            print(f"Temporary WAV file {tmp_wav_path} removed.")

            return tmp_mp3_path

        except Exception as e:
            # Removed error_output handling; now raising exceptions to be handled by Gradio
            print(f"An unexpected error occurred: {str(e)}")
            raise e

    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_file_input,  # Changed input
            remove_silence,
            cross_fade_duration_slider,
            speed_slider,
        ],
        outputs=[audio_output],  # Updated to only include audio_output
    )

    # ==================== New Button and Output Connection ====================
    show_audiobooks_btn.click(
        show_converted_audiobooks,
        inputs=[],  # No inputs needed
        outputs=[audiobooks_output],
    )
    # ==================== End of New Button and Output Connection ====================

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
    global app
    print("Starting app...")
    app.queue().launch(server_name="0.0.0.0", server_port=7860, share=share, show_api=api, debug=True)


if __name__ == "__main__":
    import sys
    print("Arguments passed to Python:", sys.argv)
    if not USING_SPACES:
        main()
    else:
        app.queue().launch(debug=True)
=======
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
    return load_model(DiT, model_cfg, ckpt_path)

F5TTS_ema_model = load_f5tts()

chat_model_state = None
chat_tokenizer_state = None

@gpu_decorator
def generate_response(messages, model, tokenizer):
    """Generate a response using the provided model and tokenizer."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        input_features=model_inputs.input_features,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    if not generated_ids:
        raise ValueError("No generated IDs returned by the model.")

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    if not generated_ids or not generated_ids[0]:
        raise ValueError("Generated IDs are empty after processing.")

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

@gpu_decorator
def infer(ref_audio_orig, ref_text, gen_text, cross_fade_duration=0.15, speed=1, show_info=gr.Info):
    """Perform inference to generate audio from text."""
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)
    except Exception as e:
        raise RuntimeError(f"Error in preprocessing reference audio and text: {e}")

    if not gen_text.strip():
        raise ValueError("Generated text is empty. Please provide valid text content.")

    try:
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio,
            ref_text,
            gen_text,
            F5TTS_ema_model,
            vocoder,
            cross_fade_duration=cross_fade_duration,
            speed=speed,
            show_info=show_info,
            progress=gr.Progress(),
        )
    except Exception as e:
        raise RuntimeError(f"Error during inference process: {e}")

    return (final_sample_rate, final_wave), ref_text

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
                type=1,
                desc='Icon',
                data=img.read()
            ))
        audio.save(mp3_path)
        print(f"Embedded cover image into {mp3_path}")
    except Exception as e:
        print(f"Failed to embed cover image into MP3: {e}")

def create_m4b(combined_wav, metadata_file, cover_image, output_m4b):
    """Create an M4B audiobook from WAV, metadata, and cover image."""
    os.makedirs(os.path.dirname(output_m4b), exist_ok=True)
    
    ffmpeg_cmd = ['ffmpeg', '-i', combined_wav, '-i', metadata_file]
    if cover_image:
        ffmpeg_cmd += ['-i', cover_image, '-map', '0:a', '-map', '2:v']
    else:
        ffmpeg_cmd += ['-map', '0:a']
    
    ffmpeg_cmd += ['-map_metadata', '1', '-c:a', 'aac', '-b:a', '192k']
    if cover_image:
        ffmpeg_cmd += ['-c:v', 'png', '-disposition:v', 'attached_pic']
    ffmpeg_cmd += [output_m4b]
    
    subprocess.run(ffmpeg_cmd, check=True)

def extract_text_and_title_from_epub(epub_path):
    """Extract text and title from an EPUB file."""
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

    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            try:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                if text:
                    text_content.append(text)
                else:
                    print(f"No text in document item {item.get_id()}.")
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

    files = [f for f in os.listdir(output_dir) if f.endswith(('.mp3', '.m4b'))]
    if not files:
        return ["No audiobooks found."]

    return [os.path.join(output_dir, f) for f in files]

def combine_wav_files(input_directory, output_directory, file_name):
    """Combine multiple WAV files into a single WAV file."""
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, file_name)
    combined_audio = AudioSegment.empty()

    input_files = sorted(
        [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(".wav")],
        key=lambda f: int(''.join(filter(str.isdigit, f)))
    )

    for file_path in input_files:
        combined_audio += AudioSegment.from_wav(file_path)

    combined_audio.export(output_file_path, format='wav')
    print(f"Combined audio saved to {output_file_path}")

def split_long_sentence(sentence, max_length=249, max_pauses=30):
    """Split long sentences into manageable parts."""
    parts = []
    while len(sentence) > max_length or sentence.count(',') + sentence.count(';') + sentence.count('.') > max_pauses:
        possible_splits = [i for i, char in enumerate(sentence) if char in ',;.' and i < max_length]
        split_at = possible_splits[-1] + 1 if possible_splits else max_length
        parts.append(sentence[:split_at].strip())
        sentence = sentence[split_at:].strip()
    parts.append(sentence)
    return parts

def create_chapter_labeled_book(ebook_file_path):
    """Create a chapter-labeled book from an ebook file."""
    ensure_directory(os.path.join(".", 'Working_files', 'Book'))

    def save_chapters_as_text(epub_path):
        directory = os.path.join(".", "Working_files", "temp_ebook")
        ensure_directory(directory)

        previous_filename = ''
        chapter_counter = 0

        for item in epub.read_epub(epub_path).get_items():
            if item.get_type() == ITEM_DOCUMENT:
                try:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text()
                    if text.strip():
                        if len(text) < 2300 and previous_filename:
                            with open(previous_filename, 'a', encoding='utf-8') as file:
                                file.write('\n' + text)
                        else:
                            previous_filename = os.path.join(directory, f"chapter_{chapter_counter}.txt")
                            chapter_counter += 1
                            with open(previous_filename, 'w', encoding='utf-8') as file:
                                file.write(text)
                                print(f"Saved chapter: {previous_filename}")
                except Exception as e:
                    print(f"Error processing chapter: {e}")

    input_ebook = ebook_file_path
    output_epub = os.path.join(".", "Working_files", "temp.epub")

    if os.path.exists(output_epub):
        os.remove(output_epub)
        print(f"Removed existing file: {output_epub}")

    if convert_to_epub(input_ebook, output_epub):
        save_chapters_as_text(output_epub)

    def process_chapter_files(folder_path, output_csv):
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Text', 'Start Location', 'End Location', 'Is Quote', 'Speaker', 'Chapter'])

            chapter_files = sorted(
                [f for f in os.listdir(folder_path) if f.startswith('chapter_') and f.endswith('.txt')],
                key=lambda x: int(re.search(r'chapter_(\d+)\.txt', x).group(1))
            )

            for filename in chapter_files:
                chapter_number = int(re.search(r'chapter_(\d+)\.txt', filename).group(1))
                file_path = os.path.join(folder_path, filename)

                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        if text:
                            text = "NEWCHAPTERABC" + text
                        sentences = nltk.tokenize.sent_tokenize(text)
                        for sentence in sentences:
                            start = text.find(sentence)
                            end = start + len(sentence)
                            writer.writerow([sentence, start, end, 'True', 'Narrator', chapter_number])
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    folder_path = os.path.join(".", "Working_files", "temp_ebook")
    output_csv = os.path.join(".", "Working_files", "Book", "Other_book.csv")
    process_chapter_files(folder_path, output_csv)

    def sort_key(filename):
        match = re.search(r'chapter_(\d+)\.txt', filename)
        return int(match.group(1)) if match else 0

    def combine_chapters(input_folder, output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        files = sorted([f for f in os.listdir(input_folder) if f.endswith(".txt")], key=sort_key)

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for i, filename in enumerate(files):
                with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    if i < len(files) - 1:
                        outfile.write("\nNEWCHAPTERABC\n")

    output_file = os.path.join(".", 'Working_files', 'Book', 'Chapter_Book.txt')
    combine_chapters(folder_path, output_file)
    ensure_directory(os.path.join(".", "Working_files", "Book"))

@gpu_decorator
def basic_tts(ref_audio_input, ref_text_input, gen_file_input, cross_fade_duration, speed):
    """Main function to convert eBooks to audiobooks."""
    try:
        generated_files = []

        for ebook in gen_file_input:
            epub_path = ebook
            if not os.path.exists(epub_path):
                raise FileNotFoundError(f"File not found: {epub_path}")

            file_type = detect_file_type(epub_path)
            if file_type != 'application/epub+zip':
                sanitized_base = sanitize_filename(os.path.splitext(os.path.basename(epub_path))[0])
                temp_epub = os.path.join("Working_files", "temp_converted", f"{sanitized_base}.epub")
                convert_to_epub(epub_path, temp_epub)
                epub_path = temp_epub

            gen_text, ebook_title = extract_text_and_title_from_epub(epub_path)
            cover_image = extract_metadata_and_cover(epub_path)

            ref_text = ref_text_input or ""

            audio_out, _ = infer(
                ref_audio_input,
                ref_text,
                gen_text,
                cross_fade_duration,
                speed,
            )

            sample_rate, wave = audio_out
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                sf.write(tmp_wav.name, wave, sample_rate)
                tmp_wav_path = tmp_wav.name

            sanitized_title = sanitize_filename(ebook_title) or f"audiobook_{int(tempfile._get_default_tempdir())}"
            tmp_mp3_path = os.path.join("Working_files", "Book", f"{sanitized_title}.mp3")
            ensure_directory(os.path.dirname(tmp_mp3_path))

            audio = AudioSegment.from_wav(tmp_wav_path)
            audio.export(tmp_mp3_path, format="mp3", bitrate="256k")

            if cover_image:
                embed_cover_into_mp3(tmp_mp3_path, cover_image)

            os.remove(tmp_wav_path)
            if cover_image and os.path.exists(cover_image):
                os.remove(cover_image)

            generated_files.append(tmp_mp3_path)

        audiobooks = show_converted_audiobooks()
        last_file = generated_files[-1] if generated_files else None

        return generated_files, last_file, audiobooks

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

        generate_btn = gr.Button("Start", variant="primary")
        audio_output = gr.Files(label="Inference Progress (After 100%, Audio Chunks are Stiched)")

        show_audiobooks_btn = gr.Button("Show All Completed Audiobooks", variant="secondary")
        audiobooks_output = gr.Files(label="Converted Audiobooks (Download Links ->)")

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
                value=0.15,
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
            outputs=[audio_output, player, audiobooks_output],
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
>>>>>>> ca85b47 (Update app to the latest version with new features and fixes)
