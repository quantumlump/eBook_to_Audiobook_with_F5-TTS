# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys

import tqdm as std_tqdm # To distinguish from any 'progress' variable that might be tqdm module
import datetime
import time

os.environ["PYTOCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"../../{os.path.dirname(os.path.abspath(__file__))}/third_party/BigVGAN/")

import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib

matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import snapshot_download, hf_hub_download
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from f5_tts.model import CFM
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
)

_ref_audio_cache = {}

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------


# chunk text into smaller pieces


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# load vocoder
def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device, hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            local_path = snapshot_download(repo_id="nvidia/bigvgan_v2_24khz_100band_256x", cache_dir=hf_cache_dir)
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16 if "cuda" in device and torch.cuda.get_device_properties(device).major >= 6 else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# transcribe


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16 if "cuda" in device and torch.cuda.get_device_properties(device).major >= 6 else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


def preprocess_ref_audio_text(ref_audio_orig, ref_text, clip_short=True, show_info=print, device=device):
    show_info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                    show_info("Audio is over 15s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 15000:
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                        show_info("Audio is over 15s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            # 3. if no proper silence found for clipping
            if len(aseg) > 15000:
                aseg = aseg[:15000]
                show_info("Audio is over 15s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    # Compute a hash of the reference audio file
    with open(ref_audio, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    if not ref_text.strip():
        global _ref_audio_cache
        if audio_hash in _ref_audio_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_audio_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_audio_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("ref_text  ", ref_text)

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]


# utils_infer.py

def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=None, 
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
    progress_start_fraction=0.0, # Overall start fraction for this infer call
    progress_end_fraction=1.0,   # Overall end fraction for this infer call
    ebook_idx=0,                 # Current ebook index
    num_ebooks=1                 # Total number of ebooks
):
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    for i, gen_text_item in enumerate(gen_text_batches): # Corrected variable name from gen_text to gen_text_item
        print(f"gen_text {i}", gen_text_item) # Corrected variable name

    if callable(show_info):
        show_info(f"Synthesizing audio in {len(gen_text_batches)} chunks for Ebook {ebook_idx+1}/{num_ebooks}...")
    else:
        print(f"Synthesizing audio in {len(gen_text_batches)} chunks for Ebook {ebook_idx+1}/{num_ebooks}...")

    return infer_batch_process(
        (audio, sr),
        ref_text,
        gen_text_batches,
        model_obj,
        vocoder,
        mel_spec_type=mel_spec_type,
        progress=progress,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        fix_duration=fix_duration,
        device=device,
        progress_start_fraction=progress_start_fraction,
        progress_end_fraction=progress_end_fraction,
        ebook_idx=ebook_idx,
        num_ebooks=num_ebooks
    )


# infer batches


# utils_infer.py

def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=None,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    progress_start_fraction=0.0,
    progress_end_fraction=1.0,
    ebook_idx=0,
    num_ebooks=1
):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []
    num_batches = len(gen_text_batches)

    tqdm_kwargs = {"disable": True}
    if progress is None:
        tqdm_kwargs = {"desc": f"Ebook {ebook_idx+1} Chunks", "unit": "chunk", "total": num_batches, "disable": False}
    
    pbar_iterator = std_tqdm.tqdm(gen_text_batches, **tqdm_kwargs)
    
    # Explicitly track start time for robust ETR
    loop_start_time = time.monotonic()


    if len(ref_text) > 0 and len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    for i, gen_text_chunk in enumerate(pbar_iterator): # Changed gen_text to gen_text_chunk
        text_list = [ref_text + gen_text_chunk] # Use gen_text_chunk
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_chunk_len = len(gen_text_chunk.encode("utf-8")) # Use gen_text_chunk
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_chunk_len / speed) # Use gen_text_chunk_len

        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated_mel_spec)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated_mel_spec)
            if rms < target_rms: # Check if this condition is still relevant or should use original audio's rms
                generated_wave = generated_wave * rms / target_rms
            generated_wave = generated_wave.squeeze().cpu().numpy()
            generated_waves.append(generated_wave)
            spectrograms.append(generated_mel_spec[0].cpu().numpy())

        if progress is not None and num_batches > 0:
            current_chunk_display_num = i + 1 # For "Chunk 1/N" display
            
            # Use time.monotonic() for a reliable elapsed time since loop start
            current_time = time.monotonic()
            elapsed_loop_time_seconds = current_time - loop_start_time
            
            etr_formatted = "ETR: Calculating..."
            # Calculate ETR based on *completed* iterations (i)
            # Condition: at least one chunk processed (i >= 1) and some time has passed
            if i >= 1 and elapsed_loop_time_seconds > 0.1: # 'i' is 0-indexed, so i=0 is the 1st chunk.
                                                          # We need i >= 1 for the *second* chunk onwards to have a base.
                time_per_chunk = elapsed_loop_time_seconds / float(i) # 'i' is number of *completed* previous chunks
                remaining_chunks = num_batches - current_chunk_display_num # Chunks yet to be processed
                
                if remaining_chunks > 0:
                    etr_seconds = time_per_chunk * remaining_chunks
                    etr_formatted = f"ETR: {str(datetime.timedelta(seconds=int(etr_seconds)))}"
                elif remaining_chunks == 0 : # Last chunk currently processing
                    etr_formatted = "ETR: Finishing..."
                else: # Should not happen if logic is correct
                    etr_formatted = "ETR: Done"

            elif i == 0 and num_batches > 1: # First chunk, but more to come
                etr_formatted = "ETR: Estimating..."
            elif num_batches <= 1 : # Only one chunk or no chunks
                 etr_formatted = ""


            progress_within_this_infer_call = current_chunk_display_num / float(num_batches) if num_batches > 0 else 1.0
            current_overall_fraction = progress_start_fraction + \
                                       (progress_within_this_infer_call * (progress_end_fraction - progress_start_fraction))
            
            desc_parts = []
            if num_ebooks > 1:
                desc_parts.append(f"Ebook {ebook_idx + 1}/{num_ebooks}")
            
            desc_parts.append(f"Chunk {current_chunk_display_num}/{num_batches}")

            if elapsed_loop_time_seconds > 0:
                elapsed_str = str(datetime.timedelta(seconds=int(elapsed_loop_time_seconds))).split('.')[0]
                desc_parts.append(f"Elapsed: {elapsed_str}")
            
            if etr_formatted:
                desc_parts.append(etr_formatted)
            
            # Add the percentage to the description
            # The 'current_overall_fraction' is the value for the progress bar itself (0.0 to 1.0)
            # If you want to show the percentage of *this inference step*, it's 'progress_within_this_infer_call'
            # If you want overall percentage, it's current_overall_fraction
            # Let's show the percentage of the current inference step (chunks)
            percent_this_infer_step = progress_within_this_infer_call * 100
            desc_parts.append(f"{percent_this_infer_step:.1f}%")

            progress_desc = " - ".join(desc_parts)
            progress(current_overall_fraction, desc=progress_desc)

    # After the loop
    if progress is not None:
        final_elapsed_time = time.monotonic() - loop_start_time
        final_desc_parts = []
        if num_ebooks > 1:
            final_desc_parts.append(f"Ebook {ebook_idx + 1}/{num_ebooks}")
        final_desc_parts.append(f"Audio synthesis complete ({num_batches} chunks)")
        if final_elapsed_time > 0:
             elapsed_str = str(datetime.timedelta(seconds=int(final_elapsed_time))).split('.')[0]
             final_desc_parts.append(f"Time: {elapsed_str}")
        final_desc_parts.append("100.0%") # Mark as 100% for this stage
        progress(progress_end_fraction, desc=" - ".join(final_desc_parts))

    # Combine waves (existing logic)
    if not generated_waves:
         return np.array([]), target_sample_rate, np.array([])


    if cross_fade_duration <= 0:
        final_wave = np.concatenate(generated_waves)
    else:
        # (Your existing cross-fade logic) ...
        final_wave = generated_waves[0]
        for i_wave in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i_wave]
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))
            if cross_fade_samples <= 0:
                final_wave = np.concatenate([prev_wave, next_wave])
                continue
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
            final_wave = np.concatenate(
                [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
            )


    combined_spectrogram = np.concatenate(spectrograms, axis=1) if spectrograms else np.array([])
    return final_wave, target_sample_rate, combined_spectrogram

# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
