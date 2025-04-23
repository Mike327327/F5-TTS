from pydub import AudioSegment, silence
import argparse
import os
import sys
import re
import librosa
from tqdm import tqdm
from importlib.resources import files
from pathlib import Path
import numpy as np
import soundfile as sf
from cached_path import cached_path
from omegaconf import OmegaConf
from scipy.signal import correlate
import whisper_timestamped as whisper
import json

if "--list_experiments" not in sys.argv:
    from f5_tts.infer.utils_infer import (
        mel_spec_type,
        target_rms,
        cross_fade_duration,
        nfe_step,
        cfg_strength,
        sway_sampling_coef,
        speed,
        fix_duration,
        infer_process,
        load_model,
        load_vocoder,
        preprocess_ref_audio_text,
        remove_silence_for_generated_wav,
    )
    from f5_tts.model import DiT

    ######################## CZ ########################
    # CKPT_FILE = "/mnt/matylda4/xluner01/F5-TTS/ckpts/ParCzech_2020_2021/model_7850000.pt"
    # MODEL_CFG = "/mnt/matylda4/xluner01/F5-TTS/src/f5_tts/configs/F5TTS_Small_train.yaml"
    # VOCAB_FILE = "/mnt/matylda4/xluner01/F5-TTS/data/ParCzech_2020_2021_char/vocab.txt"
    ######################## EN (non-causal) ########################
    # CKPT_FILE = "/mnt/matylda4/xluner01/F5-TTS/ckpts/LibriTTS_100_360_500/model_1600000_en.pt"
    # MODEL_CFG = "/mnt/matylda4/xluner01/F5-TTS/src/f5_tts/configs/F5TTS_Small_train_LibriTTS.yaml"
    # VOCAB_FILE = "/mnt/matylda4/xluner01/F5-TTS/data/LibriTTS_100_360_500_char/vocab.txt"
    ######################## EN (causal) ########################
    # CKPT_FILE = "/mnt/matylda4/xluner01/F5-TTS/ckpts/LibriTTS_100_360_500/model_1425000_en_causal.pt"
    # MODEL_CFG = "/mnt/matylda4/xluner01/F5-TTS/src/f5_tts/configs/F5TTS_Small_train_LibriTTS.yaml"
    # VOCAB_FILE = "/mnt/matylda4/xluner01/F5-TTS/data/LibriTTS_100_360_500_char/vocab.txt"
    ######################## EN (pretrained) ########################
    CKPT_FILE = "/homes/eva/xl/xluner01/.cache/huggingface/hub/models--SWivid--F5-TTS/snapshots/84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_Base/model_1200000.safetensors"
    MODEL_CFG = "/mnt/matylda4/xluner01/F5-TTS/src/f5_tts/configs/F5TTS_Base_train.yaml"
    VOCAB_FILE = "/mnt/matylda4/xluner01/F5-TTS/src/f5_tts/infer/examples/vocab.txt"
    
    SPEED = speed # 1.0
    CROSSFADE = cross_fade_duration
    VOCODER_LOCAL_PATH = "/mnt/matylda4/xluner01/F5-TTS/checkpoints/vocos-mel-24khz"
    VOCODER_NAME = mel_spec_type # vocos
    LOAD_VOCODER_FROM_LOCAL = True
    REMOVE_SILENCE = False
    
    FS = 24000
    
    LANG = "en"

ARGS = None

def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio using the F5 model.")
    
    parser.add_argument(
        "--list_experiments",
        action="store_true",  
        help="List of experiments to run with detailed descriptions and exit."
    )

    # Check for `--list_experiments` before enforcing other arguments
    if "--list_experiments" in sys.argv:
        return parser.parse_args()
    
    # reference audio and text will be the same for all generated audio
    parser.add_argument(
        "--audio_ref_file", 
        type=str, 
        required=True, 
        help="Path to the original reference audio."
    )
    parser.add_argument(
        "--text_ref_file", 
        type=str, 
        required=True, 
        help="Path to the original reference text."
    )
    parser.add_argument(
        "--text_gen_folder", 
        type=str, 
        required=True, 
        help="Path to the folder where the text to be generated is stored."
    )
    parser.add_argument(
        "--audio_gen_output_folder", 
        type=str, 
        default="./generated_audio",
        help="Path to the folder where the generated audio files will be stored."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="If set, print additional information during evaluation."
    )
    parser.add_argument(
        "--experiment",
        type=int, 
        help="If not provided, run all experiments. Otherwise, run the specified experiment (see --list_experiment)."
    )

    return parser.parse_args()

def print_verbose(text):
    if ARGS.verbose:
        print(text)
        
def remove_silence_and_get_duration_from_wav(src_file):
    aseg = AudioSegment.from_file(src_file)
    non_silent_segs = silence.split_on_silence(aseg, min_silence_len=200, silence_thresh=-50, seek_step=10)
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export("tmp_without_silence.wav", format="wav")
    y, sr = librosa.load(src_file, sr=None)  # `sr=None` preserves original sampling rate
    duration = librosa.get_duration(y=y, sr=sr)
    os.remove("tmp_without_silence.wav")
    return duration
        
def remove_silence_from_wav(src_file, out_file):
    aseg = AudioSegment.from_file(src_file)
    non_silent_segs = silence.split_on_silence(aseg, min_silence_len=200, silence_thresh=-50, seek_step=10)
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(out_file, format="wav")
    
# Temporarily disable printing
def suppress_print(func, *args, **kwargs):
    # Backup the original stdout
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')  # Redirect stdout to devnull (ignore prints)

    try:
        return func(*args, **kwargs)
    finally:
        sys.stdout = original_stdout  # Restore the original stdout
        
def load_f5_model():
    global vocoder, model_cls, model_cfg, ema_model
    
    vocoder = load_vocoder(vocoder_name=VOCODER_NAME, is_local=True, local_path=VOCODER_LOCAL_PATH)
    
    model_cls = DiT
    model_cfg = OmegaConf.load(MODEL_CFG).model.arch
  
    print("Initializing model: F5-TTS")
    print(f"Initializing vocoder: {VOCODER_NAME}")
    
    ema_model = load_model(model_cls, model_cfg, CKPT_FILE, mel_spec_type=VOCODER_NAME, vocab_file=VOCAB_FILE)

def load_whisper_timestamped_model():
    global whisper_model
    
    print("Initializing model: Whisper-timestamped")
    whisper_model = whisper.load_model("large")
    
def transcribe_audio_with_whisper(audio_path):
    audio = whisper.load_audio(audio_path)
    # result = whisper.transcribe(whisper_model, audio, language=LANG)
    result = suppress_print(whisper.transcribe, whisper_model, audio, language=LANG)

    segments = result["segments"]
    word_array = []

    for segment in segments:
        for word_info in segment.get("words", []):
            clean_word = re.sub(r'[^\w]', '', word_info["text"], flags=re.UNICODE).strip()
            word_array.append([
                clean_word.lower(),
                word_info["start"],
                word_info["end"]
            ])

    return word_array
    
def run_inference(ref_audio, ref_text, gen_text, gen_audio_path, fix_dur=fix_duration):
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    voices = {"main": main_voice}
    voices["main"]["ref_audio"], voices["main"]["ref_text"] = preprocess_ref_audio_text(
        voices["main"]["ref_audio"], voices["main"]["ref_text"]
    )

    ref_audio_ = voices["main"]["ref_audio"]
    ref_text_ = voices["main"]["ref_text"]
    gen_text_ = gen_text.strip()
    
    # fix_duration experiment (if fix_dur is set and not equal to the default fix_duration, use it)
    if fix_dur != fix_duration:
        final_fix_duration = fix_dur
    else:
        final_fix_duration = fix_duration
    
    wave, sample_rate, spectrogram = infer_process(
        ref_audio_,
        ref_text_,
        gen_text_,
        ema_model,
        vocoder,
        mel_spec_type=VOCODER_NAME,
        target_rms=target_rms,
        cross_fade_duration=CROSSFADE,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=SPEED,
        fix_duration=final_fix_duration,
    )

    with open(gen_audio_path, "wb") as f:
        sf.write(f.name, wave, sample_rate)
        if REMOVE_SILENCE:
            remove_silence_for_generated_wav(f.name)

# Based on the chunk size, divide the text into chunks
def get_chunks(text, chunk_size):
    text_words = text.split()
    
    # Remove any special characters from each word
    # \w matches Unicode word characters (letters, digits, and underscores)
    text_words = [re.sub(r'[^\w]', '', word, flags=re.UNICODE) for word in text_words]
    
    # Delete any empty items in the list
    text_words = [word for word in text_words if word != ""]
    
    # Group words into chunks of the given size
    text_chunks = []
    for i in range(0, len(text_words), chunk_size):
        chunk = " ".join(text_words[i:i+chunk_size])
        text_chunks.append(chunk)
    
    return text_chunks

# Based on the min_chars, divide the text into chunks
def get_dynamic_chunks(text, min_chars=10):
    initial_chunks = get_chunks(text, 1)
    
    text_chunks = []
    current_chunk = ""
    
    for word in initial_chunks:
        # If adding the word keeps us under the limit or we’re building the first part, keep adding
        if len(current_chunk) + len(word) + 1 <= min_chars:
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word
        else:
            # Current chunk is big enough, so we save it and start a new one
            if current_chunk:
                text_chunks.append(current_chunk)
            current_chunk = word

    # add the last chunk
    if current_chunk:
        text_chunks.append(current_chunk)
    
    return text_chunks
    
def apply_fade(audio, fade_in_duration=0.05, fade_out_duration=0.05, sr=FS):
    n_samples = len(audio)
    fade_in_samples = int(sr * fade_in_duration)
    fade_out_samples = int(sr * fade_out_duration)

    # Ensure we don't exceed audio length
    fade_in_samples = min(fade_in_samples, n_samples // 2)
    fade_out_samples = min(fade_out_samples, n_samples // 2)

    if fade_in_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_in_samples)
        audio[:fade_in_samples] *= fade_in

    if fade_out_samples > 0:
        fade_out = np.linspace(1.0, 0.0, fade_out_samples)
        audio[-fade_out_samples:] *= fade_out

    return audio

def generate_silence(duration=0.1, sr=FS):
    return np.zeros(int(duration * sr), dtype=np.float32)

# Process audio segments to find voiced intervals
def process_segments(audio_path, trim_ending_value=0, keep_only_last_segment=False):
    # Load and normalize
    audio, sr = librosa.load(audio_path, sr=None)
    # audio = audio / np.max(np.abs(audio))  # Normalize

    # Trim from end
    if trim_ending_value > 0 and keep_only_last_segment is False:
        samples_to_keep = int(len(audio) * (1 - trim_ending_value))
        audio = audio[:samples_to_keep]
        return audio

    # Frame settings
    frame_size = int(0.025 * sr)  # 25 ms
    frame_step = int(0.010 * sr)  # 10 ms

    def frame_signal(signal, frame_size, frame_step):
        num_frames = 1 + int((len(signal) - frame_size) / frame_step)
        indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_size, 1)).T
        return signal[indices]

    # Frame and compute energy
    frames = frame_signal(audio, frame_size, frame_step)
    energy = np.sum(frames ** 2, axis=1)
    threshold = np.percentile(energy, 50)  # Adjustable percentile
    low_energy_mask = energy < threshold

    # Find voiced intervals
    def get_intervals(mask, frame_step, frame_size, sr):
        intervals_samples = []
        in_interval = False
        for i, is_low in enumerate(mask):
            if not is_low and not in_interval:
                start_idx = i
                in_interval = True
            elif is_low and in_interval:
                end_idx = i
                in_interval = False
                start_sample = start_idx * frame_step
                end_sample = end_idx * frame_step + frame_size
                intervals_samples.append((start_sample, min(end_sample, len(audio))))
        if in_interval:
            start_sample = start_idx * frame_step
            intervals_samples.append((start_sample, len(audio)))
        return intervals_samples

    voiced_intervals = get_intervals(low_energy_mask, frame_step, frame_size, sr)

    # Optional: keep only the last voiced segment
    if keep_only_last_segment and voiced_intervals:
        for voiced_interval in reversed(voiced_intervals):
            # Check if the interval is long enough
            if (voiced_interval[1]-voiced_interval[0]) < 1000:
              continue
            else:
              voiced_intervals = [voiced_interval]
              break

    return voiced_intervals

# Compute cross-correlation and concatenate audio
def cross_correlation(first_audio, second_audio):
    # Trim silence
    remove_silence_from_wav(first_audio, first_audio)
    remove_silence_from_wav(second_audio, second_audio)

    # Load both audio files
    first_audio, sr1 = librosa.load(first_audio, sr=FS)
    second_audio, sr2 = librosa.load(second_audio, sr=FS)

    # Check sample rate match
    assert sr1 == sr2, "Sampling rates don't match!"
    sr = sr1  # sample rate

    # Check if either audio is empty
    if len(first_audio) == 0:
        return second_audio
    if len(second_audio) == 0:
        return first_audio

    # Cross-correlation
    corr = correlate(second_audio, first_audio, mode='full')
    lags = np.arange(-len(first_audio) + 1, len(second_audio))
    best_lag = lags[np.argmax(corr)]

    # Trim and concatenate
    if best_lag >= 0:
        second_trimmed = second_audio[best_lag:]
        final_audio = np.concatenate((first_audio, second_trimmed))
        print_verbose(f"Trimmed {best_lag} samples ({best_lag/sr:.2f} sec) from start of second audio.")
    else:
        # If second audio starts before the first, we could trim the first one
        print_verbose("First audio starts later — consider trimming it or re-aligning differently.")
        second_trimmed = second_audio
        final_audio = np.concatenate((first_audio, second_trimmed))

    return final_audio

###########################################################
###################### EXPERIMENTS ########################
###########################################################
############################ 0 ############################
# Generate audio using the model as-is, e.g., full sentence generation, conditioned on the fixed reference audio and text.
def experiment_default_per_sentence():
    experiment_id = f'experiment_default_per_sentence'
    
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = [os.path.join(ARGS.text_gen_folder, file) for file in os.listdir(ARGS.text_gen_folder)]
    gen_texts = {}
    for gen_text in gen_texts_paths:
        with open(gen_text, "r") as f:
            gen_text_file_name = os.path.basename(gen_text)
            gen_texts[gen_text_file_name] = f.read().strip()
    
    os.makedirs(Path(ARGS.audio_gen_output_folder, experiment_id), exist_ok=True)
    
    for gen_text_file_name, gen_text in tqdm(gen_texts.items()):
        output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')

        if ARGS.verbose:
            print("==========================================================================")
            print(f'Text to generate:\t {gen_text}')
            print(f'Reference text:\t\t {ref_text}')
            print(f'Reference audio:\t {ref_audio}')
            print("==========================================================================")

        # Run inference
        # run_inference(ref_audio, ref_text, gen_text, output_wav_path)
        suppress_print(run_inference, ref_audio, ref_text, gen_text, output_wav_path)
        
        # Save final concatenated audio
        print_verbose(f"Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')}")

############################ 1 ############################
# Generate audio per chunk (word, multiple words), conditioned on the fixed reference audio and text.
def experiment_per_chunk(chunk_size=1):
    experiment_id = f'experiment_per_chunk_size_{chunk_size}'
    
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = [os.path.join(ARGS.text_gen_folder, file) for file in os.listdir(ARGS.text_gen_folder)]
    gen_texts = {}
    for gen_text in gen_texts_paths:
        with open(gen_text, "r") as f:
            gen_text_file_name = os.path.basename(gen_text)
            gen_text = f.read().strip()
            gen_text_chunks = get_chunks(gen_text, chunk_size) # divide the text into chunks
            gen_texts[gen_text_file_name] = gen_text_chunks
    
    os.makedirs(Path(ARGS.audio_gen_output_folder, experiment_id), exist_ok=True)
    
    for gen_text_file_name, gen_text in tqdm(gen_texts.items()):
        final_audio = AudioSegment.silent(duration=0)
        for i, chunk in enumerate(gen_text):
            output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav')

            if ARGS.verbose:
                print("==========================================================================")
                print(f'Current chunk to generate:\t {i+1}: {chunk}')
                print(f'Reference text:\t\t {ref_text}')
                print(f'Reference audio:\t {ref_audio}')
                print("==========================================================================")

            # Run inference
            # run_inference(ref_audio, ref_text, chunk, output_wav_path)
            suppress_print(run_inference, ref_audio, ref_text, chunk, output_wav_path)
            
            # Load generated audio
            generated_audio = AudioSegment.from_wav(output_wav_path)

            # Append to final audio
            final_audio += generated_audio
            
            # Delete the generated audio file chunk
            os.remove(output_wav_path)

        # Save final concatenated audio
        final_audio.export(Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav'), format="wav")
        print_verbose(f"Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')}")

############################ 2 ############################
# Generate audio per chunk (word, multiple words), conditioned on the fixed reference & last generated audio and text.  
def experiment_per_chunk_gen_cond(chunk_size=1):
    experiment_id = f'experiment_per_chunk_size_{chunk_size}_gen_cond'
    
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = [os.path.join(ARGS.text_gen_folder, file) for file in os.listdir(ARGS.text_gen_folder)]
    gen_texts = {}
    for gen_text in gen_texts_paths:
        with open(gen_text, "r") as f:
            gen_text_file_name = os.path.basename(gen_text)
            gen_text = f.read().strip()
            gen_text_chunks = get_chunks(gen_text, chunk_size)
            gen_texts[gen_text_file_name] = gen_text_chunks
            
    os.makedirs(Path(ARGS.audio_gen_output_folder, experiment_id), exist_ok=True)
    
    original_ref_text = ref_text
    original_ref_audio = ref_audio
    
    for gen_text_file_name, gen_text in tqdm(gen_texts.items()):
        final_audio = AudioSegment.silent(duration=0)
        ref_audio = original_ref_audio
        for i, chunk in enumerate(gen_text):
            output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav')

            if ARGS.verbose:
                print("==========================================================================")
                print(f'Current chunk to generate:\t {i+1}: {chunk}')
                print(f'Reference text:\t\t {ref_text}')
                print(f'Reference audio:\t {ref_audio}')
                print("==========================================================================")

            # Run inference
            # run_inference(ref_audio, ref_text, chunk, output_wav_path)
            suppress_print(run_inference, ref_audio, ref_text, chunk, output_wav_path)
            
            # Load generated audio
            generated_audio = AudioSegment.from_wav(output_wav_path)

            # Append to final audio
            final_audio += generated_audio
            
            # Delete the generated audio file chunk
            os.remove(output_wav_path)

            # Update reference text (append new chunk)
            ref_text = original_ref_text + " " + chunk + " ."  # Keep original text + add new text
            # Concatenate the new audio with the original reference audio
            conc = AudioSegment.from_wav(original_ref_audio) + generated_audio
            conc.export(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"), format="wav")
            ref_audio = Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav") # Keep original audio + add new audio

        # Save final concatenated audio
        final_audio.export(Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav'), format="wav")
        # Delete the temporary concatenated audio file
        if os.path.exists(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav")):
            os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"))	
        print_verbose(f"Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')}")

############################ 3 ############################
# Generate audio per chunk (word, multiple words), conditioned on the fixed reference & last generated audio (with silence) and text.
def experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len_ms=100):
    experiment_id = f'experiment_per_chunk_size_{chunk_size}_gen_cond_with_silence_{silence_len_ms}_ms'
        
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = [os.path.join(ARGS.text_gen_folder, file) for file in os.listdir(ARGS.text_gen_folder)]
    gen_texts = {}
    for gen_text in gen_texts_paths:
        with open(gen_text, "r") as f:
            gen_text_file_name = os.path.basename(gen_text)
            gen_text = f.read().strip()
            gen_text_chunks = get_chunks(gen_text, chunk_size) # divide the text into chunks on space
            gen_texts[gen_text_file_name] = gen_text_chunks
            
    os.makedirs(ARGS.audio_gen_output_folder + f"/{experiment_id}", exist_ok=True)
    
    original_ref_text = ref_text
    original_ref_audio = ref_audio
    
    for gen_text_file_name, gen_text in tqdm(gen_texts.items()):
        final_audio = AudioSegment.silent(duration=0)
        ref_audio = original_ref_audio
        for i, chunk in enumerate(gen_text):
            output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav')

            if ARGS.verbose:
                print("==========================================================================")
                print(f'Current chunk to generate:\t {i+1}: {chunk}')
                print(f'Reference text:\t\t {ref_text}')
                print(f'Reference audio:\t {ref_audio}')
                print("==========================================================================")

            # Run inference
            # run_inference(ref_audio, ref_text, chunk, output_wav_path)
            suppress_print(run_inference, ref_audio, ref_text, chunk, output_wav_path)
            
            # Load generated audio
            generated_audio = AudioSegment.from_wav(output_wav_path)

            # Append to final audio
            final_audio += generated_audio
            
            # Delete the generated audio file chunk
            os.remove(output_wav_path)

            # Update reference text (append new chunk)
            ref_text = original_ref_text + " " + chunk + " ."  # Keep original text + add new text
            # Concatenate the new audio with the original reference audio
            conc = AudioSegment.from_wav(original_ref_audio) + generated_audio
            conc.export(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"), format="wav")
            ref_audio = Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav") # Keep original audio + add new audio

        # Save final concatenated audio
        final_audio.export(Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav'), format="wav")
        # Delete the temporary concatenated audio file
        os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"))	
        print_verbose(f"Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')}")

############################ 4 ############################
# Generate audio per increasing chunk (word, multiple words), conditioned on the fixed reference, obtain only the new word using DTW and concatenate with the previous audio.
#   use_whisper: Use Whisper to get the timestamps of the last word instead of DTW.
#   fade_in_out: Apply fade in and out to the generated audio, also applies silence in between the chunks.
# First 2 words are generated with fix_duration as it is just 1 (or 2) word(s) and model did not output anything; the rest is generated without fix_duration.
def experiment_dtw(use_whisper=False, fade_in_out=True):
    if use_whisper:
        experiment_id = f'experiment_dtw_whisper'
    else:
        experiment_id = f'experiment_dtw'
        
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = [os.path.join(ARGS.text_gen_folder, file) for file in os.listdir(ARGS.text_gen_folder)]
    gen_texts = {}
    for gen_text in gen_texts_paths:
        with open(gen_text, "r") as f:
            gen_text_file_name = os.path.basename(gen_text)
            gen_text = f.read().strip()
            gen_text_chunks = get_chunks(gen_text, 1) # divide the text into chunks on space
            gen_texts[gen_text_file_name] = gen_text_chunks
            
    ref_audio_duration = remove_silence_and_get_duration_from_wav(ref_audio)
            
    os.makedirs(ARGS.audio_gen_output_folder + f"/{experiment_id}", exist_ok=True)
    
    for gen_text_file_name, gen_text in tqdm(gen_texts.items()):
        final_audio = np.array([], dtype=np.float32)
        output_wav_path_prev = ""
        chunks = []
        
        for i, chunk in enumerate(gen_text):
            chunks.append(chunk)
            
            output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav')

            if ARGS.verbose:
                print("==========================================================================")
                print(f"Current chunks to generate:\t {' '.join(chunks) + ' .'}")
                print(f'Current chunk:\t\t {i+1}: {chunk}')
                print(f'Reference text:\t\t {ref_text}')
                print(f'Reference audio:\t {ref_audio}')
                print("==========================================================================")

            # Set fix_duration for initial generation
            if len(chunks) == 1 or len(chunks) == 2:
                if len(chunk) <= 5:
                    audio_duration = ref_audio_duration + 0.7
                elif len(chunk) <= 10:
                    audio_duration = ref_audio_duration + 1.0
                else:
                    audio_duration = ref_audio_duration + 1.5
                    
                # Run inference
                suppress_print(run_inference, ref_audio, ref_text, ' '.join(chunks) + ' .', output_wav_path, fix_dur=audio_duration)
            else:            
                suppress_print(run_inference, ref_audio, ref_text, ' '.join(chunks) + ' .', output_wav_path)
            
            # Load generated audio
            curr_audio_seg, _ = librosa.load(output_wav_path, sr=FS)
            
            if use_whisper:
                if len(chunks) > 1:
                    # Get timestamps of the last word
                    timestamps = transcribe_audio_with_whisper(output_wav_path)
                    print_verbose(f"Timestamps: {timestamps}")
                    word, start, end = timestamps[-1] # start and end timestamps of the last word
                    print_verbose(f"Last word details: {word.strip().lower()}, {start}, {end}")
                    
                    # Load generated audio
                    trimmed_gen_audio = curr_audio_seg[int(start * FS):int(end * FS)]
                    
                    # NOTE: save the trimmed audio
                    sf.write(Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav'), trimmed_gen_audio, samplerate=FS)
                    
                    # Append the trimmed audio to final audio
                    final_audio = np.concatenate((final_audio, trimmed_gen_audio))
                else:
                    # NOTE: save the trimmed audio
                    sf.write(Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav'), curr_audio_seg, samplerate=FS)
                    final_audio = curr_audio_seg
            else:
                if len(chunks) > 1:
                    # Load audio for the previous and current chunks as numpy arrays
                    wav_prev, sr = librosa.load(output_wav_path_prev, sr=FS)
                    wav_curr, _ = librosa.load(output_wav_path, sr=FS)

                    # Calculate MFCC features (using consistent parameters)
                    mffc_prev = librosa.feature.mfcc(y=wav_prev, sr=sr, n_mfcc=13, n_fft=480, hop_length=240)
                    mffc_curr = librosa.feature.mfcc(y=wav_curr, sr=sr, n_mfcc=13, n_fft=480, hop_length=240)

                    # Perform DTW alignment for subsequence matching
                    _, wp = librosa.sequence.dtw(mffc_prev, mffc_curr, subseq=True)

                    # Calculate overlap length in samples (frame index * hop_length)
                    overlap_len = wp[0, 1] * 240

                    # Clip the overlapping part from the current audio
                    new_samples = wav_curr[overlap_len:] if overlap_len < len(wav_curr) else np.array([], dtype=np.float32)

                    # NOTE: save the new clipped chunk
                    sf.write(Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav'), new_samples, samplerate=FS)
                    
                    # Check that faded_audio contains only 1 voiced segment (or they are close to each other)
                    # NOTE: This did not work as expected
                    # if use_vad:
                    #     voiced_intervals = process_segments(Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav'))
                    #     print_verbose(f"Voiced intervals: {voiced_intervals}")
                    #     if len(voiced_intervals) > 1:
                    #         # check if they are close to each other
                    #         for j in range(len(voiced_intervals) - 1):
                    #             # If they are not close to each other, keep the second one
                    #             if voiced_intervals[j + 1][0] - voiced_intervals[j][1] > 1000:
                    #                 new_samples_trimmed = new_samples[voiced_intervals[j + 1][0]:voiced_intervals[j + 1][1]]
                    #                 print_verbose(f"Taking the second voiced segment: {voiced_intervals[j + 1]}, current new_samples: {new_samples_trimmed}")
                    #             else:
                    #                 continue
                    #         new_samples = new_samples_trimmed
                    
                    if fade_in_out:
                        # Append the new chunk to final audio
                        faded_audio = apply_fade(new_samples, fade_in_duration=0.05, fade_out_duration=0.05, sr=FS)

                        if len(final_audio) > 0:
                            final_audio = np.concatenate((final_audio, generate_silence(0.05, sr=FS)))

                        final_audio = np.concatenate((final_audio, faded_audio))
                    else:
                        final_audio = np.concatenate((final_audio, new_samples))
                else:
                    # NOTE: save the trimmed audio
                    sf.write(Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav'), curr_audio_seg, samplerate=FS)
                    final_audio = curr_audio_seg

                output_wav_path_prev = output_wav_path

        # Save final concatenated audio
        sf.write(Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav'), final_audio, samplerate=FS)
        print_verbose(f"Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')}")
        # Delete chunk audio files
        for i in range(len(gen_text)):
            os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav'))

############################ 5 ############################
# Generate audio per chunk (word), set fix_duration value based on the length of the word.
# Fixed duration: Default variant per chunk
#   chunk_size = 1: generate audio per word (1 word).
def experiment_per_chunk_fix_duration(chunk_size=1, remove_silence=False):
    if remove_silence:
        experiment_id = f'experiment_per_chunk_size_{chunk_size}_fix_duration_silence_removed'
    else:
        experiment_id = f'experiment_per_chunk_size_{chunk_size}_fix_duration'
    
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = [os.path.join(ARGS.text_gen_folder, file) for file in os.listdir(ARGS.text_gen_folder)]
    gen_texts = {}
    for gen_text in gen_texts_paths:
        with open(gen_text, "r") as f:
            gen_text_file_name = os.path.basename(gen_text)
            gen_text = f.read().strip()
            gen_text_chunks = get_chunks(gen_text, chunk_size)
            gen_texts[gen_text_file_name] = gen_text_chunks
            
    # Calculate the duration of the reference audio without silence
    ref_audio_duration = remove_silence_and_get_duration_from_wav(ref_audio)
    
    os.makedirs(Path(ARGS.audio_gen_output_folder, experiment_id), exist_ok=True)
    
    for gen_text_file_name, gen_text in tqdm(gen_texts.items()):
        final_audio = AudioSegment.silent(duration=0)
        
        for i, chunk in enumerate(gen_text):
            output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav')

            if ARGS.verbose:
                print("==========================================================================")
                print(f'Current chunk to generate:\t {i+1}: {chunk}')
                print(f'Reference text:\t\t {ref_text}')
                print(f'Reference audio:\t {ref_audio}')
                print("==========================================================================")
                
            # Based on the length of the word, set the fix_duration value
            if len(chunk) <= 5:
                audio_duration = ref_audio_duration + 0.7
            elif len(chunk) <= 10:
                audio_duration = ref_audio_duration + 1.0
            else:
                audio_duration = ref_audio_duration + 1.5
                
            print_verbose(f"Word duration: {audio_duration} seconds, where ref_audio_duration without silence: {ref_audio_duration} seconds")

            # Run inference
            # run_inference(ref_audio, ref_text, chunk, output_wav_path)
            suppress_print(run_inference, ref_audio, ref_text, chunk, output_wav_path, fix_dur=audio_duration)
            
            # Load generated audio
            generated_audio = AudioSegment.from_wav(output_wav_path)

            # Append to final audio
            final_audio += generated_audio
            
            # Delete the generated audio file chunk
            os.remove(output_wav_path)
            
        # Save final concatenated audio
        audio_save_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')
        final_audio.export(audio_save_path, format="wav")
        print_verbose(f"Audio saved as {audio_save_path}")
        # If remove_silence is True, remove silence from the final audio
        if remove_silence:
            remove_silence_from_wav(src_file=audio_save_path, out_file=audio_save_path)

############################ 6 #############################
# Fixed duration: VAD variant
# Does NOT require trimming the reference audio from the beginning.
# 2 variants of processing the final audio:
#   final_audio_processing_id = 1: Join all output_chunks using cross_correlation
#   final_audio_processing_id = 2: Join all generated_audio_last_segment_trimmed_{X} + output_chunk_{X+1} using cross_correlation.
def experiment_per_chunk_fix_duration_gen_cond_vad(final_audio_processing_id="1"):
    experiment_id = f'experiment_per_chunk_size_1_fix_duration_processing_id_{final_audio_processing_id}_gen_cond_vad'
    
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = [os.path.join(ARGS.text_gen_folder, file) for file in os.listdir(ARGS.text_gen_folder)]
    gen_texts = {}
    for gen_text in gen_texts_paths:
        with open(gen_text, "r") as f:
            gen_text_file_name = os.path.basename(gen_text)
            gen_text = f.read().strip()
            gen_text_chunks = get_chunks(gen_text, 1)
            gen_texts[gen_text_file_name] = gen_text_chunks
            
    # Calculate the duration of the reference audio without silence
    ref_audio_duration = remove_silence_and_get_duration_from_wav(ref_audio)
    
    os.makedirs(Path(ARGS.audio_gen_output_folder, experiment_id), exist_ok=True)
    
    # Keep the original values for conditioning on the last generated audio
    original_ref_text = ref_text
    original_ref_audio = ref_audio
    
    for gen_text_file_name, gen_text in tqdm(gen_texts.items()):
        ref_audio = original_ref_audio
        ref_text = original_ref_text
        ref_appended_audio_duration = 0
        # original_reference_audio = librosa.load(ref_audio, sr=FS)[0]
        num_of_chunks = len(gen_text)
                
        for i, chunk in enumerate(gen_text):
            output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav')

            if ARGS.verbose:
                print("==========================================================================")
                print(f'Current chunk to generate:\t {i+1}: {chunk}')
                print(f'Reference text:\t\t {ref_text}')
                print(f'Reference audio:\t {ref_audio}')
                print("==========================================================================")

            # ============================================
            if len(chunk) <= 5:
                audio_duration = ref_audio_duration + ref_appended_audio_duration + 0.7
            elif len(chunk) <= 10:
                audio_duration = ref_audio_duration + ref_appended_audio_duration + 1.0
            else:
                audio_duration = ref_audio_duration + ref_appended_audio_duration + 1.5
                
            print_verbose(f"Final duration: {audio_duration}")
            print_verbose(f"ref_audio_duration without silence: {ref_audio_duration}")
            print_verbose(f"ref_appended_audio_duration: {ref_appended_audio_duration}")

            # ============================================
            # Run inference
            # run_inference(ref_audio, ref_text, chunk, output_wav_path, fix_dur=audio_duration)
            suppress_print(run_inference, ref_audio, ref_text, chunk, output_wav_path, fix_dur=audio_duration)
            
            # Load generated audio and remove original_ref_audio number of samples from the beginning
            generated_audio = librosa.load(output_wav_path, sr=FS)[0]
            # NOTE: save audio
            sf.write(output_wav_path, generated_audio, FS)
            
            # ============================================
            # Get last generated segment
            voiced_segments = process_segments(output_wav_path, keep_only_last_segment=True)
            generated_audio_last_segment = generated_audio[voiced_segments[0][0]:voiced_segments[0][1]]
            # NOTE: save audio
            generated_audio_last_segment_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'generated_audio_last_segment_{i}.wav')
            sf.write(generated_audio_last_segment_path, generated_audio_last_segment, FS)
            # Trim the last segment
            generated_audio_last_segment_trimmed = process_segments(generated_audio_last_segment_path, trim_ending_value=0.5)
            generated_audio_last_segment_trimmed_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'generated_audio_last_segment_trimmed_{i}.wav')
            sf.write(generated_audio_last_segment_trimmed_path, generated_audio_last_segment_trimmed, FS)
            
            # =============================================
            # Concatenate trimmed generated audio with the original reference audio
            ref_appended_audio_duration = len(generated_audio_last_segment_trimmed) / FS            # duration of the generated audio with trimmed end
            ref_text = original_ref_text + " " + chunk + " "                                        # keep original text + add new text
            conc = np.concatenate((librosa.load(original_ref_audio, sr=FS)[0], generated_audio_last_segment_trimmed)) # Keep original audio + add new audio with trimmed end
            sf.write(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"), conc, FS)
            ref_audio = Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav") 
            
        # Variant 1
        # Join all output_chunks using cross_correlation
        if final_audio_processing_id == "1":
            path_chunk_0 = Path(ARGS.audio_gen_output_folder, experiment_id, "output_chunk_0.wav")
            final_audio, _ = librosa.load(path_chunk_0, sr=FS)
            
            for i in range(num_of_chunks - 1):
                path_chunk_next = Path(ARGS.audio_gen_output_folder, experiment_id, f"output_chunk_{i+1}.wav")
                path_intermediate = Path(ARGS.audio_gen_output_folder, experiment_id, f"final_audio_intermediate_{i}.wav")

                sf.write(path_intermediate, final_audio, FS)
                final_audio = cross_correlation(path_intermediate, path_chunk_next)
                
            # Delete the intermediate audio files
            for i in range(num_of_chunks - 1):
                path_intermediate = Path(ARGS.audio_gen_output_folder, experiment_id, f"final_audio_intermediate_{i}.wav")
                if os.path.exists(path_intermediate):
                    os.remove(path_intermediate)
                
        # Variant 2
        # Join all generated_audio_last_segment_trimmed_{X} + output_chunk_{X+1}, then use cross_correlation to join them
        elif final_audio_processing_id == "2":
            final_audio = np.array([], dtype=np.float32)
            num_of_pairs = 0
            
            # Create pairs of generated_audio_last_segment_trimmed and output_chunk
            for i in range(num_of_chunks - 1):
                path_chunk_next = Path(ARGS.audio_gen_output_folder, experiment_id, f"output_chunk_{i+1}.wav")
                path_gen_audio_last_segment_trimmed = Path(ARGS.audio_gen_output_folder, experiment_id, f'generated_audio_last_segment_trimmed_{i}.wav')
                
                chunk_next = librosa.load(path_chunk_next, sr=FS)[0]
                gen_audio_last_segment_trimmed = librosa.load(path_gen_audio_last_segment_trimmed, sr=FS)[0]

                path_pair = Path(ARGS.audio_gen_output_folder, experiment_id, f'pair_{i}.wav')
                pair = np.concatenate((gen_audio_last_segment_trimmed, chunk_next))
                sf.write(path_pair, pair, FS)
                
                num_of_pairs += 1
                
            # Loop through the pairs and use cross_correlation to join them
            for i in range(num_of_pairs - 1):
                path_pair = Path(ARGS.audio_gen_output_folder, experiment_id, f'pair_{i}.wav')
                path_pair_next = Path(ARGS.audio_gen_output_folder, experiment_id, f'pair_{i+1}.wav')

                if i == 0:
                    final_audio = cross_correlation(path_pair, path_pair_next)
                else:
                    temp_path = Path(ARGS.audio_gen_output_folder, experiment_id, "temp_audio.wav")
                    sf.write(temp_path, final_audio, FS)
                    final_audio = cross_correlation(temp_path, path_pair_next)
                    
            # Cleanup: delete intermediate pair and temp files
            for i in range(num_of_pairs):
                pair_file = Path(ARGS.audio_gen_output_folder, experiment_id, f'pair_{i}.wav')
                if pair_file.exists():
                    os.remove(pair_file)

            temp_path = Path(ARGS.audio_gen_output_folder, experiment_id, "temp_audio.wav")
            if temp_path.exists():
                os.remove(temp_path)

        # Save final audio
        sf.write(Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav'), final_audio, FS)
        print_verbose(f"Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')}")

        # Delete the temporary concatenated audio file
        if os.path.exists(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav")):
            os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"))
        # Delete chunk audio files
        for j in range(num_of_chunks):
            if os.path.exists(Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{j}.wav')):
                os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{j}.wav'))
            if os.path.exists(Path(ARGS.audio_gen_output_folder, experiment_id, f'generated_audio_last_segment_{j}.wav')):
                os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, f'generated_audio_last_segment_{j}.wav'))
            if os.path.exists(Path(ARGS.audio_gen_output_folder, experiment_id, f'generated_audio_last_segment_trimmed_{j}.wav')):
                os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, f'generated_audio_last_segment_trimmed_{j}.wav'))

############################ 7 ############################
# Generate audio per chunk (multiple words), which consist of a minimum number of characters.
#   cond_on_last_gen_audio = False: conditioned on the fixed reference audio and text
#   cond_on_last_gen_audio = True: conditioned on the last generated audio chunk and text (without the reference audio)
def experiment_per_dynamic_sized_chunk(cond_on_last_gen_audio=False, min_chars=10):
    if cond_on_last_gen_audio:
        experiment_id = f'experiment_per_dynamic_sized_chunk__min_chars_{min_chars}_gen_cond'
    else:
        experiment_id = f'experiment_per_dynamic_sized_chunk_min_chars_{min_chars}'
    
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = [os.path.join(ARGS.text_gen_folder, file) for file in os.listdir(ARGS.text_gen_folder)]
    gen_texts = {}
    for gen_text in gen_texts_paths:
        with open(gen_text, "r") as f:
            gen_text_file_name = os.path.basename(gen_text)
            gen_text = f.read().strip()
            gen_text_chunks = get_dynamic_chunks(gen_text, min_chars=min_chars) # divide the text into chunks
            gen_texts[gen_text_file_name] = gen_text_chunks
            
    os.makedirs(Path(ARGS.audio_gen_output_folder, experiment_id), exist_ok=True)
    
    original_ref_text = ref_text
    original_ref_audio = ref_audio
    
    for gen_text_file_name, gen_text in tqdm(gen_texts.items()):
        final_audio = AudioSegment.silent(duration=0)
        ref_audio = original_ref_audio
        ref_text = original_ref_text
        for i, chunk in enumerate(gen_text):
            output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav')

            if ARGS.verbose:
                print("==========================================================================")
                print(f'Current chunk to generate:\t {i+1}: {chunk}')
                print(f'Reference text:\t\t {ref_text}')
                print(f'Reference audio:\t {ref_audio}')
                print("==========================================================================")

            # Run inference
            # run_inference(ref_audio, ref_text, chunk, output_wav_path)
            suppress_print(run_inference, ref_audio, ref_text, chunk, output_wav_path)
            
            # Load generated audio
            generated_audio = AudioSegment.from_wav(output_wav_path)

            # Append to final audio
            final_audio += generated_audio
            
            # Delete the generated audio file chunk
            os.remove(output_wav_path)
            
            # Update reference text
            if cond_on_last_gen_audio:
                ref_text = chunk + " ."
                # Save the generated audio to a temporary file
                generated_audio.export(Path(ARGS.audio_gen_output_folder, experiment_id, "chunk_tmp.wav"), format="wav")
                ref_audio = Path(ARGS.audio_gen_output_folder, experiment_id, "chunk_tmp.wav")

        # Save final concatenated audio
        final_audio.export(Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav'), format="wav")
        # Delete the temporary generated audio file
        if os.path.exists(Path(ARGS.audio_gen_output_folder, experiment_id, "chunk_tmp.wav")):
            os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, "chunk_tmp.wav"))
        print_verbose(f"Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')}")
        
############################ 8 ############################
# IMPORTANT: DOES require working with the whole generated audio (ref & gen).
#
# Fixed duration: Whisper variant
# Generate audio per chunk (word), set fix_duration value based on the length of the word.
#   Firstly, first word is generated.
#   Then, this generated word is trimmed at the end based on Whisper timestamps of the word.
#   (Idea is that the model would generate the word ending and generate the next word - this could lead to better intertwining of the words.)
#   The trimmed word is present in the reference text (only this one word)
#   Next, the generated audio is stripped of the reference audio (containing the trimmed word) and only the following audio is appended to the final audio.
def experiment_per_chunk_fix_duration_gen_cond_whisper(remove_silence=False):
    if remove_silence:
        experiment_id = f'experiment_per_chunk_size_1_fix_duration_gen_cond_whisper_silence_removed'
    else:
        experiment_id = f'experiment_per_chunk_size_1_fix_duration_gen_cond_whisper'
    
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = [os.path.join(ARGS.text_gen_folder, file) for file in os.listdir(ARGS.text_gen_folder)]
    gen_texts = {}
    for gen_text in gen_texts_paths:
        with open(gen_text, "r") as f:
            gen_text_file_name = os.path.basename(gen_text)
            gen_text = f.read().strip()
            gen_text_chunks = get_chunks(gen_text, 1)
            gen_texts[gen_text_file_name] = gen_text_chunks
            
    # Calculate the duration of the reference audio without silence
    ref_audio_duration = remove_silence_and_get_duration_from_wav(ref_audio)
    
    os.makedirs(Path(ARGS.audio_gen_output_folder, experiment_id), exist_ok=True)
    
    # Keep the original values for conditioning on the last generated audio variant
    original_ref_text = ref_text
    original_ref_audio = ref_audio
    
    for gen_text_file_name, gen_text in tqdm(gen_texts.items()):
        ref_audio = original_ref_audio
        ref_text = original_ref_text
        ref_appended_audio_duration = 0
        final_audio = np.array([], dtype=np.float32)
        i = 0
                
        for i, chunk in enumerate(gen_text):
            curr_ref_audio = librosa.load(ref_audio, sr=FS)[0] # current reference = original_ref + previously generated audio   
            
            output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav')

            if ARGS.verbose:
                print("==========================================================================")
                print(f'Current chunk to generate:\t {i+1}: {chunk}')
                print(f'Reference text:\t\t {ref_text}')
                print(f'Reference audio:\t {ref_audio}')
                print("==========================================================================")

            # ============================================
            # Based on the length of the word, set the fix_duration value            
            if len(chunk) <= 5:
                audio_duration = ref_audio_duration + ref_appended_audio_duration + 0.7
            elif len(chunk) <= 10:
                audio_duration = ref_audio_duration + ref_appended_audio_duration + 1.0
            else:
                audio_duration = ref_audio_duration + ref_appended_audio_duration + 1.5
                
            print_verbose(f"Final duration: {audio_duration}")
            print_verbose(f"ref_audio_duration without silence: {ref_audio_duration}")
            print_verbose(f"ref_appended_audio_duration: {ref_appended_audio_duration}")

            # ============================================
            # Run inference
            # run_inference(ref_audio, ref_text, chunk, output_wav_path, fix_dur=audio_duration)
            suppress_print(run_inference, ref_audio, ref_text, chunk, output_wav_path, fix_dur=audio_duration)
            
            # Load generated audio and remove curr_ref_audio number of samples from the beginning
            generated_audio = librosa.load(output_wav_path, sr=FS)[0][len(curr_ref_audio):]
            # NOTE: save audio
            sf.write(output_wav_path, generated_audio, FS)
            
            # ============================================
            # Cut part of the end of the generated audio
            trimmed_gen_audio = generated_audio
            # timestamps = transcribe_audio_with_whisper(output_wav_path)
            timestamps = suppress_print(transcribe_audio_with_whisper, output_wav_path)
            print_verbose(timestamps)
            for word, start, end in timestamps:
                if word.strip().lower() == chunk.strip().lower():
                    print_verbose(f"Word details: {word.strip().lower()}, {start}, {end}")
                    trimmed_gen_audio = generated_audio[int(start * FS):int(end * FS)]
                    break
            
            # NOTE: save audio
            # generated_ref_audio_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'trimmed_gen_audio_{i}.wav')
            # sf.write(generated_ref_audio_path, trimmed_gen_audio, FS)
            
            # =============================================
            # Concatenate trimmed generated audio with the original reference audio
            ref_appended_audio_duration = len(trimmed_gen_audio) / FS                               # duration of the generated audio with trimmed end
            ref_text = original_ref_text + " " + chunk + " "                                        # keep original text + add new text
            conc = np.concatenate((librosa.load(original_ref_audio, sr=FS)[0], trimmed_gen_audio))  # keep original audio + add new audio with trimmed end
            sf.write(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"), conc, FS)
            ref_audio = Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav") 
            
            # Save audio
            final_audio = np.concatenate((final_audio, generated_audio))
            
        # Save final concatenated audio
        audio_save_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')
        sf.write(audio_save_path, final_audio, FS)
        if remove_silence:
            remove_silence_from_wav(src_file=audio_save_path, out_file=audio_save_path)
        # Delete the temporary concatenated audio file
        if os.path.exists(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav")):
            os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"))
            
        # Delete chunk audio files
        for j in range(i+1):
            if os.path.exists(Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{j}.wav')):
                os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{j}.wav'))
                
        print_verbose(f"Audio saved as {audio_save_path}")

############################ 9 ############################
# IMPORTANT: DOES require working with the whole generated audio (ref & gen).
# Fixed duration: Final audio accumulation variant
#   Final audio is accumulated and used as reference for generation of the next word.
#   Once the reference audio is too long (10+ secs), the original reference audio is removed and only generated audio is used as reference.
#   From the results, it can be heard that conditioning on the generated audio does not work well and newly generated audio sounds low quality.
def experiment_per_chunk_fix_duration_gen_cond_acc(remove_silence=False):
    if remove_silence:
        experiment_id = f'experiment_per_chunk_size_1_fix_duration_gen_cond_acc_silence_removed'
    else:
        experiment_id = f'experiment_per_chunk_size_1_fix_duration_gen_cond_acc'
    
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = [os.path.join(ARGS.text_gen_folder, file) for file in os.listdir(ARGS.text_gen_folder)]
    gen_texts = {}
    for gen_text in gen_texts_paths:
        with open(gen_text, "r") as f:
            gen_text_file_name = os.path.basename(gen_text)
            gen_text = f.read().strip()
            gen_text_chunks = get_chunks(gen_text, 1)
            gen_texts[gen_text_file_name] = gen_text_chunks
    
    os.makedirs(Path(ARGS.audio_gen_output_folder, experiment_id), exist_ok=True)
    
    # Keep the original values for conditioning on the last generated audio variant
    original_ref_text = ref_text
    original_ref_audio = ref_audio
    
    for gen_text_file_name, gen_text in tqdm(gen_texts.items()):
        # Set the original reference audio for each generation
        ref_audio = original_ref_audio
        ref_text = original_ref_text
        # Calculate the duration of the reference audio without silence
        ref_audio_duration = remove_silence_and_get_duration_from_wav(ref_audio)
        # Initialize variables for accumulating the reference audio
        ref_appended_audio_duration = 0
        # Used to store the generated audio if the next reference audio is longer than 15 seconds
        final_audio_prev = [np.array([], dtype=np.float32)]
        final_audio = np.array([], dtype=np.float32)
        # True original reference audio
        original_reference_audio = librosa.load(ref_audio, sr=FS)[0]  
        chnks = []
        # Save idx as it will be used later to delete temporary audio files
        idx = 0
        # Current audio
        curr_generated_audio = np.array([], dtype=np.float32)
                
        for idx, chunk in enumerate(gen_text):
            chnks.append(chunk)
            
            output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{idx}.wav')

            if ARGS.verbose:
                print("==========================================================================")
                print(f'Current chunk to generate:\t {idx+1}: {chunk}')
                print(f'Reference text:\t\t {ref_text}')
                print(f'Reference audio:\t {ref_audio}')
                print("==========================================================================")
            
            if len(chunk) <= 5:
                audio_duration = ref_audio_duration + ref_appended_audio_duration + 0.5
            elif len(chunk) <= 10:
                audio_duration = ref_audio_duration + ref_appended_audio_duration + 0.8
            else:
                audio_duration = ref_audio_duration + ref_appended_audio_duration + 1.3
                
            print_verbose(f"Final duration: {audio_duration}")
            print_verbose(f"ref_audio_duration without silence: {ref_audio_duration}")
            print_verbose(f"ref_appended_audio_duration: {ref_appended_audio_duration}")

            # ============================================
            # Run inference
            # run_inference(ref_audio, ref_text, chunk, output_wav_path, fix_dur=audio_duration)
            suppress_print(run_inference, ref_audio, ref_text, chunk, output_wav_path, fix_dur=audio_duration)
            
            next_ref_audio_duration = len(librosa.load(output_wav_path, sr=FS)[0]) / FS
            print_verbose(f"Next reference audio duration: {next_ref_audio_duration}")
            
            # Load generated audio and remove reference part from the beginning
            # generated_audio = librosa.load(output_wav_path, sr=FS)[0][len(original_reference_audio):]
            # sf.write(output_wav_path, generated_audio, FS)
            
            # =============================================
            curr_generated_audio = librosa.load(output_wav_path, sr=FS)[0][len(original_reference_audio):]
            # If the next reference audio is shorter than 15 seconds, append the generated audio to the reference audio
            if next_ref_audio_duration < 10:
                ref_appended_audio_duration = len(curr_generated_audio) / FS
                ref_text += chunk + " "
                ref_audio = output_wav_path
                final_audio = curr_generated_audio
            else:
                # Save the generated audio to a temporary file, it will be considered later once everything is generated
                final_audio_prev.append(librosa.load(output_wav_path, sr=FS)[0][len(original_reference_audio):])
                # Clear current final_audio since all its content is in final_audio_prev
                final_audio = np.array([], dtype=np.float32)
                # No ref_appended_audio_duration, since what's been appended will be used as reference audio
                ref_appended_audio_duration = 0
                # Create ref_text from the chunks generated so far
                ref_text = " ".join(chnks) + " "
                # Clear the chunks since they won't be used anymore
                chnks = []
                # Since the ref_audio will change, the audio duration will change as well
                ref_audio_duration = len(curr_generated_audio) / FS
                # Save the generated audio to a temporary file as it will be used as reference audio
                sf.write(output_wav_path, curr_generated_audio, FS)
                ref_audio = output_wav_path
                # Also set the original reference audio to the current ref_audio
                original_reference_audio = librosa.load(ref_audio, sr=FS)[0]
                # Clear curr_generated_audio
                curr_generated_audio = np.array([], dtype=np.float32)
                
        # Concatenate all generated audio chunks
        if final_audio_prev:
            # Concatenate everything in the final_audio_prev list
            final_audio = np.concatenate(final_audio_prev)
            final_audio = np.concatenate((final_audio, curr_generated_audio))
        # Concatenate the final audio with the generated audio
        else:
            final_audio = curr_generated_audio
        # Save final concatenated audio
        audio_save_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')
        sf.write(audio_save_path, final_audio, FS)
        if remove_silence:
            remove_silence_from_wav(src_file=audio_save_path, out_file=audio_save_path)
        
        # Delete the temporary output_chunk audio files
        for i in range(idx + 1):
            if os.path.exists(Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav')):
                os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav'))
                
        print_verbose(f"Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')}")
        
############################ 10 ############################
# TODO Describe why this did not work.
# Sample variant - does not work, number of samples differ after each generation, thus, accumulating the reference audio samples does not work.
# Generates audio per chunk (word), set fix_duration value based on the length of the word.
# Each generated word is trimmed at the end and used as reference for the next word.

if __name__ == "__main__":   
    ARGS = parse_args()
    
    if ARGS.list_experiments:
        print("List of experiments to run:")
        print("\t 0: Generate audio using the model as-is, e.g., full sentence generation, conditioned on the fixed reference.")
        print("\t 1: Generate audio per chunk (word, multiple words), conditioned on the fixed reference.")
        print("\t 2: Generate audio per chunk (word, multiple words), conditioned on the fixed reference & last generated audio.")
        print("\t 3: Generate audio per chunk (word, multiple words), conditioned on the fixed reference & last generated audio with silence.")
        print("\t 4: Generate audio per increasing chunk (word, multiple words), conditioned on the fixed reference, obtain only the new word using DTW and concatenate with the previous audio.")
        print("\t 5: Generate audio per chunk (word, multiple words), conditioned on the fixed reference, each generation is set with fix_duration to ensure proper length.")
        print("\t 6: Generate audio per chunk (word, multiple words), conditioned on the fixed reference, each generation is set with fix_duration to ensure proper length, using VAD and cross-correlation.")
        print("\t 7: Generate audio per chunk (word, multiple words), min_num_chars is set to ensure proper length.")
        print("\t 8: Generate audio per chunk (word, multiple words), fixed duration, using Whisper timestamps to trim the generated audio.")
        print("\t 9: Generate audio per chunk (word, multiple words), accumulating the generated audio and using it as reference for the next generation.")
        exit(0)
    
    # Load the F5-TTS model
    load_f5_model()
    
    # Example usage of the run_inference function
    # run_inference(ARGS.audio_ref_file, "My wife, on the spur of the moment, managed to give the gentleman a very good dinner." , "My wife, on the spur of the moment, managed to give the gentleman a very good dinner.", "/home/m/output.wav")
    
    if ARGS.experiment is None:
        # print("Running all experiments...")
        
        # print("Running experiment 0...")
        # experiment_default_per_sentence()
        
        # print("Running experiment 1...")
        # experiment_per_chunk(chunk_size=1)
        # experiment_per_chunk(chunk_size=3)
        
        # print("Running experiment 2...")
        # experiment_per_chunk_gen_cond(chunk_size=1)
        # experiment_per_chunk_gen_cond(chunk_size=3)
        
        # print("Running experiment 3...")
        # experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len_ms=100)
        # experiment_per_chunk_gen_cond_with_silence(chunk_size=3, silence_len_ms=100)
        # experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len_ms=300)
        # experiment_per_chunk_gen_cond_with_silence(chunk_size=3, silence_len_ms=300)
        
        # print("Running experiment 4...")
        # experiment_dtw()
        # load_whisper_timestamped_model()
        # experiment_dtw(use_whisper=True)
        
        # print("Running experiment 5...")
        # experiment_per_chunk_fix_duration(chunk_size=1)
        # experiment_per_chunk_fix_duration(chunk_size=3)
        # experiment_per_chunk_fix_duration(chunk_size=1, remove_silence=True)
        # experiment_per_chunk_fix_duration(chunk_size=3, remove_silence=True)
        
        # print("Running experiment 6...")
        # experiment_per_chunk_fix_duration_gen_cond_vad(final_audio_processing_id="1")
        # experiment_per_chunk_fix_duration_gen_cond_vad(final_audio_processing_id="2")
        
        # print("Running experiment 7...")
        # experiment_per_dynamic_sized_chunk(cond_on_last_gen_audio=False)
        # experiment_per_dynamic_sized_chunk(cond_on_last_gen_audio=True)
        
        ##################################################################
        # DOES REQUIRE WORKING WITH THE WHOLE GENERATED AUDIO (ref & gen).
        ##################################################################
        
        print("Running experiment 8...")
        load_whisper_timestamped_model()        
        experiment_per_chunk_fix_duration_gen_cond_whisper(remove_silence=True)
        experiment_per_chunk_fix_duration_gen_cond_whisper(remove_silence=False)
        
        print("Running experiment 9...")
        experiment_per_chunk_fix_duration_gen_cond_acc(remove_silence=True)
        experiment_per_chunk_fix_duration_gen_cond_acc(remove_silence=False)
        
    else:
        print(f"Running experiment {ARGS.experiment}...")
        if ARGS.experiment == 0:
            experiment_default_per_sentence()
        elif ARGS.experiment == 1:
            experiment_per_chunk(chunk_size=1)
            experiment_per_chunk(chunk_size=2)
        elif ARGS.experiment == 2:
            experiment_per_chunk_gen_cond(chunk_size=1)
            experiment_per_chunk_gen_cond(chunk_size=2)
        elif ARGS.experiment == 3:
            experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len_ms=100)
            experiment_per_chunk_gen_cond_with_silence(chunk_size=2, silence_len_ms=100)
            experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len_ms=300)
            experiment_per_chunk_gen_cond_with_silence(chunk_size=2, silence_len_ms=300)
            experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len_ms=500)
            experiment_per_chunk_gen_cond_with_silence(chunk_size=2, silence_len_ms=500)
        elif ARGS.experiment == 4:
            experiment_dtw()
        elif ARGS.experiment == 5:
            experiment_per_chunk_fix_duration(chunk_size=1)
            experiment_per_chunk_fix_duration(chunk_size=2)
            experiment_per_chunk_fix_duration(chunk_size=1, cond_on_last_gen_audio=True, sliding_window_size=4)
            experiment_per_chunk_fix_duration(chunk_size=2, cond_on_last_gen_audio=True, sliding_window_size=4)
            experiment_per_chunk_fix_duration(chunk_size=1, cond_on_last_gen_audio=True, sliding_window_size=2)
            experiment_per_chunk_fix_duration(chunk_size=2, cond_on_last_gen_audio=True, sliding_window_size=2)
        elif ARGS.experiment == 6:
            experiment_per_dynamic_sized_chunk(cond_on_last_gen_audio=False)
            experiment_per_dynamic_sized_chunk(cond_on_last_gen_audio=True)
        else:
            print("Invalid experiment ID. Please select a valid experiment ID.")
            exit(0)