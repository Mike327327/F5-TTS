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
    CKPT_FILE = "/mnt/matylda4/xluner01/F5-TTS/ckpts/LibriTTS_100_360_500/model_1600000_en.pt"
    MODEL_CFG = "/mnt/matylda4/xluner01/F5-TTS/src/f5_tts/configs/F5TTS_Small_train_LibriTTS.yaml"
    VOCAB_FILE = "/mnt/matylda4/xluner01/F5-TTS/data/LibriTTS_100_360_500_char/vocab.txt"
    ######################## EN (causal) ########################
    # CKPT_FILE = "/mnt/matylda4/xluner01/F5-TTS/ckpts/LibriTTS_100_360_500/model_1425000_en_causal.pt"
    # MODEL_CFG = "/mnt/matylda4/xluner01/F5-TTS/src/f5_tts/configs/F5TTS_Small_train_LibriTTS.yaml"
    # VOCAB_FILE = "/mnt/matylda4/xluner01/F5-TTS/data/LibriTTS_100_360_500_char/vocab.txt"
    ######################## EN (pretrained) ########################
    # CKPT_FILE = "/homes/eva/xl/xluner01/.cache/huggingface/hub/models--SWivid--F5-TTS/snapshots/84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_Base/model_1200000.safetensors"
    # MODEL_CFG = "/mnt/matylda4/xluner01/F5-TTS/src/f5_tts/configs/F5TTS_Base_train.yaml"
    # VOCAB_FILE = "/mnt/matylda4/xluner01/F5-TTS/src/f5_tts/infer/examples/vocab.txt"
    
    SPEED = speed # 1.0
    CROSSFADE = cross_fade_duration
    VOCODER_LOCAL_PATH = "/mnt/matylda4/xluner01/F5-TTS/checkpoints/vocos-mel-24khz"
    VOCODER_NAME = mel_spec_type # vocos
    LOAD_VOCODER_FROM_LOCAL = True
    REMOVE_SILENCE = False

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
    
# Convert AudioSegment to numpy array (mono)
def audiosegment_to_np(audio_seg):
    samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
    if audio_seg.channels > 1:
        # If stereo, take only the first channel
        print("Audio is stereo. Taking only the first channel.")
        samples = samples.reshape((-1, audio_seg.channels))[:, 0]
    return samples

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
        os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"))	
        print_verbose(f"Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')}")

############################ 3 ############################
# Generate audio per chunk (word, multiple words), conditioned on the fixed reference & last generated audio with silence and text.
def experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len=100):
    experiment_id = f'experiment_per_chunk_size_{chunk_size}_gen_cond_with_silence_{silence_len}_ms'
        
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
def experiment_dtw(window_size=1):
    experiment_id = f'experiment_dtw_window_size_{window_size}'
        
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

            # Run inference
            # run_inference(ref_audio, ref_text, chunk, output_wav_path)
            suppress_print(run_inference, ref_audio, ref_text, ' '.join(chunks) + ' .', output_wav_path)
            
            # Load generated audio
            curr_audio_seg, _ = librosa.load(output_wav_path, sr=24000)
            
            if len(chunks) > 1:
                # Load audio for the previous and current chunks as numpy arrays
                wav_prev, sr = librosa.load(output_wav_path_prev, sr=24000)
                wav_curr, _ = librosa.load(output_wav_path, sr=24000)

                # Calculate MFCC features (using consistent parameters)
                mffc_prev = librosa.feature.mfcc(y=wav_prev, sr=sr, n_mfcc=13, n_fft=480, hop_length=240)
                mffc_curr = librosa.feature.mfcc(y=wav_curr, sr=sr, n_mfcc=13, n_fft=480, hop_length=240)

                # Perform DTW alignment for subsequence matching
                _, wp = librosa.sequence.dtw(mffc_prev, mffc_curr, subseq=True)

                # Calculate overlap length in samples (frame index * hop_length)
                overlap_len = wp[0, 1] * 240

                # Clip the overlapping part from the current audio
                new_samples = wav_curr[overlap_len:] if overlap_len < len(wav_curr) else np.array([], dtype=np.float32)

                # Save the new clipped chunk (optional for debugging)
                # sf.write(Path(ARGS.audio_gen_output_folder, experiment_id, f'check_output_chunk_{i}_new.wav'), new_samples, samplerate=sr)
                
                # Append the new chunk to final audio
                final_audio = np.concatenate((final_audio, new_samples))
    
            else:
                final_audio = curr_audio_seg

            output_wav_path_prev = output_wav_path

        # Save final concatenated audio
        sf.write(Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav'), final_audio, samplerate=24000)
        print_verbose(f"Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')}")
        # Delete chunk audio files
        for i in range(len(gen_text)):
            os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav'))

############################ 5 ############################
# Generate audio per chunk (word), set fix_duration value based on the length of the word.
#  chunk_size = 1:                     Generate audio per word (1 word).
#  cond_on_last_gen_audio = False:     Conditioned on the fixed reference audio and text.
#  cond_on_last_gen_audio = True:      Conditioned on the last generated audio chunk and text, without the reference audio.
#  sliding_window_size = 4:            Number of last generated audio chunks to be concatenated.
def experiment_per_chunk_fix_duration(chunk_size=1, cond_on_last_gen_audio=False, sliding_window_size=4):
    if cond_on_last_gen_audio:
        experiment_id = f'experiment_per_chunk_size_{chunk_size}_fix_duration_gen_cond'
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
    
    # Keep the original values for conditioning on the last generated audio variant
    original_ref_text = ref_text
    original_ref_audio = ref_audio
    
    for gen_text_file_name, gen_text in tqdm(gen_texts.items()):
        final_audio = AudioSegment.silent(duration=0)
        ref_audio = original_ref_audio
        ref_text = original_ref_text
        chunks_text_acc = []
        chunks_audio_acc = []
        
        for i, chunk in enumerate(gen_text):
            chunks_text_acc.append(chunk)
            
            output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav')

            if ARGS.verbose:
                print("==========================================================================")
                print(f'Current chunk to generate:\t {i+1}: {chunk}')
                print(f'Reference text:\t\t {ref_text}')
                print(f'Reference audio:\t {ref_audio}')
                print("==========================================================================")
                
            # Based on the length of the word, set the fix_duration value
            if len(chunk) <= 5:
                audio_duration = ref_audio_duration + 0.4
            elif len(chunk) <= 10:
                audio_duration = ref_audio_duration + 0.7
            else:
                audio_duration = ref_audio_duration + 1.0
                
            print_verbose(f"Word duration: {audio_duration} seconds, where ref_audio_duration without silence: {ref_audio_duration} seconds")

            # Run inference
            # run_inference(ref_audio, ref_text, chunk, output_wav_path)
            suppress_print(run_inference, ref_audio, ref_text, chunk, output_wav_path, fix_dur=audio_duration)
            
            # Load generated audio
            generated_audio = AudioSegment.from_wav(output_wav_path)

            # Append to final audio
            final_audio += generated_audio
            
            # Append to sliding window in order to be able to concatenate the last sliding_window_size generated audio chunks
            chunks_audio_acc.append(generated_audio)
            
            # Delete the generated audio file chunk
            os.remove(output_wav_path)
            
            # Update reference text and audio
            if cond_on_last_gen_audio and len(chunks_text_acc) >= sliding_window_size:
                ref_text = " ".join(chunks_text_acc[-sliding_window_size:]) + " ."
                # Concatenate last sliding_window_size generated audio chunks from chunks_audio_acc
                ref_audio = AudioSegment.silent(duration=0)
                for j in range(len(chunks_audio_acc) - sliding_window_size, len(chunks_audio_acc)):
                    ref_audio += chunks_audio_acc[j]
                # Save the concatenated audio to a temporary file
                ref_audio.export(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"), format="wav")
                ref_audio = Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav") 

        # Save final concatenated audio
        final_audio.export(Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav'), format="wav")
        # Delete the temporary concatenated audio file
        if cond_on_last_gen_audio:
            os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"))
        print_verbose(f"Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')}")

############################ 6 ############################
# Generate audio per chunk (multiple words), which consist of a minimum number of characters.
#   cond_on_last_gen_audio = False:     Conditioned on the fixed reference audio and text.
#   cond_on_last_gen_audio = True:      Conditioned on the last generated audio chunk and text (without the reference audio).
def experiment_per_dynamic_sized_chunk(cond_on_last_gen_audio=False):
    if cond_on_last_gen_audio:
        experiment_id = f'experiment_per_dynamic_sized_chunk_gen_cond'
    else:
        experiment_id = f'experiment_per_dynamic_sized_chunk'
    
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
            gen_text_chunks = get_dynamic_chunks(gen_text, min_chars=10) # divide the text into chunks
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
        os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, "chunk_tmp.wav"))
        print_verbose(f"Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{os.path.splitext(os.path.basename(gen_text_file_name))[0]}.wav')}")
        
if __name__ == "__main__":   
    ARGS = parse_args()
    
    if ARGS.list_experiments:
        print("List of experiments to run:")
        print("\t 0: Generate audio using the model as-is, e.g., full sentence generation, conditioned on the fixed reference.")
        print("\t 1: Generate audio per chunk (word, multiple words), conditioned on the fixed reference.")
        print("\t 2: Generate audio per chunk (word, multiple words), conditioned on the fixed reference & last generated audio.")
        print("\t 3: Generate audio per chunk (word, multiple words), conditioned on the fixed reference & last generated audio with silence.")
        print("\t 4: Generate audio per increasing chunk (word, multiple words), conditioned on the fixed reference, obtain only the new word using DTW and concatenate with the previous audio.")
        print("\t 5: Generate audio per chunk (word), set fix_duration value based on the length of the word. Conditioned on the fixed reference or the last generated audio.")
        print("\t 6: Generate audio per dynamically sized chunk (multiple words). Conditioned either on the fixed reference or the last generated audio.")
        exit(0)
    
    # Load the F5-TTS model
    load_f5_model()
    
    # Run the specified experiment
    # run_inference(ARGS.audio_ref_file, "My wife, on the spur of the moment, managed to give the gentleman a very good dinner." , "My wife, on the spur of the moment, managed to give the gentleman a very good dinner.", "/home/m/output.wav")
    
    if ARGS.experiment is None:
        print("Running all experiments...")
        
        print("Running experiment 0...")
        experiment_default_per_sentence()
        
        print("Running experiment 1...")
        experiment_per_chunk(chunk_size=1)
        # experiment_per_chunk(chunk_size=2)
        
        print("Running experiment 2...")
        experiment_per_chunk_gen_cond(chunk_size=1)
        # experiment_per_chunk_gen_cond(chunk_size=2)
        
        print("Running experiment 3...")
        experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len=100)
        experiment_per_chunk_gen_cond_with_silence(chunk_size=2, silence_len=100)
        experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len=300)
        experiment_per_chunk_gen_cond_with_silence(chunk_size=2, silence_len=300)
        
        print("Running experiment 4...")
        experiment_dtw(window_size=1)
        
        print("Running experiment 5...")
        experiment_per_chunk_fix_duration(chunk_size=1)
        
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
            experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len=100)
            experiment_per_chunk_gen_cond_with_silence(chunk_size=2, silence_len=100)
            experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len=300)
            experiment_per_chunk_gen_cond_with_silence(chunk_size=2, silence_len=300)
            experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len=500)
            experiment_per_chunk_gen_cond_with_silence(chunk_size=2, silence_len=500)
        elif ARGS.experiment == 4:
            experiment_dtw(window_size=1)
        elif ARGS.experiment == 5:
            experiment_per_chunk_fix_duration(chunk_size=1)
        else:
            print("Invalid experiment ID. Please select a valid experiment ID.")
            exit(0)