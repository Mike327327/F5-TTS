from pydub import AudioSegment
import argparse
import os
from tqdm import tqdm
from importlib.resources import files
from pathlib import Path
import numpy as np
import soundfile as sf
from cached_path import cached_path
from omegaconf import OmegaConf

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

CKPT_FILE = "/home/m/F5-TTS/ckpts/model_1600000_en.pt"
MODEL_CFG = "/home/m/F5-TTS/src/f5_tts/configs/F5TTS_Small_train.yaml"
VOCAB_FILE = "/home/m/F5-TTS/data/LibriTTS_100_360_500_char/vocab.txt"
SPEED = speed # 1.0
CROSSFADE = cross_fade_duration
VOCODER_LOCAL_PATH = "/home/m/F5-TTS/checkpoints/vocos-mel-24khz"
VOCODER_NAME = mel_spec_type # vocos
LOAD_VOCODER_FROM_LOCAL = True
REMOVE_SILENCE = False

ARGS = None

def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio using the F5 model.")
    
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
    parser.add_argument(
        "--list_experiment",
        type=str, 
        help="List of experiments to run. If selected, this program will terminate."
    )

    return parser.parse_args()

def load_f5_model():
    global vocoder, model_cls, model_cfg, ema_model
    
    vocoder = load_vocoder(vocoder_name=VOCODER_NAME, is_local=True, local_path=VOCODER_LOCAL_PATH)
    
    model_cls = DiT
    model_cfg = OmegaConf.load(MODEL_CFG).model.arch
  
    print("Initializing model: F5-TTS")
    print(f"Initializing vocoder: {VOCODER_NAME}")
    
    ema_model = load_model(model_cls, model_cfg, CKPT_FILE, mel_spec_type=VOCODER_NAME, vocab_file=VOCAB_FILE)

def run_inference(ref_audio, ref_text, gen_text, gen_audio_path):
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    voices = {"main": main_voice}
    voices["main"]["ref_audio"], voices["main"]["ref_text"] = preprocess_ref_audio_text(
        voices["main"]["ref_audio"], voices["main"]["ref_text"]
    )

    ref_audio_ = voices["main"]["ref_audio"]
    ref_text_ = voices["main"]["ref_text"]
    gen_text_ = gen_text.strip()
    
    # print(f"Reference text: {ref_text_}")
    # print(f"Generated text: {gen_text_}")
    # print(f"Reference audio: {ref_audio_}")
    
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
        fix_duration=fix_duration,
    )

    with open(gen_audio_path, "wb") as f:
        sf.write(f.name, wave, sample_rate)
        if REMOVE_SILENCE:
            remove_silence_for_generated_wav(f.name)
        if ARGS.verbose:
            print(f.name)

# Based on the chunk size, divide the text into chunks
def get_chunks(text, chunk_size):
    text_chunks = []
    text_words = text.split(" ")
    for i in range(0, len(text_words), chunk_size):
        text_chunk = " ".join(text_words[i:i+chunk_size])
        text_chunks.append(text_chunk)
    return text_chunks
    
# 0
# Generate audio using the model as-is, e.g., full sentence generation, conditioned on the fixed reference audio and text
def experiment_default_per_sentence():
    experiment_id = f'experiment_default_per_sentence'
    
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = os.listdir(ARGS.text_gen_folder, "r")
    gen_texts = {}
    for gen_text in gen_texts_paths:
        with open(gen_text, "r") as f:
            gen_text_file_name = os.path.basename(gen_text)
            gen_texts[gen_text_file_name] = f.read().strip()
    
    os.makedirs(Path(ARGS.audio_gen_output_folder, experiment_id), exist_ok=True)
    
    for gen_text_file_name, gen_text in tqdm(gen_texts):
        output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'{gen_text_file_name}.wav')

        if ARGS.verbose:
            print("=====================================")
            print(f'Text to generate:\t {gen_text}')
            print(f'Reference text:\t {ref_text}')
            print(f'Reference audio:\t {ref_audio}')
            print("=====================================")

        # Run inference
        run_inference(ref_audio, ref_text, gen_text, output_wav_path)
        
        # Save final concatenated audio
        if ARGS.verbose:
            print(f'Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{gen_text_file_name}.wav')}')

# 1
# Generate audio per chunk (word, multiple words), conditioned on the fixed reference audio and text
def experiment_per_chunk(chunk_size=1):
    experiment_id = f'experiment_per_chunk_size_{chunk_size}'
    
    final_audio = AudioSegment.silent(duration=0)
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = os.listdir(ARGS.text_gen_folder, "r")
    gen_texts = {}
    for gen_text in gen_texts_paths:
        with open(gen_text, "r") as f:
            gen_text_file_name = os.path.basename(gen_text)
            gen_text = f.read().strip()
            gen_text_chunks = get_chunks(gen_text, chunk_size) # divide the text into chunks
            gen_texts[gen_text_file_name] = gen_text_chunks
    
    os.makedirs(Path(ARGS.audio_gen_output_folder, experiment_id), exist_ok=True)
    
    for gen_text_file_name, gen_text in tqdm(gen_texts):
        for i, chunk in enumerate(gen_text):
            output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav')

            if ARGS.verbose:
                print("=====================================")
                print(f'Current chunk to generate:\t {i+1}: {chunk}')
                print(f'Reference text:\t {ref_text}')
                print(f'Reference audio:\t {ref_audio}')
                print("=====================================")

            # Run inference
            run_inference(ref_audio, ref_text, chunk, output_wav_path)
            
            # Load generated audio
            generated_audio = AudioSegment.from_wav(output_wav_path)

            # Append to final audio
            final_audio += generated_audio
            
            # Delete the generated audio file chunk
            os.remove(output_wav_path)

        # Save final concatenated audio
        final_audio.export(Path(ARGS.audio_gen_output_folder, experiment_id, f'{gen_text_file_name}.wav'), format="wav")
        if ARGS.verbose:
            print(f'Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{gen_text_file_name}.wav')}')

# 2
# Generate audio per chunk (word, multiple words), conditioned on the fixed reference & last generated audio and text         
def experiment_per_chunk_gen_cond(chunk_size=1):
    experiment_id = f'experiment_per_chunk_size_{chunk_size}_gen_cond'
    
    final_audio = AudioSegment.silent(duration=0)
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = os.listdir(ARGS.text_gen_folder, "r")
    gen_texts = {}
    for gen_text in gen_texts_paths:
        with open(gen_text, "r") as f:
            gen_text_file_name = os.path.basename(gen_text)
            gen_text = f.read().strip()
            gen_text_chunks = get_chunks(gen_text, chunk_size) # divide the text into chunks on space
            gen_texts[gen_text_file_name] = gen_text_chunks
            
    os.makedirs(Path(ARGS.audio_gen_output_folder, experiment_id), exist_ok=True)
    
    original_ref_text = ref_text
    original_ref_audio = ref_audio
    
    for gen_text_file_name, gen_text in tqdm(gen_texts):
        for i, chunk in enumerate(gen_text):
            output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav')

            if ARGS.verbose:
                print("=====================================")
                print(f'Current chunk to generate:\t {i+1}: {chunk}')
                print(f'Reference text:\t {ref_text}')
                print(f'Reference audio:\t {ref_audio}')
                print("=====================================")

            # Run inference
            run_inference(ref_audio, ref_text, chunk, output_wav_path)
            
            # Load generated audio
            generated_audio = AudioSegment.from_wav(output_wav_path)

            # Append to final audio
            final_audio += generated_audio
            
            # Delete the generated audio file chunk
            os.remove(output_wav_path)

            # Update reference text (append new chunk)
            ref_text = original_ref_text + chunk  # Keep original text + add new text
            # concatenate the new audio with the original reference audio
            conc = AudioSegment.from_wav(original_ref_audio) + generated_audio
            conc.export(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"), format="wav")
            ref_audio = Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav") # Keep original audio + add new audio

        # Save final concatenated audio
        final_audio.export(Path(ARGS.audio_gen_output_folder, experiment_id, f'{gen_text_file_name}.wav'), format="wav")
        # Delete the temporary concatenated audio file
        os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"))	
        if ARGS.verbose:
            print(f'Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{gen_text_file_name}.wav')}')

# 3
# Generate audio per chunk (word, multiple words), conditioned on the fixed reference & last generated audio and text with silence            
def experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len=100):
    experiment_id = f'experiment_per_chunk_size_{chunk_size}_gen_cond_with_silence_{silence_len}_ms'
        
    final_audio = AudioSegment.silent(duration=0)
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
    
    # Load all files in the text_gen_folder
    gen_texts_paths = os.listdir(ARGS.text_gen_folder, "r")
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
    
    for gen_text_file_name, gen_text in tqdm(gen_texts):
        for i, chunk in enumerate(gen_text):
            output_wav_path = Path(ARGS.audio_gen_output_folder, experiment_id, f'output_chunk_{i}.wav')

            if ARGS.verbose:
                print("=====================================")
                print(f"Current chunk to generate:\t {i+1}: {chunk}")
                print(f"Reference text:\t {ref_text}")
                print(f"Reference audio:\t {ref_audio}")
                print("=====================================")

            # Run inference
            run_inference(ref_audio, ref_text, chunk, output_wav_path)
            
            # Load generated audio
            generated_audio = AudioSegment.from_wav(output_wav_path)

            # Append to final audio
            final_audio += generated_audio
            
            # Delete the generated audio file chunk
            os.remove(output_wav_path)

            # Update reference text (append new chunk)
            ref_text = original_ref_text + chunk  # Keep original text + add new text
            # concatenate the new audio with the original reference audio
            conc = AudioSegment.from_wav(original_ref_audio) + generated_audio
            conc.export(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"), format="wav")
            ref_audio = Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav") # Keep original audio + add new audio

        # Save final concatenated audio
        final_audio.export(Path(ARGS.audio_gen_output_folder, experiment_id, f'{gen_text_file_name}.wav'), format="wav")
        # Delete the temporary concatenated audio file
        os.remove(Path(ARGS.audio_gen_output_folder, experiment_id, "concatenated_tmp.wav"))	
        if ARGS.verbose:
            print(f'Audio saved as {Path(ARGS.audio_gen_output_folder, experiment_id, f'{gen_text_file_name}.wav')}')

# 4
# TODO DTW experiment
def experiment_dtw():
    pass

if __name__ == "__main__":   
    ARGS = parse_args()
    
    if ARGS.list_experiment:
        print("List of experiments to run:")
        print("\t 0: Generate audio using the model as-is, e.g., full sentence generation, conditioned on the fixed reference.")
        print("\t 1: Generate audio per chunk (word, multiple words), conditioned on the fixed reference.")
        print("\t 2: Generate audio per chunk (word, multiple words), conditioned on the fixed reference & last generated audio.")
        print("\t 3: Generate audio per chunk (word, multiple words), conditioned on the fixed reference & last generated audio with silence.")
        exit(0)
    
    # Load the F5-TTS model
    load_f5_model()
    
    # Run the specified experiment
    # run_inference(ARGS.audio_ref_file, "My wife, on the spur of the moment, managed to give the gentleman a very good dinner." , "My wife, on the spur of the moment, managed to give the gentleman a very good dinner.", "/home/m/output.wav")
    
    if ARGS.experiment is None:
        print("Running all experiments...")
        if ARGS.verbose:
            print("Running experiment 0...")
        experiment_default_per_sentence()
        if ARGS.verbose:
            print("Running experiment 1...")
        experiment_per_chunk(chunk_size=1)
        experiment_per_chunk(chunk_size=2)
        if ARGS.verbose:
            print("Running experiment 2...")
        experiment_per_chunk_gen_cond(chunk_size=1)
        experiment_per_chunk_gen_cond(chunk_size=2)
        if ARGS.verbose:
            print("Running experiment 3...")
        experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len=100)
        experiment_per_chunk_gen_cond_with_silence(chunk_size=2, silence_len=100)
        experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len=300)
        experiment_per_chunk_gen_cond_with_silence(chunk_size=2, silence_len=300)
        experiment_per_chunk_gen_cond_with_silence(chunk_size=1, silence_len=500)
        experiment_per_chunk_gen_cond_with_silence(chunk_size=2, silence_len=500)
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
        else:
            print("Invalid experiment ID. Please select a valid experiment ID.")
            exit(0)