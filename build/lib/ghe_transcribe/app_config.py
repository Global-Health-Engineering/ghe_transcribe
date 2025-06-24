import os
import argparse
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE

"""
WhisperX default args
"""


args = {'model': 'small', 'model_cache_only': False, 'model_dir': None, 'device': 'cpu', 'device_index': 0, 'batch_size': 8, 'compute_type': 'float16', 'output_dir': '.', 'output_format': 'all', 'verbose': True, 'task': 'transcribe', 'language': None, 'align_model': None, 'interpolate_method': 'nearest', 'no_align': False, 'return_char_alignments': False, 'vad_method': 'pyannote', 'vad_onset': 0.5, 'vad_offset': 0.363, 'chunk_size': 30, 'diarize': False, 'min_speakers': None, 'max_speakers': None, 'temperature': 0, 'best_of': 5, 'beam_size': 5, 'patience': 1.0, 'length_penalty': 1.0, 'suppress_tokens': '-1', 'suppress_numerals': False, 'initial_prompt': None, 'condition_on_previous_text': False, 'fp16': True, 'temperature_increment_on_fallback': 0.2, 'compression_ratio_threshold': 2.4, 'logprob_threshold': -1.0, 'no_speech_threshold': 0.6, 'max_line_width': None, 'max_line_count': None, 'highlight_words': False, 'segment_resolution': 'sentence', 'threads': 0, 'hf_token': None, 'print_progress': False}
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


"""
Better args for whisperX
"""


root_path = os.path.abspath(os.path.dirname(__file__))

# To get the parent directory:
root_path = os.path.dirname(root_path)

model = {
    "default": "large-v3-turbo",
    "tiny.en": "tiny.en",
    "tiny": "tiny",
    "base.en": "base.en",
    "base": "base",
    "small.en": "small.en",
    "small": "small",
    "medium.en": "medium.en",
    "medium": "medium",
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "large": "large",
    "distil-large-v2": "distil-large-v2",
    "distil-medium.en": "distil-medium.en",
    "distil-small.en": "distil-small.en",
    "distil-large-v3": "distil-large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "turbo": "turbo",
}

device = {
    "default": "cuda" if os.getenv("TORCH_CUDA_AVAILABLE", "False").lower() == "true" else "cpu",
    "cuda": "cuda",
    "cpu": "cpu",
}

compute_type = {
    "default": "float32",
    "float16": "float16",
    "float32": "float32",
    "int8": "int8",
}

save_output = True

info = True
temperature = 0
beam_size = 5

initial_prompt = None # optional to include disfluencies, https://github.com/kaixxx/noScribe/blob/main/prompt.yml
condition_on_previous_text = False # May improve results but increases hallucinations
highlight_words = False # underline each word as it is spoken in srt and vtt

# REPLACE default args
args["model"] = model.get("default")
args["device"] = device.get("default")
args["compute_type"] = compute_type.get("default")
args["output_format"] = output_format.get("default")
args["language"] = language.get("default")
args["model_dir"] = model_dir
args["output_dir"] = output_dir
args["batch_size"] = batch_size
args["verbose"] = verbose
args["vad_onset"] = vad_onset
args["vad_offset"] = vad_offset
args["chunk_size"] = chunck_size
args["temperature"] = temperature
args["best_of"] = best_of
args["beam_size"] = beam_size
args["initial_prompt"] = initial_prompt
args["condition_on_previous_text"] = condition_on_previous_text
args["highlight_words"] = highlight_words
args["print_progress"] = print_progress
