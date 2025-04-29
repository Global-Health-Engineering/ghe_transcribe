import os
import json
from typing import Optional
from enum import Enum

from typer import Typer, Argument, Option
from huggingface_hub import login
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from torch import device as to_torch_device
from torch import set_num_threads
from torch.backends.mps import is_available as mps_is_available
from torch.cuda import is_available as cuda_is_available

from ghe_transcribe.utils import (
    diarize_text,
    timing,
    to_csv,
    to_md,
    to_wav,
    to_whisper_format,
    snip_audio,
)


class DeviceChoice(str, Enum):
    cuda = "cuda"
    mps = "mps"
    cpu = "cpu"

class ComputeTypeChoice(str, Enum):
    float32 = "float32"
    float16 = "float16"
    int8 = "int8"

class WhisperModelChoice(str, Enum):
    tiny_en = "tiny.en"
    tiny = "tiny"
    base_en = "base.en"
    base = "base"
    small_en = "small.en"
    small = "small"
    medium_en = "medium.en"
    medium = "medium"
    large_v1 = "large-v1"
    large_v2 = "large-v2"
    large_v3 = "large-v3"
    large = "large"
    distil_large_v2 = "distil-large-v2"
    distil_medium_en = "distil-medium.en"
    distil_small_en = "distil-small.en"
    distil_large_v3 = "distil-large-v3"
    large_v3_turbo = "large-v3-turbo"
    turbo = "turbo"

transcribe_config = {
    "huggingface_token": None,
    "trim": None,
    "device": None,
    "cpu_threads": 0,
    "whisper_model": "large-v3-turbo",
    "device_index": None,
    "compute_type": "float32",
    "beam_size": 5,
    "temperature": 0.0,
    "word_timestamps": None,
    "vad_filter": False,
    "min_silence_duration_ms": 2000,
    "pyannote_model": "pyannote/speaker-diarization-3.1",
    "num_speakers": None,
    "min_speakers": None,
    "max_speakers": None,
    "save_output": True,
    "info": True,
}

app = Typer(help="Transcribe and diarize an audio file.")

@timing
@app.command()
def transcribe(
    file: str = Argument(..., help="Path to the audio file."),
    huggingface_token: Optional[str] = Option(transcribe_config.get("huggingface_token"), help="Hugging Face token for authentication."),
    trim: Optional[float] = Option(transcribe_config.get("trim"), help="Trim the audio file from 0 to the specified number of seconds."),
    device: Optional[DeviceChoice] = Option(transcribe_config.get("device"), help="Device to use."),
    cpu_threads: Optional[int] = Option(transcribe_config.get("cpu_threads"), help="Number of CPU threads to use."),
    whisper_model: Optional[WhisperModelChoice] = Option(transcribe_config.get("whisper_model"), help="Faster Whisper, model to use."),
    device_index: Optional[int] = Option(transcribe_config.get("device_index"), help="Faster Whisper, index of the device to use."),
    compute_type: Optional[ComputeTypeChoice] = Option(transcribe_config.get("compute_type"), help="Faster Whisper, compute type."),
    beam_size: Optional[int] = Option(transcribe_config.get("beam_size"), help="Faster Whisper, beam size for decoding."),
    temperature: Optional[float] = Option(transcribe_config.get("temperature"), help="Faster Whisper, sampling temperature."),
    word_timestamps: Optional[bool] = Option(transcribe_config.get("word_timestamps"), help="Faster Whisper, enable word timestamps in the output."),
    vad_filter: Optional[bool] = Option(transcribe_config.get("vad_filter"), help="Faster Whisper, enable voice activity detection."),
    min_silence_duration_ms: Optional[int] = Option(transcribe_config.get("min_silence_duration_ms"), help="Faster Whisper, minimum silence duration detected by VAD in milliseconds."),
    pyannote_model: Optional[str] = Option(transcribe_config.get("pyannote_model"), help="pyannote.audio, speaker diarization model to use."),
    num_speakers: Optional[int] = Option(transcribe_config.get("num_speakers"), help="pyannote.audio, number of speakers."),
    min_speakers: Optional[int] = Option(transcribe_config.get("min_speakers"), help="pyannote.audio, minimum number of speakers."),
    max_speakers: Optional[int] = Option(transcribe_config.get("max_speakers"), help="pyannote.audio, maximum number of speakers."),
    save_output: Optional[bool] = Option(transcribe_config.get("save_output"), help="Save output to .csv and .md files."),
    info: Optional[bool] = Option(transcribe_config.get("info"), help="Print detected language information."),
):
    """Transcribe and diarize an audio file."""
    try:
        if huggingface_token:
            login(token=huggingface_token)
        else:
            with open("config.json", "r") as f:
                config = json.load(f)

            login(token=config["huggingface_token"])
    except Exception as e:
        print(f"Error: {e} Please provide a valid token through the --huggingface_token argument or configure the hugginface_token in config.json.")
        return None

    # Convert audio file to .wav
    file = to_wav(file)

    if trim is not None:
        file = snip_audio(
            file,
            os.path.splitext(file)[0] + "_snippet" + os.path.splitext(file)[1],
            0.0,
            trim,
        )

    # Device
    if device == "auto":
        device = (
            "cuda" if cuda_is_available() else "mps" if mps_is_available() else "cpu"
        )
        print(f"Using device: {device}")
    try:
        torch_device = to_torch_device(device)
    except Exception as e:
        print(f"Device Error: {e}")
        return

    # Automatic Speech Recognition (ASR): faster-whisper
    # https://github.com/SYSTRAN/faster-whisper/blob/1383fd4d3725bdf59c95d8834c629f45c6974981/faster_whisper/transcribe.py#L586

    # Create a dictionary of keyword arguments
    whisper_model_kwargs = {}
    if cpu_threads is not None:
        whisper_model_kwargs["cpu_threads"] = cpu_threads
        set_num_threads(cpu_threads)

    try:
        match device:
            case "mps" | "cpu":
                model = WhisperModel(
                    model_size_or_path=whisper_model,
                    device="cpu",
                    device_index=device_index,
                    compute_type=compute_type,
                    num_workers=1,
                    download_root=None,
                    local_files_only=False,
                    files=None,
                    **whisper_model_kwargs,
                )
            case _:
                model = WhisperModel(
                    model_size_or_path=whisper_model,
                    device=torch_device,
                    device_index=device_index,
                    compute_type=compute_type,
                    num_workers=1,
                    download_root=None,
                    local_files_only=False,
                    files=None,
                    **whisper_model_kwargs,
                )
    except Exception as e:
        print(f"WhisperModel Device Error: {e}")

    # https://github.com/SYSTRAN/faster-whisper/blob/1383fd4d3725bdf59c95d8834c629f45c6974981/faster_whisper/transcribe.py#L255

    # Create a dictionary of keyword arguments
    whisper_transcribe_kwargs = {}
    if beam_size is not None:
        whisper_transcribe_kwargs["beam_size"] = beam_size
    if temperature is not None:
        whisper_transcribe_kwargs["temperature"] = temperature
    if word_timestamps is not None:
        whisper_transcribe_kwargs["word_timestamps"] = word_timestamps
    if vad_filter is not None:
        whisper_transcribe_kwargs["vad_filter"] = vad_filter
    if min_silence_duration_ms is not None:
        whisper_transcribe_kwargs["vad_parameters"] = {"min_silence_duration_ms": min_silence_duration_ms}

    segments, info = model.transcribe(file, **whisper_transcribe_kwargs)
    generated_segments = list(segments)

    # Speaker Diarization: pyannote.audio

    # Create a dictionary of keyword arguments
    pyannote_kwargs = {}
    if num_speakers is not None:
        pyannote_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        pyannote_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        pyannote_kwargs["max_speakers"] = max_speakers

    try:
        diarization_result = Pipeline.from_pretrained(pyannote_model).to(torch_device)(
            file, **pyannote_kwargs
        )
    except Exception as e:
        print(f"Diarization Error: {e}")

    # Text alignment
    text = diarize_text(to_whisper_format(generated_segments), diarization_result)

    # Extract audio_file name
    output_file_name = "output/" + os.path.splitext(os.path.basename(file))[0]

    if save_output:
        csv = to_csv(text)
        with open(output_file_name + ".csv", "w") as f:
            f.write("\n".join(csv))
            print(f"Output saved to {output_file_name}.csv")
        md = to_md(text)
        with open(output_file_name + ".md", "w") as f:
            f.write("\n".join(md))
            print(f"Output saved to {output_file_name}.md")

    if info:
        print(
            f"Detected language {info.language} with probability {info.language_probability}"
        )
        return text

    return text


if __name__ == "__main__":
    app()