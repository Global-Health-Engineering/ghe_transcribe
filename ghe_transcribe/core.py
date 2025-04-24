import argparse
import os

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

__TRANSCRIBE_CONFIG_DEFAULT__ = {
    "snip": None,
    "device": None,
    "whisper_model": "large-v3-turbo",
    "device_index": 0,
    "compute_type": "float32",
    "cpu_threads": None,
    "beam_size": 5,
    "temperature": 0.0,
    "word_timestamps": None,
    "vad_filter": False,
    "vad_parameters": dict(min_silence_duration_ms=2000),
    "pyannote_model": "pyannote/speaker-diarization-3.1",
    "num_speakers": None,
    "min_speakers": None,
    "max_speakers": None,
    "save_output": True,
    "info": True,
}

@timing
def transcribe(audio_file,
               snip = __TRANSCRIBE_CONFIG_DEFAULT__.get("snip"),
               device = __TRANSCRIBE_CONFIG_DEFAULT__.get("device"),
               whisper_model = __TRANSCRIBE_CONFIG_DEFAULT__.get("whisper_model"),
               device_index = __TRANSCRIBE_CONFIG_DEFAULT__.get("device_index"),
               compute_type = __TRANSCRIBE_CONFIG_DEFAULT__.get("compute_type"),
               cpu_threads = __TRANSCRIBE_CONFIG_DEFAULT__.get("cpu_threads"),
               beam_size = __TRANSCRIBE_CONFIG_DEFAULT__.get("beam_size"),
               temperature = __TRANSCRIBE_CONFIG_DEFAULT__.get("temperature"),
               word_timestamps = __TRANSCRIBE_CONFIG_DEFAULT__.get("word_timestamps"),
               vad_filter = __TRANSCRIBE_CONFIG_DEFAULT__.get("vad_filter"),
               vad_parameters = __TRANSCRIBE_CONFIG_DEFAULT__.get("vad_parameters"),
               pyannote_model = __TRANSCRIBE_CONFIG_DEFAULT__.get("pyannote_model"),
               num_speakers = __TRANSCRIBE_CONFIG_DEFAULT__.get("num_speakers"),
               min_speakers = __TRANSCRIBE_CONFIG_DEFAULT__.get("min_speakers"),
               max_speakers = __TRANSCRIBE_CONFIG_DEFAULT__.get("max_speakers"),
               save_output = __TRANSCRIBE_CONFIG_DEFAULT__.get("save_output"),
               info = __TRANSCRIBE_CONFIG_DEFAULT__.get("info")
               ):
    # Convert audio file to .wav
    audio_file = to_wav(audio_file)

    if snip is not None:
        audio_file = snip_audio(
            audio_file,
            os.path.splitext(audio_file)[0] + "_snippet" + os.path.splitext(audio_file)[1],
            0.0,
            snip,
        )

    # Device
    if device is None:
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
    if vad_parameters is not None:
        whisper_transcribe_kwargs["vad_parameters"] = vad_parameters

    segments, info = model.transcribe(audio_file, **whisper_transcribe_kwargs)
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
            audio_file, **pyannote_kwargs
        )
    except Exception as e:
        print(f"Diarization Error: {e}")

    # Text alignment
    text = diarize_text(to_whisper_format(generated_segments), diarization_result)

    # Extract audio_file name
    output_file_name = "output/" + os.path.splitext(os.path.basename(audio_file))[0]

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
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize an audio file."
    )
    parser.add_argument("audio_file", type=str, help="Path to the audio file.")
    parser.add_argument("snip", type=float, default=__TRANSCRIBE_CONFIG_DEFAULT__.get("snip"), help="Snip a number of seconds of the audio file.")
    parser.add_argument(
        "--device",
        type=str,
        default=__TRANSCRIBE_CONFIG_DEFAULT__.get("device"),
        choices=["cuda", "mps", "cpu"],
        help="Device to use (cuda, mps, or cpu). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default=__TRANSCRIBE_CONFIG_DEFAULT__.get("whisper_model"),
        choices=[
            "tiny.en",
            "tiny",
            "base.en",
            "base",
            "small.en",
            "small",
            "medium.en",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "large",
            "distil-large-v2",
            "distil-medium.en",
            "distil-small.en",
            "distil-large-v3",
            "large-v3-turbo",
            "turbo",
        ],
        help="Faster Whisper model to use.",
    )
    parser.add_argument(
        "--device_index", type=int, default=__TRANSCRIBE_CONFIG_DEFAULT__.get("device_index"), help="Index of the device to use."
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default=__TRANSCRIBE_CONFIG_DEFAULT__.get("compute_type"),
        choices=["float32", "float16", "int8"],
        help="Compute type for Whisper model.",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=__TRANSCRIBE_CONFIG_DEFAULT__.get("cpu_threads"),
        help="Number of CPU threads to use for Whisper.",
    )
    parser.add_argument(
        "--beam_size", type=int, default=__TRANSCRIBE_CONFIG_DEFAULT__.get("beam_size"), help="Beam size for Whisper decoding."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=__TRANSCRIBE_CONFIG_DEFAULT__.get("temperature"),
        help="Temperature for Whisper sampling.",
    )
    parser.add_argument(
        "--word_timestamps",
        type=bool,
        default=__TRANSCRIBE_CONFIG_DEFAULT__.get("word_timestamps"),
        help="Enable word timestamps in Whisper output.",
    )
    parser.add_argument(
        "--vad_filter",
        type=bool,
        default=__TRANSCRIBE_CONFIG_DEFAULT__.get("vad_filter"),
        help="Enable voice activity detection in Whisper.",
    )
    parser.add_argument(
        "--min_silence_duration_ms",
        type=int,
        default=__TRANSCRIBE_CONFIG_DEFAULT__.get("min_silence_duration_ms"),
        help="Minimum silence duration for VAD (ms).",
    )
    parser.add_argument(
        "--pyannote_model",
        type=str,
        default=__TRANSCRIBE_CONFIG_DEFAULT__.get("pyannote_model"),
        help="Pyannote speaker diarization model to use.",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=__TRANSCRIBE_CONFIG_DEFAULT__.get("num_speakers"),
        help="Number of speakers for diarization (if known).",
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=__TRANSCRIBE_CONFIG_DEFAULT__.get("min_speakers"),
        help="Minimum number of speakers for diarization.",
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=__TRANSCRIBE_CONFIG_DEFAULT__.get("max_speakers"),
        help="Maximum number of speakers for diarization.",
    )
    parser.add_argument(
        "--save_output",
        type=bool,
        default=__TRANSCRIBE_CONFIG_DEFAULT__.get("save_output"),
        help="Save output to .csv and .md files.",
    )
    parser.add_argument(
        "--info", type=bool, default=__TRANSCRIBE_CONFIG_DEFAULT__.get("info"), help="Print detected language information."
    )

    args = parser.parse_args()

    transcribe(
        audio_file=args.audio_file,
        snip=args.snip,
        device=args.device,
        whisper_model=args.whisper_model,
        device_index=args.device_index,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        beam_size=args.beam_size,
        temperature=args.temperature,
        word_timestamps=args.word_timestamps,
        vad_filter=args.vad_filter,
        vad_parameters=dict(min_silence_duration_ms=args.min_silence_duration_ms),
        pyannote_model=args.pyannote_model,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        save_output=args.save_output,
        info=args.info,
    )
