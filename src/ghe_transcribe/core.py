import logging
import os
from enum import Enum
from tempfile import TemporaryDirectory

import yaml
from faster_whisper import WhisperModel
from torch import device as to_torch_device
from torch import set_num_threads
from torch.backends.mps import is_available as mps_is_available
from torch.cuda import is_available as cuda_is_available
from typer import Argument, Option, Typer

from ghe_transcribe.exceptions import (
    DiarizationError,
    ModelInitializationError,
)
from ghe_transcribe.utils import (
    diarize_text,
    snip_audio,
    timing,
    to_srt,
    to_txt,
    to_wav,
    to_whisper_format,
)
from pyannote.audio import Pipeline


class DeviceChoice(str, Enum):
    auto = "auto"
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
    "trim": None,
    "device": "auto",
    "cpu_threads": None,
    "whisper_model": "large-v3-turbo",
    "device_index": 0,
    "compute_type": "float32",
    "beam_size": 5,
    "temperature": 0.0,
    "word_timestamps": None,
    "vad_filter": False,
    "min_silence_duration_ms": 2000,
    "num_speakers": None,
    "min_speakers": None,
    "max_speakers": None,
    "save_output": True,
    "info": True,
}

app = Typer(help="Transcribe and diarize an audio file.")

# Set up simple logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def transcribe_core(
    file: str,
    trim: float | None = None,
    device: str | None = None,
    cpu_threads: int | None = None,
    whisper_model: str | None = None,
    device_index: int | None = None,
    compute_type: str | None = None,
    beam_size: int | None = None,
    temperature: float | None = None,
    word_timestamps: bool | None = None,
    vad_filter: bool | None = None,
    min_silence_duration_ms: int | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    save_output: bool | None = None,
    info: bool | None = None,
):
    """Transcribe and diarize an audio file.\n    \n    Args:\n        file: Path to the audio file to transcribe\n        trim: Trim audio to specified seconds (from start)\n        device: Device to use for inference (auto, cuda, mps, cpu)\n        cpu_threads: Number of CPU threads for inference\n        whisper_model: Whisper model size to use\n        device_index: Device index for multi-GPU systems\n        compute_type: Computation precision (float32, float16, int8)\n        beam_size: Beam search width for decoding\n        temperature: Sampling temperature for generation\n        word_timestamps: Enable word-level timestamps\n        vad_filter: Enable voice activity detection filter\n        min_silence_duration_ms: Minimum silence duration for VAD\n        num_speakers: Exact number of speakers (overrides min/max)\n        min_speakers: Minimum number of speakers for diarization\n        max_speakers: Maximum number of speakers for diarization\n        save_output: Save transcription to TXT and SRT files\n        info: Print detected language information\n        \n    Returns:\n        List of tuples containing (segment, speaker, text)\n        \n    Raises:\n        ModelInitializationError: If model initialization fails\n        DiarizationError: If speaker diarization fails\n        AudioConversionError: If audio conversion fails\n"""
    # Apply defaults
    trim = trim if trim is not None else transcribe_config.get("trim")
    device = device if device is not None else transcribe_config.get("device")
    cpu_threads = (
        cpu_threads if cpu_threads is not None else transcribe_config.get("cpu_threads")
    )
    whisper_model = (
        whisper_model
        if whisper_model is not None
        else transcribe_config.get("whisper_model")
    )
    device_index = (
        device_index
        if device_index is not None
        else transcribe_config.get("device_index")
    )
    compute_type = (
        compute_type
        if compute_type is not None
        else transcribe_config.get("compute_type")
    )
    beam_size = (
        beam_size if beam_size is not None else transcribe_config.get("beam_size")
    )
    temperature = (
        temperature if temperature is not None else transcribe_config.get("temperature")
    )
    word_timestamps = (
        word_timestamps
        if word_timestamps is not None
        else transcribe_config.get("word_timestamps")
    )
    vad_filter = (
        vad_filter if vad_filter is not None else transcribe_config.get("vad_filter")
    )
    min_silence_duration_ms = (
        min_silence_duration_ms
        if min_silence_duration_ms is not None
        else transcribe_config.get("min_silence_duration_ms")
    )
    num_speakers = (
        num_speakers
        if num_speakers is not None
        else transcribe_config.get("num_speakers")
    )
    min_speakers = (
        min_speakers
        if min_speakers is not None
        else transcribe_config.get("min_speakers")
    )
    max_speakers = (
        max_speakers
        if max_speakers is not None
        else transcribe_config.get("max_speakers")
    )
    save_output = (
        save_output if save_output is not None else transcribe_config.get("save_output")
    )
    info = info if info is not None else transcribe_config.get("info")

    # Relative path helper
    root_path = os.path.abspath(os.path.dirname(__file__)).replace(
        "/src/ghe_transcribe", ""
    )

    # Convert audio file to .wav
    file = to_wav(file)

    if trim is not None:
        file = snip_audio(
            file,
            os.path.splitext(file)[0]
            + f"_{int(trim)}_seconds"
            + os.path.splitext(file)[1],
            0.0,
            trim,
        )

    # Device
    if device == "auto":
        device = (
            "cuda" if cuda_is_available() else "mps" if mps_is_available() else "cpu"
        )
        logger.info(f"Using device: {device}")
    try:
        torch_device = to_torch_device(device)
    except Exception as e:
        logger.error(f"Device Error: {e}")
        raise ModelInitializationError(
            f"Failed to initialize device {device}: {e}"
        ) from e

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
        logger.error(f"WhisperModel Device Error: {e}")
        raise ModelInitializationError(
            f"Failed to initialize Whisper model: {e}"
        ) from e

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
        whisper_transcribe_kwargs["vad_parameters"] = {
            "min_silence_duration_ms": min_silence_duration_ms
        }

    segments, info = model.transcribe(file, **whisper_transcribe_kwargs)
    generated_segments = list(segments)

    # Speaker Diarization: pyannote.audio

    # Create a dictionary of keyword arguments
    pyannote_kwargs = {}

    _num_speakers = (
        num_speakers if isinstance(num_speakers, int) or num_speakers is None else None
    )
    _min_speakers = (
        min_speakers if isinstance(min_speakers, int) or min_speakers is None else None
    )
    _max_speakers = (
        max_speakers if isinstance(max_speakers, int) or max_speakers is None else None
    )

    if _num_speakers is not None:
        pyannote_kwargs["num_speakers"] = _num_speakers
    if _min_speakers is not None:
        pyannote_kwargs["min_speakers"] = _min_speakers
    if _max_speakers is not None:
        pyannote_kwargs["max_speakers"] = _max_speakers

    try:
        pyannote_config_name = "pyannote_config.yaml"

        with open(
            os.path.join(root_path, "pyannote", pyannote_config_name)
        ) as yaml_file:
            pyannote_config = yaml.safe_load(yaml_file)

        pyannote_config["pipeline"]["params"]["embedding"] = os.path.join(
            root_path, *pyannote_config["pipeline"]["params"]["embedding"].split("/")
        )
        pyannote_config["pipeline"]["params"]["segmentation"] = os.path.join(
            root_path, *pyannote_config["pipeline"]["params"]["segmentation"].split("/")
        )

        tmpdir = TemporaryDirectory("ghe_transcribe_temp")
        with open(os.path.join(tmpdir.name, pyannote_config_name), "w") as yaml_file:
            yaml.safe_dump(pyannote_config, yaml_file)

        diarization_result = Pipeline.from_pretrained(
            os.path.join(tmpdir.name, pyannote_config_name)
        ).to(torch_device)(file, **pyannote_kwargs)

    except Exception as e:
        logger.error(f"Diarization Error: {e}")
        raise DiarizationError(f"Failed to perform speaker diarization: {e}") from e

    # Text alignment
    text = diarize_text(to_whisper_format(generated_segments), diarization_result)

    # Extract audio_file name
    output_file_name = "output/" + os.path.splitext(os.path.basename(file))[0]

    if save_output:
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        txt = to_txt(text)
        with open(output_file_name + ".txt", "w") as f:
            f.write(txt)
            logger.info(f"Output saved to {output_file_name}.txt")
        srt = to_srt(text)
        with open(output_file_name + ".srt", "w") as f:
            f.write(srt)
            logger.info(f"Output saved to {output_file_name}.srt")

    if info:
        logger.info(
            f"Detected language {info.language} with probability {info.language_probability}"
        )
        return text

    return text


@app.command()
@timing
def transcribe(
    file: str = Argument(..., help="Path to the audio file."),
    trim: float | None = Option(
        transcribe_config.get("trim"),
        help="Trim the audio file from 0 to the specified number of seconds.",
    ),
    device: DeviceChoice | None = Option(
        transcribe_config.get("device"), help="Device to use."
    ),
    cpu_threads: int | None = Option(
        transcribe_config.get("cpu_threads"), help="Number of CPU threads to use."
    ),
    whisper_model: WhisperModelChoice | None = Option(
        transcribe_config.get("whisper_model"), help="Faster Whisper, model to use."
    ),
    device_index: int | None = Option(
        transcribe_config.get("device_index"),
        help="Faster Whisper, index of the device to use.",
    ),
    compute_type: ComputeTypeChoice | None = Option(
        transcribe_config.get("compute_type"), help="Faster Whisper, compute type."
    ),
    beam_size: int | None = Option(
        transcribe_config.get("beam_size"),
        help="Faster Whisper, beam size for decoding.",
    ),
    temperature: float | None = Option(
        transcribe_config.get("temperature"),
        help="Faster Whisper, sampling temperature.",
    ),
    word_timestamps: bool | None = Option(
        transcribe_config.get("word_timestamps"),
        help="Faster Whisper, enable word timestamps in the output.",
    ),
    vad_filter: bool | None = Option(
        transcribe_config.get("vad_filter"),
        help="Faster Whisper, enable voice activity detection.",
    ),
    min_silence_duration_ms: int | None = Option(
        transcribe_config.get("min_silence_duration_ms"),
        help="Faster Whisper, minimum silence duration detected by VAD in milliseconds.",
    ),
    num_speakers: int | None = Option(
        transcribe_config.get("num_speakers"),
        help="pyannote.audio, number of speakers.",
    ),
    min_speakers: int | None = Option(
        transcribe_config.get("min_speakers"),
        help="pyannote.audio, minimum number of speakers.",
    ),
    max_speakers: int | None = Option(
        transcribe_config.get("max_speakers"),
        help="pyannote.audio, maximum number of speakers.",
    ),
    save_output: bool | None = Option(
        transcribe_config.get("save_output"), help="Save output to .txt and .srt files."
    ),
    info: bool | None = Option(
        transcribe_config.get("info"), help="Print detected language information."
    ),
):
    """CLI wrapper for transcribe_core function."""
    return transcribe_core(
        file=file,
        trim=trim,
        device=device,
        cpu_threads=cpu_threads,
        whisper_model=whisper_model,
        device_index=device_index,
        compute_type=compute_type,
        beam_size=beam_size,
        temperature=temperature,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        min_silence_duration_ms=min_silence_duration_ms,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        save_output=save_output,
        info=info,
    )


if __name__ == "__main__":
    app()
