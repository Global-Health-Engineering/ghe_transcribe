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
    ArgparseGenerator,
)

__config_transcribe__ = {
    "default": {"huggingface_token": None,
                "trim": None,
                "device": "auto",
                "cpu_threads": None,
                "whisper_model": "large-v3-turbo",
                "device_index": None,
                "compute_type": "float32",
                "beam_size": 5,
                "temperature": 0.0,
                "word_timestamps": None,
                "vad_filter": False,
                "vad_parameters": {'min_silence_duration_ms': 2000},
                "pyannote_model": "pyannote/speaker-diarization-3.1",
                "num_speakers": None,
                "min_speakers": None,
                "max_speakers": None,
                "save_output": True,
                "info": True},
    "choices": {"device": ["cuda", "mps", "cpu"],
                "whisper_model": ["tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium",
                                  "large-v1", "large-v2", "large-v3", "large", "distil-large-v2", "distil-medium.en",
                                  "distil-small.en", "distil-large-v3", "large-v3-turbo", "turbo"],
                "compute_type": ["float32", "float16", "int8"]},
    }

@timing
def transcribe(audio_file: str,
               huggingface_token: str = __config_transcribe__.get("default", {}).get("huggingface_token"),
               trim: float = __config_transcribe__.get("default", {}).get("trim"),
               device: str = __config_transcribe__.get("default", {}).get("device"),
               cpu_threads: int = __config_transcribe__.get("default", {}).get("cpu_threads"),
               whisper_model: str = __config_transcribe__.get("default", {}).get("whisper_model"),
               device_index: int = __config_transcribe__.get("default", {}).get("device_index"),
               compute_type: str = __config_transcribe__.get("default", {}).get("compute_type"),
               beam_size: int = __config_transcribe__.get("default", {}).get("beam_size"),
               temperature: float = __config_transcribe__.get("default", {}).get("temperature"),
               word_timestamps: bool = __config_transcribe__.get("default", {}).get("word_timestamps"),
               vad_filter: bool = __config_transcribe__.get("default", {}).get("vad_filter"),
               vad_parameters: dict = __config_transcribe__.get("default", {}).get("vad_parameters"),
               pyannote_model: str = __config_transcribe__.get("default", {}).get("pyannote_model"),
               num_speakers: int = __config_transcribe__.get("default", {}).get("num_speakers"),
               min_speakers: int = __config_transcribe__.get("default", {}).get("min_speakers"),
               max_speakers: int = __config_transcribe__.get("default", {}).get("max_speakers"),
               save_output: bool = __config_transcribe__.get("default", {}).get("save_output"),
               info: bool = __config_transcribe__.get("default", {}).get("info")
               ):
    """Transcribe and diarize an audio file.

    Args:
        audio_file (str): Path to the audio file.
        trim (float, optional): Trim the audio file from 0 to the specified number of seconds.
        device (str, optional): Device to use.
        cpu_threads (int, optional): Number of CPU threads to use.
        whisper_model (str, optional): Faster Whisper, model to use.
        device_index (int, optional): Faster Whisper, index of the device to use.
        compute_type (str, optional): Faster Whisper, compute type.
        beam_size (int, optional): Faster Whisper, beam size for decoding.
        temperature (float, optional): Faster Whisper, sampling temperature.
        word_timestamps (bool, optional): Faster Whisper, enable word timestamps in the output.
        vad_filter (bool, optional): Faster Whisper, enable voice activity detection.
        vad_parameters (dict, optional): Faster Whisper, parameters for voice activity detection.
        pyannote_model (str, optional): pyannote.audio, speaker diarization model to use.
        num_speakers (int, optional): pyannote.audio, number of speakers.
        min_speakers (int, optional): pyannote.audio, minimum number of speakers.
        max_speakers (int, optional): pyannote.audio, maximum number of speakers.
        save_output (bool, optional): Save output to .csv and .md files.
        info (bool, optional): Print detected language information.

    Returns:
        list: A list of dictionaries, where each dictionary represents a transcribed segment
              with speaker information and timestamps.
    """
    # Convert audio file to .wav
    audio_file = to_wav(audio_file)

    if trim is not None:
        audio_file = snip_audio(
            audio_file,
            os.path.splitext(audio_file)[0] + "_snippet" + os.path.splitext(audio_file)[1],
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
    args = ArgparseGenerator(transcribe, __config_transcribe__).parse_args()
    transcribe(**vars(args))