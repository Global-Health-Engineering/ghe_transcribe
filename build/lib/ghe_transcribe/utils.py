import logging
import os
from datetime import timedelta
from functools import wraps
from time import time

import av
from torchaudio import load, save
from torchaudio.transforms import Resample

from ghe_transcribe.exceptions import AudioConversionError
from pyannote.core import Segment

logger = logging.getLogger(__name__)


def timing(func):
    """Decorator to measure and log function execution time."""

    @wraps(func)
    def wrap(*args, **kw):
        ts = time()
        result = func(*args, **kw)
        te = time()
        logger.info(f"func:{func.__name__!r} took: {te - ts:2.4f} sec")
        return result

    return wrap


# CREDIT: https://github.com/yinruiqing/pyannote-whisper


def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res["segments"]:
        start = item["start"]
        end = item["end"]
        text = item["text"]
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts


def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text


def merge_cache(text_cache):
    sentence = "".join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    # Transform SPEAKER_XX to SXX
    if isinstance(spk, str) and spk.startswith("SPEAKER_"):
        spk = "S" + spk.split("_")[1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence


__PUNC_SENT_END__ = [".", "?", "!"]


def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk

        elif text and len(text) > 0 and text[-1] in __PUNC_SENT_END__:
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text


def diarize_text(transcribe_res, diarization_result):
    """Combine transcription results with speaker diarization.

    Args:
        transcribe_res: Transcription results in Whisper format
        diarization_result: Pyannote diarization results

    Returns:
        List of tuples containing (segment, speaker, text)
    """
    timestamp_texts = get_text_with_timestamp(transcribe_res)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    result = merge_sentence(spk_text)
    return result


def to_txt(result):
    """Convert transcription results to TXT format.

    Args:
        result: List of (segment, speaker, text) tuples

    Returns:
        TXT formatted string with format: SXX: [HH:MM:SS] text
    """
    txt = []
    for seg, spk, sentence in result:
        # Format time as [HH:MM:SS]
        time = format_time_to_srt(seg.start).split(",")[0]  # Remove milliseconds
        line = f"{spk}: [{time}] {sentence}"
        txt.append(line.strip())
    return "\n".join(txt)


def to_csv(result, semicolon=False):
    """Convert transcription results to CSV format.

    Args:
        result: List of (segment, speaker, text) tuples
        semicolon: Use semicolon as separator instead of comma

    Returns:
        CSV formatted string
    """
    csv = []
    if semicolon:
        sep = ";"
    else:
        sep = ","
    start_line = "start" + sep + "end" + sep + "speaker" + sep + "sentence"
    csv.append(start_line)
    for seg, spk, sentence in result:
        if not semicolon:
            sentence = sentence.replace(",", ";")
        line = f"{format_time_to_srt(seg.start)}{sep}{format_time_to_srt(seg.end)}{sep}{spk}{sep}{sentence}"
        csv.append(line.strip())
    return "\n".join(csv)


def to_md(result):
    md = []
    previous_spk = None
    for seg, spk, sentence in result:
        if spk != previous_spk:
            md.append(f"\n{spk}")
            md.append(f"({format_time_to_iso8601(seg.start)}){sentence}".strip())
            previous_spk = spk
        else:
            md.append(f"({format_time_to_iso8601(seg.start)}){sentence}".strip())
    return "\n".join(md)


def to_srt(result):
    """Convert transcription results to SRT subtitle format.

    Args:
        result: List of (segment, speaker, text) tuples

    Returns:
        SRT formatted string
    """
    srt = []
    counter = 1
    for seg, spk, sentence in result:
        start_time = format_time_to_srt(seg.start)
        end_time = format_time_to_srt(seg.end)
        srt.append(str(counter))
        srt.append(f"{start_time} --> {end_time}")
        srt.append(f"{spk}:{sentence}")
        srt.append("")  # Add an empty line to separate subtitles
        counter += 1
    return "\n".join(srt)


def format_time_to_iso8601(seconds_float: float) -> str:
    """Formats seconds into HH:MM:SS or MM:SS or SS format."""
    delta = timedelta(seconds=seconds_float)
    parts = str(delta).split(":")
    if len(parts) == 3 and parts[0] == "0":
        return f"{parts[1]}:{parts[2].split('.')[0].zfill(2)}"
    elif len(parts) == 3:
        return f"{parts[0]}:{parts[1].zfill(2)}:{parts[2].split('.')[0].zfill(2)}"
    else:
        return f"{parts[1].split('.')[0].zfill(2)}"  # Handles cases less than an hour


def format_time_to_srt(seconds):
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    minutes, secs = divmod(int(seconds), 60)
    hours, mins = divmod(minutes, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{milliseconds:03d}"


# Convert generated segments from faster_whisper to Whisper format


def to_whisper_format(generated_segments):
    whisper_formatted_generated_segment = []
    for segment in generated_segments:
        whisper_formatted_generated_segment.append(
            {
                "id": segment.id,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": segment.tokens,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
                "words": segment.words,
                "temperature": segment.temperature,
            }
        )
    return {"segments": whisper_formatted_generated_segment}


# CREDIT: https://stackoverflow.com/a/72386137


def to_wav_pyav(in_path: str, out_path: str = None, sample_rate: int = 16000) -> str:
    """Arbitrary media files to wav"""
    if out_path is None:
        out_path = os.path.splitext(in_path)[0] + ".wav"
    with av.open(in_path) as in_container:
        in_stream = in_container.streams.audio[0]
        with av.open(out_path, "w", "wav") as out_container:
            out_stream = out_container.add_stream(
                "pcm_s16le", rate=sample_rate, layout="mono"
            )
            for frame in in_container.decode(in_stream):
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)

    return out_path


# Convert audio file to .wav


def to_wav(file_name):
    """Convert audio file to WAV format if needed.

    Args:
        file_name: Path to audio file

    Returns:
        Path to WAV file

    Raises:
        AudioConversionError: If conversion fails
    """
    if file_name.endswith(".wav"):
        return file_name
    else:
        try:
            logger.info("Converting audio file to .wav")
            out_path = to_wav_pyav(in_path=file_name)
            return out_path
        except Exception as e:
            logger.error(f"Error PyAV: {e}")
            raise AudioConversionError(
                f"Failed to convert audio file to WAV: {e}"
            ) from e


def resampling(file_name, sample_rate=16000):
    # Resample audio to 16kHz if needed
    waveform, sr = load(file_name)
    if sr != sample_rate:
        waveform = Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

    # Save the resampled audio
    save(file_name, waveform, sample_rate)


def snip_audio(input_file, output_file, start_time, duration):
    """Snip a portion of an audio file using pyAV.

    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        start_time: Start time in seconds
        duration: Duration to extract in seconds

    Returns:
        Path to output file

    Raises:
        AudioConversionError: If snipping fails
    """
    try:
        input_container = av.open(input_file)
        audio_stream = None
        for stream in input_container.streams:
            if stream.type == "audio":
                audio_stream = stream
                break

        if not audio_stream:
            logger.error(f"No audio stream found in {input_file}")
            raise AudioConversionError(f"No audio stream found in {input_file}")

        output_container = av.open(output_file, "w", format=input_container.format.name)
        output_stream = output_container.add_stream("pcm_s16le", rate=stream.rate)

        start_pts_seconds = start_time
        end_pts_seconds = start_time + duration

        for packet in input_container.demux(audio_stream):
            for frame in packet.decode():
                frame_time_seconds = frame.pts * audio_stream.time_base

                if start_pts_seconds <= frame_time_seconds < end_pts_seconds:
                    for packet_out in output_stream.encode(frame):
                        output_container.mux(packet_out)
                elif frame_time_seconds >= end_pts_seconds:
                    break

            if frame_time_seconds >= end_pts_seconds:
                break

        for packet_out in output_stream.encode():
            output_container.mux(packet_out)

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise AudioConversionError(f"Failed to snip audio: {e}") from e
    finally:
        if input_container:
            input_container.close()
        if output_container:
            output_container.close()
    return output_file
