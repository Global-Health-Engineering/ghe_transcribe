# CREDIT: https://github.com/yinruiqing/pyannote-whisper
import argparse
import os
import inspect
from functools import wraps
from time import time
from typing import Dict, Any, Callable

from av import open
from pyannote.core import Segment
from torchaudio import load, save
from torchaudio.transforms import Resample


def timing(func):
    @wraps(func)
    def wrap(*args, **kw):
        ts = time()
        result = func(*args, **kw)
        te = time()
        print(f"func:{func.__name__!r} args:[{args!r}, {kw!r}] took: {te - ts:2.4f} sec")
        return result

    return wrap


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
    timestamp_texts = get_text_with_timestamp(transcribe_res)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    result = merge_sentence(spk_text)
    return result


def to_csv(result, semicolon=False):
    csv = []
    start_line = (
        "start;end;speaker;sentence" if semicolon else "start,end,speaker,sentence"
    )
    csv.append(start_line)
    if semicolon:
        for seg, spk, sentence in result:
            line = f"{seg.start:.2f};{seg.end:.2f};{spk};{sentence}"
            csv.append(line.strip())
    else:
        # Convert all ',' in sentence to ';'
        for seg, spk, sentence in result:
            sentence = sentence.replace(",", ";")
            line = f"{seg.start:.2f},{seg.end:.2f},{spk},{sentence}"
            csv.append(line.strip())
    return csv


def spk_to_id(spk):
    # in_spk = 'SPEAKER_00'
    id = str(int(spk.split("_")[1]))
    return id


def to_md(result):
    md = []
    previous_spk = None
    for seg, spk, sentence in result:
        if spk != previous_spk:
            md.append(f"\n{spk}")
            md.append(f"({format_time(seg.start)}){sentence}".strip())
            previous_spk = spk
        else:
            md.append(f"({format_time(seg.start)}){sentence}".strip())
    return md


def md_to_csv(md_file):
    # to be implemented
    csv = []
    return csv


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
    with open(in_path) as in_container:
        in_stream = in_container.streams.audio[0]
        with open(out_path, "w", "wav") as out_container:
            out_stream = out_container.add_stream(
                "pcm_s16le", rate=sample_rate, layout="mono"
            )
            for frame in in_container.decode(in_stream):
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)

    return out_path


# Convert audio file to .wav


def to_wav(file_name):
    if file_name.endswith(".wav"):
        return file_name
    else:
        try:
            print("Converting audio file to .wav")
            out_path = to_wav_pyav(in_path=file_name)
            return out_path
        except Exception as e:
            print(f"Error PyAV: {e}")


def resampling(file_name, sample_rate=16000):
    # Resample audio to 16kHz if needed
    waveform, sr = load(file_name)
    if sr != sample_rate:
        waveform = Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

    # Save the resampled audio
    save(file_name, waveform, sample_rate)


def digit_to_string(num: int) -> str:
    if 0 <= num <= 9:
        return f"0{num}"
    else:
        return f"{num}"


def seconds_to_hours_minutes_seconds(num) -> int:
    # Expects time in seconds float or string, e.g. time = '11.27', 63.9
    seconds, minutes, hours = round(float(num)), 0, 0
    if seconds >= 60:
        minutes = seconds // 60
        seconds -= minutes * 60
    if minutes >= 60:
        hours = minutes // 60
        minutes -= hours * 60
    return seconds, minutes, hours


def format_time(num) -> str:
    seconds, minutes, hours = seconds_to_hours_minutes_seconds(num)
    try:
        if minutes == 0 and hours == 0:
            return digit_to_string(seconds)
        elif hours == 0:
            return f"{digit_to_string(minutes)}:{digit_to_string(seconds)}"
        else:
            return f"{digit_to_string(hours)}:{digit_to_string(minutes)}:{digit_to_string(seconds)}"
    except Exception as e:
        print(f"Time Formatting Error: {e}")


def snip_audio(input_file, output_file, start_time, duration):
    """
    Snips a portion of an audio file using pyAV.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the snipped audio.
        start_time (float): Start time of the snippet in seconds.
        duration (float): Duration of the snippet in seconds.
    """
    try:
        input_container = open(input_file)
        audio_stream = None
        for stream in input_container.streams:
            if stream.type == 'audio':
                audio_stream = stream
                break

        if not audio_stream:
            print(f"Error: No audio stream found in {input_file}")
            return

        output_container = open(output_file, 'w', format=input_container.format.name)
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
        print(f"Error processing file: {e}")
    finally:
        if input_container:
            input_container.close()
        if output_container:
            output_container.close()
    return output_file


class ArgparseGenerator:
    """
    Generates argparse arguments from a function's signature and a configuration dictionary.
    """

    def __init__(self, target: Callable, config_default: Dict[str, Any]):
        """
        Initializes the ArgparseGenerator.

        Args:
            target (callable): The function whose signature should be used to generate arguments.
            config_default (dict): A dictionary containing default values and optional "CHOICES_*"
                                  for the function's parameters.
        """
        self.config_default = config_default
        self.target = target
        self.parser = argparse.ArgumentParser(
            description=target.__doc__.split('\n')[0] if target.__doc__ else ""
        )
        self._add_arguments()  # Call this in the constructor

    def _add_arguments(self):
        """Adds arguments to the parser based on the target function's signature."""
        sig = inspect.signature(self.target)

        for name, param in sig.parameters.items():
            arg_name = name
            arg_type = param.annotation
            default_value = self.config_default.get("default", {}).get(name)
            choices = self.config_default.get("choices", {}).get(name)  # Get choices, if available

            if name not in self.config_default.get("default"):  # Check if the parameter is required
                self.parser.add_argument(
                    arg_name,  # No leading dashes for required arguments
                    type=arg_type,
                    help=next(
                        (line.split(':', 1)[1].strip()
                         for line in self.target.__doc__.split('\n')
                         if f'{name} (' in line),
                        f'{name.replace("_", " ").capitalize()}.'  # Fallback
                    )
                )
            else:
                if arg_type == inspect._empty:
                    arg_type = type(default_value) if default_value is not None else str

                help_text = next(
                    (line.split(':', 1)[1].strip()
                     for line in self.target.__doc__.split('\n')
                     if f'{name} (' in line),
                    f'{name.replace("_", " ").capitalize()}. Default: {default_value}'  # Fallback
                )
                if choices:
                    self.parser.add_argument(
                        f"--{arg_name}",
                        type=arg_type,
                        choices=choices,
                        default=default_value,
                        help=help_text
                    )
                else:
                    self.parser.add_argument(
                        f"--{arg_name}",
                        type=arg_type,
                        default=default_value,
                        help=help_text
                    )

    def parse_args(self) -> argparse.Namespace:
        """
        Parses the command-line arguments.

        Returns:
            argparse.Namespace: An argparse.Namespace object containing the parsed arguments.
        """
        print(self.parser.parse_args())
        return self.parser.parse_args()

    def print_help_md(self) -> str:
        """
        Prints the help message in Markdown format.

        Returns:
            str: The help message formatted as Markdown.
        """
        help_text = self.parser.format_help()
        lines = help_text.splitlines()
        markdown_output = "## " + lines[0] + "\n\n"  #  Title

        if len(lines) > 2:
            markdown_output += "### " + lines[2] + "\n\n" # Add a sub-title

        for line in lines[3:]: #Skip the first few lines
            if line.startswith(" "):
                markdown_output +=  line + "\n"
            elif line.strip() == "optional arguments:":
                markdown_output += "\n**" + line.strip() + "**\n"
            elif line.strip() == "positional arguments:":
                markdown_output += "\n**" + line.strip() + "**\n"
            else:
                markdown_output += "\n**" + line.strip() + "**\n"
        return markdown_output
