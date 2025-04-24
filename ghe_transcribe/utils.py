# CREDIT: https://github.com/yinruiqing/pyannote-whisper

import os
from functools import wraps
from time import time
from av import open
from pyannote.core import Segment
from torchaudio import load, save
from torchaudio.transforms import Resample


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' %
              (f.__name__, args, kw, te-ts))
        return result
    return wrap


def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res['segments']:
        start = item['start']
        end = item['end']
        text = item['text']
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts


def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text


def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence


__PUNC_SENT_END__ = ['.', '?', '!']


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
    start_line = 'start;end;speaker;sentence' if semicolon else 'start,end,speaker,sentence'
    csv.append(start_line)
    if semicolon:
        for seg, spk, sentence in result:
            line = f'{seg.start:.2f};{seg.end:.2f};{spk};{sentence}'
            csv.append(line.strip())
    else:
        # Convert all ',' in sentence to ';'
        for seg, spk, sentence in result:
            sentence = sentence.replace(',', ';')
            line = f'{seg.start:.2f},{seg.end:.2f},{spk},{sentence}'
            csv.append(line.strip())
    return csv


def spk_to_id(spk):
    # in_spk = "SPEAKER_00"
    id = str(int(spk.split('_')[1]))
    return id


def to_md(result):
    md = []
    previous_spk = None
    for seg, spk, sentence in result:
        if spk != previous_spk:
            md.append(f'\n{spk}')
            md.append(f'({format_time(seg.start)}){sentence}'.strip())
            previous_spk = spk
        else:
            md.append(f'({format_time(seg.start)}){sentence}'.strip())
    return md


def md_to_csv(md_file):
    # to be implemented
    csv = []
    return csv

# Convert generated segments from faster_whisper to Whisper format


def to_whisper_format(generated_segments):
    whisper_formatted_generated_segment = []
    for segment in generated_segments:
        whisper_formatted_generated_segment.append({"id": segment.id,
                                                    "seek": segment.seek,
                                                    "start": segment.start,
                                                    "end": segment.end,
                                                    "text": segment.text,
                                                    "tokens": segment.tokens,
                                                    "avg_logprob": segment.avg_logprob,
                                                    "compression_ratio": segment.compression_ratio,
                                                    "no_speech_prob": segment.no_speech_prob,
                                                    "words": segment.words,
                                                    "temperature": segment.temperature
                                                    })
    return {"segments": whisper_formatted_generated_segment}

# CREDIT: https://stackoverflow.com/a/72386137


def to_wav_pyav(in_path: str, out_path: str = None, sample_rate: int = 16000) -> str:
    """Arbitrary media files to wav"""
    if out_path is None:
        out_path = os.path.splitext(in_path)[0] + '.wav'
    with open(in_path) as in_container:
        in_stream = in_container.streams.audio[0]
        with open(out_path, 'w', 'wav') as out_container:
            out_stream = out_container.add_stream(
                'pcm_s16le',
                rate=sample_rate,
                layout='mono'
            )
            for frame in in_container.decode(in_stream):
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)

    return out_path

# Convert audio file to .wav


def to_wav(file_name):
    if file_name.endswith('.wav'):
        return file_name
    else:
        try:
            print("Converting audio file to .wav")
            out_path = to_wav_pyav(in_path=file_name)
            return out_path
        except:
            print("Error PyAV")


def resampling(file_name, sample_rate=16000):
    # Resample audio to 16kHz if needed
    waveform, sr = load(file_name)
    if sr != sample_rate:
        waveform = Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

    # Save the resampled audio
    save(file_name, waveform, sample_rate)


def digit_to_string(num: int) -> str:
    if 0 <= num <= 9:
        return f'0{num}'
    else:
        return f'{num}'


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
            return f'{digit_to_string(minutes)}:{digit_to_string(seconds)}'
        else:
            return f'{digit_to_string(hours)}:{digit_to_string(minutes)}:{digit_to_string(seconds)}'
    except Exception as e:
        print(f'Time Formatting Error: {e}')
