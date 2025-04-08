# CREDIT: https://github.com/yinruiqing/pyannote-whisper

from pyannote.core import Segment
import ffmpeg
import os

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

def to_csv(result, semicolumn=False):
    csv = []
    start_line = f'start;end;speaker;sentence' if semicolumn else 'start,end,speaker,sentence'
    csv.append(start_line)
    if semicolumn:
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

def to_md(result):
    md = []
    previous_spk = None
    for seg, spk, sentence in result:
        if spk != previous_spk:
            md.append('\n---')
            md.append(f'#### {spk}')
            md.append(f'({seg.start:.2f}) {sentence}'.strip())
            previous_spk = spk
        else:
            md.append(f'({seg.start:.2f}) {sentence}'.strip())
    return md

def md_to_csv(md_file):
    # to be implemented
    csv = []
    return csv

# Convert generated segments from faster_whisper to Whisper format
def to_whisper_format(generated_segments):
    whisper_formatted_generated_segment = []
    for segment in generated_segments:
        whisper_formatted_generated_segment.append({"id":segment.id,
                                                    "seek":segment.seek,
                                                    "start":segment.start,
                                                    "end":segment.end,
                                                    "text":segment.text,
                                                    "tokens":segment.tokens,
                                                    "avg_logprob":segment.avg_logprob,
                                                    "compression_ratio":segment.compression_ratio,
                                                    "no_speech_prob":segment.no_speech_prob,
                                                    "words":segment.words,
                                                    "temperature":segment.temperature
                                                   })
    return {"segments": whisper_formatted_generated_segment}


# Convert audio file to .wav
def to_wav(file_name):
    if file_name.endswith('.wav'):
        return file_name
    else:
        file_name_wav = os.path.splitext(file_name)[0]+'.wav'
        try:
            if os.path.exists(file_name_wav):
                print(f"File {file_name_wav} already exists", "\nUsing existing file")
                return file_name_wav
            else:
                try:
                    stream = ffmpeg.input(file_name)
                    stream = ffmpeg.output(stream, file_name_wav)
                    ffmpeg.run(stream)
                    print("Conversion successful")
                    return file_name_wav
                except ffmpeg.Error as e:
                    print(f"Error: {e.stderr.decode()}")
                    return
        except FileNotFoundError:
            print(f"File not found: {file_name}")
            return