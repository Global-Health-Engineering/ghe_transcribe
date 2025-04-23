import argparse
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from torch import device as to_torch_device
from torch import set_num_threads
from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available
import os
from ghe_transcribe.utils import to_wav, to_whisper_format, diarize_text, to_csv, to_md, timing, resampling

@timing
def transcribe(audio_file,
               device,
               whisper_model,
               device_index,
               compute_type,
               cpu_threads,
               beam_size,
               temperature,
               word_timestamps,
               vad_filter,
               vad_parameters,
               pyannote_model,
               num_speakers,
               min_speakers,
               max_speakers,
               save_output,
               info):

    # Convert audio file to .wav
    audio_file = to_wav(audio_file)

    # Resample at 16kHz
    # resampling(audio_file, sample_rate=16000)

    # Device
    if device is None: device = 'cuda' if cuda_is_available() else 'mps' if mps_is_available() else 'cpu'
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
        whisper_model_kwargs['cpu_threads'] = cpu_threads
        set_num_threads(cpu_threads)

    try:
        match device:
            case 'mps' | 'cpu':
                model = WhisperModel(model_size_or_path = whisper_model,
                                     device = 'cpu',
                                     device_index = device_index,
                                     compute_type = compute_type,
                                     num_workers = 1,
                                     download_root = None,
                                     local_files_only = False,
                                     files = None,
                                     **whisper_model_kwargs
                                     )
            case _:
                model = WhisperModel(model_size_or_path = whisper_model,
                                     device = torch_device,
                                     device_index = device_index,
                                     compute_type = compute_type,
                                     num_workers = 1,
                                     download_root = None,
                                     local_files_only = False,
                                     files = None,
                                     **whisper_model_kwargs
                                     )
    except Exception as e:
        print(f"WhisperModel Device Error: {e}")

    # https://github.com/SYSTRAN/faster-whisper/blob/1383fd4d3725bdf59c95d8834c629f45c6974981/faster_whisper/transcribe.py#L255

    # Create a dictionary of keyword arguments
    whisper_transcribe_kwargs = {}
    if beam_size is not None:
        whisper_transcribe_kwargs['beam_size'] = beam_size
    if temperature is not None:
        whisper_transcribe_kwargs['temperature'] = temperature
    if word_timestamps is not None:
        whisper_transcribe_kwargs['word_timestamps'] = word_timestamps
    if vad_filter is not None:
        whisper_transcribe_kwargs['vad_filter'] = vad_filter
    if vad_parameters is not None:
        whisper_transcribe_kwargs['vad_parameters'] = vad_parameters

    segments, info = model.transcribe(audio_file,
                                      **whisper_transcribe_kwargs)
    generated_segments = list(segments)

    # Speaker Diarization: pyannote.audio

    # Create a dictionary of keyword arguments
    pyannote_kwargs = {}
    if num_speakers is not None:
        pyannote_kwargs['num_speakers'] = num_speakers
    if min_speakers is not None:
        pyannote_kwargs['min_speakers'] = min_speakers
    if max_speakers is not None:
        pyannote_kwargs['max_speakers'] = max_speakers

    try:
        diarization_result = Pipeline.from_pretrained(pyannote_model).to(torch_device)(audio_file, **pyannote_kwargs)
    except Exception as e:
        print(f"Diarization Error: {e}")

    # Text alignment
    text = diarize_text(to_whisper_format(generated_segments), diarization_result)

    # Extract audio_file name
    output_file_name = 'output/'+os.path.splitext(os.path.basename(audio_file))[0]

    if save_output:
        csv = to_csv(text)
        with open(output_file_name+'.csv', 'w') as f:
            f.write('\n'.join(csv))
            print(f'Output saved to {output_file_name}.csv')
        md = to_md(text)
        with open(output_file_name+'.md', 'w') as f:
            f.write('\n'.join(md))
            print(f'Output saved to {output_file_name}.md')

    if info:
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        return text
    else:
        return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe and diarize an audio file.')
    parser.add_argument('audio_file', type=str, help='Path to the audio file.')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'], help='Device to use (cuda, mps, or cpu). Defaults to auto-detection.')
    parser.add_argument('--whisper_model', type=str, default='large-v3-turbo', choices=['tiny.en','tiny','base.en','base','small.en','small','medium.en','medium','large-v1','large-v2','large-v3','large','distil-large-v2','distil-medium.en','distil-small.en','distil-large-v3','large-v3-turbo','turbo'], help='Faster Whisper model to use.')
    parser.add_argument('--device_index', type=int, default=0, help='Index of the device to use.')
    parser.add_argument('--compute_type', type=str, default='float32', choices=['float32', 'float16', 'int8'], help='Compute type for Whisper model.')
    parser.add_argument('--cpu_threads', type=int, default=None, help='Number of CPU threads to use for Whisper.')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for Whisper decoding.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for Whisper sampling.')
    parser.add_argument('--word_timestamps', type=bool, default=None, help='Enable word timestamps in Whisper output.')
    parser.add_argument('--vad_filter', type=bool, default=False, help='Enable voice activity detection in Whisper.')
    parser.add_argument('--min_silence_duration_ms', type=int, default=2000, help='Minimum silence duration for VAD (ms).')
    parser.add_argument('--pyannote_model', type=str, default='pyannote/speaker-diarization-3.1', help='Pyannote speaker diarization model to use.')
    parser.add_argument('--num_speakers', type=int, default=None, help='Number of speakers for diarization (if known).')
    parser.add_argument('--min_speakers', type=int, default=None, help='Minimum number of speakers for diarization.')
    parser.add_argument('--max_speakers', type=int, default=None, help='Maximum number of speakers for diarization.')
    parser.add_argument('--save_output', type=bool, default=True, help='Save output to .csv and .md files.')
    parser.add_argument('--info', type=bool, default=True, help='Print detected language information.')

    args = parser.parse_args()

    transcribe(audio_file=args.audio_file,
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
               info=args.info)