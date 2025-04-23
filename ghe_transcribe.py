from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from torch import device as to_torch_device
from torch import set_num_threads, get_num_threads
from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available
import os
from utils import to_wav, to_whisper_format, diarize_text, to_csv, to_md, timing, resampling

@timing
def transcribe(audio_file, 
               device=None, 
               whisper_model='large-v3-turbo', 
               pyannote_model='pyannote/speaker-diarization-3.1', 
               save_output=True, 
               semicolon=False, 
               info=True):
    
    # Convert audio file to .wav
    # audio_file = to_wav(audio_file)

    # Resample at 16kHz
    resampling(audio_file, sample_rate=16000)

    # Device
    if device is None: device = 'cuda' if cuda_is_available() else 'mps' if mps_is_available() else 'cpu'
    try:
        torch_device = to_torch_device(device)
    except Exception as e:
        print(f"Device Error: {e}")
        return
    
    # CPU Threads 
    cpu_threads = get_num_threads()
    # set_num_threads(cpu_threads)

    # Automatic Speech Recognition (ASR): faster-whisper
    # https://github.com/SYSTRAN/faster-whisper/blob/1383fd4d3725bdf59c95d8834c629f45c6974981/faster_whisper/transcribe.py#L586

    try: 
        match device:
            case 'mps' | 'cpu':
                model = WhisperModel(model_size_or_path = whisper_model, 
                                     device = 'cpu',
                                     device_index = 0,
                                     compute_type = 'float32', 
                                     cpu_threads = cpu_threads,
                                     num_workers = 1,
                                     download_root = None,
                                     local_files_only = False,
                                     files = None,
                                     )
            case _:
                model = WhisperModel(model_size_or_path = whisper_model, 
                                     device = torch_device, 
                                     device_index = 0,
                                     compute_type = 'float32', 
                                     cpu_threads = cpu_threads,
                                     num_workers = 1,
                                     download_root = None,
                                     local_files_only = False,
                                     files = None,
                                     )           
    except Exception as e:
        print(f"WhisperModel Device Error: {e}")

    # https://github.com/SYSTRAN/faster-whisper/blob/1383fd4d3725bdf59c95d8834c629f45c6974981/faster_whisper/transcribe.py#L255

    segments, info = model.transcribe(audio_file, 
                                      beam_size = 5,
                                      temperature = 0.0,
                                      word_timestamps = False,
                                      vad_filter = False,
                                      vad_parameters = dict(min_silence_duration_ms=2000))
    generated_segments = list(segments)

    # Speaker Diarization: pyannote.audio


    try:
        diarization_result = Pipeline.from_pretrained(pyannote_model,
                                                      num_speakers = None,
                                                      min_speakers = None,
                                                      max_speakers = None,
                                                      ).to(torch_device)(audio_file)
    except Exception as e:
        print(f"Diarization Error: {e}")

    # Text alignment
    text = diarize_text(to_whisper_format(generated_segments), diarization_result)

    # Extract audio_file name
    output_file_name = 'output/'+os.path.splitext(os.path.basename(audio_file))[0]

    if save_output: 
        csv = to_csv(text, semicolon=semicolon)
        with open(output_file_name+'.csv', 'w') as f:
            f.write('\n'.join(csv))
            print('Output saved to'+output_file_name+'.csv')
        md = to_md(text)
        with open(output_file_name+'.md', 'w') as f:
            f.write('\n'.join(md))
            print('Output saved to'+output_file_name+'.md')

    if info: 
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        return text
    else: 
        return text