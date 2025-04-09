from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from torch import device as to_torch_device
from torch import set_num_threads, get_num_threads
from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available
import os
from utils import to_wav, to_whisper_format, diarize_text, to_csv, to_md, timing

@timing
def transcribe(audio_file, save_output=True, device=None, whisper_model='medium.en', pyannote_model='pyannote/speaker-diarization-3.1', semicolon=False, info=True):
    # Convert audio file to .wav
    audio_file = to_wav(audio_file)

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

    # Whisper model 
    try: 
        match device:
            case 'mps' | 'cpu':
                model = WhisperModel(whisper_model, device='cpu', compute_type='float32', cpu_threads=cpu_threads)
            case _:
                model = WhisperModel(whisper_model, device=torch_device, compute_type='float32')            
    except Exception as e:
        print(f"WhisperModel Device Error: {e}")

    segments, info = model.transcribe(audio_file, beam_size=5)
    generated_segments = list(segments)

    # Pyannote pipeline
    try:
        diarization_result = Pipeline.from_pretrained(pyannote_model).to(torch_device)(audio_file)
        text = diarize_text(to_whisper_format(generated_segments), diarization_result)
    except Exception as e:
        print(f"Diarization Error: {e}")

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
    else:
        if info: 
            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            return text
        else: 
            return text