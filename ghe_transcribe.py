from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from torch import device as to_torch_device
from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available
import os
from utils import to_wav, to_whisper_format, diarize_text, to_csv, to_md

def transcribe(audio_file, output_file=None, device=None, whisper_model='medium.en', pyannote_model='pyannote/speaker-diarization-3.1', semicolumn=False, info=True):
    # Convert audio file to .wav
    audio_file = to_wav(audio_file)

    # Device
    if device is None: device = 'cuda' if cuda_is_available() else 'mps' if mps_is_available() else 'cpu'
    try:
        torch_device = to_torch_device(device)
    except Exception as e:
        print(f"Device Error: {e}")
        return

    # Whisper model 
    try: 
        if device == 'mps': 
            model = WhisperModel(whisper_model, device='cpu', compute_type='float32')
        else:
            model = WhisperModel(whisper_model, device=torch_device, compute_type='float32')
    except Exception as e:
        print(f"WhisperModel Device Error: {e}")
        print("Fallbacking to CPU.")
        model = WhisperModel(whisper_model, device='cpu', compute_type='float32')

    segments, info = model.transcribe(audio_file, beam_size=5)
    generated_segments = list(segments)

    # Pyannote pipeline
    try:
        diarization_result = Pipeline.from_pretrained(pyannote_model).to(torch_device)(audio_file)
        text = diarize_text(to_whisper_format(generated_segments), diarization_result)
    except Exception as e:
        print(f"Diarization Error: {e}")

    # Remove suffix from output_file
    output_file_name = os.path.splitext(output_file)[0]

    if output_file is not None: 
        csv = to_csv(text, semicolumn=semicolumn)
        with open(output_file_name+'.csv', 'w') as f:
            f.write('\n'.join(csv))
        md = to_md(text)
        with open(output_file_name+'.md', 'w') as f:
            f.write('\n'.join(md))

    if info: 
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        return text
    else: 
        return text