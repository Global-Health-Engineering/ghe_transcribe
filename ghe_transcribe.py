from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from torch import device
from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available
from utils import to_wav, to_whisper_format, diarize_text, write_to_txt 

def transcribe(audio_file, text_file=None, info_bool=True):
    # Convert audio file to .wav
    audio_file = to_wav(audio_file)

    # Device
    device_name = 'cuda' if cuda_is_available() else 'mps' if mps_is_available() else 'cpu'
    the_device = device(device_name)

    # Whisper model 
    if device_name == 'mps': 
        # faster_whisper doesn't support 'mps'
        model = WhisperModel('medium.en', device='cpu', compute_type='float32')
    else: 
        model = WhisperModel('medium.en', device=the_device, compute_type='float32')

    segments, info = model.transcribe(audio_file, beam_size=5)
    generated_segments = list(segments)

    # Pyannote pipeline
    diarization_result = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1').to(the_device)(audio_file)
    text = diarize_text(to_whisper_format(generated_segments), diarization_result)

    # Write to text file
    if text_file is not None: write_to_txt(text, text_file, semicolumn=True)

    if info_bool: 
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        return text
    else: 
        return text