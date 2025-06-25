# Research for Development: a Literature Review

Explore these related projects and libraries for more advanced functionalities or alternative approaches.

## Transcription Libraries

  - **Whisper:** ([https://github.com/openai/whisper](https://github.com/openai/whisper)) - OpenAI's original speech-to-text model.
  - **faster-whisper:** ([https://github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)) - The fast implementation used in `ghe_transcribe`, offering significant speed improvements. See [benchmarks](https://github.com/SYSTRAN/faster-whisper/issues/1030).
  - **Deepgram Benchmarks:** ([https://deepgram.com/learn/benchmarking-top-open-source-speech-models](https://deepgram.com/learn/benchmarking-top-open-source-speech-models)) - A comparison of various open-source speech models, including Whisper.

## Speaker Diarization Libraries

  - **pyannote.audio:** ([https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)) - The speaker diarization toolkit by pyannoteAI ([https://www.pyannote.ai/](https://www.pyannote.ai/)), requiring a Hugging Face access token ([https://hf.co/settings/tokens](https://hf.co/settings/tokens)).
  - **NeMo:** ([https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)) - NVIDIA's open-source framework for conversational AI, including diarization capabilities.
  - **Pyannote vs NeMo Comparison:** ([https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300](https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300)) - A blog post comparing the two frameworks.

## Combined Transcription and Diarization Tools

  - **WhisperX:** ([https://github.com/m-bain/whisperX](https://github.com/m-bain/whisperX)) - Combines `faster-whisper` with `pyannote.audio` for transcription and diarization.
  - **whisper-diarization:** ([https://github.com/MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization)) - Integrates `faster-whisper` with `NeMo` for speaker diarization.
  - **insanely-fast-whisper:** ([https://github.com/Vaibhavs10/insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)) - Another optimized Whisper implementation combined with `pyannote.audio`.

## Graphical User Interfaces

  - **wscribe-editor:** ([https://github.com/geekodour/wscribe-editor](https://github.com/geekodour/wscribe-editor)) - An editor for transcriptions with word-level timestamps, supporting a specific JSON format (see [sample.json](https://github.com/geekodour/wscribe/blob/main/examples/output/sample.json)).
  - **QualCoder:** ([https://github.com/ccbogel/QualCoder](https://github.com/ccbogel/QualCoder)) - A general-purpose qualitative data analysis tool with some audio transcription features.

## GUI Applications with Transcription and Diarization
  - **noScribe:** ([https://github.com/kaixxx/noScribe](https://github.com/kaixxx/noScribe)) - A GUI application leveraging `faster-whisper` and `pyannote.audio`.
  - **TranscriboZH:** ([https://github.com/machinelearningZH/audio-transcription](https://github.com/machinelearningZH/audio-transcription)) - Another GUI tool based on `WhisperX`.