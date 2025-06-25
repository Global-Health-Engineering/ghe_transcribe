import os
import tempfile
from pathlib import Path

import pytest
import torch
import torchaudio

from ghe_transcribe.utils import to_wav, to_wav_pyav
from ghe_transcribe.exceptions import AudioConversionError


class TestAudioConversion:
    """Test audio conversion functionality using av."""

    @classmethod
    def setup_class(cls):
        """Create test audio files in different formats."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.sample_rate = 16000
        cls.duration = 2  # 2 seconds
        cls.frequency = 440  # A4 note

        # Generate a simple sine wave
        t = torch.linspace(0, cls.duration, int(cls.sample_rate * cls.duration))
        cls.audio_data = torch.sin(2 * torch.pi * cls.frequency * t).unsqueeze(0)

        # Create test files
        cls.wav_file = os.path.join(cls.temp_dir, "test_audio.wav")
        cls.mp3_file = os.path.join(cls.temp_dir, "test_audio.mp3")
        cls.m4a_file = os.path.join(cls.temp_dir, "test_audio.m4a")

        # Save original WAV file
        torchaudio.save(cls.wav_file, cls.audio_data, cls.sample_rate)

        # Create MP3 file using torchaudio backend
        try:
            torchaudio.save(cls.mp3_file, cls.audio_data, cls.sample_rate, format="mp3")
        except Exception:
            # If MP3 encoding not available, skip MP3 tests
            cls.mp3_file = None

        # Create M4A file using torchaudio backend
        try:
            torchaudio.save(cls.m4a_file, cls.audio_data, cls.sample_rate, format="mp4")
        except Exception:
            # If M4A encoding not available, skip M4A tests
            cls.m4a_file = None

    @classmethod
    def teardown_class(cls):
        """Clean up test files."""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_to_wav_with_wav_file(self):
        """Test that WAV files are returned unchanged."""
        result = to_wav(self.wav_file)
        assert result == self.wav_file
        assert os.path.exists(result)

    def test_to_wav_pyav_mp3_conversion(self):
        """Test conversion from MP3 to WAV using PyAV."""
        if self.mp3_file is None:
            pytest.skip("MP3 encoding not available")

        output_file = os.path.join(self.temp_dir, "converted_from_mp3.wav")
        result = to_wav_pyav(self.mp3_file, output_file)

        assert result == output_file
        assert os.path.exists(output_file)
        
        # Verify the converted file has correct properties
        converted_audio, converted_sr = torchaudio.load(output_file)
        assert converted_sr == 16000  # Should be resampled to 16kHz
        assert converted_audio.shape[0] == 1  # Should be mono
        assert converted_audio.shape[1] > 0  # Should have audio data

    def test_to_wav_pyav_m4a_conversion(self):
        """Test conversion from M4A to WAV using PyAV."""
        if self.m4a_file is None:
            pytest.skip("M4A encoding not available")

        output_file = os.path.join(self.temp_dir, "converted_from_m4a.wav")
        result = to_wav_pyav(self.m4a_file, output_file)

        assert result == output_file
        assert os.path.exists(output_file)
        
        # Verify the converted file has correct properties
        converted_audio, converted_sr = torchaudio.load(output_file)
        assert converted_sr == 16000  # Should be resampled to 16kHz
        assert converted_audio.shape[0] == 1  # Should be mono
        assert converted_audio.shape[1] > 0  # Should have audio data

    def test_to_wav_mp3_conversion(self):
        """Test high-level MP3 conversion using to_wav function."""
        if self.mp3_file is None:
            pytest.skip("MP3 encoding not available")

        result = to_wav(self.mp3_file)
        
        # Should create a WAV file with the same base name
        expected_output = os.path.splitext(self.mp3_file)[0] + ".wav"
        assert result == expected_output
        assert os.path.exists(result)
        
        # Verify the converted file properties
        converted_audio, converted_sr = torchaudio.load(result)
        assert converted_sr == 16000
        assert converted_audio.shape[0] == 1  # mono
        assert converted_audio.shape[1] > 0

    def test_to_wav_m4a_conversion(self):
        """Test high-level M4A conversion using to_wav function."""
        if self.m4a_file is None:
            pytest.skip("M4A encoding not available")

        result = to_wav(self.m4a_file)
        
        # Should create a WAV file with the same base name
        expected_output = os.path.splitext(self.m4a_file)[0] + ".wav"
        assert result == expected_output
        assert os.path.exists(result)
        
        # Verify the converted file properties
        converted_audio, converted_sr = torchaudio.load(result)
        assert converted_sr == 16000
        assert converted_audio.shape[0] == 1  # mono
        assert converted_audio.shape[1] > 0

    def test_to_wav_pyav_custom_sample_rate(self):
        """Test conversion with custom sample rate."""
        if self.mp3_file is None:
            pytest.skip("MP3 encoding not available")

        output_file = os.path.join(self.temp_dir, "converted_22khz.wav")
        custom_rate = 22050
        
        result = to_wav_pyav(self.mp3_file, output_file, sample_rate=custom_rate)
        
        assert result == output_file
        assert os.path.exists(output_file)
        
        # Verify custom sample rate
        converted_audio, converted_sr = torchaudio.load(output_file)
        assert converted_sr == custom_rate

    def test_to_wav_pyav_invalid_input(self):
        """Test conversion with invalid input file."""
        non_existent_file = os.path.join(self.temp_dir, "non_existent.mp3")
        output_file = os.path.join(self.temp_dir, "output.wav")
        
        with pytest.raises(Exception):  # Should raise an av-related exception
            to_wav_pyav(non_existent_file, output_file)

    def test_to_wav_pyav_auto_output_path(self):
        """Test conversion with automatic output path generation."""
        if self.mp3_file is None:
            pytest.skip("MP3 encoding not available")

        result = to_wav_pyav(self.mp3_file)  # No output path specified
        
        # Should create WAV file with same base name
        expected_output = os.path.splitext(self.mp3_file)[0] + ".wav"
        assert result == expected_output
        assert os.path.exists(result)

    def test_converted_audio_quality(self):
        """Test that converted audio maintains reasonable quality."""
        if self.mp3_file is None:
            pytest.skip("MP3 encoding not available")

        # Convert MP3 to WAV
        converted_wav = to_wav(self.mp3_file)
        
        # Load original and converted audio
        original_audio, _ = torchaudio.load(self.wav_file)
        converted_audio, _ = torchaudio.load(converted_wav)
        
        # Ensure they have the same number of channels
        assert original_audio.shape[0] == converted_audio.shape[0]
        
        # Check that converted audio has reasonable duration (within 10% of original)
        original_length = original_audio.shape[1]
        converted_length = converted_audio.shape[1]
        length_ratio = converted_length / original_length
        assert 0.9 <= length_ratio <= 1.1, f"Length ratio {length_ratio} outside acceptable range"