"""Audio processing and hearing capabilities for Lunaa AI"""
import os

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.recording = None
        
    def record_audio(self, duration: int = 5, output_file: str = 'recording.wav') -> str:
        """Record audio from microphone"""
        if not _AUDIO_AVAILABLE:
            return "Audio dependencies not installed"
        
        try:
            print(f"Recording for {duration} seconds...")
            recording = sd.rec(int(duration * self.sample_rate), 
                              samplerate=self.sample_rate, 
                              channels=1)
            sd.wait()
            sf.write(output_file, recording, self.sample_rate)
            return f"Audio saved to {output_file}"
        except Exception as e:
            return f"Error recording audio: {e}"
    
    def play_audio(self, audio_file: str) -> str:
        """Play audio file"""
        if not _AUDIO_AVAILABLE:
            return "Audio dependencies not installed"
        
        try:
            data, sample_rate = sf.read(audio_file)
            sd.play(data, sample_rate)
            sd.wait()
            return f"Played {audio_file}"
        except Exception as e:
            return f"Error playing audio: {e}"
    
    def transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio file to text (placeholder for Whisper integration)"""
        # This would integrate with Whisper or similar
        return "Audio transcription requires Whisper model integration"
