"""
MIDI to Audio Converter with Symbolic Manipulation
==================================================
This implementation provides a framework for MIDI to audio conversion with symbolic 
manipulation capabilities and dual synthesis backends.

Requirements:
- pretty_midi
- numpy
- torch
- fluidsynth
- librosa
"""

import os
import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
from typing import Tuple, List, Optional, Union, Dict

class MIDIProcessor:
    """
    Handles MIDI file parsing and symbolic manipulation
    """
    def __init__(self):
        pass
        
    def load_midi(self, midi_path: str) -> pretty_midi.PrettyMIDI:
        """
        Load a MIDI file into pretty_midi format
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            return midi_data
        except Exception as e:
            raise RuntimeError(f"Error loading MIDI file: {e}")
    
    def transpose(self, midi_data: pretty_midi.PrettyMIDI, semitones: int) -> pretty_midi.PrettyMIDI:
        """
        Transpose all notes in the MIDI file by the specified number of semitones
        """
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                note.pitch += semitones
                # Ensure pitch stays within valid MIDI range (0-127)
                note.pitch = max(0, min(127, note.pitch))
        return midi_data
    
    def adjust_velocity(self, midi_data: pretty_midi.PrettyMIDI, 
                        scale_factor: float = 1.0, 
                        offset: int = 0) -> pretty_midi.PrettyMIDI:
        """
        Adjust note velocities by scaling and/or adding an offset
        """
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                note.velocity = int(note.velocity * scale_factor + offset)
                # Ensure velocity stays within valid MIDI range (0-127)
                note.velocity = max(0, min(127, note.velocity))
        return midi_data
    
    def reassign_instruments(self, midi_data: pretty_midi.PrettyMIDI, 
                            instrument_map: Dict[int, int]) -> pretty_midi.PrettyMIDI:
        """
        Reassign MIDI program numbers (instruments) based on provided mapping
        
        Args:
            midi_data: MIDI data to modify
            instrument_map: Dictionary mapping original program numbers to new ones
        """
        for instrument in midi_data.instruments:
            if instrument.program in instrument_map:
                instrument.program = instrument_map[instrument.program]
        return midi_data

    def save_midi(self, midi_data: pretty_midi.PrettyMIDI, output_path: str) -> None:
        """
        Save modified MIDI data to a file
        """
        midi_data.write(output_path)


class DifferentiableSynthesizer(nn.Module):
    """
    Neural network-based synthesizer that learns to generate audio from MIDI
    using differentiable rendering techniques guided by mel-spectrogram loss
    """
    def __init__(self, 
                 sample_rate: int = 44100,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 instrument_embedding_dim: int = 64,
                 hidden_dim: int = 512):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Instrument embedding (maps program numbers to latent representations)
        self.instrument_embedding = nn.Embedding(128, instrument_embedding_dim)
        
        # Encoder for note properties (pitch, velocity, duration)
        self.note_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder to generate audio waveforms
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + instrument_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sample_rate)  # Generate 1 second of audio at a time
        )
        
        # For mel-spectrogram computation
        self.register_buffer('mel_basis', torch.from_numpy(
            librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)).float())
        
    def forward(self, notes_batch):
        """
        Generate audio for a batch of notes
        
        Args:
            notes_batch: List of (pitch, velocity, duration, program) tuples
        """
        batch_size = len(notes_batch)
        
        # Extract note properties and instrument programs
        pitches = torch.tensor([n[0] for n in notes_batch], dtype=torch.float32)
        velocities = torch.tensor([n[1] for n in notes_batch], dtype=torch.float32)
        durations = torch.tensor([n[2] for n in notes_batch], dtype=torch.float32)
        programs = torch.tensor([n[3] for n in notes_batch], dtype=torch.long)
        
        # Normalize note properties
        pitches = pitches / 127.0
        velocities = velocities / 127.0
        durations = torch.clamp(durations, 0.1, 10.0) / 10.0
        
        # Encode notes
        note_features = torch.stack([pitches, velocities, durations], dim=1)
        note_encodings = self.note_encoder(note_features)
        
        # Get instrument embeddings
        instrument_features = self.instrument_embedding(programs)
        
        # Combine note and instrument features
        combined_features = torch.cat([note_encodings, instrument_features], dim=1)
        
        # Generate audio
        waveforms = self.decoder(combined_features)
        
        return waveforms
    
    def compute_mel_spectrogram(self, audio):
        """
        Compute mel spectrogram from audio waveform
        """
        # Compute STFT
        stft = torch.stft(
            audio, 
            self.n_fft, 
            hop_length=self.hop_length, 
            window=torch.hann_window(self.n_fft).to(audio.device),
            return_complex=True
        )
        
        # Convert to power spectrogram
        power_spec = torch.abs(stft) ** 2
        
        # Convert to mel scale
        mel_spec = torch.matmul(self.mel_basis, power_spec)
        
        # Log scale
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        
        return log_mel_spec
    
    def mel_spectrogram_loss(self, generated_audio, target_audio):
        """
        Compute loss between generated and target audio using mel-spectrograms
        """
        gen_mel = self.compute_mel_spectrogram(generated_audio)
        target_mel = self.compute_mel_spectrogram(target_audio)
        
        # Mean squared error between mel-spectrograms
        loss = F.mse_loss(gen_mel, target_mel)
        
        return loss
    
    def train_with_samples(self, note_samples, audio_samples, num_epochs=100, lr=1e-4):
        """
        Train the synthesizer with real instrument samples
        
        Args:
            note_samples: List of (pitch, velocity, duration, program) tuples
            audio_samples: Corresponding audio waveforms for each note
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for i in range(len(note_samples)):
                # Forward pass
                generated_audio = self.forward([note_samples[i]])
                target_audio = audio_samples[i]
                
                # Compute loss
                loss = self.mel_spectrogram_loss(generated_audio, target_audio)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(note_samples)}")
    
    def render_midi(self, midi_data: pretty_midi.PrettyMIDI, output_path: str):
        """
        Render a complete MIDI file to audio using the trained model
        """
        # Initialize output audio array
        duration = midi_data.get_end_time()
        num_samples = int(duration * self.sample_rate)
        output_audio = np.zeros(num_samples)
        
        # Process each instrument
        for instrument in midi_data.instruments:
            program = instrument.program
            
            # Process each note
            for note in instrument.notes:
                pitch = note.pitch
                velocity = note.velocity
                duration = note.end - note.start
                start_sample = int(note.start * self.sample_rate)
                
                # Generate audio for this note
                note_audio = self.forward([(pitch, velocity, duration, program)])
                note_audio = note_audio.detach().cpu().numpy()[0]
                
                # Trim or pad note audio to match duration
                required_samples = int(duration * self.sample_rate)
                if len(note_audio) > required_samples:
                    note_audio = note_audio[:required_samples]
                else:
                    note_audio = np.pad(note_audio, (0, required_samples - len(note_audio)))
                
                # Add to output at the correct position
                end_sample = start_sample + len(note_audio)
                if end_sample <= num_samples:
                    output_audio[start_sample:end_sample] += note_audio
        
        # Normalize audio
        max_val = np.max(np.abs(output_audio))
        if max_val > 0:
            output_audio = output_audio / max_val * 0.9
        
        # Save to file
        sf.write(output_path, output_audio, self.sample_rate)


class FluidSynthBackend:
    """
    Rule-based synthesizer using FluidSynth and SoundFonts
    """
    def __init__(self, soundfont_path: str, sample_rate: int = 44100):
        """
        Initialize FluidSynth backend
        
        Args:
            soundfont_path: Path to a SoundFont file (.sf2)
            sample_rate: Audio sample rate
        """
        import fluidsynth
        
        self.sample_rate = sample_rate
        self.fs = fluidsynth.Synth()
        self.fs.start(driver="file", midi_driver="alsa_seq")
        
        # Load SoundFont
        sfid = self.fs.sfload(soundfont_path)
        self.fs.program_select(0, sfid, 0, 0)
    
    def render_midi(self, midi_data: pretty_midi.PrettyMIDI, output_path: str):
        """
        Render MIDI to audio using FluidSynth
        """
        # Save MIDI to a temporary file
        temp_midi_path = "temp_for_fluidsynth.mid"
        midi_data.write(temp_midi_path)
        
        # Use FluidSynth to render the MIDI file
        self.fs.midi_player_add(temp_midi_path)
        self.fs.player_play()
        
        # Wait for playback to complete
        import time
        duration = midi_data.get_end_time()
        time.sleep(duration)
        
        self.fs.player_stop()
        
        # For actual implementation, use direct FluidSynth Python bindings
        # to render to a file without real-time playback
        # This would typically be:
        # audio_data = self.fs.get_samples(int(duration * self.sample_rate))
        # sf.write(output_path, audio_data, self.sample_rate)
        
        # Clean up
        if os.path.exists(temp_midi_path):
            os.remove(temp_midi_path)


class MIDIToAudioConverter:
    """
    Main class for MIDI to audio conversion with optional symbolic manipulation
    """
    def __init__(self, synthesis_backend: str = "fluidsynth", 
                soundfont_path: Optional[str] = None,
                model_path: Optional[str] = None):
        """
        Initialize the converter
        
        Args:
            synthesis_backend: Either "differentiable" or "fluidsynth"
            soundfont_path: Path to SoundFont file (required for FluidSynth)
            model_path: Path to pretrained model (optional for differentiable synthesis)
        """
        self.midi_processor = MIDIProcessor()
        self.synthesis_backend = synthesis_backend
        
        if synthesis_backend == "fluidsynth":
            if soundfont_path is None:
                raise ValueError("soundfont_path must be provided for FluidSynth backend")
            self.synthesizer = FluidSynthBackend(soundfont_path)
        
        elif synthesis_backend == "differentiable":
            self.synthesizer = DifferentiableSynthesizer()
            
            # Load pretrained model if provided
            if model_path is not None and os.path.exists(model_path):
                self.synthesizer.load_state_dict(torch.load(model_path))
        
        else:
            raise ValueError("synthesis_backend must be either 'differentiable' or 'fluidsynth'")
    
    def convert(self, midi_path: str, output_path: str, 
                transpose: int = 0, 
                velocity_scale: float = 1.0,
                velocity_offset: int = 0,
                instrument_map: Optional[Dict[int, int]] = None) -> None:
        """
        Convert MIDI to audio with optional symbolic manipulations
        
        Args:
            midi_path: Path to input MIDI file
            output_path: Path for output audio file (mp3 or wav)
            transpose: Number of semitones to transpose (positive or negative)
            velocity_scale: Scale factor for note velocities
            velocity_offset: Offset to add to note velocities
            instrument_map: Dictionary mapping original program numbers to new ones
        """
        # Load MIDI
        midi_data = self.midi_processor.load_midi(midi_path)
        
        # Apply symbolic manipulations
        if transpose != 0:
            midi_data = self.midi_processor.transpose(midi_data, transpose)
        
        if velocity_scale != 1.0 or velocity_offset != 0:
            midi_data = self.midi_processor.adjust_velocity(midi_data, velocity_scale, velocity_offset)
        
        if instrument_map is not None:
            midi_data = self.midi_processor.reassign_instruments(midi_data, instrument_map)
        
        # Render to audio
        self.synthesizer.render_midi(midi_data, output_path)
        
        # Convert to MP3 if needed
        if output_path.endswith('.mp3') and not os.path.exists(output_path):
            import subprocess
            wav_path = output_path.replace('.mp3', '.wav')
            subprocess.call(['ffmpeg', '-i', wav_path, '-codec:a', 'libmp3lame', '-qscale:a', '2', output_path])
            os.remove(wav_path)


# Example usage
if __name__ == "__main__":
    # Example for FluidSynth backend
    converter = MIDIToAudioConverter(
        synthesis_backend="fluidsynth",
        soundfont_path="/path/to/soundfont.sf2"
    )
    
    # Convert a MIDI file with symbolic manipulation
    converter.convert(
        midi_path="input.mid",
        output_path="output.mp3",
        transpose=2,  # Transpose up 2 semitones
        velocity_scale=1.2,  # Increase velocity by 20%
        instrument_map={0: 24, 25: 32}  # Map piano (0) to guitar (24) and acoustic bass (25) to bass (32)
    )
    
    # Example for differentiable synthesis backend
    diff_converter = MIDIToAudioConverter(
        synthesis_backend="differentiable",
        model_path="trained_synth_model.pth"  # Optional pretrained model
    )
    
    diff_converter.convert(
        midi_path="input.mid",
        output_path="output_neural.mp3"
    )