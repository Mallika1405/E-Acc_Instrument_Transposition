"""
Differentiable Rendering Module for MIDI to Audio Conversion
===========================================================
This module implements a proper differentiable rendering approach for
MIDI to audio conversion, using gradient-based optimization to match
perceptual features of reference audio.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pretty_midi
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Union

class PhysicalModelingOscillator(nn.Module):
    """
    Differentiable oscillator for physical modeling synthesis
    """
    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Learnable parameters for the oscillator
        self.harmonic_weights = nn.Parameter(torch.zeros(16))
        self.attack_time = nn.Parameter(torch.tensor(0.01))
        self.decay_time = nn.Parameter(torch.tensor(0.1))
        self.sustain_level = nn.Parameter(torch.tensor(0.7))
        self.release_time = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, frequency: torch.Tensor, duration: torch.Tensor, 
               velocity: torch.Tensor) -> torch.Tensor:
        """
        Generate audio for a single note
        
        Args:
            frequency: Frequency in Hz
            duration: Note duration in seconds
            velocity: Note velocity (0-1)
        
        Returns:
            Waveform as a tensor
        """
        batch_size = frequency.shape[0]
        
        # Calculate number of samples
        num_samples = (duration * self.sample_rate).long()
        max_samples = num_samples.max().item()
        
        # Create time vector
        time = torch.arange(max_samples, device=frequency.device).float() / self.sample_rate
        time = time.unsqueeze(0).expand(batch_size, -1)
        
        # Generate fundamental frequency
        phase = 2 * np.pi * frequency.unsqueeze(-1) * time
        waveform = torch.sin(phase)
        
        # Add harmonics
        for i in range(2, len(self.harmonic_weights) + 2):
            harmonic_weight = F.softplus(self.harmonic_weights[i-2])
            harmonic = harmonic_weight * torch.sin(phase * i)
            waveform = waveform + harmonic
        
        # Normalize
        waveform = waveform / (1.0 + torch.sum(F.softplus(self.harmonic_weights)))
        
        # Apply ADSR envelope
        attack_samples = (F.softplus(self.attack_time) * self.sample_rate).long()
        decay_samples = (F.softplus(self.decay_time) * self.sample_rate).long()
        release_samples = (F.softplus(self.release_time) * self.sample_rate).long()
        sustain_level = torch.sigmoid(self.sustain_level)
        
        # Create envelope
        envelope = torch.zeros_like(time)
        
        # Attack phase
        attack_mask = time < F.softplus(self.attack_time).unsqueeze(-1)
        envelope = torch.where(
            attack_mask,
            time / F.softplus(self.attack_time).unsqueeze(-1),
            envelope
        )
        
        # Decay phase
        decay_start = F.softplus(self.attack_time).unsqueeze(-1)
        decay_end = decay_start + F.softplus(self.decay_time).unsqueeze(-1)
        decay_mask = (time >= decay_start) & (time < decay_end)
        decay_time = (time - decay_start) / F.softplus(self.decay_time).unsqueeze(-1)
        envelope = torch.where(
            decay_mask,
            1.0 + (sustain_level - 1.0) * decay_time,
            envelope
        )
        
        # Sustain phase
        sustain_mask = (time >= decay_end) & (time < (duration.unsqueeze(-1) - F.softplus(self.release_time).unsqueeze(-1)))
        envelope = torch.where(
            sustain_mask,
            sustain_level,
            envelope
        )
        
        # Release phase
        release_start = duration.unsqueeze(-1) - F.softplus(self.release_time).unsqueeze(-1)
        release_mask = (time >= release_start) & (time < duration.unsqueeze(-1))
        release_time = (time - release_start) / F.softplus(self.release_time).unsqueeze(-1)
        envelope = torch.where(
            release_mask,
            sustain_level * (1.0 - release_time),
            envelope
        )
        
        # Apply envelope and velocity
        output = waveform * envelope * velocity.unsqueeze(-1)
        
        # Create a mask for valid samples for each note
        mask = time < duration.unsqueeze(-1)
        output = output * mask.float()
        
        return output


class InstrumentModel(nn.Module):
    """
    Differentiable instrument model with learnable parameters
    """
    def __init__(self, 
                sample_rate: int = 44100,
                num_oscillators: int = 4):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.num_oscillators = num_oscillators
        
        # Create multiple oscillators
        self.oscillators = nn.ModuleList([
            PhysicalModelingOscillator(sample_rate) 
            for _ in range(num_oscillators)
        ])
        
        # Oscillator mixing weights
        self.mixing_weights = nn.Parameter(torch.ones(num_oscillators) / num_oscillators)
        
        # Frequency modulation parameters
        self.freq_mod_amount = nn.Parameter(torch.zeros(num_oscillators))
        self.freq_mod_rate = nn.Parameter(torch.ones(num_oscillators))
        
        # Filter parameters
        self.filter_cutoff = nn.Parameter(torch.tensor(0.5))  # Normalized cutoff
        self.filter_resonance = nn.Parameter(torch.tensor(0.1))
        
        # Effects parameters
        self.reverb_amount = nn.Parameter(torch.tensor(0.2))
        self.reverb_time = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, frequency: torch.Tensor, duration: torch.Tensor, 
               velocity: torch.Tensor) -> torch.Tensor:
        """
        Generate audio using the instrument model
        
        Args:
            frequency: Base frequency in Hz
            duration: Note duration in seconds
            velocity: Note velocity (0-1)
        
        Returns:
            Waveform as a tensor
        """
        batch_size = frequency.shape[0]
        
        # Calculate max duration for output size
        max_duration = duration.max().item()
        max_samples = int(max_duration * self.sample_rate) + 1
        
        # Output tensor
        output = torch.zeros(batch_size, max_samples, device=frequency.device)
        
        # Generate sound from each oscillator
        for i, oscillator in enumerate(self.oscillators):
            # Apply frequency modulation
            mod_amount = torch.sigmoid(self.freq_mod_amount[i])
            mod_rate = F.softplus(self.freq_mod_rate[i])
            
            # Create modulated frequency for this oscillator
            if mod_amount > 0.01:  # Only apply if modulation is significant
                # Create modulation over time
                time = torch.arange(max_samples, device=frequency.device).float() / self.sample_rate
                time = time.unsqueeze(0).expand(batch_size, -1)
                
                # Modulated frequency
                mod_freq = frequency.unsqueeze(-1) * (
                    1.0 + mod_amount * torch.sin(2 * np.pi * mod_rate * time)
                )
                
                # Generate from first time point
                osc_freq = mod_freq[:, 0]
            else:
                osc_freq = frequency
            
            # Generate from oscillator
            osc_output = oscillator(osc_freq, duration, velocity)
            
            # Mix into output with weight
            weight = F.softplus(self.mixing_weights[i])
            output = output + weight * osc_output
        
        # Normalize by sum of weights
        output = output / (torch.sum(F.softplus(self.mixing_weights)) + 1e-8)
        
        # Apply simple differentiable filter (lowpass)
        # In a full implementation, this would be a more sophisticated IIR filter
        cutoff = torch.sigmoid(self.filter_cutoff)  # 0-1
        if cutoff < 0.99:  # Only apply filtering if cutoff is not at maximum
            filter_size = 31
            t = torch.arange(-(filter_size//2), filter_size//2 + 1, device=output.device).float()
            cutoff_freq = cutoff * self.sample_rate / 2  # Convert to Hz
            sinc = torch.sin(2 * np.pi * cutoff_freq * t / self.sample_rate) / (np.pi * t)
            sinc[filter_size//2] = 2 * cutoff_freq / self.sample_rate  # Handle division by zero
            
            # Apply window
            window = 0.5 * (1 - torch.cos(2 * np.pi * torch.arange(filter_size, device=output.device) / filter_size))
            fir = sinc * window
            fir = fir / torch.sum(fir)  # Normalize
            
            # Apply filter
            output = F.pad(output, (filter_size//2, filter_size//2))
            output = F.conv1d(output.unsqueeze(1), fir.view(1, 1, -1), padding=0).squeeze(1)
        
        # Apply simple differentiable reverb
        reverb_amount = torch.sigmoid(self.reverb_amount)
        if reverb_amount > 0.01:  # Only apply if reverb is significant
            reverb_time_samples = int(F.softplus(self.reverb_time) * self.sample_rate)
            if reverb_time_samples > 1:
                # Create exponential decay
                decay = torch.exp(-torch.arange(reverb_time_samples, device=output.device) / (0.1 * reverb_time_samples))
                
                # Convolve with output (simplified)
                reverb_output = F.conv1d(
                    output.unsqueeze(1), 
                    decay.view(1, 1, -1), 
                    padding=reverb_time_samples
                ).squeeze(1)
                reverb_output = reverb_output[:, :max_samples]
                
                # Mix with dry signal
                output = (1 - reverb_amount) * output + reverb_amount * reverb_output
        
        return output


class DifferentiableRenderer(nn.Module):
    """
    End-to-end differentiable renderer for MIDI to audio conversion
    """
    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        
        self.sample_rate = sample_rate
        
        # Create instrument models for different program numbers
        # For simplicity, grouping into instrument families
        self.instrument_families = {
            "piano": (0, 7),           # Piano family
            "chromatic": (8, 15),      # Chromatic percussion
            "organ": (16, 23),         # Organ
            "guitar": (24, 31),        # Guitar
            "bass": (32, 39),          # Bass
            "strings": (40, 47),       # Strings
            "ensemble": (48, 55),      # Ensemble
            "brass": (56, 63),         # Brass
            "reed": (64, 71),          # Reed
            "pipe": (72, 79),          # Pipe
            "synth_lead": (80, 87),    # Synth Lead
            "synth_pad": (88, 95),     # Synth Pad
            "synth_effects": (96, 103), # Synth Effects
            "ethnic": (104, 111),      # Ethnic
            "percussive": (112, 119),  # Percussive
            "effects": (120, 127)      # Sound Effects
        }
        
        # Create instrument models for each family
        self.instrument_models = nn.ModuleDict({
            family: InstrumentModel(sample_rate)
            for family in self.instrument_families.keys()
        })
        
        # For mel spectrogram computation
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.register_buffer('mel_basis', torch.from_numpy(
            librosa.filters.mel(sr=sample_rate, n_fft=self.n_fft, n_mels=self.n_mels)).float())
    
    def get_instrument_model(self, program: int) -> InstrumentModel:
        """
        Get the appropriate instrument model for a given program number
        """
        for family, (start, end) in self.instrument_families.items():
            if start <= program <= end:
                return self.instrument_models[family]
        
        # Default to piano if not found
        return self.instrument_models["piano"]
    
    def render_note(self, pitch: int, velocity: int, 
                   start: float, end: float, program: int) -> Tuple[torch.Tensor, int]:
        """
        Render a single note
        
        Args:
            pitch: MIDI pitch (0-127)
            velocity: MIDI velocity (0-127)
            start: Start time in seconds
            end: End time in seconds
            program: MIDI program number (0-127)
        
        Returns:
            Tuple of (waveform, start_sample)
        """
        # Convert pitch to frequency
        frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        frequency = torch.tensor([frequency], dtype=torch.float32)
        
        # Convert velocity to amplitude (0-1)
        velocity = torch.tensor([velocity / 127.0], dtype=torch.float32)
        
        # Calculate duration
        duration = torch.tensor([end - start], dtype=torch.float32)
        
        # Get the appropriate instrument model
        instrument = self.get_instrument_model(program)
        
        # Generate audio
        audio = instrument(frequency, duration, velocity)
        
        # Calculate start sample
        start_sample = int(start * self.sample_rate)
        
        return audio[0], start_sample
    
    def render_midi(self, midi_data: pretty_midi.PrettyMIDI, 
                   target_audio: Optional[torch.Tensor] = None,
                   optimize_steps: int = 0,
                   learning_rate: float = 0.001) -> torch.Tensor:
        """
        Render a complete MIDI file to audio
        
        Args:
            midi_data: MIDI data to render
            target_audio: Optional target audio for optimization
            optimize_steps: Number of optimization steps (0 for no optimization)
            learning_rate: Learning rate for optimization
        
        Returns:
            Rendered audio as a tensor
        """
        # Calculate total duration
        duration = midi_data.get_end_time()
        num_samples = int(duration * self.sample_rate) + 1
        
        # Initialize output audio array
        output_audio = torch.zeros(num_samples)
        
        # Process each instrument
        for instrument in midi_data.instruments:
            program = instrument.program
            
            # Process each note
            for note in instrument.notes:
                # Render this note
                note_audio, start_sample = self.render_note(
                    note.pitch, note.velocity, note.start, note.end, program
                )
                
                # Add to output at the correct position
                end_sample = start_sample + len(note_audio)
                if end_sample <= num_samples:
                    output_audio[start_sample:end_sample] += note_audio
        
        # Normalize audio
        max_val = torch.max(torch.abs(output_audio))
        if max_val > 0:
            output_audio = output_audio / max_val * 0.9
        
        # If target audio is provided and optimize_steps > 0, perform optimization
        if target_audio is not None and optimize_steps > 0:
            optimized_audio = self.optimize_rendering(
                midi_data, target_audio, optimize_steps, learning_rate
            )
            return optimized_audio
        
        return output_audio
    
    def compute_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram from audio waveform
        """
        # Ensure audio is on the correct device
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        
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
        mel_spec = torch.matmul(self.mel_basis.to(audio.device), power_spec)
        
        # Log scale
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        
        return log_mel_spec
    
    def mel_spectrogram_loss(self, generated_audio: torch.Tensor, 
                            target_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between generated and target audio using mel-spectrograms
        """
        # Ensure same length
        min_len = min(len(generated_audio), len(target_audio))
        generated_audio = generated_audio[:min_len]
        target_audio = target_audio[:min_len]
        
        # Compute mel spectrograms
        gen_mel = self.compute_mel_spectrogram(generated_audio)
        target_mel = self.compute_mel_spectrogram(target_audio)
        
        # Mean squared error between mel-spectrograms
        loss = F.mse_loss(gen_mel, target_mel)
        
        return loss
    
    def optimize_rendering(self, midi_data: pretty_midi.PrettyMIDI,
                          target_audio: torch.Tensor,
                          steps: int,
                          learning_rate: float) -> torch.Tensor:
        """
        Optimize instrument parameters to match target audio
        
        Args:
            midi_data: MIDI data to render
            target_audio: Target audio to match
            steps: Number of optimization steps
            learning_rate: Learning rate for optimization
        
        Returns:
            Optimized audio as a tensor
        """
        # Set all parameters to require gradients
        for param in self.parameters():
            param.requires_grad = True
        
        # Initialize optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Optimization loop
        best_audio = None
        best_loss = float('inf')
        
        for step in range(steps):
            # Generate audio
            generated_audio = self.render_midi(midi_data)
            
            # Compute loss
            loss = self.mel_spectrogram_loss(generated_audio, target_audio)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Keep track of best result
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_audio = generated_audio.detach().clone()
            
            # Print progress
            if step % 10 == 0:
                print(f"Step {step}/{steps}, Loss: {current_loss:.6f}")
        
        # Return best audio found during optimization
        if best_audio is None:
            best_audio = generated_audio.detach()
        
        return best_audio
    
    def save_audio(self, audio: torch.Tensor, output_path: str):
        """
        Save audio tensor to file
        """
        # Convert to numpy array
        audio_np = audio.detach().cpu().numpy()
        
        # Save to file
        sf.write(output_path, audio_np, self.sample_rate)


class DifferentiableRendererWithExamples(DifferentiableRenderer):
    """
    Extended differentiable renderer that can learn from examples
    """
    def __init__(self, sample_rate: int = 44100, example_dir: Optional[str] = None):
        super().__init__(sample_rate)
        
        # Load example audio if provided
        self.examples = {}
        if example_dir and os.path.isdir(example_dir):
            self.load_examples(example_dir)
    
    def load_examples(self, example_dir: str):
        """
        Load example audio files for reference
        
        Expected directory structure:
        /example_dir/
            /piano/
                /note_60_vel_80.wav  # Middle C, velocity 80
                /note_67_vel_100.wav # G, velocity 100
            /violin/
                ...
        """
        for instrument_dir in os.listdir(example_dir):
            instrument_path = os.path.join(example_dir, instrument_dir)
            
            if os.path.isdir(instrument_path):
                # Map instrument name to program number range
                program_start = -1
                for family, (start, end) in self.instrument_families.items():
                    if family.lower() == instrument_dir.lower():
                        program_start = start
                        break
                
                if program_start >= 0:
                    # Load example files for this instrument
                    for file_name in os.listdir(instrument_path):
                        if file_name.endswith('.wav') or file_name.endswith('.mp3'):
                            # Parse pitch and velocity from filename
                            # Expected format: note_60_vel_80.wav
                            parts = file_name.split('_')
                            if len(parts) >= 4 and parts[0] == 'note' and parts[2] == 'vel':
                                try:
                                    pitch = int(parts[1])
                                    velocity = int(parts[3].split('.')[0])
                                    
                                    # Load audio
                                    audio_path = os.path.join(instrument_path, file_name)
                                    audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                                    
                                    # Store example
                                    key = (program_start, pitch, velocity)
                                    self.examples[key] = torch.tensor(audio, dtype=torch.float32)
                                    
                                    print(f"Loaded example: {instrument_dir}, pitch={pitch}, velocity={velocity}")
                                except (ValueError, IndexError):
                                    continue
    
    def get_closest_example(self, program: int, pitch: int, velocity: int) -> Optional[torch.Tensor]:
        """
        Get the closest matching example audio
        """
        # Find program family start
        program_start = -1
        for family, (start, end) in self.instrument_families.items():
            if start <= program <= end:
                program_start = start
                break
        
        if program_start < 0:
            return None
        
        # Exact match
        key = (program_start, pitch, velocity)
        if key in self.examples:
            return self.examples[key]
        
        # Find closest pitch within same program and similar velocity
        closest_pitch_diff = float('inf')
        closest_velocity_diff = float('inf')
        closest_example = None
        
        for (ex_program, ex_pitch, ex_velocity), audio in self.examples.items():
            if ex_program == program_start:
                pitch_diff = abs(ex_pitch - pitch)
                velocity_diff = abs(ex_velocity - velocity)
                
                # Prioritize pitch similarity over velocity
                if pitch_diff < closest_pitch_diff or (pitch_diff == closest_pitch_diff and velocity_diff < closest_velocity_diff):
                    closest_pitch_diff = pitch_diff
                    closest_velocity_diff = velocity_diff
                    closest_example = audio
        
        return closest_example
    
    def render_note_with_example(self, pitch: int, velocity: int, 
                               start: float, end: float, program: int,
                               optimize_steps: int = 100) -> Tuple[torch.Tensor, int]:
        """
        Render a note using example audio as reference
        """
        # Get closest example
        example = self.get_closest_example(program, pitch, velocity)
        
        if example is not None and optimize_steps > 0:
            # Use differentiable rendering to match the example
            # Convert pitch to frequency
            frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
            frequency = torch.tensor([frequency], dtype=torch.float32)
            
            # Convert velocity to amplitude (0-1)
            velocity_norm = torch.tensor([velocity / 127.0], dtype=torch.float32)
            
            # Calculate duration from example
            example_duration = len(example) / self.sample_rate
            target_duration = end - start
            duration = torch.tensor([min(example_duration, target_duration)], dtype=torch.float32)
            
            # Get the appropriate instrument model
            instrument = self.get_instrument_model(program)
            
            # Optimize instrument parameters to match example
            original_params = {}
            for name, param in instrument.named_parameters():
                original_params[name] = param.data.clone()
            
            # Optimize
            optimizer = optim.Adam(instrument.parameters(), lr=0.001)
            
            for step in range(optimize_steps):
                # Generate audio
                audio = instrument(frequency, duration, velocity_norm)
                
                # Adjust example length to match generated audio
                target = example[:len(audio[0])]
                if len(target) < len(audio[0]):
                    target = torch.nn.functional.pad(target, (0, len(audio[0]) - len(target)))
                
                # Compute loss
                loss = F.mse_loss(audio[0], target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Generate final audio with optimized parameters
            audio = instrument(frequency, torch.tensor([target_duration]), velocity_norm)
            
            # Reset parameters to original values
            for name, param in instrument.named_parameters():
                param.data = original_params[name]
            
            # Calculate start sample
            start_sample = int(start * self.sample_rate)
            
            return audio[0], start_sample
        else:
            # Fall back to regular rendering
            return super().render_note(pitch, velocity, start, end, program)
    
    def render_midi_with_examples(self, midi_data: pretty_midi.PrettyMIDI,
                                 optimize_steps: int = 100) -> torch.Tensor:
        """
        Render a complete MIDI file using examples for reference
        """
        # Calculate total duration
        duration = midi_data.get_end_time()
        num_samples = int(duration * self.sample_rate) + 1
        
        # Initialize output audio array
        output_audio = torch.zeros(num_samples)
        
        # Process each instrument
        for instrument in midi_data.instruments:
            program = instrument.program
            
            # Process each note
            for note in instrument.notes:
                # Render this note using examples
                note_audio, start_sample = self.render_note_with_example(
                    note.pitch, note.velocity, note.start, note.end, program,
                    optimize_steps=optimize_steps
                )
                
                # Add to output at the correct position
                end_sample = start_sample + len(note_audio)
                if end_sample <= num_samples:
                    output_audio[start_sample:end_sample] += note_audio
        
        # Normalize audio
        max_val = torch.max(torch.abs(output_audio))
        if max_val > 0:
            output_audio = output_audio / max_val * 0.9
        
        return output_audio


# Example usage
if __name__ == "__main__":
    # Basic usage
    renderer = DifferentiableRenderer(sample_rate=44100)
    
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI("input.mid")
    
    # Render to audio
    audio = renderer.render_midi(midi_data)
    
    # Save to file
    renderer.save_audio(audio, "output.wav")
    
    # Advanced usage with examples
    example_renderer = DifferentiableRendererWithExamples(
        sample_rate=44100,
        example_dir="instrument_samples"
    )
    
    # Render with examples
    audio_with_examples = example_renderer.render_midi_with_examples(
        midi_data,
        optimize_steps=50
    )
    
    # Save to file
    example_renderer.save_audio(audio_with_examples, "output_with_examples.wav")