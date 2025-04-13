"""
Integration module to connect the improved differentiable renderer
with the main MIDI to Audio conversion system
"""

import os
import torch
import numpy as np
import pretty_midi
import soundfile as sf
from typing import Dict, Optional

# Import the main converter and the improved differentiable renderer
from midi_to_audio_converter import MIDIProcessor
from differentiable_renderer import DifferentiableRenderer, DifferentiableRendererWithExamples


class DifferentiableRenderingBackend:
    """
    Synthesis backend using the differentiable renderer
    """
    def __init__(self, 
                sample_rate: int = 44100, 
                model_path: Optional[str] = None,
                example_dir: Optional[str] = None,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the differentiable rendering backend
        
        Args:
            sample_rate: Audio sample rate
            model_path: Path to saved model weights (optional)
            example_dir: Directory containing reference audio samples (optional)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.sample_rate = sample_rate
        self.device = device
        
        # Initialize renderer based on whether examples are provided
        if example_dir and os.path.isdir(example_dir):
            self.renderer = DifferentiableRendererWithExamples(
                sample_rate=sample_rate,
                example_dir=example_dir
            )
            self.has_examples = True
        else:
            self.renderer = DifferentiableRenderer(sample_rate=sample_rate)
            self.has_examples = False
        
        # Move to device
        self.renderer = self.renderer.to(device)
        
        # Load saved model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load saved model weights
        """
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.renderer.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load model from {model_path}: {str(e)}")
    
    def save_model(self, model_path: str):
        """
        Save model weights
        """
        try:
            torch.save(self.renderer.state_dict(), model_path)
            print(f"Saved model to {model_path}")
        except Exception as e:
            print(f"Warning: Failed to save model to {model_path}: {str(e)}")
    
    def render_midi(self, midi_data: pretty_midi.PrettyMIDI, output_path: str,
                   target_audio: Optional[np.ndarray] = None,
                   optimize_steps: int = 0):
        """
        Render MIDI to audio using differentiable rendering
        
        Args:
            midi_data: MIDI data to render
            output_path: Path for output audio file
            target_audio: Optional target audio for optimization
            optimize_steps: Number of optimization steps (0 for no optimization)
        """
        # Convert target audio to tensor if provided
        target_tensor = None
        if target_audio is not None:
            target_tensor = torch.tensor(target_audio, dtype=torch.float32).to(self.device)
        
        # Render MIDI
        with torch.no_grad():
            if self.has_examples and optimize_steps > 0:
                # Use example-based rendering
                audio = self.renderer.render_midi_with_examples(
                    midi_data,
                    optimize_steps=optimize_steps
                )
            elif target_tensor is not None and optimize_steps > 0:
                # Use optimization to match target audio
                audio = self.renderer.render_midi(
                    midi_data,
                    target_audio=target_tensor,
                    optimize_steps=optimize_steps,
                    learning_rate=0.001
                )
            else:
                # Use regular rendering
                audio = self.renderer.render_midi(midi_data)
        
        # Save audio to file
        self.renderer.save_audio(audio, output_path)
    
    def fine_tune(self, midi_path: str, target_audio_path: str, 
                 steps: int = 1000, learning_rate: float = 0.001):
        """
        Fine-tune the model to better match a reference recording
        
        Args:
            midi_path: Path to MIDI file
            target_audio_path: Path to target audio file
            steps: Number of optimization steps
            learning_rate: Learning rate for optimization
        """
        # Load MIDI and target audio
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        target_audio, _ = sf.read(target_audio_path)
        
        # Convert to mono if stereo
        if len(target_audio.shape) > 1 and target_audio.shape[1] > 1:
            target_audio = np.mean(target_audio, axis=1)
        
        # Convert to float32 and normalize
        target_audio = target_audio.astype(np.float32)
        target_audio = target_audio / np.max(np.abs(target_audio))
        
        # Convert to tensor
        target_tensor = torch.tensor(target_audio).to(self.device)
        
        # Enable gradients for all parameters
        for param in self.renderer.parameters():
            param.requires_grad = True
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.renderer.parameters(), lr=learning_rate)
        
        # Fine-tuning loop
        for step in range(steps):
            # Generate audio
            audio = self.renderer.render_midi(midi_data)
            
            # Compute loss
            loss = self.renderer.mel_spectrogram_loss(audio, target_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print progress
            if step % 10 == 0:
                print(f"Step {step}/{steps}, Loss: {loss.item():.6f}")
        
        print(f"Fine-tuning completed after {steps} steps")


class IntegratedMIDIToAudioConverter:
    """
    Integrated converter that combines the MIDI processor with 
    the improved differentiable rendering backend
    """
    def __init__(self, 
                synthesis_backend: str = "fluidsynth",
                soundfont_path: Optional[str] = None,
                model_path: Optional[str] = None,
                example_dir: Optional[str] = None,
                sample_rate: int = 44100,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the integrated converter
        
        Args:
            synthesis_backend: Either "fluidsynth", "differentiable", or "hybrid"
            soundfont_path: Path to SoundFont file (required for FluidSynth backend)
            model_path: Path to differentiable model weights (optional)
            example_dir: Path to example audio directory (optional)
            sample_rate: Audio sample rate
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.midi_processor = MIDIProcessor()
        self.synthesis_backend = synthesis_backend
        self.sample_rate = sample_rate
        
        # Import backends only when needed to avoid circular imports
        if synthesis_backend in ["fluidsynth", "hybrid"]:
            try:
                from midi_to_audio_converter import FluidSynthBackend
                if soundfont_path is None:
                    raise ValueError("soundfont_path must be provided for FluidSynth backend")
                self.fluidsynth = FluidSynthBackend(soundfont_path, sample_rate)
            except ImportError:
                print("Warning: FluidSynth not available. Make sure FluidSynth is installed.")
                self.fluidsynth = None
        else:
            self.fluidsynth = None
        
        if synthesis_backend in ["differentiable", "hybrid"]:
            self.diff_renderer = DifferentiableRenderingBackend(
                sample_rate=sample_rate,
                model_path=model_path,
                example_dir=example_dir,
                device=device
            )
        else:
            self.diff_renderer = None
    
    def convert(self, midi_path: str, output_path: str, 
                transpose: int = 0, 
                velocity_scale: float = 1.0,
                velocity_offset: int = 0,
                instrument_map: Optional[Dict[int, int]] = None,
                optimize_steps: int = 0,
                target_audio_path: Optional[str] = None) -> None:
        """
        Convert MIDI to audio with optional symbolic manipulations
        
        Args:
            midi_path: Path to input MIDI file
            output_path: Path for output audio file (mp3 or wav)
            transpose: Number of semitones to transpose (positive or negative)
            velocity_scale: Scale factor for note velocities
            velocity_offset: Offset to add to note velocities
            instrument_map: Dictionary mapping original program numbers to new ones
            optimize_steps: Number of optimization steps for differentiable rendering
            target_audio_path: Path to target audio for optimization
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
        
        # Prepare target audio if provided
        target_audio = None
        if target_audio_path and os.path.exists(target_audio_path) and self.diff_renderer is not None:
            try:
                target_audio, _ = sf.read(target_audio_path)
                # Convert to mono if stereo
                if len(target_audio.shape) > 1 and target_audio.shape[1] > 1:
                    target_audio = np.mean(target_audio, axis=1)
                # Normalize
                target_audio = target_audio / np.max(np.abs(target_audio))
            except Exception as e:
                print(f"Warning: Failed to load target audio: {str(e)}")
        
        # Render to audio using appropriate backend
        if self.synthesis_backend == "fluidsynth" and self.fluidsynth is not None:
            self.fluidsynth.render_midi(midi_data, output_path)
        
        elif self.synthesis_backend == "differentiable" and self.diff_renderer is not None:
            self.diff_renderer.render_midi(
                midi_data, 
                output_path,
                target_audio=target_audio,
                optimize_steps=optimize_steps
            )
        
        elif self.synthesis_backend == "hybrid" and self.fluidsynth is not None and self.diff_renderer is not None:
            # For hybrid mode, render with both backends and mix the results
            
            # Create temporary output paths
            fluidsynth_output = output_path + ".fluid.wav"
            diff_output = output_path + ".diff.wav"
            
            # Render with both backends
            self.fluidsynth.render_midi(midi_data, fluidsynth_output)
            self.diff_renderer.render_midi(
                midi_data, 
                diff_output,
                target_audio=target_audio,
                optimize_steps=optimize_steps
            )
            
            # Load both outputs
            fluid_audio, _ = sf.read(fluidsynth_output)
            diff_audio, sr = sf.read(diff_output)
            
            # Convert to mono if needed
            if len(fluid_audio.shape) > 1 and fluid_audio.shape[1] > 1:
                fluid_audio = np.mean(fluid_audio, axis=1)
            if len(diff_audio.shape) > 1 and diff_audio.shape[1] > 1:
                diff_audio = np.mean(diff_audio, axis=1)
            
            # Ensure same length
            min_len = min(len(fluid_audio), len(diff_audio))
            fluid_audio = fluid_audio[:min_len]
            diff_audio = diff_audio[:min_len]
            
            # Mix the audio (equal weight)
            mixed_audio = 0.5 * fluid_audio + 0.5 * diff_audio
            
            # Normalize
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio)) * 0.9
            
            # Save mixed output
            sf.write(output_path, mixed_audio, sr)
            
            # Clean up temporary files
            try:
                os.remove(fluidsynth_output)
                os.remove(diff_output)
            except:
                pass
        
        else:
            raise ValueError(f"Invalid backend configuration: {self.synthesis_backend}")
        
        # Convert to MP3 if needed
        if output_path.endswith('.mp3') and not os.path.exists(output_path):
            import subprocess
            wav_path = output_path.replace('.mp3', '.wav')
            subprocess.call(['ffmpeg', '-i', wav_path, '-codec:a', 'libmp3lame', '-qscale:a', '2', output_path])
            os.remove(wav_path)


# Example usage
if __name__ == "__main__":
    # Example for differentiable rendering backend
    converter = IntegratedMIDIToAudioConverter(
        synthesis_backend="differentiable",
        model_path="trained_diff_model.pth",
        example_dir="instrument_samples"
    )
    
    # Convert a MIDI file with symbolic manipulation
    converter.convert(
        midi_path="input.mid",
        output_path="output_diff.mp3",
        transpose=2,
        velocity_scale=1.2,
        optimize_steps=50
    )
    
    # Example for hybrid backend
    hybrid_converter = IntegratedMIDIToAudioConverter(
        synthesis_backend="hybrid",
        soundfont_path="soundfont.sf2",
        model_path="trained_diff_model.pth"
    )
    
    # Convert with reference audio
    hybrid_converter.convert(
        midi_path="input.mid",
        output_path="output_hybrid.mp3",
        target_audio_path="reference.wav",
        optimize_steps=100
    )