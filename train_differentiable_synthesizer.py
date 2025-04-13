"""
Training Script for the Differentiable Synthesizer
=================================================
This script handles data preparation and training for the neural network-based
synthesizer that learns to generate audio from MIDI notes.
"""

import os
import torch
import numpy as np
import pretty_midi
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import glob
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the differentiable renderer
from differentiable_renderer import DifferentiableRenderer, PhysicalModelingOscillator, InstrumentModel


class InstrumentSampleDataset(Dataset):
    """
    Dataset for training the differentiable synthesizer with real instrument samples
    """
    def __init__(self, sample_dir: str, sample_rate: int = 44100, duration: float = 1.0):
        """
        Initialize the dataset
        
        Args:
            sample_dir: Directory containing instrument samples
                        Expected structure: /sample_dir/instrument_name/note_[pitch].wav
            sample_rate: Audio sample rate
            duration: Target duration for audio samples (in seconds)
        """
        self.sample_rate = sample_rate
        self.target_samples = int(duration * sample_rate)
        self.samples = []
        
        # Map instrument names to MIDI program numbers
        self.instrument_map = {
            'piano': 0,
            'acoustic_guitar': 24,
            'electric_guitar': 27,
            'bass': 33,
            'violin': 40,
            'cello': 42,
            'trumpet': 56,
            'saxophone': 65,
            'flute': 73,
            'synth_lead': 80,
            'synth_pad': 88,
            'drums': 128  # Special case for drums
        }
        
        print(f"Loading samples from {sample_dir}")
        
        # Load samples
        for instrument_dir in glob.glob(os.path.join(sample_dir, "*")):
            instrument_name = os.path.basename(instrument_dir)
            
            if instrument_name in self.instrument_map:
                program = self.instrument_map[instrument_name]
                print(f"Processing {instrument_name} (program {program})")
                
                # Find all audio samples for this instrument
                for sample_path in glob.glob(os.path.join(instrument_dir, "*.wav")):
                    # Extract pitch from filename (assuming format like "note_60.wav" for middle C)
                    filename = os.path.basename(sample_path)
                    if 'note_' in filename:
                        try:
                            pitch = int(filename.split('note_')[1].split('.')[0])
                            
                            # Only use samples within valid MIDI pitch range
                            if 0 <= pitch <= 127:
                                print(f"  Found sample for pitch {pitch}: {filename}")
                                
                                # For each sample, we'll create entries with various velocities
                                for velocity in [40, 60, 80, 100, 120]:
                                    self.samples.append({
                                        'path': sample_path,
                                        'pitch': pitch,
                                        'velocity': velocity,
                                        'program': program
                                    })
                        except ValueError:
                            continue
        
        print(f"Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        audio, _ = librosa.load(sample['path'], sr=self.sample_rate, mono=True)
        
        # Scale amplitude based on velocity
        normalized_velocity = sample['velocity'] / 127.0
        audio = audio * normalized_velocity
        
        # Ensure consistent length
        if len(audio) > self.target_samples:
            audio = audio[:self.target_samples]
        else:
            audio = np.pad(audio, (0, self.target_samples - len(audio)))
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Create note descriptors
        frequency = 440.0 * (2.0 ** ((sample['pitch'] - 69) / 12.0))
        velocity = sample['velocity'] / 127.0
        duration = 1.0  # 1 second fixed duration for training
        
        # Return tensors
        return {
            'frequency': torch.tensor([frequency], dtype=torch.float32),
            'velocity': torch.tensor([velocity], dtype=torch.float32),
            'duration': torch.tensor([duration], dtype=torch.float32),
            'program': torch.tensor([sample['program']], dtype=torch.long),
            'audio': audio_tensor
        }


def train_oscillator_model(
    sample_dir: str,
    output_model_path: str,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 3e-4,
    sample_rate: int = 44100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train an oscillator model
    
    Args:
        sample_dir: Directory containing instrument samples
        output_model_path: Path to save the trained model
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        sample_rate: Audio sample rate
        device: Device to use for training ('cuda' or 'cpu')
    """
    # Create dataset and dataloader
    dataset = InstrumentSampleDataset(sample_dir, sample_rate=sample_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize oscillator model
    model = PhysicalModelingOscillator(sample_rate=sample_rate)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function (mean squared error)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Move data to device
            frequency = batch['frequency'].to(device)
            velocity = batch['velocity'].to(device)
            duration = batch['duration'].to(device)
            target_audio = batch['audio'].to(device)
            
            # Forward pass
            output_audio = model(frequency, duration, velocity)
            
            # Compute loss
            loss = criterion(output_audio, target_audio)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
        
        # Save model periodically
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{output_model_path}.e{epoch+1}")
    
    # Save final model
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f"{output_model_path}.loss.png")


def train_instrument_model(
    sample_dir: str,
    output_model_path: str,
    instrument_name: str,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 3e-4,
    sample_rate: int = 44100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train a specific instrument model
    
    Args:
        sample_dir: Directory containing instrument samples
        output_model_path: Path to save the trained model
        instrument_name: Name of the instrument to train
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        sample_rate: Audio sample rate
        device: Device to use for training ('cuda' or 'cpu')
    """
    # Create dataset with only samples from the specified instrument
    instrument_dir = os.path.join(sample_dir, instrument_name)
    if not os.path.exists(instrument_dir):
        raise ValueError(f"Instrument directory not found: {instrument_dir}")
    
    # Create a custom dataset for this instrument
    class SingleInstrumentDataset(Dataset):
        def __init__(self, instrument_dir, sample_rate=44100):
            self.sample_rate = sample_rate
            self.samples = []
            
            # Find all audio samples for this instrument
            for sample_path in glob.glob(os.path.join(instrument_dir, "*.wav")):
                # Extract pitch from filename
                filename = os.path.basename(sample_path)
                if 'note_' in filename:
                    try:
                        pitch = int(filename.split('note_')[1].split('.')[0])
                        
                        # Only use samples within valid MIDI pitch range
                        if 0 <= pitch <= 127:
                            # For each sample, create entries with various velocities
                            for velocity in [40, 60, 80, 100, 120]:
                                self.samples.append({
                                    'path': sample_path,
                                    'pitch': pitch,
                                    'velocity': velocity
                                })
                    except ValueError:
                        continue
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            # Load audio
            audio, _ = librosa.load(sample['path'], sr=self.sample_rate, mono=True)
            
            # Scale amplitude based on velocity
            normalized_velocity = sample['velocity'] / 127.0
            audio = audio * normalized_velocity
            
            # Ensure consistent length (1 second)
            target_samples = self.sample_rate
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            else:
                audio = np.pad(audio, (0, target_samples - len(audio)))
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Create note descriptors
            frequency = 440.0 * (2.0 ** ((sample['pitch'] - 69) / 12.0))
            velocity = sample['velocity'] / 127.0
            duration = 1.0
            
            return {
                'frequency': torch.tensor([frequency], dtype=torch.float32),
                'velocity': torch.tensor([velocity], dtype=torch.float32),
                'duration': torch.tensor([duration], dtype=torch.float32),
                'audio': audio_tensor
            }
    
    # Create dataset and dataloader
    dataset = SingleInstrumentDataset(instrument_dir, sample_rate=sample_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize instrument model
    model = InstrumentModel(sample_rate=sample_rate)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function (mean squared error)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Move data to device
            frequency = batch['frequency'].to(device)
            velocity = batch['velocity'].to(device)
            duration = batch['duration'].to(device)
            target_audio = batch['audio'].to(device)
            
            # Forward pass
            output_audio = model(frequency, duration, velocity)
            
            # Compute loss
            loss = criterion(output_audio, target_audio)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
        
        # Save model periodically
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{output_model_path}.e{epoch+1}")
    
    # Save final model
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(f'Training Loss - {instrument_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f"{output_model_path}.loss.png")


def train_full_renderer(
    sample_dir: str,
    output_model_path: str,
    batch_size: int = 16,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    sample_rate: int = 44100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the full differentiable renderer with all instruments
    
    Args:
        sample_dir: Directory containing instrument samples
        output_model_path: Path to save the trained model
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        sample_rate: Audio sample rate
        device: Device to use for training ('cuda' or 'cpu')
    """
    # Create dataset and dataloader
    dataset = InstrumentSampleDataset(sample_dir, sample_rate=sample_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize full renderer model
    model = DifferentiableRenderer(sample_rate=sample_rate)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Move data to device
            frequency = batch['frequency'].to(device)
            velocity = batch['velocity'].to(device)
            duration = batch['duration'].to(device)
            program = batch['program'].to(device)
            target_audio = batch['audio'].to(device)
            
            # Create mini MIDI data for rendering
            notes = []
            for i in range(len(frequency)):
                # Create a note tuple with pitch, velocity, start time, end time, program
                pitch = librosa.hz_to_midi(frequency[i].item())
                vel = int(velocity[i].item() * 127)
                start = 0.0
                end = duration[i].item()
                prog = program[i].item()
                
                notes.append((pitch, vel, start, end, prog))
            
            # Forward pass - directly render these notes
            output_audio = []
            for note in notes:
                pitch, vel, start, end, prog = note
                note_audio, _ = model.render_note(pitch, vel, start, end, prog)
                output_audio.append(note_audio)
            
            # Stack into a batch
            output_tensor = torch.stack(output_audio, dim=0)
            
            # Compute loss directly without mel spectrogram for simplicity
            # In practice, you might want to use mel spectrogram loss
            loss = torch.nn.functional.mse_loss(output_tensor, target_audio)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
        
        # Save model periodically
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{output_model_path}.e{epoch+1}")
            
            # Generate samples for evaluation
            generate_test_samples(model, f"samples_epoch_{epoch+1}", device)
    
    # Save final model
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss - Full Renderer')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f"{output_model_path}.loss.png")


def generate_test_samples(model, output_dir, device):
    """
    Generate test audio samples using the trained model
    
    Args:
        model: Trained model
        output_dir: Directory to save generated samples
        device: Device to run the model on
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Generate samples for different instruments and pitches
        instruments = [0, 24, 40, 56, 73]  # Piano, Guitar, Violin, Trumpet, Flute
        pitches = [60, 64, 67, 72]  # C4, E4, G4, C5
        
        for program in instruments:
            for pitch in pitches:
                # Generate a note
                audio, _ = model.render_note(
                    pitch=pitch,
                    velocity=80,
                    start=0.0,
                    end=2.0,
                    program=program
                )
                
                # Convert to numpy and save
                audio_np = audio.cpu().numpy()
                
                # Normalize
                audio_np = audio_np / np.max(np.abs(audio_np)) * 0.9
                
                # Save to file
                filename = f"prog{program}_pitch{pitch}.wav"
                sf.write(os.path.join(output_dir, filename), audio_np, model.sample_rate)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train the differentiable synthesizer")
    
    parser.add_argument("--sample-dir", required=True, help="Directory containing instrument samples")
    parser.add_argument("--output-model", required=True, help="Path to save the trained model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate")
    parser.add_argument("--instrument", help="Train model for specific instrument")
    parser.add_argument("--mode", choices=["oscillator", "instrument", "full"], default="full",
                       help="Training mode: oscillator, instrument, or full renderer")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Choose training mode
    if args.mode == "oscillator":
        train_oscillator_model(
            sample_dir=args.sample_dir,
            output_model_path=args.output_model,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            sample_rate=args.sample_rate,
            device=device
        )
    elif args.mode == "instrument":
        if not args.instrument:
            print("Error: --instrument must be specified for instrument mode")
            exit(1)
        
        train_instrument_model(
            sample_dir=args.sample_dir,
            output_model_path=args.output_model,
            instrument_name=args.instrument,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            sample_rate=args.sample_rate,
            device=device
        )
    else:  # full renderer
        train_full_renderer(
            sample_dir=args.sample_dir,
            output_model_path=args.output_model,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            sample_rate=args.sample_rate,
            device=device
        )
    
    print("Training complete!")