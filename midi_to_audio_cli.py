"""
Command Line Interface for MIDI to Audio Converter
=================================================
A simple CLI tool for converting MIDI files to audio with symbolic manipulations
"""

import os
import sys
import argparse
import json
from typing import Dict, Optional
import time

# Import the converter
from midi_to_audio_converter import MIDIToAudioConverter


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert MIDI files to audio with symbolic manipulations")
    
    # Input/output files
    parser.add_argument("input", help="Input MIDI file path")
    parser.add_argument("-o", "--output", help="Output audio file path (default: input file with .mp3 extension)")
    
    # Synthesis backend
    parser.add_argument("-b", "--backend", choices=["fluidsynth", "differentiable"], default="fluidsynth",
                      help="Synthesis backend (default: fluidsynth)")
    parser.add_argument("-sf", "--soundfont", help="SoundFont file for FluidSynth backend")
    parser.add_argument("-m", "--model", help="Model file for Differentiable backend")
    
    # Symbolic manipulations
    parser.add_argument("-t", "--transpose", type=int, default=0,
                      help="Transpose by N semitones (default: 0)")
    parser.add_argument("-vs", "--velocity-scale", type=float, default=1.0,
                      help="Scale note velocities by factor (default: 1.0)")
    parser.add_argument("-vo", "--velocity-offset", type=int, default=0,
                      help="Add offset to note velocities (default: 0)")
    parser.add_argument("-im", "--instrument-map", help="JSON file with instrument mappings")
    
    # Additional options
    parser.add_argument("--sample-rate", type=int, default=44100,
                      help="Audio sample rate in Hz (default: 44100)")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Enable verbose output")
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments"""
    # Check input file
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return False
    
    # Set default output file if not specified
    if not args.output:
        base_name, _ = os.path.splitext(args.input)
        args.output = f"{base_name}.mp3"
    
    # Check backend requirements
    if args.backend == "fluidsynth" and not args.soundfont:
        print("Error: FluidSynth backend requires a SoundFont file (--soundfont)")
        return False
    
    if args.backend == "differentiable" and not args.model:
        print("Error: Differentiable backend requires a model file (--model)")
        return False
    
    # Check if SoundFont file exists
    if args.soundfont and not os.path.isfile(args.soundfont):
        print(f"Error: SoundFont file '{args.soundfont}' does not exist")
        return False
    
    # Check if model file exists
    if args.model and not os.path.isfile(args.model):
        print(f"Error: Model file '{args.model}' does not exist")
        return False
    
    # Check if instrument map file exists
    if args.instrument_map and not os.path.isfile(args.instrument_map):
        print(f"Error: Instrument map file '{args.instrument_map}' does not exist")
        return False
    
    return True


def load_instrument_map(file_path: Optional[str]) -> Optional[Dict[int, int]]:
    """Load instrument mappings from a JSON file"""
    if not file_path:
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert string keys to integers
        return {int(k): int(v) for k, v in data.items()}
    
    except Exception as e:
        print(f"Error loading instrument map: {str(e)}")
        return None


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    try:
        # Load instrument map if specified
        instrument_map = load_instrument_map(args.instrument_map)
        
        # Print configuration if verbose
        if args.verbose:
            print("MIDI to Audio Conversion:")
            print(f"  Input:             {args.input}")
            print(f"  Output:            {args.output}")
            print(f"  Backend:           {args.backend}")
            if args.backend == "fluidsynth":
                print(f"  SoundFont:         {args.soundfont}")
            else:
                print(f"  Model:             {args.model}")
            print(f"  Transpose:         {args.transpose} semitones")
            print(f"  Velocity Scale:    {args.velocity_scale}")
            print(f"  Velocity Offset:   {args.velocity_offset}")
            print(f"  Instrument Map:    {args.instrument_map if args.instrument_map else 'None'}")
            print(f"  Sample Rate:       {args.sample_rate} Hz")
            print()
        
        # Initialize converter
        start_time = time.time()
        
        if args.verbose:
            print("Initializing converter...")
        
        if args.backend == "fluidsynth":
            converter = MIDIToAudioConverter(
                synthesis_backend="fluidsynth",
                soundfont_path=args.soundfont
            )
        else:  # differentiable
            converter = MIDIToAudioConverter(
                synthesis_backend="differentiable",
                model_path=args.model
            )
        
        # Convert MIDI to audio
        if args.verbose:
            print("Converting MIDI to audio...")
        
        converter.convert(
            midi_path=args.input,
            output_path=args.output,
            transpose=args.transpose,
            velocity_scale=args.velocity_scale,
            velocity_offset=args.velocity_offset,
            instrument_map=instrument_map
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print success message
        print(f"Conversion successful! Output saved to: {args.output}")
        
        if args.verbose:
            print(f"Conversion completed in {duration:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()