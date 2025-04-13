from midi_to_audio_converter import MIDIToAudioConverter

# Initialize with FluidSynth (the simpler approach)
converter = MIDIToAudioConverter(
    synthesis_backend="fluidsynth",
    soundfont_path="FluidR3_GM.sf2"  # Your SoundFont file - update this path
)

# Convert a MIDI file
converter.convert(
    midi_path="your_midi_file.mid",  # Your MIDI file - update this path
    output_path="output.mp3",
    transpose=0,                     # No transposition
    velocity_scale=1.0               # Original velocities
)

print("Conversion complete!")