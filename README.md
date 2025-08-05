# E-Acc_Instrument_Transposition

## 🚀 Getting Started 

### 1. Clone the repo
```bash
git clone https://github.com/Malilka/E-Acc_Instrument_Transposition.git
cd E-Acc_Instrument_Transposition
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

## 🎯 Purpose

   - A 10-second pitch:
     > ⚡ *A fast and flexible MIDI-to-audio synthesizer. Whether you want speed (FluidSynth) or realism (AI rendering), this tool turns MIDI into clean, expressive audio. *


## 🎶 Proposed Project Pipeline

   - ### 🛠 Project Pipeline Overview

      → **`Raw Audio` (.wav)**
     
      → **`Audio Transcription`** : Uses **Onsets and Frames model** to transcribe raw polyphonic audio into MIDI. Suitable for piano-like input.
     
      → **`Symbolic Manipulation`** :
        - With **pretty_midi**, we transform the MIDI file by updating the following parameters:
            - Pitch transposition (i.e. what notes are played)
            - Velocity adjustment (i.e. how hard they're hit)
            - Instrument program reassignment
              
      → **`Audio Synthesis`** (choose one):
      - **Differentiable Rendering (AI-based)**: Learns synthesis with Mel-spectrogram loss. High quality.
      - **FluidSynth (rule-based)**: Rule-based SoundFont synthesis. Practical baseline.

           
      → **`Evaluation (symbolic + audio metrics)`** : Assess the output both symbolically (is the MIDI accurate?) and audibly (does the final sound actually sound like the intended instrument?)
     - **Symbolic**: Note onset, pitch, rhythm accuracy
     - **Audio**: MelCD, spectral convergence, instrument classification, perceptual tests

      → **`Final Output`**: **.wav** in a new instrument’s voice





 - ### 🔁 Simplified Pipeline with Stages

   - **Input: .wav**  
   - **Transcription** : **.wav -> MIDI**  
   - **Symbolic Manipulation** : **MIDI -> new MIDI**  
   - **Synthesis** : **new MIDI -> .wav (new instrument)**
     

## 💻 File Overview

| File | Description |
|------|-------------|
| `integration_code.py` | End-to-end pipeline controller |
| `differentiable_renderer.py` | Differentiable synthesis backend |
| `train_differentiable_synthesizer.py` | Training loop for learned audio renderer |
| `midi_to_audio_converter.py` | MIDI to audio using FluidSynth |
| `simple_converter.py` | Simpler CLI wrapper for FluidSynth synthesis |
| `midi_to_audio_cli.py` | Command-line interface for conversion |
| `requirements.txt` | Python dependencies |


## 👤 Contributors
- [Mallika Dasgupta](https://github.com/Mallika1405)
- [Ashish Bamba](https://github.com/AshishBamba05)

## 📑 Relevant Documents
- [MuseScore+ E/Acc Document](https://docs.google.com/document/d/1oZA30UqnOtiMxfK-RZ8UEz_G2kqgnJWKfGTHXXf4PLQ/edit?tab=t.0#heading=h.d8uue6rjq29x)

## 📚 References

Key foundations:

- Sumino et al., _Differentiable Rendering for Instrument Transposition_ ([AIMC 2020](https://arxiv.org/abs/2008.04956))
- Mor et al., _Universal Music Translation Network_ ([arXiv 2018](https://arxiv.org/abs/1805.07848))
- Hawthorne et al., _Onsets and Frames_ ([GitHub](https://github.com/magenta/magenta/tree/main/magenta/models/onsets_frames_transcription))
