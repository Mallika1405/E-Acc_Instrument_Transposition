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

## 📌 About

This repository is part of a larger research initiative @ ***CSES E/Acc (Effective Accelerationism)*** exploring AI-powered music processing pipelines — from audio transcription to symbolic manipulation and neural resynthesis. The project forms a key component of **MuseScore+**, an experimental tool for expressive music transformation.

🔬 This work will be presented at the ***San Diego Tech Conference (SDTC)*** on October 2025, as part of a collaborative research effort focused on intelligent audio synthesis.



## 🎯 Purpose

A 10-second pitch: ⚡ Meet MuseScore+ — your all-in-one AI music engine. Go from raw audio to a brand-new voice in seconds. Transcribe with precision, manipulate with power, and synthesize with style.
     
   - ✅ ***Studio-Grade Accuracy:*** high‑quality transcription using the **Onsets and Frames** model
        
   - 🚀 ***Blazing Fast:*** **FluidSynth** baseline renders quickly with SoundFonts; great for demos and batch jobs
        
   - 🎚 ***Fully Customizable:*** Transpose pitch, scale velocities, remap instruments via CLI or Python API


## 🎶 Full Proposed Project Pipeline

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

## 📦 Scope of This Repository

This repository focuses specifically on the **audio synthesis** component of the full instrument transposition pipeline. It includes:

- ✅ A CLI + Python API for MIDI-to-audio conversion
- ✅ Two backends: rule-based (FluidSynth) and AI-based (Differentiable Renderer)
- ✅ Tools to train the differentiable synthesizer on WAV instrument samples
- ❌ Does not handle audio transcription (e.g. .wav → MIDI) — this is out-of-scope
- ❌ Does not perform symbolic manipulation (e.g. pitch shift, velocity changes) directly, but supports MIDI pre-processing via config

Use this repo if you want to **render expressive audio from MIDI**, compare synthesis strategies, or experiment with AI-driven timbre modeling.


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


## Relevant Libraries / Frameworks
- PyTorch
- PrettyMIDI


## 👤 Contributors
- [Mallika Dasgupta](https://github.com/Mallika1405)
- [Ashish Bamba](https://github.com/AshishBamba05)

## 📑 Relevant Documents

For more information regarding any relevant component to this research project pipeline, please feel free to refer to the following document: 

- [MuseScore+ E/Acc Document](https://docs.google.com/document/d/1oZA30UqnOtiMxfK-RZ8UEz_G2kqgnJWKfGTHXXf4PLQ/edit?tab=t.0#heading=h.d8uue6rjq29x)

## 📚 References

Key foundations:

- Sumino et al., _Differentiable Rendering for Instrument Transposition_ ([AIMC 2020](https://arxiv.org/abs/2008.04956))
- Mor et al., _Universal Music Translation Network_ ([arXiv 2018](https://arxiv.org/abs/1805.07848))
- Hawthorne et al., _Onsets and Frames_ ([GitHub](https://github.com/magenta/magenta/tree/main/magenta/models/onsets_frames_transcription))
