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

## Purpose

   - A 10-second pitch: 

## 🎶 Proposed Project Pipeline

   - ### 🎯 Project Pipeline Overview

      → **Raw Audio (.wav)**
     
      → **Asudio Transcription** : Uses Onsets and Frames model to transcribe raw polyphonic audio into MIDI. Suitable for piano-like input.
     
      → **Symbolic Manipulation** :
        - With pretty_midi, allows:
            - Pitch transposition
            - Velocity adjustment
            - Instrument program reassignment
     
      → **Audio Synthesis** (choose one):
           → **Differentiable Rendering (AI-based)** :  Learns synthesis with Mel-spectrogram loss. High quality.
           → **FluidSynth (rule-based)** : Rule-based SoundFont synthesis. Practical baseline.
     
      → **Evaluation** (symbolic + audio metrics) : 
      
      → **Final Output: `.wav` in a new instrument’s voice**



 - ### 🔁 Simplified Pipeline with Stages

   - **Input: .wav**  
   - **Transcription** : **.wav -> MIDI**  
   - **Symbolic Manipulation** : **.MIDI -> .prettyMIDI**  
   - **Synthesis** : **.prettyMIDI -> .wav (new instrument)**
     

## 💻 File Overview

## 👤 Contributors
- [Mallika Dasgupta](https://github.com/Mallika1405)
- [Ashish Bamba](https://github.com/AshishBamba05)

## 📚 References

Key foundations:

- Sumino et al., _Differentiable Rendering for Instrument Transposition_ ([AIMC 2020](https://arxiv.org/abs/2008.04956))
- Mor et al., _Universal Music Translation Network_ ([arXiv 2018](https://arxiv.org/abs/1805.07848))
- Hawthorne et al., _Onsets and Frames_ ([GitHub](https://github.com/magenta/magenta/tree/main/magenta/models/onsets_frames_transcription))
