# EEG-Based Imagined Speech Communication System

A brain-computer interface (BCI) prototype that decodes pre-recorded imagined-speech EEG signals into communication words, constructs natural assistive sentences via an LLM, and speaks them aloud via Text-to-Speech (TTS).

---

## 1. Project Objective & Conceptual Pipeline

The primary goal of this project is to assist individuals with severe motor or vocal disabilities (e.g., ALS, locked-in syndrome) to communicate. The system decodes the neural signals generated when a person actively imagines pronouncing a specific word or phoneme.

```
Person imagines a word
         ↓
Pre-recorded EEG recording (.npz, .mat, .cnt)
         ↓
EEG Preprocessing & Channel Harmonization (122 channels @ 256 Hz)
         ↓
Trained Deep Learning Model (Gate + CNN + Self-Attention + BiLSTM + Fusion)
         ↓
Predicted Imagined Word (e.g., "pot", "water", "help")
         ↓
Assistive LLM Service (gpt-4o-mini / deterministic fallback)
         ↓
Natural First-Person Sentence (e.g., "I need some water.")
         ↓
Text-To-Speech (gTTS / pyttsx3)
         ↓
Audio Playback (.mp3 / .wav)
```

> [!IMPORTANT]
> **College Demonstration Notice**:
> This prototype uses pre-recorded EEG recordings from the University of Toronto **Kara One** database because physical EEG electrode hardware and live acquisition headsets are not included in the college demonstration setting.
>
> **Scope & Capabilities**:
> The system **does not** read arbitrary thoughts or perform mind-reading. It classifies predefined imagined-speech categories learned from EEG training data.

---

## 2. Research Architecture & Technical Specifications

The system is derived from the research implementation in `notebooks/project (1).ipynb`:

### A. Preprocessing Pipeline
1. **Filtering**: 4th-order Butterworth IIR Bandpass filter (0.5 Hz – 100.0 Hz) and Notch filter (50.0 Hz and 60.0 Hz) to eliminate mains power line interference without edge distortion.
2. **Spatial Re-referencing**: Common Average Reference (CAR), subtracting across-channel mean at every timepoint.
3. **Resampling**: Resamples signals from native 1000 Hz to target 256 Hz.
4. **Epoch Standardization**: Center-cropping or center-padding to exactly 1280 samples (5.0 seconds).
5. **Artifact Handling**: Outlier rejection based on peak-to-peak modified Z-score using Median Absolute Deviation (MAD threshold = 4.0).
6. **Normalization**: Per-channel Z-score normalization across the temporal axis.
7. **Canonical Channel Harmonization**: Zero-pads and aligns channels into a canonical 122-channel layout (62 named 10-20 EEG channels + 60 generic channel slots). Generates a 1D binary mask `channel_mask` of length 122.

### B. Deep Learning Model (`ImaginedSpeechModel`)
- **Learned Channel Gate** (3,281 parameters): Learns dynamic per-channel, per-trial importance weights using a shared Conv1D temporal encoder, subject/dataset embeddings (dim=8), and an MLP with Sigmoid activation. Enforces hard channel masking.
- **Temporal CNN Backbone**: Separable depthwise convolution (`kernel=25`, `depthwise_mult=2`) and pointwise convolution (`1x1`) reducing temporal dimension by a factor of 16 (1280 samples → 80 time steps, 64 feature maps).
- **Multi-Head Self-Attention**: 4 heads, `embed_dim=64`, LayerNorm, and residual connection capturing long-range temporal dependencies across the 80 time steps.
- **Bidirectional LSTM**: 2 layers, `hidden_size=128`, producing 256-dimensional forward/backward temporal representations.
- **Feature Fusion**: Global Average Pooling over time for both Attention (64) and BiLSTM (256), concatenated (320) and projected through an ELU dense layer to a 128-dimensional embedding.
- **Dataset-Specific Heads**: Dedicated classification heads for Kara One (11 classes), FEIS (16 classes), and Nguyen (3 classes).
- **Total Model Parameters**: **698,347** (identical to research notebook).

### C. Assistive LLM & TTS
- **EEG Model Role**: EEG signal → Single predicted word.
- **LLM Role**: Word → First-person assistive communication sentence (e.g. `water` → `"I need water."`).
- **TTS Role**: Sentence → Audio file served via `/audio/{filename}`.

---

## 3. Training vs Inference: Strictly Enforced Separation

```
TRAINING PHASE (Executed ONCE in Research/Notebook):
Kara One EEG Dataset → Preprocessing → Model Training (100 epochs) → Saved Checkpoint (kara_one_model.pth)

APPLICATION PHASE (FastAPI Production Backend):
Startup: Load kara_one_model.pth ONCE → model.eval() → Resident in memory
Request: Upload EEG → Preprocess → Forward Inference (torch.inference_mode()) → Output
```

> [!CAUTION]
> **NO RETRAINING**:
> During API prediction, there is **no** `model.fit()`, **no** backward pass, **no** optimizer step, and **no** retraining. Inference runs strictly in evaluation mode (`torch.inference_mode()`) in milliseconds.

---

## 4. Directory Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application & lifespan loader
│   │   ├── config.py                   # Environment & runtime settings
│   │   ├── api/
│   │   │   ├── routes_health.py        # /api/health, /api/model-info, /
│   │   │   ├── routes_prediction.py    # POST /api/predict (EEG file upload)
│   │   │   ├── routes_demo.py          # GET /api/demo-samples, POST /api/predict-demo
│   │   │   ├── routes_llm.py           # POST /api/generate-sentence
│   │   │   └── routes_tts.py           # POST /api/text-to-speech, GET /audio/{file}
│   │   ├── models/
│   │   │   ├── imagined_speech_model.py# Complete PyTorch architecture
│   │   │   └── model_loader.py         # Singleton in-memory model manager
│   │   ├── services/
│   │   │   ├── eeg_preprocessing.py    # 7-step signal processing & harmonization
│   │   │   ├── inference_service.py    # End-to-end inference orchestrator
│   │   │   ├── llm_service.py          # Assistive LLM & deterministic fallback
│   │   │   └── tts_service.py          # gTTS & pyttsx3 speech synthesizer
│   │   └── schemas/
│   │       └── prediction.py           # Pydantic request/response schemas
│   ├── requirements.txt
│   └── .env.example
├── models/
│   ├── kara_one_model.pth              # Exported PyTorch inference weights
│   ├── model_config.json               # Canonical model configuration
│   ├── label_mapping.json              # 11-class Kara One mapping
│   ├── channel_mapping.json            # 122-channel canonical layout
│   └── preprocessing_config.json       # Filter frequencies and window sizes
├── data/
│   └── kara_one/
│       ├── raw/                        # Place downloaded MMxx.tar.bz2 archives here
│       ├── processed/                  # Extracted subject trial directories
│       └── demo_samples/               # Pre-processed .npz demo samples
├── scripts/
│   ├── setup_dataset.py                # Dataset source verification & directory setup
│   ├── prepare_kara_one.py             # Archive extraction & preprocessing pipeline
│   ├── create_demo_samples.py          # Generates testable demo samples
│   ├── export_model.py                 # Exports weights & configuration
│   └── validate_backend_against_notebook.py # Notebook vs backend validation
├── notebooks/
│   └── project (1).ipynb               # Authoritative research notebook
├── tests/
│   ├── test_health.py
│   ├── test_model_loading.py
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   ├── test_llm.py
│   └── test_tts.py
├── .env
└── README.md
```

---

## 5. Installation & Setup Workflow

### Step 1: Environment Setup
Ensure Python 3.10+ is installed:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

### Step 2: Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
*(Optional: Add `OPENAI_API_KEY=your_key` if you want OpenAI cloud generation; otherwise the built-in deterministic fallback operates completely offline).*

### Step 3: Export Model Checkpoint
Initialize the model architecture and inference checkpoint:
```powershell
python scripts/export_model.py
```
This generates `models/kara_one_model.pth` and verifies all metadata JSON files.

### Step 4: Validate Backend Against Notebook
Run the validation script to verify that parameters, tensor shapes, and logits match the notebook exactly:
```powershell
python scripts/validate_backend_against_notebook.py
```

### Step 5: (Optional) Prepare Raw Kara One Dataset
If you have downloaded the Kara One archives from the University of Toronto:
1. Place `MM05.tar.bz2`, `MM08.tar.bz2`, etc. in `data/kara_one/raw/`.
2. Run extraction and preprocessing:
   ```powershell
   python scripts/prepare_kara_one.py
   ```
If you do not have the large 3GB raw archives yet, generate the ready demo samples:
```powershell
python scripts/create_demo_samples.py
```

### Step 6: Start FastAPI Backend
```powershell
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
The server will start at `http://localhost:8000`. Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## 6. College Demonstration Guide

During your presentation, follow this sequence:

1. **Check System Health**:
   Send a GET request to verify model is loaded in memory:
   ```powershell
   curl -X GET "http://localhost:8000/api/health"
   ```
2. **List Demo Samples**:
   ```powershell
   curl -X GET "http://localhost:8000/api/demo-samples"
   ```
3. **Execute Prediction on Demo Sample**:
   ```powershell
   curl -X POST "http://localhost:8000/api/predict-demo/sample_001"
   ```
4. **Inspect Results**:
   - **EEG Visualization**: Channel waveforms sampled across the 5.0-second epoch.
   - **Prediction**: Decoded word (e.g. `pot`), model confidence (e.g. `0.842`), and top-5 alternative classes.
   - **Evaluation**: `ground_truth: "pot"`, `correct: true`.
   - **Assistive Communication**: `"I want a pot."` (generated by LLM).
   - **Audio URL**: `/audio/pot_...mp3`.
   - **Inference Time**: Displays processing latency in milliseconds.

---

## 7. Example API Requests & Responses

### POST `/api/predict` (Multipart Upload)
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@data/kara_one/demo_samples/sample_001.npz" \
  -F "dataset=kara_one" \
  -F "subject=MM05"
```

### Response:
```json
{
  "success": true,
  "input": {
    "filename": "sample_001.npz",
    "dataset": "Kara One",
    "subject": "MM05"
  },
  "prediction": {
    "word": "pot",
    "confidence": 0.842
  },
  "top_predictions": [
    { "word": "pot", "confidence": 0.842 },
    { "word": "pat", "confidence": 0.081 },
    { "word": "/diy/", "confidence": 0.035 }
  ],
  "ground_truth": "pot",
  "evaluation": {
    "correct": true
  },
  "communication": {
    "sentence": "I want a pot.",
    "generation_mode": "fallback"
  },
  "audio": {
    "url": "/audio/pot_7a1b9c2d.mp3"
  },
  "processing": {
    "inference_time_ms": 18.42,
    "preprocessing_time_ms": 4.15,
    "device": "cpu"
  },
  "visualization": {
    "sampling_rate": 256,
    "duration_seconds": 5.0,
    "time_points": [0.0, 0.0156, 0.0312, "..."],
    "waveforms": [
      { "channel": "CZ", "values": [-0.14, 0.22, 0.45, "..."] },
      { "channel": "FZ", "values": [0.05, -0.12, 0.31, "..."] }
    ]
  }
}
```

---

## 8. Limitations & Future Real-Time Hardware Architecture

1. **Hardware Acquisition**: Current version processes pre-recorded EEG files. In a production clinic or research lab, the `file_path` input to `InferenceService` can be substituted with an LSL (Lab Streaming Layer) socket stream connected to an active 64-channel EEG amplifier (e.g. Brain Products, g.tec, or OpenBCI).
2. **Signal-to-Noise Ratio (SNR)**: Imagined speech signals have low SNR compared to motor imagery; spatial filtering and the Learned Channel Gate are crucial to suppress non-speech muscular artifacts.
3. **Thought Generalization**: The classifier identifies discrete vocabulary classes (`11 classes in Kara One`). Open-ended continuous thought decoding remains an open research challenge.
