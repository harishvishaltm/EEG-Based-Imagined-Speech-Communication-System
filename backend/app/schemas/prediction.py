"""Pydantic schemas for request and response validation."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    word: str
    confidence: float


class PredictionResult(BaseModel):
    word: str
    confidence: float


class EvaluationResult(BaseModel):
    correct: bool


class CommunicationResult(BaseModel):
    sentence: str
    generation_mode: str


class AudioResult(BaseModel):
    url: str


class ProcessingResult(BaseModel):
    inference_time_ms: float
    preprocessing_time_ms: Optional[float] = None
    device: str


class ChannelWaveform(BaseModel):
    channel: str
    values: List[float]


class VisualizationData(BaseModel):
    sampling_rate: int
    duration_seconds: float
    time_points: List[float]
    waveforms: List[ChannelWaveform]


class PredictionResponse(BaseModel):
    success: bool
    input: Dict[str, Any]
    prediction: PredictionResult
    top_predictions: List[TopPrediction]
    ground_truth: Optional[str] = None
    evaluation: Optional[EvaluationResult] = None
    communication: Optional[CommunicationResult] = None
    audio: Optional[AudioResult] = None
    processing: ProcessingResult
    visualization: Optional[VisualizationData] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    dataset: str
    mode: str
    is_calibrated: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    dataset: str
    number_of_classes: int
    classes: List[str]
    input_shape: List[int]
    sampling_frequency: int
    model_loaded: bool
    device: str
    architecture: str


class DemoSampleItem(BaseModel):
    id: str
    dataset: str
    subject: str
    label: Optional[str] = None
    file_size_bytes: int


class GenerateSentenceRequest(BaseModel):
    word: str
    use_memory: Optional[bool] = False


class GenerateSentenceResponse(BaseModel):
    word: str
    sentence: str
    generation_mode: str


class TextToSpeechRequest(BaseModel):
    text: str
    word_hint: Optional[str] = None


class TextToSpeechResponse(BaseModel):
    text: str
    audio_url: str
    backend_used: str


class VectorPredictionRequest(BaseModel):
    eeg: List[List[float]] = Field(..., description="Numerical 2D EEG vector (channels x time samples)")
    channel_names: Optional[List[str]] = Field(None, description="Optional channel names (e.g. ['CZ', 'FZ', ...])")
    sfreq: Optional[float] = Field(256.0, description="Sampling rate in Hz")
    dataset: Optional[str] = Field("kara_one", description="Dataset identifier")
    subject: Optional[str] = Field("MM05", description="Subject identifier")
    ground_truth: Optional[str] = Field(None, description="Optional ground truth label for demonstration")

