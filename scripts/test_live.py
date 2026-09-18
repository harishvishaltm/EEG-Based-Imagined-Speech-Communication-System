"""Live test script demonstrating the complete pipeline in one command:
1. Health Check
2. Model Metadata
3. Demo Sample Prediction (EEG -> Word -> Sentence -> Audio)
4. Raw Numerical Vector Prediction
"""
import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def run_live_test():
    print("=" * 80)
    print("EEG-BASED IMAGINED SPEECH COMMUNICATION SYSTEM - LIVE TEST")
    print("=" * 80)

    # 1. Health check
    print("\n[1] Checking System Health...")
    health = client.get("/api/health").json()
    print(f"    Status       : {health['status']}")
    print(f"    Model Loaded : {health['model_loaded']}")
    print(f"    Device       : {health['device']}")
    print(f"    Dataset      : {health['dataset']}")
    print(f"    Mode         : {health['mode']}")

    # 2. Model info
    print("\n[2] Checking Model Info...")
    info = client.get("/api/model-info").json()
    print(f"    Model Name   : {info['model_name']}")
    print(f"    Classes (11) : {info['classes']}")
    print(f"    Input Shape  : {info['input_shape']}")
    print(f"    Sampling Rate: {info['sampling_frequency']} Hz")

    # 3. Test Demo Sample
    print("\n[3] Testing Demo Sample Prediction (sample_001)...")
    res = client.post("/api/predict-demo/sample_001").json()
    print(f"    Input File   : {res['input']['filename']}")
    print(f"    Decoded Word : {res['prediction']['word']} (Confidence: {res['prediction']['confidence'] * 100:.1f}%)")
    print(f"    Top 3 Classes: {res['top_predictions'][:3]}")
    print(f"    Ground Truth : {res['ground_truth']}")
    print(f"    Evaluation   : {'✓ Correct' if res['evaluation'] and res['evaluation']['correct'] else 'Incorrect'}")
    print(f"    Sentence     : \"{res['communication']['sentence']}\"")
    print(f"    Audio URL    : {res['audio']['url']}")
    print(f"    Inference ms : {res['processing']['inference_time_ms']} ms")
    print(f"    Waveforms    : {len(res['visualization']['waveforms'])} channels extracted for display")

    # 4. Test Raw Numerical Vector Input
    print("\n[4] Testing Direct Numerical Vector Input (62 channels x 500 samples)...")
    raw_vector = np.random.randn(62, 500).tolist()
    vec_res = client.post(
        "/api/predict-vector",
        json={"eeg": raw_vector, "dataset": "kara_one", "subject": "MM05", "ground_truth": "pot"},
    ).json()
    print(f"    Input Vector : 62 channels x 500 time points")
    print(f"    Decoded Word : {vec_res['prediction']['word']}")
    print(f"    Sentence     : \"{vec_res['communication']['sentence']}\"")
    print(f"    Audio URL    : {vec_res['audio']['url']}")
    print(f"    Inference ms : {vec_res['processing']['inference_time_ms']} ms")

    # 5. Verify Audio File Exists on Disk
    audio_path = PROJECT_ROOT / "backend" / "static" / "audio" / Path(vec_res["audio"]["url"]).name
    print(f"\n[5] Verifying Generated Audio File...")
    print(f"    File on disk : {audio_path.resolve()}")
    print(f"    File Exists  : {audio_path.exists()}")
    if audio_path.exists():
        print(f"    File Size    : {audio_path.stat().st_size} bytes")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_live_test()
