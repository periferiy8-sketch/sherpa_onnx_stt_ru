from flask import Flask, request, jsonify
from flask_cors import CORS
import sherpa_onnx
import wave
import numpy as np
import os
import tempfile
import requests

print("🚀 Starting STT server from:", __file__)

app = Flask(__name__)
CORS(app)

# === MODEL SETUP ===
MODEL_DIR = "./model"
MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ru-2024-09-18.tar.bz2"

if not os.path.exists(os.path.join(MODEL_DIR, "model_ready")):
    print("📥 Downloading Sherpa-ONNX model...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    archive_path = "/tmp/model.tar.bz2"
    with open(archive_path, "wb") as f:
        f.write(requests.get(MODEL_URL).content)

    os.system(f"tar -xjf {archive_path} -C {MODEL_DIR}")

    inner = os.path.join(MODEL_DIR, "sherpa-onnx-zipformer-ru-2024-09-18")
    if os.path.exists(inner):
        for f in os.listdir(inner):
            os.rename(os.path.join(inner, f), os.path.join(MODEL_DIR, f))
        os.rmdir(inner)

    os.remove(archive_path)
    open(os.path.join(MODEL_DIR, "model_ready"), "w").close()
    print("✅ Model ready")

# === RECOGNIZER ===
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{MODEL_DIR}/encoder.onnx",
    decoder=f"{MODEL_DIR}/decoder.onnx",
    joiner=f"{MODEL_DIR}/joiner.onnx",
    tokens=f"{MODEL_DIR}/tokens.txt",
    num_threads=1,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
)

print("✅ Sherpa-ONNX initialized")

# === UTILS ===
def read_wav(path):
    with wave.open(path, "rb") as wf:
        samples = np.frombuffer(wf.readframes(-1), dtype=np.int16).astype(np.float32)
        samples /= 32768.0
        return samples, wf.getframerate()

# === ROUTES ===
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "OK",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json(force=True)
    if "audioData" not in data:
        return jsonify({"error": "audioData missing"}), 400

    audio_bytes = bytes(data["audioData"])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_bytes)
        path = tmp.name

    samples, sr = read_wav(path)
    os.remove(path)

    stream = recognizer.create_stream()
    stream.accept_waveform(sr, samples)
    recognizer.decode_stream(stream)

    return jsonify({"text": stream.result.text.strip()})

# === START ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
