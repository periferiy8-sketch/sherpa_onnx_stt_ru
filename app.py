from flask import Flask, request, jsonify
import sherpa_onnx
import wave
import numpy as np
import os
import tempfile
import time
import requests

app = Flask(__name__)

# Путь к модели
MODEL_DIR = "./model"
MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ru-2024-09-18.tar.bz2"

# Скачиваем модель, если её нет
if not os.path.exists(os.path.join(MODEL_DIR, "model_ready")):
    print("📥 Downloading Sherpa-ONNX Russian model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open("/tmp/model.tar.bz2", "wb") as f:
        f.write(requests.get(MODEL_URL).content)
    os.system("tar -xjf /tmp/model.tar.bz2 -C " + MODEL_DIR)
    os.rename(os.path.join(MODEL_DIR, "sherpa-onnx-zipformer-ru-2024-09-18"), os.path.join(MODEL_DIR, "temp"))
    os.system("mv " + os.path.join(MODEL_DIR, "temp/*") + " " + MODEL_DIR)
    os.system("rm -rf " + os.path.join(MODEL_DIR, "temp") + " /tmp/model.tar.bz2")
    with open(os.path.join(MODEL_DIR, "model_ready"), "w") as f:
        f.write("done")
    print("✅ Model ready at " + MODEL_DIR)

# Загружаем модель один раз при старте
recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer(
    model=MODEL_DIR,
    tokens=os.path.join(MODEL_DIR, "tokens.txt"),
    num_threads=1,
    debug=False,
)

print("✅ Sherpa-ONNX STT ready!")

def read_wav(file_path):
    """Читает WAV, возвращает (samples, sample_rate)"""
    with wave.open(file_path, "rb") as wf:
        assert wf.getnchannels() == 1, "Only mono supported"
        assert wf.getsampwidth() == 2, "Only 16-bit supported"
        sample_rate = wf.getframerate()
        frames = wf.readframes(-1)
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate

@app.route('/health')
def health():
    return "OK", 200

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files['audio']
    if not audio_file.filename.lower().endswith('.wav'):
        return jsonify({"error": "Only WAV files supported"}), 400

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        start_time = time.time()
        samples, sample_rate = read_wav(tmp_path)
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        elapsed = time.time() - start_time
        print(f"⏱️  Transcribed in {elapsed:.2f} sec: '{text}'")
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)