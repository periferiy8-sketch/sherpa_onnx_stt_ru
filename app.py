from flask import Flask, request, jsonify
import sherpa_onnx
import wave
import numpy as np
import os
import tempfile
import requests

app = Flask(__name__)

# === Настройка модели ===
MODEL_DIR = "./model"
MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ru-2024-09-18.tar.bz2"

# Скачиваем модель при первом запуске, если не готова
if not os.path.exists(os.path.join(MODEL_DIR, "model_ready")):
    print("📥 Downloading Sherpa-ONNX Russian model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open("/tmp/model.tar.bz2", "wb") as f:
        f.write(requests.get(MODEL_URL).content)
    os.system(f"tar -xjf /tmp/model.tar.bz2 -C {MODEL_DIR}")
    inner_path = os.path.join(MODEL_DIR, "sherpa-onnx-zipformer-ru-2024-09-18")
    if os.path.exists(inner_path):
        for item in os.listdir(inner_path):
            os.rename(os.path.join(inner_path, item), os.path.join(MODEL_DIR, item))
        os.rmdir(inner_path)
    os.remove("/tmp/model.tar.bz2")
    with open(os.path.join(MODEL_DIR, "model_ready"), "w") as f:
        f.write("done")
    print("✅ Model ready.")

# === Инициализация распознавателя ===
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=os.path.join(MODEL_DIR, "encoder.onnx"),
    decoder=os.path.join(MODEL_DIR, "decoder.onnx"),
    joiner=os.path.join(MODEL_DIR, "joiner.onnx"),
    tokens=os.path.join(MODEL_DIR, "tokens.txt"),
    num_threads=1,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
    debug=False,
)
print("✅ Sherpa-ONNX STT initialized.")

# === Вспомогательные функции ===
def read_wav(file_path):
    with wave.open(file_path, "rb") as wf:
        assert wf.getnchannels() == 1, "Only mono supported"
        assert wf.getsampwidth() == 2, "Only 16-bit supported"
        sample_rate = wf.getframerate()
        frames = wf.readframes(-1)
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate

# === Эндпоинты ===
@app.route('/health')
def health():
    return "OK", 200

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    if not data or 'audioData' not in data:
        return jsonify({"error": "Missing 'audioData' in JSON"}), 400

    try:
        audio_bytes = bytes(data['audioData'])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)
            tmp_path = tmp.name

        samples, sample_rate = read_wav(tmp_path)
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()

        os.unlink(tmp_path)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Запуск ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)