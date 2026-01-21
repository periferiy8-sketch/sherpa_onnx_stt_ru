# Установка зависимостей при первом запуске
import os
import sys

def install_deps():
    if not os.path.exists(".deps_installed"):
        print("📦 Installing dependencies...")
        os.system(f"{sys.executable} -m pip install flask flask-cors sherpa-onnx requests numpy")
        with open(".deps_installed", "w") as f:
            f.write("ok")
        print("✅ Dependencies installed.")

install_deps()

# Импорты после установки
from flask import Flask, request, jsonify
from flask_cors import CORS
import sherpa_onnx
import wave
import numpy as np
import tempfile
import requests

print("🚀 Starting STT server...")

app = Flask(__name__)
CORS(app)

# === Настройка модели ===
MODEL_DIR = "./model"
MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ru-2024-09-18.tar.bz2"

if not os.path.exists(os.path.join(MODEL_DIR, "model_ready")):
    print("📥 Downloading Sherpa-ONNX Russian model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Скачиваем архив
    archive_path = "/tmp/model.tar.bz2"
    with open(archive_path, "wb") as f:
        f.write(requests.get(MODEL_URL).content)
    
    # Распаковываем
    os.system(f"tar -xjf {archive_path} -C {MODEL_DIR}")
    
    # Перемещаем файлы из подпапки
    inner_dir = os.path.join(MODEL_DIR, "sherpa-onnx-zipformer-ru-2024-09-18")
    if os.path.exists(inner_dir):
        for item in os.listdir(inner_dir):
            os.rename(os.path.join(inner_dir, item), os.path.join(MODEL_DIR, item))
        os.rmdir(inner_dir)
    
    os.remove(archive_path)
    with open(os.path.join(MODEL_DIR, "model_ready"), "w") as f:
        f.write("done")
    print("✅ Model ready.")

# === Конфигурация распознавателя ===
config = sherpa_onnx.OfflineRecognizerConfig(
    model_config=sherpa_onnx.OfflineModelConfig(
        transducer=sherpa_onnx.OfflineTransducerModelConfig(
            encoder=os.path.join(MODEL_DIR, "encoder.onnx"),
            decoder=os.path.join(MODEL_DIR, "decoder.onnx"),
            joiner=os.path.join(MODEL_DIR, "joiner.onnx"),
        ),
        tokens=os.path.join(MODEL_DIR, "tokens.txt"),
        num_threads=1,
    ),
    feat_config=sherpa_onnx.FeatureConfig(
        sample_rate=16000,
        feature_dim=80
    ),
    decoding_method="greedy_search",
)

recognizer = sherpa_onnx.OfflineRecognizer(config)
print("✅ Sherpa-ONNX STT initialized.")

# === Вспомогательные функции ===
def read_wav(file_path):
    with wave.open(file_path, "rb") as wf:
        assert wf.getnchannels() == 1, "Only mono supported"
        assert wf.getsampwidth() == 2, "Only 16-bit supported"
        frames = wf.readframes(-1)
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, wf.getframerate()

# === Эндпоинты ===
@app.route('/health')
def health():
    return "OK", 200

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json(force=True)
    if not data or 'audioData' not in 
        return jsonify({"error": "Missing 'audioData'"}), 400

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
    app.run(host='0.0.0.0', port=5000)