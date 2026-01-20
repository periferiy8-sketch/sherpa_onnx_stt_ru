# sherpa_onnx_stt_ru

Быстрый STT для русского языка на базе Sherpa-ONNX.

- Модель: `zipformer-ru-2024-09-18`
- Задержка: 1–3 сек
- Хостинг: Koyeb (бесплатно, без сна)

## Деплой на Koyeb

1. Создайте аккаунт на [koyeb.com](https://www.koyeb.com/) (через GitHub).
2. Создайте **Web Service** → укажите этот репозиторий.
3. Build Command:
   ```bash
   chmod +x setup_model.sh && ./setup_model.sh && pip install -r requirements.txt