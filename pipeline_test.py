# pipeline_test.py
import gc
import os
import torch
import soundfile as sf
import ollama
from faster_whisper import WhisperModel
import voxcpm

# --- Настройки ---
REFERENCE_AUDIO = r"D:\Programs\AI\MyVoice.mp3"  # ← ваш файл с голосом
OLLAMA_MODEL = "qwen2.5:3b"
TARGET_LANG = "English"

torch.backends.cuda.matmul.allow_tf32 = True

if not os.path.exists(REFERENCE_AUDIO):
    print(f"❌ Файл не найден: {REFERENCE_AUDIO}")
    exit()

ollama_client = ollama.Client(host="http://localhost:11434")

# --- Шаг 1: STT (Whisper) ---
print("\n[1/3] Распознавание речи...")
stt = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = stt.transcribe(REFERENCE_AUDIO)
original_text = " ".join([s.text.strip() for s in segments])
print(f"  Язык: {info.language}")
print(f"  Текст: {original_text}")

# Выгружаем Whisper из памяти перед загрузкой VoxCPM
del stt
gc.collect()
torch.cuda.empty_cache()
print("  Whisper выгружен из памяти")

# --- Шаг 2: Перевод ---
print("\n[2/3] Перевод...")
response = ollama_client.chat(
    model=OLLAMA_MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are a translator. Output ONLY the translated text. No explanations, no notes, no alternatives."
        },
        {
            "role": "user",
            "content": f"Translate from {info.language} to {TARGET_LANG}:\n{original_text}"
        }
    ]
)
translated = response["message"]["content"].strip()
print(f"  Результат: {translated}")

# --- Шаг 3: TTS (VoxCPM2) ---
print("\n[3/3] Загрузка VoxCPM2 и синтез голоса...")
tts = voxcpm.VoxCPM.from_pretrained(
    "./pretrained_models/VoxCPM2",
    load_denoiser=False,        # ← ключевое исправление
    zipenhancer_model_id=None,  # ← явно отключаем
    optimize=False,
    device="cuda",
)
print("  ✅ Модель загружена, генерация...")

wav = tts.generate(
    text=translated,
    reference_wav_path=REFERENCE_AUDIO,
    cfg_value=2.0,
    inference_timesteps=10,
    normalize=False,
    denoise=False,
)

sr = tts.tts_model.sample_rate
output_path = "pipeline_output.wav"
sf.write(output_path, wav, sr)
print(f"\n✅ Готово! Сохранено: {output_path}")