# step2_tts.py
import json
import torch
import soundfile as sf
import voxcpm

REFERENCE_AUDIO = r"D:\Programs\AI\MyVoice.mp3"  # ← ваш файл с голосом
MAX_CHARS = 300  # ← ограничение длины для теста

torch.backends.cuda.matmul.allow_tf32 = True

with open("stt_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Обрезаем текст до первого предложения для теста
translated = data["translated"]
# Берём только первое предложение
first_sentence = translated.split(".")[0] + "."
print(f"Текст для синтеза ({len(first_sentence)} символов):")
print(f"  {first_sentence}")

print("\nЗагрузка VoxCPM2...")
tts = voxcpm.VoxCPM.from_pretrained(
    "./pretrained_models/VoxCPM2",
    load_denoiser=False,
    zipenhancer_model_id=None,
    optimize=False,
    device="cuda",
)
print("✅ Модель загружена, генерация...")

wav = tts.generate(
    text=first_sentence,
    reference_wav_path=REFERENCE_AUDIO,
    cfg_value=2.0,
    inference_timesteps=10,
    max_len=512,   # ← явно ограничиваем буфер генерации
    normalize=False,
    denoise=False,
)

sr = tts.tts_model.sample_rate
sf.write("pipeline_output.wav", wav, sr)
print("✅ Готово! pipeline_output.wav")