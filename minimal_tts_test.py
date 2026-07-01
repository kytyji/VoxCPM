# minimal_tts_test.py
import torch
import soundfile as sf
import voxcpm

torch.backends.cuda.matmul.allow_tf32 = True

print("Загрузка модели...")
tts = voxcpm.VoxCPM.from_pretrained(
    "./pretrained_models/VoxCPM2",
    optimize=False,
    device="cuda",
    
)
print("✅ Модель загружена!")

print("Генерация...")
wav = tts.generate(
    text="Hello, this is a test, you motherfucker bitch you know",
    reference_wav_path=None,   # без референса — просто тест голоса
    cfg_value=2.0,
    inference_timesteps=10,
    normalize=False,
    denoise=False,
)
sr = tts.tts_model.sample_rate
sf.write("minimal_output.wav", wav, sr)
print(f"✅ Готово! minimal_output.wav")