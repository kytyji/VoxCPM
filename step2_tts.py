import json
import torch
import soundfile as sf
import voxcpm

REFERENCE_AUDIO = r"D:\Programs\AI\MyVoice.mp3"  # ← ваш файл с голосом
MODEL_PATH = "./pretrained_models/VoxCPM2"

torch.backends.cuda.matmul.allow_tf32 = True

with open("stt_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"[TTS] Перевод: {data['translated']}")
print(f"[TTS] VRAM до загрузки: {torch.cuda.memory_allocated()/1e9:.2f} GB")

tts = voxcpm.VoxCPM.from_pretrained(
    hf_model_id=MODEL_PATH,
    load_denoiser=False,
    optimize=False,
    device="cuda",
)

print(f"[TTS] VRAM после загрузки: {torch.cuda.memory_allocated()/1e9:.2f} GB")

wav = tts.generate(
    text=data["translated"],
    reference_wav_path=REFERENCE_AUDIO,
    cfg_value=2.0,
    inference_timesteps=10,
    normalize=False,
    denoise=False,
)

sr = tts.tts_model.sample_rate
sf.write("pipeline_output.wav", wav, sr)
print(f"[TTS] ✅ pipeline_output.wav (sr={sr})")