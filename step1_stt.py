# step1_stt.py
import json
import ollama
from faster_whisper import WhisperModel

REFERENCE_AUDIO = r"D:\Programs\AI\MyVoice.mp3"  # ← ваш файл с голосом
OLLAMA_MODEL = "qwen2.5:3b"
TARGET_LANG = "English"

print("[1/2] Распознавание речи...")
stt = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = stt.transcribe(REFERENCE_AUDIO)
original_text = " ".join([s.text.strip() for s in segments])
print(f"  Язык: {info.language}")
print(f"  Текст: {original_text}")

print("[2/2] Перевод...")
client = ollama.Client(host="http://localhost:11434")
response = client.chat(
    model=OLLAMA_MODEL,
    messages=[
        {"role": "system", "content": "You are a translator. Output ONLY the translated text. No explanations."},
        {"role": "user", "content": f"Translate from {info.language} to {TARGET_LANG}:\n{original_text}"}
    ]
)
translated = response["message"]["content"].strip()
print(f"  Перевод: {translated}")

with open("stt_result.json", "w", encoding="utf-8") as f:
    json.dump({"original": original_text, "translated": translated, "lang": info.language}, f, ensure_ascii=False)

print("\n✅ Результат сохранён в stt_result.json")