from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cpu", compute_type="int8")

# укажите путь к вашей записи голоса
segments, info = model.transcribe(r"D:\Programs\AI\MyVoice.mp3")

print(f"Язык: {info.language}")
print(f"Уверенность: {info.language_probability:.0%}")
print("Текст:")
for segment in segments:
    print(segment.text)