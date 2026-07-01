import ollama

client = ollama.Client(host="http://localhost:11434")

def translate(text, source_lang, target_lang):
    response = client.chat(
        model="qwen2.5:7b-instruct-q4_K_M",  # или ваше точное имя
        messages=[
            {
                "role": "system",
                "content": "You are a translator. Output ONLY the translated text. No explanations, no notes, no alternatives."
            },
            {
                "role": "user", 
                "content": f"Translate from {source_lang} to {target_lang}:\n{text}"
            }
        ]
    )
    return response["message"]["content"].strip()

print(translate("Привет, это тест голосового перевода", "Russian", "English"))
print(translate("Привет, это тест голосового перевода", "Russian", "Chinese"))
print(translate("Hello, how are you today?", "English", "Russian"))