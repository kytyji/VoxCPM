import subprocess
import sys
import json
import time

PYTHON = sys.executable  # тот же venv

def run_step(script: str, label: str) -> bool:
    print(f"\n{'='*40}\n▶ {label}\n{'='*40}")
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, script],
        capture_output=False,   # вывод в консоль в реальном времени
        text=True
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"✗ {label} завершился с ошибкой (code {result.returncode})")
        return False
    print(f"✓ {label} завершён за {elapsed:.1f}с")
    return True

if __name__ == "__main__":
    if not run_step("step1_stt.py", "STT + Translation"):
        sys.exit(1)

    # Проверяем промежуточный результат
    with open("stt_result.json", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n📝 Распознано ({data['lang']}): {data['original']}")
    print(f"🌐 Перевод: {data['translated']}")

    if not run_step("step2_tts.py", "TTS Synthesis (VoxCPM2)"):
        sys.exit(1)

    print("\n✅ Пайплайн завершён → pipeline_output.wav")