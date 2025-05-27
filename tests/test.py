import sys
from pathlib import Path
# 현재 파일의 경로
current_path = Path(__file__).resolve()
# 부모 경로
parent_path = current_path.parent.parent
# 부모 경로를 sys.path에 추가
sys.path.append(str(parent_path))

import whisper
import speech_recognition as sr

model = whisper.load_model("base", device="cuda")
recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("말해주세요...")
    audio = recognizer.listen(source)

    with open("temp.wav", "wb") as f:
        f.write(audio.get_wav_data())

result = model.transcribe("temp.wav")
print("Whisper 인식 결과:", result["text"])
