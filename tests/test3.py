import os
import time
import whisper
import speech_recognition as sr

# Whisper 모델 로드
model = whisper.load_model("base")
#save_dir = "C:\\test\\speech_recognition\\whisper\\tests\\audio"
save_dir = "/data/whisper/tests/audio"

# Recognizer 초기화
recognizer = sr.Recognizer()

def listen_and_transcribe():
    with sr.Microphone() as source:
        print("잠시 대기 중... (말하면 녹음 시작)")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        print("녹음 완료. 처리 중...")

        # 파일 저장
        filename = f"{save_dir}\\voice_{int(time.time())}.wav"
        with open(filename, "wb") as f:
            f.write(audio.get_wav_data())

        # Whisper로 인식
        result = model.transcribe(filename)
        print("🗣️ 인식 결과:", result["text"])

        # 필요 없으면 삭제
        #os.remove(filename)

# 반복적으로 실행
while True:
    try:
        listen_and_transcribe()
    except KeyboardInterrupt:
        print("\n종료합니다.")
        break
    except Exception as e:
        print("오류 발생:", e)
