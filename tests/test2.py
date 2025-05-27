import pyaudio
import webrtcvad
import collections
import numpy as np
import whisper

# 모델 불러오기
model = whisper.load_model("base")

# 오디오 설정
RATE = 16000
CHANNELS = 1
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  # 프레임당 샘플 수
FORMAT = pyaudio.paInt16

vad = webrtcvad.Vad()
vad.set_mode(2)  # 0~3 사이, 3이 가장 민감함

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=FRAME_SIZE)

print("Listening...")

def bytes_to_float32(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np

ring_buffer = collections.deque(maxlen=10)  # 최근 음성 판단 저장용
triggered = False
voiced_frames = []

try:
    while True:
        frame = stream.read(FRAME_SIZE, exception_on_overflow=False)

        is_speech = vad.is_speech(frame, RATE)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 6:  # 6 * 30ms = 180ms 이상 음성 감지되면 말 시작
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 6:  # 180ms 이상 무음이면 말 끝으로 판단
                triggered = False

                # 녹음된 음성 합침
                audio_data = b''.join(voiced_frames)
                voiced_frames = []

                # float32로 변환 후 인식
                audio_float = bytes_to_float32(audio_data)
                result = model.transcribe(audio_float, fp16=False)
                print("Recognized:", result["text"])

                ring_buffer.clear()

except KeyboardInterrupt:
    print("Stopped")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
