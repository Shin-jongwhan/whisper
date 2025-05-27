import whisper
import webrtcvad
import wave
import contextlib
import numpy as np
from scipy.io import wavfile
import os
import tempfile

# ----- 무음 제거용 함수 -----
def vad_split(wav_path, aggressiveness=2, frame_duration_ms=30):
    vad = webrtcvad.Vad(aggressiveness)
    with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000), "지원되지 않는 샘플레이트"
        pcm_data = wf.readframes(wf.getnframes())
        frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 16bit=2byte
        segments = []
        for i in range(0, len(pcm_data), frame_size):
            frame = pcm_data[i:i+frame_size]
            if len(frame) < frame_size:
                break
            is_speech = vad.is_speech(frame, sample_rate)
            if is_speech:
                segments.append((i // 2, i + frame_size))  # (시작 byte idx, 끝 byte idx)
    # byte index → time(sec)
    result = []
    last_end = 0
    for start_byte, end_byte in segments:
        start_sec = start_byte / (sample_rate * 2)
        end_sec = end_byte / (sample_rate * 2)
        if result and start_sec - result[-1][1] < 1.0:
            result[-1] = (result[-1][0], end_sec)
        else:
            result.append((start_sec, end_sec))
    return result, sample_rate

# ----- Whisper 인식 -----
def transcribe_with_timestamps(audio_path, language="ko"):
    print("🔊 무음 제거 중...")
    speech_segments, sample_rate = vad_split(audio_path)

    model = whisper.load_model("medium")

    result = []
    for idx, (start, end) in enumerate(speech_segments):
        print(f"🎙️ segment {idx+1}: {start:.2f}s ~ {end:.2f}s")
        with wave.open(audio_path, 'rb') as wf:
            wf.setpos(int(start * sample_rate))
            frames = wf.readframes(int((end - start) * sample_rate))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wavfile.write(tmp.name, sample_rate, np.frombuffer(frames, dtype=np.int16))
            tmp.flush()
            seg_result = model.transcribe(tmp.name, language=language)
            segment_text = seg_result.get("text", "").strip()
            result.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "segment": segment_text
            })
        os.remove(tmp.name)

    return result

# ----- 실행 -----
if __name__ == "__main__":
    AUDIO_FILE = "/data/test_audio.wav"  # 🔁 여기에 오디오 파일 경로 입력

    output = transcribe_with_timestamps(AUDIO_FILE)

    print("\n📄 최종 결과:")
    for item in output:
        print(item)
