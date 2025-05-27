import librosa
import numpy as np
import pandas as pd
import os

def extract_framewise_features(file_path, frame_duration=0.5):
    y, sr = librosa.load(file_path, sr=None)
    frame_length = int(frame_duration * sr)
    hop_length = frame_length  # 프레임 간 겹침 없음

    data = []

    for start_sample in range(0, len(y) - frame_length + 1, hop_length):
        y_frame = y[start_sample:start_sample + frame_length]
        time_stamp = start_sample / sr

        # 피치 (F0)
        f0, voiced_flag, _ = librosa.pyin(y_frame,
                                          fmin=librosa.note_to_hz('C2'),
                                          fmax=librosa.note_to_hz('C7'),
                                          sr=sr)
        pitch = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0

        # 기타 특징
        rms = np.mean(librosa.feature.rms(y=y_frame))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_frame))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_frame, sr=sr))
        mfccs = librosa.feature.mfcc(y=y_frame, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)

        row = {
            'time': round(time_stamp, 2),
            'pitch': pitch,
            'rms': rms,
            'zcr': zcr,
            'spectral_centroid': spectral_centroid,
        }

        for i, mfcc in enumerate(mfcc_means):
            row[f'mfcc_{i+1}'] = mfcc

        data.append(row)

    return pd.DataFrame(data)


# 경로 설정
file_path = "/data/test_audio.wav"  # 분석할 .wav 파일 경로
output_path = "/data/test_audio_whisper_result.csv"

# 디렉토리 생성 (없을 경우)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 특징 추출 및 저장
df = extract_framewise_features(file_path)
df.to_csv(output_path, index=False)

print(f"✅ 분석 결과가 {output_path} 에 저장되었습니다.")
