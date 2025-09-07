# elevenlabs_tts.py — FINAL

import os
from typing import List

# ======================
# Config: 음성 템플릿/보이스
# ======================

# ElevenLabs용 템플릿 (예시)
TTS_ELEVENLABS_TEMPLATES = {
    "default": {"voice": "Rachel", "stability": 0.4, "similarity_boost": 0.8},
    "soft": {"voice": "Bella", "stability": 0.6, "similarity_boost": 0.9},
    "fast": {"voice": "Elli", "stability": 0.3, "similarity_boost": 0.7},
}

# Polly 보이스 맵 (예시)
TTS_POLLY_VOICES = {
    "default_male": {"voice_id": "Seoyeon", "lang": "ko-KR"},
    "korean_female1": {"voice_id": "Seoyeon", "lang": "ko-KR"},
    "korean_male1": {"voice_id": "Takumi", "lang": "ja-JP"},  # 필요에 맞게 수정
}


# ======================
# 실제 합성 함수 (stub)
# ======================

def synthesize_with_polly(ssml: str, out_path: str, polly_voice_key: str = "korean_female1"):
    """
    AWS Polly로 SSML 합성.
    실제 구현은 boto3 사용 (Polly client.synthesize_speech).
    """
    try:
        # TODO: boto3 Polly client 연결
        # 현재는 더미 파일 생성
        with open(out_path, "wb") as f:
            f.write(b"")  # 빈 파일 placeholder
        print(f"[Polly] Generated TTS: {out_path} ({polly_voice_key})")
    except Exception as e:
        print(f"[error] Polly TTS 실패: {e}")
        raise


def synthesize_with_elevenlabs(ssml: str, out_path: str, voice_template: str = "default"):
    """
    ElevenLabs API로 SSML 합성.
    실제 구현은 requests.post(...) 호출.
    """
    try:
        # TODO: ElevenLabs API 호출 코드
        # 현재는 더미 파일 생성
        with open(out_path, "wb") as f:
            f.write(b"")  # 빈 파일 placeholder
        print(f"[11Labs] Generated TTS: {out_path} ({voice_template})")
    except Exception as e:
        print(f"[error] ElevenLabs TTS 실패: {e}")
        raise


# ======================
# Main API
# ======================

def generate_tts_per_line(
    ssml_list: List[str],
    provider: str = "polly",
    template: str = "default",
    polly_voice_key: str = "korean_female1",
) -> List[str]:
    """
    주어진 SSML 문자열 리스트를 줄 단위로 음성 파일 생성.
    - provider: "polly" | "elevenlabs"
    - 반환: 생성된 오디오 파일 경로 리스트
    """
    if not ssml_list:
        return []

    audio_paths: List[str] = []
    os.makedirs("assets/auto/tts_lines", exist_ok=True)

    for i, ssml in enumerate(ssml_list):
        out_path = os.path.join("assets", "auto", "tts_lines", f"line_{i:03d}.mp3")
        try:
            if provider.lower() == "polly":
                synthesize_with_polly(ssml, out_path, polly_voice_key=polly_voice_key)
            else:
                synthesize_with_elevenlabs(ssml, out_path, voice_template=template)
            audio_paths.append(out_path)
        except Exception as e:
            print(f"[warn] line {i} TTS 실패: {e}")
            audio_paths.append("")  # 실패 시 빈 문자열로 채움

    return audio_paths
