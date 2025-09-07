# generate_timed_segments.py — FINAL
from __future__ import annotations

import os
import re
import json
from typing import List, Dict, Any, Tuple

# LLM 배치 함수 (분절/SSML)
from ssml_converter import (
    breath_linebreaks_batch,
    convert_lines_to_ssml_batch,
    koreanize_if_english,
)

# 줄별 TTS 합성 (프로젝트 내 기존 함수 사용)
from elevenlabs_tts import generate_tts_per_line

# 오디오 병합/길이 계산용
from pydub import AudioSegment


# =========================
# ASS Subtitles (templates)
# =========================
SUBTITLE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "default": {
        "script_info": {
            "Title":         "Generated Subtitles",
            "ScriptType":    "v4.00+",
            "WrapStyle":     "0",
            "PlayResX":      "1080",
            "PlayResY":      "1920",
            "ScaledBorderAndShadow": "yes",
        },
        "v4_styles": [
            # Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic,
            # Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment,
            # MarginL, MarginR, MarginV, Encoding
            "Style: Default,Pretendard,64,&H00FFFFFF,&H000000FF,&H00000000,&H7F000000,0,0,0,0,100,100,0,0,1,4,0,2,40,40,80,1",
        ],
        "events_format": "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    }
}


def _ass_time(t: float) -> str:
    """seconds -> h:mm:ss.cc (ASS format centiseconds)"""
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _sanitize_text_for_ass(text: str) -> str:
    # ASS에서 { } 는 제어코드. 이스케이프
    t = (text or "").replace("{", r"\{").replace("}", r"\}")
    # 여러 공백 정리
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _quantize_segments(segments: List[Dict[str, Any]], fps: float = 24.0) -> List[Dict[str, Any]]:
    """프레임 경계에 맞춰 start/end를 살짝 보정(선택적)."""
    if fps <= 0:
        return segments
    q = []
    step = 1.0 / fps
    for seg in segments:
        st = round(seg["start"] / step) * step
        et = round(seg["end"] / step) * step
        if et <= st:
            et = st + step
        q.append({**seg, "start": st, "end": et})
    return q


def generate_ass_subtitle(events: List[Dict[str, Any]], ass_path: str, template_name: str = "default",
                          strip_trailing_punct_last: bool = True) -> str:
    tpl = SUBTITLE_TEMPLATES.get(template_name, SUBTITLE_TEMPLATES["default"])
    info = tpl["script_info"]
    styles = tpl["v4_styles"]
    fmt = tpl["events_format"]

    os.makedirs(os.path.dirname(ass_path), exist_ok=True)

    # 마지막 줄 문장부호 제거 옵션
    if strip_trailing_punct_last and events:
        last = events[-1]
        last_text = re.sub(r"[.!?…]+$", "", last["text"]).strip()
        events[-1] = {**last, "text": last_text}

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
        f.write("\n[V4+ Styles]\n")
        for s in styles:
            f.write(s + "\n")
        f.write("\n[Events]\n")
        f.write(fmt + "\n")
        for seg in events:
            start = _ass_time(seg["start"])
            end = _ass_time(seg["end"])
            text = _sanitize_text_for_ass(seg["text"])
            # 기본 스타일(Default), 중앙 하단 정렬 가정
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

    return ass_path


# =========================
# SSML 안전 보정 유틸
# =========================
def _validate_ssml(ssml: str) -> str:
    """
    기본적인 SSML 유효성 보정:
    - <speak> 루트 보장
    - 연속 break 방지
    - 공백/개행 정리
    """
    t = (ssml or "").strip()
    if not t:
        return "<speak></speak>"

    # 루트 래핑 보장
    if not re.search(r"^\s*<\s*speak\b", t, flags=re.IGNORECASE):
        t = f"<speak>{t}</speak>"

    # 연속 break 축약
    t = re.sub(r"(</?speak>)", r"\1\n", t)  # 가독성
    t = re.sub(r"(<break[^>]*>\s*){2,}", r"<break time=\"200ms\"/>", t, flags=re.IGNORECASE)
    # prosody 중첩/중복 공백 정리
    t = re.sub(r">\s+<", "><", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def _ssml_safe_or_fallback(original_text: str, ssml: str) -> Tuple[str, bool]:
    """
    - SSML가 너무 빈약하거나 비정형이면 '원문을 안전 prosody로 감싼' 결정적 폴백을 사용
    - 반환: (ssml, forced_fallback)
    """
    t = (ssml or "").strip()
    if not t or len(re.sub(r"<[^>]+>", "", t).strip()) == 0:
        # 내용이 없으면 폴백
        safe = f"<speak><prosody rate=\"105%\" volume=\"medium\">{original_text}</prosody></speak>"
        return safe, True
    # 기타 간단 보정
    return _validate_ssml(t), False


def _summarize_line_pitch(ssml: str) -> float:
    """
    간단한 pitch 요약값(스타일/색상 선택 참고치). prosody pitch 속성 탐지.
    """
    m = re.search(r'pitch\s*=\s*"([^"]+)"', ssml, flags=re.IGNORECASE)
    if not m:
        return 0.0
    val = m.group(1)
    # 예: +10%, -5%, x-high 등
    pm = re.search(r"([+-]?\d+)\s*%", val)
    if pm:
        try:
            return float(pm.group(1)) / 100.0
        except Exception:
            return 0.0
    named = {"x-low": -0.4, "low": -0.2, "medium": 0.0, "high": 0.2, "x-high": 0.4}
    return named.get(val.strip().lower(), 0.0)


# =========================
# Audio helpers
# =========================
def merge_audio_files(audio_paths: List[str], out_path: str) -> List[Dict[str, float]]:
    """
    줄별 오디오 파일을 순서대로 하나로 합치고,
    각 구간의 start/end(초)를 반환한다.
    """
    if not audio_paths:
        return []

    segs: List[Dict[str, float]] = []
    cursor_ms = 0
    mixed = AudioSegment.silent(duration=0)

    for p in audio_paths:
        if not p or not os.path.exists(p):
            # 비어 있으면 300ms 공백을 삽입
            gap = AudioSegment.silent(duration=300)
            segs.append({"start": cursor_ms / 1000.0, "end": (cursor_ms + 300) / 1000.0})
            mixed += gap
            cursor_ms += 300
            continue

        clip = AudioSegment.from_file(p)
        dur = len(clip)
        start = cursor_ms
        end = cursor_ms + dur
        segs.append({"start": start / 1000.0, "end": end / 1000.0})
        mixed += clip
        cursor_ms = end

    # 출력 저장
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mixed.export(out_path, format="mp3")
    return segs


# =========================
# MAIN API
# =========================
def generate_subtitle_from_script(
    script_text: str,
    ass_path: str,
    provider: str = "polly",
    template: str = "default",
    polly_voice_key: str = "korean_female1",
    subtitle_lang: str = "ko",
    translate_only_if_english: bool = False,
    strip_trailing_punct_last: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str] | None, str]:
    """
    영상용 세그먼트 + SSML + 자막 생성 (LLM 2회 + TTS)
      1) 전체 대본 분절(LLM 1회) -> base_lines
      2) SSML 변환(LLM 1회)       -> ssml_list
      3) 줄별 TTS (LLM 없음)
      4) 병합 & 세그먼트 구성
      5) ASS 생성
    반환:
      segments_base(list[{start,end,text,ssml,pitch}]), audio_paths(list[str]) or None, ass_path(str)
    """

    # --- 1) 전체 대본 → 분절 라인 배열(LLM 1회)
    base_lines = breath_linebreaks_batch(script_text or "")
    base_lines = [ln.strip() for ln in base_lines if ln and str(ln).strip()]
    if not base_lines:
        return [], None, ass_path

    # --- 2) SSML 변환(LLM 1회) : 자막은 원문 유지, 발화용만 보정
    ssml_input_lines = [koreanize_if_english(ln) for ln in base_lines]
    ssml_list = convert_lines_to_ssml_batch(ssml_input_lines)

    # --- 3) SSML 안전 보정
    safe_ssml_list: List[str] = []
    for orig, frag in zip(base_lines, ssml_list):
        frag2, forced = _ssml_safe_or_fallback(orig, frag)
        safe_ssml_list.append(_validate_ssml(frag2))

    # Polly 엔진 사용 시 <speak> 래핑 확정 (보수적 보장)
    if provider.lower() == "polly":
        tmp: List[str] = []
        for s in safe_ssml_list:
            t = s.strip()
            if not re.search(r"^\s*<\s*speak\b", t, flags=re.IGNORECASE):
                t = f"<speak>{t}</speak>"
            tmp.append(t)
        safe_ssml_list = tmp

    # --- 4) 줄별 TTS(LLM 아님) → 오디오 병합 & 세그먼트 타임스탬프 생성
    audio_paths = generate_tts_per_line(
        safe_ssml_list,
        provider=provider,
        template=template,
        polly_voice_key=polly_voice_key,
    )
    if not audio_paths:
        return [], None, ass_path

    full_audio_file_path = os.path.join("assets", "auto", "_mix_audio.mp3")
    segments_raw = merge_audio_files(audio_paths, full_audio_file_path)

    # --- 5) base 세그먼트 구성 (자막=원문, 음성=SSML)
    n = min(len(segments_raw), len(base_lines), len(safe_ssml_list))
    if n <= 0:
        return [], audio_paths, ass_path

    segments_base: List[Dict[str, Any]] = []
    for i in range(n):
        s = segments_raw[i]
        orig_text = base_lines[i]        # ✅ 자막용: 원문 (예: "365km")
        ssml_text = safe_ssml_list[i]    # ✅ 발화용: SSML (예: "<speak>삼백육십오킬로미터</speak>")
        pitch_sum = _summarize_line_pitch(ssml_text)

        segments_base.append({
            "start": float(s["start"]),
            "end":   float(s["end"]),
            "text":  re.sub(r"\s+", " ", orig_text).strip(),
            "ssml":  ssml_text,
            "pitch": pitch_sum,
        })

    # --- 6) ASS 생성 (프레임 정렬 후 렌더)
    try:
        events = _quantize_segments(segments_base, fps=24.0)
        ass_path = generate_ass_subtitle(events, ass_path, template_name=template, strip_trailing_punct_last=strip_trailing_punct_last)
    except Exception as e:
        print(f"[error] ASS 생성 중 오류: {e}")

    return segments_base, audio_paths, ass_path
