# main.py — FINAL (batch-only LLM calls: segmentation, SSML, image keywords)

import os
import hashlib
from typing import List, Dict, Any, Optional

import streamlit as st

# -------------------------------
# Imports (배치 함수 & 파이프라인들)
# -------------------------------

from generate_timed_segments import generate_subtitle_from_script

from keyword_generator import generate_image_keywords_per_line_batch  # 이미지 키워드 배치(1회)
from image_generator import generate_images_for_topic, generate_videos_for_topic
from video_maker import create_video_with_segments
from upload import upload_to_youtube
from persona import generate_response_from_persona  # 대본 전처리/개선이 필요한 경우 사용 (선택)
# from best_subject_subtitle_extractor import extract_best_subject  # 필요 시 사용

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Video Maker (Batch LLM x3)", layout="wide")

# -------------------------------
# Session guards / cache helpers
# -------------------------------
if "_job_running" not in st.session_state:
    st.session_state["_job_running"] = False

def _hash_text(s: str) -> str:
    h = hashlib.sha256()
    h.update((s or "").encode("utf-8"))
    return h.hexdigest()

def _cache_get(key: str, default=None):
    return st.session_state.get(key, default)

def _cache_set(key: str, val):
    st.session_state[key] = val

# -------------------------------
# UI
# -------------------------------
st.title("🎬 Video Maker — LLM Calls Fixed to 3x per Video")

with st.sidebar:
    st.subheader("TTS / Subtitles")
    tts_provider = st.selectbox("TTS Provider", options=["polly", "elevenlabs"], index=0)
    voice_template = st.selectbox("Voice Template", options=["default", "soft", "fast"], index=0)
    polly_voice_key = st.selectbox("Polly Voice", options=["korean_female1", "korean_male1", "default_male"], index=0)
    subtitle_lang = st.selectbox("Subtitle Lang", options=["ko", "en"], index=0)
    translate_only_if_english = st.checkbox("Translate only if line is English", value=False)

    st.subheader("Rendering")
    style = st.selectbox("Visual Style", options=["default", "emotional"], index=0)
    add_title_overlay = st.checkbox("Add Title Overlay", value=True)
    bgm_path = st.text_input("BGM path (optional)", value="")

    st.subheader("Upload")
    do_upload = st.checkbox("Upload to YouTube after render", value=False)
    youtube_title = st.text_input("YouTube Title", value="AI 자동 생성 영상")
    youtube_description = st.text_area("YouTube Description", value="AI로 생성된 숏폼입니다.")

st.markdown("#### Input Script")
script_text = st.text_area(
    "Paste your script here",
    height=220,
    placeholder="예) 365km 같은 표기는 자막으로는 그대로, 발음은 삼백육십오킬로미터로 읽히게 처리됩니다.",
)

col_btn1, col_btn2, col_btn3 = st.columns([1,1,2])
run_clicked = col_btn1.button("Generate video")
reset_clicked = col_btn2.button("Reset caches")

if reset_clicked:
    # 간단한 캐시 초기화
    for k in list(st.session_state.keys()):
        if str(k).startswith(("lines_", "ssml_", "kw_", "segments_", "_orig_lines_for_tts", "_used_br_lines")):
            del st.session_state[k]
    st.success("Caches reset.")

# -------------------------------
# Preview helpers (no extra LLM)
# -------------------------------
def _log_ssml_preview_from_segments(segments: List[Dict[str, Any]], idx: int = 0):
    """
    추가 LLM 호출 없이, 이미 생성된 segments에서 1개 라인의 SSML을 미리보기로 표시.
    """
    if not segments:
        return
    i = max(0, min(idx, len(segments)-1))
    st.write(f"🧪 [SSML 미리보기] index={i}, text='{segments[i]['text']}'")
    st.code(segments[i]["ssml"], language="xml")

# -------------------------------
# Run Job
# -------------------------------
def run_pipeline_job(job: Dict[str, Any]) -> Optional[str]:
    """
    전체 파이프라인
    LLM calls:
      - inside generate_subtitle_from_script: segmentation(1) + SSML(1)
      - below: generate_image_keywords_per_line_batch(1)
      => TOTAL 3
    """
    try:
        st.write("🚀 Starting job...")

        # 0) Input
        raw_script = (job.get("script_text") or "").strip()
        if not raw_script:
            st.error("❌ 입력 대본이 없습니다.")
            return None

        # (Optional) 1) Persona / Final script
        # 필요 시에만 사용. 불필요하면 주석 처리 가능.
        final_script = job.get("final_script")
        if not final_script:
            # 아래 라인 주석 처리하면 LLM 호출 0. (현재는 persona가 LLM일 수 있음)
            final_script = generate_response_from_persona(raw_script)

        # 2) Subtitles + SSML + audio (LLM: segmentation+SSML = 2 calls)
        ass_output = os.path.join("assets", "auto", "subtitles.ass")
        os.makedirs(os.path.dirname(ass_output), exist_ok=True)

        segments, audio_clips, ass_path = generate_subtitle_from_script(
            final_script,
            ass_output,
            provider=job.get("tts_provider", "polly"),
            template=job.get("voice_template", "default"),
            polly_voice_key=job.get("polly_voice_key", "korean_female1"),
            subtitle_lang=job.get("subtitle_lang", "ko"),
            translate_only_if_english=job.get("translate_only_if_english", False),
            strip_trailing_punct_last=True,
        )
        if not segments:
            st.error("❌ 세그먼트 생성 실패")
            return None

        # Show a preview without extra LLM
        with st.expander("SSML preview (no extra LLM)"):
            _log_ssml_preview_from_segments(segments, idx=0)

        # 3) Image keywords (LLM: 1 call)
        line_texts = [seg["text"] for seg in segments]
        key_kw = f"kw_{hash(tuple(line_texts))}"
        image_keywords = _cache_get(key_kw)
        if image_keywords is None:
            image_keywords = generate_image_keywords_per_line_batch(line_texts)  # LLM 1회
            _cache_set(key_kw, image_keywords)

        # 4) Media fetch (no LLM)
        image_paths: List[Optional[str]] = []
        if job.get("style") != "emotional":
            for kw in image_keywords:
                paths = generate_images_for_topic(kw, max_results=1)
                image_paths.append(paths[0] if paths else None)
        else:
            # emotional 스타일이면 영상 위주로
            for kw in image_keywords:
                vpaths = generate_videos_for_topic(kw, max_results=1)
                image_paths.append(vpaths[0] if vpaths else None)

        # 5) Compose video (no LLM)
        final_audio_path = "assets/auto/_mix_audio.mp3"  # 내부에서 사용/생성되는 경로일 수 있음
        output_video_path = os.path.join("assets", "auto", "video.mp4")
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        title_for_overlay = job.get("topic") or (line_texts[0] if add_title_overlay and line_texts else "")
        video_path = create_video_with_segments(
            image_paths=image_paths,
            segments=segments,
            audio_path=final_audio_path,
            topic_title=title_for_overlay if add_title_overlay else "",
            include_topic_title=add_title_overlay,
            bgm_path=job.get("bgm_path", ""),
            save_path=output_video_path,
            ass_path=ass_path,
        )

        if not video_path or not os.path.exists(video_path):
            st.error("❌ 영상 합성 실패")
            return None

        # 6) Upload (optional, no LLM)
        if job.get("upload", False):
            youtube_url = upload_to_youtube(
                video_path,
                title=job.get("youtube_title", "AI 자동 생성 영상"),
                description=job.get("youtube_description", "AI로 생성된 숏폼입니다.")
            )
            if youtube_url:
                st.success(f"✅ 업로드 완료: {youtube_url}")
                st.session_state["youtube_link"] = youtube_url
        else:
            st.success("✅ 영상 생성 완료")
            st.video(video_path)
            st.session_state["final_video_path"] = video_path

        return video_path
    except Exception as e:
        st.error(f"❌ 파이프라인 실행 중 오류: {e}")
        return None

# -------------------------------
# Button handler with rerun guard
# -------------------------------
if run_clicked and not st.session_state["_job_running"]:
    st.session_state["_job_running"] = True
    try:
        job = {
            "script_text": script_text,
            "tts_provider": tts_provider,
            "voice_template": voice_template,
            "polly_voice_key": polly_voice_key,
            "subtitle_lang": subtitle_lang,
            "translate_only_if_english": translate_only_if_english,
            "style": style,
            "bgm_path": bgm_path,
            "upload": do_upload,
            "youtube_title": youtube_title,
            "youtube_description": youtube_description,
        }
        run_pipeline_job(job)
    finally:
        st.session_state["_job_running"] = False

# -------------------------------
# Developer helpers (optional)
# -------------------------------
with st.expander("ℹ️ Notes"):
    st.markdown("""
- LLM 호출은 고정 3회입니다.
  1) `generate_subtitle_from_script` 내부의 **분절 1회**
  2) `generate_subtitle_from_script` 내부의 **SSML 1회**
  3) 본문 하단의 `generate_image_keywords_per_line_batch` **1회**
- 미리보기/로그는 **추가 LLM 호출 없음** — 이미 만들어진 segments에서 SSML을 가져옵니다.
- 자막(text)은 원문, 발음은 SSML을 사용하므로 "365km" 표기는 그대로, 발음은 "삼백육십오킬로미터"로 처리됩니다.
    """)

# -------------------------------
# Post-run panels (no extra LLM)
# -------------------------------
with st.expander("🔍 Generated keywords & assets (no extra LLM)"):
    if "final_video_path" in st.session_state:
        st.write("**Final video path:**", st.session_state["final_video_path"])
    if "youtube_link" in st.session_state:
        st.write("**YouTube:**", st.session_state["youtube_link"])

    # 가능한 키워드 캐시 키들을 탐색해서 보여주기 (세션에 있으면 노출)
    try:
        kw_keys = [k for k in st.session_state.keys() if str(k).startswith("kw_")]
        if kw_keys:
            k = kw_keys[-1]
            kws = st.session_state[k]
            st.write(f"**Image keywords** (len={len(kws)}):")
            st.code("\n".join(kws[:50]), language="text")
    except Exception as e:
        st.write(f"Keyword cache view error: {e}")

# -------------------------------
# Local runner (optional)
# -------------------------------
if __name__ == "__main__":
    st.write("👋 Running in Streamlit context. Use the UI to start a job.")

