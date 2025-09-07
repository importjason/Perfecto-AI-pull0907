# ssml_converter.py — FINAL (batch-only, JSON I/O, safe)
from __future__ import annotations

import json
import re
from typing import List

# === 프로젝트 공용 LLM 호출 래퍼 ===
#   - 모든 LLM 호출은 이 함수 1곳만 사용
from persona import generate_response_from_persona as _llm


# === 공용 유틸 ===
def _complete_with_any_llm(prompt: str) -> str:
    """
    프로젝트 공용 LLM 호출. 반드시 문자열만 반환하도록 강제.
    """
    out = _llm(prompt)
    # Streamlit / LangChain 등에서 dict가 올 수 있으므로 방어
    if not isinstance(out, str):
        try:
            out = json.dumps(out, ensure_ascii=False)
        except Exception:
            out = str(out)
    return out.strip()


def _json_loads_strict(raw: str):
    """
    모델이 설명/마크다운을 섞는 사고 대비:
    - 첫 번째 '[' 부터 마지막 ']' 구간만 추출하여 JSON 파싱 시도
    - 실패 시 개행 단위로 list를 구성(임시 폴백)
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        # 설명이 앞뒤에 섞였을 경우 bracket strip
        l = raw.find('[')
        r = raw.rfind(']')
        if l != -1 and r != -1 and l < r:
            try:
                return json.loads(raw[l:r+1])
            except Exception:
                pass
        # 최후 폴백: 줄 단위로 리스트 구성
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def _has_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))


def koreanize_if_english(text: str) -> str:
    """
    자막은 원문 그대로 유지하고, '발화용 입력'만 이 함수를 거칩니다.
    - 한국어가 이미 섞여 있으면 그대로 반환
    - 전부 영문/숫자라면 SSML 쪽에서 자연 발화가 되도록 최소 치환
      (단위 몇 개만 한글 단위로 치환: km, m, kg 등)
    - 숫자 → 한글 숫자 표기는 LLM(SSML 변환 프롬프트)에서 처리
    """
    t = (text or "").strip()
    if not t:
        return t
    if _has_korean(t):
        return t

    # 최소 단위 치환 (너무 과도하게 바꾸지 않는다)
    # 예: 365km -> 365 킬로미터 (숫자 한글화는 SSML 단계에서 처리)
    unit_map = {
        r"\bkm\b": " 킬로미터",
        r"\bm\b": " 미터",
        r"\bkg\b": " 킬로그램",
        r"\bg\b": " 그램",
        r"\bcm\b": " 센티미터",
        r"\bmm\b": " 밀리미터",
        r"\bmi\b": " 마일",
        r"\blb\b": " 파운드",
        r"\bft\b": " 피트",
        r"\bin\b": " 인치",
        r"\bh\b": " 시간",
        r"\bmin\b": " 분",
        r"\bs\b": " 초",
        r"\b%": " 퍼센트",
    }
    out = t
    for pat, rep in unit_map.items():
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)

    # 중복 공백 정리
    out = re.sub(r"\s+", " ", out).strip()
    return out


# === 프롬프트 (기존 본문을 유지하면서, 배치/JSON I/O만 강제) ===

# 1) 호흡/분절 프롬프트
BREATH_PROMPT = """
역할: 너는 한국어/다국어 대본을 숏폼 영상용 '호흡 단위'로 분절하는 편집기다.
출력은 오직 JSON 배열(lines)만 내야 한다. 마크다운/설명/주석을 포함하지 마라.

[규칙]
- 원문 보존: 단어/어미/문장부호 변경 금지. 분절만 수행.
- 입력의 개행(\n)은 '상위 문단 경계'로만 간주. 문단 내부에서만 호흡 분할.
- 빈 문자열 요소를 만들지 말 것.
- 공백 정리 외 텍스트 변형 금지.
"""

# 2) SSML 변환 프롬프트
SSML_PROMPT = """
역할: 너는 한국어 대본을 Amazon Polly/일반 TTS에서 안정적으로 읽히는 SSML로 변환하는 변환기다.
출력은 오직 JSON 배열(ssml_list)만 내야 한다. 마크다운/설명/주석을 포함하지 마라.

[불변 규칙]
1) 입력 라인 수 == 출력 SSML 수 (길이와 순서 완전 동일)
2) 각 SSML은 <speak>...</speak> 루트로 감싼다
3) 허용 태그: <speak>, <prosody>, <break>
4) 원문 의미를 바꾸지 말 것. 불필요한 문장 합치기/분할/재배열 금지
5) 숫자+단위가 있으면 한국어 자연 발화가 되도록 적절히 읽히게 조정 가능
6) 과도한 <break> 연쇄 금지(연속 2회 이상 금지), rate/volume은 과하게 조정하지 말 것
"""


# === 공개 API (배치 2개) ===

def breath_linebreaks_batch(script_text: str) -> List[str]:
    """
    전체 대본 1개를 입력하고, LLM에서 분절된 라인 배열을 JSON으로 1회 반환.
    - 출력: ["라인1", "라인2", ...]
    - 빈 요소 제거/공백 정리 수행
    """
    text = (script_text or "").strip()
    if not text:
        return []

    prompt = f"""
{BREATH_PROMPT}

[입력/출력 형식(중요)]
- 입력은 전체 대본 1개(개행 포함 가능)
- 출력은 JSON 배열 lines: ["라인1","라인2",...]
- 마크다운/설명/주석 금지. 오직 JSON만.
- 빈 문자열 요소 금지

입력:
<<<SCRIPT
{text}
SCRIPT>>>
"""
    raw = _complete_with_any_llm(prompt)
    lines = _json_loads_strict(raw)

    # 정리
    if not isinstance(lines, list):
        lines = [str(lines)]
    lines = [str(ln).strip() for ln in lines if isinstance(ln, (str, int, float)) and str(ln).strip()]
    return lines


def convert_lines_to_ssml_batch(lines: List[str]) -> List[str]:
    """
    라인 배열을 통째로 입력하고, 동일 길이의 SSML 배열(JSON)을 1회 반환.
    - 출력: ["<speak>...<speak>", ...]
    - 길이/순서 동일성 검증
    """
    if not lines:
        return []

    # 발화용 입력(영문/숫자만인 라인은 최소 한글 단위 치환)
    speak_inputs = [koreanize_if_english(ln) for ln in lines]

    prompt = f"""
{SSML_PROMPT}

[입력/출력 형식(중요)]
- 입력은 JSON 배열 lines: ["라인1","라인2",...]
- 출력은 JSON 배열 ssml_list: ["<speak>..</speak>", ...]
- 배열 길이와 순서는 반드시 동일
- 마크다운/설명/주석 금지. 오직 JSON만.

입력(JSON):
{json.dumps({"lines": speak_inputs}, ensure_ascii=False)}
"""
    raw = _complete_with_any_llm(prompt)
    ssml_list = _json_loads_strict(raw)

    if not isinstance(ssml_list, list):
        raise ValueError("SSML 배치 결과가 JSON 배열이 아닙니다.")
    if len(ssml_list) != len(lines):
        raise AssertionError(f"SSML 배열 길이 불일치: {len(ssml_list)} != {len(lines)}")

    # <speak> 래핑 보장 & 기본 정리
    out: List[str] = []
    for s in ssml_list:
        t = (s or "").strip()
        if not t:
            t = lines[len(out)]  # 비었으면 원문으로 폴백(아래에서 <speak>로 감쌈)
        if not re.search(r"^\s*<\s*speak\b", t, flags=re.IGNORECASE):
            t = f"<speak>{t}</speak>"
        out.append(t)
    return out
