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
역할: 호흡 라인브레이커. 줄바꿈만 조정. 원문/문장부호 불변. 출력은 최종 텍스트만.

규칙
0) 원문 완전 보존: 글자·공백·숫자·단위·어미·어순 변경 금지. 줄바꿈(\n)만 추가.
1) 포맷 정리: 빈 줄 금지(연속 개행 없음); 각 줄 앞뒤 공백 제거; 중복 공백은 1칸으로.
2) 값+단위/부호 결합 유지: 6,371킬로미터, 696,342킬로미터, -173도, 100배 등은 한 줄에서 분리 금지.
3) 보조용언·문말 결합 유지: “… 수 있/없습니다”, “… 것입니다/이다/해야 한다/할 수 없다”는 중간에서 끊지 말고 한 줄.
4) 라벨→값 분리: “X는” 다음 줄에 “값 …입니다/이다” 배치. (예: “태양 반지름은” ⏎ “696,342킬로미터입니다.”)
5) 질문 규칙:
   5-1) 머리말 단독: “왜일까요/무엇일까요/어떻게 될까요”는 단독 줄, 설명은 다음 줄.
   5-2) ‘?’ 뒤 줄바꿈 허용(질문 끝 새 리듬 시작). 쉼표/물음표 등 문장부호는 원문 그대로 유지.
6) 전환·담화 표지:
   6-1) {대신, 하지만/그러나, 그리고/또한, 따라서/즉/그러니까, 한쪽으로는/다른 쪽으로는}이 문두면 단독 줄 허용(강조).
   6-2) 단, 뒤가 아주 짧은 주어·지시어면 같은 줄로 묶음(예: “하지만 우리는”, “한쪽으로는 태양의”, “다른 쪽으로는 태양이”).
7) 쉼표 뒤 선택 분리: 의미 단위가 분명하면 쉼표 뒤에서 줄바꿈 가능. 단, 숫자 천단위 쉼표는 예외(분리 금지).
8) 길이 가이드: 한 줄 3–6단어 또는 8–18자 권장. 1–2단어 라인 남발 금지(“대신” 등 강조 단어는 예외).
9) 명사구/조사 경계: 명사구 내부·조사 바로 앞/뒤에서 어색한 분할 금지.
10) 첫·마지막 문장 보존: 입력의 **첫 문장**과 **마지막 문장**은 **단독 한 줄로 유지**하며 **내부 분절·병합 금지**(문장부호 포함 그대로).

검수: R2–R5 위반, 빈 줄/공백 정리, 숫자 천단위 쉼표 오분할 여부를 확인하고, 문제 있으면 줄바꿈만 재조정 후 최종본만 출력.

샷(예시)
A) 왜일까요, 단순히 커지는 것만으로 모든 것이 바뀝니다!
→
왜일까요,
단순히 커지는 것만으로
모든 것이 바뀝니다!

B) 태양 반지름은 696,342킬로미터입니다. 지구 반지름은 6,371킬로미터입니다.
→
태양 반지름은
696,342킬로미터입니다.
지구 반지름은
6,371킬로미터입니다.

C) 대신 무엇이 생기느냐면, 압력과 중력이 지배하는 죽음의 세계입니다.
→
대신
무엇이 생기느냐면
압력과 중력이 지배하는
죽음의 세계입니다.

D) 생명체는 살아남을 수 없습니다.
→
생명체는 살아남을 수 없습니다.

E) 하지만 우리는 태양을 잃었습니다.
→
하지만 우리는
태양을 잃었습니다.

"""

# 2) SSML 변환 프롬프트
SSML_PROMPT = """
[불변 규칙 — 반드시 지켜]
1) 원문 보존: 단어·어순·어미(경어체/평서체) 절대 변경 금지.
   - 각 문장의 끝 어미는 입력 그대로 유지한다. (예: "~습니다/~합니다/~이다/~다" 등)
   - 문장 끝 어미를 다른 형태로 바꾸지 말 것.
2) 숫자·단위 표기, 고유명사 그대로 유지.
3) 허용 태그: <speak>, <prosody>, <break>만.
4) 허용 문장부호: 물음표(?)와 쉼표(,)만. 마침표(.)/느낌표(!)/줄임표(…)는 출력 금지.
5) 일시정지 규칙:
   - 구(절) 사이: <break time="20ms"/>
   - 문장 사이: <break time="50ms"/>
   - 90ms 초과 금지, 20ms+50ms 연속 사용 금지(중복 브레이크 금지).
6) 변환은 ‘분할’만 한다. 재작성·의역·어휘 치환 금지.
   - 쉼표는 추가해도 되지만, 단어/어미는 그대로여야 한다.

[억양/속도 설계]
- 훅/질문/경고: rate 160~165%, pitch +15~+25%
- 일반 설명/정보: rate 140~155%, pitch -10%~+5%
- 결론/단정/무거운 문장: rate 130~140%, pitch -15%~-20%
- 같은 문장 내 2~3개의 구(절)로 분할하고, 의미가 고조되면 뒤 구절의 rate/pitch를 최대 +5%p 상향,
  침잠이면 최대 -5%p 하향.

[끝맺음 ‘말꼬리’ 짧게 (대본 수정 없이)]
- 문장 마지막 구절(원문 어미 그대로)에만 미세 조정:
  그 구절을 <prosody rate="원래값+5%" pitch="+3%">…</prosody>로 감싼 뒤,
  바로 <break time="50ms"/> 또는 다음 문장으로 넘어간다.
- 이 조정은 어미 텍스트를 바꾸지 않고 발화만 또렷하게 만든다.

[출력 형식]
- 최상위: <speak><prosody volume="+2dB"> … </prosody></speak>
- 각 구(절): <prosody rate="…" pitch="…">원문 일부(어미 포함, 원형 유지)</prosody>
- 구(절) 사이는 30ms, 문장 사이는 50ms. 중복 브레이크 금지.

[검증 체크리스트(내부 적용 후 통과된 경우에만 출력)]
- <speak> 루트, 허용 태그 외 사용 없음?
- break는 20ms/50ms만, 90ms 이하, 연속 중복 없음?
- 각 문장의 마지막 prosody 텍스트가 원문 마지막 어미와 ‘완전히 동일’한가?
- 재작성/치환/어미 변경 없이 원문 부분문자열로만 구성했는가?
- 물음표/쉼표 외의 마침표/느낌표/줄임표를 쓰지 않았는가?
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
