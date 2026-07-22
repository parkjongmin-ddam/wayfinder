# 완전 로컬로 돌아가는 멀티소스 라우팅 RAG 에이전트를 만들고 측정하기

> Wayfinder — LangGraph로 라우팅·폴백·검증을 갖춘 RAG 에이전트를, GPU 6GB 로컬 PC에서
> API 키 없이 **결정적으로 재현 가능**하게 구축한 기록.

## 왜 만들었나

RAG 튜토리얼은 대부분 "만들었다"에서 끝난다. "내부 벡터검색 ↔ 외부 웹검색 라우팅/폴백"은
Adaptive/Corrective RAG의 표준 예제이고, 경로를 하나 더 넣는 것 자체는 차별점이 아니다.
이 프로젝트가 증명하려는 것은 세 가지다.

1. **라우팅을 측정한다** — routing accuracy, 폴백 발동률, 경로별 faithfulness.
2. **모든 결정을 눈으로 감사할 수 있게 한다** — 질의 1건당 한 줄 결정 트레이스.
3. **재현 가능하게 배포한다** — 링크(혹은 명령 몇 줄)만으로 남이 똑같이 돌린다.

여기에 더해, 이번 라운드의 목표는 **전체 파이프라인을 로컬 하드웨어에서 API 키 없이**
돌리는 것이었다. 라우터·grader·임베딩·답변·벡터스토어를 전부 로컬로 내렸다.

## 아키텍처: 노드는 그대로, 시임만 갈아 끼운다

그래프 토폴로지는 고정이다.

```
START → route → retrieve | web_search → grade → answer → verify → END
                          grade  → web_search   (1-hop 폴백)
                          verify → answer        (1-hop 재생성)
```

- **route** — LLM이 질의를 A(semantic)/B(keyword)/C(web)로 분류. 스키마 이탈 시 semantic으로
  안전 폴백(런타임 방어).
- **grade** — 검색 근거의 충분성을 판정. 부실하면 웹으로 **1-hop** 폴백.
- **answer** — 근거 기반 합성. 웹 발췌는 **데이터로만** 취급(프롬프트 인젝션 격리) + URL 인용.
- **verify** — 답변 후 **faithfulness 게이트**. 임계 미달이면 더 엄격한 프롬프트로 **1회 재생성**.

핵심 설계 원칙은 **모든 외부 의존을 프로토콜 뒤에 두는 것**이다. `Retriever`, `WebSearcher`,
`Grader`, 임베더, LLM 팩토리 — 전부 시임이다. 그래서 **같은 컴파일된 그래프**가 오프라인
(mock/stub) → 호스티드(OpenAI/Anthropic) → **완전 로컬(Ollama + pgvector)** 로,
노드 수정 0줄로 갈아탄다. 바뀌는 것은 config뿐이다.

## 로컬 스택으로 내리기

| 역할 | 로컬 모델 | 비고 |
|---|---|---|
| 라우터 / grader | `qwen2.5:3b` | 빠른 분류기, 100% GPU |
| 임베딩 | `nomic-embed-text` | 768차원 |
| 답변 합성 | `llama3.1:8b` | 근거 합성 |
| 벡터스토어 | pgvector (Docker) | 코사인 `<=>` |

측정 환경은 **GTX 1660 SUPER 6GB**. 여기서 나온 실측값:

- `qwen2.5:3b` — **86.8 tok/s, 100% GPU** (6GB에 완전 적재). 라우터/grader에 이상적.
- `llama3.1:8b` — **23.6 tok/s, 25%/75% CPU·GPU 분산** (5.6GB라 일부가 CPU로).

6GB VRAM에서 8B는 완전히 안 올라가지만, 라우터/grader를 3B로 내리면 분류·판정은 GPU에서
빠르게, 답변만 8B로 처리하는 구성이 자연스럽게 나온다. `LLM_PROVIDER=ollama` 하나로
provider-aware 기본값이 이 조합을 자동 선택한다.

## grader를 "seam"에서 "진짜 판정기"로

grader는 provider 시임 덕에 `LLM_PROVIDER=ollama`이면 자동으로 로컬 `qwen2.5:3b`에 붙는다.
문제는 초기 구현이 **binary**(SUFFICIENT/INSUFFICIENT → 0.85/0.25 고정)라 거칠었다는 것.

이를 **RAGAS식 등급 판정**으로 승격했다: 1~5점 근거 충분성 → 0~1 점수로 매핑하고
`grade_threshold`와 비교. 파싱 실패 시에는 라우터의 off-schema 방어와 같은 철학으로
**중립(sufficient) 기본값**을 두되, 답변 후 `verify` faithfulness 게이트가 백스톱이 된다.

라이브 결과:
- in-corpus "What is chunking?" → 로컬 grader가 **5/5** → 폴백 없이 corpus 근거 답변.
- out-of-corpus "capital of France?" → corpus 저평가 → **웹 폴백**.

## 소형 로컬 모델의 함정과 결정성

작은 모델은 jagged intelligence를 갖는다. 실제로 초기 데모에서 3B 라우터가 에러코드 질의를
keyword(B) 대신 semantic(A)으로 오분류했고, grader 판정은 실행마다 흔들려 같은 질의가
어떤 날은 근거 답변, 어떤 날은 웹 폴백으로 갈렸다.

해법은 **temperature=0을 기본값으로** 두는 것이었다. 라우터·grader는 분류기이고 답변은
근거 기반이므로 0.0이 옳은 기본이며, 무엇보다 **결정 트레이스가 재현 가능**해진다.
적용 후 `demo_trace.py`는 **두 번 실행해도 byte-identical**이다 — 소형 로컬 모델에서도.

```
A semantic  | route=A | fallback=no  | faith=0.76
B keyword   | route=B | fallback=no  | faith=0.70
C web       | route=C | fallback=no  | faith=0.17
A→fallback  | route=A | fallback=web | faith=0.18
```

## 배운 것 / 남은 것

- **시임 설계가 전부다.** 프로토콜 뒤에 의존을 숨겨두니 로컬화가 "코드 재작성"이 아니라
  "config 스위치"가 됐다. 노드는 한 줄도 안 바뀌었다.
- **소형 모델엔 결정성 장치가 먼저다.** temperature=0 + 스키마 이탈/파싱 실패 기본값 같은
  런타임 방어가 없으면 데모조차 재현이 안 된다.
- **측정이 서사를 만든다.** "만들었다"가 아니라 "6GB에서 3B는 87 tok/s, 8B는 24 tok/s로
  돌고, 라우팅은 세 경로가 결정적으로 분기한다"가 이야기가 된다.

남은 것: route C 실제 웹 답변(Tavily 연결), 관리형 배포, 데모 영상.

## 재현하기

```sh
ollama pull qwen2.5:3b llama3.1:8b nomic-embed-text
docker run -d --name wayfinder-pgvector -e POSTGRES_PASSWORD=wayfinder \
  -e POSTGRES_DB=wayfinder -p 5433:5432 pgvector/pgvector:pg17
pip install -e ".[ollama,pgvector]"
# .env: LLM_PROVIDER=ollama, PGVECTOR_CONNINFO=postgresql://postgres:wayfinder@localhost:5433/wayfinder
python scripts/ingest.py
python scripts/demo_trace.py
```

API 키 0개. 두 번 돌리면 같은 트레이스가 나온다.
