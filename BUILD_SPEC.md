# langconnect-agent — Build Spec (Claude Code 핸드오프)

> 이 문서는 langconnect-v2를 채용 포트폴리오용 통합 에이전트로 발전시키는 작업의
> 단일 기준 문서입니다. 대화 맥락 없이 이 문서만으로 작업을 이어갈 수 있도록
> self-contained로 작성되었습니다. `CLAUDE.md`로 리포지토리 루트에 두어도 됩니다.

---

## 1. 목적과 배경

- 작성자는 IAM/ADFS 엔지니어(5년)로, AI/RAG 엔지니어로 커리어 전환 중.
- 최근 AI 포지션 서류 탈락. 사유의 핵심은 "역량 부족"이 아니라 **"즉시 투입 가능한 실전 배포 증거 부족"** — 즉 프로젝트들이 "학습·실습"으로 프레이밍되어 있고 **배포된 완성작이 없다는 것**.
- 따라서 이 프로젝트의 목표는 새 기술 학습이 아니라, **이미 가진 재료를 하나의 배포·운영된 포트폴리오로 완성**하는 것.

## 2. 무엇을 만드는가 — 축 1·2·3 통합

langconnect-v2를 기반으로 세 역량 축을 **하나의 배포작 안에서 맞물리게** 한다.

| 축 | 내용 | 재료 상태 |
|---|---|---|
| 1. RAG 품질/평가 | pgvector 파이프라인 + RAGAS 다전략 평가 | 이미 보유 → 재사용 |
| 2. 에이전트/오케스트레이션 | LangGraph 다중 소스 라우팅 + 폴백 | 신규 (핵심 갭) |
| 3. MCP/통합 | 완성 에이전트를 MCP 서버로 노출 | 기존 FastMCP 자산 승격 |

세 축을 따로 증명하면 "조금씩 할 줄 안다"는 얕은 인상이 되지만, 한 배포작에서
맞물리면 **"RAG 파이프라인을 에이전트로 오케스트레이션하고 MCP로 노출해 배포까지 한 사람"**
이라는 이야기가 된다. 후자가 목표 직무의 JD와 일치한다.

## 3. 명시적 비(非)목표 — Out of Scope

- **권한/IdP/ADFS 각도 전면 배제.** 작성자의 IAM 배경을 전면에 내세우지 않는다.
  이 프로젝트는 순수하게 RAG·LLM·MCP 역량을 증명한다. (초기 논의에서 permission-aware /
  authz 방향을 검토했으나 폐기함.)
- **과(過)범위 금지.** 각 축은 "증명에 필요한 최소"로 자른다. 화려함이 아니라
  "끝까지 배포됐다"가 이 포트폴리오의 전부다.

## 4. 차별성은 어디서 나오는가 (중요 — 오해 방지)

**차별성은 경로 개수나 툴 개수에서 나오지 않는다.** 특히:

- **"내부 벡터검색 ↔ 외부 웹검색 라우팅/폴백"은 LangGraph 튜토리얼의 표준 예제**
  (Adaptive RAG / Corrective RAG)다. 웹검색 경로(C)를 넣는 것은 에이전트 서사를
  완성하는 **무대**일 뿐, 그 자체로 차별점이 아니다. C를 "차별점"으로 취급하지 말 것.

차별성은 다음 세 곳에서만 나온다:

1. **RAGAS 자산으로 라우팅을 정당화** — 경로 분기·폴백 트리거 기준을 기존 다전략
   실험 데이터로 근거한다. 라우터를 감이 아니라 측정값으로 설계한다.
2. **라우팅 자체를 평가** — routing accuracy, 폴백 발동률, 경로별 faithfulness를 측정한다.
   "만들었다"가 아니라 "측정하고 개선했다".
3. **배포·관측·완성도** — 이번 탈락의 실제 사유였던 지점. 가장 큰 차별점.

## 5. 아키텍처

### 5.1 에이전트 구조 (축 2)
- **라우터 노드 (LLM 기반, 빠른 모델)**: 질의 의도를 분류해 경로 선택. 규칙 기반
  if-else가 아니라 LLM 분류여야 한다(이게 "라우터 함수 하나"로 안 보이게 하는 핵심).
  - **라우터 기본 폴백 (런타임 방어).** 라우터는 비결정적 LLM이므로 출력이 A/B/C
    스키마를 벗어나거나 파싱 실패하면 **기본 경로(semantic, A)로 폴백**한다. 정확도
    "측정"(Phase 4)과 별개로, 측정 이전에 안전값부터 둔다. (근거: LLM은 할루시네이션·
    jagged intelligence를 갖는다는 전제로 방어선을 먼저 친다 — Software 3.0 검토 반영.)
- **3경로**:
  - A) semantic 벡터검색 (개념형 질의)
  - B) keyword·hybrid 검색 (고유명사·정확 매칭)
  - C) 외부 웹검색 — **Tavily** (코퍼스에 없는 최신 정보)
    - **인젝션 격리 (필수).** C 경로는 외부 콘텐츠를 답변 합성 노드에 주입하므로
      프롬프트 인젝션 표면이다. 웹 발췌는 **데이터로만 취급하고 지시문으로 취급하지
      않는다**(시스템 프롬프트로 격리 + 출처 URL 표기). 최소 조치이지만 리뷰어가
      정확히 찌르는 지점이라, 오히려 4절 '완성도' 차별점으로 홍보한다.
- **답변 합성 노드 (강한 모델)**: 검색 결과에 근거해 답변. provider 교체 가능하게.
  웹(C) 발췌를 받을 때는 위 인젝션 격리 원칙을 따른다(발췌 = 데이터, 지시문 아님).
- **폴백 루프 (self-correcting)**: 선택 경로의 근거가 부실하면 대체 경로로 폴백.
  **1-hop으로 상한** (예: 내부 → 웹). 폴백 판정에 RAGAS식 근거 충분성 평가를 붙인다.

### 5.2 서빙 아키텍처 (확정)
- 프론트엔드(agent-chat-ui)는 **LangGraph Server가 자동 노출하는 API**와 통신.
  에이전트용 HTTP 엔드포인트를 FastAPI로 직접 짜지 않는다.
- 기존 langconnect-v2의 FastAPI 자산: 검색/RAG 로직을 **그래프 노드 안으로 흡수하고
  FastAPI는 최소화/제거**. 별도 서비스로 유지할 이유가 없으면 걷어낸다.
- 즉 이 프로젝트의 API 계층 = LangGraph Server.

### 5.3 모델 정책
- 라우터 = 빠른 모델(예: Claude Haiku / GPT-4o-mini급). 답변 = 강한 모델(Claude/GPT 프런티어).
- provider는 `LLM_PROVIDER`로 교체(mock/anthropic/openai). 두 provider를 동시에 쓰는 게
  아니라 **갈아 끼울 수 있음**을 보여주는 설계.

### 5.4 결정 트레이스 / 시각적 검증 (Software 3.0 검토 반영 — 갭 1)
- Karpathy가 검증 가속의 1순위 지렛대로 꼽는 것은 **사람이 눈으로 감사하는 화면**
  ("vision is a highway to the brain")이다. 이 프로젝트의 핵심 산출물이 "라우팅 결정"인
  만큼, 그 결정을 **개발 중에(=Phase 2~3) 빠르게 눈으로 확인**할 수 있어야 한다.
- **처방**: 완성형 웹 UI(Phase 6)와 별개로, **초경량 결정 트레이스**를 Phase 2에
  먼저 넣는다. CLI/노트북에서 질의 1건당 다음을 한 줄로 출력:
  `{질의 → 선택 경로(A/B/C) → 라우터 근거 → grade 점수 → 폴백 여부}`.
  - 이는 agent-chat-ui의 "경로/폴백 표시" 기능의 **최소 버전을 Phase 2로 당긴 것**이며,
    Phase 6의 UI는 이 트레이스를 시각화하는 것으로 확장된다(중복 작업 아님).
  - 이 트레이스는 LangSmith trace(Phase 2 게이트)와 상호보완: LangSmith는 사후 분석,
    로컬 트레이스는 개발 루프 중 즉시 확인용.

## 6. 확정된 결정 로그

- 대상 프로젝트: **langconnect-v2** (신규 0-베이스 아님, 기존 자산 발전)
- 에이전트 유형: **다중 소스 라우팅 + 폴백 하이브리드**
- C 경로 웹검색 API: **Tavily**
- 라우터 LLM: **빠른 모델**, 답변: **강한 모델**, provider swap 가능
- API 계층: **LangGraph Server**, 기존 FastAPI는 노드로 흡수
- 배포: 공개 접근(일반 사용자 링크 접근) 확정. **관리형(LangGraph Platform) 권장**.
  - self-host(azlab 등 본인 인프라 공개 배포)는 선택적 대안. **로컬 개발과는 다른 단계.**
- 관측/평가: **LangSmith** (기존에 shelve했던 Langfuse 대체)

## 7. 현재 코드 상태 — Phase 1 스켈레톤 (완료·검증됨)

단일 semantic 경로 그래프 골격이 구현되어 있고, 컨테이너에서 실행 검증 완료
(smoke test 2건 통과, topology `START → retrieve → answer → END`).
**DB·API 키 없이** Stub 리트리버 + mock LLM으로 실행된다.

```
src/langconnect_agent/
  state.py        # AgentState — routing/fallback 필드까지 미리 정의(스키마 안정성)
  config.py       # 라우터/답변 모델 분리, top_k, max_fallbacks
  retrievers.py   # Retriever 프로토콜 + StubRetriever + LangConnectRetriever(연결 seam)
  llm.py          # provider-agnostic 팩토리 (mock | anthropic | openai)
  nodes.py        # Phase 1: retrieve(semantic) + answer
  graph.py        # build_graph() + module-level `graph` (langgraph dev용)
tests/
  test_graph_smoke.py
langgraph.json / pyproject.toml / .env.example / README.md
```

### 실행
```bash
pip install -e .          # 실모델: pip install -e ".[anthropic]"
cp .env.example .env      # 기본 LLM_PROVIDER=mock (오프라인)
pytest -q                 # Phase 1 게이트: 2건 통과
```

### 연결 seam (→ langconnect-v2) — 가장 중요
그래프는 `retrievers.py`의 `Retriever` 프로토콜에만 의존한다:
```python
def search(self, query: str, k: int = 5, route: str = "semantic") -> list[Document]: ...
```
`LangConnectRetriever.search()`를 실제 pgvector로 채우면(질의를 수집 시점 임베딩 모델로
임베딩 → 유사도 검색 → 행을 `Document`로 매핑) **노드 수정 없이 Phase 1 parity 도달**.

## 8. 로드맵 (staged — 각 Phase는 검증 게이트 통과 전 다음으로 넘어가지 않음)

### Phase 0 — 베이스라인 잠금
- 현행 RAG 파이프라인 로컬 재현
- 고정 평가셋: 3경로 유형 커버(개념형/고유명사형/최신정보형)
- 현재 검색 출력 + 전략별 RAGAS 점수 스냅샷
- **게이트**: 평가셋 재현 가능 + 베이스라인 지표 저장

### Phase 1 — LangGraph 그래프화 (단일 경로, 동등성) — *스켈레톤 완료, parity 대기*
- State 정의(완료), 단일 semantic 경로 end-to-end(완료)
- **남은 것**: StubRetriever 자리에 실제 pgvector 연결
- **게이트**: 단일 경로 출력이 Phase 0 베이스라인과 일치(parity)

### Phase 2 — 다중 소스 라우팅 (축2 핵심)
- 라우터 노드(빠른 LLM 의도 분류 → 경로 선택), **스키마 이탈 시 semantic 기본 폴백**(5.1)
- 경로 A/B/C(Tavily) 구현, 답변 노드에 provider swap 지점
- **C 경로 인젝션 격리**(5.1): 웹 발췌 = 데이터로만 취급 + 출처 URL 표기
- **초경량 결정 트레이스**(5.4): `질의 → 경로 → 근거 → grade → 폴백` 한 줄 출력
- LangSmith에 라우팅 결정 trace
- **게이트**: 3개 질의 유형이 각각 의도한 경로로 라우팅 + 결정 트레이스로 눈으로 확인
  가능 + 라우터가 깨진 출력을 뱉어도 semantic으로 안전 폴백

### Phase 3 — 폴백 루프 (self-correcting)
- 근거 충분성 판정 노드(grade, RAGAS식)
- 부실 시 대체 경로 폴백, **1-hop 상한**
- **게이트**: 내부 근거 약한 질의가 웹으로 폴백해 정상 복구

### Phase 4 — 평가 레이어 (차별성)
- Phase 0 평가셋을 LangSmith dataset 등록
- routing accuracy / 폴백 발동률 / 경로별 faithfulness 측정
- 라우팅 기준을 RAGAS 데이터로 정당화하는 문서
- **게이트**: 라우팅 정확도 + 폴백 동작 + 경로별 품질이 대시보드로 확인

### Phase 5 — MCP 노출 (축3)
- 완성 그래프를 MCP 서버로 래핑(기존 FastMCP "검색 MCP" → "에이전트 MCP" 승격)
- Claude Desktop 등록 후 실제 툴 호출 검증
- **게이트**: Claude Desktop에서 에이전트 MCP 툴 호출 end-to-end 동작

### Phase 6 — 프론트엔드 + 배포
- agent-chat-ui + useStream, UI에 선택 경로 + 폴백 여부 표시
- checkpointer(Postgres)로 스레드 지속성
- 관리형(LangGraph Platform) 배포 권장, 프론트는 Vercel
- **게이트**: 외부인이 링크만으로 3경로 데모 재현

### Phase 7 — 패키징
- README(아키텍처 다이어그램) + 블로그 + 데모 영상 + LangSmith 스크린샷
- 이력서 문구 재작성: "학습·실습" → "배포·운영"
- **게이트**: 링크만으로 재현 가능

## 9. 스코프 안전장치 (미완성 방지)
- 에이전트 노드 4~5개, MCP 툴 1~2개, 폴백 1-hop.
- Phase 1은 반드시 단일 경로부터. 3경로를 동시에 짜기 시작하면 뼈대가 흔들린다.
- 각 축은 최소로 자른다. "배포됐다"가 완성도의 기준.

## 10. Claude Code 즉시 다음 작업 (택1)
1. **Phase 1 parity 마무리**: `LangConnectRetriever.search()`를 실제 langconnect-v2
   pgvector에 연결. 필요 입력: 기존 리트리버 함수/모듈, DB 연결·세션 코드, 임베딩 모델
   설정, 테이블/컬럼 스키마.
2. **Phase 2 착수**: Stub 위에서 라우터 노드 + 3경로(A/B/C, Tavily) 분기 + 조건부 엣지 구현.

## 11. 병행 학습
LangChain Academy LangGraph 코스를 Phase 1~3와 병행. 수동 시청이 아니라 각 개념을
해당 Phase에 즉시 적용하는 방식으로만.

## 12. 미확정 항목
- 관리형 vs self-host 최종 택1 (Phase 6 진입 시 요금 확인 후 결정)
- 답변 합성 기본 모델을 Claude로 둘지 GPT로 둘지 (swap 가능하므로 우선순위 낮음)
- keyword·hybrid(B) 검색 구현 방식 (langconnect-v2 기존 지원 여부에 따라)
