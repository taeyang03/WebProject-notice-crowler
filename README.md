# 🚀 대학 공지사항 자율형 RAG 에이전트 및 통합 관리 시스템
> **과제명**: LangChain 및 LangGraph 기반 Agent 서비스 구현 프로젝트 (최종 평가 과제)

본 프로젝트는 **LangGraph**와 **FastAPI**를 융합하여 대학 학사 공지사항을 자율적으로 수집(Crawling), 의미론적 검색 및 요약(RAG), 그리고 시스템 제어 명령어 체계까지 결합한 **하이브리드 AI 학사 비서 에이전트**입니다. 사용자의 복잡한 자연어 요청 맥락을 스스로 이해하고 자율적으로 도구(Tool)를 호출하거나 RAG 파이프라인으로 라우팅하는 지능형 워크플로우를 갖추고 있습니다.

---

## ✨ 과제 주요 요구사항 반영 및 코드 분석 (Evaluation Matrix)

과제 공지사항의 필수 요건들이 실제 소스코드 내에 다음과 같이 견고하게 설계 및 반영되었습니다.

| 과제 요구사항 | 프로젝트 구현 내용 및 소스코드 매핑 위치 |
| :--- | :--- |
| **1. 자율적 도구 호출**<br>(최소 2개 이상의 Tool) | - `@tool` 데코레이터를 기반으로 에이전트 전용 도구 구현 완료<br>- `get_current_date_and_time`: 실시간 일정 매칭을 위한 시간 정보 도구<br>- `get_user_academic_context`: 사용자의 학과, 학년, 관심사 맞춤형 컨텍스트 도구<br>- `server.py`의 `tools = [get_current_date_and_time, get_user_academic_context]` 및 `ToolNode` 연동 완료 |
| **2. RAG 파이프라인 구축**<br>(최소 1개 이상) | - **Chroma** 벡터 데이터베이스(`notice_collection`) 기반 시맨틱 검색 파이프라인 구축<br>- `OpenAIEmbeddings(text-embedding-3-small)` 모델을 이용해 크롤링된 공지사항 본문을 임베딩하고 자율 질의응답 및 요약 기능 수행 |
| **3. 멀티턴 대화 및 메모리** | - LangGraph 내장 체크포인터인 **`MemorySaver`**를 연동하여 세션(`thread_id`)별 독립적인 대화 맥락 유지<br>- `add_messages` 툴을 통한 상태 업데이트로 멀티턴 대화 제어 |
| **4. StateGraph 및 조건부 분기**<br>(Conditional Edge 필수) | - `StateGraph(AgentState)` 구조 설계<br>- 시스템 명령어(`!su`, `!kw add` 등) 감지 시 `command_node`로 분기하거나, 사용자 의도 분석 결과(`intent_type`)에 따라 자율적으로 RAG 검색, 요약, 혹은 일반 대화 노드로 라우팅하는 **조건부 분기(Conditional Edge)** 로직 완비 |
| **5. 미들웨어 적용**<br>(최소 1개 이상) | - **안정성/운영 관점의 다중 가드레일 미들웨어 도입**<br>- `slowapi` (`Limiter`)를 활용한 API 속도 제한(Rate Limiting)으로 DoS 방지 및 가드레일 구축<br>- `GZipMiddleware`를 통한 네트워크 리소스 최적화<br>- `tenacity` 라이브러리의 `@retry` 메커니즘을 크롤러 네트워크 통신 부에 적용하여 일시적 장애 복구력 확보 |
| **6. 구조화된 출력 파서**<br>(OutputParser/Pydantic) | - **Pydantic 구조화 출력을 멀티 지점에서 활용**<br>- `UserIntent`: 사용자 요청의 키워드, 날짜 필터, 의도 유형 등을 파싱하여 RAG 전처리 수행<br>- `KeywordExtraction`: LLM 크롤링 파트에서 공지 본문 내 태그를 3~5개 형태로 완벽히 구조화 출력(`with_structured_output`) |
| **7. API Key 분리 관리** | - `load_dotenv()` 기반 프로젝트 루트 내 `.env` 파일로 중요 자격 증명(`OPENAI_API_KEY`) 하드코딩 없이 철저히 분리 |

---

## 🧠 2. 시스템 아키텍처 및 워크플로우 (Architecture & Workflow)

본 에이전트는 사용자의 자연어 입력이 들어오면 내부 상태망(`AgentState`)을 거쳐 동적으로 흐름을 결정합니다.

```mermaid
graph TD
    START([사용자 입력 요청]) --> A{"명령어 검증 (! 시작여부)"}
    
    A -->|Yes| B[command_node]
    B --> END([즉시 응답 반환])
    
    A -->|No| C[intent_analyzer_node]
    C --> D{"intent_type 조건부 분기"}
    
    D -->|summary / search| E[RAG_retrieval_node]
    E --> F[llm_generation_node]
    
    D -->|general| G[llm_with_tools_node]
    G --> H{"도구 호출 필요 여부"}
    H -->|Yes| I[tool_node]
    H -->|No| F
    I --> G
    
    F --> END
