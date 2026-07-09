# 🚀 대학 공지사항 자율형 RAG 에이전트 및 통합 관리 시스템
> **과제명**: LangChain 및 LangGraph 기반 Agent 서비스 구현 프로젝트 (최종 평가 과제)

본 프로젝트는 **LangGraph**와 **FastAPI**를 융합하여 대학 학사 공지사항을 자율적으로 수집(Crawling), 의미론적 검색 및 요약(RAG), 그리고 시스템 제어 명령어 체계까지 결합한 **하이브리드 AI 학사 비서 에이전트**입니다. 사용자의 복잡한 자연어 요청 맥락을 스스로 이해하고 자율적으로 도구(Tool)를 호출하거나 RAG 파이프라인으로 라우팅하는 지능형 워크플로우를 갖추고 있습니다.

---

## ✨ 과제 주요 요구사항 반영 및 코드 분석 (Evaluation Matrix)

과제 공지사항의 필수 요건들이 실제 소스코드 내에 다음과 같이 견고하게 설계 및 반영되었습니다.

| 과제 요구사항 | 프로젝트 구현 내용 및 소스코드 매핑 위치 |
| :--- | :--- |
| **1. 자율적 도구 호출**<br>(최소 2개 이상의 Tool) | - `@tool` 데코레이터를 기반으로 에이전트 전용 도구 구현 완료<br>  - `get_current_date_and_time`: 실시간 일정 매칭을 위한 시간 정보 도구<br>  - `get_user_academic_context`: 사용자의 학과, 학년, 관심사 맞춤형 컨텍스트 도구<br>  - `server.py`의 `tools = [get_current_date_and_time, get_user_academic_context]` 및 `ToolNode` 연동 완료 |
| **2. RAG 파이프라인 구축**<br>(최소 1개 이상) | - **Chroma** 벡터 데이터베이스(`notice_collection`) 기반 시맨틱 검색 파이프라인 구축<br> - `OpenAIEmbeddings(text-embedding-3-small)` 모델을 이용해 크롤링된 공지사항 본문을 임베딩하고 자율 질의응답 및 요약 기능 수행 |
| **3. 멀티턴 대화 및 메모리** | - LangGraph 내장 체크포인터인 **`MemorySaver`**를 연동하여 세션(`thread_id`)별 독립적인 대화 맥락 유지<br> - `add_messages` 툴을 통한 상태 업데이트로 멀티턴 대화 제어 |
| **4. StateGraph 및 조건부 분기**<br>(Conditional Edge 필수) | - `StateGraph(AgentState)` 구조 설계<br> - 시스템 명령어(`!su`, `!kw add` 등) 감지 시 `command_node`로 분기하거나, 사용자 의도 분석 결과(`intent_type`)에 따라 자율적으로 RAG 검색, 요약, 혹은 일반 대화 노드로 라우팅하는 **조건부 분기(Conditional Edge)** 로직 완비 |
| **5. 미들웨어 적용**<br>(최소 1개 이상) | - **안정성/운영 관점의 다중 가드레일 미들웨어 도입**<br>  - `slowapi` (`Limiter`)를 활용한 API 속도 제한(Rate Limiting)으로 DoS 방지 및 가드레일 구축<br>  - `GZipMiddleware`를 통한 네트워크 리소스 최싱화<br>  - `tenacity` 라이브러리의 `@retry` 메커니즘을 크롤러 네트워크 통신 부에 적용하여 일시적 장애 복구력 확보 |
| **6. 구조화된 출력 파서**<br>(OutputParser/Pydantic) | - **Pydantic 구조화 출력을 멀티 지점에서 활용**<br>  - `UserIntent`: 사용자 요청의 키워드, 날짜 필터, 의도 유형 등을 파싱하여 RAG 전처리 수행<br>  - `KeywordExtraction`: LLM 크롤링 파트에서 공지 본문 내 태그를 3~5개 형태로 완벽히 구조화 출력(`with_structured_output`) |
| **7. API Key 분리 관리** | - `load_dotenv()` 기반 프로젝트 루트 내 `.env` 파일로 중요 자격 증명(`OPENAI_API_KEY`) 하드코딩 없이 철저히 분리 |

---

## 🛠️ 1. 환경 설정 (Environment Setup)

### 필수 패키지 설치
본 시스템은 Python 3.10 이상의 환경에서 정상 작동합니다. 터미널에서 다음 명령어를 실행하여 의존성 라이브러리(`requirements.txt`)를 설치해 주세요.

```bash
pip install fastapi uvicorn pydantic dotenv slowapi tenacity beautifulsoup4 httpx aiosmtplib
pip install langchain langchain-core langchain-openai langchain-chroma langgraph

```

### 환경 변수 설정 (`.env`)

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 아래와 같이 필수 API Key 및 에이전트 식별 정보를 입력합니다.

```env
# OpenAI API Key (텍스트 임베딩 및 자율 답변 생성용)
OPENAI_API_KEY=sk-proj-YourActualOpenAiApiKeyHere...

# LangChain 내부 User-Agent 정보 설정
USER_AGENT=notice-rag-agent/1.0

```

### 시스템 디렉토리 구조

```text
📂 project-root/
 ├── 📄 server.py                 # FastAPI 웹서버 & LangGraph Agent 핵심 비즈니스 로직
 ├── 📄 enterprise_crawler.py     # 대학 사이트 타겟팅 스케줄러 & LLM 키워드 추출 크롤러
 ├── 📄 .env                      # API 자격 증명 관리 파일 (보안 주의)
 ├── 📄 feedbacks.jsonl           # 사용자 평가/피드백 실시간 적재 데이터 스트림
 ├── 📂 chroma_db/                # 공지사항 벡터 및 유저 구독 정보가 지속 보관되는 Vector DB
 └── 📂 public/                   # 정적 웹 자원 및 인터랙티브 UI 서비스 컴포넌트
      └── 📄 index.html           # 학사 비서 웹 인터페이스 화면

```

---

## 🧠 2. 시스템 아키텍처 및 워크플로우 (Architecture & Workflow)

본 에이전트는 사용자의 자연어 입력이 들어오면 내부 상태망(`AgentState`)을 거쳐 동적으로 흐름을 결정합니다.

```mermaid
graph TD
    START([사용자 입력 요청]) --> A{명령어 검증 (! 시작여부)}
    
    A -->|Yes| B[command_node]
    B -->|사용자 관리 / 세션 전환 / 키워드 구독| END([즉시 응답 반환])
    
    A -->|No (자연어)| C[intent_analyzer_node]
    C -->|UserIntent Pydantic 분석| D{intent_type 조건부 분기}
    
    D -->|'summary' / 'search'| E[RAG_retrieval_node]
    E -->|Recency-Aware Post Filtering 반년이내 및 마감일 필터링| F[llm_generation_node]
    
    D -->|'general' / Tool 호출 필요 시| G[llm_with_tools_node]
    G -->|자율 도구 선택| H[tool_node]
    H --> F
    
    F --> END

```

### 핵심 차별화 기능 (Key Features)

1. **데이터 최신성 필터링 (Recency-Aware Post Filtering):**
유사도가 높은 벡터 데이터라 할지라도 메타데이터상의 마감일(`end_date`)이 현재 시점보다 과거이거나, 작성일(`date`) 기준 180일을 초과한 과거 공지사항은 포스트 필터링 로직을 통해 자동으로 컨텍스트에서 배제하여 정보의 신뢰성을 극대화합니다.
2. **동적 가변 결과 추출:**
사용자가 "공지사항 5개 보여줘"와 같이 구체적인 숫자를 자연어로 요청할 경우, Pydantic 파서(`result_count`)가 이를 파싱하여 유연하게 컨텍스트 검색 수량을 조절합니다.

---

## 💻 3. 사용법 및 실행 안내 (Usage Guide)

### 백엔드 서버 구동

```bash
python server.py

```

* 서버는 기본적으로 `http://127.0.0.1:8000` 주소에서 대기합니다.
* 웹 브라우저를 통해 해당 주소에 접속하면 세련된 UI의 `public/index.html` 모니터링 및 실시간 대화 화면을 만나보실 수 있습니다.

### 채팅창 지원 시스템 명령어 체계

채팅 입력란에 `!` 기호로 시작하는 명령어를 입력하여 데이터베이스 내 유저 세션 상태 및 키워드 구독 정보를 관리할 수 있습니다.

| 명령어 그룹 | 명령어 패턴 | 설명 |
| --- | --- | --- |
| **계정 및 세션 관리** | `!su [이름] [이메일]` | 해당 사용자 계정으로 로그인 및 세션(컨텍스트 권한)을 즉시 전환합니다. |
|  | `!whoami` | 현재 세션의 로그인 상태 및 등록된 맞춤 알림 키워드를 상시 확인합니다. |
|  | `!exit` | 현재 로그인된 계정에서 로그아웃 처리를 수행합니다. |
| **사용자 DB 제어** | `!user add [이름] [이메일]` | 새로운 사용자를 인메모리/벡터 시스템에 안전하게 등록합니다. |
|  | `!user ls` | 시스템에 등록되어 있는 전체 사용자 계정 리스트를 출력합니다. |
|  | `!user rm [이름] [이메일]` | 특정 사용자 데이터를 데이터베이스에서 안전하게 완전 삭제합니다. |
| **구독 키워드 제어** | `!kw add [키워드...]` | 현재 계정에 푸시 알림 타겟팅을 위한 맞춤형 학사 알림 키워드를 복수 등록합니다. |
|  | `!kw rm [키워드...]` | 기등록된 알림 키워드 중 불필요한 단어를 선택 삭제합니다. |
|  | `!kw ls` | 현재 계정이 구독 중인 모든 알림 키워드를 일목요연하게 조회합니다. |

---

## 📈 4. 크롤링 및 실시간 알림 스케줄러 (`enterprise_crawler.py`)

타겟팅 대학 사이트(`충북대학교 소프트웨어학과`, `충북대학교 홈페이지`)의 공지사항을 주기적으로 파싱하는 독립 백그라운드 프로세스입니다.

* **네트워크 폴트 톨러런스:** 지수 백오프 기반 `tenacity.retry` 메커니즘을 내장하여 간헐적 네트워크 차단 시 자동 재시도합니다.
* **LLM 기반 자동 태깅 및 벡터화:** 단순 텍스트 수집에 그치지 않고, 수집 시점에 `gpt-4o-mini` 모델을 사용해 핵심 태그 3~5개를 Pydantic 구조로 추출하여 메타데이터에 함께 빌드 및 ChromaDB에 적재합니다.
* **키워드 일치 푸시 알림:** 등록된 사용자의 관심 키워드와 신규 수집된 공지의 자동 추출 태그가 매칭될 경우, `aiosmtplib`를 활용한 이메일 알림 발송 로직이 비동기 병렬(`asyncio.gather`)로 즉각 트리거됩니다.

```bash
python enterprise_crawler.py
*(실운영 환경 가동 시 6시간 주기로 무한 루프 스케줄링이 활성화됩니다.)*
