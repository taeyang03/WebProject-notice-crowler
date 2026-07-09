# LangGraph 기반 대학 공지사항 RAG 및 관리 시스템

본 프로젝트는 LangGraph와 FastAPI를 활용하여 대학 공지사항을 효율적으로 검색하고 요약하여 제공하는 RAG(Retrieval-Augmented Generation) 시스템입니다. 기간이 만료되었거나 일정 기간(180일)이 경과한 데이터는 검색 대상에서 제외하여 정보의 최신성과 신뢰성을 유지합니다.

---

## 1. 환경 설정 (Environment Setup)

본 시스템을 구동하기 위해 필요한 의존성 패키지 설치 및 환경 변수 설정 안내입니다. Python 3.10 이상의 환경을 권장합니다.

### 필수 패키지 설치
터미널에서 아래 명령어를 실행하여 핵심 라이브러리를 설치합니다.

```bash
pip install fastapi uvicorn pydantic dotenv slowapi
pip install langchain langchain-core langchain-openai langchain-chroma langgraph
