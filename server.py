import json
import os
import datetime
import time
from pathlib import Path
from typing import List, Optional, AsyncGenerator, TypedDict, Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

# LangGraph 내장 메모리 체크포인터
from langgraph.checkpoint.memory import MemorySaver

# =====================================================================
# 1. 환경변수 및 기본 설정
# =====================================================================
load_dotenv()
os.environ.setdefault("USER_AGENT", "notice-rag-agent/1.0")

app = FastAPI(title="LangGraph 기반 공지사항 RAG & 관리 에이전트")

llm_logic = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
llm_stream = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True)
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

CHROMA_PERSIST_DIR = "./chroma_db"

# 1-1. 공지사항 스토어
vector_store = Chroma(
    collection_name="notice_collection",
    embedding_function=embedder,
    persist_directory=CHROMA_PERSIST_DIR
)

# 1-2. 유저 구독 정보 스토어
user_store = Chroma(
    collection_name="user_subscriptions",
    embedding_function=embedder,
    persist_directory=CHROMA_PERSIST_DIR
)

# =====================================================================
# 2. Pydantic 및 State 정의
# =====================================================================
class ChatRequest(BaseModel):
    input: str
    sessionId: str = "default_user"

class UserIntent(BaseModel):
    search_keyword: Optional[str] = Field(description="검색에 사용할 핵심 키워드 1~2개 (없으면 null)")
    date_filter: str = Field(description="'today' | '3_days' | '1_week' | '1_month' | 'all'")
    site_filter: str = Field(description="'A대학교' | 'B재단' | 'C창업포털' | 'all'")
    intent_type: str = Field(description="'summary' (오늘 요약), 'search' (특정 조건 검색), 'general' (일반 대화)")
    # [개선] 사용자의 구체적인 개수 요청을 인지하기 위한 필드 추가
    result_count: int = Field(default=3, description="유저가 구체적으로 요청한 결과의 개수 (예: '10개'면 10, '5개 찾아줘'면 5, 특별한 언급이 없으면 기본값 3)")

class FeedbackRequest(BaseModel):
    sessionId: str
    question: str
    answer: str
    is_positive: bool

class AgentState(TypedDict):
    input: str
    sessionId: str
    messages: Annotated[List[BaseMessage], add_messages]
    logged_in_user: Optional[str]
    intent: Optional[UserIntent]
    context: str
    source_documents: List[Document]
    next_node: Optional[str]
    error: Optional[str]
    response: str

# =====================================================================
# 3. 내부 헬퍼 함수 및 툴 정의
# =====================================================================
def get_user_keywords(email: Optional[str]) -> str:
    if not email:
        return ""
    user_data = user_store.get(ids=[email])
    if user_data["ids"] and user_data["metadatas"]:
        return user_data["metadatas"][0].get("keywords", "")
    return ""

def filter_recent_documents(docs: List[Document], max_days: int = 180) -> List[Document]:
    now = datetime.datetime.now()
    filtered_docs = []
    
    for doc in docs:
        metadata = doc.metadata
        
        # 1. 마감일(end_date) 기반 기간 만료 체크
        end_date_str = metadata.get("end_date")
        if end_date_str:
            try:
                clean_end = end_date_str.replace(".", "-").split()[0].strip()
                end_date = datetime.datetime.strptime(clean_end, "%Y-%m-%d")
                if now > end_date:
                    continue
            except Exception:
                pass
                
        # 2. 작성일(date) 기반 반년 이상 경과 체크
        doc_date_str = metadata.get("date")
        if doc_date_str:
            try:
                clean_date = doc_date_str.replace(".", "-").split()[0].strip()
                doc_date = datetime.datetime.strptime(clean_date, "%Y-%m-%d")
                if (now - doc_date).days > max_days:
                    continue
            except Exception:
                pass
                
        filtered_docs.append(doc)
        
    return filtered_docs

@tool
def get_current_date_and_time() -> str:
    """현재 날짜와 요일, 정확한 시각 정보를 반환합니다."""
    now = datetime.datetime.now()
    weeks = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    return f"현재 시각은 {now.strftime('%Y-%m-%d')} ({weeks[now.weekday()]}) {now.strftime('%H시 %M분')} 입니다."

@tool
def get_user_academic_context(session_id: str) -> str:
    """유저의 대학생 맞춤형 메타데이터를 조회합니다."""
    return "소속: 소프트웨어학과 / 학년: 3학년 / 관심 분야: 국가장학금 및 IT 취업 공지"

tools = [get_current_date_and_time, get_user_academic_context]
tool_node = ToolNode(tools)
llm_with_tools = llm_stream.bind_tools(tools)

# =====================================================================
# 4. LangGraph Nodes 정의
# =====================================================================

async def command_node(state: AgentState, config: RunnableConfig) -> dict:
    raw_input = state["input"].strip()
    parts = raw_input.split()
    cmd = parts[0].lower()
    
    logged_in_email = state.get("logged_in_user")
    response_text = ""
    new_logged_in_user = logged_in_email

    try:
        if cmd in ["!help", "!man"]:
            response_text = (
                "📚 **System Commands**\n"
                "`!whoami` : 현재 접속 계정 확인\n"
                "`!su [이름] [이메일]` : 계정 접속 (로그인)\n"
                "`!exit` : 로그아웃\n"
                "`!user ls` : 전체 사용자 목록 조회\n"
                "`!user add [이름] [이메일]` : 사용자 추가\n"
                "`!user rm [이름] [이메일]` : 사용자 삭제\n"
                "`!kw ls` / `!kw add [단어]` / `!kw rm [단어]` : 키워드 관리 (로그인 필요)"
            )
        elif cmd == "!whoami":
            if logged_in_email:
                user_data = user_store.get(ids=[logged_in_email])
                if user_data["ids"]:
                    name = user_data["metadatas"][0].get("name", "Unknown")
                    kws = user_data["metadatas"][0].get("keywords", "")
                    response_text = f"🐧 현재 접속자: {name} ({logged_in_email})\n🔑 구독 키워드: [{kws if kws else '없음'}]"
                else:
                    response_text = "⚠️ DB에서 유저 정보를 찾을 수 없습니다. 다시 `!su` 해주세요."
            else:
                response_text = "👤 현재 로그인된 계정이 없습니다. (`!su` 명령어를 사용하세요)"
        elif cmd == "!su":
            if len(parts) < 3:
                response_text = "❌ 사용법: `!su [이름] [이메일]`"
            else:
                name, email = parts[1], parts[2]
                user_data = user_store.get(ids=[email])
                if user_data["ids"] and user_data["metadatas"][0].get("name") == name:
                    new_logged_in_user = email
                    response_text = f"✅ `{name}` 계정으로 전환되었습니다."
                else:
                    response_text = "❌ 이름 또는 이메일이 일치하는 계정이 없습니다."
        elif cmd == "!exit":
            new_logged_in_user = None
            response_text = "👋 로그아웃 되었습니다."
        elif cmd == "!user":
            subcmd = parts[1] if len(parts) > 1 else ""
            if subcmd == "ls":
                results = user_store.get()
                if not results["ids"]:
                    response_text = "텅~ (등록된 유저가 없습니다.)"
                else:
                    user_list = [f"- {m.get('name')} ({m.get('email')})" for m in results["metadatas"]]
                    response_text = "👥 **전체 유저 목록**\n" + "\n".join(user_list)
            elif subcmd == "add":
                if len(parts) < 4:
                    response_text = "❌ 사용법: `!user add [이름] [이메일]`"
                else:
                    name, email = parts[2], parts[3]
                    if user_store.get(ids=[email])["ids"]:
                        response_text = "❌ 이미 존재하는 이메일입니다."
                    else:
                        doc = Document(page_content="", metadata={"name": name, "email": email, "keywords": ""})
                        user_store.add_documents(documents=[doc], ids=[email])
                        response_text = f"✅ 사용자 `{name}` 등록 완료."
            elif subcmd == "rm":
                if len(parts) < 4:
                    response_text = "❌ 사용법: `!user rm [이름] [이메일]`"
                else:
                    name, email = parts[2], parts[3]
                    user_data = user_store.get(ids=[email])
                    if user_data["ids"] and user_data["metadatas"][0].get("name") == name:
                        user_store.delete(ids=[email])
                        response_text = f"🗑️ 사용자 `{name}` 삭제 완료."
                        if logged_in_email == email:
                            new_logged_in_user = None
                    else:
                        response_text = "❌ 정보가 일치하는 유저가 없습니다."
            else:
                response_text = "❌ 잘못된 명령어입니다. (`ls`, `add`, `rm` 사용 가능)"
        elif cmd == "!kw":
            if not logged_in_email:
                response_text = "❌ 접근 권한 거부: 먼저 `!su` 명령어로 로그인하세요."
            else:
                subcmd = parts[1] if len(parts) > 1 else ""
                user_data = user_store.get(ids=[logged_in_email])
                name = user_data["metadatas"][0].get("name", "Unknown")
                current_kws = [k.strip() for k in user_data["metadatas"][0].get("keywords", "").split(",") if k.strip()]
                
                if subcmd == "ls":
                    response_text = f"🏷️ 내 키워드: {', '.join(current_kws) if current_kws else '없음'}"
                elif subcmd == "add":
                    if len(parts) < 3:
                        response_text = "❌ 사용법: `!kw add [단어1] [단어2]...`"
                    else:
                        new_kws = parts[2:]
                        merged_kws = list(set(current_kws + new_kws))
                        kw_str = ", ".join(merged_kws)
                        
                        user_store.delete(ids=[logged_in_email])
                        doc = Document(page_content=kw_str, metadata={"name": name, "email": logged_in_email, "keywords": kw_str})
                        user_store.add_documents(documents=[doc], ids=[logged_in_email])
                        response_text = f"✅ 키워드 추가됨. (현재: {kw_str})"
                elif subcmd == "rm":
                    if len(parts) < 3:
                        response_text = "❌ 사용법: `!kw rm [단어1] [단어2]...`"
                    else:
                        remove_kws = parts[2:]
                        merged_kws = [kw for kw in current_kws if kw not in remove_kws]
                        kw_str = ", ".join(merged_kws)
                        
                        user_store.delete(ids=[logged_in_email])
                        doc = Document(page_content=kw_str, metadata={"name": name, "email": logged_in_email, "keywords": kw_str})
                        user_store.add_documents(documents=[doc], ids=[logged_in_email])
                        response_text = f"🗑️ 키워드 삭제됨. (현재: {kw_str if kw_str else '없음'})"
                else:
                    response_text = "❌ 잘못된 명령어입니다. (`ls`, `add`, `rm` 사용 가능)"
        else:
            response_text = f"❌ 알 수 없는 명령어입니다: `{cmd}`. `!help`를 입력하세요."

    except Exception as e:
        response_text = f"⚠️ 명령어 실행 중 오류 발생: {e}"

    return {
        "response": response_text, 
        "messages": [("ai", response_text)],
        "logged_in_user": new_logged_in_user
    }

# ── [4-2] 의도 분석 노드 ──
intent_extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 사용자의 질문 및 대화 맥락을 파악하여 최적의 검색 조건과 요청 사항을 분석하는 라우팅 전문가야.
# Context
- 현재 날짜 및 요일: {current_date_info}
- 참조 가능한 사이트 이름 리스트: ["A대학교", "B재단", "C창업포털"]
* 중요: 사용자가 몇 개의 소식/공지를 보여달라고 구체적인 숫자(예: 10개, 5개 등)를 요구했다면 이를 정확히 추출하여 result_count에 반영해야 해."""),
    MessagesPlaceholder("messages")
])

async def extract_intent_node(state: AgentState, config: RunnableConfig) -> dict:
    now = datetime.datetime.now()
    weeks = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    current_date_info = f"{now.strftime('%Y-%m-%d')} ({weeks[now.weekday()]})"
    
    chain = intent_extractor_prompt | llm_logic.with_structured_output(UserIntent)
    intent = await chain.ainvoke({
        "current_date_info": current_date_info,
        "messages": state["messages"]
    }, config=config)
    return {"intent": intent}

# ── [4-3] 유저 맞춤형 RAG 검색 노드 ──
async def rag_node(state: AgentState, config: RunnableConfig) -> dict:
    intent = state["intent"]
    base_query = intent.search_keyword if intent.search_keyword else state["input"]
    
    # 유저가 구체적으로 요청한 개수 파악 (기본값 3)
    limit = intent.result_count if intent.result_count else 3
    
    user_kws = get_user_keywords(state.get("logged_in_user"))
    if user_kws:
        search_query = f"{base_query} {user_kws.replace(',', ' ')}"
    else:
        search_query = base_query
        
    # [수정] 유저가 요청한 limit 수치에 맞춰 동적으로 오버샘플링 풀을 확장
    fetch_k = max(25, limit * 3)
    raw_docs = await vector_store.asimilarity_search(search_query, k=fetch_k)
    
    # [수정] 유저가 요청한 limit 개수만큼 슬라이싱
    docs = filter_recent_documents(raw_docs, max_days=180)[:limit]
    
    context_list = []
    for d in docs:
        site = d.metadata.get('site_name', intent.site_filter if intent.site_filter != 'all' else '알수없음')
        title = d.metadata.get('title', '공지사항 공고')
        context_list.append(f"[공지사항 정보]\n사이트명: {site}\n제목: {title}\n본문: {d.page_content}")
    context = "\n\n".join(context_list)
    
    # [수정] 프롬프트 내부의 개수 제한 지침을 유저 요청치({limit})로 동적 변경
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""너는 유저가 신뢰할 수 있는 공지사항 안내원이야.
데이터베이스에서 검색된 [참조 데이터]만을 기반으로 유저의 질문과 가장 관련성이 높은 최상위 {{limit}}개 항목을 찾아 정확하게 답변해야 해.
아주 중요한 규칙: 마크다운 절대 금지 (ERROR 방지)
오직 줄바꿈(엔터)과 숫자(1., 2.), 그리고 일반 텍스트만 사용해서 답변을 작성해야 합니다.
[참조 데이터]
{{context}}"""),
        MessagesPlaceholder("messages")
    ])
    
    chain = rag_prompt | llm_stream
    res = await chain.ainvoke({
        "context": context, 
        "limit": limit,
        "messages": state["messages"]
    }, config=config)
    
    return {"context": context, "response": res.content, "messages": [res], "source_documents": docs}

# ── [4-4] 유저 맞춤형 최신 공지사항 요약 노드 ──
async def summary_node(state: AgentState, config: RunnableConfig) -> dict:
    intent = state["intent"]
    base_query = "최신 공지사항 학사 학정 고시"
    
    # 유저가 구체적으로 요청한 개수 파악 (기본값 3)
    limit = intent.result_count if intent.result_count else 3
    
    user_kws = get_user_keywords(state.get("logged_in_user"))
    if user_kws:
        search_query = f"{base_query} {user_kws.replace(',', ' ')}"
    else:
        search_query = base_query

    # [수정] 유저가 요청한 limit 수치에 맞춰 동적으로 오버샘플링 풀을 확장
    fetch_k = max(35, limit * 3)
    raw_docs = await vector_store.asimilarity_search(search_query, k=fetch_k)
    
    # [수정] 유저가 요청한 limit 개수만큼 슬라이싱
    docs = filter_recent_documents(raw_docs, max_days=180)[:limit]
    
    context_list = []
    for i, d in enumerate(docs, 1):
        site = d.metadata.get('site_name', '알수없음')
        title = d.metadata.get('title', f'최신 공지사항 {i}')
        context_list.append(f"[공지사항 {i}]\n사이트명: {site}\n제목: {title}\n본문: {d.page_content}")
    context = "\n\n".join(context_list)
    
    # [수정] 프롬프트 내부의 요약 선별 개수 제약을 유저 요청치({limit})로 동적 매핑
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""너는 대학생에게 가장 중요하고 시급한 학내 소식을 선별해 전달하는 족집게 학사 비서야.
제공된 본문 리스트를 분석해 유저의 관심 분야와 밀접하며 가장 중요한 핵심 공지사항 {{limit}}가지를 선별해줘.
# Constraint (중요 제약 조건)
1. 마크다운 기호(#, *, -, _, --- 등)를 절대 사용하지 마. 
2. 오직 순수한 텍스트(Plain Text)로만 출력해.
# Input Data
{{context}}"""),
        MessagesPlaceholder("messages")
    ])
    
    chain = summary_prompt | llm_stream
    res = await chain.ainvoke({
        "context": context, 
        "limit": limit,
        "messages": state["messages"]
    }, config=config)
    
    return {"context": context, "response": res.content, "messages": [res], "source_documents": docs}

# ── [4-5] 일반 대화 및 툴 처리 노드 ──
async def general_node(state: AgentState, config: RunnableConfig) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 친절하고 도움을 주는 학교 생활 어시스턴트다. 공지사항 외에 날짜나 시간, 일상 대화, 유저 정보 관련 요청이 오면 적절한 도구(Tools)를 활용해 답변해줘."),
        MessagesPlaceholder("messages")
    ])
    
    chain = prompt | llm_with_tools
    res = await chain.ainvoke({
        "messages": state["messages"]
    }, config=config)
    
    return {"messages": [res], "response": res.content}

# =====================================================================
# 5. 그래프 라우팅 및 구조 조립
# =====================================================================
def route_start(state: AgentState) -> str:
    if state["input"].strip().startswith("!"):
        return "command_node"
    return "extract_intent"

def route_intent_edge(state: AgentState) -> str:
    if not state.get("intent") or not hasattr(state["intent"], "intent_type"):
        return "general"
    return state["intent"].intent_type
    
def route_after_general(state: AgentState) -> str:
    if "messages" in state and state["messages"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("command_node", command_node)
workflow.add_node("extract_intent", extract_intent_node)
workflow.add_node("rag_generation", rag_node)
workflow.add_node("summary_generation", summary_node)
workflow.add_node("general_generation", general_node)
workflow.add_node("tools", tool_node)

workflow.add_conditional_edges(START, route_start, {
    "command_node": "command_node",
    "extract_intent": "extract_intent"
})

workflow.add_conditional_edges("extract_intent", route_intent_edge, {
    "summary": "summary_generation",
    "search": "rag_generation",
    "general": "general_generation"
})

workflow.add_conditional_edges("general_generation", route_after_general, {
    "tools": "tools",
    END: END
})

workflow.add_edge("tools", "general_generation")

for node in ["command_node", "summary_generation", "rag_generation"]:
    workflow.add_edge(node, END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# =====================================================================
# 6. API 엔드포인트 및 미들웨어
# =====================================================================
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(GZipMiddleware, minimum_size=500)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.post("/api/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, body: ChatRequest):
    async def generate() -> AsyncGenerator[str, None]:
        try:
            config = {"configurable": {"thread_id": body.sessionId}}
            graph_input = {
                "input": body.input, 
                "sessionId": body.sessionId,
                "messages": [("user", body.input)]
            }
            
            async for event in graph.astream_events(graph_input, config=config, version="v2"):
                kind = event["event"]
                name = event.get("name", "")

                if kind == "on_chain_start" and name == "command_node":
                    yield f"data: {json.dumps({'status': 'step', 'message': '시스템 명령어를 처리하고 있습니다...'}, ensure_ascii=False)}\n\n"
                    
                elif kind == "on_chain_end" and name == "command_node":
                    output = event.get("data", {}).get("output", {})
                    if "response" in output:
                        yield f"data: {json.dumps({'text': output['response']}, ensure_ascii=False)}\n\n"

                elif kind == "on_chain_start":
                    if name == "extract_intent":
                        yield f"data: {json.dumps({'status': 'step', 'message': '질문 의도를 분석하고 있습니다...'}, ensure_ascii=False)}\n\n"
                    elif name == "rag_generation":
                        yield f"data: {json.dumps({'status': 'step', 'message': '관련 공지사항을 DB에서 검색 중입니다...'}, ensure_ascii=False)}\n\n"
                    elif name == "summary_generation":
                        yield f"data: {json.dumps({'status': 'step', 'message': '최신 공지사항을 요약하고 있습니다...'}, ensure_ascii=False)}\n\n"
                    elif name == "general_generation":
                        yield f"data: {json.dumps({'status': 'step', 'message': '답변을 생성하고 있습니다...'}, ensure_ascii=False)}\n\n"

                elif kind == "on_chat_model_stream":
                    if event.get("metadata", {}).get("langgraph_node") == "extract_intent":
                        continue
                        
                    chunk_content = event["data"]["chunk"].content
                    if chunk_content:
                        yield f"data: {json.dumps({'text': chunk_content}, ensure_ascii=False)}\n\n"

                elif kind == "on_chain_end" and name in ["rag_generation", "summary_generation"]:
                    output = event.get("data", {}).get("output", {})
                    if "source_documents" in output:
                        docs = output["source_documents"]
                        sources = []
                        seen_urls = set()
                        for d in docs:
                            url = d.metadata.get("url")
                            title = d.metadata.get("title", "공지사항 링크")
                            if url and url not in seen_urls:
                                sources.append({"title": title, "url": url})
                                seen_urls.add(url)
                                
                        if sources:
                            yield f"data: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(), 
        media_type="text/event-stream", 
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    state = graph.get_state(config)
    if state and hasattr(state, "values") and "messages" in state.values:
        history = []
        for msg in state.values["messages"]:
            if msg.type in ["human", "ai"]:
                role = "user" if msg.type == "human" else "ai"
                history.append({"role": role, "content": msg.content})
        return history
    return []

@app.delete("/api/chat/history/{session_id}")
async def delete_chat_history(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    state = graph.get_state(config)
    if state and hasattr(state, "values") and "messages" in state.values:
        delete_messages = [RemoveMessage(id=m.id) for m in state.values["messages"]]
        graph.update_state(config, {"messages": delete_messages})
        return {"status": "success", "message": "대화 기록이 성공적으로 삭제되었습니다."}
    return {"status": "not_found", "message": "삭제할 대화 기록이 존재하지 않습니다."}

@app.post("/api/chat/feedback")
async def submit_feedback(req: FeedbackRequest):
    feedback_entry = req.model_dump()
    feedback_entry["timestamp"] = datetime.datetime.now().isoformat()
    
    feedback_file = Path("feedbacks.jsonl")
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")
    return {"status": "success", "message": "소중한 피드백이 저장되었습니다."}

# =====================================================================
# 7. 정적 파일 서빙 및 앱 실행 구문
# =====================================================================
app.mount("/", StaticFiles(directory="public", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)