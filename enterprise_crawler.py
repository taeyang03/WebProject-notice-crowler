from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

import asyncio
import hashlib
import logging
import re
from datetime import datetime
from urllib.parse import urljoin
from typing import List, Set

from dotenv import load_dotenv
from bs4 import BeautifulSoup

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

import os
from email.message import EmailMessage
import aiosmtplib
import logging

load_dotenv()

# ── 로깅 설정 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.FileHandler("crawler.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger("NoticeCrawler")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
CHROMA_PERSIST_DIR = "./chroma_db"

# ── 타겟 사이트 설정 ──
SITES_CONFIG = {
    "충북대학교_소프트웨어학과": {
        "base_url": "https://software.cbnu.ac.kr/",
        "list_url": "https://software.cbnu.ac.kr/sub0401",
        "list_selector": "table.bd_lst tbody tr, table tbody tr",
        "title_selector": "td.title a, a.hx",
        "date_selector": "td.time, td.date, td:nth-child(4)",
        "content_selector": "div.xe_content, div.rd_body",
        "is_dynamic": False
    },
    "충북대학교_홈페이지": {
        "base_url": "https://www.cbnu.ac.kr/www/selectBbsNttView.do", 
        "list_url": "https://www.cbnu.ac.kr/www/selectBbsNttList.do?bbsNo=8&key=813",
        "list_selector": "table tbody tr", 
        "title_selector": "td.p-subject a, td.subject a", 
        "date_selector": "td:nth-child(6), td.reg_date",   
        "content_selector": "div.viewcontentbox", 
        "is_dynamic": True 
    }
}

# ── Vector DB 설정 ──
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name="notice_collection",
    embedding_function=embedder,
    persist_directory=CHROMA_PERSIST_DIR
)

user_store = Chroma(
    collection_name="user_subscriptions",
    embedding_function=embedder,
    persist_directory=CHROMA_PERSIST_DIR
)


def add_user_subscription(email: str, keywords: List[str]):
    """사용자의 구독 키워드를 ChromaDB에 추가하거나 업데이트합니다."""
    keyword_str = ", ".join(keywords)
    
    # ID를 이메일로 지정하여 중복을 방지하고 데이터 갱신을 용이하게 함
    doc = Document(
        page_content=keyword_str, 
        metadata={"email": email, "keywords": keyword_str}
    )
    user_store.add_documents(documents=[doc], ids=[email])
    logger.info(f"👤 사용자 구독 정보 업데이트 완료: {email} -> [{keyword_str}]")

def delete_user_subscription(email: str):
    """특정 사용자의 구독 정보를 DB에서 삭제합니다."""
    try:
        user_store.delete(ids=[email])
        logger.info(f"👤 사용자 구독 정보 삭제 완료: {email}")
    except Exception as e:
        logger.error(f"구독 정보 삭제 실패 (존재하지 않는 이메일): {email}")

def get_all_user_subscriptions() -> List[dict]:
    """ChromaDB에 저장된 모든 사용자의 구독 정보를 불러옵니다."""
    results = user_store.get()
    user_list = []
    
    if results and "metadatas" in results and results["metadatas"]:
        for meta in results["metadatas"]:
            if meta and "email" in meta and "keywords" in meta:
                # DB에 문자열로 저장된 키워드를 다시 리스트 형태로 복원
                kw_list = [k.strip() for k in meta["keywords"].split(",") if k.strip()]
                user_list.append({
                    "email": meta["email"], 
                    "keywords": kw_list
                })
    return user_list
    
def get_existing_ids() -> Set[str]:
    result = vector_store.get()
    return set(result["ids"]) if result and "ids" in result else set()

def generate_doc_id(site_name: str, title: str, date_str: str) -> str:
    return hashlib.md5(f"{site_name}_{title}_{date_str}".encode("utf-8")).hexdigest()

# ── Pydantic 모델 (LLM 구조화된 출력용) ──
class KeywordExtraction(BaseModel):
    keywords: List[str] = Field(description="공지사항 본문에서 추출한 핵심 키워드 3~5개")

# ── 비동기 네트워크 & LLM 함수 ──
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)))
async def fetch_static_html(url: str) -> str:
    async with httpx.AsyncClient(timeout=15.0, headers=HEADERS, verify=False) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text

async def extract_keywords_with_llm(title: str, content: str) -> str:
    """LLM을 사용해 본문에서 핵심 태그 3~5개를 추출하고 쉼표로 연결하여 반환합니다."""
    try:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        structured_model = model.with_structured_output(KeywordExtraction)
        
        prompt = f"다음 공지사항의 제목과 본문을 분석하여 대학생에게 유용한 핵심 키워드(태그)를 3~5개 추출해줘.\n제목: {title}\n본문: {content[:1000]}"
        res = await structured_model.ainvoke(prompt)
        
        # ChromaDB 메타데이터 저장을 위해 리스트를 문자열로 변환
        return ", ".join(res.keywords)
    except Exception as e:
        logger.error(f"키워드 추출 실패: {e}")
        return ""

async def extract_text_from_image(image_url: str) -> str:
    try:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        msg = HumanMessage(content=[
            {"type": "text", "text": "공지사항 첨부파일 또는 포스터 이미지의 텍스트를 추출하고 주요 일정을 요약해줘."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ])
        res = await model.ainvoke([msg])
        return res.content
    except Exception as e:
        return f"[이미지 분석 실패: {e}]"

# ── 푸시 알림 트리거 함수 ──
async def trigger_push_notification(email: str, title: str, url: str):
    """Gmail SMTP를 활용하여 비동기로 이메일 푸시 알림을 발송합니다."""
    
    sender_email = os.getenv("GMAIL_SENDER")
    app_password = os.getenv("GMAIL_APP_PASSWORD")

    if not sender_email or not app_password:
        logger.error(" Gmail 환경변수(GMAIL_SENDER, GMAIL_APP_PASSWORD)가 설정되지 않았습니다.")
        return

    # 이메일 객체 생성 및 메타데이터 설정
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = email
    msg["Subject"] = f"[공지 알림] 관심 키워드 새 글: {title}"

    # 본문 내용 작성 (HTML 형식도 지원합니다)
    content = f"""
    안녕하세요! 구독하신 키워드가 포함된 새로운 공지사항이 등록되었습니다.

         제목: {title}
         확인하기: {url}

    본 메일은 대학생 공지사항 알림 시스템을 통해 자동 발송되었습니다.
    """
    msg.set_content(content)

    try:
        # 비동기로 이메일 발송
        await aiosmtplib.send(
            msg,
            hostname="smtp.gmail.com",
            port=465,
            use_tls=True,
            username=sender_email,
            password=app_password,
        )
        logger.info(f"[푸시 발송 성공] {email} 님께 알림 전송: {title}")
        
    except Exception as e:
        logger.error(f"[푸시 발송 실패] {email} 님께 알림 전송 중 오류 발생: {e}")
# ── 개별 사이트 크롤링 로직 ──
async def scrape_site(site_name: str, config: dict, existing_ids: Set[str], browser=None) -> List[Document]:
    logger.info(f"[{site_name}] 크롤링 동작 중...")
    new_docs = []
    
    try:
        html = await fetch_static_html(config["list_url"])
        soup = BeautifulSoup(html, "html.parser")
        articles = soup.select(config["list_selector"])
        
        if not articles:
            logger.warning(f"[{site_name}] 게시글 목록을 찾을 수 없습니다.")
            return []
            
        logger.info(f"[{site_name}] {len(articles)}개의 게시글을 발견했습니다.")

        for art in articles:
            title_elem = art.select_one(config["title_selector"])
            date_elem = art.select_one(config["date_selector"])
            
            if not title_elem or not date_elem: 
                continue
                
            title = title_elem.text.strip()
            date_str = date_elem.text.strip()
            href = title_elem.get("href", "")
            detail_url = urljoin(config["base_url"], href)
            doc_id = generate_doc_id(site_name, title, date_str)
            
            if doc_id in existing_ids: 
                continue
                
            logger.info(f"   ↳ 새 글 발견: {title} ({detail_url})")
            
            detail_html = await fetch_static_html(detail_url)
            detail_soup = BeautifulSoup(detail_html, "html.parser")
            content_elem = detail_soup.select_one(config["content_selector"])
            
            # 이미지 및 첨부파일 원본 URL 추출
            images = detail_soup.find_all("img")
            image_urls = [urljoin(detail_url, img.get("src", "")) for img in images if img.get("src")]
            
            attachments = detail_soup.find_all("a", href=re.compile(r"(download|file|attach|down)", re.I))
            attachment_urls = [urljoin(detail_url, a.get("href", "")) for a in attachments if a.get("href")]

            if content_elem:
                raw_text = content_elem.get_text(separator='\n', strip=True)
                content_text = re.sub(r'\n{2,}', '\n', raw_text)
            else:
                logger.warning(f"   ↳ [{title}] 본문 내용을 찾지 못했습니다.")
                content_text = detail_soup.body.text.strip()[:1000]

            # 본문이 부실하고 이미지가 있는 포스터 공지사항인 경우 Vision API로 텍스트 추출 보강
            if len(content_text) < 100 and image_urls:
                logger.info(f"   ↳ 텍스트 보강을 위해 이미지를 분석합니다.")
                image_text = await extract_text_from_image(image_urls[0])
                content_text += f"\n\n[이미지 텍스트 분석 결과]\n{image_text}"

            # LLM을 이용해 핵심 키워드(태그) 추출
            tags = await extract_keywords_with_llm(title, content_text)

            full_content = f"제목: {title}\n작성일: {date_str}\n키워드: {tags}\n본문: {content_text}"
            
            # 메타데이터 구성 (ChromaDB 제약으로 리스트형은 콤마 연결 문자열로 저장)
            metadata = {
                "id": doc_id, 
                "site_name": site_name, 
                "title": title, 
                "url": detail_url, 
                "created_at": date_str,
                "tags": tags,
                "image_urls": ", ".join(image_urls[:3]),       # 최대 3개까지만 저장
                "attachment_urls": ", ".join(attachment_urls[:3])
            }
            
            new_docs.append(Document(page_content=full_content, metadata=metadata))
            
    except Exception as e:
        logger.error(f"[{site_name}] 크롤링 중 에러 발생: {e}")
        
    return new_docs

# ── 병렬 스케줄링 및 DB 저장, 알림 처리 ──
async def crawl_and_update_db():
    existing_ids = get_existing_ids()
    all_new_docs: List[Document] = []
    
    tasks = [scrape_site(name, cfg, existing_ids, None) for name, cfg in SITES_CONFIG.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
        
    for res in results:
        if isinstance(res, list): all_new_docs.extend(res)
        
    if not all_new_docs:
        logger.info("업데이트된 새 공지가 없습니다.")
        return

    logger.info(f"총 {len(all_new_docs)}개의 새 공지를 Chroma Vector DB에 추가합니다...")
    doc_ids = [doc.metadata["id"] for doc in all_new_docs]
    vector_store.add_documents(documents=all_new_docs, ids=doc_ids)
    logger.info(f"{len(all_new_docs)}건 DB 추가 완료.")

    user_subscriptions = get_all_user_subscriptions()
    
    if not user_subscriptions:
        logger.info("등록된 구독자가 없어 알림을 발송하지 않습니다.")
        return

    # 발송 태스크를 모아둘 리스트 생성 및 검색 조건 최적화
    notification_tasks = []
    
    for doc in all_new_docs:
        # LLM이 추출한 핵심 키워드 문자열만 가져옴
        doc_tags = doc.metadata.get("tags", "")
        
        for user in user_subscriptions:
            for kw in user["keywords"]:
                # doc.page_content를 검색 대상에서 제외하고 태그에서만 검색
                if kw in doc_tags:
                    # 메일을 즉시 보내지 않고 비동기 태스크 리스트에 추가
                    notification_tasks.append(
                        trigger_push_notification(
                            email=user["email"], 
                            title=doc.metadata["title"], 
                            url=doc.metadata["url"]
                        )
                    )
                    break  # 한 공지에 대해 중복 알림 방지

    # 모아둔 이메일 발송 작업을 한 번에 병렬 실행
    if notification_tasks:
        logger.info(f"총 {len(notification_tasks)}건의 푸시 알림 발송을 시작합니다.")
        await asyncio.gather(*notification_tasks, return_exceptions=True)
        logger.info("모든 푸시 알림 발송이 완료되었습니다.")

# ── 스케줄러 ──
async def run_scheduler():
    logger.info("크롤링 및 알림 시스템 초기화를 시작합니다.")
    
    # [개선 2] 테스트용 하드코딩 구독 데이터 제거 (실운영 시 주석 처리 또는 삭제)
    # add_user_subscription("student1@example.com", ["장학", "근로", "등록금"])
    # add_user_subscription("student2@example.com", ["인턴", "소프트웨어", "채용", "대회"])
    
    await crawl_and_update_db()
    logger.info(f"1회차 크롤링이 완료되었습니다.")
    logger.info(f"다음 크롤링 대기중...")
    
    # 6시간마다 반복
    while True:
        await asyncio.sleep(6 * 3600)
        await crawl_and_update_db()

if __name__ == "__main__":
    try:
        logger.info("크롤러 스크립트를 시작합니다...")
        asyncio.run(run_scheduler())
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 크롤러가 종료되었습니다.")
    except Exception as e:
        logger.error(f"크롤러 실행 중 치명적인 오류 발생: {e}")