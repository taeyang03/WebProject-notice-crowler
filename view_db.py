import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# .env 파일에 있는 OPENAI_API_KEY 로드
load_dotenv()

CHROMA_PERSIST_DIR = "./chroma_db"
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# DB 연결
vector_store = Chroma(
    collection_name="notice_collection",
    embedding_function=embedder,
    persist_directory=CHROMA_PERSIST_DIR
)

# 💡 DB의 모든 데이터 가져오기 (limit 옵션 제거)
all_data = vector_store.get()

# 총 데이터 개수 출력
total_count = len(all_data['ids'])
print(f"✅ 총 저장된 데이터 개수: {total_count}개")
print("=" * 60)

# 데이터가 없을 경우 처리
if total_count == 0:
    print("DB에 저장된 데이터가 없습니다.")
else:
    # 모든 데이터 반복 출력
    for i in range(total_count):
        print(f"[{i+1}] 문서 ID: {all_data['ids'][i]}")
        
        # 💡 메타데이터 내부 컬럼을 하나씩 풀어서 모두 출력
        print("📌 메타데이터:")
        for key, value in all_data['metadatas'][i].items():
            print(f"   - {key}: {value}")
        
        # 본문 텍스트 전체 출력 (말줄임표 없음!)
        content = all_data['documents'][i]
        print(f"📝 본문 내용:\n{content}")
        print("-" * 60)