import dashscope
import chromadb
from config import API_KEY, CHROMA_PATH, COLLECTION_NAME  # 👈 从 config 导入配置

dashscope.api_key = API_KEY

# 1. 把文字变成向量
def get_embedding(text):
    try:
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v2",
            input=text
        )
        return resp.output['embeddings'][0]['embedding']
    except Exception as e:
        print(f"❌ 向量化失败: {e}")
        return None

# 2. 👇 新增：专门用来获取数据库连接的函数
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)

# 3. 检索逻辑
def search(query, chunks):
    # 直接调用上面写好的 get_collection 拿抽屉
    collection = get_collection()

    # 核心判断：如果库里没有数据，才进行向量化！
    if collection.count() == 0:
        print("⏳ 首次运行，正在将文本块转换为向量并写入硬盘...")
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if embedding:
                collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    ids=[str(i)]
                )
        print("✅ 向量化入库完成！")
    else:
        print(f"✅ 已从硬盘读取本地知识库，共 {collection.count()} 条数据，无需重新向量化！")

    # 把用户的问题变成向量
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []

    # 去库里查
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    return results['documents'][0]