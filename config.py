import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL = "qwen-plus"

# 文本切块配置
CHUNK_SIZE = 300
CHUNK_OVERLAP = 10

# 👇👇👇 加上这两行数据库配置 👇👇👇
CHROMA_PATH = "/tmb/chroma_db"        # 向量库存哪个文件夹
COLLECTION_NAME = "my_kb"          # 数据库里的集合名字