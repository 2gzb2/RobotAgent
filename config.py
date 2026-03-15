import os
import datetime
from dotenv import load_dotenv

load_dotenv()


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# 使用国内镜像站下载模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ================= 1. 听 (Ear) =================
ASR_MODEL_PATH = os.getenv("ASR_MODEL_PATH", "model")
VAD_SILENCE_DURATION = 1.5
MAX_RECORD_TIME = 20
POST_TTS_LISTEN_DELAY = 0.6

# ================= 2. 说 (Mouth) =================
TTS_VOICE = "zh-CN-XiaoxiaoNeural"
TTS_OUTPUT_FILE = os.getenv("TTS_OUTPUT_FILE", "temp_audio.mp3")

# ================= 3. 脑 (Brain / RAG) =================
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_MODEL = os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3")
RAG_PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "./chroma_db")
DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
KNOWLEDGE_FILE_PATH = os.getenv("KNOWLEDGE_FILE_PATH", os.path.join(DATA_FOLDER, "knowledge.txt"))
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
MEMORY_DB_URL = os.getenv("MEMORY_DB_URL", "sqlite:///memory.db")
MEMORY_SESSION_ID = datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S")

# 是否强制重建知识库？
# True: 每次启动都清空并重新读取 txt (适合演示前/更新文档后)
# False: 如果数据库存在，直接加载旧的 (启动速度极快)
FORCE_REBUILD_KNOWLEDGE = _get_bool_env("FORCE_REBUILD_KNOWLEDGE", False)
