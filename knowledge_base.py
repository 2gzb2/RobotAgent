import os

# 仅在显式要求时强制离线，默认允许“优先本地缓存，缺失时再走镜像下载”。
force_offline = os.getenv("EMBEDDING_FORCE_OFFLINE", "0").lower()
if force_offline == "1" or force_offline == "true" or force_offline == "yes":
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import RAG_PERSIST_DIR, EMBEDDING_MODEL_NAME, FORCE_REBUILD_KNOWLEDGE


class KnowledgeBase:
    @staticmethod
    def _resolve_local_embedding_model(model_name: str) -> str:
        """优先使用本地缓存，避免代理波动导致 HuggingFace 元数据请求失败。"""
        if os.path.exists(model_name):
            return model_name

        if "/" not in model_name:
            return model_name

        repo_cache_dir = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "huggingface",
            "hub",
            f"models--{model_name.replace('/', '--')}",
        )
        snapshots_dir = os.path.join(repo_cache_dir, "snapshots")
        refs_main = os.path.join(repo_cache_dir, "refs", "main")

        if os.path.exists(refs_main):
            with open(refs_main, "r", encoding="utf-8") as f:
                revision = f.read().strip()
            snapshot_path = os.path.join(snapshots_dir, revision)
            if os.path.exists(snapshot_path):
                return snapshot_path

        if os.path.isdir(snapshots_dir):
            snapshot_dirs = [
                os.path.join(snapshots_dir, name)
                for name in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, name))
            ]
            if snapshot_dirs:
                return sorted(snapshot_dirs)[-1]

        return model_name

    def __init__(self):
        self.persist_dir = RAG_PERSIST_DIR
        local_only_env = os.getenv("EMBEDDING_LOCAL_ONLY", "1").lower()
        local_only = True
        if local_only_env == "0" or local_only_env == "false" or local_only_env == "no":
            local_only = False

        # 默认优先命中本地缓存，只有显式关闭时才回退到仓库名下载。
        if local_only:
            model_name = self._resolve_local_embedding_model(EMBEDDING_MODEL_NAME)
        else:
            model_name = EMBEDDING_MODEL_NAME
        print(f" [RAG] 正在加载 Embedding 模型...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # 有显卡改 cuda，没有显卡改 cpu
                encode_kwargs={'normalize_embeddings': True}  # 归一化向量，确保 L2 距离在 0~2 范围内
            )
        except Exception as e:
            print(f" Embedding 加载失败: {e}")
            raise

        # 初始化向量数据库连接
        self.vector_db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="robot_memory"
        )

    def init_data(self, file_path: str):
        """智能初始化：根据配置决定是否重写知识库"""

        # 1. 检查是否需要跳过
        # 注意：这里访问 _collection 会有黄色警告，请忽略它，这是获取条数最快的方法
        try:
            doc_count = self.vector_db._collection.count()
        except Exception:
            doc_count = 0

        if not FORCE_REBUILD_KNOWLEDGE and doc_count > 0:
            print(f" [RAG] 检测到已有知识库 ({doc_count} 条)，跳过重建。")
            return

        # 2. 如果强制重写，先清空
        if FORCE_REBUILD_KNOWLEDGE and os.path.exists(self.persist_dir):
            print(" [RAG] 正在清空旧知识库...")
            self.vector_db.delete_collection()  # 清空集合

            # 重新初始化 DB 对象以确保干净
            self.vector_db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name="robot_memory"
            )

        # 3. 开始注入
        self._rebuild_from_file(file_path)

    def _rebuild_from_file(self, file_path: str):
        """读取文件 -> 智能切分 -> 存入数据库 (内部方法)"""
        if not os.path.exists(file_path):
            print(f"[!] 找不到知识库文件: {file_path}")
            return

        print(f" [RAG] 正在读取并处理: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        if not full_text.strip():
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        docs = text_splitter.create_documents([full_text])

        print(f" [RAG] 正在注入 {len(docs)} 个知识片段...")
        self.vector_db.add_documents(docs)
        print(f" [RAG] 知识注入完成！")

    def search(self, query: str, top_k: int = 3):
        try:
            results = self.vector_db.similarity_search_with_score(query, k=top_k)
            valid_docs = []

            for doc, score in results:
                # 分数越低越相似，1.2 是经验阈值（需配合 normalize_embeddings=True 使用）
                if score < 1.2:
                    valid_docs.append(f"资料{len(valid_docs) + 1}: {doc.page_content}")

            return "\n".join(valid_docs) if valid_docs else ""
        except Exception as e:
            print(f" 检索出错: {e}")
            return ""


