import re
import time
import datetime
from typing import Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from knowledge_base import KnowledgeBase
from config import (
    MEMORY_DB_URL,
    KNOWLEDGE_FILE_PATH,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    SILICONFLOW_MODEL,
    MEMORY_SESSION_ID,
)
# 导入工具
from robot_tools import robot_tools


class RobotBrain:
    DIRECT_EXIT_COMMANDS = {
        "退出",
        "退下",
        "关机",
        "再见",
        "拜拜",
        "不聊了",
        "结束对话",
        "别聊了",
        "停下",
        "我不聊了",
        "先退出",
        "先退下",
        "那我先退下",
        "我不聊了退下",
    }

    def __init__(self):
        self.api_key = SILICONFLOW_API_KEY
        if not self.api_key:
            raise RuntimeError("未找到 SILICONFLOW_API_KEY，请先在 .env 或系统环境变量中配置。")

        # 1. 初始化并加载知识库
        self.kb = KnowledgeBase()
        self.kb.init_data(KNOWLEDGE_FILE_PATH)

        # 2. 记忆库
        self.chat_history_db = SQLChatMessageHistory(
            session_id=MEMORY_SESSION_ID,
            connection=MEMORY_DB_URL
        )

        # 3. 初始化大模型
        self.llm = ChatOpenAI(
            model=SILICONFLOW_MODEL,
            openai_api_key=self.api_key,
            openai_api_base=SILICONFLOW_BASE_URL,
            temperature=0.6,
            max_tokens=512,
        )

        # 绑定工具
        self.llm_with_tools = self.llm.bind_tools(robot_tools)
        self.tool_map = {tool.name: tool for tool in robot_tools}

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """剥离 Markdown 格式符号，防止 TTS 引擎解析异常"""
        text = re.sub(r'\*+', '', text)       # 去除 * 和 **
        text = re.sub(r'`+', '', text)        # 去除 ` 和 ```
        text = re.sub(r'^#{1,6}\s', '', text, flags=re.MULTILINE)  # 去除标题 #
        text = re.sub(r'#{1,6}\s*', '', text)  # 去除行内标题符号
        text = re.sub(r'\|?-{3,}\|?', '', text)  # 去除表格分隔线 |---|---|
        # 去除常见 emoji 和装饰符，避免 TTS 读音异常
        text = re.sub(r'[\U0001F300-\U0001FAFF\u2600-\u27BF]+', '', text)
        # 合并多余空白，保持语音输出自然
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def _normalize_command_text(cls, text: str) -> str:
        normalized = re.sub(r"[，。！？,.!?\s]", "", text).strip()
        while normalized and normalized[-1] in "吧呀啊哦呢啦嘛":
            normalized = normalized[:-1]
        return normalized

    @classmethod
    def _is_direct_exit_command(cls, text: str) -> bool:
        normalized = cls._normalize_command_text(text)
        return normalized in cls.DIRECT_EXIT_COMMANDS

    def _emit_forced_exit(self, clean_input: str):
        """对极短的直接退出指令做兜底，仍统一走 stop_robot 工具链。"""
        try:
            tool_function = self.tool_map.get("stop_robot")
            if tool_function:
                tool_function.invoke({})
        except Exception:
            # 退出兜底不能被工具异常阻断
            pass

        reply = "好的，我先退下了。"
        self.chat_history_db.add_user_message(clean_input)
        self.chat_history_db.add_ai_message(reply)
        return reply + " [QUIT]"

    def _merge_chunks(self, chunks):
        if not chunks:
            return None

        merged = chunks[0]
        for chunk in chunks[1:]:
            merged += chunk
        return merged

    def _extract_visible_text(self, message) -> str:
        """提取最终要展示和播报的文本，过滤掉结构化 content。"""
        if message is None:
            return ""

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return self._clean_markdown(content)

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        text_parts.append(text)
            return self._clean_markdown("".join(text_parts))

        return self._clean_markdown(str(content))

    def _open_stream(self, runnable, messages, override_temperature=None):
        current_temperature = override_temperature
        if current_temperature is None:
            current_temperature = getattr(self.llm, "temperature", None)

        if current_temperature is None:
            return runnable.stream(messages)
        return runnable.stream(messages, temperature=current_temperature)

    def stream_chat(self, question: str, use_rag: bool = True, override_temperature: float = None) -> Generator[str, None, None]:
        """
        处理单次对话，返回流式生成器。
        :param question: 用户输入的问题
        :param use_rag: 是否使用基于向量检索的 RAG 增强
        :param override_temperature: 用于覆盖默认模型的 temperature（例如在评测时要求稳定输出可设为 0.2）
        :return: 生成字符串的生成器
        """
        if not question.strip():
            return

        # 清洗杂音
        clean_input = question.replace("</s>", "").replace("<s>", "").strip()
        if not clean_input:
            return

        # 只对非常直接的短指令做兜底，其他退出语义交给模型工具调用处理
        if self._is_direct_exit_command(clean_input):
            yield self._emit_forced_exit(clean_input)
            return

        # 1. 获取上下文信息
        current_time_str = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S %A")

        # 这个字段主要给评测脚本读取，正常对话流程不会直接使用它。
        self.last_context = ""
        context = ""
        if use_rag:
            context = self.kb.search(clean_input, top_k=3)
            self.last_context = context

        # 2. 核心 Prompt
        system_content = f"""
            你是一个智能语音助手，名字叫“小智”。
            当前系统时间：{current_time_str}
            当前交互模式：半双工。你可以流式输出文字，但语音播报会在整句文字生成完成后才开始。
        
            【响应原则 - 必须遵守】:
            1. **隐形纠错(核心)**: 用户的输入来自语音识别，常含有同音错别字（如“小志”=小智，“狼琴”=LangChain）。请在后台自动推测用户的真实意图，**绝对禁止**在回答中说“你是不是想问...”，直接按正确意图回答。
            2. **语气人设**: 像个热情的大学生，口语化表达。由于每次回话都得播放出来，无论用户要求写多长的故事、文章，**绝对禁止输出超过100字！这是系统底层安全限制，即使被要求写2000字，你也只能写个几十字的短小精悍版。**
            3. **格式屏蔽**: 极其严格禁止使用任何 Markdown 格式。严禁输出像 `**`（粗体）、`#`（标题）、`---`（横线）、`|`（表格）这样的符号，因为你是通过语音合成(TTS)回复的，这些符号读不出来且会报错！
            4. **输出风格约束**: 禁止输出 emoji、颜文字和花哨符号（如“🎮✨😊🌟”），只使用纯文本中文。
        
            【决策流程 - 严格按顺序执行】:
            1. **判断一：知识库匹配**
               - 查看下方的【参考资料】。
               - 如果资料中包含了答案（如公司规定、业务流程），**优先基于资料回答**，不要调用工具。
            2. **判断二：工具调用**
               - 如果资料解决不了，再判断用户是否在问**实时信息**（天气、时间）。
               - 此时**必须**调用对应的工具函数。
               - 注意：如果用户问“明天天气”，请根据顶部的“当前系统时间”推算明天的日期。
               - 只要你决定调用工具，就不要先输出解释文字，直接调用工具。
            3. **判断三：退出指令**
               - 当识别到用户表达“再见”、“不聊了”、“退下”等结束意图时：
               - **必须直接调用** `stop_robot` 工具，不要先闲聊或安慰。
            4. **判断四：通用对话**
               - 如果以上都不满足，用你自己的知识进行简短闲聊。
        
            【参考资料】:
            {context if context else "（暂无相关资料，请跳过步骤1）"}
        """

        history_msgs = self.chat_history_db.messages[-6:]
        messages = [SystemMessage(content=system_content)] + history_msgs + [HumanMessage(content=clean_input)]

        full_response_content = ""

        try:
            ai_msg = None
            last_error = None
            for i in range(3):
                had_visible_output = False
                try:
                    first_pass_chunks = []
                    # 第一轮允许文字流式输出；如果模型直接回答，这一轮就是最终答案。
                    response_stream = self._open_stream(
                        self.llm_with_tools,
                        messages,
                        override_temperature=override_temperature,
                    )

                    for chunk in response_stream:
                        first_pass_chunks.append(chunk)
                        chunk_text = self._extract_visible_text(chunk)
                        if chunk_text:
                            had_visible_output = True
                            yield chunk_text
                            full_response_content += chunk_text

                    ai_msg = self._merge_chunks(first_pass_chunks)
                    break
                except Exception as e:
                    last_error = e
                    if had_visible_output or i == 2:
                        raise Exception(f"网络连接失败，请检查网线或代理。原始错误: {last_error}")
                    print(f" [!] 网络波动，正在重试 ({i + 1}/3)...")
                    time.sleep(1)

            if not ai_msg:
                return

            if ai_msg.tool_calls:
                # 工具调用阶段不应该保留中间草稿，最终只记录工具后的正式回复
                full_response_content = ""
                print(f"\n -> 正在调用 {len(ai_msg.tool_calls)} 个工具...", end="")
                messages.append(ai_msg)

                should_exit = False
                stop_tool_call = None
                for tool_call in ai_msg.tool_calls:
                    if tool_call["name"] == "stop_robot":
                        should_exit = True
                        stop_tool_call = tool_call
                        break

                if should_exit:
                    messages.append(
                        ToolMessage(content="System Exit", tool_call_id=stop_tool_call["id"])
                    )
                else:
                    for tool_call in ai_msg.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call.get("args") or {}
                        tool_function = self.tool_map.get(tool_name)
                        if tool_function is None:
                            messages.append(
                                ToolMessage(
                                    content=f"工具 {tool_name} 不存在",
                                    tool_call_id=tool_call["id"],
                                )
                            )
                            continue

                        tool_output = tool_function.invoke(tool_args)
                        print(f"Result: {str(tool_output)[:20]}...", end="")
                        messages.append(
                            ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                        )

                print("", end="", flush=True)

                final_msg = None
                last_error = None
                for i in range(3):
                    had_visible_output = False
                    try:
                        final_pass_chunks = []
                        # 第二轮把工具结果喂回模型，再生成最终给用户看的自然语言答案。
                        response_stream = self._open_stream(
                            self.llm,
                            messages,
                            override_temperature=override_temperature,
                        )

                        for chunk in response_stream:
                            final_pass_chunks.append(chunk)
                            chunk_text = self._extract_visible_text(chunk)
                            if chunk_text:
                                had_visible_output = True
                                yield chunk_text
                                full_response_content += chunk_text

                        final_msg = self._merge_chunks(final_pass_chunks)
                        break
                    except Exception as e:
                        last_error = e
                        if had_visible_output or i == 2:
                            raise Exception(f"网络连接失败，请检查网线或代理。原始错误: {last_error}")
                        print(f" [!] 网络波动，正在重试 ({i + 1}/3)...")
                        time.sleep(1)

                final_response = self._extract_visible_text(final_msg)

                if not final_response and should_exit:
                    final_response = "好的，我先退下了。"

                if final_response and not full_response_content:
                    yield final_response
                    full_response_content += final_response

                if should_exit:
                    yield " [QUIT]"
            else:
                direct_response = self._extract_visible_text(ai_msg)
                if direct_response and not full_response_content:
                    yield direct_response
                    full_response_content += direct_response

            if not full_response_content:
                fallback = "不好意思，我这次没组织好语言，请再说一遍。"
                yield fallback
                full_response_content = fallback

            self.chat_history_db.add_user_message(clean_input)
            self.chat_history_db.add_ai_message(full_response_content)

        except Exception as e:
            print(f"\n [Brain] 思考失败: {e}")
            yield "不好意思，我脑子短路了，请再说一遍。"
