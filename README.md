# 智能语音助手系统 (RobotAgent) - 本科毕业设计

> 基于 DeepSeek-V3 大模型、Sherpa-Onnx 离线语音识别与 LangChain 架构的实时交互机器人。

## 项目简介

本项目旨在构建一个响应迅速、具备知识库检索（RAG）与工具调用能力的智能语音助手。系统采用“耳-脑-口”仿生架构，实现了从语音输入识别、语义理解、外部工具调用到语音合成的全流程闭环。

核心亮点在于实现了半双工交互：语音识别与终端文本回复保持流式，语音播报在整句文本生成完成后再开始。系统支持用户在机器人说话时通过按键实时打断，并针对国内网络环境进行了优化。

当前公开仓库聚焦主功能演示，只保留运行项目所需的核心代码、配置模板和说明文档；本地测试、评测和论文分析脚本不纳入公开版本。

## 技术架构

| 模块 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| 听 (Ear) | `Sherpa-Onnx` | 离线流式语音识别，支持中英双语，响应速度快，无需联网。 |
| 脑 (Brain) | `DeepSeek-V3` + `LangChain` | 接入 SiliconFlow API，利用 ReAct 范式实现工具调用与逻辑推理。 |
| 口 (Mouth) | `Edge-TTS` | 微软超逼真语音合成，文本生成完成后统一播报。 |
| 记忆 (Memory) | `SQLite` | 轻量级嵌入式数据库，持久化存储对话上下文。 |
| 知识库 (RAG) | `ChromaDB` + `text2vec` | 本地向量数据库 + 中文 Embedding 模型，保障隐私与检索准确率。 |

## 核心功能

1. 半双工流式交互：ASR 与终端文本回复为流式处理，TTS 在文本生成完成后统一播报。
2. RAG 知识增强：内置本地知识库（`knowledge.txt`），可回答关于知识库的私有数据。
3. 工具调用：支持天气、时间和退出控制等能力。
4. 智能打断机制：用户可随时按空格键打断 AI 发言，无需等待说完。
5. 鲁棒性设计：内置 HuggingFace 国内镜像源，并带有基础重试逻辑。
6. 环境变量配置：把 API Key 放到 `.env` 或系统环境变量中，避免把敏感信息写入源码仓库。

## 目录结构

```text
RobotAgent/
├── main.py              # 主程序入口（控制层）
├── robot_brain.py       # 大脑核心（LangChain, DeepSeek, Tools）
├── robot_ear.py         # 耳朵核心（Sherpa-Onnx 语音识别）
├── robot_mouth.py       # 嘴巴核心（Edge-TTS 语音合成与播放）
├── robot_tools.py       # 工具箱（天气、时间、系统指令）
├── config.py            # 全局配置文件
├── knowledge_base.py    # RAG 知识库管理
├── data/
│   └── knowledge.txt    # 知识库源文件
├── model/               # 存放离线模型文件
├── requirements.txt     # 项目依赖列表
└── README.md            # 项目说明文档
```

## 环境配置

1. 复制配置模板：

```powershell
Copy-Item .env.example .env
```

2. 打开 `.env`，填写你自己的真实 `SILICONFLOW_API_KEY`。
3. `.env` 仅用于本地运行，已加入 `.gitignore`，不要提交到 GitHub。

如果你不想使用 `.env`，也可以直接把这些变量配置到系统环境变量中。

## 模型准备

语音识别依赖离线模型文件，但大型 ONNX 文件不适合直接放进普通 Git 仓库。

首次运行前，请确认 `model/` 目录下存在以下文件：

- `encoder.int8.onnx`
- `decoder.int8.onnx`
- `tokens.txt`

说明：

- 仓库默认忽略 `model/*.onnx`，避免推送到 GitHub 时因大文件失败。
- 如果你从其他机器复制项目，需要把离线下载好的模型文件手动放回 `model/` 目录。
- 更详细的说明见 `model/README.md`。

## 安装与运行

```powershell
pip install -r requirements.txt
python main.py
```

## 首次运行会自动生成的文件

以下内容属于本地配置、运行缓存或评测产物，删除后不影响源码仓库的完整性，因此默认不建议提交到 GitHub：

- `.env`
- `memory.db`
- `temp_audio.mp3`
- `chroma_db/`
- `chroma_db_recovered/`
- `chroma_test_new/`
- `__pycache__/`
- `.idea/`
- `.vscode/`
- `results_no_rag.json`
- `results_with_rag.json`
- `rag_eval_comparison.csv`
- `rag_eval_summary.csv`
- `latency_data.json`
- `latency_chart.png`
- `model/*.onnx`

其中：

- `memory.db` 会在运行时自动生成。
- `chroma_db/` 会在首次构建知识库后自动生成。
- `temp_audio.mp3` 是 TTS 临时输出文件。
- `model/*.onnx` 需要你本地单独准备，但不应直接提交到普通 GitHub 仓库。
