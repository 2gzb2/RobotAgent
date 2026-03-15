import os
import warnings
import logging
import threading
import keyboard
import time
import re
import config

# ================= 消音区 =================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
# =========================================

from robot_ear import RobotEar
from robot_brain import RobotBrain
from robot_mouth import RobotMouth

TTS_OUTPUT_FILE = config.TTS_OUTPUT_FILE
POST_TTS_LISTEN_DELAY = 0.6
if hasattr(config, "POST_TTS_LISTEN_DELAY"):
    try:
        POST_TTS_LISTEN_DELAY = float(config.POST_TTS_LISTEN_DELAY)
        if POST_TTS_LISTEN_DELAY < 0:
            POST_TTS_LISTEN_DELAY = 0.0
    except (TypeError, ValueError):
        POST_TTS_LISTEN_DELAY = 0.6

# 全局退出标志
is_running = True


def check_exit_key():
    """后台线程：随时监听 ctrl+q 键退出"""
    global is_running
    while is_running:
        if keyboard.is_pressed('ctrl+q'):
            print("\n\n [!] 检测到强制退出指令 (Ctrl+Q)...")
            is_running = False
            os._exit(0)
        time.sleep(0.1)


def print_banner():
    print("\n" + "=" * 60)
    print("   [ 智能语音助手系统 v3 (半双工版本) ]")
    print("   核心驱动: DeepSeek-V3 + EdgeTTS + Sherpa-Onnx")
    print("-" * 60)
    print("   [ 文本 ]  流式输出 | [ 语音 ]  回复完成后播报 ")
    print("   [ 空格键 ]  打断 AI 说话 ")
    print("   [ ctrl+q ]  退出系统 ")
    print("=" * 60 + "\n")


def main():
    global is_running
    print_banner()

    # Ctrl+Q 是“强制退出”，因此放在独立后台线程里，避免主循环卡在录音或播报阶段。
    exit_thread = threading.Thread(target=check_exit_key, daemon=True)
    exit_thread.start()

    print("[*] 系统初始化中...", end="", flush=True)

    try:
        brain = RobotBrain()
        ear = RobotEar()
        mouth = RobotMouth()
    except Exception as e:
        print(f"\n 初始化失败: {e}")
        return

    if os.path.exists(TTS_OUTPUT_FILE):
        try:
            os.remove(TTS_OUTPUT_FILE)
        except OSError:
            pass

    print(" [+] 系统就绪。\n")
    print("-" * 30)

    try:
        while is_running:
            # 1. 听
            print("\n ------------------------------\n")
            print(" [...] 识别中: ", end="", flush=True)
            user_text = ear.listen()
            if not user_text:
                continue

            # 2. 想
            # 增加 \n\n 实现空两行，区分“识别中”和“思考中”
            # 输出后不换行，且后面不跟内容
            print("\n\n[>>>] 思考中...", end="", flush=True)

            full_response = ""
            is_first_chunk = True

            for chunk in brain.stream_chat(user_text):
                # 1. 收到第一个字时，另起一行显示名字
                if is_first_chunk:
                    print("\n[Brain] 小智: ", end="", flush=True)
                    is_first_chunk = False

                # 2. 直接打印，包含 [QUIT]
                print(chunk, end="", flush=True)

                # 3. 拼接完整句子 (用于后续逻辑判断)
                full_response += chunk

            print("")  # 结束换行

            # 3. 处理退出信号
            should_exit = False
            clean_response = full_response

            if re.search(r"\[QUIT\]", full_response, re.IGNORECASE):
                should_exit = True
                # [QUIT] 只是主程序内部控制标记，不应该被送进 TTS。
                clean_response = re.sub(r"\[QUIT\]", "", full_response, flags=re.IGNORECASE).strip()

            # 4. 说 (拿到拼接好的完整句子，送去读)
            # 只有当有内容时才读，防止空语音报错
            if clean_response:
                mouth.speak(clean_response)
                if POST_TTS_LISTEN_DELAY > 0:
                    time.sleep(POST_TTS_LISTEN_DELAY)

            if should_exit:
                is_running = False
                print("\n 收到大脑的退出指令，系统关闭中...")
                print(" 再见！")
                break

    except KeyboardInterrupt:
        is_running = False
        print("\n [!] 用户强制中断")
    finally:
        print("")


if __name__ == "__main__":
    main()
