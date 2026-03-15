import asyncio
import edge_tts
import pygame
import keyboard
import time
from config import TTS_VOICE, TTS_OUTPUT_FILE


class RobotMouth:
    def __init__(self):
        pygame.mixer.init()
        self.voice = TTS_VOICE
        self.output_file = TTS_OUTPUT_FILE

    async def _generate_audio(self, text):
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(self.output_file)

    def speak(self, text: str):
        if not text:
            return

        # 生成语音
        try:
            asyncio.run(self._generate_audio(text))
        except Exception as e:
            print(f" 语音生成失败: {e}")
            return

        # 播放语音
        self._play_audio_with_interrupt()

    def _play_audio_with_interrupt(self):
        audio_loaded = False
        try:
            pygame.mixer.music.load(self.output_file)
            audio_loaded = True
            pygame.mixer.music.play()

            # 播放阶段只处理“空格打断”这一条公开交互规则。
            while pygame.mixer.music.get_busy():
                if keyboard.is_pressed('space'):
                    pygame.mixer.music.stop()
                    print(" [!] 用户打断播放")
                    return

                time.sleep(0.05)
        except Exception as e:
            print(f" 播放失败: {e}")
        finally:
            # 只有在成功加载过音频时，才做卸载，避免清理阶段覆盖原始异常。
            if audio_loaded:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
                try:
                    pygame.mixer.music.unload()
                except Exception:
                    pass
