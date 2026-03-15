import sherpa_onnx
import pyaudio
import numpy as np
import os
import time
from config import ASR_MODEL_PATH, VAD_SILENCE_DURATION, MAX_RECORD_TIME


class RobotEar:
    def __init__(self):
        self.model_dir = ASR_MODEL_PATH
        self.sample_rate = 16000
        self.chunk_size = 1024

        # 检查关键文件
        required_files = ["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"]
        for f in required_files:
            if not os.path.exists(os.path.join(self.model_dir, f)):
                raise RuntimeError(f"模型文件缺失: {f}")

        print(" 正在加载语音识别模型...", end="")
        try:
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
                tokens=f"{self.model_dir}/tokens.txt",
                encoder=f"{self.model_dir}/encoder.int8.onnx",
                decoder=f"{self.model_dir}/decoder.int8.onnx",
                num_threads=1,
                sample_rate=self.sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
                enable_endpoint_detection=True,
                # 使用配置文件里的 1.5 秒
                rule1_min_trailing_silence=VAD_SILENCE_DURATION,
            )
            print("")
        except Exception as e:
            raise RuntimeError(f"语音识别模型加载失败: {e}") from e

        try:
            self.audio = pyaudio.PyAudio()
        except Exception as e:
            raise RuntimeError(f"PyAudio 初始化失败: {e}") from e

    def __del__(self):
        try:
            if hasattr(self, "audio"):
                self.audio.terminate()
        except Exception:
            pass

    def listen(self):
        stream = self.recognizer.create_stream()
        audio_stream = None

        # 记录开始时间，用于超时强制打断
        start_time = time.time()

        # 临时存放识别到的文本
        last_text = ""

        try:
            audio_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

            while True:
                # 1. 检查是否超时 (比如环境太吵，一直不断句，强行停止)
                # 超时时尽量返回最后一次识别结果，而不是直接丢空，减少用户重说成本。
                if time.time() - start_time > MAX_RECORD_TIME:
                    print(f"\n 说话时间过长 ({MAX_RECORD_TIME}s)，强制停止识别。")
                    return last_text

                samples = audio_stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(samples, dtype=np.float32)
                stream.accept_waveform(self.sample_rate, audio_array)

                while self.recognizer.is_ready(stream):
                    self.recognizer.decode_stream(stream)

                result = self.recognizer.get_result(stream)

                # 实时显示部分结果
                if result:
                    last_text = result
                    print(f"\r[...] 识别中: {result}", end="", flush=True)

                # 2. 正常检测到静音结束
                if self.recognizer.is_endpoint(stream):
                    final_result = self.recognizer.get_result(stream)
                    if final_result:
                        return final_result
                    else:
                        # 只是噪音，重置流
                        self.recognizer.reset(stream)

        except Exception as e:
            print(f"\n 麦克风异常: {e}")
            return ""
        finally:
            if audio_stream is not None:
                try:
                    audio_stream.stop_stream()
                except Exception:
                    pass
                try:
                    audio_stream.close()
                except Exception:
                    pass
