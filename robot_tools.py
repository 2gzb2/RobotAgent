import requests
import datetime
from langchain_core.tools import tool


# ================= 定义工具函数 =================

@tool
def get_current_time():
    """获取当前的日期、时间、星期几。"""
    now = datetime.datetime.now()
    week_days = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return now.strftime(f"%Y年%m月%d日 %H:%M {week_days[now.weekday()]}")


@tool
def get_weather(city: str):
    """
    查询某个城市的实时天气。
    参数 city: 城市名称 (例如: "北京", "蒙城县")。
    """
    url = f"https://wttr.in/{city}?format=3&lang=zh"

    # 策略1：强制直连（绕过代理），适用于未开代理或代理仅为系统级代理的场景
    # 策略2：使用系统默认代理，适用于开启了 TUN/全局代理且直连被拦截的场景
    strategies = [
        ("直连", {"http": None, "https": None}),
        ("系统代理", None),
    ]

    last_error = None
    for name, proxies in strategies:
        try:
            kwargs = {"timeout": 8}
            if proxies is not None:
                kwargs["proxies"] = proxies
            response = requests.get(url, **kwargs)
            if response.status_code == 200:
                return f"查询结果: {response.text.strip()}"
        except Exception as e:
            last_error = e

    return f"天气查询失败（直连与代理均不通）: {last_error}"


@tool
def stop_robot():
    """
    当用户明确表示要结束对话（如“再见”、“关机”、“退下”、“不聊了”）时，调用此工具。
    不需要任何参数。
    """
    return "SYSTEM_EXIT_SIGNAL"


# 导出工具列表
robot_tools = [get_current_time, get_weather, stop_robot]