from langchain.chat_models import init_chat_model
from sympy.physics.units import temperature

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, ZHIPU_API_KEY, \
    ZHIPU_BASE_URL, HUNYUAN_API_KEY, HUNYUAN_BASE_URL

# 初始化Deepseek模型
deepseek_llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# 初始化千问模型
dashscope_llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    temperature = 0 # 温度参数，控制模型的随机性，0表示确定性，1表示随机性
)

# 初始化智谱模型
zhipu_llm = init_chat_model(
    model="glm-5",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
)

# 初始化混元模型
hunyuan_llm = init_chat_model(
    model="hunyuan-turbos-latest",
    model_provider="openai",
    api_key=HUNYUAN_API_KEY,
    base_url=HUNYUAN_BASE_URL,
)
