import os
from openai import OpenAI

# 从环境变量中获取您的API KEY，配置方法见：https://www.volcengine.com/docs/82379/1399008
api_key = os.getenv('ARK_API_KEY')

client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="b7a14aed-bdb0-419c-b8f5-0bc22a66dccd",
)

system_prompt = "你是我的人工智能助手，协助我分析图片内容并回答相关问题。"
response = client.responses.create(
    model="ep-20260215004450-nf8t8",
    input=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "你好",
        }
    ]
)

print(response)