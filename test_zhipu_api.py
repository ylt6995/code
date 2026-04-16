import os
import openai

def main():
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少环境变量 ZHIPU_API_KEY。\n"
            "请在同一个 PowerShell 里先执行：\n"
            '$env:ZHIPU_API_KEY="你的智谱key"\n'
            '$env:ZHIPU_BASE_URL="https://open.bigmodel.cn/api/paas/v4"\n'
            '$env:ZHIPU_MODEL="glm-4.7"\n'
            "再运行：python test_zhipu_api.py\n"
        )

    base_url = os.getenv("ZHIPU_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4"
    model = os.getenv("ZHIPU_MODEL") or "glm-4.7"

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个严格按要求输出JSON的助手。"},
            {"role": "user", "content": "请只输出一个JSON：{\"collusionSuspicionScore\":0,\"riskLevel\":\"Low\"}"},
        ],
        timeout=60,
        temperature=0.1,
        top_p=0.9,
        max_tokens=200,
    )
    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()
