import os
import openai

def main():
    api_key = os.environ["GPT_API_KEY"]
    base_url = os.environ.get("GPT_BASE_URL", "https://api.chatanywhere.tech/v1")
    model = os.environ.get("GPT_MODEL", "gpt-4o-mini")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "请只输出一个JSON：{\"collusionSuspicionScore\":0,\"riskLevel\":\"Low\"}"}],
        timeout=60,
    )
    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()