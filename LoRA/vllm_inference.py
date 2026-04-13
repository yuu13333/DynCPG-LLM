from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def get_response(prompt):
    chat_response = client.chat.completions.create(
        # model="/hy-tmp/models/Qwen2.5-Coder-32B-Instruct_cpgql-2_10",
        model="/hy-tmp/models/Qwen3-Coder-30B-A3B-Instruct_cpgql-2_10",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        presence_penalty=1.05,
    )
    print("Chat response:", chat_response.choices[0].message.content)


with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()
    get_response(text)