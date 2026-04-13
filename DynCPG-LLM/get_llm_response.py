from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com")


def get_ds_response(llm_messages, temperature):
    response = client.chat.completions.create(
        model="deepseek-reasoner",  # 思考模式
        messages=llm_messages,
        temperature=temperature,
        stream=False,
        response_format={
            'type': 'json_object'
        }
    )
    return response


def get_query_response(prompt):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    chat_response = client.chat.completions.create(
        # model="/hy-tmp/models/Qwen2.5-Coder-32B-Instruct_cpgql-2_10",
        model="/hy-tmp/Qwen2.5-Coder-32B-Instruct_cpgql-2_24",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        presence_penalty=1.05
    )
    # 基础queries
    queries = ast.literal_eval(chat_response.choices[0].message.content)
    return queries