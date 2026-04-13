from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com")


def get_response(messages, temperature):
    response = client.chat.completions.create(
        model="deepseek-reasoner",  # 思考模式
        messages=messages,
        temperature=temperature,
        stream=False,
        response_format={
            'type': 'json_object'
        }
    )
    return response