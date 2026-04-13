from unsloth import FastLanguageModel
import concurrent.futures

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/hy-tmp/models/Qwen2.5-Coder-32B-Instruct",
    # model_name = "/hy-tmp/models/Qwen3-Coder-30B-A3B-Instruct",
    max_seq_length = 28000,
    local_files_only=True
)

from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen2.5" # qwen2.5
    # chat_template = "qwen3-instruct" # qwen3
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference


messages = [
    {"role": "user", "content": f"{text}"},
]
inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024, use_cache = True)

# test example 1
text = ""
with open("test1.txt", "r", encoding="utf-8") as f:
    text = f.read()


messages = [
    {"role": "user", "content": f"{text}"},
]
inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024, use_cache = True)

# test example 2
text = ""
with open("test2.txt", "r", encoding="utf-8") as f:
    text = f.read()


messages = [
    {"role": "user", "content": f"{text}"},
]
inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024, use_cache = True)
