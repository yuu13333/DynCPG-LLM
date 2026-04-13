from unsloth import FastLanguageModel, is_bfloat16_supported, FastModel
import torch
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import json
import logging
import argparse
from sklearn.model_selection import train_test_split
import bitsandbytes as bnb
import os
from unsloth import FastModel
from unsloth import is_bfloat16_supported
from peft import get_peft_model


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cpg_query_finetune.log')
        ]
    )
    return logging.getLogger(__name__)


def load_json_dataset(file_path: str, eval_split_ratio: float = 0.1) -> tuple[Dataset, Dataset]:
    data = []
    with open(file_path, 'r') as f:
        json_lines = f.readlines()
        for line in json_lines:
            item = json.loads(line)
            data.append(item)

    # 需要一个system prompt
    with open("/train/cpgql_system_prompt.txt", 'r') as f:
        system_prompt = f.read()

    conversations = []
    for item in data:
        code, desc, queries = item["func_code"], item["context_desc"], item["queries"]
        conv = [
            {"role": "user",
             "content": f"{system_prompt}\nGiven code: {code}\nRequired Context Description: {desc}".strip()},
            {"role": "assistant", "content": f"{queries}"}
        ]
        conversations.append({"conversations": conv})

    train_data, eval_data = train_test_split(conversations, test_size=eval_split_ratio, random_state=42)

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
          
    return train_dataset, eval_dataset


def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(
        convo,
        tokenize=False,
        add_generation_prompt=False
    ) for convo in convos]
    return {"text": texts}


def find_all_linear_names(model):
    # 4bit量化linear
    cls = bnb.nn.Linear4bit
    lora_module_names = set()

    for name, module in model.named_modules():  # named_modules模型中所有子模块的名称和对应的模块实例
        if isinstance(module, cls):  # 当模块是4bit量化
            names = name.split('.')  # 名称用.分割 取最后一个模块名 方便LoraConfig的target_modules设置
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # 指的是输出层
        lora_module_names.remove('lm_head')

    return list(lora_module_names)


def setup_model_and_tokenizer(args, logger):
    model_dir = "unsloth/Qwen3-Coder-30B-A3B-Instruct"
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        cache_dir="/hy-tmp"
    )

    target_modules = find_all_linear_names(model)  # 先找所有的4bit linear，每一个linear都需要添加适配器
    logger.info(f"Target Modules: {str(target_modules)}")

    # 在指定层插入 LoRA adapter，实现高效微调
    model = FastModel.get_peft_model(
        model,
        r=args.rank,  # LoRA适配器的低秩矩阵的维度（秩），直接影响可训练参数数量
        target_modules=target_modules,  # 哪些模型层的权重矩阵需要添加LoRA适配器
        lora_alpha=args.alpha,  # 控制LoRA适配器输出的缩放权重（即适配器结果乘以 alpha/r）
        lora_dropout=0.1,  # 丢弃率
        bias="none",  # 不要偏置
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,  # 随机种子，确保可复现
        use_rslora=False,  # 使用RSLoRA可提升稳定性 可以尝试下原始lora和rslora
        loftq_config=None,  # 无特殊需求保持none
    )


    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen3-instruct",
    )  # 匹配分词器模版

    return model, tokenizer


def setup_trainer(model, tokenizer, train_dataset, eval_dataset, args, logger):
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            per_device_eval_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_ratio=0.10,
            num_train_epochs = 3, # Set this for 1 full training run.
            # max_steps = 15,
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use TrackIO/WandB etc
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=5,
            output_dir=args.model_output_dir,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            greater_is_better=False
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    return trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="unsloth/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="/train/training_dataset.jsonl")  
    parser.add_argument("--max_seq_length", type=int, default=28000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--logging_steps", type=int, default=2)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--experiment_name", default="cpgql")
    parser.add_argument("--output_dir", type=str, default="/hy-tmp/models")
    parser.add_argument("--eval_split_ratio", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=8, help="LoRA attention dimension (rank)")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha parameter")
    args = parser.parse_args()
    args.model_output_dir = f"{args.output_dir}/{args.model_name.split('/')[-1]}_{args.experiment_name}"
    return args


def main():
    args = parse_args()
    logger = setup_logging()

    logger.info(f"Loading model: {args.model_name}")
    model, tokenizer = setup_model_and_tokenizer(args, logger)

    logger.info(
        f"Loading dataset from: {args.dataset_path} and splitting for evaluation ({args.eval_split_ratio * 100:.0f}/{100 - args.eval_split_ratio * 100:.0f})")
    train_dataset, eval_dataset = load_json_dataset(args.dataset_path, args.eval_split_ratio)

    # ShareGPT -> ChatML
    train_dataset = standardize_sharegpt(train_dataset)
    eval_dataset = standardize_sharegpt(eval_dataset)

    train_dataset = train_dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True
    )
    eval_dataset = eval_dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True
    )

    logger.info("Setting up trainer")
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset, args, logger)

    logger.info("Starting training")
    trainer_stats = trainer.train()

    logger.info("Saving model")
    best_checkpoint = trainer.state.best_model_checkpoint
    print(best_checkpoint)
    model.save_pretrained(args.model_output_dir)
    tokenizer.save_pretrained(args.model_output_dir)

    model.save_pretrained_merged(args.model_output_dir + "-2_10", tokenizer, save_method="merged_16bit", )

    logger.info("Training completed")
    return trainer_stats


if __name__ == "__main__":
    main()
