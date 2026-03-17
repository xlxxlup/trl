"""
判别式奖励模型训练脚本

奖励模型用于对模型生成的回复进行打分，通常用于RLHF训练流程。
训练数据需要包含成对的偏好数据，让模型学会给好的回复更高分数。

数据格式示例：
{
    "prompt": "什么是人工智能？",
    "chosen": "人工智能（AI）是计算机科学的一个分支...",
    "rejected": "我不知道。"
}

训练目标：
- chosen回复的分数 > rejected回复的分数
- 通常使用margin loss或ranking loss
"""

import json
from datasets import Dataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model


def load_reward_data(jsonl_path):
    """加载奖励模型训练数据

    Args:
        jsonl_path: JSONL格式的数据文件路径

    Returns:
        Dataset: 包含prompt, chosen, rejected字段的Dataset
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"]
            })
    return Dataset.from_list(data)


def main():
    # 模型路径
    model_path = "/data/public/Qwen/Qwen2.5/Qwen3-4B-Instruct-2507/"

    # 加载序列分类模型（用于输出奖励分数）
    # num_labels=1 表示输出单个标量分数
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,  # 奖励模型输出单个分数
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 设置pad_token（如果模型没有）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA配置（用于奖励模型）
    # 注意：奖励模型是分类任务，所以task_type是SEQ_CLS
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="SEQ_CLS",  # 序列分类任务
    )

    # 应用LoRA到模型
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 加载数据
    # train_dataset = load_reward_data("/data2/xiangl/projtct/trl/data/reward_train_data.jsonl")

    # 也可以使用HuggingFace Hub上的数据集
    # from datasets import load_dataset
    # train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

    # 创建奖励模型训练器
    trainer = RewardTrainer(
        model=model,
        args=RewardConfig(
            output_dir=".output/reward_model_output",
            learning_rate=1e-4,  # 奖励模型通常使用较大的学习率
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=100,
            # 奖励模型特定参数
            max_length=1024,  # 最大序列长度
            center_rewards_coefficient=0.01,  # 奖励中心化系数，让奖励均值接近0
            # 其他参数
            remove_unused_columns=False,
            report_to=["tensorboard"],
            gradient_checkpointing=True,  # 启用梯度检查点以节省显存
        ),
        train_dataset=None,  # 在这里传入你的训练数据集
        # eval_dataset=None,  # 可选：验证集
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # 开始训练
    trainer.train()

    # 保存奖励模型
    trainer.save_model("./reward_model_output/final_model")
    tokenizer.save_pretrained("./reward_model_output/final_model")

    print("奖励模型训练完成！模型已保存到 ./reward_model_output/final_model")
    print("注意：这是LoRA适配器权重，使用时需要加载到基础模型上")


def inference_with_reward_model(model_path, tokenizer_path, prompt, response):
    """
    使用训练好的奖励模型进行推理的示例函数

    Args:
        model_path: 训练好的模型路径
        tokenizer_path: tokenizer路径
        prompt: 输入提示
        response: 模型回复

    Returns:
        reward_score: 奖励分数
    """
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # 加载基础模型
    base_model = AutoModelForSequenceClassification.from_pretrained(
        tokenizer_path,
        num_labels=1,
        trust_remote_code=True,
    )

    # 加载LoRA权重
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 构造输入
    text = f"{prompt}{response}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

    # 获取奖励分数
    with torch.no_grad():
        outputs = model(**inputs)
        reward_score = outputs.logits[0][0].item()

    return reward_score


if __name__ == "__main__":
    main()

    # 推理示例
    # score = inference_with_reward_model(
    #     model_path="./reward_model_output/final_model",
    #     tokenizer_path="/data/public/Qwen/Qwen2.5/Qwen3-4B-Instruct-2507/",
    #     prompt="什么是人工智能？",
    #     response="人工智能是计算机科学的一个分支..."
    # )
    # print(f"奖励分数: {score}")
