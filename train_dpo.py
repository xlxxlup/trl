"""
DPO (Direct Preference Optimization) 训练脚本

DPO需要偏好数据格式：
- prompt: 输入提示
- chosen: 偏好的回复
- rejected: 不被偏好的回复

示例数据格式：
{
    "prompt": "什么是人工智能？",
    "chosen": "人工智能（AI）是计算机科学的一个分支...",
    "rejected": "我不知道。"
}
"""

import json
from datasets import Dataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'  # 使用两张卡
import swanlab
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def load_dpo_data(jsonl_path):
    """加载DPO训练数据

    Args:
        jsonl_path: JSONL格式的数据文件路径

    Returns:
        Dataset: 包含prompt, chosen, rejected字段的Dataset
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            prompt_messages = item["prompt_messages"]

            # 提取除最后一条user消息外的所有消息作为context
            # 最后一条user消息是需要模型回答的问题
            messages = []
            for i, msg in enumerate(prompt_messages):
                if i == len(prompt_messages) - 1:  # 最后一条消息
                    # 跳过，这是待回答的问题
                    pass
                else:
                    messages.append(msg)

            data.append({
                "prompt": messages,  # 历史对话（不含最后一条user消息）
                "chosen": [{"role": "assistant", "content": item["chosen"]}],
                "rejected": [{"role": "assistant", "content": item["rejected"]}]
            })
    return Dataset.from_list(data)


def main():
    # 初始化SwanLab
    swanlab.init(
        project="Qwen3-4B-SDFT-DPO",  # 项目名称
        experiment_name="dpo-checkpoint343-lr1e5",  # 实验名称
        description="DPO training on SFT checkpoint-343 with LoRA r=64"
    )

    # 模型路径 - 使用SFT后的checkpoint
    model_path = "/data2/xiangl/projtct/SDFT/output/checkpoint-343"

    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype="bfloat16",  # 使用bfloat16精度训练
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,  # 修复tokenizer regex问题
    )

    # 设置pad_token（如果模型没有）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA配置 - 在SFT模型基础上继续训练
    peft_config = LoraConfig(
        r=64,            # 增大LoRA秩以获得更好的表达能力
        lora_alpha=32,   # LoRA缩放因子
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 目标模块
        task_type="CAUSAL_LM",
    )

    # 加载数据
    # 需要准备包含prompt, chosen, rejected的数据
    train_dataset = load_dpo_data("/data2/xiangl/projtct/reward/dpo_training_data_messages.jsonl")

    # 也可以使用HuggingFace Hub上的数据集
    # from datasets import load_dataset
    # train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

    # 创建DPO训练器
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 如果为None，会使用当前模型作为参考模型（frozen）
        args=DPOConfig(
            output_dir=".output/dpo_output",
            learning_rate=1e-5,  # DPO学习率，可以根据实际情况调整
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            max_steps=1000,
            logging_steps=2,
            save_steps=100,
            # DPO特定参数
            beta=0.1,  # 控制与参考模型的偏离程度，值越小偏离越大
            max_length=2048,  # 最大序列长度
            loss_type="sigmoid",  # 损失类型：sigmoid, hinge, ipo, etc.
            label_smoothing=0.0,  # 标签平滑，用于Robust DPO
            # 其他参数
            remove_unused_columns=False,
            report_to=["swanlab"],  # 日志上报工具
            gradient_checkpointing=True,  # 启用梯度检查点以节省显存
            warmup_ratio=0.03,  # 预热比例
        ),
        train_dataset=train_dataset,  # 在这里传入你的训练数据集
        peft_config=peft_config,  # DPOTrainer会自动应用LoRA
    )

    # 开始训练
    trainer.train()

    # 保存LoRA模型
    trainer.save_model(".output/dpo_output/final_model")
    tokenizer.save_pretrained(".output/dpo_output/final_model")

    print("DPO LoRA训练完成！模型已保存到 .output/dpo_output/final_model")
    print("注意：这只是LoRA适配器权重，需要与SFT checkpoint合并使用")


if __name__ == "__main__":
    main()
