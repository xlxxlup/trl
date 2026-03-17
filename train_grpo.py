import json
from datasets import Dataset
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
from trl import GRPOTrainer, GRPOConfig


def load_grpo_data(jsonl_path):
    """加载GRPO训练数据"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "prompt": item["prompt"],
                "metadata": item["metadata"]
            })
    return Dataset.from_list(data)


def custom_reward_func(prompts, completions, **kwargs):
    """自定义奖励函数"""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # 简单示例：基于补全长度给予奖励
        reward = len(completion) / 100.0
        rewards.append(reward)
    return rewards


def main():
    # 加载数据
    train_dataset = load_grpo_data("/data2/xiangl/projtct/trl/data/grpo_train_data.jsonl")

    # 创建GRPO训练器
    trainer = GRPOTrainer(
        model="/data/public/Qwen/Qwen2.5/Qwen3-4B-Instruct-2507/",
        reward_funcs=custom_reward_func,
        train_dataset=train_dataset,
        args=GRPOConfig(
            output_dir=".output/grpo_output",
            learning_rate=1e-6,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_generations=4,
            max_completion_length=256,
            max_steps=10,
            logging_steps=1,
            log_completions=True,
            num_completions_to_print=2,
            beta=0.001,
            temperature=1.0,
            remove_unused_columns=False,  # 保留metadata列供奖励函数使用
        )
    )

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model("./grpo_output/final_model")


if __name__ == "__main__":
    main()
