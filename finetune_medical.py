# ==============================================================================
# 🏥 医疗大模型微调脚本 (QLoRA)
# 功能:
# - 使用我们之前生成的医疗CoT数据集进行模型微调。
# - 采用QLoRA技术，实现高效的4-bit量化微调。
# - 动态加载JSON数据集并进行格式化。
# - 将所有配置项集中管理，方便调整。
#
# v1.5 更新:
# - 解决了因缺少 `tensorboard` 库导致的 RuntimeError。
# - 将默认的 `report_to` 参数从 "tensorboard" 修改为 "none"，以避免不必要的依赖。
#
# v1.4 更新:
# - 解决了 `SFTTrainer` 不接受 `max_seq_length` 参数的 TypeError。
# ==============================================================================

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# ================================
# 1. 配置参数
# ================================
class ModelConfig:
    # --- 模型与路径配置 (已根据您的提供进行更新) ---
    # 要微调的基础模型ID (本地路径)
    base_model_id: str = "/home/mw/input/models2179" 
    # 训练数据路径
    dataset_path: str = "output/medical_cot_alpaca_v1.json"
    # 微调过程中检查点和日志的输出目录
    output_dir: str = "./results"
    # 最终保存LoRA适配器的路径
    final_model_path: str = "/home/mw/input/qwen4b-medical-cot"

    # --- QLoRA 量化配置 ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --- LoRA 配置 (适用于Qwen2/3系列模型) ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # --- 训练参数 ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=1,
        max_steps=-1,
        save_steps=50,
        fp16=True,
        optim="paged_adamw_32bit",
        # 关键修复：禁用TensorBoard报告，以避免依赖错误。
        # 如果您想使用TensorBoard，请先运行 `pip install tensorboard`，然后将此项改回 "tensorboard"
        report_to="none",
        logging_dir=f"{output_dir}/logs",
        save_strategy="steps",
        load_best_model_at_end=False,
    )

# ================================
# 2. 主训练流程
# ================================
def main():
    # 加载配置
    config = ModelConfig()
    
    # --- 加载数据集 ---
    print(f"🔄 正在从 {config.dataset_path} 加载数据集...")
    dataset = load_dataset("json", data_files=config.dataset_path, split="train")
    print(f"✅ 数据集加载成功，共 {len(dataset)} 条记录。")

    # --- 加载模型和Tokenizer ---
    print(f"🔄 正在加载基础模型: {config.base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        quantization_config=config.bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, trust_remote_code=True)
    
    tokenizer.model_max_length = 1024
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ 模型和Tokenizer加载完成。")

    # --- 定义数据格式化函数 ---
    def formatting_func(example):
        # 为每个样本构建符合模型聊天模板的消息列表
        messages = [
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": example['output']}
        ]
        # 使用tokenizer应用聊天模板
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # --- 初始化SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        args=config.training_args,
        train_dataset=dataset,
        peft_config=config.peft_config,
        formatting_func=formatting_func,
    )
    
    # --- 开始训练 ---
    print("🚀 开始模型微调...")
    trainer.train()
    print("🎉 模型微调完成！")

    # --- 保存最终模型 ---
    print(f"💾 正在将训练好的LoRA适配器保存到 {config.final_model_path}...")
    os.makedirs(config.final_model_path, exist_ok=True)
    trainer.save_model(config.final_model_path)
    tokenizer.save_pretrained(config.final_model_path)
    print("✅ 模型保存成功！")

if __name__ == "__main__":
    main()
