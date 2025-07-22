# ==============================================================================
# 🏥 微调前后模型效果对比脚本
# 功能:
# - 同时加载基础模型和微调后的模型。
# - 对预设的同一个问题列表，分别获取两个模型的回答。
# - 将回答并排打印，方便直观地对比微调带来的效果提升。
# ==============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import warnings

# 忽略一些无害的警告
warnings.filterwarnings("ignore")

# ================================
# 1. 配置参数 (请确保路径与您的训练配置一致)
# ================================
# 基础模型ID (必须与训练时使用的模型一致)
BASE_MODEL_ID = "/home/mw/input/models2179"
# LoRA适配器路径 (finetune_medical.py中final_model_path的路径)
LORA_ADAPTER_PATH = "/home/mw/input/qwen4b-medical-cot"

# ================================
# 2. 待测试的问题列表
# ================================
TEST_QUESTIONS = [
    "血热的临床表现是什么?",
    "帕金森叠加综合征的辅助治疗有些什么？",
    "卵巢癌肉瘤的影像学检查有些什么？",
]

# ================================
# 3. 模型加载与生成函数
# ================================

def load_models_and_tokenizer():
    """加载基础模型、微调后模型和分词器"""
    print("🔄 正在加载模型和LoRA适配器，此过程可能需要一些时间...")
    
    # --- 使用相同的量化配置 ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # --- 加载基础模型 ---
    print("...正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 加载微调后的模型 (基础模型 + LoRA适配器) ---
    print("...正在加载微调后的模型...")
    tuned_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    
    # 确保两个模型都处于评估模式
    base_model.eval()
    tuned_model.eval()
    
    print("✅ 基础模型和微调后模型均已加载完成！")
    return base_model, tuned_model, tokenizer


def get_model_response(model, tokenizer, question):
    """获取单个模型对问题的回答"""
    messages = [{"role": "user", "content": question}]
    input_tensor = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_tensor, 
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    return response

# ================================
# 4. 主执行流程
# ================================
def main():
    base_model, tuned_model, tokenizer = load_models_and_tokenizer()

    print("\n" + "="*80)
    print("🚀 开始进行模型回答对比测试...")
    print("="*80)

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"\n\n--- ❓ 问题 {i+1}: {question} ---\n")
        
        # --- 获取基础模型的回答 ---
        print("⏳ 正在获取 [基础模型] 的回答...")
        base_response = get_model_response(base_model, tokenizer, question)
        print("\n" + "-"*35 + " [基础模型回答] " + "-"*35)
        print(base_response)
        
        # --- 获取微调模型的回答 ---
        print("\n⏳ 正在获取 [微调后模型] 的回答...")
        tuned_response = get_model_response(tuned_model, tokenizer, question)
        print("\n" + "-"*35 + " [微调后模型回答] " + "-"*35)
        print(tuned_response)
        
        print("\n" + "="*80)
        
    print("\n🎉 所有问题对比完成！")

if __name__ == "__main__":
    main()