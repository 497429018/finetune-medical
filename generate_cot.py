# ==============================================================================
# 🏥 医疗CoT数据生成脚本 (Notebook展示版)
# 功能:
# - 以单线程、逐条处理的简单模式运行，方便调试和观察。
# - 默认只处理10条数据用于快速验证。
# - 输出详细、分阶段的日志，完美复现您提供的日志格式，适合在Notebook中展示。
# - 在一个脚本内完成数据生成、分析、转换和保存的全过程。
# ==============================================================================

import json
import os
import ray
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# 导入Hugging Face和distilabel相关库
from datasets import load_dataset, Dataset, DatasetDict
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration

# ================================
# 1. 核心函数定义
# ================================

access_key = "sk-792*********************"

def cleanup_ray():
    """安全地关闭并清理Ray环境。"""
    try:
        if ray.is_initialized():
            print("🧹 正在清理Ray环境...")
            ray.shutdown()
            print("✅ Ray环境已清理")
    except Exception as e:
        print(f"⚠️ 清理Ray环境时出现错误: {e}")

def load_local_dataset(data_path: str, file_pattern: str) -> Optional[Dataset]:
    """从本地路径加载数据集。"""
    print("📖 加载数据集...")
    path = Path(data_path)
    if not path.exists():
        print(f"❌ 数据路径不存在: {path}")
        return None

    if path.is_dir():
        print(f"🔍 在目录 {path} 中查找符合模式 '{file_pattern}' 的文件...")
        files = list(path.glob(f"{file_pattern}.json")) + list(path.glob(f"{file_pattern}.jsonl"))
        if not files:
            print(f"❌ 在目录 {path} 中未找到符合模式 '{file_pattern}' 的JSON文件。")
            return None
        
        print(f"📁 找到 {len(files)} 个JSON文件:")
        for f in files:
            print(f"  - {f.name}")
        file_paths = [str(f) for f in files]
        return load_dataset("json", data_files=file_paths, split="train")
    else:
        return load_dataset("json", data_files=str(path), split="train")

def build_distilabel_pipeline(config: dict) -> Pipeline:
    """构建Distilabel生成管道。"""
    print("\n🔧 构建生成管道...")
    generation_kwargs = {
        "max_new_tokens": config['max_new_tokens'],
        "temperature": config['temperature'],
        "top_p": config['top_p'],
    }
    
    print(f"🤖 使用模型: {config['model']}")
    if "reasoner" in config['model'].lower():
        print("🧠 DeepSeek Reasoner模型，推理功能将自动启用")
    print(f"📝 使用模板: {config['prompt_template']}")
    print(f"⚙️ 生成参数: {generation_kwargs}")

    with Pipeline(name="medical-cot-generation-simple").ray() as pipeline:
        llm = OpenAILLM(
            base_url="https://api.deepseek.com/v1",
            api_key=access_key,
            model=config['model'],
            generation_kwargs=generation_kwargs,
        )
        TextGeneration(
            name="text_generation",
            llm=llm,
            template=config['prompt_template'],
            input_batch_size=1, # 逐条处理
            resources=StepResources(replicas=1), # 单进程
        )
    print("✅ 管道构建成功!")
    return pipeline

def run_generation_pipeline(config: dict) -> Optional[DatasetDict]:
    """运行完整的生成管道。"""
    cleanup_ray()
    
    print("\n🚀 开始运行医疗CoT数据生成管道")
    print("="*50)
    print("📋 配置参数:")
    for key, value in config.items():
        print(f"  模型: {config['model']}")
    print("="*50)

    dataset = load_local_dataset(config['data_path'], config['file_pattern'])
    if dataset is None:
        return None
        
    print(f"✅ 数据集加载成功! 包含 {len(dataset)} 条记录")
    print(f"📊 数据集字段: {dataset.column_names}")
    
    print("\n📋 数据样例:")
    for i, example in enumerate(dataset.select(range(min(3, len(dataset))))):
        print(f"  样例 {i+1}: {str(example['instruction'])[:30]}...")
    
    # 采样10条数据进行测试
    num_to_process=3
    if len(dataset) > num_to_process:
        print(f"\n⚠️ 注意：将只处理前3条数据用于简单测试。")
        dataset = dataset.select(range(num_to_process))

    pipeline = build_distilabel_pipeline(config)
    
    print("\n🎯 开始生成过程...")
    print("⏳ 这可能需要几分钟时间，请耐心等待...")
    
    distiset = pipeline.run(dataset=dataset, use_cache=False)
    
    print("🎉 生成完成!")
    return distiset

def analyze_and_convert_results(distiset: Optional[DatasetDict], output_dir: str):
    """分析生成结果并转换为Alpaca格式，打印详细日志。"""
    print("\n" + "="*50)
    print("📊 第二步: 分析和转换结果")
    print("="*50)
    
    if distiset is None:
        print("❌ 没有可分析的结果")
        return None

    try:
        train_data = distiset['default']['train']
        print(f"✅ 成功提取训练数据")
        print(f"📈 数据条数: {len(train_data)}")
        print(f"🏷️ 字段名称: {train_data.column_names}")
    except (KeyError, TypeError) as e:
        print(f"❌ 提取训练数据失败: {e}")
        return None

    print("\n📋 详细数据分析:")
    cot_keywords = ['思考', '分析', '首先', '其次', '因此', '推理', '步骤', '根据', '基于']
    
    for i, item in enumerate(train_data):
        print(f"\n--- 第 {i+1} 条数据 ---")
        print(f"📝 指令: {item['instruction']}")
        print(f"📥 输入: {item.get('input', '(空)')}")
        print(f"📤 原始输出: {str(item['output'])[:100]}...")
        print(f"🤖 生成内容: {str(item['generation'])[:150]}...")
        print(f"🏷️ 模型: {item['model_name']}")
        has_cot = any(keyword in item['generation'] for keyword in cot_keywords)
        print(f"🧠 包含推理过程: {'✅ 是' if has_cot else '❌ 否'}")
        print(f"📏 生成长度: {len(item['generation'])} 字符")
        
    print("\n🔄 转换为Alpaca格式...")
    alpaca_data = [
        {"instruction": item["instruction"], "input": item.get("input", ""), "output": item["generation"]}
        for item in train_data
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = "v1"
    alpaca_file = Path(output_dir) / f"medical_cot_alpaca-{timestamp}.json"
    
    with open(alpaca_file, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print("✅ 数据保存完成!")
    print(f"📁 Alpaca格式: {alpaca_file}")

    print("\n📊 质量统计:")
    total_count = len(alpaca_data)
    cot_count = sum(1 for item in alpaca_data if any(keyword in item['output'] for keyword in cot_keywords))
    avg_length = sum(len(item['output']) for item in alpaca_data) / total_count if total_count > 0 else 0
    
    print(f"📈 总数据条数: {total_count}")
    print(f"🧠 包含推理的数据: {cot_count}")
    print(f"📊 推理覆盖率: {cot_count/total_count*100:.1f}%")
    print(f"📏 平均生成长度: {avg_length:.0f} 字符")

    print("\n📋 Alpaca格式样例:")
    for i, item in enumerate(alpaca_data[:2]):
        print(f"\n样例 {i+1}:")
        print(f"instruction: {item['instruction']}")
        print(f"input: {item['input']}")
        print(f"output: {item['output'][:300]}...")
        print("-" * 40)
        
    return alpaca_data

# ================================
# 2. 主执行流程
# ================================
def main():
    """主执行函数"""
    print("本项目来源于和鲸社区，使用转载需要标注来源")
    print("\n🏥 医疗CoT数据生成系统")
    print("=" * 60)
    
    # 检查API密钥
    # if not os.getenv("DEEPSEEK_API_KEY"):
    #     print("❌ 错误: 请先设置环境变量 'DEEPSEEK_API_KEY'")
    #     return
    #access_key = os.getenv("DEEPSEEK_API_KEY")
    #access_key = "sk-792eddb3cdd9480cbe8d0a420c632261"


    config = {
        'model': 'deepseek-reasoner',
        'data_path': 'raw',
        'file_pattern': 'train*',
        'prompt_template': '{{ instruction }}',
        'temperature': 0.1,
        'top_p': 0.8,
        'max_new_tokens': 4096,
        'output_dir': 'output'
    }
    
    print("⚙️ 当前配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # --- 步骤1: 运行生成管道 ---
    print("\n🚀 第一步: 运行生成管道")
    distiset_result = run_generation_pipeline(config)
    
    # --- 步骤2: 分析和转换结果 ---
    if distiset_result:
        analyze_and_convert_results(distiset_result, config['output_dir'])
    else:
        print("❌ 生成失败，无法进行分析。")

    # --- 步骤3: 最终总结 ---
    print("\n" + "="*60)
    print("🎉 任务流程执行完毕!")
    print("="*60)
    
    cleanup_ray()

if __name__ == "__main__":
    main()