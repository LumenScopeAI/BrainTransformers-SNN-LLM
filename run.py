import torch
from transformers import AutoTokenizer, BrainGPTForCausalLM
from tqdm import tqdm
from torch.utils.data import DataLoader

# 定义模型和分词器的路径
model_path = "/path/to/your/model"

# 加载模型和分词器
print("Loading model and tokenizer...")
model = BrainGPTForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 将模型移到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model loaded on {device}")

# 定义一个函数来生成文本
def generate_text(messages, max_new_tokens=50):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 测试生成函数
def test_generation(test_prompts):
    print("\nTesting text generation:")
    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": "你是一个知识渊博的助手。"},
            {"role": "user", "content": prompt}
        ]
        print(f"\nPrompt: {prompt}")
        generated = generate_text(messages)
        print(f"Generated: {generated}")

test_prompts = [
    "请解释勾股定理！",
    "什么是人工智能？",
    "请写一首关于春天的诗。",
    "解释量子计算的基本原理。",
    "如何制作一个简单的披萨？"
]

print("\nTesting text generation before STDP training:")
test_generation(test_prompts)