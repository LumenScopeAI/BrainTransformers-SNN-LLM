import torch
from transformers import AutoTokenizer, BrainGPTForCausalLM
from tqdm import tqdm
from torch.utils.data import DataLoader

# 定义模型和分词器的路径
model_path = "/media/tangshi/AI001/test_ANN2SNN/model"

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

# 打印模型中的Synapsis层
def print_synapsis_layers(model):
    print("\nSynapsis layers in the model:")
    count = 0
    for name, module in model.named_modules():
        if type(module).__name__ == 'Synapsis':
            print(f"Found Synapsis layer: {name}")
            count += 1
    print(f"Total Synapsis layers found: {count}")

# 启用 Synapsis 模块中的 STDP 训练
def enable_stdp_training(model):
    count = 0
    for module in model.modules():
        if type(module).__name__ == 'Synapsis':
            if hasattr(module, 'use_stdp'):
                module.use_stdp = True
                count += 1
    print(f"Enabled STDP training for {count} Synapsis layers")

# 禁用 Synapsis 模块中的 STDP 训练
def disable_stdp_training(model):
    count = 0
    for module in model.modules():
        if type(module).__name__ == 'Synapsis':
            if hasattr(module, 'use_stdp'):
                module.use_stdp = False
                count += 1
    print(f"Disabled STDP training for {count} Synapsis layers")

# 重置神经元的状态
def reset_neuron_states(model):
    count = 0
    for module in model.modules():
        if type(module).__name__ == 'Synapsis':
            if hasattr(module, 'reset'):
                module.reset()
                count += 1
    print(f"Reset states for {count} Synapsis layers")

# 准备测试提示
test_prompts = [
    "请解释勾股定理！",
    "什么是人工智能？",
    "请写一首关于春天的诗。",
    "解释量子计算的基本原理。",
    "如何制作一个简单的披萨？"
]

# 准备无监督训练数据
training_data = [
    "这是一个训练样本。",
    "另一个示例文本用于训练。",
    "春天到了，花儿开了。",
    "机器学习和人工智能是热门领域。",
    "今天天气真好，我们去郊游吧。"
]

# 创建数据加载器
data_loader = DataLoader(training_data, batch_size=2, shuffle=True)

# 主实验流程
print_synapsis_layers(model)

print("\nTesting text generation before STDP training:")
test_generation(test_prompts)

def stdp_train(model, data_loader, num_epochs):
    print("\n开始 STDP 训练...")
    for epoch in range(num_epochs):
        print(f"\n开始第 {epoch+1} 轮训练...")
        total_loss = 0
        
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
            # 启用 STDP 训练
            enable_stdp_training(model)
            
            # 对每个样本进行处理
            for sample in batch:
                messages = [
                    {"role": "system", "content": "你是一个知识渊博的助手。"},
                    {"role": "user", "content": sample}
                ]
                
                # 使用当前样本生成文本，触发神经元活动
                _ = generate_text(messages)
            
            # STDP 更新和损失计算
            batch_loss = 0
            for module in model.modules():
                if type(module).__name__ == 'Synapsis' and hasattr(module, 'stdp_update'):
                    batch_loss += module.stdp_update()
            
            total_loss += batch_loss
            print(f"Batch STDP loss: {batch_loss:.4f}")
            
            # 禁用 STDP 训练
            disable_stdp_training(model)
            
            # 重置神经元状态
            reset_neuron_states(model)
        
        print(f"第 {epoch+1} 轮平均损失: {total_loss / len(data_loader):.4f}")

    print("\nSTDP 训练完成")

# 执行 STDP 训练
num_epochs = 3
stdp_train(model, data_loader, num_epochs)

print("\n测试 STDP 训练后的文本生成:")
test_generation(test_prompts)