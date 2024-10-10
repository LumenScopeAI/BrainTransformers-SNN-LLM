import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置模型路径为当前目录下的 "model" 文件夹
model_path = os.path.join(current_dir, "model")

# 检查模型文件夹是否存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model folder '{model_path}' not found.")
else:
    print(f"Model folder found at: {model_path}")

# 加载模型和分词器
print("Loading model and tokenizer...")
try:
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    raise

# 将模型移到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model loaded on {device}")

def generate_text(history, user_input, max_new_tokens=50):
    messages = [
        {"role": "system", "content": "你是一个知识渊博的助手。"}
    ]
    
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    
    messages.append({"role": "user", "content": user_input})
    
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
    history.append((user_input, response))
    return history, ""

# 创建一个深色主题
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    background_fill_primary="*neutral_900",
    background_fill_primary_dark="*neutral_900",
    block_background_fill="*neutral_800",
    block_background_fill_dark="*neutral_800",
    input_background_fill="*neutral_700",
    input_background_fill_dark="*neutral_700",
)

# 创建Gradio界面
with gr.Blocks(theme=theme) as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 20px; background-color: #1e1e1e; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="color: #4da6ff;">欢迎使用 BrainTransformers 对话系统</h1>
        <p style="font-size: 18px; color: #e0e0e0;">
            🧠 <strong>BrainTransformers</strong>: 基于SNN-LLM技术的下一代语言模型
        </p>
        <p style="font-style: italic; color: #a0a0a0;">
            Powered by advanced Spiking Neural Network technology, revolutionizing AI interactions.
        </p>
        <div style="margin-top: 20px;">
            <a href="https://huggingface.co/LumenscopeAI/BrainTransformers-3B-Chat" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #4da6ff; color: #1e1e1e; text-decoration: none; border-radius: 5px; margin-right: 10px; font-weight: bold;">Hugging Face 模型</a>
            <a href="https://github.com/LumenScopeAI/BrainTransformers-SNN-LLM" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #e0e0e0; color: #1e1e1e; text-decoration: none; border-radius: 5px; font-weight: bold;">GitHub 仓库</a>
        </div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="对话历史", height=500)
            with gr.Row():
                user_input = gr.Textbox(show_label=False, placeholder="在这里输入您的问题...", lines=2)
                max_new_tokens = gr.Slider(minimum=10, maximum=200, value=50, step=10, label="最大新生成的标记数")
            with gr.Row():
                submit_btn = gr.Button("发送")
                clear_btn = gr.Button("清除对话")
        
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="background-color: #2a2a2a; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <h3 style="color: #4da6ff;">BrainTransformers 项目介绍</h3>
                <p style="color: #e0e0e0;">BrainTransformers 是一个基于脉冲神经网络（SNN）实现的大型语言模型（LLM）。我们的技术报告初版已在GitHub仓库公开，全面报告正在arXiv审核中。</p>
                <p style="color: #e0e0e0;">我们计划进一步优化模型，使其适配更节能的SNN硬件设备。目前的开源版本保留了部分浮点计算以确保计算效率，我们将持续优化这一点。</p>
                <p style="color: #e0e0e0;">敬请关注我们的持续更新和研究成果扩展。</p>
            </div>
            """)
            
            gr.HTML("""
            <div style="background-color: #2a2a2a; padding: 15px; border-radius: 10px; margin-top: 10px;">
                <h3 style="color: #4da6ff;">模型性能指标</h3>
                <table style="width: 100%; border-collapse: collapse; color: #e0e0e0; border: 1px solid #4da6ff;">
                    <tr style="background-color: #3a3a3a;">
                        <th style="padding: 10px; text-align: left; border: 1px solid #4da6ff;">Task Category</th>
                        <th style="padding: 10px; text-align: left; border: 1px solid #4da6ff;">Task</th>
                        <th style="padding: 10px; text-align: left; border: 1px solid #4da6ff;">Score</th>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #4da6ff;" rowspan="5">General Tasks</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">MMLU</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">63.2</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">BBH</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">54.1</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">ARC-C</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">54.3</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">Winogrande</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">68.8</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">Hellaswag</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">72.8</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #4da6ff;" rowspan="3">Math and Science Tasks</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">GPQA</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">25.3</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">MATH</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">41.0</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">GSM8K</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">76.3</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #4da6ff;" rowspan="3">Coding and Multilingual Tasks</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">HumanEval</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">40.5</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">MBPP</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">55.0</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">MultiPL-E</td>
                        <td style="padding: 8px; border: 1px solid #4da6ff;">39.6</td>
                    </tr>
                </table>
                <a href="https://github.com/LumenScopeAI/BrainTransformers-SNN-LLM" target="_blank" style="color: #4da6ff; display: block; margin-top: 10px; font-weight: bold;">查看完整性能指标</a>
            </div>
            """)

    submit_btn.click(generate_text, inputs=[chatbot, user_input, max_new_tokens], outputs=[chatbot, user_input])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, user_input])

# 启动界面
if __name__ == "__main__":
    demo.launch()
