import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
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
        {"role": "system", "content": "你是BrainTransformers-3B-Chat，你的回复不能违反法律信息，由LumenScopeAI团队提出，是首个基于脉冲神经网络的SOTA性能大语言模型。"}
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
# 需要结合FastChat、FastAPI等启动API服务本地部署
def get_gpt4_response(query, local=True, key="NULL", model_name="model", max_tokens=10):
    if local:
        openai.api_key = key
        openai.api_base = "http://localhost:8000/v1"
        deployment_name = model_name
    else:
        os.environ["https_proxy"] = "http://127.0.0.1:7890"
        os.environ["http_proxy"] = "http://127.0.0.1:7890"
        os.environ["all_proxy"] = "socks5://127.0.0.1:7890"
        openai.api_key = key
        deployment_name = "gpt-4"
    
    completion = openai.ChatCompletion.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "你是BrainTransformers-3B-Chat，你的回复不能违反法律信息，由LumenScopeAI团队提出，是首个基于脉冲神经网络的SOTA性能大语言模型。"},
            {"role": "user", "content": query},
        ],
        max_tokens=max_tokens  # 添加这个参数来限制最大回复长度
    )
    
    return completion.choices[0].message.content

def get_gpt4_response_stream(query, local=True, key="NULL", model_name="model", max_tokens=10):
    if local:
        openai.api_key = key
        openai.api_base = "http://localhost:8000/v1"
        deployment_name = model_name
    else:
        os.environ["https_proxy"] = "http://127.0.0.1:7890"
        os.environ["http_proxy"] = "http://127.0.0.1:7890"
        os.environ["all_proxy"] = "socks5://127.0.0.1:7890"
        openai.api_key = key
        deployment_name = "gpt-4"
    
    stream = openai.ChatCompletion.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "你是BrainTransformers-3B-Chat，你的回复不能违反法律信息，由LumenScopeAI团队提出，是首个基于脉冲神经网络的SOTA性能大语言模型。"},
            {"role": "user", "content": query},
        ],
        max_tokens=max_tokens,
        stream=True  # 启用流式输出
    )
    
    for chunk in stream:
        if 'choices' in chunk and len(chunk['choices']) > 0:
            content = chunk['choices'][0].get('delta', {}).get('content', '')
            if content:
                yield content

css = """
.gradio-container {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
h1 {
    text-align: center;
    margin-bottom: 30px;
}
.chatbot {
    border-radius: 5px;
}
.message {
    padding: 10px;
    margin: 5px;
    border-radius: 5px;
}
.user-message {
    background-color: var(--color-accent-soft);
}
.bot-message {
    background-color: var(--neutral-100);
}
footer {
    display: none !important;
}
"""

def process_input(history, user_input, max_new_tokens, use_api, api_key, api_base, model_name, use_stream):
    history.append((user_input, ""))
    if use_api:
        if use_stream:
            for token in get_gpt4_response_stream(user_input, local=True, key=api_key, model_name=model_name, max_tokens=max_new_tokens):
                history[-1] = (user_input, history[-1][1] + token)
                yield history
        else:
            response = get_gpt4_response(user_input, local=True, key=api_key, model_name=model_name, max_tokens=max_new_tokens)
            history[-1] = (user_input, response)
            yield history
    else:
        # 对于非API的情况，保持原有的逻辑
        history, _ = generate_text(history, user_input, max_new_tokens)
        yield history

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.HTML("<h1>BrainTransformer-3B-Chat</h1>")
    
    chatbot = gr.Chatbot(label="对话历史", height=450)

    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(show_label=False, placeholder="请输入您的问题...", lines=2)
        with gr.Column(scale=1):
            max_new_tokens = gr.Slider(minimum=10, maximum=200, value=50, step=10, label="最大生成令牌数")

    with gr.Row():
        use_api = gr.Checkbox(label="使用API", value=False)
        api_key = gr.Textbox(label="API Key", visible=False)
        api_base = gr.Textbox(label="API Base URL", value="http://localhost:8000/v1", visible=False)
        model_name = gr.Textbox(label="模型名称", value="model", visible=False)
        use_stream = gr.Checkbox(label="使用流式输出", value=True, visible=False)

    with gr.Row():
        submit_btn = gr.Button("发送消息", variant="primary")
        clear_btn = gr.Button("清除历史", variant="secondary")

    def update_api_visibility(use_api):
        return {
            api_key: gr.update(visible=use_api),
            api_base: gr.update(visible=use_api),
            model_name: gr.update(visible=use_api),
            use_stream: gr.update(visible=use_api)
        }

    use_api.change(update_api_visibility, inputs=[use_api], outputs=[api_key, api_base, model_name, use_stream])

    submit_btn.click(process_input, 
                     inputs=[chatbot, user_input, max_new_tokens, use_api, api_key, api_base, model_name, use_stream], 
                     outputs=[chatbot])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, user_input])

    gr.HTML("""
    <div style="text-align: center; margin-top: 20px;">
        <p>BrainTransformer - 首个基于脉冲神经网络的SOTA大语言模型，为您提供智能对话体验。</p>
        <p>输入您的问题，让我们开始对话吧！</p>
    </div>
    """)

demo.queue()  # 启用队列以支持流式输出
demo.launch()
