# BrainTransformers: SNN-LLM

Based on BrainTransformers, BrainGPTForCausalLM is a Large Language Model (LLM) implemented using Spiking Neural Networks (SNN). Our technical report will be uploaded to arXiv as soon as possible. We plan to further optimize the model at the operator level and adapt it for hardware compatibility, enabling BrainGPTForCausalLM to be deployed on more energy-efficient SNN hardware devices.

## Model Availability

- The current pre-trained model parameters have been published on ModelScope: [DataLinguistic/BrainTransformers-3B-Chat](https://www.modelscope.cn/models/DataLinguistic/BrainTransformers-3B-Chat)
- The current pre-trained model parameters have been published on Hugging Face.[LumenscopeAI/BrainTransformers-3B-Chat](https://huggingface.co/LumenscopeAI/BrainTransformers-3B-Chat)

## Repository

The github link is: [LumenScopeAI/BrainTransformers-SNN-LLM](https://github.com/LumenScopeAI/BrainTransformers-SNN-LLM)

## Model Performance

Below are the performance metrics of our 3B model on various benchmarks:

### General Tasks

| Dataset | Performance |
|---------|-------------|
| MMLU | 63.2 |
| MMLU-pro | 33.3 |
| MMLU-redux | 61.3 |
| BBH | 54.1 |
| ARC-C | 54.3 |
| Trurhfulqa | 47.1 |
| Winogrande | 68.8 |
| Hellaswag | 72.8 |

### Math and Science Tasks

| Dataset | Performance |
|---------|-------------|
| GPQA | 25.3 |
| Theoremqa | 26.4 |
| MATH | 41.0 |
| MMLU-stem | 60.2 |
| GSM8K | 76.3 |

### Coding and Multilingual Tasks

| Dataset | Performance |
|---------|-------------|
| HumanEval | 40.5 |
| HumanEval+ | 34.6 |
| MBPP | 55.0 |
| MBPP+ | 47.5 |
| MultiPL-E | 39.6 |
| Multi-Exam | 52.6 |
| Multi-Understanding | 73.9 |
| Multi-Mathematics | 47.1 |
| Multi-Translation | 28.2 |

## Usage

### Generate Text
```python
import torch
from transformers import AutoTokenizer, BrainGPTForCausalLM

model_path = "/path/to/your/model"
model = BrainGPTForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(messages, max_new_tokens=50):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Example usage
messages = [
    {"role": "system", "content": "You are a knowledgeable assistant."},
    {"role": "user", "content": "Explain the Pythagorean theorem."}
]
response = generate_text(messages)
print(response)
```
