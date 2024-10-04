# BrainTransformers: SNN-LLM

Based on BrainTransformers, BrainGPTForCausalLM is a Large Language Model (LLM) implemented using Spiking Neural Networks (SNN). Our technical report has been submitted to arXiv and is currently in the "on hold" status, pending review. It will be available for public access as soon as the review process is completed. We plan to further optimize the model at the operator level and adapt it for hardware compatibility, enabling BrainGPTForCausalLM to be deployed on more energy-efficient SNN hardware devices.

The current open-source version retains some floating-point calculations to ensure computational efficiency. We will continue to optimize this. Some detailed explanations are provided in the comments within the source code.

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

## Acknowledgments

The model was trained using ANN-Base-Qwen2, with a total of three training stages, including SNN-specific neuron synaptic plasticity training. The technical report is still being prepared. Please note that SNN models do not support ANN fine-tuning techniques. We are currently developing specialized fine-tuning code tools for SNN models. Our open-source model has achieved leading SOTA results, and we welcome your stars.

This repository includes a complete transformers package, which can directly replace the transformers package in your development environment. This allows compatibility with our SNN-Base-LLM without affecting existing usage.
