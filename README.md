# BrainTransformers: SNN-LLM

Based on BrainTransformers, BrainGPTForCausalLM is a Large Language Model (LLM) implemented using Spiking Neural Networks (SNN). Our technical report will be uploaded to arXiv as soon as possible. We plan to further optimize the model at the operator level and adapt it for hardware compatibility, enabling BrainGPTForCausalLM to be deployed on more energy-efficient SNN hardware devices.

## Model Availability

- The current pre-trained model parameters have been published on ModelScope: [DataLinguistic/BrainTransformers-3B-Chat](https://www.modelscope.cn/models/DataLinguistic/BrainTransformers-3B-Chat)
- The current pre-trained model parameters have been published on Hugging Face.[LumenscopeAI/BrainTransformers-3B-Chat](https://huggingface.co/LumenscopeAI/BrainTransformers-3B-Chat)

## Repository

The github link is: [LumenScopeAI/BrainTransformers-SNN-LLM](https://github.com/LumenScopeAI/BrainTransformers-SNN-LLM)

## Usage

### Generate Text
```
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


---
license: apache-2.0
---
