# Fine-Tuning LLaMA3 and Falcon on Arabic-English Instruction Dataset using QLoRA
## Project Overview
This project involves fine-tuning LLaMA3 and Falcon models on both Arabic and English datasets using the QLoRA technique. The goal is to enhance the models' performance and behavior in multilingual contexts, improving the accuracy and effectiveness of language models across these two languages.

## Datasets
We utilized the following datasets from Hugging Face:

English Dataset: garage-bAInd/Open-Platypus
Arabic Dataset: akbargherbal/six_millions_instruction_dataset_for_arabic_llm_ft
## Data Preprocessing
To preprocess the data, we removed duplicates with the same meaning using the embedding model sentence-transformers/all-MiniLM-L6-v2. This model is suitable for handling multilingual data. We retained the longer sentence when duplicates were found, as it provides more guidance to the model for better training.

## Training Process
The training process was similar for both the Falcon model and the LLaMA3 model. We chose these models because they are open-source, trained on large corpora, and perform well on various benchmarks. Additionally, they are multilingual.

## Steps:
Quantization: We quantized the models using bitsandbytes to reduce memory usage.
Parameter-Efficient Fine-Tuning (PEFT): We utilized techniques such as LoRA to make fine-tuning faster and cheaper, reduce catastrophic forgetting, and minimize hallucinations.
Results
Both models showed similar improvements, but LLaMA3 provided slightly better responses.

## Usage
Here is the code to use the fine-tuned LLaMA3 model with Python:

```python
from transformers import pipeline

model = 'Mohamed-Sami/llama-3-8b-fine-tuned-ar-en'
pipe = pipeline('text-generation', model=model)

# Run text generation pipeline with our model
prompt = "What is a large language model?"
instruction = f"### Instruction:\n{prompt}\n\n### Response:\n"
result = pipe(instruction)
print(result[0]['generated_text'][len(instruction):])
```
## Conclusion
This project successfully fine-tuned LLaMA3 and Falcon models on Arabic and English datasets using QLoRA, resulting in improved multilingual language models. The use of quantization and PEFT techniques ensured efficient and cost-effective training while maintaining model performance.
