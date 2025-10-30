---
layout: post
title: "오픈소스 LLM 파인튜닝 완벽 가이드: 실전 예제로 배우기"
date: 2024-02-11 14:00:00 +0900
categories: [ai, tutorials]
description: "오픈소스 LLM을 파인튜닝하는 방법을 처음부터 끝까지 상세하게 다룹니다. LoRA, QLoRA 등 최신 기법과 실전 예제 코드를 포함합니다."
---

오픈소스 LLM을 자신의 데이터로 파인튜닝하면 특정 도메인에서 더 나은 성능을 얻을 수 있습니다. 이 글에서는 파인튜닝의 기초부터 실전 구현까지 모든 과정을 상세히 다룹니다.

## LLM 파인튜닝이란?

파인튜닝(Fine-tuning)은 사전 학습된 대규모 언어 모델을 특정 작업이나 도메인에 맞게 추가 학습시키는 과정입니다.

### 파인튜닝이 필요한 경우

- **도메인 특화**: 의료, 법률, 금융 등 전문 분야
- **톤앤매너**: 브랜드에 맞는 응답 스타일
- **특수 작업**: 데이터 추출, 분류, 요약 등
- **언어 지원**: 특정 언어나 방언 지원 강화
- **비용 절감**: 큰 모델 대신 작은 모델 최적화

### 파인튜닝 vs 프롬프트 엔지니어링

| 특징 | 프롬프트 엔지니어링 | 파인튜닝 |
|------|-------------------|---------|
| 비용 | 낮음 | 중~높음 |
| 시간 | 즉시 | 수시간~수일 |
| 전문성 요구 | 낮음 | 높음 |
| 성능 개선 | 제한적 | 대폭 향상 가능 |
| 데이터 요구량 | 적음 | 많음 (수백~수천 샘플) |

## 파인튜닝 기법 비교

### 1. Full Fine-tuning
모델의 모든 파라미터를 업데이트하는 전통적인 방식입니다.

**장점:**
- 최고의 성능 달성 가능
- 모델 구조를 완전히 활용

**단점:**
- 막대한 GPU 메모리 필요 (예: LLaMA 7B = ~28GB)
- 긴 학습 시간
- 높은 비용

### 2. LoRA (Low-Rank Adaptation)
작은 어댑터 레이어만 학습하는 효율적인 방법입니다.

**장점:**
- GPU 메모리 사용량 대폭 감소 (~3배)
- 빠른 학습 속도
- 여러 LoRA 어댑터를 교체하며 사용 가능

**단점:**
- Full fine-tuning보다 약간 낮은 성능
- 추가적인 추론 오버헤드 (미미함)

### 3. QLoRA (Quantized LoRA)
LoRA에 양자화를 결합한 최신 기법입니다.

**장점:**
- 극도로 적은 메모리 사용 (~10배 감소)
- 소비자용 GPU로도 학습 가능
- LoRA와 유사한 성능

**단점:**
- 약간 느린 학습 속도
- 구현 복잡도 증가

## 환경 설정

### 필요한 하드웨어

| 모델 크기 | Full Fine-tuning | LoRA | QLoRA |
|----------|-----------------|------|-------|
| 7B | A100 40GB | RTX 3090 24GB | RTX 3060 12GB |
| 13B | A100 80GB | A100 40GB | RTX 3090 24GB |
| 70B | 8xA100 | 2xA100 | A100 40GB |

### Python 환경 구성

```bash
# 가상환경 생성
conda create -n llm-finetune python=3.10
conda activate llm-finetune

# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 필수 라이브러리 설치
pip install transformers==4.36.0
pip install datasets==2.16.0
pip install peft==0.7.0  # LoRA 지원
pip install bitsandbytes==0.41.3  # QLoRA 지원
pip install accelerate==0.25.0
pip install trl==0.7.10  # 강화학습 기반 파인튜닝

# 모니터링 도구
pip install wandb tensorboard
```

## 실전 예제 1: LoRA로 감정 분석 모델 만들기

### 데이터셋 준비

```python
# dataset_prep.py
import json
from datasets import Dataset

# 한국어 감정 분석 데이터 예시
data = [
    {
        "instruction": "다음 문장의 감정을 분석해주세요.",
        "input": "오늘 정말 기분이 좋아요!",
        "output": "긍정적 (positive) - 행복하고 즐거운 감정이 표현되어 있습니다."
    },
    {
        "instruction": "다음 문장의 감정을 분석해주세요.",
        "input": "이 영화는 정말 실망스러웠어요.",
        "output": "부정적 (negative) - 실망과 불만족이 드러납니다."
    },
    {
        "instruction": "다음 문장의 감정을 분석해주세요.",
        "input": "날씨가 흐리네요.",
        "output": "중립적 (neutral) - 객관적인 사실 진술로 특별한 감정이 없습니다."
    },
    # 실제로는 최소 500~1000개 이상 필요
]

# Alpaca 형식으로 변환
def format_instruction(sample):
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""

# 데이터셋 생성
dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.1)

# 저장
dataset.save_to_disk("./sentiment_dataset")
print(f"Training samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")
```

### LoRA 파인튜닝 코드

```python
# finetune_lora.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_from_disk

# 1. 모델과 토크나이저 로드
model_name = "beomi/llama-2-ko-7b"  # 한국어 LLaMA 모델
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 2. LoRA 설정
lora_config = LoraConfig(
    r=16,  # LoRA rank (낮을수록 메모리 절약, 높을수록 성능 향상)
    lora_alpha=32,  # LoRA scaling factor
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # LLaMA 모델의 attention과 MLP 레이어
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# LoRA 어댑터 적용
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 출력 예: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%

# 3. 데이터 준비
dataset = load_from_disk("./sentiment_dataset")

def format_instruction(sample):
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""

def preprocess_function(examples):
    # 텍스트 포맷팅
    texts = [format_instruction(ex) for ex in examples]
    
    # 토크나이징
    model_inputs = tokenizer(
        texts,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # labels 설정 (causal LM은 input_ids를 그대로 사용)
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    return model_inputs

# 데이터셋 전처리
tokenized_train = dataset["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

tokenized_test = dataset["test"].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["test"].column_names
)

# 4. 학습 설정
training_args = TrainingArguments(
    output_dir="./lora-sentiment-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # 실질적 배치 크기 = 4 * 4 = 16
    learning_rate=2e-4,
    fp16=True,  # Mixed precision 학습
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",  # 메모리 효율적인 옵티마이저
    report_to="tensorboard",
)

# 5. Trainer 설정 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 학습 시작
print("Starting training...")
trainer.train()

# 6. 모델 저장
model.save_pretrained("./lora-sentiment-final")
tokenizer.save_pretrained("./lora-sentiment-final")
print("Training complete! Model saved to ./lora-sentiment-final")
```

### 추론 및 테스트

```python
# inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 베이스 모델 로드
base_model_name = "beomi/llama-2-ko-7b"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# LoRA 어댑터 로드 및 병합
model = PeftModel.from_pretrained(base_model, "./lora-sentiment-final")
model = model.merge_and_unload()  # LoRA를 베이스 모델에 병합

# 추론 함수
def analyze_sentiment(text):
    prompt = f"""### Instruction:
다음 문장의 감정을 분석해주세요.

### Input:
{text}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Response 부분만 추출
    response = response.split("### Response:")[-1].strip()
    
    return response

# 테스트
test_sentences = [
    "이 제품 정말 최고예요! 강추합니다!",
    "배송이 너무 늦어서 짜증나네요.",
    "보통 수준이에요. 가격대비 괜찮습니다.",
]

for sentence in test_sentences:
    print(f"\n입력: {sentence}")
    print(f"분석: {analyze_sentiment(sentence)}")
```

## 실전 예제 2: QLoRA로 대화형 챗봇 만들기

QLoRA는 4bit 양자화를 사용하여 메모리를 극도로 절약하면서 파인튜닝합니다.

### QLoRA 파인튜닝 코드

```python
# finetune_qlora.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# 1. 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # 이중 양자화로 추가 메모리 절약
    bnb_4bit_quant_type="nf4",  # Normal Float 4bit
    bnb_4bit_compute_dtype=torch.bfloat16  # 계산은 bfloat16으로
)

# 2. 모델 로드 (4bit 양자화 적용)
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True  # HuggingFace 토큰 필요
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. k-bit 학습 준비
model = prepare_model_for_kbit_training(model)

# 4. LoRA 설정 (QLoRA는 LoRA + 양자화)
lora_config = LoraConfig(
    r=64,  # 더 큰 rank 사용 가능 (메모리 여유가 있으므로)
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 5. 대화 데이터셋 준비
# 고객 서비스 챗봇 예시
conversations = [
    {
        "messages": [
            {"role": "user", "content": "제품 배송은 얼마나 걸리나요?"},
            {"role": "assistant", "content": "일반 배송은 2-3일, 빠른 배송은 1일이 소요됩니다. 주말과 공휴일은 제외됩니다."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "환불 정책이 궁금합니다."},
            {"role": "assistant", "content": "구매 후 14일 이내 미개봉 제품은 전액 환불 가능합니다. 개봉 제품은 제품 상태에 따라 부분 환불됩니다."}
        ]
    },
    # 더 많은 대화 데이터...
]

# ChatML 형식으로 변환
def format_chat(example):
    formatted = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"<|user|>\n{content}\n"
        else:
            formatted += f"<|assistant|>\n{content}\n"
    formatted += "<|endoftext|>"
    return {"text": formatted}

from datasets import Dataset
dataset = Dataset.from_list(conversations)
dataset = dataset.map(format_chat)

# 6. SFTTrainer 설정 (Supervised Fine-Tuning)
training_args = TrainingArguments(
    output_dir="./qlora-chatbot-output",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # 4bit이므로 배치 크기 1로도 충분
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    bf16=True,  # bfloat16 사용
    logging_steps=5,
    save_strategy="epoch",
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512,
)

# 7. 학습 시작
trainer.train()

# 8. 저장
model.save_pretrained("./qlora-chatbot-final")
tokenizer.save_pretrained("./qlora-chatbot-final")
```

### QLoRA 모델 추론

```python
# inference_qlora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 베이스 모델 로드
base_model_name = "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(base_model, "./qlora-chatbot-final")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 대화 함수
def chat(user_message, conversation_history=""):
    prompt = conversation_history + f"<|user|>\n{user_message}\n<|assistant|>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = response.split("<|assistant|>")[-1].strip()
    
    return assistant_response

# 대화 테스트
print("챗봇: 안녕하세요! 무엇을 도와드릴까요?")
conversation = ""

while True:
    user_input = input("사용자: ")
    if user_input.lower() in ["quit", "exit", "종료"]:
        break
    
    response = chat(user_input, conversation)
    print(f"챗봇: {response}")
    
    # 대화 이력 업데이트
    conversation += f"<|user|>\n{user_input}\n<|assistant|>\n{response}\n"
```

## 실전 예제 3: 도메인 특화 모델 (의료 QA)

의료 분야처럼 전문적인 도메인에서 더 정확한 답변을 위한 파인튜닝 예시입니다.

### 의료 데이터셋 구성

```python
# medical_dataset.py
import json
from datasets import Dataset

# 의료 QA 데이터 예시
medical_data = [
    {
        "question": "당뇨병의 주요 증상은 무엇인가요?",
        "answer": """당뇨병의 주요 증상은 다음과 같습니다:

1. 다음(多飮) - 과도한 갈증
2. 다뇨(多尿) - 잦은 소변
3. 다식(多食) - 과도한 배고픔
4. 체중 감소
5. 피로감
6. 시력 저하
7. 상처 치유 지연

이러한 증상이 지속된다면 혈당 검사를 받아보시기 바랍니다."""
    },
    {
        "question": "고혈압 관리를 위한 생활습관은?",
        "answer": """고혈압 관리를 위한 주요 생활습관:

1. 식이요법
   - 나트륨 섭취 제한 (하루 2,000mg 이하)
   - DASH 식단 (과일, 채소, 저지방 유제품)
   - 포화지방 줄이기

2. 운동
   - 주 5일, 30분 이상 유산소 운동
   - 걷기, 수영, 자전거 타기 권장

3. 체중 관리
   - 정상 BMI(18.5-24.9) 유지
   
4. 금연 및 절주
5. 스트레스 관리
6. 규칙적인 혈압 측정"""
    },
    # 실제로는 수천 개의 의료 QA 필요
]

def format_medical_qa(sample):
    return {
        "text": f"""아래는 의료 관련 질문과 전문적인 답변입니다.

### 질문:
{sample['question']}

### 답변:
{sample['answer']}

"""
    }

dataset = Dataset.from_list(medical_data)
dataset = dataset.map(format_medical_qa)
dataset.save_to_disk("./medical_qa_dataset")
```

### 의료 모델 파인튜닝 (LoRA + 특수 설정)

```python
# finetune_medical.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_from_disk

# 모델 설정
model_name = "beomi/llama-2-ko-7b"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRA 설정 (의료 도메인에 최적화)
lora_config = LoraConfig(
    r=32,  # 전문 도메인이므로 높은 rank 사용
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# 데이터셋 로드
dataset = load_from_disk("./medical_qa_dataset")
train_test = dataset.train_test_split(test_size=0.1)

# 학습 설정 (의료 도메인 특화)
training_args = TrainingArguments(
    output_dir="./medical-lora-output",
    num_train_epochs=5,  # 전문 도메인은 더 많은 에폭
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,  # 낮은 학습률로 안정적 학습
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=5,
    load_best_model_at_end=True,
    warmup_ratio=0.1,  # 더 긴 워밍업
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    report_to="wandb",  # Weights & Biases 로깅
)

# SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=1024,  # 긴 의료 답변을 위해 길이 증가
)

# 학습
trainer.train()

# 저장
model.save_pretrained("./medical-model-final")
tokenizer.save_pretrained("./medical-model-final")
```

## 고급 기법

### 1. 데이터 증강 (Data Augmentation)

```python
# data_augmentation.py
from transformers import pipeline

# Back-translation으로 데이터 증강
translator_ko_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")
translator_en_ko = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ko")

def augment_with_backtranslation(text):
    # 한국어 -> 영어 -> 한국어
    en_text = translator_ko_en(text)[0]['translation_text']
    augmented = translator_en_ko(en_text)[0]['translation_text']
    return augmented

original = "당뇨병의 주요 증상은 무엇인가요?"
augmented = augment_with_backtranslation(original)
print(f"원본: {original}")
print(f"증강: {augmented}")

# Paraphrasing으로 데이터 증강
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
paraphrase_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

def paraphrase(text, num_return_sequences=3):
    inputs = paraphrase_tokenizer(f"paraphrase: {text}", return_tensors="pt")
    outputs = paraphrase_model.generate(
        **inputs,
        max_length=128,
        num_return_sequences=num_return_sequences,
        num_beams=5
    )
    return [paraphrase_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

paraphrases = paraphrase("당뇨병의 주요 증상은 무엇인가요?")
for i, p in enumerate(paraphrases, 1):
    print(f"변형 {i}: {p}")
```

### 2. 커스텀 Loss 함수

```python
# custom_loss.py
import torch
import torch.nn as nn
from transformers import Trainer

class WeightedLossTrainer(Trainer):
    """중요한 토큰에 더 높은 가중치를 부여하는 커스텀 Trainer"""
    
    def __init__(self, *args, important_token_ids=None, weight_multiplier=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.important_token_ids = important_token_ids or []
        self.weight_multiplier = weight_multiplier
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Cross Entropy Loss 계산
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # 중요한 토큰에 가중치 부여
        weights = torch.ones_like(shift_labels, dtype=torch.float)
        for token_id in self.important_token_ids:
            weights[shift_labels == token_id] *= self.weight_multiplier
        
        weighted_loss = (loss * weights.view(-1)).mean()
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss

# 사용 예시
important_tokens = tokenizer.encode("당뇨병 고혈압 증상 치료", add_special_tokens=False)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    important_token_ids=important_tokens,
    weight_multiplier=2.5
)
```

### 3. 멀티태스크 학습

```python
# multitask_learning.py
from datasets import concatenate_datasets, load_from_disk

# 여러 작업의 데이터셋 로드
sentiment_dataset = load_from_disk("./sentiment_dataset")
qa_dataset = load_from_disk("./medical_qa_dataset")
summarization_dataset = load_from_disk("./summarization_dataset")

# 작업 식별자 추가
def add_task_prefix(example, task_name):
    example["text"] = f"[TASK: {task_name}]\n{example['text']}"
    return example

sentiment_with_task = sentiment_dataset.map(
    lambda x: add_task_prefix(x, "SENTIMENT")
)
qa_with_task = qa_dataset.map(
    lambda x: add_task_prefix(x, "QA")
)
summarization_with_task = summarization_dataset.map(
    lambda x: add_task_prefix(x, "SUMMARIZATION")
)

# 데이터셋 결합
combined_dataset = concatenate_datasets([
    sentiment_with_task,
    qa_with_task,
    summarization_with_task
])

# 셔플
combined_dataset = combined_dataset.shuffle(seed=42)

# 이제 combined_dataset으로 파인튜닝
```

## 평가 및 벤치마킹

### 정량적 평가

```python
# evaluation.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np

# 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "beomi/llama-2-ko-7b",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./medical-model-final")
tokenizer = AutoTokenizer.from_pretrained("beomi/llama-2-ko-7b")

# 테스트 데이터 로드
test_dataset = load_from_disk("./medical_qa_dataset")["test"]

# Perplexity 계산
def calculate_perplexity(model, tokenizer, texts):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts):
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            outputs = model(**inputs, labels=inputs["input_ids"])
            
            loss = outputs.loss
            num_tokens = inputs["input_ids"].numel()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

texts = [example["text"] for example in test_dataset]
ppl = calculate_perplexity(model, tokenizer, texts)
print(f"Perplexity: {ppl:.2f}")

# BLEU, ROUGE 점수 계산
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

def evaluate_generation(model, tokenizer, questions, reference_answers):
    model.eval()
    bleu_scores = []
    rouge = Rouge()
    rouge_scores = []
    
    for question, reference in tqdm(zip(questions, reference_answers)):
        prompt = f"""### 질문:
{question}

### 답변:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated.split("### 답변:")[-1].strip()
        
        # BLEU 점수
        reference_tokens = reference.split()
        generated_tokens = generated.split()
        bleu = sentence_bleu([reference_tokens], generated_tokens)
        bleu_scores.append(bleu)
        
        # ROUGE 점수
        try:
            scores = rouge.get_scores(generated, reference)[0]
            rouge_scores.append(scores)
        except:
            continue
    
    avg_bleu = np.mean(bleu_scores)
    avg_rouge1 = np.mean([s["rouge-1"]["f"] for s in rouge_scores])
    avg_rouge2 = np.mean([s["rouge-2"]["f"] for s in rouge_scores])
    avg_rougeL = np.mean([s["rouge-l"]["f"] for s in rouge_scores])
    
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L: {avg_rougeL:.4f}")

questions = [ex["question"] for ex in test_dataset]
answers = [ex["answer"] for ex in test_dataset]
evaluate_generation(model, tokenizer, questions, answers)
```

### 정성적 평가

```python
# human_evaluation.py
import random

def create_evaluation_set(test_data, num_samples=50):
    """사람이 평가할 샘플 생성"""
    samples = random.sample(test_data, num_samples)
    
    results = []
    for i, sample in enumerate(samples):
        question = sample["question"]
        reference = sample["answer"]
        generated = generate_answer(model, tokenizer, question)
        
        results.append({
            "id": i,
            "question": question,
            "reference": reference,
            "generated": generated,
            "rating": None,  # 1-5점 척도
            "comments": ""
        })
    
    return results

# CSV로 저장하여 사람이 평가
import pandas as pd
eval_set = create_evaluation_set(test_dataset, num_samples=50)
df = pd.DataFrame(eval_set)
df.to_csv("human_evaluation.csv", index=False)
print("human_evaluation.csv 파일을 열어서 rating과 comments를 작성해주세요.")
```

## 모델 배포

### 1. HuggingFace Hub에 업로드

```python
# upload_to_hub.py
from huggingface_hub import HfApi, create_repo

# 로그인 (토큰 필요)
from huggingface_hub import login
login(token="your_huggingface_token")

# 저장소 생성
repo_name = "my-finetuned-medical-llm"
create_repo(repo_name, private=True)

# 모델 업로드
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

# README 작성
readme = """---
language: ko
license: llama2
tags:
- medical
- healthcare
- korean
---

# Korean Medical LLM

이 모델은 의료 QA 데이터셋으로 파인튜닝된 한국어 LLaMA 모델입니다.

## 사용 예시

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("beomi/llama-2-ko-7b")
model = PeftModel.from_pretrained(model, "your-username/my-finetuned-medical-llm")
tokenizer = AutoTokenizer.from_pretrained("your-username/my-finetuned-medical-llm")

# 추론 코드...
```

## 성능

- Perplexity: 15.2
- BLEU: 0.42
- ROUGE-L: 0.58

## 주의사항

이 모델은 교육 목적으로만 사용되어야 하며, 실제 의료 진단에 사용해서는 안 됩니다.
"""

with open("README.md", "w") as f:
    f.write(readme)

api = HfApi()
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=f"your-username/{repo_name}"
)
```

### 2. FastAPI로 서빙

```python
# serve_model.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn

app = FastAPI(title="Medical LLM API")

# 모델 로드 (시작 시 한 번만)
print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "beomi/llama-2-ko-7b",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./medical-model-final")
tokenizer = AutoTokenizer.from_pretrained("beomi/llama-2-ko-7b")
model.eval()
print("Model loaded!")

class QueryRequest(BaseModel):
    question: str
    max_tokens: int = 200
    temperature: float = 0.7

class QueryResponse(BaseModel):
    answer: str
    tokens_used: int

@app.post("/generate", response_model=QueryResponse)
async def generate_answer(request: QueryRequest):
    try:
        prompt = f"""### 질문:
{request.question}

### 답변:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("### 답변:")[-1].strip()
        
        return QueryResponse(
            answer=answer,
            tokens_used=len(outputs[0])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Docker 컨테이너화

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 모델과 코드 복사
COPY ./medical-model-final ./medical-model-final
COPY serve_model.py .

# 포트 노출
EXPOSE 8000

# 실행
CMD ["python3", "serve_model.py"]
```

```bash
# Docker 빌드 및 실행
docker build -t medical-llm-api .
docker run --gpus all -p 8000:8000 medical-llm-api

# 테스트
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "당뇨병의 증상은 무엇인가요?"}'
```

## 트러블슈팅

### 1. CUDA Out of Memory 오류

```python
# 메모리 최적화 팁

# 1) Gradient Checkpointing 활성화
model.gradient_checkpointing_enable()

# 2) 배치 크기 줄이기
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # 최소값
    gradient_accumulation_steps=32,  # 실질적 배치 크기 유지
)

# 3) Mixed Precision 사용
training_args = TrainingArguments(
    fp16=True,  # 또는 bf16=True
)

# 4) Optimizer 상태 오프로딩
training_args = TrainingArguments(
    optim="adafactor",  # 메모리 효율적인 옵티마이저
)

# 5) 시퀀스 길이 줄이기
training_args = TrainingArguments(
    max_seq_length=512,  # 1024 대신
)
```

### 2. 학습이 불안정할 때

```python
# 안정적인 학습을 위한 설정

training_args = TrainingArguments(
    learning_rate=1e-5,  # 더 낮은 학습률
    warmup_ratio=0.1,  # 워밍업 증가
    weight_decay=0.01,  # 정규화
    max_grad_norm=1.0,  # Gradient Clipping
    lr_scheduler_type="cosine",  # 부드러운 학습률 감소
)

# LoRA alpha 조정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,  # alpha = 2 * r (일반적 규칙)
)
```

### 3. 과적합 방지

```python
# 과적합 방지 기법

# 1) 데이터 증강
from nlpaug.augmenter.word import SynonymAug
aug = SynonymAug()
augmented_text = aug.augment(original_text)

# 2) Dropout 증가
lora_config = LoraConfig(
    lora_dropout=0.1,  # 기본값 0.05 대신
)

# 3) Early Stopping
from transformers import EarlyStoppingCallback

trainer = Trainer(
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 4) 정규화
training_args = TrainingArguments(
    weight_decay=0.01,
)
```

### 4. 생성 품질이 낮을 때

```python
# 생성 품질 개선

# 1) 다양한 디코딩 전략 시도
outputs = model.generate(
    **inputs,
    # Beam Search
    num_beams=5,
    no_repeat_ngram_size=3,
    
    # Top-k + Top-p Sampling
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    
    # Repetition Penalty
    repetition_penalty=1.2,
    
    # Length Penalty
    length_penalty=1.0,
)

# 2) 프롬프트 엔지니어링
prompt = f"""당신은 전문 의료 AI입니다. 정확하고 상세하게 답변해주세요.

### 질문:
{question}

### 답변:
"""

# 3) Few-shot 예시 추가
prompt = f"""아래는 의료 질문에 대한 전문적인 답변 예시입니다.

질문: 고혈압의 정상 수치는?
답변: 정상 혈압은 수축기 120mmHg 미만, 이완기 80mmHg 미만입니다.

질문: {question}
답변:"""
```

## 성능 비교 및 선택 가이드

### 모델 크기별 추천

| 모델 크기 | 추천 용도 | 필요 GPU | 파인튜닝 방법 |
|----------|----------|---------|-------------|
| 1B-3B | 간단한 분류, 챗봇 | GTX 1080 Ti | Full / LoRA |
| 7B | 일반적인 QA, 생성 | RTX 3090 | LoRA / QLoRA |
| 13B | 전문 도메인, 복잡한 추론 | A100 40GB | QLoRA |
| 30B-70B | 최고 품질 필요 시 | 2x A100 | QLoRA |

### 데이터셋 크기별 가이드

- **100-500 샘플**: Few-shot prompting 고려
- **500-2,000 샘플**: LoRA 파인튜닝 시작
- **2,000-10,000 샘플**: Full LoRA 효과
- **10,000+ 샘플**: Full fine-tuning 고려

## 실전 체크리스트

### 파인튜닝 시작 전

- [ ] 명확한 목표 설정 (작업, 성능 지표)
- [ ] 충분한 데이터 확보 (최소 500개)
- [ ] 데이터 품질 검증 (오타, 포맷 일관성)
- [ ] 베이스 모델 선택 (언어, 크기 고려)
- [ ] GPU 리소스 확인

### 학습 중

- [ ] 학습 곡선 모니터링 (Loss, Perplexity)
- [ ] 주기적으로 샘플 생성 테스트
- [ ] GPU 메모리 사용량 체크
- [ ] 과적합 징후 확인 (Train/Val loss 차이)

### 학습 후

- [ ] 정량적 평가 (BLEU, ROUGE, Perplexity)
- [ ] 정성적 평가 (사람의 판단)
- [ ] 다양한 엣지 케이스 테스트
- [ ] 베이스 모델과 비교
- [ ] 배포 전 안전성 검증

## 추가 리소스

### 유용한 오픈소스 도구

- **Axolotl**: 파인튜닝 자동화 프레임워크
- **LM Studio**: GUI 기반 파인튜닝 툴
- **Weights & Biases**: 실험 추적 및 시각화
- **vLLM**: 빠른 추론 서버

### 추천 학습 자료

- HuggingFace Transformers 공식 문서
- PEFT (Parameter-Efficient Fine-Tuning) 라이브러리
- QLoRA 논문: "QLoRA: Efficient Finetuning of Quantized LLMs"
- LoRA 논문: "LoRA: Low-Rank Adaptation of Large Language Models"

### 데이터셋 소스

- **HuggingFace Datasets**: 수천 개의 공개 데이터셋
- **AI Hub (한국어)**: 한국어 NLP 데이터셋
- **OpenOrca**: 고품질 instruction 데이터
- **Alpaca**: Instruction following 데이터

## 마무리

오픈소스 LLM 파인튜닝은 처음에는 복잡해 보이지만, 적절한 도구와 기법을 사용하면 비교적 적은 리소스로도 훌륭한 결과를 얻을 수 있습니다.

핵심 포인트:
1. **목적에 맞는 방법 선택**: Full fine-tuning vs LoRA vs QLoRA
2. **품질 좋은 데이터**: 양보다 질이 중요
3. **체계적인 평가**: 정량적 + 정성적 평가 병행
4. **반복적 개선**: 실험과 분석을 통한 지속적 개선

이 가이드의 예제 코드를 기반으로 자신만의 특화된 LLM을 만들어보세요. 질문이나 문제가 있다면 HuggingFace 포럼이나 GitHub Issues를 활용하시기 바랍니다.

다음 글에서는 파인튜닝된 모델을 프로덕션 환경에 배포하고 모니터링하는 방법을 다루겠습니다.

