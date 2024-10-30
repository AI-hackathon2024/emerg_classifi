import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file

# FastAPI app 생성
app = FastAPI()

# 모델 디렉토리 경로
model_dir = '/home/ubuntu/work_space/emerg_classifi/saved_model'

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, state_dict=None)
state_dict = load_file(f"{model_dir}/model.safetensors")
model.load_state_dict(state_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

class TextInput(BaseModel):
    text: str

@app.post("/predict/")
async def predict(input: TextInput):
    # 예측 수행
    text = input.text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    return {"predicted_class": predictions.item()}

# 애플리케이션 실행 (command line에서 uvicorn으로 실행)
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload
