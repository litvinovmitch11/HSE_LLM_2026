import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

class MLModel:
    def __init__(self, model_path="./my_finance_llm", base_model_id="Qwen/Qwen2.5-1.5B"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загружаем токенизатор из нашей папки
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
        # 1. Загружаем базовую модель (можно в 4-bit для экономии памяти на проде)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_id,
            num_labels=2,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # 2. Накатываем наши обученные LoRA веса
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> str:
        processed_text = text[:500] 
        
        inputs = self.tokenizer(
            processed_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item()

        if pred_class == 1:
            result = f"Ожидается сильное движение (Уверенность: {confidence:.1%})"
        else:
            result = f"Новость пройдет незаметно (Уверенность: {confidence:.1%})"
            
        return result

model = MLModel()
