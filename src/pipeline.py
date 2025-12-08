# src/pipeline.py

# 감정 분류용 inference 클래스
# 학습은 하지 않고 pretrained 모델을 불러와서 예측만 진행함

from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentClassifier:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        preferred_device: Optional[str] = None,
    ):
        # tokenizer / model 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # device 선택 로직
        if preferred_device:
            self.device = preferred_device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.model.eval()  # inference 모드로 설정

        # 모델 출력 id → 문자열 label
        self.label_map = {
            0: "negative",
            1: "positive",
        }

    def _prepare_inputs(self, texts: List[str]):
        # tokenizer로 텍스트를 모델 입력 형식으로 변환
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        # tensor들을 모두 device로 이동
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def predict_one(self, text: str) -> str:
        # 한 문장만 예측할 때 사용
        model_in = self._prepare_inputs([text])
        out = self.model(**model_in)
        logits = out.logits.squeeze(0)

        cls_id = int(torch.argmax(logits).item())
        return self.label_map[cls_id]

    @torch.no_grad()
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[str]:
        # 여러 문장을 batch 단위로 처리
        preds: List[str] = []

        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            model_in = self._prepare_inputs(chunk)
            out = self.model(**model_in)

            logits = out.logits
            cls_ids = torch.argmax(logits, dim=1).cpu().tolist()

            preds.extend(self.label_map[c] for c in cls_ids)

        return preds


# 간단 테스트
if __name__ == "__main__":
    clf = SentimentClassifier()

    samples = [
        "I absolutely loved this!",
        "This was extremely disappointing.",
    ]

    for s in samples:
        print(s, "->", clf.predict_one(s))
