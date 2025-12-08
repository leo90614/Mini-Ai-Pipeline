# src/baseline.py

# 간단한 규칙 기반 감정 분류기
# 특정 단어가 포함되어 있으면 positive/negative로 판단함

import re
from typing import List

# 긍정/부정 단어 목록
POS_WORDS = {
    "good", "great", "excellent", "amazing", "awesome", "love", "like", "happy", "fantastic", "nice"
}

NEG_WORDS = {
    "bad", "terrible", "awful", "hate", "boring", "disappointed", "poor", "worst", "sad", "angry"
}


def _tokenize_basic(text: str) -> List[str]:
    # 소문자로 맞추고, 알파벳만 남김
    cleaned = text.lower()
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    return cleaned.split()


def predict_rule(text: str) -> str:
    # 텍스트를 토큰으로 나눠서 단어 개수 비교
    toks = _tokenize_basic(text)

    pos_cnt = sum(1 for w in toks if w in POS_WORDS)
    neg_cnt = sum(1 for w in toks if w in NEG_WORDS)

    # 단순한 규칙: 긍정 단어가 더 많으면 positive
    if pos_cnt > neg_cnt:
        return "positive"

    # 나머지는 negative로 처리 (tie 포함)
    return "negative"


def predict_batch(texts: List[str]) -> List[str]:
    # 여러 문장 한번에 처리
    return [predict_rule(t) for t in texts]


# 간단 테스트
if __name__ == "__main__":
    samples = [
        "I love this movie so much!",
        "This is the worst thing ever.",
        "It was good but also a bit disappointing."
    ]
    for s in samples:
        print(s, "->", predict_rule(s))
