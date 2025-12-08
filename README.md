# Mini-AI-Pipeline: Simple Sentiment Classification Project

This project builds a small end-to-end AI pipeline that classifies short English sentences into **positive** or **negative** sentiment.  

---

## 1. Task Definition

### What are we trying to do?
We want a system that reads a short sentence and predicts whether the sentiment is **positive** or **negative**.

### Why is this interesting or useful?
People write countless reviews, comments, and reactions every day.  
No one can manually read everything. A sentiment classifier can quickly show how people feel about something.

Even though this task is simple, it nicely shows the full AI workflow:  
**data → baseline → model → evaluation → reflection**

### Input → Output
- **Input:** one short English sentence  
- **Output:** `"positive"` or `"negative"`

### Success Criteria
- The AI model should outperform a basic rule-based baseline.  
- Accuracy and F1-score should clearly improve.  
- Predictions should also make sense when we manually check examples.

---

## 2. Dataset

I created a small dataset of **300 sentences** (150 positive, 150 negative).  
Example format:

text,label
The food was amazing,positive
The app keeps crashing,negative


The file is here:
data/sentiment_data.csv


---

## 3. Baseline System (Rule-based)

Before using a real model, we need a simple baseline.

My baseline works like this:

- I made two tiny lists: positive words and negative words.
- If a sentence contains more positive words → predict **positive**
- Otherwise → predict **negative**

This method is extremely limited, but it shows why we need a real model.

File:
src/baseline.py



---

## 4. AI Pipeline (Transformer Model)

For the actual model, I used a lightweight pretrained Transformer:
textattack/distilbert-base-uncased-SST-2


I did **not** train a new model.  
Instead, I wrapped this pretrained model in a small class:
SentimentClassifier


This class handles:
- Tokenization  
- Batch inference  
- Device selection (CPU/GPU)  
- Converting logits → labels  

File:
src/pipeline.py



---

## 5. Evaluation

Metrics used:
- Accuracy  
- Precision  
- Recall  
- F1-score  

Why?  
Because sentiment classification is a binary task, and these metrics clearly show how much the model improves over the baseline.

---

## 6. Results

### Baseline (Rule-based)
- Very low recall  
- Predicts **negative** most of the time unless it sees obvious positive keywords  
- Misses many positive examples  

### AI Model (Transformer)
- Much higher accuracy and F1-score  
- Understands a wide variety of expressions  
- Handles subtle positive wording that baseline cannot  

A comparison plot is shown inside the notebook.

---

## 7. Case Study (Understanding Errors)

Here are a few examples where the baseline failed but the model succeeded:

### Example 1
- Text: *"The movie ending was beautiful"*  
- Baseline: **negative**  
- Model: **positive**  
- Reason: baseline doesn't know the word "beautiful"

### Example 2
- Text: *"The chair is comfortable for long hours"*  
- Baseline: **negative**  
- Model: **positive**  
- Reason: baseline only matches very specific words like "good" or "great"

### Example 3
- Text: *"The teacher explained things clearly"*  
- Baseline: **negative**  
- Model: **positive**  
- Reason: baseline ignores context and synonyms

This shows why AI models are necessary for real language tasks.

---

## 8. Reflection

What I learned from this project:

- Baselines are important: they show how much the AI actually improves.
- An AI pipeline includes more than the model: data, preprocessing, evaluation, and analysis all matter.
- Pretrained models make it easy to build working pipelines quickly.
- Simple rule-based systems break easily when language becomes slightly more complex.
- Checking mistakes manually helps understand model strengths and weaknesses.

Overall, this project helped me clearly see how all pieces of an AI system fit together.

---

## 9. How to Run (Colab)

Clone the repo:
!git clone https://github.com/leo90614/Mini-Ai-Pipeline.git


Add `src` to the Python path:

```python
import sys
sys.path.append("/content/Mini-Ai-Pipeline/src")


Open and run the notebook:
!git clone https://github.com/leo90614/Mini-Ai-Pipeline.git


