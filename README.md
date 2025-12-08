# Mini-AI-Pipeline: Simple Sentiment Classification Project

This project builds a small end-to-end AI pipeline that classifies short English sentences into **positive** or **negative** sentiment.  
The goal is to clearly show how a simple AI workflow works from data to evaluation.

---

## 1. Task Definition

### What are we trying to do?
We want a system that reads a short sentence and predicts whether the sentiment is **positive** or **negative**.

### Why is this useful?
People write countless reviews, comments, and reactions every day.  
No one can manually read everything, so a sentiment classifier helps quickly understand how people feel.

Even though this task is simple, it nicely demonstrates the full AI workflow:

**data → baseline → model → evaluation → reflection**

### Input → Output
- **Input:** one short English sentence  
- **Output:** `"positive"` or `"negative"`

### Success Criteria
- The AI model should outperform the basic rule-based baseline.  
- Accuracy and F1-score should clearly improve.  
- Predictions should also make sense when checking example sentences manually.

---

## 2. Dataset

I created a custom dataset of **300 sentences**, with an almost perfectly balanced number of positive and negative examples.

**Example format:**

| text                     | label    |
|--------------------------|----------|
| The food was amazing     | positive |
| The app keeps crashing   | negative |

The file is here:
**```data/sentiment_data.csv```**


---

## 3. Baseline System (Rule-based)

Before using a real model, we need a simple baseline.

My baseline works like this:

- I made two tiny lists: positive words and negative words.
- If a sentence contains more positive words → predict **positive**
- Otherwise → predict **negative**

This method is extremely limited, but it shows why we need a real model.

File: **```src/baseline.py```**



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

File: **```src/pipeline.py```**



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

In our final dataset, the baseline reached around **0.60 accuracy, 1.00 precision, 0.20 recall**, and **0.33 F1-score**.

### AI Model (Transformer)
- Much higher accuracy and F1-score  
- Understands a wide variety of expressions  
- Handles subtle positive wording that baseline cannot

The transformer model achieved about **0.98 accuracy, 0.97 precision, 1.00 recall**, and **0.98 F1-score**.
This is a huge improvement over the baseline!

A comparison plot is shown inside the notebook.

---

## 7. Case Study (Understanding Errors)

Here are some examples from the test set where the baseline failed but the model succeeded.  
These examples help show why the rule-based system struggles while the transformer model performs much better.

### Example 1
- **Text:** "The tour guide was knowledgeable"  
- **Baseline:** negative  
- **Model:** positive  
- **Reason:** the baseline does not recognize the word “knowledgeable” as positive.

### Example 2
- **Text:** "The movie trailer looked exciting"  
- **Baseline:** negative  
- **Model:** positive  
- **Reason:** the baseline only detects simple positive words like “good” or “great,” and misses many natural expressions.

### Example 3
- **Text:** "The pizza crust was crispy"  
- **Baseline:** negative  
- **Model:** positive  
- **Reason:** the baseline cannot understand positive descriptions outside its tiny keyword list.

### Example 4
- **Text:** "The bread was soft and warm"  
- **Baseline:** negative  
- **Model:** positive  

### Example 5
- **Text:** "The hotel staff smiled at us"  
- **Baseline:** negative  
- **Model:** positive  

These examples clearly show that the rule-based baseline is extremely limited, while the transformer model can understand much more natural phrasing and context.

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


