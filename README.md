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

The dataset contains **exactly 300 short English sentences**, roughly balanced between `positive` and `negative`.  
The goal was to keep the sentences simple and diverse, so that both the baseline and the AI model can be evaluated clearly.

### Format
Each entry consists of a text sentence and a sentiment label:


| text                     | label    |
|--------------------------|----------|
| The food was amazing     | positive |
| The app keeps crashing   | negative |


The file is here:
**```data/sentiment_data.csv```**


### Train/Test Split
We split the dataset into **80% training** and **20% testing** using a randomized split.  
Because the dataset is balanced, this split also results in a nearly balanced test set.

### Preprocessing
Minimal preprocessing was applied, since the Transformer tokenizer handles most steps automatically:

- **Lowercasing:** handled by the pretrained model (uncased tokenizer).  
- **Tokenization:** using the DistilBERT tokenizer (`AutoTokenizer`).  
- **Truncation:** sentences longer than the model’s maximum length are automatically truncated.  
- **Padding:** shorter sentences are padded during batching.  
- **No manual text cleaning:** we intentionally kept sentences simple and natural to simulate a realistic small dataset.

These steps ensure that the input is standardized before being fed into both the baseline and the AI pipeline.


---

## 3. Baseline System (Rule-based)

Before building the actual AI model, I first designed a very simple **naïve baseline** that does not rely on any complex machine learning.  
The goal of this baseline is to give us a reference point so we can measure how much the AI pipeline improves the task.

### How the baseline works
The baseline uses a small, hand-crafted set of keywords:

- A short list of **positive** words (e.g., *good, nice, helpful, amazing*).
- A short list of **negative** words (e.g., *bad, terrible, awful, boring*).

For each sentence:
1. Count how many positive keywords appear.  
2. Count how many negative keywords appear.  
3. If positive > negative → predict **positive**.  
4. Otherwise → predict **negative**.

This approach does **not** look at grammar, synonyms, sentence structure, or the overall meaning.  
It simply checks whether a few specific words appear.

### Why this is considered naïve
- It cannot understand common positive words not in the tiny list (e.g., *knowledgeable*, *crispy*, *exciting*).
- It ignores context (e.g., sarcasm, negation like *“not good”*).
- It breaks easily when sentences use more natural expressions.
- It treats all sentences as bags of words with no structure.

### When the baseline fails
This rule-based method usually fails on:
- Descriptive adjectives (*warm*, *soft*, *crispy*).  
- Subtle positivity (e.g., *“the staff smiled at us”*).  
- Sentences with no explicit “keyword.”  
- Slightly long or complex sentences.

Because of these limitations, this baseline gives us a good contrast point when evaluating the transformer model.

File: **```src/baseline.py```**



---


---

## 4. AI Pipeline (Transformer Model)

For the main system, we built a small but realistic **AI pipeline** using a pretrained transformer model:

```textattack/distilbert-base-uncased-SST-2```

I did **not** fine-tune the model.  
I only used it for inference, which is allowed by the assignment.

### Pipeline Stages

The pipeline has four clear stages, similar to what the assignment describes:


### **1. Preprocessing**
I used the DistilBERT tokenizer to:
- lowercase text (because it's *uncased*)  
- tokenize the sentence into subword units  
- apply truncation if the text is too long  
- pad sequences during batching  

No manual cleaning was required because the tokenizer handles formatting reliably.


### **2. Representation (Embeddings)**
The tokenized input is passed into the DistilBERT encoder, which outputs:
- contextual embeddings for each token  
- a pooled representation for the entire sentence  

These embeddings capture the meaning of the sentence far better than simple keyword matching.


### **3. Decision Stage**
The pretrained classification head converts embeddings into:
- logits for **positive** and **negative**  
- I took the `argmax` to produce the final label

This replaces the “keyword counting” logic used in the baseline.


### **4. Optional Post-processing**
After prediction:
- I converted the model’s numeric outputs (0/1) into readable labels (“positive”, “negative”).  
- I batched predictions for faster evaluation.

No additional thresholds or reranking were necessary.



### Why the pipeline works better
- It understands synonyms and subtle phrasing (e.g., *crispy*, *knowledgeable*, *smiled*).  
- It uses context, not just individual words.  
- It handles various writing styles and sentence structures.  
- Pretrained knowledge from large datasets makes the predictions robust even with a small dataset.


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

### Performance Summary (Baseline vs Model)

| Model               | Accuracy | Precision | Recall | F1-score |
|--------------------|----------|-----------|--------|----------|
| Rule-based Baseline | 0.60     | 1.00      | 0.20   | 0.33     |
| Transformer Model   | 0.98     | 0.97      | 1.00   | 0.98     |

The transformer model clearly outperforms the baseline across all metrics, especially recall and F1-score.


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


A comparison bar graph is shown inside the notebook.

---

## 7. Case Study (Understanding Errors)

Below are several examples from the test set where the baseline classifier failed but the transformer model succeeded.  
These examples show the limitations of rule-based methods and why a learned model is necessary.

---

### **Example 1**
**Text:** "The tour guide was knowledgeable"  
- **Baseline:** negative  
- **Model:** positive  
- **Why?**  
  The baseline only checks a tiny list of “positive words.”  
  It does not understand that “knowledgeable” carries positive meaning.  
  The transformer model correctly learns this from context.

---

### **Example 2**
**Text:** "The movie trailer looked exciting"  
- **Baseline:** negative  
- **Model:** positive  
- **Why?**  
  Words like “exciting” are not in the baseline’s keyword list,  
  so the baseline mislabels it as negative.  
  The model generalizes to unseen positive expressions.

---

### **Example 3**
**Text:** "The pizza crust was crispy"  
- **Baseline:** negative  
- **Model:** positive  
- **Why?**  
  The baseline cannot interpret descriptive adjectives like “crispy.”  
  The model captures these finer-grained semantics.

---

### **Example 4**
**Text:** "The bread was soft and warm"  
- **Baseline:** negative  
- **Model:** positive  
- **Why?**  
  “Soft” and “warm” are implicitly positive, but not in the baseline dictionary.  
  The transformer recognizes the positive tone of the sentence.

---

### **Example 5**
**Text:** "The hotel staff smiled at us"  
- **Baseline:** negative  
- **Model:** positive  
- **Why?**  
  The rule-based system cannot detect subtle positive cues like “smiled.”  
  The model understands actions and sentiment beyond simple keywords.

---

Together, these examples show how the baseline fails whenever sentences use natural phrasing or subtle positive descriptions, while the transformer model successfully handles these variations in everyday language.


---

## 8. Reflection

What I learned from this project:

- Baselines are important: they show how much the AI actually improves.
- An AI pipeline includes more than the model: data, preprocessing, evaluation, and analysis all matter.
- Pretrained models make it easy to build working pipelines quickly.
- Simple rule-based systems break easily when language becomes slightly more complex.
- Checking mistakes manually helps understand model strengths and weaknesses.

I also found that accuracy and F1-score captured the performance of this task quite well, especially because the dataset was balanced. Recall was particularly informative in showing the baseline’s weaknesses.

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


