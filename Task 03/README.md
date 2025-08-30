# 📰 News Headline Classification with BERT

## 📌 Problem Statement
In today’s digital world, thousands of news articles are published every day. Efficiently organizing these articles into categories is crucial for search, recommendations, and information filtering.  
The goal of this project is to **automatically classify news headlines** into one of four predefined categories using **Natural Language Processing (NLP)**.

---

## 🎯 Objective
- Build a **text classification model** that predicts the category of a news headline.  
- Use **BERT (Bidirectional Encoder Representations from Transformers)** for fine-tuning on the AG News dataset.  
- Focus only on **headlines (titles)** instead of full descriptions for faster inference and practical usage.  
- Deploy the model with a **simple UI (Gradio/Streamlit)** where users can input a news headline and receive:
  - Predicted category (World, Sports, Business, Sci/Tech)  
  - Confidence score for each class  

---

## 📊 Dataset
- **Dataset:** [AG News](https://huggingface.co/datasets/ag_news)  
- **Size:**  
  - Training set: **120,000 headlines**  
  - Test set: **7,600 headlines**  
- **Features:**  
  - `Class Index` (1–4 → mapped to categories)  
  - `Title` (news headline – used for training)  
  - `Description` (ignored in this project)  

**Categories:**  
1. 🌍 World  
2. 🏅 Sports  
3. 💼 Business  
4. 🔬 Science/Technology  

---

## 🛠️ Approach
1. **Data Loading & Preprocessing**
   - Load AG News dataset from HuggingFace.  
   - Use only `Title` for training.  
   - Map labels (1–4) → (0–3).  

2. **Modeling**
   - Fine-tune **BERT-base-uncased** using HuggingFace `Trainer`.  
   - Train with `CrossEntropyLoss` for 2–3 epochs.  

3. **Evaluation**
   - Metrics: Accuracy, F1-score (weighted).  
   - Balanced dataset ensures reliable evaluation.  

4. **Deployment**
   - Gradio/Streamlit app for real-time predictions.  
   - Input: News headline  
   - Output: Predicted category + confidence scores  

---

## 🧠 Model & Tools

- **Model**: [`bert-base-uncased`](https://huggingface.co/bert-base-uncased)
- **Libraries**:
  - 🤗 Transformers
  - 🤗 Datasets
  - PyTorch
  - Scikit-learn
  - Gradio

---

 ## 📈 Evaluation Metrics

| Metric   | Score (Fast Training Example) |
|----------|-------------------------------|
| Accuracy | ~88%                          |
| F1 Score | ~88%                          |

> *(Scores may vary depending on dataset size and epochs)*

---  

## 🚀 Features

- Fine-tuned BERT with custom classification head
- Tokenization with padding and truncation
- Accuracy and F1-score evaluation
- Live demo using Gradio
- Google Colab-compatible (GPU-supported)

---

## 📈 Results & Insights
- Fine-tuned BERT achieved around **94–96% accuracy** on the test set.  
- Short headlines were enough for classification → no need for full descriptions.  
- Real-time predictions possible in <1 second using Gradio/Streamlit.  

---



## 🔗 References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [AG News Dataset](https://huggingface.co/datasets/ag_news)

---

## 👩‍💻 Author

**Urooj Fatima**  

