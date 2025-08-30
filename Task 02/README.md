# Auto Tagging Support Tickets Using LLM

## 📌 Problem Statement
Organizations often receive thousands of **support queries** written in free text (e.g., login errors, billing issues, connectivity problems). Manually tagging these tickets into categories like *Billing*, *Authentication*, or *Connectivity* is:
- Time-consuming  
- Inconsistent across agents  
- Not scalable as ticket volume grows  

The objective of this project is to **automatically classify support queries into categories** using **Large Language Models (LLMs)**. This enables faster routing of issues to the right teams and improves customer support efficiency.  

---

## 🎯 Objective
- Build an **LLM-based pipeline** that automatically tags support queries.  
- Compare **zero-shot**, **few-shot**, and **fine-tuned** performance.  
- Generate **top-3 probable tags** for each support query.  

---

## 📂 Dataset
The dataset consists of free-text support tickets with two main fields:  
- **support_query** → description of the issue raised by the customer  
- **resolution_tag** → assigned category/label (e.g., Billing, Connectivity, Authentication)  

---

## ⚙️ Methodology
. **Zero-Shot Classification**
   - Used a pre-trained LLM (e.g., `facebook/bart-large-mnli`)  
   - Directly predicted categories without training  


. **Evaluation**
   - Top-3 prediction accuracy  
   - Confusion matrix to analyze misclassifications  

---

## 📊 Results / Observations
- Zero-shot performed reasonably well but struggled with domain-specific queries.  
- Few-shot improved classification on niche categories.  
- Fine-tuning achieved the **highest accuracy and consistency**.  
- Providing **top-3 probable tags** helped mitigate single-label errors.  

---




## ✅ Key Insights
- LLMs provide strong baselines for text classification tasks.  
- Few-shot learning is a **cost-effective** way to boost performance.  
- Auto-tagging improves ticket routing efficiency and reduces manual effort.  


---
## 👩‍💻 Author  
**Urooj Fatima**  
AI/ML Engineering Intern  
DevelopersHub Corporation  
