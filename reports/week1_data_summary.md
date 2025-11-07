
# EDA Summary (Laptop – Subtask 3)

- Total records: 5773
- Unique aspects: 956
- Unique categories: 121
- Unique opinion words: 1156
- Average Valence: 5.94
- Average Arousal: 6.67

# 📊 Week 1 Data Exploration Summary — DimABSA 2026 Track A (Laptop Domain)

## 1️⃣ Overview
This exploratory analysis examined the **Laptop** subset of the **Subtask 3 (DimASQP)** dataset from the DimABSA 2026 Track A competition.  
The goal was to understand the data structure, sentiment distribution, and linguistic patterns before modeling.

Each record contains:
- **Text** – customer review sentence  
- **Aspect**, **Category**, **Opinion** – explicit or implicit targets and descriptors  
- **VA** – combined *Valence (Affective Polarity)* and *Arousal (Intensity)* scores  

---

## 2️⃣ Valence and Arousal Distributions
![Valence and Arousal Histograms](../notebooks/imgs/valence_arousal.png)

- **Valence** values cluster around 6 – 8 → majority of reviews express **positive sentiment**.  
- **Arousal** peaks around 6 – 7 → opinions are written with **moderate-to-high emotional intensity**.  
- Very few instances fall below 3 → limited negative samples.  

**Interpretation →** Dataset is *positively biased* with moderately expressive emotional tone.

---

## 3️⃣ Category Frequency
![Category Distribution](../notebooks/imgs/category.png)

- Over 50 fine-grained categories appear (e.g., `LAPTOP#GENERAL`, `BATTERY#PERFORMANCE`, `HARDWARE#OPERATION`).  
- Text overlap occurs due to large category diversity.  

**Interpretation →** Coverage is **broad and detailed**; later preprocessing may require *category grouping* to simplify modeling.

---

## 4️⃣ Top Aspects
![Top Aspects](../notebooks/imgs/aspect.png)

| Rank | Aspect | Notes |
|------|---------|-------|
| 1 | `NULL` | Implicit aspects (no explicit noun target) |
| 2 | `laptop` | General product-level comments |
| 3 | `screen` | Display quality and size |
| 4 | `keyboard` | Usability & design feedback |
| 5 | `battery life` | Performance duration |

**Interpretation →** Most opinions focus on **core hardware**, with several **implicit aspect** cases that need careful handling during preprocessing.

---

## 5️⃣ Top Opinion Words
![Top Opinions](../notebooks/imgs/opinion.png)

Frequent terms: **great, good, love, nice, fast, excellent, easy, perfect**.  
Presence of `"NULL"` marks sentences lacking explicit opinion words.

**Interpretation →** Lexical field dominated by **positive adjectives**, aligning with Valence ≈ 6 – 8.

---

## 6️⃣ Valence vs Arousal Relationship
![Valence vs Arousal](../notebooks/imgs/va_relation.png)

- Scatterplot reveals a **V-shaped trend**:  
  - High Arousal accompanies both *high* and *low* Valence extremes.  
  - Mid-range Valence (~5) shows calmer Arousal (~5 – 6).  
- **Correlation ≈ 0.6**, indicating moderate positive association.

**Interpretation →** Emotional intensity increases with both strong positivity and negativity—consistent with affective theory.

---

## 7️⃣ Key Insights for Preprocessing
- **Handle NULL fields:** expand or impute implicit aspect/opinion pairs.  
- **Normalize categories:** merge rare labels under higher-level types.  
- **Balance sentiment:** consider oversampling low-Valence samples.  
- **Tokenization:** plan to use Transformer tokenizers (e.g., BERT/DeBERTa).  
- **Output format:** each `(Text, Aspect, Category, Opinion)` should map to `(Valence, Arousal)` pair.

---

## Conclusion
The laptop dataset in Subtask 3 is rich, positively skewed, and emotionally expressive.  
These findings establish a foundation for **Week 2 preprocessing** and **model training**, ensuring consistency and interpretability across subtasks.

---
