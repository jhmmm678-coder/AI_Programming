# EF2039 Project 01 – AI-Powered Financial News Sentiment Dashboard

**Student ID:** 20240110 
**Name:** Jungho Moon

## 1. Project Overview

This project builds a simple web dashboard that analyzes financial news
headlines or summaries using a pre-trained AI model, **FinBERT**.
The app estimates the sentiment of each news item (negative / neutral / positive)
and assigns simple issue tags such as:

- Regulation / Legal Risk  
- Earnings  
- Growth  
- ESG / Climate  
- Other  

The goal is to help retail investors quickly understand the overall mood
and main issues around a specific stock or topic without reading every article
in detail.

## 2. AI Model and Motivation

### 2.1 Pre-trained Model

- **Model name:** `ProsusAI/finbert`
- **Type:** BERT-based transformer model
- **Domain:** Financial text sentiment analysis
- **Source:** Hugging Face Hub (pre-trained weights)

I chose FinBERT because it was trained specifically on financial texts such as
earnings reports and financial news. Therefore, it is more suitable for my
use case than a generic sentiment model trained on movie reviews or tweets.

### 2.2 How the Model Is Used

I do **not** train a model from scratch.  
Instead, I load the pre-trained FinBERT model from source code and
use it to:

1. Tokenize each news text  
2. Run a forward pass through the model  
3. Obtain class probabilities for negative / neutral / positive  
4. Select the label with the highest probability  

This satisfies the requirement of using at least one AI model from source code
with pre-trained weights.

## 3. System Pipeline

The overall pipeline is:

1. **Input**
   - User uploads a CSV file containing a `text` column, or
   - User manually pastes news texts (one per line).

2. **Preprocessing**
   - Convert text to lowercase internally.
   - Truncate to a maximum sequence length for the model (256 tokens).

3. **Sentiment Analysis**
   - Use FinBERT to predict sentiment labels and probabilities
     for each news item.

4. **Issue Tagging (Rule-based)**
   - Apply simple keyword rules to assign tags:
     - "lawsuit", "regulation", "fine" → Regulation / Legal Risk  
     - "earnings", "profit", "loss" → Earnings  
     - "growth", "market share" → Growth  
     - "carbon", "renewable", "solar" → ESG / Climate  
   - If no keyword matches, the tag is set to "Other".

5. **Aggregation and Visualization**
   - Count how many news items are negative / neutral / positive.
   - Count how many news items have each issue tag.
   - Display results as bar charts and a detailed table.

6. **Output**
   - Streamlit web page with charts, metrics, and a table.
   - Button to download the analyzed results as a CSV file.

## 4. Files and Directory Structure

```text
EF2039_Proj01_YourID_YourName/
├── app.py                # Streamlit app (UI + visualization)
├── analysis.py           # Sentiment and tagging logic with FinBERT
├── requirements.txt      # Python dependencies
├── README.md             # Project description (this file)
└── .gitignore            # Optional, ignore venv and cache files
