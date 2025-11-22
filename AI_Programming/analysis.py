# analysis.py
"""
Sentiment analysis, issue tagging, and risk estimation
for financial news using a pre-trained FinBERT model.
"""
# library/module import part
import os
import torch
import pandas as pd
# Hugging Face Transformers: FinBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 1. Load pre-trained FinBERT model
# On the school server, we first try to use a local "../finbert" folder.
# If it does not exist (e.g., on TA's machine), we fall back to the Hugging Face model name.

LOCAL_MODEL_DIR = "../finbert"   # relative to the AI_Programming folder

if os.path.isdir(LOCAL_MODEL_DIR):
    MODEL_NAME = LOCAL_MODEL_DIR
else:
    MODEL_NAME = "ProsusAI/finbert"
# use cuda when GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load FinBERT tokenizer & sequence classification model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


# 2. Simple heuristics for filtering and risk

UI_KEYWORDS = [
    "menu", "watchlist", "video", "markets", "livestream",
    "sign in", "sign up", "skip navigation", "search", "home",
]


def is_probably_ui_text(text: str, min_words: int = 3) -> bool:
    """
    Very short texts or typical UI/menu words are treated as non-news.
    Returns True if the text looks like UI, not a real headline.
    """
    # None, or not string, is difficult to view as news → UI
    if not text or not isinstance(text, str):
        return True

    lower = text.lower().strip()

    # Very short strings are likely to be buttons or menu labels
    word_count = len(lower.split())
    if word_count < min_words:
        return True

    # Typical navigation / UI terms
    for kw in UI_KEYWORDS:
        if kw in lower:
            return True

    return False


def compute_risk_level(sentiment_label: str, tags: list, scores: dict) -> str:
    """
    Simple heuristic risk score for investors.

    - High Risk:
        * negative sentiment with high negative probability
        * or regulation/legal risk with high negative probability
    - Medium Risk:
        * neutral sentiment but regulation or earnings related
        * default case when not clearly low or high
    - Low Risk:
        * clearly positive sentiment without regulation risk
    """
    negative_prob = scores.get("negative", 0.0)
    positive_prob = scores.get("positive", 0.0)

    # Risk classification based on conditions
    if "Regulation / Legal Risk" in tags and negative_prob >= 0.4:
        return "High"
    if sentiment_label.lower() == "negative" and negative_prob >= 0.5:
        return "High"

    if sentiment_label.lower() == "neutral" and (
        "Regulation / Legal Risk" in tags or "Earnings" in tags
    ):
        return "Medium"

    if sentiment_label.lower() == "positive" and positive_prob >= 0.5:
        return "Low"

    return "Medium"


# 3. Core FinBERT sentiment + rule-based tags

def predict_sentiment(text: str):
    """
    Run sentiment analysis on a single news text using FinBERT.

    Returns:
        dict with:
            - label: predicted sentiment label (e.g., 'positive')
            - scores: dictionary of label -> probability
    """
    # Return 'unknow' unless it's a valid string
    if not text or not isinstance(text, str):
        return {"label": "unknown", "scores": {}}
    # Convert input sentences to token ID/attention mask with talknizer
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    # Deactivate gradient calculation because it is an inference step
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Calculate probability for each label with softmax
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    # FinBERT uses id2label such as {0: 'negative', 1: 'neutral', 2: 'positive'}
    id2label = model.config.id2label
    pred_id = int(probs.argmax())
    pred_label = id2label[pred_id]
    # Convert probability for each label to dictionary
    scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "label": pred_label,
        "scores": scores,
    }


def assign_tags(text: str):
    """
    Assign simple rule-based issue tags to a news text.

    Tags examples:
        - 'Regulation / Legal Risk'
        - 'Earnings'
        - 'Growth'
        - 'ESG / Climate'
        - 'Other'
    """
    # Classified as 'Other' unless it is a valid text
    if not text or not isinstance(text, str):
        return ["Other"]

    text_lower = text.lower()
    tags = []

    # Regulation / Legal risk
    regulation_keywords = [
        "lawsuit", "regulation", "ban", "fine", "probe",
        "investigation", "antitrust", "legal", "court",
    ]
    if any(word in text_lower for word in regulation_keywords):
        tags.append("Regulation / Legal Risk")

    # Earnings and financial performance
    earnings_keywords = [
        "earnings", "revenue", "profit", "loss", "guidance",
        "q1", "q2", "q3", "q4", "quarter", "dividend",
    ]
    if any(word in text_lower for word in earnings_keywords):
        tags.append("Earnings")

    # Growth and expansion
    growth_keywords = [
        "growth", "expansion", "market share", "demand",
        "sales increase", "record", "new market",
    ]
    if any(word in text_lower for word in growth_keywords):
        tags.append("Growth")

    # ESG / climate / renewable
    esg_keywords = [
        "esg", "carbon", "emissions", "climate",
        "sustainability", "renewable", "solar", "wind",
        "green", "environment",
    ]
    if any(word in text_lower for word in esg_keywords):
        tags.append("ESG / Climate")

    if not tags:
        tags.append("Other")

    return tags


# 4. DataFrame-level analysis & evaluation

def analyze_news_dataframe(df: pd.DataFrame, text_column: str = "text"):
    """
    Apply sentiment analysis and issue tagging to all rows in a DataFrame.

    - Filters out rows that look like UI/menu text
    - Adds sentiment scores, tags, and a simple risk level

    Args:
        df: input DataFrame containing a column with news texts
        text_column: name of the column containing the news text

    Returns:
        result_df: analyzed rows
        dropped_count: number of rows skipped as non-news
    """
    results = []
    dropped_count = 0
    # Perform analysis while traversing each row
    for _, row in df.iterrows():
        text = str(row.get(text_column, ""))

        # 1) Skip UI/menu-like texts
        if is_probably_ui_text(text):
            dropped_count += 1
            continue

        # 2) FinBERT sentiment
        sentiment = predict_sentiment(text)
        label = sentiment["label"]
        scores = sentiment["scores"]

        positive_score = scores.get("positive", 0.0)
        neutral_score = scores.get("neutral", 0.0)
        negative_score = scores.get("negative", 0.0)

        # 3) Tags + risk level
        tags = assign_tags(text)
        risk_level = compute_risk_level(label, tags, scores)
        # Save analysis results for one row as a dictionary
        results.append(
            {
                text_column: text,
                "sentiment_label": label,
                "sentiment_positive": positive_score,
                "sentiment_neutral": neutral_score,
                "sentiment_negative": negative_score,
                "tags": "; ".join(tags),
                "risk_level": risk_level,
            }
        )
    # Converting the complete list of results to DataFrame
    result_df = pd.DataFrame(results)
    return result_df, dropped_count


def evaluate_model(df: pd.DataFrame, text_column: str = "text", label_column: str = "label"):
    """
    Evaluate FinBERT predictions against manually labeled data.

    Args:
        df: DataFrame with columns 'text' and 'label'
        text_column: name of the column containing the news text
        label_column: name of the column containing the true label

    Returns:
        accuracy (float) and confusion matrix DataFrame
    """
    from sklearn.metrics import accuracy_score, confusion_matrix

    true_labels = []
    pred_labels = []
    # Collect FinBERT predictive labels and actual labels for each row
    for _, row in df.iterrows():
        text = str(row.get(text_column, ""))
        true_label = str(row.get(label_column, "")).lower()

        sentiment = predict_sentiment(text)
        predicted_label = sentiment["label"].lower()

        true_labels.append(true_label)
        pred_labels.append(predicted_label)
    # 1) Calculate overall accuracy
    acc = accuracy_score(true_labels, pred_labels)
    # 2) Calculate the confusion matrix (label order is fixed as negative, neutral, and positive)
    cm = confusion_matrix(true_labels, pred_labels, labels=["negative", "neutral", "positive"])
    cm_df = pd.DataFrame(
        cm,
        index=["true_negative", "true_neutral", "true_positive"],
        columns=["pred_negative", "pred_neutral", "pred_positive"],
    )

    return acc, cm_df
