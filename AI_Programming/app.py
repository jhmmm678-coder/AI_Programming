# app.py
"""
Streamlit dashboard for financial news sentiment, tagging, risk level,
and evaluation of FinBERT on labeled data.
"""
# library/module import part
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from analysis import analyze_news_dataframe, evaluate_model

# streamlit app preferences
st.set_page_config(
    page_title="News Sentiment Dashboard",
    layout="wide",
)

# output app main title & breif descriptions
st.title("AI-Powered Financial News Sentiment & Risk Dashboard")
st.write(
    """
This app analyzes financial news headlines or summaries using a pre-trained FinBERT model.
It shows sentiment distribution (negative / neutral / positive), simple issue tags
(Earnings, Regulation, Growth, ESG), and a heuristic risk level (High / Medium / Low).
In evaluation mode, it also computes accuracy and a confusion matrix on labeled data.
"""
)

# Sidebar: Mode selection
st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Select mode:",
    ["Dashboard (unlabeled data)", "Evaluation (labeled data)"],
)
# option selection: CSV file vs. paste text
st.sidebar.header("Input Options")
upload_option = st.sidebar.selectbox(
    "How do you want to provide news data?",
    ["Upload CSV file", "Paste texts manually"],
)
# Set the column name containing news text in CSV (default: 'text')
text_column_name = st.sidebar.text_input(
    "Name of text column (if CSV is used):",
    value="text",
)
# button to start analyzing
analyze_button = st.sidebar.button("Run Analysis")

# dataframe clearing
news_df = None
# CSV uploading case
if upload_option == "Upload CSV file":
    # upload csv file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # create pandas dataframe
        news_df = pd.read_csv(uploaded_file)
        # preview
        st.write("Preview of uploaded data:")
        st.dataframe(news_df.head())
# text uploading case
else:
    manual_texts = st.text_area(
        "Paste one news headline or summary per line:",
        height=200,
        placeholder=(
            "Example:\n"
            "Tesla shares jump after strong Q3 earnings report\n"
            "Company X stock plunges as regulators launch investigation"
        ),
    )
    # Process if it is not a blank input
    if manual_texts.strip():
        # Divide by line, remove both spaces + remove blank lines
        lines = [
            line.strip()
            for line in manual_texts.split("\n")
            if line.strip()
        ]
        # Create DataFrame with a user-specified column name
        news_df = pd.DataFrame({text_column_name: lines})
        st.write("Preview of manually entered data:")
        st.dataframe(news_df.head())


# Main logic: if the button is pressed
if analyze_button:
    # input data is not prepared
    if news_df is None:
        st.warning("Please upload a CSV file or paste some texts before running analysis.")
    else:
        # 1) Dashboard mode: sentiment + tags + risk + filtering
        if mode == "Dashboard (unlabeled data)":
            with st.spinner("Running sentiment analysis..."):
                analyzed_df, dropped_count = analyze_news_dataframe(
                    news_df,
                    text_column=text_column_name,
                )

            if analyzed_df.empty:
                st.warning("No valid news texts after filtering. Please check your input.")
            else:
                st.success(f"Analysis completed! Filtered out {dropped_count} non-news / UI rows.")

                # Sentiment summary
                st.subheader("Sentiment Summary")
                sentiment_counts = analyzed_df["sentiment_label"].value_counts()
                total_news = len(analyzed_df)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total News (after filtering)", total_news)
                col2.metric("Positive", int(sentiment_counts.get("positive", 0)))
                col3.metric("Neutral", int(sentiment_counts.get("neutral", 0)))
                col4.metric("Negative", int(sentiment_counts.get("negative", 0)))

                # Sentiment distribution chart
                st.subheader("Sentiment Distribution")

                fig, ax = plt.subplots()
                sentiment_counts.reindex(
                    ["negative", "neutral", "positive"]
                ).plot(kind="bar", ax=ax)
                ax.set_xlabel("Sentiment")
                ax.set_ylabel("Count")
                ax.set_title("Number of news by sentiment label")
                st.pyplot(fig)

                # Risk level distribution
                st.subheader("Risk Level Distribution")
                risk_counts = analyzed_df["risk_level"].value_counts()
                fig_risk, ax_risk = plt.subplots()
                risk_counts.reindex(["High", "Medium", "Low"]).plot(kind="bar", ax=ax_risk)
                ax_risk.set_xlabel("Risk level")
                ax_risk.set_ylabel("Count")
                ax_risk.set_title("Number of news by risk level")
                st.pyplot(fig_risk)

                # Tags distribution
                st.subheader("Issue Tags Distribution")
                tags_series = (
                    analyzed_df["tags"]
                    .fillna("")
                    .str.split(";")
                    .explode()
                    .str.strip()
                )
                tag_counts = tags_series.value_counts()

                fig2, ax2 = plt.subplots()
                tag_counts.plot(kind="bar", ax=ax2)
                ax2.set_xlabel("Tag")
                ax2.set_ylabel("Count")
                ax2.set_title("Number of news by issue tag")
                st.pyplot(fig2)

                # Show full table
                st.subheader("Detailed Results")
                st.dataframe(analyzed_df)

                # Download analyzed results
                csv_download = analyzed_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download analyzed CSV",
                    data=csv_download,
                    file_name="analyzed_news.csv",
                    mime="text/csv",
                )

        # 2) Evaluation mode: labeled data performance
        else:  # mode == "Evaluation (labeled data)"
            st.info("Upload a CSV file with columns 'text' and 'label' to evaluate FinBERT.")

            if upload_option != "Upload CSV file":
                st.warning("For evaluation mode, please use 'Upload CSV file' and include labels.")
            elif "label" not in news_df.columns:
                st.error("The uploaded CSV must contain a 'label' column with true sentiment labels.")
            else:
                with st.spinner("Evaluating FinBERT on labeled data..."):
                    acc, cm_df = evaluate_model(
                        news_df,
                        text_column=text_column_name,
                        label_column="label",
                    )

                st.success(f"Evaluation completed! Accuracy: {acc:.3f}")

                st.subheader("Confusion Matrix")
                st.dataframe(cm_df)
