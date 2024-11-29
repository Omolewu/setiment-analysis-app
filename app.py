import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from io import StringIO
import time

# Set page configuration
st.set_page_config(
    page_title="Financial News Sentiment Analyzer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the FinBERT model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

def analyze_sentiment(text, tokenizer, model):
    """Analyze sentiment of given text using FinBERT"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    probabilities = probabilities.detach().numpy()[0]
    
    labels = ['negative', 'neutral', 'positive']
    sentiment = labels[np.argmax(probabilities)]
    confidence = float(max(probabilities))
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': dict(zip(labels, probabilities))
    }

def main():
    # Header
    st.title("📊 Financial News Sentiment Analyzer")
    st.markdown("""
    Analyze the sentiment of financial news using state-of-the-art FinBERT model.
    Choose between single text analysis or batch processing via CSV upload.
    """)
    
    # Load model
    with st.spinner("Loading FinBERT model..."):
        tokenizer, model = load_model()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📝 Single Text Analysis", "📁 Batch Processing"])
    
    # Single Text Analysis Tab
    with tab1:
        st.subheader("Analyze Individual Text")
        text_input = st.text_area(
            "Enter financial news headline or article",
            height=100,
            placeholder="Enter your text here..."
        )
        
        if st.button("Analyze Sentiment", key="single_analyze"):
            if text_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    result = analyze_sentiment(text_input, tokenizer, model)
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['confidence'] * 100,
                        title={'text': f"Confidence: {result['sentiment'].title()}"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': {
                                   'positive': "#28a745",
                                   'negative': "#dc3545",
                                   'neutral': "#6c757d"
                               }[result['sentiment']]}}
                    ))
                    st.plotly_chart(fig)
                
                with col2:
                    # Probabilities bar chart
                    prob_df = pd.DataFrame({
                        'Sentiment': list(result['probabilities'].keys()),
                        'Probability': list(result['probabilities'].values())
                    })
                    fig = px.bar(
                        prob_df,
                        x='Sentiment',
                        y='Probability',
                        color='Sentiment',
                        color_discrete_map={
                            'positive': "#28a745",
                            'negative': "#dc3545",
                            'neutral': "#6c757d"
                        }
                    )
                    st.plotly_chart(fig)
            else:
                st.warning("Please enter some text to analyze.")
    
    # Batch Processing Tab
    with tab2:
        st.subheader("Batch Analysis")
        st.markdown("Upload a CSV file with a column named 'text' containing the news headlines or articles.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column.")
                    return
                
                with st.spinner("Analyzing batch data..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, row in df.iterrows():
                        result = analyze_sentiment(row['text'], tokenizer, model)
                        results.append(result)
                        progress_bar.progress((i + 1) / len(df))
                    
                    results_df = pd.DataFrame(results)
                    
                    # Display overall sentiment distribution
                    st.subheader("Sentiment Distribution")
                    sentiment_counts = results_df['sentiment'].value_counts()
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'positive': "#28a745",
                            'negative': "#dc3545",
                            'neutral': "#6c757d"
                        }
                    )
                    st.plotly_chart(fig)
                    
                    # Display detailed results table
                    st.subheader("Detailed Results")
                    results_df['confidence'] = results_df['confidence'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(
                        results_df[['text', 'sentiment', 'confidence']],
                        column_config={
                            "sentiment": st.column_config.Column(
                                "Sentiment",
                                help="Predicted sentiment",
                                width="medium",
                            ),
                            "confidence": st.column_config.Column(
                                "Confidence",
                                help="Model confidence score",
                                width="small",
                            ),
                        },
                        hide_index=True,
                    )
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()