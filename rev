import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gensim
from gensim import corpora
from gensim.models import LdaModel
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import time
import pickle
import os
from PIL import Image
from io import BytesIO
import base64

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="Restaurant Review Sentiment Analyzer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        color: #888888;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    .positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .negative {
        color: #F44336;
        font-weight: bold;
    }
    .neutral {
        color: #FFC107;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def preprocess_text(text):
    """Preprocess text by removing special characters, lowercasing, and removing stopwords"""
    if isinstance(text, str):
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        # Keep negation words as they're important for sentiment
        negations = {"no", "not", "nor", "neither", "never", "none", "wasn't", "weren't", "doesn't", "don't"}
        stop_words = stop_words - negations
        
        filtered_tokens = [w for w in tokens if w not in stop_words]
        
        return " ".join(filtered_tokens)
    return ""

def load_models():
    """Load or train the models for sentiment analysis"""
    # Create directory for models if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Placeholder for ML model - in production, you'd load a trained model
    # For demo purposes, we'll train a simple model on a small sample dataset
    try:
        # Try to load the saved ML model
        with open('models/nb_model.pkl', 'rb') as f:
            ml_model = pickle.load(f)
        st.sidebar.success("✓ ML model loaded successfully")
    except:
        st.sidebar.warning("⚠️ Training a simple model for demo purposes...")
        # Sample data for training
        sample_data = pd.DataFrame({
            'review': [
                "The food was absolutely delicious and the service was excellent!",
                "Great ambience, but the food was mediocre at best.",
                "Terrible service, waited for an hour to get our food.",
                "The best dining experience I've had in years!",
                "Food was cold when served, and the waiter was rude.",
                "Beautiful restaurant with amazing views, but overpriced food.",
                "The chef's special was out of this world!",
                "Disappointing experience overall, would not recommend.",
                "Friendly staff and reasonable prices, will return!",
                "Appetizers were good but main courses were disappointing."
            ],
            'sentiment': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
        })
        
        # Preprocess the reviews
        sample_data['processed_review'] = sample_data['review'].apply(preprocess_text)
        
        # Train a simple model
        X_train, X_test, y_train, y_test = train_test_split(
            sample_data['processed_review'], 
            sample_data['sentiment'], 
            test_size=0.2, 
            random_state=42
        )
        
        ml_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('clf', MultinomialNB())
        ])
        
        ml_model.fit(X_train, y_train)
        
        # Save the model
        with open('models/nb_model.pkl', 'wb') as f:
            pickle.dump(ml_model, f)
        
        st.sidebar.success("✓ Simple model trained and saved")
    
    # Load the transformer model
    @st.cache_resource
    def load_transformer_model():
        try:
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            return tokenizer, model
        except Exception as e:
            st.error(f"Error loading transformer model: {e}")
            return None, None
    
    tokenizer, transformer_model = load_transformer_model()
    
    # Load VADER sentiment analyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    
    return ml_model, tokenizer, transformer_model, vader_analyzer

def predict_sentiment_ml(text, model):
    """Predict sentiment using the ML model"""
    processed_text = preprocess_text(text)
    pred = model.predict([processed_text])[0]
    pred_proba = model.predict_proba([processed_text])[0]
    confidence = pred_proba[1] if pred == 1 else pred_proba[0]
    return "Positive" if pred == 1 else "Negative", confidence

def predict_sentiment_transformer(text, tokenizer, model):
    """Predict sentiment using the transformer model"""
    if tokenizer is None or model is None:
        return "Model not loaded", 0.5
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred].item()
    
    return "Positive" if pred == 1 else "Negative", confidence

def predict_emotion(text, analyzer):
    """Predict emotion using VADER sentiment analyzer"""
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        return "Happy 😊", compound
    elif compound <= -0.05:
        return "Angry 😡", abs(compound)
    else:
        return "Neutral 😐", 0.5

def predict_aspect(text):
    """Predict aspect (Food, Service, Ambience) using keyword matching"""
    text = text.lower()
    food_keywords = ['food', 'dish', 'meal', 'taste', 'flavor', 'chef', 'delicious', 
                     'appetizer', 'dessert', 'menu', 'cuisine', 'ingredient']
    service_keywords = ['service', 'staff', 'waiter', 'waitress', 'server', 'customer', 
                        'manager', 'hostess', 'reservation', 'wait', 'attentive']
    ambience_keywords = ['ambience', 'atmosphere', 'decor', 'interior', 'music', 'lighting',
                         'table', 'chair', 'comfortable', 'noisy', 'quiet', 'romantic']
    
    food_count = sum(word in text for word in food_keywords)
    service_count = sum(word in text for word in service_keywords)
    ambience_count = sum(word in text for word in ambience_keywords)
    
    max_count = max(food_count, service_count, ambience_count)
    
    if max_count == 0:
        return "General", 0.33
    
    if max_count == food_count:
        return "Food", food_count / (food_count + service_count + ambience_count)
    elif max_count == service_count:
        return "Service", service_count / (food_count + service_count + ambience_count)
    else:
        return "Ambience", ambience_count / (food_count + service_count + ambience_count)

def generate_wordcloud(text):
    """Generate a wordcloud from text"""
    if not text:
        return None
    
    wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
    return wc

def perform_topic_modeling(reviews, num_topics=3):
    """Perform LDA topic modeling on a list of reviews"""
    # Preprocess the reviews
    processed_reviews = [preprocess_text(review).split() for review in reviews if isinstance(review, str)]
    
    if not processed_reviews:
        return [], []
    
    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(processed_reviews)
    corpus = [dictionary.doc2bow(review) for review in processed_reviews]
    
    # Train LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        alpha='auto',
        random_state=42
    )
    
    # Get topics
    topics = []
    for topic_id in range(num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=5)
        topics.append([word for word, _ in topic_words])
    
    # Get topic distribution for each document
    topic_distribution = [lda_model[doc] for doc in corpus]
    
    return topics, topic_distribution

def create_sidebar():
    """Create the sidebar navigation"""
    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Single Review Analysis", "Batch Analysis", "About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Model Settings")
    model_weight = st.sidebar.slider(
        "ML vs. Transformer Weight",
        0.0, 1.0, 0.5,
        help="0 = ML only, 1 = Transformer only"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📚 Resources")
    st.sidebar.markdown("[GitHub Repository](https://github.com/yourusername/restaurant-sentiment)")
    st.sidebar.markdown("[Report an Issue](https://github.com/yourusername/restaurant-sentiment/issues)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔄 Models Status")
    
    return page, model_weight

def render_home_page():
    """Render the home page"""
    st.markdown("<h1 class='main-header'>🍽️ Restaurant Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
    
    # Project description
    st.markdown("""
    <div class='highlight'>
    <h2 class='sub-header'>📖 Project Overview</h2>
    <p>Welcome to the Advanced Restaurant Review Sentiment Analyzer! This tool uses advanced machine learning and deep learning techniques to analyze customer feedback for restaurants.</p>
    <p>Unlike traditional sentiment analysis, this model goes beyond basic polarity detection and introduces aspect-based analysis, emotion detection, real-time sentiment visualization, and topic modeling to provide deeper insights into customer reviews.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='highlight'>
        <h2 class='sub-header'>🚀 Features</h2>
        <ul>
            <li><strong>Hybrid Sentiment Analysis</strong> (ML + Deep Learning)</li>
            <li><strong>Emotion Detection</strong> (Happy, Angry, Neutral)</li>
            <li><strong>Aspect-Based Analysis</strong> (Food, Service, Ambience)</li>
            <li><strong>Topic Modeling</strong> using LDA</li>
            <li><strong>Real-time Visualization</strong> with charts and wordclouds</li>
            <li><strong>Batch Processing</strong> for large datasets</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='highlight'>
        <h2 class='sub-header'>🛠️ Tech Stack</h2>
        <ul>
            <li><strong>Python</strong> 🐍 - Core programming language</li>
            <li><strong>Scikit-learn</strong> - For traditional ML models</li>
            <li><strong>Transformers (BERT)</strong> - For deep learning analysis</li>
            <li><strong>VADER</strong> - For emotion detection</li>
            <li><strong>Gensim</strong> - For topic modeling</li>
            <li><strong>Streamlit</strong> 🚀 - For interactive dashboard</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Get started section
    st.markdown("""
    <div class='highlight'>
    <h2 class='sub-header'>🚀 Get Started</h2>
    <p>Choose a page from the sidebar to begin analyzing restaurant reviews:</p>
    <ul>
        <li><strong>Single Review Analysis</strong> - Analyze a single review in detail</li>
        <li><strong>Batch Analysis</strong> - Upload a CSV file with multiple reviews</li>
        <li><strong>About</strong> - Learn more about the project and methodology</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample reviews
    st.markdown("<h2 class='sub-header'>📝 Sample Reviews to Try</h2>", unsafe_allow_html=True)
    sample_reviews = [
        "The food was absolutely delicious and the service was excellent!",
        "Great ambience, but the food was mediocre at best.",
        "Terrible service, waited for an hour to get our food.",
        "The best dining experience I've had in years!"
    ]
    
    for i, review in enumerate(sample_reviews):
        if st.button(f"Try Sample {i+1}", key=f"sample_{i}"):
            st.session_state.sample_review = review
            st.session_state.page = "Single Review Analysis"
            st.experimental_rerun()

def render_single_review_page(ml_model, tokenizer, transformer_model, vader_analyzer, model_weight):
    """Render the single review analysis page"""
    st.markdown("<h1 class='main-header'>🔍 Single Review Analysis</h1>", unsafe_allow_html=True)
    
    # Input form
    st.markdown("<h2 class='sub-header'>📝 Enter a Restaurant Review</h2>", unsafe_allow_html=True)
    
    # Check if there's a sample review from the home page
    initial_text = ""
    if 'sample_review' in st.session_state:
        initial_text = st.session_state.sample_review
        st.session_state.sample_review = ""
    
    review_text = st.text_area(
        "Review Text",
        value=initial_text,
        height=150,
        help="Enter a restaurant review to analyze"
    )
    
    analyze_button = st.button("🔍 Analyze Review")
    
    if analyze_button and review_text:
        with st.spinner("Analyzing review..."):
            # Add a slight delay to show the spinner
            time.sleep(0.5)
            
            # ML model prediction
            ml_sentiment, ml_confidence = predict_sentiment_ml(review_text, ml_model)
            
            # Transformer model prediction
            transformer_sentiment, transformer_confidence = predict_sentiment_transformer(
                review_text, tokenizer, transformer_model
            )
            
            # Combined prediction
            combined_sentiment = transformer_sentiment if transformer_confidence > ml_confidence else ml_sentiment
            combined_confidence = model_weight * transformer_confidence + (1 - model_weight) * ml_confidence
            
            # Emotion prediction
            emotion, emotion_confidence = predict_emotion(review_text, vader_analyzer)
            
            # Aspect prediction
            aspect, aspect_confidence = predict_aspect(review_text)
            
            # Display results
            st.markdown("<h2 class='sub-header'>📊 Analysis Results</h2>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            # Sentiment card
            with col1:
                st.markdown("""
                <div class='metric-card'>
                <h3>Overall Sentiment</h3>
                """, unsafe_allow_html=True)
                
                sentiment_class = "positive" if combined_sentiment == "Positive" else "negative"
                st.markdown(f"""
                <h2 class='{sentiment_class}'>{combined_sentiment}</h2>
                <p>Confidence: {combined_confidence:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Emotion card
            with col2:
                st.markdown("""
                <div class='metric-card'>
                <h3>Detected Emotion</h3>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <h2>{emotion}</h2>
                <p>Confidence: {emotion_confidence:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Aspect card
            with col3:
                st.markdown("""
                <div class='metric-card'>
                <h3>Primary Aspect</h3>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <h2>{aspect}</h2>
                <p>Confidence: {aspect_confidence:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Model comparison
            st.markdown("<h3>🤖 Model Comparison</h3>", unsafe_allow_html=True)
            
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                ml_color = "green" if ml_sentiment == "Positive" else "red"
                st.markdown(f"""
                <div class='metric-card'>
                <h4>Traditional ML Model</h4>
                <p>Prediction: <span style='color:{ml_color};'>{ml_sentiment}</span></p>
                <p>Confidence: {ml_confidence:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with comp_col2:
                transformer_color = "green" if transformer_sentiment == "Positive" else "red"
                st.markdown(f"""
                <div class='metric-card'>
                <h4>Transformer Model (BERT)</h4>
                <p>Prediction: <span style='color:{transformer_color};'>{transformer_sentiment}</span></p>
                <p>Confidence: {transformer_confidence:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("<h3>📊 Visualizations</h3>", unsafe_allow_html=True)
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Word cloud
                st.markdown("<h4>Word Cloud</h4>", unsafe_allow_html=True)
                wc = generate_wordcloud(review_text)
                if wc:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("Unable to generate word cloud for this review.")
            
            with viz_col2:
                # Sentiment breakdown
                st.markdown("<h4>Sentiment Analysis Breakdown</h4>", unsafe_allow_html=True)
                
                # Polarity scores from VADER
                scores = vader_analyzer.polarity_scores(review_text)
                
                # Create a bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Positive', 'Neutral', 'Negative'],
                    y=[scores['pos'], scores['neu'], scores['neg']],
                    marker_color=['#4CAF50', '#FFC107', '#F44336']
                ))
                
                fig.update_layout(
                    title="VADER Sentiment Scores",
                    xaxis_title="Sentiment",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1]),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Key Insights
            st.markdown("<h3>🔑 Key Insights</h3>", unsafe_allow_html=True)
            
            # Generate some insights based on the analysis
            insights = []
            
            # Sentiment insight
            if combined_sentiment == "Positive":
                insights.append(f"This is a <span class='positive'>positive review</span> with {combined_confidence:.0%} confidence.")
            else:
                insights.append(f"This is a <span class='negative'>negative review</span> with {combined_confidence:.0%} confidence.")
            
            # Emotion insight
            if emotion == "Happy 😊":
                insights.append("The customer appears to be <span class='positive'>happy</span> with their experience.")
            elif emotion == "Angry 😡":
                insights.append("The customer appears to be <span class='negative'>upset</span> with their experience.")
            else:
                insights.append("The customer has <span class='neutral'>mixed feelings</span> about their experience.")
            
            # Aspect insight
            insights.append(f"The review primarily focuses on the restaurant's <strong>{aspect.lower()}</strong>.")
            
            # Display insights
            for insight in insights:
                st.markdown(f"<div class='highlight'>{insight}</div>", unsafe_allow_html=True)
    
    else:
        if not review_text and analyze_button:
            st.warning("Please enter a review to analyze.")

def render_batch_analysis_page(ml_model, tokenizer, transformer_model, vader_analyzer, model_weight):
    """Render the batch analysis page"""
    st.markdown("<h1 class='main-header'>📊 Batch Review Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='highlight'>Upload a CSV file containing restaurant reviews to analyze multiple reviews at once.</div>", unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load the data
            df = pd.read_csv(uploaded_file)
            
            # Check if the CSV has the required column
            if 'review' not in df.columns:
                st.error("The CSV file must contain a 'review' column.")
                return
            
            st.success(f"Successfully loaded {len(df)} reviews.")
            
            # Display a sample of the data
            st.markdown("<h3>📝 Sample Reviews</h3>", unsafe_allow_html=True)
            st.dataframe(df.head())
            
            # Analyze button
            if st.button("📊 Analyze All Reviews"):
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Results DataFrame
                results = []
                
                # Analyze each review
                for i, row in enumerate(df.itertuples()):
                    review = row.review if hasattr(row, 'review') else ""
                    
                    if isinstance(review, str) and review.strip():
                        # ML model prediction
                        ml_sentiment, ml_confidence = predict_sentiment_ml(review, ml_model)
                        
                        # Transformer model prediction
                        transformer_sentiment, transformer_confidence = predict_sentiment_transformer(
                            review, tokenizer, transformer_model
                        )
                        
                        # Combined prediction
                        combined_sentiment = transformer_sentiment if transformer_confidence > ml_confidence else ml_sentiment
                        combined_confidence = model_weight * transformer_confidence + (1 - model_weight) * ml_confidence
                        
                        # Emotion prediction
                        emotion, _ = predict_emotion(review, vader_analyzer)
                        
                        # Aspect prediction
                        aspect, _ = predict_aspect(review)
                        
                        # Add to results
                        results.append({
                            'review': review,
                            'sentiment': combined_sentiment,
                            'confidence': combined_confidence,
                            'emotion': emotion,
                            'aspect': aspect
                        })
                    
                    # Update progress
                    progress = (i + 1) / len(df)
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzed {i+1} of {len(df)} reviews...")
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                # Convert results to DataFrame
                results_df = pd.DataFrame(results)
                
                # Add results to the original DataFrame
                for col in ['sentiment', 'confidence', 'emotion', 'aspect']:
                    if col in results_df.columns:
                        df[col] = results_df[col]
                
                # Display results
                st.markdown("<h2 class='sub-header'>📊 Analysis Results</h2>", unsafe_allow_html=True)
                
                # Display the full results
                st.dataframe(df)
                
                # Download link
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_analysis_results.csv">Download Results as CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Visualizations
                st.markdown("<h2 class='sub-header'>📈 Insights</h2>", unsafe_allow_html=True)
                
                # Sentiment distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h3>Sentiment Distribution</h3>", unsafe_allow_html=True)
                    sentiment_counts = df['sentiment'].value_counts()
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        color=sentiment_counts.index,
                        color_discrete_map={'Positive': '#4CAF50', 'Negative': '#F44336'},
                        title="Review Sentiment Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("<h3>Emotion Distribution</h3>", unsafe_allow_html=True)
                    emotion_counts = df['emotion'].value_counts()
                    fig = px.pie(
                        values=emotion_counts.values,
                        names=emotion_counts.index,
                        title="Emotion Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Aspect Analysis
                st.markdown("<h3>Aspect Analysis</h3>", unsafe_allow_html=True)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    aspect_counts = df['aspect'].value_counts()
                    fig = px.bar(
                        x=aspect_counts.index,
                        y=aspect_counts.values,
                        color=aspect_counts.index,
                        title="Review Aspects"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col4:
                    # Aspect vs Sentiment
                    aspect_sentiment = pd.crosstab(df['aspect'], df['sentiment'])
                    fig = px.bar(
                        aspect_sentiment,
                        title="Sentiment by Aspect",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Topic Modeling
                st.markdown("<h3>Topic Modeling</h3>", unsafe_allow_html=True)
                
                # Perform topic modeling
                topics, _ = perform_topic_modeling(df['review'].tolist(), num_topics=3)
                
                if topics:
                    col5, col6, col7 = st.columns(3)
                    
                    for i, (topic_words, col) in enumerate(zip(topics, [col5, col6, col7])):
                        with col:
                            st.markdown(f"<h4>Topic {i+1}</h4>", unsafe_allow_html=True)
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            for word in topic_words:
                                st.markdown(f"- {word}", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                
                # Word Cloud of all reviews
                st.markdown("<h3>Word Cloud of All Reviews</h3>", unsafe_allow_html=True)
