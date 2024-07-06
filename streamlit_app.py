!pip install streamlit
!pip install pandas
!pip install matplotlib
!pip install wordcloud
!pip install plotly
!pip install numpy

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objs as go
from collections import Counter
import numpy as np
import re

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Sentiment Analysis Dashboard',
    page_icon='🌍',  # Middle East world emoji
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def load_data():
    """Load the cleaned data from GitHub."""
    url = 'https://raw.githubusercontent.com/dnlaql/Sentiment-Analysis-miniproject/main/cleaned_combined_data_with_aspects_and_sentiments.csv'
    return pd.read_csv(url)

# Load data
combined_df = load_data()

# Convert 'createdAt' to datetime if it exists
if 'createdAt' in combined_df.columns:
    combined_df['createdAt'] = pd.to_datetime(combined_df['createdAt'])

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# 🌍 Sentiment Analysis Dashboard

Explore sentiment data related to global brand boycotts amid the Israel-Palestine conflict in late 2022. Use the filters to customize your view and gain insights.
'''

# Sidebar for user input
st.sidebar.header('Select Your Preferences:')
selected_sentiment = st.sidebar.selectbox('Sentiment', options=['All', 'POSITIVE', 'NEGATIVE'])
selected_aspect = st.sidebar.selectbox('Aspect', options=['All'] + combined_df['Aspect'].unique().tolist())
selected_date = st.sidebar.date_input('Date', [])
selected_tokens = st.sidebar.text_input('Tokens (comma-separated)', '')

# Reset Button
if st.sidebar.button('Reset'):
    st.experimental_rerun()

# Filter data based on selections
filtered_df = combined_df.copy()

if selected_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['Sentiment'] == selected_sentiment]

if selected_aspect != 'All':
    filtered_df = filtered_df[filtered_df['Aspect'] == selected_aspect]

if selected_date:
    filtered_df = filtered_df[filtered_df['createdAt'] == pd.to_datetime(selected_date)]

if selected_tokens:
    token_list = [token.strip() for token in selected_tokens.split(',')]
    filtered_df = filtered_df[filtered_df['tokens'].apply(lambda x: all(token in x for token in token_list))]

# Clean tokens by removing unwanted characters
def clean_tokens(tokens):
    cleaned = re.sub(r'[^\w\s]', '', tokens)  # Remove punctuation
    return cleaned

# Apply cleaning function to the 'lemmatized_tokens' column
combined_df['cleaned_tokens'] = combined_df['lemmatized_tokens'].apply(lambda x: clean_tokens(str(x)))
filtered_df['cleaned_tokens'] = filtered_df['lemmatized_tokens'].apply(lambda x: clean_tokens(str(x)))

# Display filtered data with interactive elements
st.markdown("### Text Preview")

# Add search functionality
search_term = st.text_input('Search by keyword:')
if search_term:
    filtered_df = filtered_df[filtered_df['text'].str.contains(search_term, case=False, na=False)]

# Get the number of records in the filtered dataset
max_records = len(filtered_df)

if max_records >= 5:
    num_records = st.slider('Number of records to display', min_value=5, max_value=max_records, value=min(10, max_records))
else:
    num_records = st.slider('Number of records to display', min_value=1, max_value=max_records, value=max_records)

st.dataframe(filtered_df[['text', 'Sentiment', 'Aspect']].head(num_records))

# Create a 3D WordCloud visualization using Plotly
st.markdown("### WordCloud Most Frequent Keyword")

if not filtered_df.empty:
    # Generate word frequencies from the cleaned tokens
    combined_words = ' '.join(combined_df['cleaned_tokens']).split()
    combined_word_freq = Counter(combined_words).most_common(50)
    max_freq = max(freq for word, freq in combined_word_freq)

    words = ' '.join(filtered_df['cleaned_tokens']).split()
    word_freq = Counter(words).most_common(50)

    # Prepare data for 3D scatter plot
    x = np.random.randn(len(word_freq))
    y = np.random.randn(len(word_freq))
    z = np.random.randn(len(word_freq))
    text = [word for word, freq in word_freq]
    size = [freq for word, freq in word_freq]

    # Determine the maximum size from the entire dataset
    font_sizes = [max(freq/max_freq*150, 30) for freq in size]  # Triple the size and ensure minimum font size for visibility

    # Create a colorscale
    colorscale = px.colors.sequential.Plasma

    # Normalize the frequency to match the colorscale
    norm_size = [freq / max(size) for freq in size]

    # Map the normalized frequency to the colorscale
    colors = [colorscale[int(norm * (len(colorscale) - 1))] for norm in norm_size]

    # Create 3D scatter plot with text annotations
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        text=text,
        textfont=dict(size=font_sizes, color=colors),
        marker=dict(
            size=font_sizes,
            sizemode='diameter',
            sizeref=max(size)/50,
            color=size,
            colorscale='Plasma',
            showscale=True
        )
    )])
    fig.update_layout(
        title="3D WordCloud",
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        )
    )

    st.plotly_chart(fig)
else:
    st.warning("No data available for the selected filters.")

# Plot sentiment distribution
st.markdown("### Total Sentiment")
if not filtered_df.empty:
    sentiment_count = filtered_df['Sentiment'].value_counts()
    fig = px.pie(values=sentiment_count.values, names=sentiment_count.index, title="Sentiment Distribution")
    st.plotly_chart(fig)
    
    total_count = sentiment_count.sum()
    st.markdown(f"<h2 style='text-align: center; color: white;'>Total Sentiment: {total_count}</h2>", unsafe_allow_html=True)
    
    positive_count = sentiment_count.get('POSITIVE', 0)
    negative_count = sentiment_count.get('NEGATIVE', 0)

    # Create sentiment atmosphere gauge
    sentiment_score = positive_count / total_count * 100 if total_count > 0 else 0

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = sentiment_score,
        title = {'text': "Sentiment Atmosphere"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 20], 'color': 'red'},
                {'range': [20, 40], 'color': 'orange'},
                {'range': [40, 60], 'color': 'yellow'},
                {'range': [60, 80], 'color': 'lightgreen'},
                {'range': [80, 100], 'color': 'green'}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_score}}))

    st.plotly_chart(fig)

else:
    st.warning("No data available for the selected filters.")

# Plot aspect distribution with sentiment
st.markdown("### Aspect Distribution with Sentiment")
if not filtered_df.empty and {'Aspect', 'Sentiment'}.issubset(filtered_df.columns):
    aspect_sentiment_count = filtered_df.groupby(['Aspect', 'Sentiment']).size().unstack().fillna(0)
    if not aspect_sentiment_count.empty:
        aspect_sentiment_count = aspect_sentiment_count.reset_index()
        if 'POSITIVE' not in aspect_sentiment_count.columns:
            aspect_sentiment_count['POSITIVE'] = 0
        if 'NEGATIVE' not in aspect_sentiment_count.columns:
            aspect_sentiment_count['NEGATIVE'] = 0
        aspect_sentiment_count = aspect_sentiment_count.melt(id_vars=['Aspect'], value_vars=['POSITIVE', 'NEGATIVE'], var_name='Sentiment', value_name='Count')
        
        fig = px.bar(aspect_sentiment_count, x='Aspect', y='Count', color='Sentiment', barmode='group', title="Aspect Distribution with Sentiment")
        st.plotly_chart(fig)
    else:
        st.warning("No data available for the selected filters.")
else:
    st.warning("No data available for the selected filters or time column is missing.")
