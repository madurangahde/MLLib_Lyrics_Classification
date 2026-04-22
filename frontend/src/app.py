import streamlit as st
from inferencer import predict_genre
import plotly.graph_objects as go
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

st.set_page_config(page_title="Lyric Genre Predictor", page_icon="🎵", layout="wide")

model_path = "../model_stage4_merged_Trans_way_new"

spark = (
    SparkSession.builder.appName("Genre Prediction").master("local[*]").getOrCreate()
)
model = PipelineModel.load(model_path)

# Theme colors for consistent UI visuals.
BG_GRADIENT = "linear-gradient(145deg, #0b1020 0%, #131a30 45%, #1a223d 100%)"
CARD_BG = "rgba(23, 31, 54, 0.82)"
TEXT_DARK = "#e5e7eb"
ACCENT = "#ff8a5c"
ACCENT_ALT = "#4ea8de"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG_GRADIENT};
    }}
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }}
    .hero-card {{
        background: {CARD_BG};
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 10px 28px rgba(2, 6, 23, 0.35);
        margin-bottom: 1.25rem;
    }}
    .hero-title {{
        color: {TEXT_DARK};
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }}
    .hero-subtitle {{
        color: #cbd5e1;
        font-size: 1.0rem;
        margin-bottom: 0;
    }}
    .stTextArea textarea {{
        border-radius: 14px !important;
        border: 1px solid #334155 !important;
        background: #111827 !important;
        color: {TEXT_DARK} !important;
    }}
    .stButton > button {{
        border-radius: 12px;
        border: 0;
        background: linear-gradient(90deg, {ACCENT} 0%, #ffb347 100%);
        color: white;
        font-weight: 700;
        padding: 0.55rem 1rem;
        box-shadow: 0 6px 16px rgba(255, 122, 89, 0.28);
    }}
    .stButton > button:hover {{
        filter: brightness(0.96);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Lyric Genre Predictor</div>
        <p class="hero-subtitle">Paste a song lyric and get a genre prediction with confidence and probability breakdown.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Add a welcome message and instructions
# st.markdown("""
# ## Welcome to the Lyric Genre Predictor!

# Enter the lyrics of a song in the box below and click "Predict Genre" to see the predicted genre.

# ### Available Genres: Pop, Country, Blues, Rock, Jazz, Reggae, Hip-Hop, Classic

# """)

# Create a text area for lyrics input
lyrics = st.text_area("Enter the lyrics here:", height=300)

# Add a button to trigger the prediction
if st.button("Predict Genre"):
    if lyrics.strip() == "":
        st.error("Please enter some lyrics.")
    else:
        with st.spinner("Predicting..."):
            # Call the predict_genre function
            predicted_genre, predicted_prob, genre_prob_dict = predict_genre(
                spark, model, lyrics
            )

        # Display the predicted genre and confidence
        st.success(
            f"Predicted Genre: {predicted_genre} (Confidence: {predicted_prob:.2%})"
        )

        # Prepare data for visualizations
        genre_labels = [
            "pop",
            "country",
            "blues",
            "rock",
            "jazz",
            "reggae",
            "hip hop",
            "Alternative Rock",
        ]
        probabilities = [genre_prob_dict[i] for i in range(len(genre_labels))]
        # Find the index and value of the maximum probability
        max_prob_index = probabilities.index(max(probabilities))
        max_genre = genre_labels[max_prob_index]
        max_prob = probabilities[max_prob_index]
        color_palette = [
            "#ff7a59",
            "#f2c14e",
            "#6a9c89",
            "#4d96ff",
            "#8f6ed5",
            "#00a896",
            "#ef476f",
            "#2c7da0",
        ]

        # Create a donut chart with all genres and highlight the top prediction.
        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=genre_labels,
                    values=probabilities,
                    hole=0.45,
                    sort=False,
                    textinfo="label+percent",
                    textposition="inside",
                    marker=dict(
                        colors=color_palette,
                        line=dict(color="#ffffff", width=2),
                    ),
                    pull=[0.12 if i == max_prob_index else 0 for i in range(len(genre_labels))],
                    hovertemplate="<b>%{label}</b><br>Probability: %{percent}<extra></extra>",
                )
            ]
        )
        pie_fig.update_layout(
            title=f"Genre Probability Share (Top: {max_genre})",
            width=520,
            height=430,
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color=TEXT_DARK),
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0.5, xanchor="center"),
            margin=dict(t=70, b=65, l=20, r=20),
        )
        pie_fig.add_annotation(
            text=f"Top<br><b>{max_genre}</b><br>{max_prob:.1%}",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color=ACCENT_ALT),
        )

        # Create bar chart (Probability distribution for each genre)
        bar_fig = go.Figure(
            data=[
                go.Bar(
                    x=genre_labels,
                    y=probabilities,
                    text=[f"{p:.1%}" for p in probabilities],
                    textposition="outside",
                    marker_color=[
                        ACCENT if i == max_prob_index else ACCENT_ALT
                        for i in range(len(genre_labels))
                    ],
                )
            ]
        )
        bar_fig.update_layout(
            title="Probability Distribution by Genre",
            xaxis_title="Genre",
            yaxis_title="Probability",
            width=520,
            height=430,
            yaxis=dict(range=[0, max(0.1, max(probabilities) * 1.25)]),
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color=TEXT_DARK),
            margin=dict(t=70, b=40, l=30, r=30),
        )

        # Display visualizations side by side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(pie_fig, use_container_width=True)
        with col2:
            st.plotly_chart(bar_fig, use_container_width=True)

        result = predicted_genre
