import streamlit as st
from inferencer import predict_genre
import plotly.graph_objects as go
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

model_path = "../model_stage4_merged"

spark = (
    SparkSession.builder.appName("Genre Prediction").master("local[*]").getOrCreate()
)
model = PipelineModel.load(model_path)

# Set the title of the app
st.title("Lyric Genre Predictor")

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

        # Prepare data for pie chart
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
        other_prob = sum(
            prob for i, prob in enumerate(probabilities) if i != max_prob_index
        )

        # Create pie chart (Maximum genre vs. Other)
        pie_labels = [max_genre, "Other"]
        pie_values = [max_prob, other_prob]
        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=pie_labels,
                    values=pie_values,
                    textinfo="label+percent",
                    marker=dict(colors=["#ff6384", "#d3d3d3"]),
                    pull=[0.1, 0],  # Pull out the max genre slice
                )
            ]
        )
        pie_fig.update_layout(title="Predicted Genre vs. Other", width=400, height=400)

        # Create bar chart (Probability distribution for each genre)
        bar_fig = go.Figure(
            data=[
                go.Bar(
                    x=genre_labels,
                    y=probabilities,
                    marker_color=[
                        "#ff6384" if i == max_prob_index else "#36a2eb"
                        for i in range(len(genre_labels))
                    ],
                )
            ]
        )
        bar_fig.update_layout(
            title="Probability Distribution by Genre",
            xaxis_title="Genre",
            yaxis_title="Probability",
            width=400,
            height=400,
        )

        # Display visualizations side by side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(pie_fig, use_container_width=True)
        with col2:
            st.plotly_chart(bar_fig, use_container_width=True)

        result = predicted_genre
