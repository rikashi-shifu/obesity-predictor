import plotly.express as px
import pandas as pd


def create_probability_visualization(probabilities):
    """Interactive probability distribution chart"""
    prob_df = pd.DataFrame(
        list(probabilities.items()), columns=["Class", "Probability"]
    )
    prob_df = prob_df.sort_values("Probability", ascending=True)

    fig = px.bar(
        prob_df,
        x="Probability",
        y="Class",
        orientation="h",
        title="Obesity Level Prediction Probabilities",
        color="Probability",
        color_continuous_scale="viridis",
    )

    fig.update_traces(texttemplate="%{x:.1%}", textposition="outside")
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Probability",
        yaxis_title="Obesity Level",
    )

    return fig
