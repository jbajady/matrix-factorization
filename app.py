import json
from pathlib import Path

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.data_loader import load_movies, load_ratings
from utils.recommendation import PREDICTION_PATHS, generate_recommendations


PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_data
def load_dashboard_data():
    ratings = load_ratings()
    movies = load_movies()
    return ratings, movies


@st.cache_data
def load_model_metrics():
    metrics_path = PROJECT_ROOT / "reports" / "model_metrics.json"
    if not metrics_path.is_file():
        return {}
    with metrics_path.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide")
    st.title("Movie Recommendation Dashboard")
    st.write(
        "Explore your rating history and compare personalized recommendations "
        "from SVD and PMF."
    )

    missing_models = [
        model.upper()
        for model, path in PREDICTION_PATHS.items()
        if not path.is_file()
    ]
    if missing_models:
        st.error(
            "Prediction files are missing for "
            f"{', '.join(missing_models)}. Train both models before launching "
            "the recommendation dashboard."
        )
        st.code("python -m models.svd_model\npython -m models.pmf_model")
        return

    ratings, movies = load_dashboard_data()
    user_ids = sorted(ratings["UserID"].unique())
    if not user_ids:
        st.error("No users were found in the ratings data.")
        return

    selected_user = st.number_input(
        "User ID",
        min_value=int(user_ids[0]),
        max_value=int(user_ids[-1]),
        value=int(user_ids[0]),
        step=1,
        help=f"Choose a MovieLens user ID between {user_ids[0]} and {user_ids[-1]}.",
    )
    top_n = st.slider("Recommendations per model", min_value=5, max_value=20, value=10)

    user_history = ratings.loc[ratings["UserID"] == selected_user]
    if user_history.empty:
        st.error(f"User ID {selected_user} has no ratings in the dataset.")
        return

    st.subheader(f"Your highest-rated movies — user {selected_user}")
    top_rated = (
        user_history.sort_values(
            ["Rating", "Timestamp", "MovieID"],
            ascending=[False, False, True],
        )
        .merge(movies[["MovieID", "Title", "Genres"]], on="MovieID", how="left")
        .head(10)
    )
    st.dataframe(
        top_rated[["Title", "Genres", "Rating"]].rename(
            columns={"Rating": "Your rating"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    try:
        svd_recommendations = generate_recommendations(
            int(selected_user), "svd", top_n
        )
        pmf_recommendations = generate_recommendations(
            int(selected_user), "pmf", top_n
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(f"Could not generate recommendations: {error}")
        return

    st.subheader("Recommendations")
    svd_column, pmf_column = st.columns(2)
    with svd_column:
        st.markdown("#### SVD")
        st.dataframe(
            svd_recommendations.rename(
                columns={"PredictedRating": "Predicted rating"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    with pmf_column:
        st.markdown("#### PMF")
        st.dataframe(
            pmf_recommendations.rename(
                columns={"PredictedRating": "Predicted rating"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("SVD vs. PMF predictions")
    figure, axes = plt.subplots(
        1, 2, figsize=(15, max(5, top_n * 0.4)), sharex=True
    )
    for axis, model, recommendations, color in (
        (axes[0], "SVD", svd_recommendations, "#4C78A8"),
        (axes[1], "PMF", pmf_recommendations, "#F58518"),
    ):
        ranked = recommendations.sort_values("PredictedRating")
        axis.barh(ranked["Title"], ranked["PredictedRating"], color=color)
        axis.set(title=f"{model} top {len(ranked)}", xlabel="Predicted rating", xlim=(1, 5))
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)

    metrics = load_model_metrics()
    if metrics:
        st.subheader("Model evaluation")
        metric_columns = st.columns(3)
        for column, key, label in zip(
            metric_columns,
            ("SVD_RMSE", "PMF_RMSE", "PMF_vs_SVD_improvement_%"),
            ("SVD RMSE", "PMF RMSE", "PMF improvement"),
        ):
            if key in metrics:
                suffix = "%" if key.endswith("%") else ""
                column.metric(label, f"{float(metrics[key]):.3f}{suffix}")


if __name__ == "__main__":
    main()