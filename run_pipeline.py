"""Rebuild all data artifacts, train both models, and generate reports."""

from utils.data_validation import validate_dataset
from utils.matrix_creation import prepare_matrix_data, save_user_item_matrix
from utils.model_visualization import generate_model_visualizations
from models.svd_model import run_svd
from models.pmf_model import run_pmf


def run_pipeline(user_id: int = 1) -> dict[str, float]:
    """Execute the complete recommender workflow in dependency order."""
    validate_dataset()

    normalized_matrix, _, _, _ = prepare_matrix_data()
    save_user_item_matrix(normalized_matrix)

    run_svd(k=50)
    run_pmf(n_factors=30, epochs=15)
    return generate_model_visualizations(user_id=user_id, top_n=10)


if __name__ == "__main__":
    run_pipeline()
