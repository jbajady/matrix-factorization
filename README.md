# Matrix Factorization Movie Recommender

## Architecture

### This diagram shows how your files communicate with each other
```mermaid

flowchart TD
    A[MovieLens Dataset]

    A --> B[data_loader.py]
    B --> C[matrix_creation.py]

    C --> D[User-Item Matrix]

    D --> E[SVD Model]
    D --> F[PMF Model]

    E --> G[SVD Predictions]
    F --> H[PMF Predictions]

    G --> I[Model Evaluation]
    H --> I

    I --> J[RMSE Comparison]

    G --> K[Recommendation Engine]
    H --> K

    K --> L[Top Movie Recommendations]

    L --> M[app.py]

    I --> N[Reports]



```

### This diagram focuses on the machine learning pipeline.


```mermaid

flowchart TD
    A[Raw Dataset] --> B[Data Loading]
    B --> C[Data Cleaning]
    C --> D[Exploratory Data Analysis]
    D --> E[User-Item Matrix]

    E --> F[Train/Test Split]

    F --> G[SVD]
    F --> H[PMF]

    G --> I[Predicted Ratings]
    H --> J[Predicted Ratings]

    I --> K[Evaluation]
    J --> K

    K --> L[Calculate RMSE]

    L --> M[Compare Models]
    M --> N[Best Model]

    N --> O[Generate Recommendations]
```

## Preparing the user-item matrix

Run `python utils/matrix_creation.py` from the repository root to load the
MovieLens data, split ratings into training and test sets, and create a
user-item matrix from the training ratings. The matrix is centered by each
user's mean rating, and missing ratings are represented as `0` after
normalization. The result is saved to `processed/user_item_matrix.csv`.

See the [project documentation](doc/README.md) for setup instructions,
repository structure, data details, and the current workflow.
