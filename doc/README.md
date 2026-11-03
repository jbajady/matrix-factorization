# Project Documentation

This folder documents the MovieLens matrix-factorization recommender project.
The project currently includes a ratings-data loader and a script for creating
a normalized user-item matrix. The model, application, and notebook files are
empty in the current checkout, so model training, recommendations, and a user
interface are not currently runnable from those files.

## Setup

1. Use a Python installation compatible with the pinned packages in
   [`requirements.txt`](../requirements.txt).
2. From the repository root, create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, activate it with:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/` | MovieLens ratings, user, and movie data files and their dataset notes |
| `utils/data_loader.py` | Loads the `::`-delimited ratings, users, and movies files |
| `utils/data_split.py` | Randomly splits ratings into training and test sets (80/20, seed 42) |
| `utils/matrix_creation.py` | Creates and saves the normalized user-item matrix |
| `processed/` | Generated processed data, including `user_item_matrix.csv` |
| `models/` | Intended location for model code; the model files are empty currently |
| `reports/` | Existing model metrics, predictions, and visualization artifacts |
| `app.py` | Intended application entry point; empty currently |
| `Movie_Recommender_System.ipynb` | Intended notebook; empty currently |

## Create the user-item matrix

Run this command from the repository root:

```bash
python utils/matrix_creation.py
```

The script loads `data/ratings.dat`, partitions ratings with an 80/20
train/test split, and pivots the training data into a matrix indexed by user ID
and movie ID. It subtracts each user's mean training rating from that user's
ratings, then fills missing entries with `0` (zero deviation). The resulting
matrix is rounded to four decimal places, stored as `float32`, and written to
`processed/user_item_matrix.csv`.

The test split is created but is not used to evaluate a model by this script.
The saved matrix contains only the training portion of ratings.

## Dataset

The files in `data/` use the MovieLens `::`-separated format:

- `ratings.dat`: `UserID::MovieID::Rating::Timestamp`
- `users.dat`: `UserID::Gender::Age::Occupation::Zip-code`
- `movies.dat`: `MovieID::Title::Genres`

The dataset details and usage terms are in [`data/README`](../data/README).
In particular, review those terms before redistributing or using the data
commercially.

## models

### SVD (Singular Value Decomposition)


## Current status

The root [`README.md`](../README.md) includes an architecture outline, and
`reports/` contains generated result artifacts. However, the model modules,
application entry point, and notebook are empty in the current checkout. The
architecture outline should therefore be treated as a project overview, not as
a claim that those components can currently be run.
