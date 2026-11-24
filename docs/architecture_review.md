# Architecture and Implementation Review

## Scope

This review evaluates the current implementation of the Matrix Factorization Movie Recommender System using the actual source files in the project. It does not propose or apply code changes.

The project has these main stages:

1. Load MovieLens ratings, users, and movies.
2. Split ratings into training and test sets.
3. Build and save a normalized training User-Item Matrix.
4. Train SVD and PMF.
5. Evaluate both models using the same test ratings.
6. Generate recommendation reports and comparison visualizations.
7. Present results through Streamlit.

## Executive Summary

The current implementation is mathematically valid and satisfies the main project requirements. The principal issue is architectural repetition: several functions independently split the ratings dataset and load the same files.

The repeated splitting is currently deterministic because every call uses `random_state=42`, but a centralized split is safer and easier to audit. It also ensures that SVD, PMF, and visualization cannot accidentally evaluate different test sets.

## 1. Data Splitting Strategy

### Existing implementation

The split is performed with:

```python
train_test_split(
    ratings,
    test_size=0.2,
    random_state=42,
)
```

It is called in:

- `utils/matrix_creation.py` in `prepare_matrix_data()`.
- `models/svd_model.py` in `run_svd()`.
- `models/pmf_model.py` in `run_pmf()`.
- `utils/model_visualization.py` in `generate_model_visualizations()`.

### Why `prepare_matrix_data()` splits

The split is used to create a training-only matrix:

```python
train_data, test_data = split_data(ratings)
user_movie_matrix = create_user_movie_matrix(
    train_data,
    all_users,
    all_movies,
)
```

This prevents test ratings from entering `processed/user_item_matrix.csv`.

The test DataFrame is returned from `prepare_matrix_data()`, but the current model scripts and pipeline do not use that returned test split. They create their own split instead.

### Why `run_svd()` splits again

SVD must have:

- `train_ratings` to calculate training user means.
- `test_ratings` to calculate held-out RMSE.

It loads the saved centered matrix with `load_matrix()`, reloads raw ratings with `load_ratings()`, and recreates the fixed split.

### Why `run_pmf()` splits again

PMF uses:

- `train_ratings` for latent-factor SGD updates.
- `test_ratings` for held-out RMSE.

It does not use the saved dense matrix for factor updates. It derives `user_ids` and `movie_ids` from the matrix dimensions, but trains from observed training triples.

### Why visualization splits again

Visualization needs test ratings to:

- calculate SVD RMSE.
- calculate PMF RMSE.
- generate predicted-versus-actual scatter plots.

The train DataFrame is unused in the visualization function. Only `test_ratings` is required there.

### Is repeated splitting unnecessary?

It is redundant architecturally, but not automatically incorrect.

Because all calls use the same loader, split implementation, row order, test size, and random seed, they should produce the same split.

Repeated splitting may become inconsistent if:

- ratings are loaded in a different order;
- the split implementation changes;
- preprocessing changes before one component runs;
- a model is given a previously saved or filtered ratings DataFrame;
- the main pipeline is extended without preserving the same split.

### Does `random_state=42` guarantee the same split?

It guarantees reproducibility for the same input DataFrame passed to the same deterministic `train_test_split()` implementation.

It does not guarantee identical output if a component loads a different or modified DataFrame. It also does not make a shared split an explicit architectural contract.

### Better approach

A centralized pipeline should split once:

```text
load ratings
    -> validate ratings
    -> split once with random_state=42
    -> build the training matrix
    -> train SVD with train/test pair
    -> train PMF with train/test pair
    -> evaluate and visualize with test pair
```

This is preferable because it makes the exact split explicit and reuses it across all components.

## 2. Matrix Creation

### Why the matrix uses `train_data`

The saved matrix represents information available during training.

Using `train_data` prevents test ratings from becoming visible to SVD during factorization. Test ratings are retained separately for later RMSE evaluation.

The matrix still contains all known users and movies:

```python
matrix.reindex(
    index=all_users,
    columns=all_movies,
)
```

Therefore, the dimensions are complete, while values outside the training observations become missing and are filled with zero deviations.

### Why the complete ratings dataset is not used

Using the complete dataset in the saved training matrix would include test observations. This could produce data leakage and make the test RMSE optimistic.

A complete observed matrix is suitable as a preprocessing artifact or as the input to a final model retrained after evaluation. It is not suitable for the matrix used during held-out testing unless test ratings are guaranteed to be excluded from model updates.

### Why test ratings remain a DataFrame

Test ratings are naturally represented as observed triples:

```text
(UserID, MovieID, Rating)
```

A dense test matrix is unnecessary. The DataFrame directly supports exact test-pair lookup in `evaluate_rmse()` and predicted-versus-actual plots.

### Data-leakage verdict

The current matrix preparation is mathematically consistent with preventing test leakage:

- test ratings are not inserted into the training matrix;
- SVD is trained from the training matrix;
- PMF is trained from observed training triples;
- test ratings are used only for evaluation.

## 3. SVD Implementation

### Why SVD loads the saved matrix

`run_svd()` loads `processed/user_item_matrix.csv` because that file is the normalized training matrix:

```python
centered_matrix = load_matrix()
```

SVD receives this matrix as input to `scipy.sparse.linalg.svds()`.

### Why SVD reloads the original ratings

It reloads raw ratings to obtain:

- the training DataFrame for user means;
- the test DataFrame for RMSE.

The original ratings are not directly added to `svds()` for factorization.

### Why user means are calculated again

The saved matrix is centered. `svds()` reconstructs centered rating deviations:

$$
\hat R_{\text{centered}} = U_k \Sigma_k V_k^T
$$

The reconstruction must restore the rating-scale baseline:

$$
\hat R_{u,m} = \bar r_u + \hat R_{\text{centered},u,m}
$$

The means are derived from `train_ratings`, not `test_ratings`, so test ratings do not affect the prediction baseline.

The implementation clips the final result to `[1, 5]`.

### Why SVD needs test ratings

Test ratings provide the actual observed values for RMSE:

```python
rmse = evaluate_rmse(
    test_ratings,
    predictions,
    user_ids,
    movie_ids,
)
```

Test ratings are never passed to `svds()` or used to calculate user means.

### How `svds()` creates predictions

`svds()` returns:

- `user_factors`: $U$
- `singular_values`: $\Sigma$
- `item_factors_t`: $V^T$

Their reconstruction is:

```python
centered_predictions = (
    user_factors * singular_values
) @ item_factors_t
```

This produces a prediction for every user/item pair, after which user means and rating bounds are restored.

The implementation saves the full result as `reports/svd_predictions.npy`.

### Project consistency

SVD is consistent with the project requirements:

- uses `scipy.sparse.linalg.svds`;
- trains on the processed training matrix;
- calculates RMSE on test ratings;
- saves a full predicted NumPy matrix;
- saves SVD RMSE in `model_metrics.json`.

Missing training entries are represented as zero deviations. This is an introductory dense-matrix convention. It is valid for the project but not necessarily optimal in a production recommender.

## 4. PMF Implementation

### Why PMF uses `train_ratings`

PMF performs SGD over observed rating triples. It does not require an entry for every user/movie pair.

Using `train_ratings` avoids:

- iterating over 6040 × 3706 dense combinations;
- treating missing entries as zero-error observations;
- storing a separate centered matrix for PMF.

### Why PMF uses `global_mean`

The PMF implementation centers every training rating around the training mean:

```python
global_mean = float(ratings.mean())
centered_ratings = ratings - global_mean
```

The latent factors learn deviations:

$$
r_{u,m} \approx \mu + p_u^T q_m
$$

The global mean is a reasonable baseline before considering user and item latent preferences.

### Why PMF evaluates against `test_ratings`

After factor learning, the complete prediction matrix is reconstructed and compared against the held-out rows.

This is correct evaluation: test ratings are ground truth, while test rows are never used in factor updates.

### PMF correctness

The implementation is correct as a regularized matrix-factorization/PMF-style SGD implementation:

- random latent factors are initialized reproducibly;
- each observed training rating contributes an SGD gradient;
- user and item factors are updated;
- regularization reduces large factor values;
- factors do not receive updates from missing matrix entries;
- training MSE is logged per epoch;
- test RMSE is held out.

It is not a fully Bayesian probabilistic graphical model. It is a common PMF-style maximum-likelihood/regularized SGD implementation.

## 5. Model Comparison

### Same test ratings

SVD and PMF should receive exactly the same test rows because each call uses:

- `load_ratings()`;
- `test_size=0.2`;
- `random_state=42`;
- the same pandas/scikit-learn split implementation.

No test leakage was found in either model.

### Fairness

The comparison is fair in the following respects:

- both use the same test rows;
- both calculate RMSE using the same function;
- both use the same user/movie ID space;
- both save all-pair prediction matrices;
- both predict on the `[1, 5]` rating scale.

The algorithms differ by design:

- SVD uses a global low-rank decomposition of a centered dense matrix.
- PMF uses SGD over observed user/movie rating triples.

This difference is the intended experimental comparison.

### Does visualization need to split again?

Only if it is an independent function. In the current standalone design, yes: it loads ratings and recreates the test split.

In a centralized design, the visualization function should accept `test_ratings` instead of calling `split_data()`.

The current line:

```python
train_ratings, test_ratings = split_data(load_ratings())
```

should treat `train_ratings` as unused. Keeping or removing it has no effect on the visualization itself.

### Data-leakage verdict

No direct test leakage is visible in the current implementation.

Potential weakness is architectural rather than mathematical:

- the split is repeated rather than shared;
- all components rely on identical input ordering;
- report regeneration depends on the split remaining reproducible;
- the stored metrics are not tied to an explicit persisted split artifact.

## 6. Architecture Comparison

### Approach A: Current implementation

Advantages:

- modular standalone scripts;
- easy to run each stage separately;
- uses a deterministic seed;
- no direct test leakage;
- functional dashboard and report generation.

Disadvantages:

- repeated loading and splitting;
- one source of truth is missing;
- matrix preparation returns test data that model scripts do not consume;
- standalone model scripts can be run in a different order;
- pipeline logs do not prove the split object was shared;
- visualization recalculates metrics instead of consuming model outputs.

### Approach B: Centralized pipeline

Advantages:

- load and validate once;
- split once;
- same train/test objects passed everywhere;
- matrix preparation uses explicit training data;
- models evaluate against exactly the same test DataFrame;
- easier to test, cache, and reproduce;
- clear separation of training and evaluation code;
- lower risk of accidental future inconsistencies.

Recommended pipeline:

```text
ratings = load_ratings()
validate(ratings)

train_ratings, test_ratings = split_data(ratings)

matrix = create_user_movie_matrix(
    train_ratings,
    all_users,
    all_movies,
)
normalized_matrix = normalize(matrix)
save(normalized_matrix)

svd_result = run_svd(
    train_ratings,
    test_ratings,
)

pmf_result = run_pmf(
    train_ratings,
    test_ratings,
)

visualize(
    test_ratings,
    svd_result,
    pmf_result,
)
```

This redesign preserves the mathematics while making the software contract explicit.

## 7. Final Verdict

### Correct parts

- `random_state=42` provides reproducibility for a fixed input.
- The matrix dimensions include all original users and movies.
- The training matrix is built only from training ratings.
- Test ratings are excluded from SVD factorization.
- PMF trains from observed training triples.
- SVD restores user means and clips predictions.
- PMF uses a training global mean and factor regularization.
- Both models can be evaluated by the same RMSE function.
- Full prediction matrices and model factors are saved.
- Comparison plots and recommendation CSVs are generated.

### Unnecessary or redundant parts

- Repeated `split_data()` calls.
- Returning `test_data` from matrix preparation if no caller consumes it.
- Recalculating user means and the PMF split in separate entry points.
- The unused `train_ratings` variable in visualization.
- Re-evaluating metrics during visualization instead of consuming model outputs.

### Potential bugs or inconsistencies

- Independent splitting relies on loading data in exactly the same order.
- A future preprocessing change could change one split while others remain unchanged.
- Matrix IDs are critical: prediction arrays and the CSV must keep the same row/column ordering.
- Visualization can regenerate metrics that differ from the metrics JSON if model artifacts become stale.
- The pipeline has no persisted split identity or artifact checksum.

### Project requirement verdict

The current implementation satisfies the project instructions:

- MovieLens data is loaded and processed.
- A fixed 80/20 split is reproducible.
- A dense normalized training matrix is saved.
- SVD uses `scipy.sparse.linalg.svds`.
- PMF performs regularized factor training.
- Both prediction matrices are saved.
- PMF convergence and required comparison plots are generated.
- Recommendations and dashboard support are implemented.

### Recommendation

Keep the same training-matrix and held-out-test strategy. Centralize only the split and evaluation flow:

1. Load and validate ratings once.
2. Split once with `random_state=42`.
3. Build and save the training matrix.
4. Pass the same train/test objects to SVD and PMF.
5. Pass test predictions/ratings to visualization.
6. Record metrics once and use those artifacts consistently.

This is better software architecture. It does not require changing the mathematics of the current implementation.
