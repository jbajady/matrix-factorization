# Matrix Factorization Models: SVD and PMF

## 1. Introduction

Matrix Factorization is a technique used in recommendation systems to discover hidden patterns in user preferences and item characteristics.

In a movie recommendation system, we start with a **User-Item Rating Matrix**:

- Rows represent users.
- Columns represent movies.
- Values represent ratings given by users.
- Missing values represent movies that users have not rated.

The objective is to predict missing ratings and recommend movies that users are likely to enjoy.

This project explores two matrix factorization approaches:

1. Singular Value Decomposition (SVD)
2. Probabilistic Matrix Factorization (PMF)

Both approaches learn lower-dimensional representations of users and movies, called **latent factors**.

---

# 2. Singular Value Decomposition (SVD)

## 2.1 What is SVD?

Singular Value Decomposition is a mathematical technique that decomposes a matrix into three smaller matrix components.

Given a rating matrix \(R\):

\[
R = U \Sigma V^T
\]

Where:

- **U (User Matrix):** Represents users in a latent-factor space.
- **Σ (Singular Values):** Contains the importance of each latent dimension.
- **Vᵀ (Movie Matrix):** Represents movies in the same latent-factor space.

The decomposition allows us to represent the original matrix using a smaller number of dimensions.

## 2.2 How does SVD work?

The mathematical process can be understood through the following steps:

**Step 1: Analyze the original matrix**

SVD analyzes the relationships and patterns contained in the rating matrix.

**Step 2: Identify important directions**

The decomposition identifies latent directions that capture patterns in the data.

Each direction has an associated singular value.

A larger singular value indicates a direction that contributes more to representing the original matrix.

**Step 3: Select the number of latent factors (k)**

Instead of keeping every singular value, we select the top \(k\) largest singular values.

This produces a reduced representation:

\[
R \approx U_k \Sigma_k V_k^T
\]

Where:

- \(U_k\) contains the first \(k\) columns of U.
- \(\Sigma_k\) contains the \(k\) largest singular values.
- \(V_k^T\) contains the first \(k\) rows of Vᵀ.

**Step 4: Reconstruct the approximate matrix**

The three reduced components are multiplied together to produce an approximation of the original rating matrix.

This approximation captures the most important patterns while reducing dimensionality.

## 2.3 Understanding the parameter k

The parameter \(k\) controls the number of latent dimensions retained by the model.

For example:

- \(k = 10\): Retains 10 latent dimensions.
- \(k = 50\): Retains 50 latent dimensions.
- \(k = 100\): Retains 100 latent dimensions.

A smaller \(k\) reduces memory usage and computation but may lose important information.

A larger \(k\) retains more information but increases computational cost and may capture noise.

The optimal value should be selected through validation experiments.

## 2.4 SVD for recommendation

Once the matrix has been decomposed, we can approximate a user's rating for a movie.

The predicted rating is:

\[
\hat r_{ui} = p_u^T q_i
\]

Where:

- \(p_u\) represents the user latent vector.
- \(q_i\) represents the movie latent vector.
- The dot product estimates the user's preference for that movie.

Movies with higher predicted ratings can be recommended to the user.

## 2.5 Important consideration: Missing ratings

Traditional SVD operates on numerical matrices.

It does not automatically understand that a missing rating is unknown.

If missing values are replaced with zero, SVD treats those zeros as actual numerical observations.

This can introduce bias into the decomposition.

Therefore, missing-rating handling must be considered carefully when applying SVD to sparse recommendation datasets.

Possible approaches include:

- Filling missing values using a suitable baseline or imputation strategy.
- Using an appropriate sparse or incomplete-matrix factorization method.
- Using a missing-data-aware matrix factorization approach.

---

# 3. Probabilistic Matrix Factorization (PMF)

## 3.1 What is PMF?

Probabilistic Matrix Factorization is a machine learning approach that learns latent representations of users and items from observed ratings.

Unlike a conventional SVD decomposition, PMF directly learns user and movie latent vectors by optimizing a probabilistic model.

The rating matrix is approximated by:

\[
R \approx P Q^T
\]

Where:

- **P:** User latent-factor matrix.
- **Q:** Movie latent-factor matrix.

Each user and each movie is represented by a vector of \(k\) latent factors.

## 3.2 How does PMF work?

**Step 1: Initialize latent factors**

PMF creates initial latent vectors for users and movies.

For example, if \(k = 2\):

\[
p_u = [0.8, 0.3]
\]

\[
q_i = [0.5, 0.7]
\]

These values are initially learned from the training process rather than manually assigned.

**Step 2: Predict ratings**

PMF calculates the predicted rating using the dot product:

\[
\hat r_{ui} = p_u^T q_i
\]

For example:

\[
\hat r_{ui} = (0.8 \times 0.5) + (0.3 \times 0.7)
\]

\[
\hat r_{ui} = 0.61
\]

**Step 3: Calculate prediction error**

The predicted rating is compared with the actual observed rating.

\[
e_{ui} = r_{ui} - \hat r_{ui}
\]

For example, if the actual rating is 5 and the prediction is 4:

\[
e_{ui} = 5 - 4 = 1
\]

**Step 4: Update latent factors**

PMF uses optimization techniques such as Gradient Descent or Stochastic Gradient Descent (SGD) to update user and movie vectors.

The objective is to reduce the difference between predicted and observed ratings.

**Step 5: Repeat the process**

The model repeats prediction, error calculation, and parameter updates over multiple training iterations (epochs).

Over time, the learned latent vectors become better at predicting user preferences.

## 3.3 The probabilistic approach

PMF assumes that observed ratings can be modeled probabilistically.

A common formulation assumes Gaussian observation noise:

\[
p(R \mid P,Q,\sigma^2)
=
\prod_{(u,i)\in\Omega}
\mathcal{N}
\left(
r_{ui}\mid p_u^Tq_i,\sigma^2
\right)
\]

Where:

- \(\Omega\) is the set of observed user-item ratings.
- \(\sigma^2\) represents the observation noise variance.
- \(\mathcal{N}\) is the Gaussian probability distribution.

PMF also commonly places Gaussian priors on user and movie latent factors.

These priors encourage reasonable factor values and help prevent overfitting.

The resulting optimization objective is commonly expressed as:

\[
\mathcal{L} =
\frac{1}{2}
\sum_{(u,i)\in\Omega}
(r_{ui}-p_u^Tq_i)^2
+
\frac{\lambda_P}{2}\sum_u \|p_u\|^2
+
\frac{\lambda_Q}{2}\sum_i \|q_i\|^2
\]

The first term measures prediction error.

The other terms are regularization penalties that prevent latent factors from becoming excessively large.

## 3.4 Handling missing ratings

One important advantage of PMF for recommendation systems is that it can train directly on observed ratings.

It only calculates prediction errors for user-item pairs that have actual ratings.

Missing ratings are not treated as zero.

This makes PMF suitable for sparse recommendation datasets such as MovieLens.

---

# 4. SVD vs PMF

| Feature | SVD | PMF |
|---|---|---|
| Main approach | Mathematical decomposition | Probabilistic learning |
| Main components | U, Σ, Vᵀ | P, Q |
| Learning mechanism | Matrix decomposition | Optimization |
| Latent factors | Extracted through decomposition | Learned through training |
| Missing ratings | Requires careful handling in conventional SVD | Uses observed ratings directly |
| Optimization | Depends on the SVD algorithm | Commonly uses SGD or related methods |
| Main parameter | Number of components (k) | Number of factors (k) |
| Regularization | Depends on the implementation | Commonly included |

## 4.1 Main conceptual difference

SVD decomposes a matrix into mathematical components to identify its most important directions.

PMF learns user and movie representations by minimizing prediction error under a probabilistic model.

Both approaches aim to discover latent relationships between users and movies, but they use different learning mechanisms.

---

# 5. Model Evaluation

Both models should be evaluated using data that was not used during training.

The goal is to measure how accurately each model predicts unseen ratings.

## 5.1 Root Mean Squared Error (RMSE)

RMSE measures the average magnitude of prediction errors, with larger errors penalized more heavily.

\[
RMSE =
\sqrt{
\frac{1}{N}
\sum_{j=1}^{N}
(r_j-\hat r_j)^2
}
\]

A lower RMSE indicates better rating prediction accuracy.

## 5.2 Comparing different values of k

We can train models using different latent-factor counts.

For example:

- \(k = 10\)
- \(k = 20\)
- \(k = 50\)
- \(k = 100\)

The models can then be compared using validation RMSE.

The best value of \(k\) should balance prediction quality and computational efficiency.

## 5.3 Recommendation quality

Prediction accuracy is not the only important metric.

A recommendation system should also be evaluated based on whether it recommends relevant movies.

Additional evaluation metrics may include:

- Precision@K
- Recall@K
- MAP@K
- NDCG@K

These metrics measure the quality and ranking of recommended items.

---

# 6. Conclusion

SVD and PMF are two important approaches to matrix factorization in recommendation systems.

SVD provides a mathematical decomposition that reduces dimensionality and extracts important patterns from rating matrices.

PMF learns latent user and movie representations through probabilistic modeling and optimization, while naturally supporting training on observed ratings only.

In this project, comparing both approaches helps demonstrate how different matrix factorization techniques can learn user preferences and generate movie recommendations.

The final model selection should be based on validation performance, recommendation quality, and computational requirements.