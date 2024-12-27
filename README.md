# Movie Recommendation System: A Complete Guide for Beginners

## Table of Contents
1. [Basic Concepts](#basic-concepts)
2. [How Matrix Factorization Works](#how-matrix-factorization-works)
3. [Code Walkthrough](#code-walkthrough)
4. [Performance Improvements](#performance-improvements)
5. [Example Usage](#example-usage)

## Basic Concepts

### What is a Recommendation System?
Imagine Netflix trying to guess which movies you'll like. That's a recommendation system! It looks at:
- What movies you've watched and rated
- What other similar users have liked
- Patterns in everyone's viewing habits

### The Rating Matrix
We start with a big table (matrix) where:
- Each row is a user
- Each column is a movie
- Each cell contains a rating (1-5 stars) or is empty

Example:
```
         Titanic  Star Wars  Alien  Jaws
User 1     5         4        ?      3
User 2     ?         5        2      ?
User 3     3         ?        4      5
```
The '?' marks are what we're trying to predict!

### Why Matrix Factorization?
Think of it like this:
1. Every movie has certain features (action, romance, special effects, etc.)
2. Every user has certain preferences for these features
3. We can multiply these together to predict ratings!

## How Matrix Factorization Works

### The Math Made Simple
Let's break it down:

1. **User Features (Matrix P)**:
   ```
   User 1: [Likes Action: 0.8, Likes Romance: 0.3]
   User 2: [Likes Action: 0.2, Likes Romance: 0.9]
   ```

2. **Movie Features (Matrix Q)**:
   ```
   Titanic:    [Action Content: 0.1, Romance Content: 0.9]
   Star Wars:  [Action Content: 0.9, Romance Content: 0.2]
   ```

3. **Predicting a Rating**:
   ```python
   # For User 1 rating Titanic:
   Rating = (0.8 × 0.1) + (0.3 × 0.9) = 0.35
   ```

### Why It Works
- Users who like similar movie features will get similar recommendations
- Movies with similar features will be recommended to similar users
- The system learns these features automatically from the ratings!

## Detailed Code Walkthrough

### 1. Data Loading and Preparation (`load_and_prepare_data`)

Let's say we have this raw data in ratings.csv:
```
user_id  movie_id  rating  timestamp
1        50        4.0     981234567
1        172       3.0     981234568
2        50        5.0     981234569
```

The function processes it like this:

```python
def load_and_prepare_data(ratings_path='ratings.csv'):
    # 1. Read the CSV file
    ratings_df = pd.read_csv(ratings_path, sep='\t')
    
    # 2. Clean column names (remove extra spaces and tabs)
    ratings_df.columns = ratings_df.columns.str.strip().str.replace('\t', '')
    
    # 3. Create user and movie ID mappings
    user_ids = sorted(ratings_df['user_id'].unique())  # e.g., [1, 2, 3, ...]
    movie_ids = sorted(ratings_df['movie_id'].unique())  # e.g., [50, 172, ...]
    
    # Create dictionaries for quick lookup
    user_map = {id: i for i, id in enumerate(user_ids)}  # e.g., {1: 0, 2: 1, ...}
    movie_map = {id: i for i, id in enumerate(movie_ids)}  # e.g., {50: 0, 172: 1, ...}
    
    # 4. Create the rating matrix (R)
    R = np.zeros((len(user_ids), len(movie_ids)))  # All zeros initially
    
    # 5. Fill in the known ratings
    for _, row in ratings_df.iterrows():
        i = user_map[row['user_id']]      # Convert user_id to matrix index
        j = movie_map[row['movie_id']]    # Convert movie_id to matrix index
        R[i, j] = row['rating']           # Add rating to matrix
```

The result is matrix R:
```
     Movie0(50)  Movie1(172)
User0(1)  4.0         3.0
User1(2)  5.0         0.0     # 0.0 means no rating
```

### 2. Matrix Factorization Class (`ImprovedMF`)

#### 2.1 Initialization
```python
def __init__(self, R, K=100, learning_rate=0.005, reg=0.02, iterations=50):
    self.R = R          # Our rating matrix from above
    self.num_users, self.num_items = R.shape  # e.g., (2 users, 2 movies)
    self.K = K          # Number of latent features (e.g., 100)
    self.learning_rate = learning_rate  # How fast we learn
    self.reg = reg      # Regularization strength
    self.iterations = iterations  # How many times to iterate
```

#### 2.2 Training Setup
```python
def train(self):
    # 1. Initialize matrices using Xavier/Glorot method
    limit = np.sqrt(6 / (self.num_users + self.num_items))
    
    # User feature matrix (P)
    self.P = np.random.uniform(-limit, limit, (self.num_users, self.K))
    # For 2 users, K=3 (simplified), might look like:
    # User features
    # [[ 0.1  0.2  0.3]    # User 0's preferences for 3 features
    #  [-0.1  0.4  0.2]]   # User 1's preferences for 3 features
    
    # Movie feature matrix (Q)
    self.Q = np.random.uniform(-limit, limit, (self.num_items, self.K))
    # For 2 movies, K=3 (simplified), might look like:
    # Movie features
    # [[ 0.2  0.3  0.1]    # Movie 0's values for 3 features
    #  [ 0.4  0.1  0.3]]   # Movie 1's values for 3 features
    
    # 2. Initialize biases
    self.b_u = np.zeros(self.num_users)   # User bias, e.g., [0.0, 0.0]
    self.b_i = np.zeros(self.num_items)   # Movie bias, e.g., [0.0, 0.0]
    self.b = np.mean(self.R[np.where(self.R != 0)])  # Global average rating
```

#### 2.3 The Training Process

Let's follow one complete training step:

```python
# 1. Get one known rating
user_id = 0      # First user
movie_id = 1     # Second movie
actual_rating = 3.0  # From our example data

# 2. Calculate predicted rating
# Step 2.1: Get dot product of user and movie features
user_features = self.P[user_id]    # e.g., [0.1, 0.2, 0.3]
movie_features = self.Q[movie_id]  # e.g., [0.4, 0.1, 0.3]
dot_product = np.dot(user_features, movie_features)
# dot_product = (0.1 × 0.4) + (0.2 × 0.1) + (0.3 × 0.3) = 0.17

# Step 2.2: Add biases
predicted_rating = (self.b +                  # Global average (e.g., 3.5)
                   self.b_u[user_id] +        # User bias (e.g., 0.2)
                   self.b_i[movie_id] +       # Movie bias (e.g., -0.1)
                   dot_product)               # Features dot product (0.17)
# predicted_rating = 3.5 + 0.2 + (-0.1) + 0.17 = 3.77

# 3. Calculate error
error = actual_rating - predicted_rating  # 3.0 - 3.77 = -0.77

# 4. Update parameters
# 4.1 Update biases
self.b_u[user_id] += self.learning_rate * (error - self.reg * self.b_u[user_id])
# If learning_rate = 0.005, reg = 0.02:
# New user_bias = 0.2 + 0.005 * (-0.77 - 0.02 * 0.2) = 0.196

self.b_i[movie_id] += self.learning_rate * (error - self.reg * self.b_i[movie_id])
# New movie_bias = -0.1 + 0.005 * (-0.77 - 0.02 * -0.1) = -0.104

# 4.2 Update feature matrices
# Update user features
self.P[user_id] += self.learning_rate * (error * self.Q[movie_id] - self.reg * self.P[user_id])
# For first feature:
# 0.1 + 0.005 * (-0.77 * 0.4 - 0.02 * 0.1) = 0.098

# Update movie features
self.Q[movie_id] += self.learning_rate * (error * self.P[user_id] - self.reg * self.Q[movie_id])
# For first feature:
# 0.4 + 0.005 * (-0.77 * 0.1 - 0.02 * 0.4) = 0.396
```

### 3. Early Stopping Implementation

```python
# Track best model performance
best_val_rmse = float('inf')
no_improvement = 0

for iteration in range(self.iterations):
    # Train one epoch...
    
    # Calculate validation RMSE
    val_rmse = self.calculate_rmse(val_data)
    
    # Check if we improved
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        no_improvement = 0
    else:
        no_improvement += 1
    
    # Stop if no improvement for 3 iterations
    if no_improvement >= 3:
        print("Early stopping!")
        break
```

### 4. RMSE Calculation

```python
def calculate_rmse(self, data):
    errors = []
    for i, j, r in data:
        pred = self.get_rating(i, j)
        errors.append((r - pred) ** 2)
    
    return np.sqrt(np.mean(errors))

# Example:
# Actual ratings:  [4.0, 3.0, 5.0]
# Predictions:     [3.8, 3.2, 4.7]
# Squared errors:  [(4.0-3.8)², (3.0-3.2)², (5.0-4.7)²]
#                = [0.04, 0.04, 0.09]
# Mean:           0.057
# RMSE:          √0.057 = 0.238
```

### 5. Making Predictions

```python
def predict(self, user_id, movie_id):
    """Predict rating for a specific user-movie pair"""
    if user_id < self.num_users and movie_id < self.num_items:
        return self.get_rating(user_id, movie_id)
    return self.b  # Return global average for unknown users/movies

def get_rating(self, i, j):
    """Calculate rating using learned parameters"""
    # Example with numbers:
    # Global bias:     3.5
    # User bias:       0.2    (this user rates slightly higher)
    # Movie bias:     -0.1    (this movie gets slightly lower ratings)
    # Dot product:     0.17   (from user and movie features)
    # Final rating:    3.77
    return self.b + self.b_u[i] + self.b_i[j] + self.P[i].dot(self.Q[j])
```

## Deep Dive: Initialization Methods and Biases

### 1. Understanding Xavier/Glorot Initialization

#### What is it?
Xavier/Glorot initialization is a smart way to set initial values for our matrices that helps the model learn faster and better. Instead of using completely random values, we choose values that are likely to work well with our neural network-like structure.

#### The Math Behind It
```python
limit = np.sqrt(6 / (num_users + num_items))
```

Let's break this down with real numbers:
```python
# Example with 1000 users and 2000 movies:
num_users = 1000
num_items = 2000
limit = np.sqrt(6 / (1000 + 2000))
      = np.sqrt(6 / 3000)
      = np.sqrt(0.002)
      ≈ 0.0447
```

#### Why This Formula?
1. The `6` comes from the optimal variance (2/(fan_in + fan_out))
2. We divide by (num_users + num_items) to scale based on matrix size
3. Square root ensures values aren't too small

#### np.random.uniform Explained
```python
self.P = np.random.uniform(-limit, limit, (self.num_users, self.K))
```

Let's break down np.random.uniform:
1. **Parameters**:
   - `-limit`: Lower bound (e.g., -0.0447)
   - `limit`: Upper bound (e.g., 0.0447)
   - `(self.num_users, self.K)`: Shape of output matrix

2. **Example Output**:
```python
# With num_users=3, K=4, limit=0.0447
P = np.random.uniform(-0.0447, 0.0447, (3, 4))

# Might produce:
[[ 0.0123  -0.0332   0.0401  -0.0156]  # User 1's features
 [-0.0445   0.0201  -0.0078   0.0321]  # User 2's features
 [ 0.0067  -0.0234   0.0445  -0.0109]] # User 3's features
```

#### Why Not Simple Random Numbers?
Let's compare initialization methods:

1. **Bad: All Zeros**
```python
P = np.zeros((num_users, K))
# Result:
[[0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0]]
# Problem: Model can't learn because all gradients will be same
```

2. **Bad: Large Random Numbers**
```python
P = np.random.rand(num_users, K) * 10  # Values between 0 and 10
# Result:
[[7.2, 3.1, 9.4, 5.5],
 [2.8, 8.9, 1.2, 6.7],
 [4.5, 7.8, 2.3, 9.1]]
# Problem: Large values cause exploding gradients
```

3. **Good: Xavier/Glorot (Our Method)**
```python
limit = np.sqrt(6 / (num_users + num_items))
P = np.random.uniform(-limit, limit, (num_users, K))
# Result:
[[ 0.0123, -0.0332,  0.0401, -0.0156],
 [-0.0445,  0.0201, -0.0078,  0.0321],
 [ 0.0067, -0.0234,  0.0445, -0.0109]]
# Advantage: Values are just right for learning
```

### 2. Understanding Bias Initialization

#### What are Biases?
Biases help capture basic patterns in ratings that aren't related to movie-user interactions:

1. **Global Bias (self.b)**:
```python
self.b = np.mean(self.R[np.where(self.R != 0)])

# Example:
R = [[4, 3, 0],    # 0 means no rating
     [5, 0, 2],
     [0, 4, 3]]

# Calculate mean:
existing_ratings = [4, 3, 5, 2, 4, 3]
global_bias = sum(existing_ratings) / len(existing_ratings)
           = 21 / 6
           = 3.5
```

2. **User Bias (self.b_u)**:
```python
self.b_u = np.zeros(self.num_users)

# Initially:
b_u = [0.0, 0.0, 0.0]  # One per user

# After some training, might become:
b_u = [0.5, -0.3, 0.2]  # Meaning:
# - User 1 tends to rate 0.5 stars higher than average
# - User 2 tends to rate 0.3 stars lower than average
# - User 3 tends to rate 0.2 stars higher than average
```

3. **Movie Bias (self.b_i)**:
```python
self.b_i = np.zeros(self.num_items)

# Initially:
b_i = [0.0, 0.0, 0.0]  # One per movie

# After training, might become:
b_i = [0.8, -0.1, -0.4]  # Meaning:
# - Movie 1 typically gets rated 0.8 stars above average
# - Movie 2 typically gets rated 0.1 stars below average
# - Movie 3 typically gets rated 0.4 stars below average
```

#### Impact on Predictions

Let's see how biases affect a prediction:

```python
# Without biases:
rating = np.dot(user_features, movie_features)
# Example:
user_features = [0.1, 0.2, 0.3]
movie_features = [0.2, 0.3, 0.1]
rating = (0.1 × 0.2) + (0.2 × 0.3) + (0.3 × 0.1)
       = 0.02 + 0.06 + 0.03
       = 0.11  # Too low for a 1-5 rating scale!

# With biases:
rating = global_bias + user_bias + movie_bias + np.dot(user_features, movie_features)
# Example:
global_bias = 3.5    # Average rating across all users/movies
user_bias = 0.5      # This user rates higher than average
movie_bias = -0.1    # This movie gets slightly lower ratings
dot_product = 0.11   # From above calculation

rating = 3.5 + 0.5 + (-0.1) + 0.11
       = 4.01  # A much more reasonable prediction!
```

#### Why Initialize Biases to Zero?

1. **Start Neutral**:
   - Let the model learn biases from data
   - Don't make assumptions about users/movies

2. **Gradual Learning**:
   ```python
   # During training:
   error = actual_rating - predicted_rating
   self.b_u[user_id] += learning_rate * (error - reg * self.b_u[user_id])
   
   # Example:
   # If actual=4.0, predicted=3.5, learning_rate=0.005, reg=0.02:
   error = 4.0 - 3.5 = 0.5
   new_bias = 0 + 0.005 * (0.5 - 0.02 * 0)
            = 0.0025
   # Bias slowly moves in the right direction
   ```

3. **Regularization Effect**:
   - Starting at zero helps regularization work better
   - Prevents biases from growing too large too quickly

## Understanding the Impact of Each Component

### 1. Bias Terms
Without biases:
```
Prediction = User_Features · Movie_Features
           = 0.17
```

With biases:
```
Prediction = Global_Average + User_Bias + Movie_Bias + (User_Features · Movie_Features)
           = 3.5 + 0.2 + (-0.1) + 0.17
           = 3.77
```

### 2. Regularization
Without regularization:
```python
self.P[user_id] += learning_rate * error * self.Q[movie_id]
# Features might grow too large
```

With regularization:
```python
self.P[user_id] += learning_rate * (error * self.Q[movie_id] - reg * self.P[user_id])
# The reg * self.P[user_id] term pulls values back toward zero
```

### 3. Early Stopping
Example progression:
```
Iteration 1: Train RMSE = 1.2, Val RMSE = 1.3
Iteration 2: Train RMSE = 1.0, Val RMSE = 1.1
Iteration 3: Train RMSE = 0.9, Val RMSE = 1.0
Iteration 4: Train RMSE = 0.8, Val RMSE = 1.1  # Validation got worse
Iteration 5: Train RMSE = 0.7, Val RMSE = 1.2  # Even worse
Iteration 6: Train RMSE = 0.6, Val RMSE = 1.3  # Stop here!
```

## Tips for Optimal Performance

1. **Data Preprocessing**:
   - Remove users with very few ratings (less than 5)
   - Remove movies with very few ratings (less than 10)
   - This helps the model learn better patterns

2. **Hyperparameter Selection**:
   ```python
   K = 100              # Enough features to capture patterns
   learning_rate = 0.005  # Small enough to converge
   reg = 0.02           # Prevents overfitting
   iterations = 50      # Enough to learn, with early stopping
   ```

3. **Memory Usage**:
   For a dataset with:
   - 1000 users
   - 2000 movies
   - K = 100 features
   
   Memory needed:
   - P matrix: 1000 × 100 × 8 bytes = 800KB
   - Q matrix: 2000 × 100 × 8 bytes = 1.6MB
   - Biases: (1000 + 2000) × 8 bytes = 24KB
   - Total: ~2.4MB

## Common Issues and Solutions

1. **High RMSE**:
   - Increase K (more features)
   - Decrease regularization
   - Check for data quality issues

2. **Slow Training**:
   - Decrease K
   - Increase learning rate (carefully)
   - Use mini-batches

3. **Overfitting**:
   - Increase regularization
   - Decrease K
   - Add more training data

## Performance Improvements

### 1. Xavier/Glorot Initialization
- **What**: Smart way to start the matrices
- **Why**: Prevents training from getting stuck
- **Impact**: ~10% faster training

### 2. Bias Terms
- **What**: Account for user/movie rating tendencies
- **Why**: Some users always rate high/low
- **Impact**: Improves RMSE by ~5-10%

### 3. Early Stopping
- **What**: Stop when not improving
- **Why**: Prevents overfitting
- **Impact**: Better final predictions

### 4. Regularization
- **What**: Prevents extreme predictions
- **Why**: Makes model more general
- **Impact**: More stable predictions

## Example Usage

### Basic Use
```python
# Load your data
R = load_and_prepare_data()

# Create and train model
mf = ImprovedMF(R, K=100)
mf.train()

# Get predictions
rating = mf.predict(user_id=5, movie_id=10)
```

### Tips for Best Results
1. Choose K wisely:
   - Too small (K=10): Not enough detail
   - Too large (K=500): Too slow, might overfit
   - Just right (K=100): Good balance

2. Watch the validation RMSE:
   - If it starts going up, you're overfitting
   - Early stopping helps catch this

## Results
On MovieLens 1M dataset:
- Training RMSE: ~0.87
- Validation RMSE: ~0.89
- Better than baseline (average rating): ~1.0

## Requirements
```
numpy
pandas
matplotlib
scikit-learn
