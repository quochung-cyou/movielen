import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ImprovedMF():
    def __init__(self, R, K=100, learning_rate=0.005, reg=0.02, iterations=50):
        """
        Simple but effective Matrix Factorization
        
        Parameters:
        - R: rating matrix (users x movies)
        - K: number of latent factors (default: 100)
        - learning_rate: how fast to learn (default: 0.005)
        - reg: regularization to prevent overfitting (default: 0.02)
        - iterations: number of training iterations (default: 50)
        """
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.learning_rate = learning_rate
        self.reg = reg
        self.iterations = iterations
        
    def train(self):
        # Initialize matrices with reasonable values
        limit = np.sqrt(6 / (self.num_users + self.num_items))
        self.P = np.random.uniform(-limit, limit, (self.num_users, self.K))
        self.Q = np.random.uniform(-limit, limit, (self.num_items, self.K))
        
        # Initialize biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create training data
        train_data = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        # Split into train/validation
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
        
        print("Starting training...")
        training_process = []
        best_val_rmse = float('inf')
        no_improvement = 0
        
        for iteration in range(self.iterations):
            # Shuffle training data
            np.random.shuffle(train_data)
            
            # Train on each rating
            for i, j, r in train_data:
                # Calculate prediction and error
                pred = self.get_rating(i, j)
                error = r - pred
                
                # Update biases
                self.b_u[i] += self.learning_rate * (error - self.reg * self.b_u[i])
                self.b_i[j] += self.learning_rate * (error - self.reg * self.b_i[j])
                
                # Update user and movie matrices
                self.P[i] += self.learning_rate * (error * self.Q[j] - self.reg * self.P[i])
                self.Q[j] += self.learning_rate * (error * self.P[i] - self.reg * self.Q[j])
            
            # Calculate RMSE on train and validation
            train_rmse = self.calculate_rmse(train_data)
            val_rmse = self.calculate_rmse(val_data)
            training_process.append((train_rmse, val_rmse))
            
            # Early stopping check
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                no_improvement = 0
            else:
                no_improvement += 1
            
            if iteration % 5 == 0:
                print(f"Iteration {iteration}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")
            
            # Stop if no improvement for 3 iterations
            if no_improvement >= 3:
                print("Early stopping!")
                break
        
        return training_process
    
    def get_rating(self, i, j):
        """Predict rating for user i and movie j"""
        return self.b + self.b_u[i] + self.b_i[j] + self.P[i].dot(self.Q[j])
    
    def calculate_rmse(self, data):
        """Calculate RMSE for given data points"""
        errors = []
        for i, j, r in data:
            pred = self.get_rating(i, j)
            errors.append((r - pred) ** 2)
        return np.sqrt(np.mean(errors))
    
    def predict(self, user_id, movie_id):
        """Predict rating for a specific user-movie pair"""
        if user_id < self.num_users and movie_id < self.num_items:
            return self.get_rating(user_id, movie_id)
        return self.b  # Return global average for unknown users/movies

def load_and_prepare_data(ratings_path='ratings.csv', movies_path='movies.csv', users_path='users.csv'):
    """Load and prepare the MovieLens data"""
    print("Loading data...")
    
    try:
        # Read files with tab delimiter
        ratings_df = pd.read_csv(ratings_path, sep='\t')
        
        # Clean column names
        ratings_df.columns = ratings_df.columns.str.strip().str.replace('\t', '')
        
        # Create user-movie matrix
        user_ids = sorted(ratings_df['user_id'].unique())
        movie_ids = sorted(ratings_df['movie_id'].unique())
        
        # Create ID mappings
        user_map = {id: i for i, id in enumerate(user_ids)}
        movie_map = {id: i for i, id in enumerate(movie_ids)}
        
        # Create and fill the rating matrix
        R = np.zeros((len(user_ids), len(movie_ids)))
        for _, row in ratings_df.iterrows():
            i = user_map[row['user_id']]
            j = movie_map[row['movie_id']]
            R[i, j] = row['rating']
        
        print(f"Created matrix with {len(user_ids)} users and {len(movie_ids)} movies")
        return R, user_map, movie_map
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

if __name__ == "__main__":
    # Load data
    print("Starting MovieLens Matrix Factorization...")
    R, user_map, movie_map = load_and_prepare_data()
    
    # Train model
    mf = ImprovedMF(R)
    history = mf.train()
    
    # Plot results
    train_rmse, val_rmse = zip(*history)
    plt.figure(figsize=(10, 6))
    plt.plot(train_rmse, label='Training RMSE')
    plt.plot(val_rmse, label='Validation RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()
    
    # Print final performance
    print(f"\nFinal Training RMSE: {train_rmse[-1]:.4f}")
    print(f"Final Validation RMSE: {val_rmse[-1]:.4f}")
