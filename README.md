# Netflix-Based Movie Recommendation System

## Overview
This project implements a **Netflix-based Movie Recommendation System** using a **Stacked Autoencoder**. The system is designed to provide **personalized movie recommendations** based on user preferences and historical ratings.

## Features
- Utilizes **Stacked Autoencoders** for collaborative filtering-based recommendations.
- Achieves **Root Mean Squared Error (RMSE)** of **0.9177 on the training set** and **0.9674 on the test set**.
- Captures complex data patterns for improved recommendation accuracy.
- Implements **unsupervised learning** techniques to extract latent user preferences.

## Dataset
The dataset consists of user-movie ratings, similar to the **Netflix Prize dataset**. It contains:
- **User IDs**: Unique identifiers for users.
- **Movie IDs**: Unique identifiers for movies.
- **Ratings**: Numerical ratings given by users to movies.

## Model Architecture
The system is built using a **Stacked Autoencoder**, a deep neural network architecture consisting of:
- **Encoder Layers**: Compress input data into a latent representation.
- **Bottleneck Layer**: Captures essential features.
- **Decoder Layers**: Reconstructs data from the latent representation.

## Technologies Used
- **Python**
- **TensorFlow / PyTorch** (for building the Autoencoder)
- **NumPy, Pandas** (for data preprocessing)
- **Matplotlib, Seaborn** (for visualization)
- **Scikit-learn** (for evaluation metrics)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/netflix-autoencoder.git
   cd netflix-autoencoder
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```sh
   python train_autoencoder.py
   ```

## Usage
- Train the model using the provided dataset.
- Use the trained model to predict movie ratings for users.
- Generate personalized recommendations based on predicted ratings.

## Evaluation Metrics
- **Root Mean Squared Error (RMSE)**: Used to measure model performance.
- **Loss Function**: Mean Squared Error (MSE) optimization.

## Future Improvements
- Integrate with real-world movie streaming data.
- Optimize model hyperparameters for better performance.
- Implement hybrid recommendation techniques (collaborative + content-based).

## Author
**Mihir**

For any questions or collaborations, feel free to reach out!

