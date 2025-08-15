# Federated Learning for Content Caching

This project implements a federated learning approach for content caching optimization using autoencoder-based models. The system uses TensorFlow Federated (TFF) to train models across distributed clients while maintaining data privacy.

## Project Overview

The project focuses on building an efficient content caching system using federated learning with the following key components:

1. Autoencoder-based model for content popularity prediction
2. Federated learning implementation using TensorFlow Federated
3. Various caching strategies comparison including:
   - Oracle
   - Autoencoder-based
   - Random
   - Modified ε-greedy
   - Most Frequently Used
   - Thompson Sampling

## Requirements

The project requires the following Python packages:
- tensorflow-federated
- nest_asyncio
- xlsxwriter
- jax (>= 0.4.9)
- cachetools (>= 4.2.1)
- tensorflow
- pandas
- numpy
- matplotlib
- scikit-learn

## Dataset

The project uses the MovieLens 1M dataset (`ratings_1m.csv`) which contains movie ratings from users. The dataset is processed to create:
- User-specific data splits (train/validation/test)
- Federated datasets for distributed learning
- Binary interaction matrices for content popularity prediction

## Architecture

### Autoencoder Model

The project implements an autoencoder with configurable hidden layer sizes (2, 10, and 100 neurons tested) with the following architecture:
- Input layer: movie interaction vector
- Hidden layer: dense layer with ReLU activation
- Output layer: reconstruction of the input vector

### Federated Learning Implementation

The federated learning system includes:
1. Server initialization and update functions
2. Client update mechanism with local optimization
3. Weight aggregation using Federated Averaging (FedAvg)
4. Iterative process for model training across rounds

## Caching Strategies

The project implements and compares various caching strategies:

1. **Oracle**: Optimal caching based on perfect knowledge
2. **Autoencoder-based**: Using the trained federated model predictions
3. **Random**: Random content selection
4. **Modified ε-greedy**: Combination of exploration and exploitation
5. **Most Frequently Used**: Based on historical content popularity
6. **Thompson Sampling**: Probabilistic approach to content selection

## Results

The project evaluates different aspects:

1. **Model Performance**
   - Loss and accuracy metrics across training rounds
   - Cache efficiency comparisons
   - Impact of hidden layer size on performance

2. **Cache Efficiency Comparison**
   - Comparison across different cache sizes (50-500)
   - Performance evaluation of different strategies
   - Analysis of autoencoder architectures with varying neuron counts

3. **Thompson Sampling Analysis**
   - Implementation with different cache sizes and rounds
   - Cache efficiency measurements
   - Comparison with other strategies

## Key Findings

1. Cache efficiency increases with cache size across all strategies
2. The autoencoder-based approach performs better than random and m-epsilon greedy strategies
3. Increasing the number of neurons in the autoencoder's hidden layer improves cache efficiency
4. Thompson sampling shows competitive performance with adaptive behavior

## Usage

1. Setup the environment and install required packages
2. Prepare the MovieLens dataset
3. Configure the federated learning parameters (clients, rounds)
4. Train the model using the federated learning implementation
5. Evaluate different caching strategies
6. Analyze results using the provided visualization tools

## Results Visualization

The project includes visualization tools for:
1. Cache efficiency vs. cache size comparisons
2. Performance analysis of different autoencoder architectures
3. Training metrics across federated learning rounds

Results are saved in an Excel file ('Results.xlsx') for detailed analysis.

## Future Work

Potential areas for improvement and extension:
1. Testing with larger datasets and more clients
2. Implementing additional caching strategies
3. Optimizing the autoencoder architecture
4. Exploring different federated learning algorithms
5. Implementing real-time adaptation mechanisms
