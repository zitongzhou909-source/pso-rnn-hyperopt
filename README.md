🧭 Overview

This project implements Particle Swarm Optimization (PSO) for tuning the hyperparameters of a Recurrent Neural Network (RNN) in time-series forecasting tasks.
The goal is to improve model performance through automated optimization of learning rate, hidden size, dropout, and other parameters.


⚙️ Features

* PSO-based hyperparameter optimization for RNN
* Validation and test evaluation using MAE and RMSE metrics
* Built-in data preprocessing and z-score normalization
* Full implementation in PyTorch
* Clear, modular code structure and detailed comments


📁 Project Structure

Currently, the project mainly contains pso_rnn1.py, which serves as the core script for PSO-based RNN hyperparameter optimization and time-series forecasting.
Future updates will focus on improving the PSO algorithm — including strategies such as adaptive inertia weight, neighborhood topology, and convergence control — to enhance optimization stability and accuracy.


📊 Example Results

| Method            |   MAE  |    RMSE |
| ----------------- | -------| ------- |
| PSO-RNN (Current) | 949.79 | 1098.05 |

Note: The results may vary slightly due to random initialization and hardware differences.


🧠 Key Concepts

Particle Swarm Optimization (PSO)
A population-based stochastic optimization technique inspired by bird flocking behavior. Each particle represents a candidate solution and iteratively updates its position based on its own experience and the swarm’s global best.

Recurrent Neural Network (RNN)
An architecture capable of modeling sequential dependencies. It takes past observations as inputs to predict future values, which makes it suitable for time-series forecasting.


🧩 Example Workflow

1. Split data into train and test sets (last 18 months as test).
2. Normalize the data (z-score).
3. Use PSO to search for optimal hyperparameters.
4. Train RNN with the best configuration.
5. Evaluate on test data (MAE, RMSE).


🧑‍💻 Author

Zhou Zitong,
Graduate Student, Nagoya University,
Research Focus: Data Science, Machine Learning, Optimization
