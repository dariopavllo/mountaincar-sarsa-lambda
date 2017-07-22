# MountainCar with SARSA(λ)
This project implements an agent that solves the Mountain Car task using the SARSA(λ) algorithm, that is, a variant of the original SARSA algorithm that uses eligibility traces to improve convergence speed.
The Mountain Car problem is a standard benchmark for reinforcement learning (RL) algorithms, and poses some challenges due to the fact that the agent must be modelled with continuous states, whereas traditional RL techniques such as Q-Learning or SARSA require discrete states. In this project, continuous states have been mapped to discrete states using an approach known as function approximation.

# How to run
A Python notebook has been provided in `Playground.ipynb`, and provides three features:
* Run an experiment with many agents and plot the escape latency with respect to the number of trials. The implementation spawns multiple threads that run in parallel.
* Visualize the behaviour of the agent in real time, with an interactive GUI.
* Plot a vector field that shows the most likely action in a certain state.

You need a suitable Python 3 distribution with SciPy to run this project (e.g. Anaconda).

# Report
You can find a report in `Report.pdf` that describes how the algorithm works in detail, as well as the experimental results.

# Disclaimer
This project was realized as a course project during my studies at the École polytechnique fédérale de Lausanne (EPFL). The software is provided "as is", without warranty of any kind.