# 🎮 2048 Adversarial AI Agent

This project implements a high-performance AI agent for the game **2048**, framing gameplay as an **adversarial search problem** and leveraging classic techniques from AI research — including **expectiminimax**, **alpha-beta pruning**, and **Bayesian-tuned heuristics**.

Inspired by the paper [_"An AI Plays 2048"_ (Nie, Hou, An – Stanford CS229)](https://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf), this agent consistently reaches the 2048 tile and frequently achieves 4096+ scores across multiple runs, demonstrating intelligent planning under uncertainty and real-time constraints.

---

## 🧠 Core Techniques

- **Adversarial Search:** Models 2048 gameplay using **expectiminimax** to account for both agent actions and random tile placements (2 or 4).
- **Alpha-Beta Pruning:** Prunes branches of the search tree to optimize decision-making within a fixed time limit (~0.2 seconds per move).
- **Iterative Deepening:** Dynamically adjusts depth based on computation time to ensure best move selection within time constraints.
- **Heuristic Evaluation Function:** Combines multiple handcrafted features with Bayesian-optimized weights:
  - **Snake Pattern Adherence:** Encourages large tiles to cluster in a predefined optimal path.
  - **Smoothness:** Penalizes grids with large adjacent differences (harder to merge).
  - **Empty Cell Count:** Rewards open space for flexibility and avoiding deadlocks.
  - **Corner Control:** Rewards keeping the max tile in the top-left corner.

---

## 🧪 Performance

Sample run results (each group = 10 runs):

Group 1: [2048, 2048, 2048, 2048, 2048]
Group 2: [2048, 2048, 2048, 1024, 1024]
Group 3: [4096, 2048, 2048, 2048, 2048]
Group 4: [2048, 2048, 1024, 1024, 1024]
Group 5: [4096, 2048, 2048, 2048, 1024]

Results demonstrate consistent high-tile outcomes across randomized tile placements and game states.

---

## ⚙️ How It Works

- `getMove()` runs iterative deepening using expectiminimax and alpha-beta pruning.
- Heuristics are applied at terminal nodes to score each grid state.
- `maximize()` and `minimize()` simulate the player and the environment, respectively.
- A `chance()` function treats random tile placement as a probabilistic minimizer.

---

## 📂 Project Structure

├── IntelligentAgent.py # Core AI agent logic
├── BaseAI.py # Abstract agent interface
├── GameManager.py # Controls game loop and timing
├── Grid.py # Grid representation and move logic
├── README.md # You're here


---

## 🛠 Technologies

- **Language:** Python 3
- **Techniques:** Expectiminimax, Alpha-Beta Pruning, Heuristic Search, Bayesian Optimization
- **Tools:** NumPy (optional), Object-Oriented Programming

---

## 🧩 Key Insights

- **Adversarial framing** improves strategy quality vs. greedy or myopic agents.
- **Heuristic design** plays a crucial role in achieving high-tile performance.
- **Real-time pruning and iterative depth** balance computational cost and game responsiveness.

---

## 📈 Future Enhancements

- Incorporate **Monte Carlo Tree Search (MCTS)** for stochastic sampling
- Extend to **reinforcement learning** agents for self-play training
- Add **visualizer** for gameplay analysis and replay
