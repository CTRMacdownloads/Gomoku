# Gomoku AI (Qt6 + Python)
**Course Project: Foundations of Artificial Intelligence**

A Gomoku (Five-in-a-Row) game featuring a C++/Qt6 frontend and two distinct Python AI engines.

## 📂 Project Structure

* **Frontend (C++/Qt6)**: Handles GUI, game rules, and process management (`src/`).
* **AI Engine 1 (A*)**: Traditional Minimax search with Alpha-Beta pruning (`ai_stdin.py`).
* **AI Engine 2 (RL)**: Convolutional Policy Network trained via self-play (`RL.py`, `gomoku_policy.pt`).
* **Training Pipeline**: Scripts for data generation and model training (`train_policy.py`, etc.).

## 🚀 Build & Run

### Prerequisites
* **C++**: Qt6, CMake
* **Python**: Python 3.x, PyTorch, NumPy (for RL engine)

### Compilation
From the `003` directory:
```bash
cmake -S . -B build
cmake --build build --config Release

### Running the game
./build/Gomoku

