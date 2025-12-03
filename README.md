Gomoku: A* Search and RL Policy Engine
Course project for Foundations of Artificial Intelligence.
This repository contains a Qt6-based Gomoku game with two different AI
engines:


a traditional A*/Minimax search engine; and


a reinforcement-learning (RL) policy network trained from self-play
and human–AI games.


The final version of the project used in our report is under the
003/ directory.

Repository Structure


003/src/ – C++/Qt6 sources


boardwidget.* – renders the 15×15 board and handles mouse input


gamemodel.* – owns board state, legal-move checking, win/draw detection


gamecontroller.* – orchestrates human vs AI / AI vs AI, talks to AiClient


aiclient.* – abstract interface for AI engines


python_ai_process.* – launches Python, sends/receives JSON messages
over stdin / stdout




003/ai_stdin.py – original A*/Minimax engine (JSON / stdio interface)


003/RL.py – RL policy engine that loads gomoku_policy.pt


003/generate_data.py – A* self-play to generate gomoku_dataset.pt


003/human-ai_generate.py – builds human_dataset.pt from
ai_online_log.jsonl (human–AI games)


003/train_policy.py – trains the convolutional PolicyNet
using cross-entropy loss


003/gomoku_dataset.pt, 003/human_dataset.pt – training datasets


003/gomoku_policy.pt – final trained policy model used by RL.py



Build Instructions (Qt / C++)
From the repository root:
cd 003
cmake -S . -B build
cmake --build build --config Release
./build/Gomoku

This will open the Qt6 GUI for Gomoku.
By default the C++ framework can be wired to either engine via
PythonAiProcess:


to use the A* engine, start ai_stdin.py;


to use the RL engine, start RL.py (which loads gomoku_policy.pt).


Both engines use the same JSON protocol:
{
  "size": 15,
  "board": [[0,1,0,...], [0,2,0,...], ...],
  "player": "black" or "white",
  "last": [row, col]
}

and reply with:
{ "row": r, "col": c }


Data Generation and Training
To regenerate the datasets and the RL policy model:
cd 003

# 1. A* self-play dataset (AI vs AI using ai_stdin.py)
python3 generate_data.py        # produces gomoku_dataset.pt

# 2. Human–AI dataset from online logs
python3 human-ai_generate.py    # consumes ai_online_log.jsonl,
                                # produces human_dataset.pt

# 3. Train the policy network
python3 train_policy.py         # produces gomoku_policy.pt

train_policy.py trains a small convolutional policy network
(PolicyNet) using cross-entropy loss and reports training / validation
accuracy. The saved weights are loaded by RL.py at runtime when the
RL engine is selected.

Project Description (Short)


Frontend: Qt6 GUI written in C++. BoardWidget is responsible for
rendering the board; GameModel stores the state and enforces rules;
GameController coordinates turns and communicates with AI engines
through the AiClient interface and PythonAiProcess.


A* Engine: ai_stdin.py implements threat-based move generation
plus depth-limited Minimax with alpha–beta pruning. It plays both as a
standalone opponent and as a teacher for dataset generation.


RL Engine: RL.py implements a convolutional policy network that
maps a 2-channel board tensor (current player vs opponent stones) to
move logits. It supports both greedy argmax selection and stochastic
softmax sampling.


Training Pipeline: generate_data.py runs A* self-play;
human-ai_generate.py reconstructs human–AI games from JSON logs;
train_policy.py trains PolicyNet on the union of AI and human
state–action pairs.


This README describes the latest project structure and how to build and
run both AI engines, in line with the course rubric.

Authors


Xianhao Dai – Qt framework and A* search engine


Qiyou Wu – RL pipeline (data generation, training scripts, RL runtime)



