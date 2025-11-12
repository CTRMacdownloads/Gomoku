# Gomoku
A full-stack Gomoku platform with a self-trainable AI (MCTS/AlphaZero-style), supporting human-vs-AI, AI-vs-AI play, review, and self-play training pipeline.


rm -rf build
cmake -S . -B build
cmake --build build --config Release
./build/GomokuQt6
