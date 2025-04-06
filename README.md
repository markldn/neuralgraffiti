![Grupo 4](https://github.com/user-attachments/assets/088a8ed3-7636-44af-a69d-5fe6d2e140fe)

# Neural Graffiti

Neural Graffiti is a lightweight, memory-modulation layer for transformer language models. It introduces a small, persistent internal state — the spray — which influences the model's hidden states at inference time.

This approach requires no retraining or fine-tuning. Instead, it wraps a base model and injects a feedback signal derived from prior input embeddings. The spray evolves slowly across interactions, introducing subtle behavioral drift based on past context.

Inspired by the principles of state-based systems like Liquid Neural Networks, Neural Graffiti applies continuous memory influence at the output layer of the transformer. It is minimal, modular, and compatible with any model that exposes hidden states.

Tested on Gemma 2B/3B.


