🤖Decoder-Only Models with Progressive Neural Networks🤖
--
This repository contains implementations of decoder-only Transformer models integrated with Progressive Neural Networks (PNN), designed for text generation and continual learning.
These models are autoregressive, making them ideal for tasks like natural language generation, and leverage PNN to adapt to new tasks without forgetting previous ones.

🥇Overview
--
The repository provides a lightweight, inference-focused implementation of a decoder-only Transformer model with PNN. 
The model supports efficient text generation using a single PNN column, with features like mixed precision (FP16) for reduced memory usage and key-value caching for fast inference. 
It is suitable for researchers and developers interested in exploring continual learning and generative AI.

🥈Features
--
1️⃣Decoder-Only Architecture: Autoregressive model optimized for text generation.

2️⃣Progressive Neural Networks (PNN): Enables continual learning by adding task-specific columns while preserving knowledge from prior tasks.

3️⃣Mixed Precision (FP16): Reduces memory footprint, enabling deployment on modest hardware.

4️⃣Key-Value Caching: Accelerates inference for sequential text generation.

5️⃣Modular Design: Easily extendable for additional PNN columns or model configurations.

6️⃣Inference-Only: Focused on text generation, with no training components.

🥉Installation
--
1) Clone the Repository:

```
git clone https://github.com/Aviation169/Decoder_model.git
cd llama.py
```

2) Set Up a Virtual Environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3) Install Dependencies:

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

4) Verify PyTorch installation:

```
import torch
print(torch.__version__, torch.cuda.is_available())
```

📅License
--

This project is licensed under the MIT License. See the LICENSE file for details.

📒Contact
--

For questions or support, contact the repository maintainer at `akajay14955j@gmail.com` or open an issue on GitHub.
