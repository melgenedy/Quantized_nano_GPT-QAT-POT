This code is based on NanoGPT repositoray,
https://github.com/karpathy/nanoGPT


The main readme for nanoGPT is renamed to "README_NanoGPT.md"

The main files that contain my contribution in quantizing character level model (trained on Shakespeare dataset) with QAT and both uniforam and power of two quantization,

- train_1_qat_pot.py ==> to train quantized model QAT of character based on Shakespeare dataset. Both normal quantization and power of two (PoT) are supported
- train_1_orig ==> has the original training without QAT
- fakequant_pot.py ==> Includes the PoT fake quantize class (FakeQuantize_PoT), and the differentiable (STE) Power-of-two implementation class
- test_pot_example.ipynb ==> Has some basic trials and illustration about Power of two (PoT) quantization
- config/train_shakespeare_char_qat.py ==> includes training and model configurations parameters for QAT
- directory "out-shakespeare-char-qat" ==> has pretrained LLM model for caracter based model, that can be used as baseline for QAT to start with and fine tuning with quantization inserted
- Used some small suggestions and fixes by LLMs. Also, differnet questions and discussions with LLMs

