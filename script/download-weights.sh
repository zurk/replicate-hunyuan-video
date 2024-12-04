#! /bin/bash
huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./ckpts/llava-llama-3-8b-v1_1-transformers
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./ckpts/text_encoder_2
python /HunyuanVideo/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir ./ckpts/llava-llama-3-8b-v1_1-transformers --output_dir ./ckpts/text_encoder
