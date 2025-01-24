import torch
import torch.nn as nn
from model import TransformerModel  # or however you define your model classes
from transformers import AutoTokenizer
import gradio as gr

# Load half-precision state_dict
checkpoint = torch.load("model_weights_fp16.pt", map_location="cpu")
state_dict_fp16 = checkpoint["model_state_dict"]

# Create model in FP16
model = TransformerModel(
    vocab_size=49152,
    hidden_size=576,
    num_hidden_layers=30,
    num_attention_heads=9,
    intermediate_size=1536,
    num_key_value_heads=3,
    max_position_embeddings=2048,
    rms_norm_eps=1e-5,
    hidden_act="silu",
    tie_word_embeddings=True,
)

# Convert model to half precision
model.half()

# Load the half-precision weights
model.load_state_dict(state_dict_fp16, strict=False)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")

def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, do_sample=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

gr.Interface(fn=generate_text, inputs="text", outputs="text").launch()
