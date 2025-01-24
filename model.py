import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for transformers.
    """
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary positional embedding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_heads, head_dim).
            seq_len (int): Sequence length.

        Returns:
            torch.Tensor: Output tensor with rotary positional embeddings applied.
        """
        batch_size, seq_len, num_heads, head_dim = x.shape

        # Generate position indices
        position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(-1)

        # Generate frequencies
        freqs = torch.exp(
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device) * -(torch.log(torch.tensor(self.theta)) / head_dim)
        )

        # Compute sinusoids
        sinusoid = position * freqs
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)

        # Reshape sin and cos to match the input tensor's shape
        sin = sin.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1, head_dim // 2)
        cos = cos.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1, head_dim // 2)

        # Apply rotary embeddings
        x_rotated = x.clone()
        x_rotated[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        x_rotated[..., 1::2] = x[..., 1::2] * cos + x[..., 0::2] * sin

        return x_rotated

from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    """
    A single transformer block with self-attention and feed-forward layers.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_key_value_heads: int,
        rms_norm_eps: float,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads

        # Ensure the hidden size is divisible by the number of attention heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )

        # Self-attention layers
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # Feed-forward layers
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

        # Normalization layers
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Activation function
        self.act = nn.SiLU() if hidden_act == "silu" else nn.GELU()

        # Rotary positional embedding
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module._forward(inputs[0], inputs[1])
            return custom_forward

        # Use gradient checkpointing
        return checkpoint(create_custom_forward(self), x, attention_mask)

    def _forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.input_norm(x)

        # Project inputs to query, key, and value
        batch_size, seq_len, _ = x.shape

        # Reshape queries for multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)

        # Reshape keys and values for key-value heads
        k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply rotary positional embedding
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Add residual connection
        x = residual + attn_output

        # Feed-forward network
        residual = x
        x = self.post_attention_norm(x)
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        ff_output = self.down_proj(gate * up)

        # Add residual connection
        x = residual + ff_output

        return x

class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        rms_norm_eps: float,
        hidden_act: str = "silu",
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings

        # Embedding layers (skip quantization for these)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(max_position_embeddings, hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_key_value_heads=num_key_value_heads,
                rms_norm_eps=rms_norm_eps,
                hidden_act=hidden_act,
            )
            for _ in range(num_hidden_layers)
        ])

        # Final normalization layer
        self.final_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Output layer (tied to input embeddings if specified)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embed tokens and positions
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        token_embeddings = self.embed_tokens(input_ids)
        position_embeddings = self.embed_positions(position_ids)
        x = token_embeddings + position_embeddings

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final normalization
        x = self.final_norm(x)

        # Output logits
        logits = self.lm_head(x)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            max_length (int): Maximum length of the generated sequence.
            temperature (float): Sampling temperature. Higher values mean more random sampling.
            top_k (int): Top-k sampling. Only the top-k tokens are considered.
            do_sample (bool): Whether to sample from the distribution or take the argmax.

        Returns:
            torch.Tensor: Generated token IDs of shape (batch_size, max_length).
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get the logits for the last token
                logits = self(input_ids)[:, -1, :]

                # Apply temperature
                logits = logits / temperature

                # Top-k sampling
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(logits, top_k)
                    logits[logits < top_k_values[:, -1].unsqueeze(-1)] = -float("Inf")

                # Convert logits to probabilities
                probs = F.softmax(logits, dim=-1)

                # Sample or take the argmax
                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)

                # Append the next token to the input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids

# Create the model based on the configuration
def create_model_from_config(config: dict) -> TransformerModel:
    model_config = config["model"]["model_config"]
    return TransformerModel(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["hidden_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        intermediate_size=model_config["intermediate_size"],
        num_key_value_heads=model_config["num_key_value_heads"],
        max_position_embeddings=model_config["max_position_embeddings"],
        rms_norm_eps=model_config["rms_norm_eps"],
        hidden_act=model_config["hidden_act"],
        tie_word_embeddings=model_config["tie_word_embeddings"],
    )

# Example usage
if __name__ == "__main__":
    import json

    # Load the configuration file
    with open("config_smollm2_135M.json", "r") as f:
        config = json.load(f)

    # Create the model
    model = create_model_from_config(config)
    print(model)