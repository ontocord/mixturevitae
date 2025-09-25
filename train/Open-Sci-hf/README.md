# Opensci model hf

```bash
pip install -r requirements.txt
```

```bash
python example.py
```

```
OpensciForCausalLM(
  (model): OpensciModel(
    (embed_tokens): Embedding(50304, 2048)
    (layers): ModuleList(
      (0-23): 24 x OpensciDecoderLayer(
        (self_attn): OpensciAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (q_norm): OpensciRMSNorm((64,), eps=1e-05)
          (k_norm): OpensciRMSNorm((64,), eps=1e-05)
        )
        (mlp): OpensciMLP(
          (gate_proj): Linear(in_features=2048, out_features=5440, bias=True)
          (up_proj): Linear(in_features=2048, out_features=5440, bias=True)
          (down_proj): Linear(in_features=5440, out_features=2048, bias=True)
          (act_fn): SiLU()
        )
        (input_layernorm): OpensciRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): OpensciRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): OpensciRMSNorm((2048,), eps=1e-05)
    (rotary_emb): OpensciRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=50304, bias=False)
)
```