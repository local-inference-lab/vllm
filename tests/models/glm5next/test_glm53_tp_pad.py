# No-GPU unit test for the TP3 padding overlay: run inside the R17 image with the overlays mounted,
#   VLLM_GLM53_TP_HEAD_PAD=72,66 TP=3, model dir at /model.
import torch, vllm.distributed as dist
dist.get_tensor_model_parallel_world_size = lambda: 3  # no process group in this test
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig
from vllm.transformers_utils.configs.glm53_tp_pad import pad_head_weights, routed_expert_intermediate, shared_expert_intermediate, vocab_padding_size
c = Glm5NextConfig.from_pretrained("/model"); t = c.text_config
assert (t.num_attention_heads, t.linear_num_heads, t.linear_attn_config["num_heads"], t.num_attention_heads_ckpt, t.linear_num_heads_ckpt) == (72, 66, 66, 64, 64)
t2 = type(t)(**t.to_dict()); assert (t2.num_attention_heads, t2.num_attention_heads_ckpt) == (72, 64), "re-instantiation not idempotent"
assert routed_expert_intermediate(t, 3) == 2304 and routed_expert_intermediate(t, 3, mtp=True) == 2304 and routed_expert_intermediate(t, 4) == 2048 and routed_expert_intermediate(t, 2) == 2048 and routed_expert_intermediate(t, 8) == 2048
assert shared_expert_intermediate(t, 3) == 2112 and shared_expert_intermediate(t, 4) == 2048 and vocab_padding_size() == 192
P = "layers.%d.%s"  # prefix-relative names, as AutoWeightsLoader passes them to Glm5NextModel.load_weights
bf = torch.bfloat16
cases = {
 P % (0, "self_attn.q_proj.weight"): ((8192, 4096), (8448, 4096)),
 P % (0, "self_attn.b_proj.weight"): ((64, 4096), (66, 4096)),
 P % (0, "self_attn.forget_gate.A_log"): ((64,), (66,)),
 P % (0, "self_attn.dt_bias"): ((8192,), (8448,)),
 P % (0, "self_attn.k_conv1d.weight"): ((8192, 1, 4), (8448, 1, 4)),
 P % (0, "self_attn.f_b_proj.weight"): ((8192, 128), (8448, 128)),
 P % (0, "self_attn.g_b_proj.weight"): ((8192, 128), (8448, 128)),
 P % (0, "self_attn.o_proj.weight"): ((4096, 8192), (4096, 8448)),
 P % (0, "self_attn.f_a_proj.weight"): ((128, 4096), (128, 4096)),
 P % (0, "mlp.gate_proj.weight"): ((12288, 4096), (12288, 4096)),
 P % (3, "self_attn.q_b_proj.weight"): ((16384, 1536), (18432, 1536)),
 P % (3, "self_attn.kv_b_proj.weight"): ((32768, 512), (36864, 512)),
 P % (3, "self_attn.o_proj.weight"): ((4096, 16384), (4096, 18432)),
 P % (3, "self_attn.indexer.wq_b.weight"): ((4096, 1536), (4096, 1536)),
 P % (3, "mlp.shared_experts.gate_proj.weight"): ((2048, 4096), (2112, 4096)),
 P % (3, "mlp.shared_experts.up_proj.weight"): ((2048, 4096), (2112, 4096)),
 P % (3, "mlp.shared_experts.down_proj.weight"): ((4096, 2048), (4096, 2112)),
 P % (45, "self_attn.kv_b_proj.weight"): ((32768, 512), (36864, 512)),
 P % (45, "mlp.shared_experts.down_proj.weight"): ((4096, 2048), (4096, 2112)),
 P % (3, "mlp.experts.0.gate_proj.weight"): ((2048, 2048), (2304, 2048)),
 P % (3, "mlp.experts.0.up_proj.weight_scale"): ((2048, 256), (2304, 256)),
 P % (3, "mlp.experts.7.down_proj.weight"): ((4096, 1024), (4096, 1152)),
 P % (3, "mlp.experts.7.down_proj.weight_scale"): ((4096, 128), (4096, 144)),
 P % (3, "mlp.experts.7.down_proj.weight_scale_2"): ((), ()),
 P % (45, "mlp.experts.0.gate_proj.weight"): ((2048, 4096), (2304, 4096)),
 P % (45, "mlp.experts.0.gate_proj.weight_scale"): ((2048, 128), (2304, 128)),
 P % (45, "mlp.experts.0.down_proj.weight"): ((4096, 2048), (4096, 2304)),
 P % (45, "mlp.experts.0.down_proj.weight_scale"): ((4096, 64), (4096, 72)),
 "model.layers.45.self_attn.o_proj.weight": ((4096, 16384), (4096, 18432)),
 "lm_head.weight": ((154880, 4096), (154880, 4096)),
}
def dt(n):
    if "A_log" in n or "dt_bias" in n or n.endswith("_scale_2"): return torch.float32
    if ".experts." in n and ".shared" not in n:
        mtp = "layers.45." in n
        if n.endswith("weight_scale"): return torch.uint8 if mtp else torch.float8_e4m3fn
        return torch.float8_e4m3fn if mtp else torch.uint8
    return bf
ws = [(n, torch.ones(s, dtype=dt(n))) for n, (s, _) in cases.items()]
out = dict(pad_head_weights(ws, t))
for n, (_, want) in cases.items():
    got = tuple(out[n].shape); assert got == want, (n, got, want)
    if got != cases[n][0]:  # padded region must be zero, original region intact
        assert float(out[n].float().sum()) == float(torch.ones(cases[n][0]).sum()), n
print("PAD TEST OK:", len(cases), "tensors")
