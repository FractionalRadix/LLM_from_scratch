from GPT_config_values import GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M
from chapter04 import GPTModel

from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(
    #model_size="124M", models_dir="gpt2"
    #model_size="355M", models_dir="gpt2"
    #model_size="774M", models_dir="gpt2"
    model_size="1558M", models_dir="gpt2"
)

print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

#print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)

model_configs = {
    "gpt2-small (124M)" : { "emb_dim":  768, "n_layers": 12, "n_heads": 12 },
    "gpt2-medium (355M)": { "emb_dim": 1024, "n_layers": 24, "n_heads": 16 },
    "gpt2-large (774M)" : { "emb_dim": 1280, "n_layers": 36, "n_heads": 20 },
    "gpt2-xl (1558)"    : { "emb_dim": 1600, "n_layers": 48, "n_heads": 25 },
}


#model_name = "gpt2-small (124M)"
#NEW_CONFIG=GPT_CONFIG_124M.copy()

#model_name = "gpt2-medium (355M)"
#NEW_CONFIG=GPT_CONFIG_355M.copy()

#model_name = "gpt2-large (774M)"
#NEW_CONFIG=GPT_CONFIG_744M.copy()

model_name = "gpt2-xl (1558)"
NEW_CONFIG=GPT_CONFIG_1558M.copy()

NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()


