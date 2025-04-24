
def get_config():
    max_eps_length = 100
    config = {
        "PPO":{
            "critic_coef": 1,
            "policy_kl_range":0.0008,
            "policy_params": 20,
            "gamma":0.998,
            "gae_lambda":0.95,
            "value_clip": 0.2,
        },
        "LSTM":{
            "max_eps_length":max_eps_length + 50,
            "seq_length":64,
            "hidden_size":128,
            "embed_size": 64,
        },
        "entropy_coef":{
            "start": 0.01,
            "end": 0,
            "step": 100_000
        },
        "lr":1e-5,
        "num_epochs": 5,
        "num_game_per_batch":2,
        "max_grad_norm": 0.5,
        "n_mini_batch": 4,
        "rewards": [0,1,0], # [lose,win,not_done]
        "set_detect_anomaly": True,
        "normalize_advantage": True,
    }
    return config
