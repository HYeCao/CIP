from utilis.config import Config
default_config = Config({
    "project_name": "CAE-minor-hps",
    "seed": 0,
    "tag": "dmc",
    "start_steps": 5e3,
    "cuda": True,
    "device": 0,
    "num_steps": 1000001,
    
    "env_name": "HalfCheetah-v2", 
    "eval": True,
    "eval_episodes": 10,
    "eval_interval": 10,
    "replay_size": 1000000,

    "policy": "Gaussian",   
    "gamma": 0.99, 
    "tau": 0.005,
    "lr": 0.0003,
    "alpha": 0.2,
    "quantile": 0.9,
    "automatic_entropy_tuning": True,
    # "batch_size": 512,
    "batch_size": 256,
    "updates_per_step": 1,
    "target_update_interval": 2,
    # "target_update_interval": 1,
    # "hidden_size": 1024,
    "hidden_size": 256,
    "msg": "default"
})

