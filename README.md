# Causal Information Prioritization for Efficient Reinforcement Learning


This repository is the official PyTorch implementation of **CIP**. 

# 🛠️ Installation Instructions

First, create a virtual environment and install all required packages. 
~~~
conda create -n cae python=3.8
pip install -r requirements.txt
~~~


## 💻 Code Usage

If you would like to run CIP on a standard version of a certain `task`, please use `main_causal.py` to train CIP policies.
~~~
export MUJOCO_GL="osmesa"
~~~
~~~
xvfb-run -a python main_causal.py --env_name task
~~~
If you would like to run CIP on a sparse reward version of a certain `task`, please follow the command below.
~~~
python main_causal.py --env_name task --reward_type sparse
~~~

## 📝 Citation

If you use our method or code in your research, please consider citing the paper as follows:

```
@article{cao2025causal,
  title={Causal information prioritization for efficient reinforcement learning},
  author={Cao, Hongye and Feng, Fan and Yang, Tianpei and Huo, Jing and Gao, Yang},
  journal={arXiv preprint arXiv:2502.10097},
  year={2025}
}
```

## 🙏 Acknowledgement

CIP is licensed under the MIT license. MuJoCo and DeepMind Control Suite are licensed under the Apache 2.0 license. 