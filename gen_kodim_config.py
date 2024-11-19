def cfg_str(num: str) -> str:
    return f"""input: data/kodak/kodim{num}.png
output: results/exps/kodak/kodim{num}.cool
workdir: null
disable_wandb: False
load_models: False

enc_cfg:
  n_itr: 100000
  start_lr: 1e-2 # This should be removed, as it is hardcoded in every phase.
  n_train_loops: 2
  std_recipe_name: c3x

dec_cfgs:
  - config_name: hop
    arm: 16,2
    layers_synthesis: 48-1-linear-relu,X-1-linear-none,X-3-residual-relu,X-3-residual-none
    n_ft_per_res: 1,1,1,1,1,1,1
    ups_k_size: 8
    ups_preconcat_k_size: 7
  - config_name: mop
    arm: 16,2
    layers_synthesis: 16-1-linear-relu,X-1-linear-none,X-3-residual-relu,X-3-residual-none
    n_ft_per_res: 1,1,1,1,1,1,1
    ups_k_size: 8
    ups_preconcat_k_size: 7
  - config_name: lop
    arm: 8,2
    layers_synthesis: 16-1-linear-relu,X-1-linear-none,X-3-residual-relu,X-3-residual-none
    n_ft_per_res: 1,1,1,1,1,1,1
    ups_k_size: 8
    ups_preconcat_k_size: 7
  - config_name: vlop
    arm: 8,1
    layers_synthesis: 8-1-linear-relu,X-1-linear-none,X-3-residual-none
    n_ft_per_res: 1,1,1,1,1,1,1
    ups_k_size: 8
    ups_preconcat_k_size: 7"""


def generate_cfg(num: str):
    with open(f"cfg/exps/2024-11-15/kodim{num}.yaml", "w") as f:
        f.write(cfg_str(num))


if __name__ == "__main__":
    for i in range(1, 25):
        num_str = str(i).zfill(2)
        generate_cfg(num_str)
