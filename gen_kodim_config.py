from pathlib import Path


def cfg_str(img_num: list[int], lambda_value: float, cfg_dir: Path) -> str:
    input_files_lines = "\n".join(
        [f"  - data/kodak/kodim{num:02}.png" for num in img_num]
    )
    template_path = cfg_dir / "kodim_template.yamltemplate"
    template = template_path.read_text()
    template = template.replace("{input_files_list}", input_files_lines)
    template = template.replace("{lambda_value}", str(lambda_value))
    return template


def save_cfg(config: str, dir: Path, name: str) -> None:
    name = name.replace(".yaml", "")
    filename = dir / f"{name}.yaml"
    with open(filename, "w") as f:
        f.write(config)


if __name__ == "__main__":
    exps_dir = Path("cfg/exps/2024-11-21")
    # 1 value of lambda and 8 images per config
    # so that 5 lmbda x 3 img = 15 configs that can be ran by
    # 8 sbatch jobs that run 2 configs in parallel.
    cnt = 0
    for lmbda in [0.0001, 0.0004, 0.001, 0.004, 0.02]:
        for img_start, img_end in [(1, 8), (9, 16), (17, 24)]:
            config = cfg_str(
                img_num=list(range(img_start, img_end + 1)),
                lambda_value=lmbda,
                cfg_dir=exps_dir,
            )
            save_cfg(config, dir=exps_dir, name=f"kodim_config_{cnt:02}")
            cnt += 1
