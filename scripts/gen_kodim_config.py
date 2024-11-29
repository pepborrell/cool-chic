from pathlib import Path
import argparse


def cfg_str(img_num: list[int], lambda_value: float, template_file: Path) -> str:
    input_files_lines = "\n".join(
        [f"  - data/kodak/kodim{num:02}.png" for num in img_num]
    )
    template = template_file.read_text()
    template = template.replace("{input_files_list}", input_files_lines)
    template = template.replace("{lambda_value}", str(lambda_value))
    return template


def save_cfg(config: str, dir: Path, name: str, template: bool = False) -> None:
    name = name.replace(".yaml", "")
    filename = dir / f"{name}.yaml"
    if template:
        filename = filename.with_suffix(".yamltemplate")
    with open(filename, "w") as f:
        f.write(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    args = parser.parse_args()

    template_path: Path = args.template
    assert template_path.exists()
    exps_dir = template_path.parent

    # 1 value of lambda and 8 images per config
    # so that 5 lmbda x 3 img = 15 configs that can be ran by
    # 8 sbatch jobs that run 2 configs in parallel.
    cnt = 0
    for lmbda in [0.0001, 0.0004, 0.001, 0.004, 0.02]:
        for img_start, img_end in [(1, 8), (9, 16), (17, 24)]:
            config = cfg_str(
                img_num=list(range(img_start, img_end + 1)),
                lambda_value=lmbda,
                template_file=template_path,
            )
            save_cfg(config, dir=exps_dir, name=f"kodim_config_{cnt:02}")
            cnt += 1
