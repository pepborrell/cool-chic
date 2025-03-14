import argparse
from pathlib import Path


def cfg_str(lambda_value: float, template_file: Path) -> str:
    template = template_file.read_text()
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

    cnt = 0
    # 1 value of lambda per config
    for lmbda in [0.0001, 0.0004, 0.001, 0.004, 0.02]:
        config = cfg_str(lambda_value=lmbda, template_file=template_path)
        save_cfg(config, dir=exps_dir, name=f"config_{cnt:02}")
        cnt += 1
