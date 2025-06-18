import argparse
from pathlib import Path

CONFIG_NUM_TO_LMBDA = {
    "00": 0.0001,
    "01": 0.0002,
    "02": 0.0004,
    "03": 0.001,
    "04": 0.004,
    "05": 0.02,
}
LMBDA_TO_CONFIG_NUM = {v: k for k, v in CONFIG_NUM_TO_LMBDA.items()}


def cfg_str(lambda_value: float, template_file: Path) -> str:
    template = template_file.read_text()
    template = template.replace("{lambda_value}", str(lambda_value))

    # Sometimes different lambdas need different weights to start from.
    template = template.replace(
        "{lmbda_config_num}", f"config_{LMBDA_TO_CONFIG_NUM[lambda_value]}"
    )
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
    for lmbda in CONFIG_NUM_TO_LMBDA.values():
        config = cfg_str(lambda_value=lmbda, template_file=template_path)
        save_cfg(config, dir=exps_dir, name=f"config_{cnt:02}")
        cnt += 1
