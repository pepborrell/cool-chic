from coolchic.enc.component.coolchic import CoolChicEncoder
from coolchic.hypernet.hypernet import CoolchicWholeNet, DeltaWholeNet, NOWholeNet
from coolchic.utils.paths import CONFIG_DIR
from coolchic.utils.types import HypernetRunConfig, load_config


def flops_coolchic():
    print("-" * 10, "DeltaWholeNet", "-" * 10)
    config_path = CONFIG_DIR / "exps" / "delta-hn" / "like_no_cchic.yaml"
    cfg = load_config(config_path, HypernetRunConfig)
    net = DeltaWholeNet(config=cfg.hypernet_cfg)
    net.mean_decoder.param.set_image_size((512, 512))
    net.mean_decoder.get_flops()

    delta_flops = net.mean_decoder.total_flops
    print(f"{delta_flops=}")
    print(net.mean_decoder.flops_str)

    print("-" * 10, "NOWholeNet", "-" * 10)
    config_path = CONFIG_DIR / "exps" / "no-cchic" / "batching.yaml"
    cfg = load_config(config_path, HypernetRunConfig)
    net = NOWholeNet(config=cfg.hypernet_cfg)
    net.mean_decoder.param.set_image_size((512, 512))
    net.mean_decoder.get_flops()

    no_flops = net.mean_decoder.total_flops
    print(f"{no_flops=}")
    print(net.mean_decoder.flops_str)

    print("-" * 10, "CoolchicEncoder", "-" * 10)
    print(f"{net.mean_decoder.param.img_size=}")
    enc = CoolChicEncoder(param=net.mean_decoder.param)
    enc.get_flops()

    enc_flops = enc.total_flops
    print(f"{enc_flops=}")
    print(enc.flops_str)

    print(f"{100*enc_flops / no_flops=}%")


#### NOW FOR HYPERNETS ####
def flops_hnet():
    config_path = CONFIG_DIR / "exps" / "big-rerun" / "control.yaml"
    cfg = load_config(config_path, HypernetRunConfig)
    wholenet = CoolchicWholeNet(config=cfg.hypernet_cfg)

    print("-" * 10, "LatentHypernet", "-" * 10)
    latent = wholenet.hypernet.latent_hn
    latent.get_flops()
    latent_flops = latent.total_flops
    print(f"{latent_flops=}")
    print(latent.flops_str)

    print("-" * 10, "SynthHypernet", "-" * 10)
    synth = wholenet.hypernet.synthesis_hn
    synth.get_flops()
    synth_flops = synth.total_flops
    print(f"{synth_flops=}")
    print(synth.flops_str)

    print("-" * 10, "ArmHypernet", "-" * 10)
    arm = wholenet.hypernet.arm_hn
    arm.get_flops()
    arm_flops = arm.total_flops
    print(f"{arm_flops=}")
    print(arm.flops_str)

    print("-" * 10, "Backbone", "-" * 10)
    backbone = wholenet.hypernet.hn_backbone
    backbone.get_flops()
    backbone_flops = backbone.total_flops
    print(f"{backbone_flops=}")
    print(backbone.flops_str)


# flops_coolchic()
flops_hnet()
