import argparse
import tempfile
from pathlib import Path

from tqdm import tqdm

from coolchic.enc.bitstream.encode import encode_video
from coolchic.enc.component.video import load_video_encoder


def real_bitstream_rate_bpp(
    bitstream_path: Path, video_encoder_savepath: Path
) -> float:
    real_rate_byte = bitstream_path.stat().st_size
    # Not very elegant but look at the first frame cool-chic to get the video resolution
    video_encoder = load_video_encoder(video_encoder_savepath)
    assert (
        video_encoder.all_frame_encoders["0"][0].coolchic_encoder_param.img_size
        is not None
    )
    h, w = video_encoder.all_frame_encoders["0"][0].coolchic_encoder_param.img_size
    real_rate_bpp = (
        real_rate_byte * 8 / (h * w * len(video_encoder.coding_structure.frames))
    )
    return real_rate_bpp


def save_bitstream(path_video_encoder: Path, output_path: Path) -> None:
    video_encoder = load_video_encoder(path_video_encoder)
    encode_video(video_encoder, output_path, hls_sig_blksize=16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("results/exps/only_latents"),
    )
    args = parser.parse_args()

    encoder_paths = [
        file
        for file in args.experiments_root.rglob("*video_encoder.pt")
        if file.is_file()
    ]
    print(f"{len(encoder_paths)=}")
    for encoder in tqdm(encoder_paths):
        with tempfile.NamedTemporaryFile() as tmp_file:
            save_bitstream(encoder, Path(tmp_file.name))
            rate = real_bitstream_rate_bpp(Path(tmp_file.name), encoder)
            with open(encoder.parent / "real_rate_bpp.txt", "w") as f:
                f.write(str(rate))
