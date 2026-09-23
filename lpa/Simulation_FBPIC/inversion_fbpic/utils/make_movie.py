import shutil
import subprocess
from pathlib import Path

# Check if ffmpeg is installed
if shutil.which("ffmpeg") is None:
    raise RuntimeError("ffmpeg is required for make_movie but was not found on PATH.")


def make_movie(
    images_dir: Path | str,
    image_prefix: str = "",
    filename: str = "out",
    framerate: int = 10,
    image_extension: str = "png",
    *,
    optimas: bool = False,
    movie_dir: Path | str | None = None,
) -> Path | None:
    """
    Make a movie from the image files in the given directory. Requires ffmpeg to be installed.

    Args:
        images_dir (Path | str): Path to the directory containing the image files.
        image_prefix (str): Prefix of the image files.
        filename (str): Name of the output movie file (without extension).
        framerate (int): Frame rate of the output movie.
        image_extension (str): Extension of the image files. Defaults to "png".
        optimas (bool): If True, use cwd-relative glob and close stdin so ffmpeg does not
            hang when spawned from Optimas workers. Defaults to False (legacy command).
        movie_dir (Path | str | None): Directory for the output movie. Defaults to
            `images_dir`.

    Returns:
        Path to the output movie file, or None if the operation failed.
    """
    images_dir = Path(images_dir)
    movie_dir = images_dir if movie_dir is None else Path(movie_dir)
    movie_dir.mkdir(parents=True, exist_ok=True)
    movie_file = movie_dir / f"{filename}.mp4"

    if optimas:
        images = sorted(images_dir.glob(f"{image_prefix}*.{image_extension}"))
        if not images:
            print(
                f"make_movie: no images matching {image_prefix}*.{image_extension} "
                f"in {images_dir}"
            )
            return None

        # Match manual workflow: glob from inside stills_dir, libx264, yuv420p.
        # -nostdin + stdin=DEVNULL: ffmpeg otherwise blocks on stdin when spawned
        # from Optimas/subprocess (prints version banner then hangs indefinitely).
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-framerate",
            str(framerate),
            "-pattern_type",
            "glob",
            "-i",
            f"{image_prefix}*.{image_extension}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(movie_file.resolve()),
        ]
        print(
            f"make_movie: encoding {len(images)} frames at {framerate} fps -> {movie_file}"
        )
        try:
            subprocess.run(
                cmd,
                cwd=images_dir,
                check=True,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            print(f"make_movie: ffmpeg failed with exit code {exc.returncode}")
            return None
    else:
        cmd = [
            "ffmpeg",
            "-framerate",
            str(framerate),
            "-pattern_type",
            "glob",
            "-i",
            f"{images_dir}/{image_prefix}*.{image_extension}",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(movie_file),
        ]
        try:
            subprocess.run(cmd, shell=False, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Error making movie: {exc.returncode}")
            return None

    print(f"Movie created successfully: {movie_file}")
    return movie_file
