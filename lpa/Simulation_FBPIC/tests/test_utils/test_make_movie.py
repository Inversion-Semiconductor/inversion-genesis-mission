from pathlib import Path
from unittest.mock import patch

from inversion_fbpic.utils.make_movie import make_movie


def test_make_movie_writes_to_requested_movie_dir(tmp_path: Path) -> None:
    images_dir = tmp_path / "stills"
    images_dir.mkdir()
    (images_dir / "frame_0001.png").touch()
    movie_dir = tmp_path / "movies"

    with patch("inversion_fbpic.utils.make_movie.subprocess.run") as run:
        movie_file = make_movie(
            images_dir=images_dir,
            image_prefix="frame_",
            filename="simulation",
            movie_dir=movie_dir,
        )

    assert movie_file == movie_dir / "simulation.mp4"
    assert movie_dir.is_dir()
    assert run.call_args.args[0][-1] == str(movie_file)


def test_make_movie_optimas_writes_to_requested_movie_dir(tmp_path: Path) -> None:
    images_dir = tmp_path / "stills"
    images_dir.mkdir()
    (images_dir / "frame_0001.png").touch()
    movie_dir = tmp_path / "movies"

    with patch("inversion_fbpic.utils.make_movie.subprocess.run") as run:
        movie_file = make_movie(
            images_dir=images_dir,
            image_prefix="frame_",
            filename="simulation",
            movie_dir=movie_dir,
            optimas=True,
        )

    assert movie_file == movie_dir / "simulation.mp4"
    assert run.call_args.args[0][-1] == str(movie_file.resolve())
    assert run.call_args.kwargs["cwd"] == images_dir
