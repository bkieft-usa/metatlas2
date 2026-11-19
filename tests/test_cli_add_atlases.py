"""Tests for the add-atlases command wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


def test_add_atlases_forwards_custom_flag_to_python(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("dirname", "basename", "id", "cat"):
        (bin_dir / command).symlink_to(f"/usr/bin/{command}")

    captured_args = tmp_path / "docker_args.txt"
    docker = bin_dir / "docker"
    docker.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$DOCKER_ARGS_FILE"\n',
        encoding="utf-8",
    )
    docker.chmod(0o755)

    data_dir = tmp_path / "data"
    main_db_dir = data_dir / "databases" / "main_db"
    main_db_dir.mkdir(parents=True)
    (main_db_dir / "metatlas.duckdb").touch()
    (main_db_dir / ".zenodo_version").write_text(
        "https://doi.org/10.5281/zenodo.22968291\n",
        encoding="utf-8",
    )
    config_path = data_dir / "atlases.yaml"
    config_path.touch()

    env = dict(
        os.environ,
        PATH=str(bin_dir),
        HOME=str(tmp_path),
        METATLAS_DATA_DIR=str(data_dir),
        DOCKER_ARGS_FILE=str(captured_args),
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "metatlas2.sh"
    subprocess.run(
        [
            "/bin/bash",
            str(script),
            "add-atlases",
            "--config_path",
            str(config_path),
            "--custom",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    args = captured_args.read_text(encoding="utf-8").splitlines()
    assert any("add_atlases_to_db" in arg and "custom=True" in arg for arg in args)
