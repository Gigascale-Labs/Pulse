from pathlib import Path

import yaml
from typer.testing import CliRunner

from pulse.cli import app


runner = CliRunner()


def test_validate_config_command(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "device": "cpu",
                },
                "clustering": {
                    "kmeans": {
                        "backend": "sklearn",
                    }
                },
                "output": {
                    "persona_dump_dir": str(tmp_path / "outputs"),
                },
            }
        )
    )

    result = runner.invoke(
        app,
        ["validate-config", "--config", str(config_path)],
    )

    assert result.exit_code == 0
    assert "Config is valid." in result.output