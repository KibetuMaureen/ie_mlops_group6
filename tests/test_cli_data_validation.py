import subprocess
import os

def test_validator_cli_success(tmp_path):
    # Prepare paths
    root = os.path.dirname(os.path.dirname(__file__))
    script = os.path.join(root, "src", "data_validation", "data_validation.py")
    data_path = os.path.join(root, "data", "raw", "fraudTrain.csv")
    config_path = os.path.join(root, "config.yaml")
    output_path = tmp_path / "output.csv"

    # Run the script as if from the command line
    result = subprocess.run(
        [
            "python",
            script,
            data_path,
            config_path,
            "--output_csv",
            str(output_path)
        ],
        capture_output=True,
        text=True
    )

    # Assert it exited successfully
    assert result.returncode == 0
    assert "Data validation completed successfully." in result.stdout or result.stderr
    assert output_path.exists()

def test_validator_cli_fails_on_bad_file():
    result = subprocess.run(
        ["python", "src/data_validation/data_validation.py", "not_found.csv", "config.yaml"],
        capture_output=True,
        text=True
    )
    assert result.returncode != 0
    assert "Failed to load data file" in result.stderr
