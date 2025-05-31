import sys
import pytest
import logging
from unittest.mock import patch, MagicMock
from src import main


def run_main_with_args(args):
    with patch.object(sys, 'argv', args):
        with patch('src.main.sys.exit') as mock_exit:
            main.main()
            return mock_exit


@patch('src.main.get_data')
@patch('src.main.validate_schema')
@patch('src.main.run_model_pipeline')
@patch('src.main.run_inference')
@patch('src.main.load_config')
@patch('src.main.setup_logging')
def test_main_all_stages(
    mock_setup_logging,
    mock_load_config,
    mock_run_inference,
    mock_run_model_pipeline,
    mock_validate_schema,
    mock_get_data
):
    # Arrange
    mock_load_config.return_value = {
        "logging": {},
        "data_validation": {"schema": {"columns": []}},
        "inference": {"input_csv": "input.csv", "output_csv": "output.csv"}
    }
    mock_get_data.return_value = MagicMock(shape=(10, 5), head=lambda: "head")

    # Act
    with patch.object(sys, 'argv', ['src/main.py', '--stage', 'all']):
        with patch('builtins.print'):
            with patch('src.main.logger.info'):
                with patch('src.main.logger.exception'):
                    with patch('src.main.sys.exit') as mock_exit:
                        main.main()
                        # Assert exit called zero times on success
                        assert not mock_exit.called

    # Assert calls
    mock_load_config.assert_called_once()
    mock_setup_logging.assert_called_once()
    mock_get_data.assert_called_once()
    mock_validate_schema.assert_called_once()
    mock_run_model_pipeline.assert_called_once()
    mock_run_inference.assert_not_called()  # infer only on 'infer' stage


def test_main_infer_stage_calls_run_inference():
    with patch.object(sys, 'argv', ['src/main.py', '--stage', 'infer']):
        with patch('src.main.load_config') as mock_load_config, \
             patch('src.main.setup_logging'), \
             patch('src.main.run_inference') as mock_run_inference, \
             patch('src.main.sys.exit') as mock_exit:

            mock_load_config.return_value = {
                "logging": {},
                "inference": {
                    "input_csv": "input.csv",
                    "output_csv": "output.csv"
                    }
            }
            main.main()
            mock_run_inference.assert_called_once_with(
                "input.csv",
                "config.yaml",
                "output.csv"
                )
            assert not mock_exit.called


@pytest.mark.parametrize("func_name", [
    "load_config",
    "get_data",
    "validate_schema",
    "run_model_pipeline",
    "run_inference",
])
def test_main_failure_triggers_exit(func_name):
    with patch.object(sys, 'argv', ['src/main.py', '--stage', 'all']), \
         patch('src.main.load_config') as mock_load_config, \
         patch('src.main.setup_logging'), \
         patch(f'src.main.{func_name}') as mock_func, \
         patch('src.main.sys.exit') as mock_exit, \
         patch('src.main.logger.exception') as mock_log_exc:

        mock_load_config.return_value = {
            "logging": {},
            "data_validation": {"schema": {"columns": []}},
            "inference": {"input_csv": "input.csv", "output_csv": "output.csv"}
        }

        if func_name == "load_config":
            mock_load_config.side_effect = FileNotFoundError("Config missing")
        else:
            mock_func.side_effect = RuntimeError("Test error")

        main.main()
        assert mock_exit.called
        mock_log_exc.assert_called()


def test_load_config_invalid_file(tmp_path):
    invalid_config_path = tmp_path / "invalid_config.yaml"
    invalid_config_path.write_text("!!!invalid_yaml")

    with pytest.raises(Exception):
        main.load_config(str(invalid_config_path))


def test_setup_logging_creates_handlers(tmp_path):
    log_file = tmp_path / "test.log"
    config = {
        "log_file": str(log_file),
        "level": "DEBUG",
        "format": "%(message)s",
        "datefmt": "%H:%M:%S"
    }
    main.setup_logging(config)
    logger = logging.getLogger()
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
