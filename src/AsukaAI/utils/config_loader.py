# utils/config_loader.py
import logging
from typing import Dict, Any, Tuple
from utils import json_handler as jh # Assuming json_handler.py is in utils/
from utils import constants # Import constants to access DEBUG_MODE if needed for logging

def load_settings_from_json(
    logger: logging.Logger, # Logger is passed in, its level already respects DEBUG_MODE
    json_handler_instance: jh.JsonHandler,
    config_definitions: Dict[str, Tuple[str, Any]],
    module_name: str = "Module"
) -> Dict[str, Any]:
    loaded_config_values: Dict[str, Any] = {}
    config_file_path = json_handler_instance.config_file
    all_defaults_used = False

    try:
        for var_name, (json_key, default_value) in config_definitions.items():
            loaded_config_values[var_name] = json_handler_instance.get_setting(json_key, default_value)
        logger.info(
            f"{module_name} configuration processed using '{config_file_path}' "
            "(defaults applied where necessary)."
        )
    except Exception as e:
        logger.error(
            f"Critical error processing {module_name} configuration from '{config_file_path}': {e}. "
            f"Falling back to all default settings for {module_name}.",
            exc_info=True
        )
        all_defaults_used = True
        for var_name, (_, default_value) in config_definitions.items():
            loaded_config_values[var_name] = default_value
    
    # Log final values if global DEBUG_MODE is on, or if all defaults were forced,
    # or if the logger for this module is explicitly set to DEBUG.
    if constants.DEBUG_MODE or all_defaults_used or logger.isEnabledFor(logging.DEBUG):
        summary = {k: (loaded_config_values[k] if not isinstance(loaded_config_values[k], str) or len(loaded_config_values[k]) < 100 else loaded_config_values[k][:97] + "...") for k in config_definitions}
        logger.debug(f"{module_name} final configuration values: {summary}")
        
    return loaded_config_values