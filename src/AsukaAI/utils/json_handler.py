import json
import os
from typing import Any, Dict, Optional
from functools import wraps

class JsonHandler:
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config: Dict[str, Any] = self.load_config()

    @staticmethod
    def config_file_exists(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"Configuration file '{self.config_file}' not found.")
            return func(self, *args, **kwargs)
        return wrapper

    def load_config(self) -> dict[str, Any]: # Use lowercase dict for modern type hinting
        try:
            if not os.path.exists(self.config_file):
                # Return empty dict if file doesn't exist, allowing optional config
                return {}
            with open(self.config_file, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON in '{self.config_file}': {e}")

    @config_file_exists
    def save_config(self) -> None:
        with open(self.config_file, 'w') as file:
            json.dump(self.config, file, indent=4)

    def get_setting(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def set_setting(self, key: str, value: Any) -> None:
        keys = key.split('.')
        d = self.config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        self.save_config()

    def delete_setting(self, key: str) -> None:
        keys = key.split('.')
        d = self.config
        for k in keys[:-1]:
            if k not in d:
                return
            d = d[k]
        if keys[-1] in d:
            del d[keys[-1]]
            self.save_config()

    @config_file_exists
    def reset_config(self) -> None:
        self.config = {}
        self.save_config()

    def get_all_settings(self) -> dict[str, Any]: # Use lowercase dict
        return self.config.copy()

    def update_settings(self, new_settings: dict[str, Any]) -> None: # Use lowercase dict
        self.config.update(new_settings)
        self.save_config()