import json
import os
from typing import Any, Dict 

class JsonHandler:
    def __init__(self, config_file: str = 'config.json'): # Default path
        self.config_file = config_file
        self.config: Dict[str, Any] = self._load_config_internal()

    def _load_config_internal(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.config_file):
                return {}
            with open(self.config_file, 'r', encoding='utf-8') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON in '{self.config_file}': {e}") from e
        except Exception as e:
            raise IOError(f"Could not load configuration file '{self.config_file}': {e}") from e

    def save_config(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.config_file) or '.', exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as file:
                json.dump(self.config, file, indent=4)
        except IOError as e:
            raise

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
            if not isinstance(d, dict):
                raise TypeError(f"Configuration structure error: expected a dictionary at path leading to '{k}'")
        d[keys[-1]] = value
        self.save_config()

    def delete_setting(self, key: str) -> None:
        keys = key.split('.')
        d = self.config
        for i, k_segment in enumerate(keys[:-1]):
            if isinstance(d, dict) and k_segment in d:
                d = d[k_segment]
            else:
                return
        if isinstance(d, dict) and keys[-1] in d:
            del d[keys[-1]]
            self.save_config()

    def reset_config(self) -> None:
        self.config = {}
        self.save_config()

    def get_all_settings(self) -> Dict[str, Any]:
        return self.config.copy()

    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        self.config.update(new_settings)
        self.save_config()