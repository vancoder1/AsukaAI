import json
import logging
import modules.logging_config as lf

logger = lf.configure_logger(__name__)

class JsonHandler:
    def __init__(self, 
                 config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.config_file, 'r') as file:
                config = json.load(file)
                logger.info("Configuration loaded successfully")
                return config
        except FileNotFoundError:
            logger.warning("Configuration file not found, initializing with an empty configuration")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return {}

    def save_config(self):
        try:
            with open(self.config_file, 'w') as file:
                json.dump(self.config, file, indent=4)
                logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def get_setting(self, key, default=None):
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value.get(k, default)
                if value is default:
                    break
            return value
        except Exception as e:
            logger.error(f"Error getting setting {key}: {e}")
            return default

    def set_setting(self, key, value):
        try:
            keys = key.split('.')
            d = self.config
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
            self.save_config()
            logger.info(f"Setting {key} updated to {value}")
        except Exception as e:
            logger.error(f"Error setting {key} to {value}: {e}")