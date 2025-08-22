from typing import Dict, Any, Optional
import yaml
import os

class AdvancedConfig:
    def __init__(self, base_config_path: str = "config/settings.yaml"):
        self.base_config = self._load_config(base_config_path)
        self.runtime_overrides = {}
        
    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def override_setting(self, key: str, value: Any):
        """Runtime'da ayar değiştir"""
        self.runtime_overrides[key] = value
        print(f"Override applied: {key} = {value}")
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Override'lar uygulanmış final config"""
        config = self.base_config.copy()
        for key, value in self.runtime_overrides.items():
            self._set_nested_value(config, key, value)
        return config
    
    def _set_nested_value(self, config: Dict, key: str, value: Any):
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    def save_current_config(self, output_path: str):
        """Mevcut konfigürasyonu kaydet"""
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.get_effective_config(), f, ensure_ascii=False, indent=2)