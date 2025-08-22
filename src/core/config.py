import os
import yaml
from dotenv import load_dotenv
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str = "config/settings.yaml"):
        load_dotenv("config/.env")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            self._config = yaml.safe_load(file)
        
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Noktalı notasyonla config değerlerine erişim (örn: 'model.name')"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    @property
    def model_name(self) -> str:
        return self.get('model.name', 'gemini-2.5-flash')
    
    @property
    def temperature(self) -> float:
        return self.get('model.temperature', 0.1)