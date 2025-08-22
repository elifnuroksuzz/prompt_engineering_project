import google.generativeai as genai
import time
import hashlib
from typing import Optional, Dict
from .config import Config

class ModelManager:
    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self.mock_mode = config.get('model.mock_mode', False)
        
        # Mock responses cache - farklı promptlar için farklı yanıtlar
        self.mock_responses = self._initialize_mock_responses()
        
        if not self.mock_mode:
            self._initialize_model()
        else:
            print("Mock mode enabled - using realistic simulated responses")
    
    def _initialize_model(self) -> None:
        """Gemini modelini başlat"""
        genai.configure(api_key=self.config.gemini_api_key)
        self._model = genai.GenerativeModel(self.config.model_name)
    
    def _initialize_mock_responses(self) -> Dict[str, str]:
        """Farklı problem türleri için farklı mock yanıtları"""
        return {
            # Mathematical reasoning responses
            "elma_portakal": """
Adım adım çözelim:

1. Elma fiyatına 'e', portakal fiyatına 'p' diyelim
2. Problemden: 5e + 3p = 42
3. Ayrıca: e = p + 2 (elma 2 TL daha pahalı)
4. İkinci denklemi birincide yerine koyalım: 5(p + 2) + 3p = 42
5. Açalım: 5p + 10 + 3p = 42
6. Birleştirelim: 8p + 10 = 42
7. 8p = 32
8. p = 4

Portakal fiyatı 4 TL, elma fiyatı: e = 4 + 2 = 6 TL

Yanıt: 1 kilo elma: 6 TL, 1 kilo portakal: 4 TL
""",
            "kahve_cay": """
Adım adım çözelim:

1. Kahve fiyatına 'k', çay fiyatına 'ç' diyelim
2. Problemden: 4k + 2ç = 26
3. Ayrıca: k = ç + 3 (kahve 3 TL daha pahalı)
4. İkinci denklemi birincide yerine koyalım: 4(ç + 3) + 2ç = 26
5. Açalım: 4ç + 12 + 2ç = 26
6. Birleştirelim: 6ç + 12 = 26
7. 6ç = 14
8. ç = 2.33... ≈ 2

Çay fiyatı 2 TL, kahve fiyatı: k = 2 + 3 = 5 TL

Yanıt: 1 kahve: 5 TL, 1 çay: 2 TL
""",
            "bilet": """
Adım adım çözelim:

1. Çocuk biletine 'ç', yetişkin biletine 'y' diyelim
2. Problemden: 6ç + 4y = 84
3. Ayrıca: ç = y - 6 (çocuk bileti 6 TL daha ucuz)
4. İkinci denklemi birincide yerine koyalım: 6(y - 6) + 4y = 84
5. Açalım: 6y - 36 + 4y = 84
6. Birleştirelim: 10y - 36 = 84
7. 10y = 120
8. y = 12

Yetişkin bileti 12 TL, çocuk bileti: ç = 12 - 6 = 6 TL

Yanıt: Çocuk bileti: 6 TL, Yetişkin bileti: 12 TL
""",
            # Sentiment classification responses
            "positive": "Olumlu",
            "negative": "Olumsuz", 
            "neutral": "Nötr"
        }
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Prompt ile metin üret - gerçekçi mock responses ile"""
        if self.mock_mode:
            return self._generate_smart_mock_response(prompt)
        
        for attempt in range(max_retries):
            try:
                response = self._model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                error_msg = str(e)
                
                # 429 (rate limit) hatası için bekleme
                if "429" in error_msg and attempt < max_retries - 1:
                    wait_time = 60 * (attempt + 1)  # 60, 120, 180 saniye
                    print(f"Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                raise RuntimeError(f"Model generation failed: {error_msg}")
    
    def _generate_smart_mock_response(self, prompt: str) -> str:
        """Prompt içeriğine göre akıllı mock yanıt üret"""
        prompt_lower = prompt.lower()
        
        # Mathematical reasoning problems
        if "elma" in prompt_lower and "portakal" in prompt_lower:
            return self.mock_responses["elma_portakal"]
        elif "kahve" in prompt_lower and "çay" in prompt_lower:
            return self.mock_responses["kahve_cay"]
        elif "bilet" in prompt_lower or ("çocuk" in prompt_lower and "yetişkin" in prompt_lower):
            return self.mock_responses["bilet"]
        
        # Sentiment classification 
        elif "sınıflandır" in prompt_lower or "duygu" in prompt_lower:
            # Prompt içeriğine göre sentiment belirle
            if any(word in prompt_lower for word in ["güzel", "harika", "sevdim", "tavsiye"]):
                return self.mock_responses["positive"]
            elif any(word in prompt_lower for word in ["kötü", "sinir", "hayal kırıklığı", "geç"]):
                return self.mock_responses["negative"]  
            else:
                return self.mock_responses["neutral"]
        
        # Default fallback
        else:
            return "Mock response - content not recognized"
    
    def is_ready(self) -> bool:
        """Model hazır mı kontrolü"""
        return self.mock_mode or self._model is not None