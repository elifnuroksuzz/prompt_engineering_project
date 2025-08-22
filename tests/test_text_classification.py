import unittest
import sys
import os

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.config import Config
from src.core.model_manager import ModelManager
from src.prompts.prompt_library import PromptLibrary
from src.tasks.text_classification import TextClassificationTask

class TestTextClassification(unittest.TestCase):
    def setUp(self):
        # Test konfigürasyonu
        os.environ['GEMINI_API_KEY'] = 'test_key'
        self.config = Config('config/settings.yaml')
        
        # Mock model manager (gerçek API çağrısı yapmadan test için)
        class MockModelManager:
            def generate(self, prompt):
                return "Olumlu"  # Test için sabit yanıt
            
            def is_ready(self):
                return True
        
        self.model_manager = MockModelManager()
        self.prompt_library = PromptLibrary()
        self.task = TextClassificationTask(
            self.model_manager, 
            self.prompt_library, 
            self.config
        )
    
    def test_label_extraction(self):
        """Etiket çıkarma fonksiyonunu test et"""
        test_cases = [
            ("Olumlu", "Olumlu"),
            ("Bu metin olumsuz bir duygu içeriyor", "Olumsuz"),
            ("Nötr bir ifade", "Nötr"),
            ("Belirsiz yanıt", None)
        ]
        
        for response, expected in test_cases:
            with self.subTest(response=response):
                result = self.task._extract_label(response)
                self.assertEqual(result, expected)
    
    def test_prompt_generation(self):
        """Prompt oluşturma fonksiyonunu test et"""
        data_item = {"input_text": "Test metni"}
        
        prompt = self.task._generate_prompt("zero_shot", data_item)
        self.assertIn("Test metni", prompt)
        self.assertIn("sınıflandır", prompt)

if __name__ == '__main__':
    unittest.main()