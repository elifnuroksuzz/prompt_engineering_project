import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

class DataHandler:
    def __init__(self, output_dir: str = "data/output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_results(self, results_df: pd.DataFrame, task_name: str) -> str:
        """Sonuçları dosyaya kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_name}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        results_df.to_csv(filepath, index=False, encoding='utf-8')
        return filepath
    
    def save_json(self, data: Dict[str, Any], filename: str) -> str:
        """JSON verisi kaydet"""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        return filepath
    
    def load_results(self, filepath: str) -> pd.DataFrame:
        """Kaydedilmiş sonuçları yükle"""
        return pd.read_csv(filepath, encoding='utf-8')
    
    def create_summary_report(self, all_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Tüm görevler için özet rapor oluştur"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "tasks": {},
            "overall_stats": {}
        }
        
        total_tests = 0
        total_accuracy = []
        
        for task_name, results_df in all_results.items():
            if not results_df.empty and 'Accuracy' in results_df.columns:
                valid_accuracies = results_df['Accuracy'].dropna()
                
                summary["tasks"][task_name] = {
                    "total_tests": len(results_df),
                    "successful_tests": len(valid_accuracies),
                    "average_accuracy": float(valid_accuracies.mean()) if not valid_accuracies.empty else 0.0,
                    "best_strategy": self._get_best_strategy(results_df)
                }
                
                total_tests += len(results_df)
                total_accuracy.extend(valid_accuracies.tolist())
        
        summary["overall_stats"] = {
            "total_tests": total_tests,
            "overall_accuracy": float(np.mean(total_accuracy)) if total_accuracy else 0.0
        }
        
        return summary
    
    def _get_best_strategy(self, results_df: pd.DataFrame) -> str:
        """En iyi performans gösteren stratejiyi bul"""
        if 'Prompt Type' not in results_df.columns or 'Accuracy' not in results_df.columns:
            return "Unknown"
        
        strategy_performance = results_df.groupby('Prompt Type')['Accuracy'].mean()
        if not strategy_performance.empty:
            return strategy_performance.idxmax()
        
        return "Unknown"