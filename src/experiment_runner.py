from typing import Dict, List, Any
import pandas as pd
from .core.config import Config
from .core.model_manager import ModelManager
from .prompts.prompt_library import PromptLibrary
from .tasks.text_classification import TextClassificationTask
from .evaluation.metrics import EvaluationMetrics
from .utils.data_handler import DataHandler
from .analytics.report_generator import ReportGenerator

class ExperimentRunner:
    def __init__(self, config_path: str = "config/settings.yaml"):
        print("ExperimentRunner initializing...")
        self.config = Config(config_path)
        print("Config loaded")
        self.model_manager = ModelManager(self.config)
        print("Model manager created")
        self.prompt_library = PromptLibrary()
        print("Prompt library created")
        self.evaluator = EvaluationMetrics()
        self.data_handler = DataHandler(self.config.get('evaluation.output_dir', 'data/output'))
        self.report_generator = ReportGenerator()
        self.tasks = {}
        print("About to initialize tasks...")
        self._initialize_tasks()
        print(f"Initialization complete. Tasks: {list(self.tasks.keys())}")
    
    def _initialize_tasks(self):
        """Görevleri başlat"""
        print("Starting task initialization...")
        
        # Text classification
        if self.config.get('tasks.text_classification.enabled', True):
            self.tasks['text_classification'] = TextClassificationTask(
                self.model_manager, 
                self.prompt_library, 
                self.config
            )
            print("Text classification task added")
        
        # Mathematical reasoning
        try:
            from .tasks.mathematical_reasoning import MathematicalReasoningTask
            self.tasks['mathematical_reasoning'] = MathematicalReasoningTask(
                self.model_manager, 
                self.prompt_library, 
                self.config
            )
            print("Mathematical reasoning task added")
        except Exception as e:
            print(f"Error adding mathematical reasoning task: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"Final tasks dictionary: {list(self.tasks.keys())}")
    
    def run_single_task(self, task_name: str, strategies: List[str] = None) -> pd.DataFrame:
        """Tek bir görevi çalıştır"""
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not found. Available tasks: {list(self.tasks.keys())}")
        
        if strategies is None:
            strategies = self.config.get(f'tasks.{task_name}.strategies', 
                                       ['zero_shot', 'one_shot', 'few_shot'])
        
        print(f"Running {task_name} with strategies: {strategies}")
        task = self.tasks[task_name]
        results_df = task.run_experiment(strategies)
        
        # Sonuçları kaydet
        if self.config.get('evaluation.save_results', True):
            filepath = self.data_handler.save_results(results_df, task_name)
            print(f"Results saved to: {filepath}")
        
        return results_df
    
    def run_all_tasks(self) -> Dict[str, pd.DataFrame]:
        """Tüm görevleri çalıştır"""
        all_results = {}
        
        for task_name in self.tasks.keys():
            try:
                results_df = self.run_single_task(task_name)
                all_results[task_name] = results_df
                print(f"✓ {task_name} completed")
            except Exception as e:
                print(f"✗ {task_name} failed: {str(e)}")
        
        # Genel özet rapor oluştur
        summary = self.data_handler.create_summary_report(all_results)
        self.data_handler.save_json(summary, "experiment_summary.json")
        
        # HTML raporu oluştur
        try:
            report_path = self.report_generator.generate_comprehensive_report(all_results)
            print(f"Comprehensive report generated: {report_path}")
        except Exception as e:
            print(f"Warning: Could not generate HTML report: {e}")
        
        return all_results
    
    def get_performance_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Performans özetini getir"""
        return self.evaluator.calculate_strategy_performance(results_df)
    
    def print_results_summary(self, results_df: pd.DataFrame):
        """Sonuçları konsola yazdır"""
        print("\n" + "="*50)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*50)
        
        if 'Accuracy' in results_df.columns:
            performance = self.get_performance_summary(results_df)
            print("\nPerformance by Strategy:")
            print(performance.to_string(index=False))
        
        print(f"\nTotal tests: {len(results_df)}")
        print(f"Successful tests: {results_df['Accuracy'].notna().sum()}")
        
        if 'Accuracy' in results_df.columns:
            avg_accuracy = results_df['Accuracy'].mean()
            print(f"Average accuracy: {avg_accuracy:.3f}")
    
    def add_custom_task(self, task_name: str, task_instance):
        """Özel görev ekleme"""
        self.tasks[task_name] = task_instance
        print(f"Custom task '{task_name}' added")
    
    def list_available_tasks(self) -> List[str]:
        """Mevcut görevleri listele"""
        return list(self.tasks.keys())
    
    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """Görev hakkında bilgi al"""
        if task_name not in self.tasks:
            return {"error": f"Task '{task_name}' not found"}
        
        task = self.tasks[task_name]
        return {
            "name": task.get_task_name(),
            "test_data_count": len(task.get_test_data()),
            "available_strategies": ["zero_shot", "one_shot", "few_shot"] # Default
        }