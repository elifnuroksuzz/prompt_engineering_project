import time
import psutil
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import gc

class BenchmarkRunner:
    def __init__(self, experiment_runner):
        self.experiment_runner = experiment_runner
        self.benchmark_results = []
    
    def run_performance_benchmark(self, task_name: str, iterations: int = 3) -> Dict[str, Any]:
        """Performans benchmark'ı çalıştır"""
        results = {
            'task_name': task_name,
            'iterations': iterations,
            'execution_times': [],
            'memory_usage_mb': [],
            'timestamp': datetime.now().isoformat()
        }
        
        for i in range(iterations):
            print(f"Benchmark iteration {i+1}/{iterations} for {task_name}")
            
            # Memory temizle
            gc.collect()
            
            # Başlangıç ölçümleri
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Task çalıştır
            try:
                _ = self.experiment_runner.run_single_task(task_name)
                success = True
            except Exception as e:
                print(f"Benchmark iteration {i+1} failed: {e}")
                success = False
            
            # Bitiş ölçümleri
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            if success:
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                results['execution_times'].append(execution_time)
                results['memory_usage_mb'].append(max(0, memory_delta))  # Negatif değerleri önle
                
                print(f"  Execution time: {execution_time:.2f}s")
                print(f"  Memory delta: {memory_delta:.2f}MB")
        
        # İstatistikler
        if results['execution_times']:
            results['avg_execution_time'] = sum(results['execution_times']) / len(results['execution_times'])
            results['avg_memory_usage'] = sum(results['memory_usage_mb']) / len(results['memory_usage_mb'])
            results['min_execution_time'] = min(results['execution_times'])
            results['max_execution_time'] = max(results['execution_times'])
        
        self.benchmark_results.append(results)
        return results
    
    def generate_benchmark_report(self) -> pd.DataFrame:
        """Benchmark raporu oluştur"""
        if not self.benchmark_results:
            return pd.DataFrame()
        
        report_data = []
        for result in self.benchmark_results:
            report_data.append({
                'Task': result['task_name'],
                'Iterations': result['iterations'],
                'Avg Time (s)': f"{result.get('avg_execution_time', 0):.2f}",
                'Min Time (s)': f"{result.get('min_execution_time', 0):.2f}",
                'Max Time (s)': f"{result.get('max_execution_time', 0):.2f}",
                'Avg Memory (MB)': f"{result.get('avg_memory_usage', 0):.2f}",
                'Timestamp': result['timestamp']
            })
        
        return pd.DataFrame(report_data)