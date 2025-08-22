import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.experiment_runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(description='Prompt Engineering Experiment Runner')
    
    # Ana komutlar
    parser.add_argument('--list-tasks', action='store_true', 
                       help='Available tasks listesini göster')
    parser.add_argument('--task-info', type=str, 
                       help='Belirli bir task hakkında bilgi al')
    parser.add_argument('--task', type=str, 
                       help='Çalıştırılacak task adı')
    parser.add_argument('--strategies', nargs='+', 
                       help='Kullanılacak prompt stratejileri')
    parser.add_argument('--run-all', action='store_true',
                       help='Tüm task\'ları çalıştır')
    
    # Gelişmiş özellikler
    parser.add_argument('--benchmark', action='store_true',
                       help='Performance benchmark çalıştır')
    parser.add_argument('--compare-strategies', action='store_true',
                       help='Stratejileri karşılaştır')
    parser.add_argument('--config', type=str, default='config/settings.yaml',
                       help='Config dosyası yolu')
    parser.add_argument('--output-format', choices=['console', 'csv', 'json', 'html'],
                       default='console', help='Çıktı formatı')
    
    args = parser.parse_args()
    
    try:
        runner = ExperimentRunner(args.config)
        
        if args.list_tasks:
            tasks = runner.list_available_tasks()
            print("Available tasks:")
            for task in tasks:
                print(f"  - {task}")
        
        elif args.task_info:
            info = runner.get_task_info(args.task_info)
            if "error" in info:
                print(f"Error: {info['error']}")
            else:
                print(f"Task Information for '{args.task_info}':")
                print(f"  name: {info['name']}")
                print(f"  test_data_count: {info['test_data_count']}")
                print(f"  available_strategies: {info['available_strategies']}")
        
        elif args.run_all:
            print("Running all available tasks...")
            all_results = runner.run_all_tasks()
            print("\n" + "="*60)
            print("ALL TASKS COMPLETED")
            print("="*60)
            
            # Özet rapor
            for task_name, results_df in all_results.items():
                print(f"\n{task_name.upper()}:")
                if 'Accuracy' in results_df.columns:
                    avg_acc = results_df['Accuracy'].mean()
                    print(f"  Average Accuracy: {avg_acc:.3f}")
                    best_strategy = results_df.groupby('Prompt Type')['Accuracy'].mean().idxmax()
                    print(f"  Best Strategy: {best_strategy}")
        
        elif args.benchmark:
            try:
                from src.utils.benchmark_runner import BenchmarkRunner
                benchmark = BenchmarkRunner(runner)
                
                print("Running performance benchmarks...")
                for task_name in runner.list_available_tasks():
                    print(f"\nBenchmarking {task_name}...")
                    result = benchmark.run_performance_benchmark(task_name, iterations=3)
                    print(f"Average execution time: {result.get('avg_execution_time', 0):.2f}s")
                
                # Benchmark raporu
                report_df = benchmark.generate_benchmark_report()
                print("\n" + "="*50)
                print("BENCHMARK REPORT")
                print("="*50)
                print(report_df.to_string(index=False))
                
            except ImportError:
                print("Benchmark runner not available")
        
        elif args.compare_strategies:
            if not args.task:
                print("--task parameter required for strategy comparison")
                return
            
            print(f"Comparing all strategies for {args.task}...")
            # Tüm mevcut stratejileri al
            available_strategies = ["vanilla", "zero_shot", "one_shot", "few_shot", "zero_shot_cot", "few_shot_cot"]
            
            results_df = runner.run_single_task(args.task, available_strategies)
            
            print("\n" + "="*50)
            print("STRATEGY COMPARISON")
            print("="*50)
            
            strategy_performance = runner.get_performance_summary(results_df)
            print(strategy_performance.to_string(index=False))
            
            # En iyi ve en kötü strateji
            if 'Accuracy' in results_df.columns:
                avg_by_strategy = results_df.groupby('Prompt Type')['Accuracy'].mean()
                best = avg_by_strategy.idxmax()
                worst = avg_by_strategy.idxmin()
                print(f"\nBest Strategy: {best} ({avg_by_strategy[best]:.3f})")
                print(f"Worst Strategy: {worst} ({avg_by_strategy[worst]:.3f})")
        
        elif args.task:
            results_df = runner.run_single_task(args.task, args.strategies)
            runner.print_results_summary(results_df)
            
            # Çıktı formatına göre kaydet
            if args.output_format != 'console':
                timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if args.output_format == 'csv':
                    filename = f"results_{args.task}_{timestamp}.csv"
                    results_df.to_csv(f"data/output/{filename}", index=False)
                    print(f"\nResults saved to: data/output/{filename}")
                
                elif args.output_format == 'json':
                    filename = f"results_{args.task}_{timestamp}.json"
                    results_df.to_json(f"data/output/{filename}", orient='records', indent=2)
                    print(f"\nResults saved to: data/output/{filename}")
                
                elif args.output_format == 'html':
                    try:
                        from src.analytics.report_generator import ReportGenerator
                        report_gen = ReportGenerator()
                        report_path = report_gen.generate_comprehensive_report({args.task: results_df})
                        print(f"\nHTML report generated: {report_path}")
                    except ImportError:
                        print("HTML report generator not available")
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()