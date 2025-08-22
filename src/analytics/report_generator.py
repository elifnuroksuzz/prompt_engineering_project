import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import json
from datetime import datetime
import os

class ReportGenerator:
    def __init__(self, output_dir: str = "data/output/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_comprehensive_report(self, all_results: Dict[str, pd.DataFrame]) -> str:
        """Kapsamlı analiz raporu oluştur"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"comprehensive_report_{timestamp}.html")
        
        html_content = self._create_html_report(all_results)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _create_html_report(self, all_results: Dict[str, pd.DataFrame]) -> str:
        """HTML raporu oluştur"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prompt Engineering Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; }
                .task-section { margin: 30px 0; border: 1px solid #ddd; padding: 20px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 3px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .chart { margin: 20px 0; }
            </style>
        </head>
        <body>
        """
        
        # Header
        html += f"""
        <div class="header">
            <h1>Prompt Engineering Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total Tasks Analyzed: {len(all_results)}</p>
        </div>
        """
        
        # Task-specific analysis
        for task_name, results_df in all_results.items():
            html += self._create_task_section(task_name, results_df)
        
        # Overall comparison
        html += self._create_comparison_section(all_results)
        
        html += "</body></html>"
        return html
    
    def _create_task_section(self, task_name: str, results_df: pd.DataFrame) -> str:
        """Görev-spesifik bölüm oluştur"""
        section = f"""
        <div class="task-section">
            <h2>{task_name.replace('_', ' ').title()}</h2>
        """
        
        if 'Accuracy' in results_df.columns:
            # Performance metrics
            avg_accuracy = results_df['Accuracy'].mean()
            best_strategy = results_df.groupby('Prompt Type')['Accuracy'].mean().idxmax()
            
            section += f"""
            <div class="metric">Average Accuracy: <strong>{avg_accuracy:.3f}</strong></div>
            <div class="metric">Best Strategy: <strong>{best_strategy}</strong></div>
            <div class="metric">Total Tests: <strong>{len(results_df)}</strong></div>
            """
            
            # Strategy comparison table
            strategy_stats = results_df.groupby('Prompt Type')['Accuracy'].agg(['mean', 'std', 'count'])
            section += "<h3>Strategy Performance</h3>"
            section += strategy_stats.to_html(classes="strategy-table")
        
        section += "</div>"
        return section
    
    def _create_comparison_section(self, all_results: Dict[str, pd.DataFrame]) -> str:
        """Karşılaştırma bölümü"""
        section = """
        <div class="task-section">
            <h2>Cross-Task Analysis</h2>
        """
        
        # Task performance summary
        task_summary = {}
        for task_name, results_df in all_results.items():
            if 'Accuracy' in results_df.columns:
                task_summary[task_name] = {
                    'avg_accuracy': results_df['Accuracy'].mean(),
                    'total_tests': len(results_df),
                    'best_strategy': results_df.groupby('Prompt Type')['Accuracy'].mean().idxmax()
                }
        
        section += "<h3>Task Performance Summary</h3><table>"
        section += "<tr><th>Task</th><th>Avg Accuracy</th><th>Total Tests</th><th>Best Strategy</th></tr>"
        
        for task, stats in task_summary.items():
            section += f"""
            <tr>
                <td>{task.replace('_', ' ').title()}</td>
                <td>{stats['avg_accuracy']:.3f}</td>
                <td>{stats['total_tests']}</td>
                <td>{stats['best_strategy']}</td>
            </tr>
            """
        
        section += "</table></div>"
        return section