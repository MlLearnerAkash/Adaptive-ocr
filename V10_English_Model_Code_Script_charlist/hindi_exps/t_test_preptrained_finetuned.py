
#!/usr/bin/env python3
"""
T-Test Calculator for OCR Model Predictions
==========================================

This script calculates t-tests between OCR model predictions and ground truth,
as well as between different models. It supports various file formats and 
provides comprehensive statistical analysis.

Author: Manus AI
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings
import matplotlib.pyplot as plt

class OCRTTestCalculator:
    """
    A comprehensive class for calculating t-tests on OCR model predictions.
    
    Supports multiple file formats and provides various statistical tests
    including paired t-tests, independent t-tests, and accuracy comparisons.
    """
    
    def __init__(self, file_path: str, delimiter: str = None, 
                 column_names: List[str] = None):
        """
        Initialize the calculator with data file.
        
        Args:
            file_path: Path to the text file containing predictions
            delimiter: Delimiter used in the file (auto-detected if None)
            column_names: Names of columns (auto-detected if None)
        """
        self.file_path = Path(file_path)
        self.delimiter = delimiter
        self.column_names = column_names
        self.data = None
        self.results = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and parse the data file with automatic format detection.
        
        Returns:
            DataFrame containing the loaded data
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Try different delimiters if not specified
        delimiters = [self.delimiter] if self.delimiter else ["\t", ",", " ", "|"]
        
        for delim in delimiters:
            try:
                # Read first few lines to determine structure
                with open(self.file_path, "r", encoding="utf-8") as f:
                    sample_lines = [f.readline().strip() for _ in range(5)]
                
                # Skip empty lines
                sample_lines = [line for line in sample_lines if line]
                
                if not sample_lines:
                    raise ValueError("File appears to be empty")
                
                # Check if first line might be header
                first_line_parts = sample_lines[0].split(delim)
                
                # Try to read with pandas
                df = pd.read_csv(self.file_path, delimiter=delim, 
                                header=0 if self._is_header(first_line_parts) else None)
                
                # If successful and has reasonable number of columns
                if len(df.columns) >= 2:
                    self.delimiter = delim
                    break
                    
            except Exception as e:
                continue
        else:
            raise ValueError("Could not parse the file with any common delimiter")
        
        # Set column names if not provided
        if self.column_names:
            if len(self.column_names) != len(df.columns):
                raise ValueError(f"Number of column names ({len(self.column_names)}) "
                               f"doesn't match number of columns ({len(df.columns)})")
            df.columns = self.column_names
        elif df.columns.dtype == "int64":  # No header was detected
            if len(df.columns) == 2:
                df.columns = ["GT", "Model1"]
            elif len(df.columns) == 3:
                df.columns = ["GT", "Model1", "Model2"]
            else:
                df.columns = [f"Column_{i}" for i in range(len(df.columns))]
        
        self.data = df
        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Delimiter detected: '{self.delimiter}'")
        
        return df
    
    def _is_header(self, parts: List[str]) -> bool:
        """Check if the first line appears to be a header."""
        # If any part contains non-numeric characters, likely a header
        for part in parts:
            try:
                float(part)
            except ValueError:
                return True
        return False
    
    def calculate_accuracy(self, gt_col: str, pred_col: str) -> float:
        """
        Calculate accuracy between ground truth and predictions.
        
        Args:
            gt_col: Name of ground truth column
            pred_col: Name of prediction column
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        correct = (self.data[gt_col] == self.data[pred_col]).sum()
        total = len(self.data)
        accuracy = correct / total
        
        print(f"Accuracy for {pred_col}: {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    def calculate_word_recognition_rate(self, gt_col: str, pred_col: str) -> float:
        """
        Calculate Word Recognition Rate (WRR) - same as accuracy for word-level.
        
        Args:
            gt_col: Name of ground truth column
            pred_col: Name of prediction column
            
        Returns:
            WRR as a float between 0 and 1
        """
        return self.calculate_accuracy(gt_col, pred_col)
    
    def paired_ttest(self, model1_col: str, model2_col: str, gt_col: str) -> Dict:
        """
        Perform paired t-test between two models' accuracies.
        
        Args:
            model1_col: Name of first model's prediction column
            model2_col: Name of second model's prediction column
            gt_col: Name of ground truth column
            
        Returns:
            Dictionary containing test results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Calculate per-sample correctness
        correct1 = (self.data[gt_col] == self.data[model1_col]).astype(int)
        correct2 = (self.data[gt_col] == self.data[model2_col]).astype(int)
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(correct1, correct2)
        
        # Calculate effect size (Cohen's d for paired samples)
        diff = correct1 - correct2
        cohen_d = np.mean(diff) / np.std(diff, ddof=1)
        
        # Calculate accuracies
        acc1 = np.mean(correct1)
        acc2 = np.mean(correct2)
        
        result = {
            'test_type': 'paired_ttest',
            'model1': model1_col,
            'model2': model2_col,
            'model1_accuracy': acc1,
            'model2_accuracy': acc2,
            'accuracy_difference': acc1 - acc2,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohen_d': cohen_d,
            'degrees_freedom': len(correct1) - 1,
            'sample_size': len(correct1),
            'significant': p_value < 0.05
        }
        
        self.results[f'paired_ttest_{model1_col}_vs_{model2_col}'] = result
        return result
    
    def mcnemar_test(self, model1_col: str, model2_col: str, gt_col: str) -> Dict:
        """
        Perform McNemar's test for comparing two models.
        
        Args:
            model1_col: Name of first model's prediction column
            model2_col: Name of second model's prediction column
            gt_col: Name of ground truth column
            
        Returns:
            Dictionary containing test results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Calculate correctness for both models
        correct1 = (self.data[gt_col] == self.data[model1_col])
        correct2 = (self.data[gt_col] == self.data[model2_col])
        
        # Create contingency table
        both_correct = (correct1 & correct2).sum()
        model1_only = (correct1 & ~correct2).sum()
        model2_only = (~correct1 & correct2).sum()
        both_wrong = (~correct1 & ~correct2).sum()
        
        # McNemar's test statistic
        if model1_only + model2_only == 0:
            # No discordant pairs
            chi2_stat = 0
            p_value = 1.0
        else:
            chi2_stat = (abs(model1_only - model2_only) - 1)**2 / (model1_only + model2_only)
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        
        result = {
            'test_type': 'mcnemar_test',
            'model1': model1_col,
            'model2': model2_col,
            'both_correct': both_correct,
            'model1_only_correct': model1_only,
            'model2_only_correct': model2_only,
            'both_wrong': both_wrong,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        self.results[f'mcnemar_{model1_col}_vs_{model2_col}'] = result
        return result
    
    def bootstrap_confidence_interval(self, gt_col: str, pred_col: str, 
                                    n_bootstrap: int = 1000, 
                                    confidence: float = 0.95) -> Dict:
        """
        Calculate bootstrap confidence interval for accuracy.
        
        Args:
            gt_col: Name of ground truth column
            pred_col: Name of prediction column
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary containing confidence interval results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        correct = (self.data[gt_col] == self.data[pred_col]).astype(int)
        n_samples = len(correct)
        
        # Bootstrap sampling
        bootstrap_accuracies = []
        np.random.seed(42)  # For reproducibility
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(correct, size=n_samples, replace=True)
            bootstrap_accuracy = np.mean(bootstrap_sample)
            bootstrap_accuracies.append(bootstrap_accuracy)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_accuracies, lower_percentile)
        ci_upper = np.percentile(bootstrap_accuracies, upper_percentile)
        
        result = {
            'model': pred_col,
            'accuracy': np.mean(correct),
            'confidence_level': confidence,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'n_bootstrap': n_bootstrap
        }
        
        return result
    
    def comprehensive_analysis(self) -> Dict:
        """
        Perform comprehensive statistical analysis on all available models.
        
        Returns:
            Dictionary containing all analysis results
        """
        if self.data is None:
            self.load_data()
        
        results = {'summary': {}, 'tests': {}}
        
        # Identify GT column and model columns
        columns = list(self.data.columns)
        gt_col = None
        model_cols = []
        
        # Try to identify GT column
        for col in columns:
            if 'gt' in col.lower() or 'ground' in col.lower() or 'truth' in col.lower():
                gt_col = col
                break
        
        if gt_col is None:
            gt_col = columns[0]  # Assume first column is GT
            print(f"Warning: GT column not clearly identified. Using '{gt_col}' as ground truth.")
        
        # Remaining columns are model predictions
        model_cols = [col for col in columns if col != gt_col]
        
        if not model_cols:
            raise ValueError("No model prediction columns found")
        
        print(f"Ground Truth column: {gt_col}")
        print(f"Model columns: {model_cols}")
        
        # Calculate accuracies and confidence intervals
        for model_col in model_cols:
            accuracy = self.calculate_accuracy(gt_col, model_col)
            ci = self.bootstrap_confidence_interval(gt_col, model_col)
            
            results['summary'][model_col] = {
                'accuracy': accuracy,
                'confidence_interval': ci
            }
        
        # Perform pairwise comparisons if multiple models
        if len(model_cols) >= 2:
            for i in range(len(model_cols)):
                for j in range(i + 1, len(model_cols)):
                    model1, model2 = model_cols[i], model_cols[j]
                    
                    # Paired t-test
                    ttest_result = self.paired_ttest(model1, model2, gt_col)
                    results['tests'][f'ttest_{model1}_vs_{model2}'] = ttest_result
                    
                    # McNemar's test
                    mcnemar_result = self.mcnemar_test(model1, model2, gt_col)
                    results['tests'][f'mcnemar_{model1}_vs_{model2}'] = mcnemar_result
        
        self.results.update(results)
        return results
    
    def print_results(self, results: Dict = None):
        """Print formatted results."""
        if results is None:
            results = self.results
        
        print("\n" + "="*60)
        print("OCR MODEL COMPARISON RESULTS")
        print("="*60)
        
        # Print summary
        if 'summary' in results:
            print("\nMODEL ACCURACIES:")
            print("-" * 40)
            for model, data in results['summary'].items():
                acc = data['accuracy']
                ci = data['confidence_interval']
                print(f"{model:15s}: {acc:.4f} (95% CI: {ci['ci_lower']:.4f}-{ci['ci_upper']:.4f})")
        
        # Print statistical tests
        if 'tests' in results:
            print("\nSTATISTICAL TESTS:")
            print("-" * 40)
            
            for test_name, test_data in results['tests'].items():
                if test_data['test_type'] == 'paired_ttest':
                    print(f"\nPaired t-test: {test_data['model1']} vs {test_data['model2']}")
                    print(f"  Accuracy difference: {test_data['accuracy_difference']:.4f}")
                    print(f"  t-statistic: {test_data['t_statistic']:.4f}")
                    print(f"  p-value: {test_data['p_value']:.6f}")
                    print(f"  Cohen's d: {test_data['cohen_d']:.4f}")
                    print(f"  Significant: {'Yes' if test_data['significant'] else 'No'}")
                
                elif test_data['test_type'] == 'mcnemar_test':
                    print(f"\nMcNemar's test: {test_data['model1']} vs {test_data['model2']}")
                    print(f"  Both correct: {test_data['both_correct']}")
                    print(f"  Only {test_data['model1']} correct: {test_data['model1_only_correct']}")
                    print(f"  Only {test_data['model2']} correct: {test_data['model2_only_correct']}")
                    print(f"  Both wrong: {test_data['both_wrong']}")
                    print(f"  Chi-square statistic: {test_data['chi2_statistic']:.4f}")
                    print(f"  p-value: {test_data['p_value']:.6f}")
                    print(f"  Significant: {'Yes' if test_data['significant'] else 'No'}")

    def plot_accuracies(self, results: Dict = None, output_path: str = "accuracies.png"):
        """
        Generate a bar plot of model accuracies with confidence intervals.
        
        Args:
            results: Dictionary containing analysis results (defaults to self.results)
            output_path: Path to save the plot
        """
        if results is None:
            results = self.results
        
        if not results or 'summary' not in results:
            print("No summary results available for plotting accuracies.")
            return
        
        models = []
        accuracies = []
        ci_lowers = []
        ci_uppers = []
        
        for model, data in results["summary"].items():
            models.append(model)
            accuracies.append(data["accuracy"])
            ci_lowers.append(data["confidence_interval"]["ci_lower"])
            ci_uppers.append(data["confidence_interval"]["ci_upper"])
            
        # Calculate error bars
        errors = [np.array(accuracies) - np.array(ci_lowers), 
                  np.array(ci_uppers) - np.array(accuracies)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(models, accuracies, yerr=errors, capsize=5, color="#4CAF50")
        ax.set_ylabel("Word Recognition Rate (WRR)")
        ax.set_title("OCR Model Accuracies with 95% Confidence Intervals")
        ax.set_ylim(bottom=0, top=1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, acc in enumerate(accuracies):
            ax.text(i, acc + 0.02, f"{acc:.4f}", ha='center', va='bottom')
            
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Accuracy plot saved to: {output_path}")
        plt.close()
        
    def generate_report(self, results: Dict = None, output_path: str = "report.md"):
        """
        Generate a Markdown report of the analysis results.
        
        Args:
            results: Dictionary containing analysis results (defaults to self.results)
            output_path: Path to save the report
        """
        if results is None:
            results = self.results
            
        report_content = []
        report_content.append("# OCR Model Comparison Report\n")
        report_content.append("This report summarizes the statistical analysis of OCR model predictions.\n")
        
        # Fix the f-string issue by using format() instead
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        report_content.append(f"Generated on: {timestamp}\n")
        
        # Summary
        if 'summary' in results:
            report_content.append("## 1. Model Accuracies\n")
            report_content.append("| Model | Accuracy | 95% Confidence Interval |\n")
            report_content.append("|-------|----------|-------------------------|\n")
            for model, data in results["summary"].items():
                acc = data["accuracy"]
                ci = data["confidence_interval"]
                report_content.append(f"| {model} | {acc:.4f} | [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] |\n")
            report_content.append("\n")
            
        # Statistical Tests
        if 'tests' in results:
            report_content.append("## 2. Statistical Tests\n")
            for test_name, test_data in results["tests"].items():
                if test_data["test_type"] == "paired_ttest":
                    report_content.append(f"### 2.1 Paired t-test: {test_data['model1']} vs {test_data['model2']}\n")
                    report_content.append(f"- **Accuracy Difference**: {test_data['accuracy_difference']:.4f}\n")
                    report_content.append(f"- **t-statistic**: {test_data['t_statistic']:.4f}\n")
                    report_content.append(f"- **p-value**: {test_data['p_value']:.6f}\n")
                    report_content.append(f"- **Cohen's d**: {test_data['cohen_d']:.4f}\n")
                    report_content.append(f"- **Significant (p < 0.05)**: {'Yes' if test_data['significant'] else 'No'}\n\n")
                elif test_data["test_type"] == "mcnemar_test":
                    report_content.append(f"### 2.2 McNemar's Test: {test_data['model1']} vs {test_data['model2']}\n")
                    report_content.append(f"- **Both Correct**: {test_data['both_correct']}\n")
                    report_content.append(f"- **{test_data['model1']} Only Correct**: {test_data['model1_only_correct']}\n")
                    report_content.append(f"- **{test_data['model2']} Only Correct**: {test_data['model2_only_correct']}\n")
                    report_content.append(f"- **Both Wrong**: {test_data['both_wrong']}\n")
                    report_content.append(f"- **Chi-square statistic**: {test_data['chi2_statistic']:.4f}\n")
                    report_content.append(f"- **p-value**: {test_data['p_value']:.6f}\n")
                    report_content.append(f"- **Significant (p < 0.05)**: {'Yes' if test_data['significant'] else 'No'}\n\n")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(report_content)
        print(f"Report saved to: {output_path}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Calculate t-tests for OCR model predictions')
    parser.add_argument('--file_path', help='Path to the text file containing predictions')
    parser.add_argument('--delimiter', '-d', help='Delimiter used in the file')
    parser.add_argument('--columns', '-c', nargs='+', 
                       help='Column names (e.g., GT Model1 Model2)')
    parser.add_argument('--output', '-o', help='Output file for raw JSON results')
    parser.add_argument('--plot', action='store_true', help='Generate accuracy plot (accuracies.png)')
    parser.add_argument('--report', action='store_true', help='Generate markdown report (report.md)')
    
    args = parser.parse_args()
    
    try:
        # Initialize calculator
        calculator = OCRTTestCalculator(
            file_path=args.file_path,
            delimiter=args.delimiter,
            column_names=args.columns
        )
        
        # Perform comprehensive analysis
        results = calculator.comprehensive_analysis()
        
        # Print results to console
        calculator.print_results(results)
        
        # Save raw JSON results if output file specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nRaw results saved to: {args.output}")
        
        # Generate plot if requested
        if args.plot:
            calculator.plot_accuracies(results, output_path="accuracies.png")
        
        # Generate report if requested
        if args.report:
            calculator.generate_report(results, output_path="report.md")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()