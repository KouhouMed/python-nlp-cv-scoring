import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from src.models.bert_model import BERTScorer
from src.models.word2vec_model import Word2VecScorer
from src.models.chatgpt_model import ChatGPTScorer
from src.config.scorer_config import ScorerConfig


class ModelBenchmarker:
    def __init__(self, cv_folder, job_desc_folder, manual_scores_file):
        self.config = ScorerConfig()
        self.cv_folder = cv_folder
        self.job_desc_folder = job_desc_folder
        self.manual_scores_file = manual_scores_file
        self.models = {
            "BERT": BERTScorer(),
            "Word2Vec": Word2VecScorer(),
            "ChatGPT": ChatGPTScorer(),
        }
        self.results = {}

    def load_text_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def load_manual_scores(self):
        return pd.read_csv(self.manual_scores_file)

    def run_benchmarks(self):
        manual_scores = self.load_manual_scores()

        for index, row in manual_scores.iterrows():
            cv_file = os.path.join(self.cv_folder, row["cv_filename"])
            job_desc_file = os.path.join(self.job_desc_folder, row["job_desc_filename"])

            cv_text = self.load_text_file(cv_file)
            job_desc_text = self.load_text_file(job_desc_file)

            for model_name, model in self.models.items():
                score = model.score_cv(cv_text, job_desc_text)
                top_skills = model.extract_top_skills(cv_text, job_desc_text)

                if model_name not in self.results:
                    self.results[model_name] = []

                self.results[model_name].append(
                    {
                        "cv": row["cv_filename"],
                        "job_desc": row["job_desc_filename"],
                        "model_score": score,
                        "manual_score": row["manual_score"],
                        "top_skills": top_skills,
                    }
                )

    def calculate_metrics(self):
        metrics = {}
        for model_name, scores in self.results.items():
            model_scores = [s["model_score"] for s in scores]
            manual_scores = [s["manual_score"] for s in scores]

            mse = mean_squared_error(manual_scores, model_scores)
            mae = mean_absolute_error(manual_scores, model_scores)
            pearson_corr, _ = pearsonr(manual_scores, model_scores)
            spearman_corr, _ = spearmanr(manual_scores, model_scores)

            metrics[model_name] = {
                "MSE": mse,
                "MAE": mae,
                "Pearson Correlation": pearson_corr,
                "Spearman Correlation": spearman_corr,
            }

        return metrics

    def plot_results(self, metrics):
        self._plot_error_metrics(metrics)
        self._plot_correlation_heatmap(metrics)
        self._plot_scatter_comparisons()
        self._plot_skill_frequency()

    def _plot_error_metrics(self, metrics):
        plt.figure(figsize=(12, 6))
        models = list(metrics.keys())
        mse_values = [m["MSE"] for m in metrics.values()]
        mae_values = [m["MAE"] for m in metrics.values()]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width / 2, mse_values, width, label="MSE")
        rects2 = ax.bar(x + width / 2, mae_values, width, label="MAE")

        ax.set_ylabel("Error")
        ax.set_title("MSE and MAE by Model")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        plt.savefig("benchmark_error_metrics.png")
        plt.close(fig)

    def _plot_correlation_heatmap(self, metrics):
        models = list(metrics.keys())
        corr_data = {
            "Pearson": [m["Pearson Correlation"] for m in metrics.values()],
            "Spearman": [m["Spearman Correlation"] for m in metrics.values()],
        }
        corr_df = pd.DataFrame(corr_data, index=models)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
        plt.title("Correlation Coefficients")
        plt.tight_layout()
        plt.savefig("benchmark_correlation_heatmap.png")
        plt.close()

    def _plot_scatter_comparisons(self):
        for model_name, scores in self.results.items():
            plt.figure(figsize=(10, 8))
            model_scores = [s["model_score"] for s in scores]
            manual_scores = [s["manual_score"] for s in scores]

            plt.scatter(manual_scores, model_scores, alpha=0.6)
            plt.xlabel("Manual Scores")
            plt.ylabel("Model Scores")
            plt.title(f"{model_name} vs Manual Scores")

            # Add diagonal line
            min_score = min(min(manual_scores), min(model_scores))
            max_score = max(max(manual_scores), max(model_scores))
            plt.plot([min_score, max_score], [min_score, max_score], "r--")

            # Add correlation coefficient
            correlation = np.corrcoef(manual_scores, model_scores)[0, 1]
            plt.annotate(
                f"Correlation: {correlation:.4f}",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
            )

            plt.tight_layout()
            plt.savefig(f"benchmark_scatter_{model_name}.png")
            plt.close()

    def _plot_skill_frequency(self):
        all_skills = {}
        for model_scores in self.results.values():
            for score in model_scores:
                for skill in score["top_skills"]:
                    skill_name = skill["compétence"]
                    if skill_name not in all_skills:
                        all_skills[skill_name] = 0
                    all_skills[skill_name] += 1

        # Sort skills by frequency
        sorted_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)
        skills, frequencies = zip(*sorted_skills[:20])  # Top 20 skills

        plt.figure(figsize=(15, 10))
        plt.bar(skills, frequencies)
        plt.xticks(rotation=90)
        plt.xlabel("Skills")
        plt.ylabel("Frequency")
        plt.title("Top 20 Most Frequently Extracted Skills")
        plt.tight_layout()
        plt.savefig("benchmark_skill_frequency.png")
        plt.close()

    def analyze_skill_extraction(self):
        skill_analysis = {model: {} for model in self.models}

        for model_name, scores in self.results.items():
            all_skills = []
            for score in scores:
                all_skills.extend(
                    [skill["compétence"] for skill in score["top_skills"]]
                )

            skill_freq = pd.Series(all_skills).value_counts()
            skill_analysis[model_name] = skill_freq.to_dict()

        return skill_analysis

    def run_full_benchmark(self):
        print("Running benchmarks...")
        self.run_benchmarks()

        print("Calculating metrics...")
        metrics = self.calculate_metrics()

        print("Plotting results...")
        self.plot_results(metrics)

        print("Analyzing skill extraction...")
        skill_analysis = self.analyze_skill_extraction()

        return metrics, skill_analysis

    def save_results(self, output_file):
        with open(output_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Model",
                    "CV",
                    "Job Description",
                    "Model Score",
                    "Manual Score",
                    "Top Skills",
                ]
            )
            for model, scores in self.results.items():
                for score in scores:
                    writer.writerow(
                        [
                            model,
                            score["cv"],
                            score["job_desc"],
                            score["model_score"],
                            score["manual_score"],
                            "; ".join(
                                [
                                    f"{skill['compétence']} ({skill['score']:.2f})"
                                    for skill in score["top_skills"]
                                ]
                            ),
                        ]
                    )
