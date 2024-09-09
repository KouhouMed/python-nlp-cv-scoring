from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from src.scoring.cv_scorer import BaseScorer
from src.utils.skill_extractor import extract_skills_certifications_projects
from src.config.scorer_config import ScorerConfig
import numpy as np


class BERTScorer(BaseScorer):
    def __init__(self):
        self.config = ScorerConfig()
        model_name = self.config.get("models.bert.model_name", "camembert-base")
        print(f"Loading BERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        print("BERT model loaded successfully")

    def get_bert_embedding(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()

    def cosine_sim(self, a, b):
        return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]

    def score_cv(self, cv_text, job_description):
        cv_embedding = self.get_bert_embedding(cv_text)
        job_embedding = self.get_bert_embedding(job_description)
        base_similarity = self.cosine_sim(cv_embedding, job_embedding)

        # Ensure base_similarity is between 0 and 1
        base_similarity = (base_similarity + 1) / 2

        skills, certifications, projects = extract_skills_certifications_projects(
            cv_text
        )

        # Apply experience multiplier
        total_experience = sum(years for _, years in skills.items())
        max_experience_years = self.config.get("global.max_experience_years", 20)
        experience_multiplier = min(
            1 + (total_experience / max_experience_years),
            self.config.get("skill_scoring.experience_multiplier_cap", 2.0),
        )

        # Apply certification boost
        cert_boost = 1 + (
            len(certifications)
            * self.config.get("skill_scoring.certification_boost", 0.1)
        )

        # Apply project relevance boost
        project_embeddings = [self.get_bert_embedding(project) for project in projects]
        project_relevance = (
            np.mean(
                [
                    self.cosine_sim(proj_emb, job_embedding)
                    for proj_emb in project_embeddings
                ]
            )
            if projects
            else 0
        )
        project_boost = 1 + (
            project_relevance
            * self.config.get("skill_scoring.project_relevance_boost_cap", 0.2)
        )

        # Calculate weighted score
        skill_weight = self.config.get("global.skill_weight", 0.4)
        experience_weight = self.config.get("global.experience_weight", 0.3)
        certification_weight = self.config.get("global.certification_weight", 0.2)
        project_weight = self.config.get("global.project_weight", 0.1)

        final_score = (
            base_similarity * skill_weight
            + (experience_multiplier - 1) * experience_weight
            + (cert_boost - 1) * certification_weight
            + (project_boost - 1) * project_weight
        )

        return min(final_score, 1.0)

    def extract_top_skills(self, cv_text, job_description):
        skills, certifications, projects = extract_skills_certifications_projects(
            cv_text
        )
        job_embedding = self.get_bert_embedding(job_description)

        skill_scores = {}
        for skill, years in skills.items():
            skill_embedding = self.get_bert_embedding(skill)
            base_score = self.cosine_sim(skill_embedding, job_embedding)

            # Ensure base_score is between 0 and 1
            base_score = (base_score + 1) / 2

            # Apply experience multiplier
            max_skill_experience = self.config.get(
                "skill_scoring.max_skill_experience", 10
            )
            experience_multiplier = min(
                1 + (years / max_skill_experience),
                self.config.get("skill_scoring.experience_multiplier_cap", 2.0),
            )

            # Apply certification boost
            cert_boost = (
                self.config.get("skill_scoring.certification_boost", 1.2)
                if any(cert.lower() in skill.lower() for cert in certifications)
                else 1.0
            )

            # Apply project relevance boost
            project_boost = 1.0
            for project in projects:
                project_embedding = self.get_bert_embedding(project)
                project_relevance = self.cosine_sim(project_embedding, job_embedding)
                if skill.lower() in project.lower():
                    project_boost = max(
                        project_boost,
                        1
                        + (
                            project_relevance
                            * self.config.get(
                                "skill_scoring.project_relevance_boost_cap", 0.2
                            )
                        ),
                    )

            # Calculate final score
            final_score = (
                base_score * experience_multiplier * cert_boost * project_boost
            )

            skill_scores[skill] = {
                "compétence": skill,
                "score": final_score,
                "années": years,
            }

        # Normalize scores
        max_score = max(item["score"] for item in skill_scores.values())
        if max_score > 0:
            for skill in skill_scores:
                skill_scores[skill]["score"] /= max_score

        top_skills_count = self.config.get("global.top_skills_count", 3)
        top_skills = sorted(
            skill_scores.values(), key=lambda x: x["score"], reverse=True
        )[:top_skills_count]
        return top_skills
