import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from src.scoring.cv_scorer import BaseScorer
from src.utils.skill_extractor import extract_skills_certifications_projects
from src.config.scorer_config import ScorerConfig


class Word2VecScorer(BaseScorer):
    def __init__(self):
        self.config = ScorerConfig()
        model_name = self.config.get(
            "models.word2vec.model_name", "fasttext-wiki-news-subwords-300"
        )
        print(f"Loading Word2Vec model: {model_name}")
        self.model = api.load(model_name)
        print("Word2Vec model loaded successfully")

    def get_document_vector(self, text):
        words = text.lower().split()
        word_vectors = [self.model[word] for word in words if word in self.model]
        if not word_vectors:
            return np.zeros(self.model.vector_size)
        return np.mean(word_vectors, axis=0)

    def cosine_sim(self, a, b):
        return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]

    def score_cv(self, cv_text, job_description):
        cv_vector = self.get_document_vector(cv_text)
        job_vector = self.get_document_vector(job_description)
        base_similarity = self.cosine_sim(cv_vector, job_vector)

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
        project_vectors = [self.get_document_vector(project) for project in projects]
        project_relevance = (
            np.mean(
                [self.cosine_sim(proj_vec, job_vector) for proj_vec in project_vectors]
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
        job_vector = self.get_document_vector(job_description)

        skill_scores = {}
        for skill, years in skills.items():
            skill_vector = self.get_document_vector(skill)
            base_score = self.cosine_sim(skill_vector, job_vector)

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
                project_vector = self.get_document_vector(project)
                project_relevance = self.cosine_sim(project_vector, job_vector)
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
