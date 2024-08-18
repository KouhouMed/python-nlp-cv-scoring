import re

SKILLS = [
    "Python", "Java", "C++", "JavaScript", "TypeScript", "PHP", "Ruby", "Go", "Rust",
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Oracle", "SQLite",
    "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Express.js", "Django", "Flask",
    "Spring", "Hibernate", "ASP.NET", "Ruby on Rails",
    "Machine Learning", "Deep Learning", "Apprentissage automatique", "Apprentissage profond",
    "Traitement du langage naturel", "Vision par ordinateur",
    "AWS", "Azure", "Google Cloud", "Heroku", "Docker", "Kubernetes",
    "Git", "SVN", "CI/CD", "Jenkins", "GitLab CI", "GitHub Actions",
    "Agile", "Scrum", "Kanban", "DevOps", "TDD", "BDD",
    "Analyse de données", "Big Data", "Hadoop", "Spark", "Tableau", "Power BI",
    "Excel", "VBA", "R", "SPSS", "SAS",
    "Gestion de projet", "Leadership", "Communication", "Travail d'équipe",
    "Français", "Anglais", "Allemand", "Espagnol", "Italien", "Chinois", "Arabe",
    "Marketing digital", "SEO", "SEM", "Réseaux sociaux", "Contenu marketing",
    "Adobe Creative Suite", "Photoshop", "Illustrator", "InDesign", "Premiere Pro",
    "UI/UX Design", "Figma", "Sketch", "Adobe XD",
    "Cybersécurité", "Cryptographie", "Ethical Hacking", "Forensics",
    "IoT", "Blockchain", "Réalité virtuelle", "Réalité augmentée",
    "Lean Six Sigma", "Gestion de la qualité", "ISO 9001",
    "ITIL", "PRINCE2", "PMP", "Certification Scrum"
]

CERTIFICATIONS = [
    "AWS Certified Solutions Architect", "Certified Information Systems Security Professional (CISSP)",
    "Project Management Professional (PMP)", "Certified ScrumMaster (CSM)",
    "Certified Information Security Manager (CISM)", "Certified Ethical Hacker (CEH)",
    "Cisco Certified Network Associate (CCNA)", "CompTIA A+", "CompTIA Network+", "CompTIA Security+",
    "ITIL Foundation", "Microsoft Certified: Azure Solutions Architect Expert",
    "Certified Information Systems Auditor (CISA)", "Certified in Risk and Information Systems Control (CRISC)",
    "Google Certified Professional Cloud Architect", "Certified Information Security Auditor (CISA)",
    "Offensive Security Certified Professional (OSCP)", "Certified Information Privacy Professional (CIPP)",
    "Certified Cloud Security Professional (CCSP)", "Salesforce Certified Administrator",
    "Oracle Certified Professional (OCP)", "Red Hat Certified Engineer (RHCE)",
    "VMware Certified Professional (VCP)", "Certified Kubernetes Administrator (CKA)",
    "Certified Scrum Product Owner (CSPO)", "PMI Agile Certified Practitioner (PMI-ACP)",
    "Certified Ethical Hacker (CEH)", "GIAC Security Essentials (GSEC)",
    "Certified Data Professional (CDP)", "Tableau Desktop Certified Professional"
]


def extract_skills_with_experience(text):
    text = text.lower()
    extracted_skills = {}
    year_pattern = r'(\d+(?:[,.]\d+)?)\s*(?:an(?:s|nées?)?|a(?:ns?)?)'

    for skill in SKILLS:
        skill_lower = skill.lower()
        if skill_lower in text:
            skill_index = text.index(skill_lower)
            search_start = max(0, skill_index - 100)
            search_end = min(len(text), skill_index + 100)
            search_text = text[search_start:search_end]

            experience_match = re.search(year_pattern, search_text)
            years = float(experience_match.group(1).replace(',', '.')) if experience_match else 0

            extracted_skills[skill] = years

    return extracted_skills


def extract_certifications(text):
    text = text.lower()
    return [cert for cert in CERTIFICATIONS if cert.lower() in text]


def extract_projects(text):
    project_pattern = r'(?:projet|project)s?\s*:\s*((?:(?!projet|project).)+)'
    projects = re.findall(project_pattern, text, re.IGNORECASE | re.DOTALL)
    return [project.strip() for project in projects]


def extract_skills_certifications_projects(text):
    skills = extract_skills_with_experience(text)
    certifications = extract_certifications(text)
    projects = extract_projects(text)
    return skills, certifications, projects


