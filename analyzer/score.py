import re
import os
from joblib import load

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.joblib")

# -----------------------
# Helper Functions
# -----------------------

def find_years_of_experience(text):
    text = text.lower()
    years = re.findall(r"(\d{4})\s*[–-]\s*(\d{4})", text)  
    ranges = [int(b) - int(a) for a,b in years] if years else []
    explicit = re.findall(r"(\d+)\+?\s+years?", text)
    explicit = [int(x) for x in explicit] if explicit else []
    candidates = ranges + explicit
    return max(candidates) if candidates else 0


def count_skills(text, skill_list):
    t = text.lower()
    return sum(1 for s in skill_list if s.lower() in t)


def format_checks(text):
    checks = {}
    checks['has_email'] = bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))
    checks['has_phone'] = bool(re.search(r"\b\d{10}\b", re.sub(r"\D","", text))) or \
                          bool(re.search(r"\+?\d[\d \-]{7,}\d", text))
    checks['has_section'] = any(h in text.lower() for h in ("experience","education","skills","projects","certifications"))
    return checks

# -----------------------
# Main Analyze Function
# -----------------------

def analyze_resume(text):
    data = load(MODEL_PATH)
    vect = data['vect']
    model = data['model']

    X = vect.transform([text])
    pred_prob = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else model.decision_function(X)
    score = int(pred_prob.max() * 100) if hasattr(pred_prob, '__len__') else int(min(max(pred_prob, 0), 1) * 100)

    # Helper scoring
    yrs = find_years_of_experience(text)
    skill_score = count_skills(text, ["python", "django", "sql", "javascript", "html", "css"]) * 10
    formatting = format_checks(text)

    # -----------------------
    # Dynamic Tips Generation
    # -----------------------
    tips = []

    # If model predicts low probability of good resume
    bad_prob = pred_prob[0]  # assuming label 0 = bad
    good_prob = pred_prob[1] if len(pred_prob) > 1 else 1 - bad_prob

    if bad_prob > 0.5:
        # Weak points / improvements
        if "python" not in text.lower() and "django" not in text.lower():
            tips.append("Add relevant technical skills like Python, Django, or JavaScript.")
        if "projects" not in text.lower():
            tips.append("Include projects to demonstrate practical experience.")
        if yrs < 2:
            tips.append("Add more detailed work experience or internships.")
        if len(text.split()) < 300:
            tips.append("Consider adding more content for better readability.")
        if not all(formatting.values()):
            tips.append("Include proper sections (Email, Phone, Skills, Education, Projects).")
    else:
        # Strong points
        if skill_score > 30:
            tips.append("Strong technical skills detected. Highlight them in the resume summary.")
        if yrs >= 3:
            tips.append(f"Good experience of {yrs} years. Mention key achievements for impact.")
        if all(formatting.values()):
            tips.append("Well-formatted resume. Keep consistent structure and sections.")
        if "projects" in text.lower():
            tips.append("Projects section is strong. Highlight measurable results.")

    return {
        "skill_score": min(skill_score, 100),
        "experience_score": min(yrs * 10, 100),
        "education_score": 100 if "bachelor" in text.lower() or "master" in text.lower() else 40,
        "format_score": 100 if all(formatting.values()) else 60,
        "final_score": score,
        "tips": tips
    }
