# analyzer/train_model.py

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump

# --- Synthetic Dataset ---

good_resumes = [
"Software Engineer with 3+ years experience in Python, Django, REST APIs, PostgreSQL. Built scalable backend systems and optimized database queries by 40%. Worked with Docker, AWS EC2, and CI/CD pipelines.",
"Full-stack developer with hands-on experience in React.js, Node.js, MongoDB. Developed 12+ production-ready web applications. Strong skills in Git, testing, and API design.",
"Data Analyst skilled in SQL, Excel, Power BI, Python (Pandas, NumPy). Created dashboards that reduced reporting time by 60%. Experience with ETL pipelines.",
"Machine Learning Engineer experienced in scikit-learn, TensorFlow, NLP, and MLOps. Built classification models with 92% accuracy. Designed end-to-end ML pipelines.",
"Cybersecurity Analyst with knowledge of vulnerability scanning, SIEM tools, network monitoring, and threat detection. Completed CEH certification.",
"Backend Developer strong in Java, Spring Boot, Hibernate, MySQL, Redis. Designed microservices architecture for high-traffic systems.",
"Cloud Engineer with expertise in AWS: EC2, S3, Lambda, CloudWatch, IAM. Automated deployments using Terraform.",
"DevOps Engineer skilled in Kubernetes, Jenkins, Docker, Ansible. Improved build times by 35% through pipeline optimization.",
"AI Engineer experienced in NLP, text classification, vectorization, and model deployment. Built resume analyzer and sentiment classifier.",
"Business Analyst with strong knowledge of SDLC, documentation, wireframing, and stakeholder communication. Proficient in JIRA.",
"Android Developer with apps reaching 10K+ downloads. Skilled in Kotlin, Jetpack Compose, Firebase, Room DB.",
"iOS Developer experienced in Swift, Xcode, MVVM architecture. Built secure payment flow integrations.",
"Digital Marketer skilled in Google Ads, SEO, analytics, content strategy. Improved organic traffic by 120%.",
"Project Manager with 4+ years experience in Agile, Scrum, sprint planning, and team leadership.",
"UI/UX Designer skilled in Figma, prototypes, wireframes, user testing, and responsive design.",
"ML Engineer with focus on computer vision — OpenCV, YOLO, CNNs. Built object detection systems.",
"Software Test Engineer skilled in automation using Selenium & PyTest. Improved test coverage by 60%.",
"Blockchain Developer skilled in Solidity, smart contracts, web3.js, and Truffle. Deployed NFT marketplace on testnet.",
"Data Engineer with experience in Airflow, Snowflake, Spark, distributed data pipelines.",
"Cloud Architect with 6+ years on Azure, designing scalable architectures and security policies.",
"Python Developer specializing in automation, APIs, FastAPI, data scripts.",
"React Developer with 20+ component libraries built and optimized front-end performance.",
"Network Engineer skilled in routing, switching, firewalls, and Cisco tools.",
"AI Research Intern with projects on embeddings, transformers, and vector stores.",
"Marketing Analyst skilled in market research, segmentation, campaign analysis.",
"Quality Analyst with good experience in manual testing, defect tracking, and reporting.",
"NLP Engineer building chatbot systems, intent classification, entity extraction.",
"PHP Developer skilled in Laravel, MySQL, REST APIs, and cloud integration.",
"Systems Engineer with strong Linux, shell scripting, and automation knowledge.",
"Data Scientist with deep expertise in statistical modeling and feature engineering.",
"HR Specialist experienced in talent acquisition, onboarding, and HR analytics.",
"Mechanical Engineer with CAD design, simulation, and manufacturing experience.",
"Electrical Engineer with PLC automation and IoT experience.",
"AI Intern who built LLM applications, vector search, and embeddings.",
"Web Developer strong in HTML, CSS, JS, Bootstrap, and responsive design.",
"Python + Django Developer who built hospital resource management platform.",
"SQL Developer strong in queries, optimization, stored procedures.",
"Cloud DevOps Intern with Docker, CI/CD, cloud security basics.",
"Mobile App Developer with cross-platform Flutter experience.",
"Financial Analyst skilled in forecasting, modeling, Excel automation.",
"Game Developer with Unity, C#, gameplay scripting experience.",
"IT Support Engineer experienced in troubleshooting and ticketing tools.",
"Data Entry Specialist with high accuracy and automation tools.",
"Automation Engineer with Python RPA and workflow automation.",
"Technical Writer with documentation, API guidelines, tutorials.",
"Product Manager experienced in roadmapping and feature planning.",
"Operations Analyst with process optimization experience.",
"Resume Writer skilled in formatting and improving job applications.",
"AI Resume Analyzer Developer with ML and feature extraction knowledge."
]


bad_resumes = [
"Looking for any job. I know computer and mobile. Can do anything.",
"I have done some projects but don’t remember details. Need a job urgently.",
"Fresher. No experience. Basic knowledge of MS Word. Hardworking.",
"I can do coding if someone teaches me. I have interest in software.",
"Worked in shop for some time. Good communication.",
"I know Python little bit. Not fully confident. Want opportunity.",
"Made a website once but not sure where it is now.",
"I can do data entry fast. No certifications.",
"Just completed degree. No internships or projects.",
"I want to get job in IT. I am passionate and good learner.",
"Not much experience, but I will try my best.",
"I have computer knowledge and browsing skills.",
"I can work under pressure. No technical skills.",
"I worked in a small startup, mostly helping with tasks.",
"Know Java basics. Have not built projects yet.",
"Looking for work-from-home job. Good typing speed.",
"I am not great in coding but want to improve.",
"I know HTML a little. Learning CSS.",
"Trying to learn programming. Beginner level.",
"Completed BTech. No projects done.",
"Have some certificates but lost them.",
"Helped friend in his project. No documentation.",
"I know C but forget syntax sometimes.",
"I worked in NGO volunteering. No tech work.",
"Would like to join IT company. No experience.",
"Good at communication. Not good at coding.",
"Made basic calculator in Python.",
"Worked as office assistant for 3 months.",
"I know Excel at basic level.",
"Participated in college fest organization.",
"Open to learn new things.",
"Interested in AI but no project.",
"Made resume analyzer idea but no code.",
"No GitHub profile.",
"Not familiar with databases.",
"Learning JavaScript slowly.",
"Studied machine learning theory only.",
"Did small internship but no real tasks.",
"Can work with team. Limited skills.",
"I know Canva designing.",
"Good leadership skills.",
"Basic knowledge of computer networks.",
"Completed online courses only.",
"No contributions to projects.",
"Beginner level coding.",
"Still exploring career paths.",
"Learning Python from YouTube.",
"Looking for entry-level role."
]


# Combine datasets
corpus = good_resumes + bad_resumes
labels = [1] * len(good_resumes) + [0] * len(bad_resumes)

# --- Train Model ---
vect = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vect.fit_transform(corpus)

model = LogisticRegression(max_iter=300)
model.fit(X, labels)

# --- Save Model ---
model_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(model_dir, exist_ok=True)

dump({"vect": vect, "model": model}, os.path.join(model_dir, "model.joblib"))

print("🎉 Model trained and saved to analyzer/models/model.joblib")
