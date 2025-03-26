from flask import Flask, request, render_template, send_from_directory
import os
import docx2txt
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
import json

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PREVIOUS_RESUMES'] = 'previous_resumes.json'  # File to store previous resumes

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Ensure previous_resumes.json file exists and is valid
if not os.path.exists(app.config['PREVIOUS_RESUMES']):
    with open(app.config['PREVIOUS_RESUMES'], 'w') as file:
        json.dump([], file)  # Initialize with an empty list

# Load BERT model for embeddings
bert_model = SentenceTransformer('all-MiniLM-L6-v2')


# Load previous resumes from file
def load_previous_resumes():
    try:
        with open(app.config['PREVIOUS_RESUMES'], 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


# Save previous resumes to file
def save_previous_resumes(resumes):
    with open(app.config['PREVIOUS_RESUMES'], 'w') as file:
        json.dump(resumes, file)


# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()


# Function to extract text from a DOCX file
def extract_text_from_docx(file_path):
    return docx2txt.process(file_path).strip()


# Function to extract text from a TXT file
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


# Function to determine file type and extract text
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    return ""


# Function to check if the text is a resume
def is_resume(text):
    resume_keywords = [
        "experience", "education", "skills", "summary", "objective",
        "work history", "projects", "certifications", "achievements",
        "contact information", "phone", "email", "linkedin"
    ]
    keyword_count = sum(1 for keyword in resume_keywords if keyword.lower() in text.lower())
    return keyword_count >= 3


# Enhanced extract_meaningful_keywords function
def extract_meaningful_keywords(text):
    # Tech acronyms and their expansions
    tech_acronyms = {
        'ztca': 'zscaler trust certification',
        'mlh': 'major league hacking',
        'aiml': 'ai ml',
        'aws': 'amazon web services',
        'gcp': 'google cloud platform',
        'cicd': 'ci cd',
        'rest': 'rest api',
        'graphql': 'graph ql',
        'nosql': 'no sql'
    }

    # Common technical terms we want to keep even if they're short
    important_short_words = {
        'ai', 'ml', 'db', 'ui', 'ux', 'api', 'ci', 'cd', 'dev', 'ops',
        'qa', 'sql', 'js', 'ts', 'os', 'aws', 'gcp', 'az', 'cli', 'sdk',
        'ide', 'oop', 'tcp', 'udp', 'ip', 'dns', 'ssl', 'ssh', 'json',
        'xml', 'csv', 'iot', 'vr', 'ar', 'http', 'https', 'ftp', 'smtp'
    }

    # Replace known acronyms first
    for acronym, expansion in tech_acronyms.items():
        text = re.sub(rf'\b{acronym}\b', expansion, text, flags=re.IGNORECASE)

    # Keep only alphanumeric and specific tech-relevant characters
    text = re.sub(r'[^a-zA-Z0-9#+./&\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).lower()

    # Split camelCase and snake_case
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'_', ' ', text)

    words = re.findall(r'\b[\w+#&.]+\b', text)

    # Extended stop words list for tech context
    extended_stop_words = stop_words.union({
        'using', 'used', 'use', 'via', 'based', 'etc', 'including',
        'like', 'example', 'also', 'within', 'without', 'among',
        'want', 'need', 'make', 'made', 'many', 'much', 'may',
        'know', 'work', 'working', 'worked', 'new', 'old', 'good',
        'better', 'best', 'high', 'higher', 'low', 'lower', 'able',
        'various', 'different', 'several', 'multiple', 'across'
    })

    # Categories of words to exclude
    exclude_categories = {
        # Months
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        # Days
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        # Common verbs
        'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
        # Pronouns
        'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'she',
        'her', 'hers', 'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs',
        # Other common non-technical words
        'hello', 'hi', 'dear', 'regards', 'sincerely', 'thank', 'thanks', 'looking',
        'forward', 'contact', 'phone', 'email', 'address', 'city', 'state', 'country',
        'zip', 'code', 'number', 'mobile', 'cell', 'linkedin', 'github', 'portfolio'
    }

    meaningful_keywords = set()
    for word in words:
        # Keep important short technical terms
        if word in important_short_words:
            meaningful_keywords.add(word)
            continue

        # Skip if in extended stop words or exclusion categories
        if (word in extended_stop_words or
                word in exclude_categories or
                len(word) < 3 and word not in important_short_words):
            continue

        # Skip years (19xx, 20xx)
        if re.match(r'^(19|20)\d{2}$', word):
            continue

        # Skip common non-technical patterns
        if (re.match(r'^[a-z]+\d+$', word) and word not in important_short_words):  # e.g. "item1", "project2"
            continue

        meaningful_keywords.add(word)

    # Manual filtering of some common non-keywords
    non_keywords = {
        'com', 'http', 'https', 'www', 'html', 'href', 'mailto', 'src',
        'img', 'alt', 'title', 'div', 'span', 'class', 'id', 'src', 'rel',
        'col', 'row', 'px', 'rem', 'em', 'pt', 'width', 'height', 'style',
        'true', 'false', 'null', 'undefined', 'type', 'name', 'value'
    }
    meaningful_keywords = meaningful_keywords - non_keywords

    return meaningful_keywords


# Enhanced technology term normalization
def normalize_tech_terms(keywords):
    tech_mappings = {
        # Programming languages
        'js': 'javascript',
        'ts': 'typescript',
        'py': 'python',
        'rb': 'ruby',
        'pl': 'perl',
        'cs': 'csharp',
        'fs': 'fsharp',
        'go': 'golang',
        'kt': 'kotlin',
        'scala': 'scala',
        'swift': 'swift',
        'objc': 'objectivec',

        # Frameworks and libraries
        'node': 'nodejs',
        'express': 'expressjs',
        'react': 'reactjs',
        'vue': 'vuejs',
        'angular': 'angularjs',
        'django': 'djangorest',
        'flask': 'flask',
        'spring': 'springboot',

        # Databases
        'mongo': 'mongodb',
        'postgres': 'postgresql',
        'msql': 'mysql',
        'mssql': 'sqlserver',
        'sqlite': 'sqlite3',

        # Tools and platforms
        'vscode': 'visualstudiocode',
        'vs': 'visualstudio',
        'gcloud': 'googlecloud',
        'azure': 'microsoftazure',
        'k8s': 'kubernetes',
        'docker': 'docker',
        'jenkins': 'jenkins',
        'git': 'git',
        'github': 'github',
        'gitlab': 'gitlab',
        'jira': 'jira',
        'trello': 'trello',

        # Other tech terms
        'oop': 'objectoriented',
        'fp': 'functionalprogramming',
        'rest': 'restapi',
        'graphql': 'graphql',
        'grpc': 'grpc',
        'soap': 'soap',
        'xml': 'xml',
        'json': 'json',
        'yaml': 'yaml',
        'csv': 'csv',
        'html': 'html5',
        'css': 'css3',
        'sass': 'sass',
        'less': 'less',
        'tailwind': 'tailwindcss',
        'bootstrap': 'bootstrap',
        'materialui': 'materialui',
        'redux': 'redux',
        'mobx': 'mobx',
        'jest': 'jest',
        'mocha': 'mocha',
        'jasmine': 'jasmine',
        'karma': 'karma',
        'webpack': 'webpack',
        'babel': 'babel',
        'gulp': 'gulp',
        'grunt': 'grunt',
        'npm': 'npm',
        'yarn': 'yarn',
        'pip': 'pip',
        'maven': 'maven',
        'gradle': 'gradle',
        'ant': 'ant',
        'make': 'make',
        'cmake': 'cmake',
        'ninja': 'ninja',
        'bazel': 'bazel',
        'buck': 'buck',
        'pants': 'pants',
        'sbt': 'sbt',
        'lein': 'lein',
        'cargo': 'cargo',
        'mix': 'mix',
        'rebar': 'rebar',
        'stack': 'stack',
        'cabal': 'cabal',
        'nuget': 'nuget',
        'paket': 'paket',
        'composer': 'composer',
        'pear': 'pear',
        'pipenv': 'pipenv',
        'poetry': 'poetry',
        'conda': 'conda',
        'virtualenv': 'virtualenv',
        'venv': 'venv',
        'pyenv': 'pyenv',
        'rbenv': 'rbenv',
        'nvm': 'nvm',
        'nvs': 'nvs',
        'fnm': 'fnm',
        'volta': 'volta',
        'asdf': 'asdf',
        'jenv': 'jenv',
        'sdkman': 'sdkman',
        'chocolatey': 'chocolatey',
        'homebrew': 'homebrew',
        'apt': 'apt',
        'yum': 'yum',
        'dnf': 'dnf',
        'pacman': 'pacman',
        'zypper': 'zypper',
        'portage': 'portage',
        'nix': 'nix',
        'guix': 'guix',
        'snap': 'snap',
        'flatpak': 'flatpak',
        'appimage': 'appimage',
        'docker': 'docker',
        'podman': 'podman',
        'lxc': 'lxc',
        'lxd': 'lxd',
        'kvm': 'kvm',
        'qemu': 'qemu',
        'virtualbox': 'virtualbox',
        'vmware': 'vmware',
        'hyperv': 'hyperv',
        'vagrant': 'vagrant',
        'terraform': 'terraform',
        'packer': 'packer',
        'ansible': 'ansible',
        'puppet': 'puppet',
        'chef': 'chef',
        'salt': 'salt',
        'cfengine': 'cfengine',
        'rundeck': 'rundeck',
        'jenkins': 'jenkins',
        'teamcity': 'teamcity',
        'bamboo': 'bamboo',
        'gitlabci': 'gitlabci',
        'githubactions': 'githubactions',
        'circleci': 'circleci',
        'travisci': 'travisci',
        'drone': 'drone',
        'argo': 'argo',
        'tekton': 'tekton',
        'spinnaker': 'spinnaker',
        'flux': 'flux',
        'helm': 'helm',
        'kustomize': 'kustomize',
        'skaffold': 'skaffold',
        'tilt': 'tilt',
        'octant': 'octant',
        'lens': 'lens',
        'k9s': 'k9s',
        'kubectl': 'kubectl',
        'oc': 'openshift',
        'istio': 'istio',
        'linkerd': 'linkerd',
        'consul': 'consul',
        'envoy': 'envoy',
        'nginx': 'nginx',
        'apache': 'apache',
        'caddy': 'caddy',
        'traefik': 'traefik',
        'haproxy': 'haproxy',
        'varnish': 'varnish',
        'squid': 'squid',
        'bind': 'bind',
        'unbound': 'unbound',
        'powerdns': 'powerdns',
        'knot': 'knot',
        'nsd': 'nsd',
        'pdns': 'powerdns',
        'dnsmasq': 'dnsmasq',
        'postfix': 'postfix',
        'sendmail': 'sendmail',
        'exim': 'exim',
        'qmail': 'qmail',
        'opensmtpd': 'opensmtpd',
        'haraka': 'haraka',
        'wildduck': 'wildduck',
        'mailcow': 'mailcow',
        'iredmail': 'iredmail',
        'zimbra': 'zimbra',
        'roundcube': 'roundcube',
        'squirrelmail': 'squirrelmail',
        'horde': 'horde',
        'rainloop': 'rainloop',
        'afterlogic': 'afterlogic',
        'nextcloud': 'nextcloud',
        'owncloud': 'owncloud',
        'seafile': 'seafile',
        'syncthing': 'syncthing',
        'resilio': 'resilio',
        'dropbox': 'dropbox',
        'gdrive': 'googledrive',
        'onedrive': 'onedrive',
        'box': 'box',
        'pcloud': 'pcloud',
        'mega': 'mega',
        'tresorit': 'tresorit',
        'spideroak': 'spideroak',
        'sync': 'sync',
        'syncplicity': 'syncplicity',
        'egnyte': 'egnyte',
        'citrix': 'citrix',
        'sharefile': 'sharefile',
        'nextcloud': 'nextcloud',
        'owncloud': 'owncloud',
        'seafile': 'seafile',
        'syncthing': 'syncthing',
        'resilio': 'resilio',
        'dropbox': 'dropbox',
        'gdrive': 'googledrive',
        'onedrive': 'onedrive',
        'box': 'box',
        'pcloud': 'pcloud',
        'mega': 'mega',
        'tresorit': 'tresorit',
        'spideroak': 'spideroak',
        'sync': 'sync',
        'syncplicity': 'syncplicity',
        'egnyte': 'egnyte',
        'citrix': 'citrix',
        'sharefile': 'sharefile'
    }

    normalized = set()
    for keyword in keywords:
        normalized.add(tech_mappings.get(keyword.lower(), keyword.lower()))
    return normalized


# Function to display extracted keywords
def display_extracted_keywords(text, source):
    keywords = extract_meaningful_keywords(text)
    keywords = normalize_tech_terms(keywords)
    print(f"Extracted keywords from {source}: {keywords}")
    return keywords


# Check if a section exists using regex
def has_section(text, section):
    pattern = rf'(?i)\b{section}\b.*(\n|\r)'
    return bool(re.search(pattern, text))


# Generate resume improvement suggestions
def generate_improvements(resume_text, job_description):
    suggestions = []
    sections = ['experience', 'education', 'skills', 'projects']
    for section in sections:
        if not has_section(resume_text, section):
            suggestions.append(f"Consider adding a dedicated '{section.capitalize()}' section.")

    job_keywords = display_extracted_keywords(job_description, "Job Description")
    resume_keywords = display_extracted_keywords(resume_text, "Resume")
    missing_keywords = job_keywords - resume_keywords

    if missing_keywords:
        suggestions.append(f"Consider including these relevant keywords: {', '.join(list(missing_keywords)[:5])}.")

    # Tech-specific suggestions
    tech_keywords = {'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'ci/cd', 'microservices'}
    missing_tech = tech_keywords & job_keywords - resume_keywords
    if missing_tech:
        suggestions.append(f"Add these trending tech skills: {', '.join(missing_tech)}.")

    leadership_phrases = ["led a team", "managed a team", "supervised", "team lead", "project lead"]
    if not any(phrase in resume_text.lower() for phrase in leadership_phrases):
        suggestions.append("Highlight any leadership experience you may have.")

    if len(resume_text.split()) < 300:
        suggestions.append("Your resume seems too short; consider adding more details about your experience.")
    elif len(resume_text.split()) > 800:
        suggestions.append("Your resume seems too long; consider making it more concise.")

    if 'certification' not in resume_text.lower():
        suggestions.append("Consider including any relevant certifications to strengthen your profile.")
    if 'projects' not in resume_text.lower():
        suggestions.append("Mention any key projects you've worked on to showcase your experience.")
    if 'achievements' not in resume_text.lower():
        suggestions.append("Highlight any achievements or awards to stand out from other candidates.")

    return suggestions[:6]  # Return top 6 suggestions


@app.route("/")
def home():
    return render_template('matchresume.html')


@app.route("/matcher", methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')

        if not job_description or not resume_files:
            return render_template('matchresume.html', message="Please upload resumes and enter a job description.")

        # Extract and display keywords from the job description
        job_keywords = display_extracted_keywords(job_description, "Job Description")

        resume_texts = []
        resume_names = []
        resume_suggestions = {}

        for resume_file in resume_files:
            if resume_file.filename == '':
                continue

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(file_path)
            resume_text = extract_text(file_path)

            if resume_text:
                if not is_resume(resume_text):
                    resume_suggestions[resume_file.filename] = ["This file does not appear to be a valid resume."]
                    continue

                # Extract and display keywords from the resume
                resume_keywords = display_extracted_keywords(resume_text, f"Resume: {resume_file.filename}")

                resume_texts.append(resume_text)
                resume_names.append(resume_file.filename)
                resume_suggestions[resume_file.filename] = generate_improvements(resume_text, job_description)

        if not resume_texts:
            return render_template('matchresume.html', message="No valid resumes were processed.")

        job_embedding = bert_model.encode(job_description)
        resume_embeddings = bert_model.encode(resume_texts)

        similarities = cosine_similarity([job_embedding], resume_embeddings)[0]
        similarity_scores = [(resume_names[i], float(similarities[i])) for i in range(len(resume_names))]
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        top_resumes = [x[0] for x in similarity_scores]
        top_scores = [round(x[1] * 100, 2) for x in similarity_scores]
        top_suggestions = [resume_suggestions[x[0]] for x in similarity_scores]

        previous_resumes = load_previous_resumes()
        for i, resume_name in enumerate(top_resumes):
            previous_resumes.append({
                "filename": resume_name,
                "score": top_scores[i],
                "suggestions": top_suggestions[i]
            })
        save_previous_resumes(previous_resumes)

        better_resumes = [
            {
                "filename": resume['filename'],
                "score": resume['score'],
                "suggestions": resume['suggestions'],
                "file_path": os.path.join(app.config['UPLOAD_FOLDER'], resume['filename'])
            }
            for resume in previous_resumes if resume['score'] > top_scores[0]
        ]

        return render_template('matchresume.html', message="Top matching resumes:", top_resumes=top_resumes,
                               similarity_scores=top_scores, resume_suggestions=top_suggestions,
                               better_resumes=better_resumes)

    return render_template('matchresume.html')


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route("/previous_resumes")
def previous_resumes():
    previous_resumes = load_previous_resumes()
    return render_template('previous_resumes.html', previous_resumes=previous_resumes)


if __name__ == '__main__':
    app.run(debug=True)