import base64
import io
import logging
import os
import re
import textwrap

from flask import Flask, request, jsonify

import openai
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adapte si besoin
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from PIL import ImageEnhance, ImageOps

# ------------------------------------------
# CONFIG / MAPPING / PROMPTS
# ------------------------------------------

POINTS_CONTROLE = [
    "Intitulé du produit",
    "Coordonnées du fournisseur",
    "Estampille",
    "Présence d’une certification (VRF, VVF, BIO, VPF)",
    "Mode de réception (Frais, Congelé)",
    "Conditionnement / Emballage",
    "Température",
    "Conservation",
    "Présence d’une DLC / DLUO",
    "Espèce",
    "Origine",
    "Contaminants (1881/2006 , 2022/2388)",
    "Corps Etranger",
    "VSM",
    "Aiguilles",
    "Date du document",
    "Composition du produit",
    "Process",
    "Critères Microbiologiques",
    "Critères physico-chimiques"
]

MAPPING_SYNONYMES = """
Certaines informations de la fiche technique peuvent apparaître sous des intitulés différents. Voici des équivalences :
- "Intitulé du produit" : "Dénomination légale", "Nom du produit", "Produit"
- "Estampille" : "Estampille sanitaire", "N° d’agrément", "Sanitary mark"
- "Coordonnées du fournisseur" : "Adresse fournisseur", "Nom et adresse du fabricant"
- "Origine" : "Origine", "Pays d’origine", "Origine viande"
- "DLC / DLUO" : "Durée de vie", "Date limite de consommation", "Use by", "Durée étiquetée", "Durée", "Validité", "à consommer avant.."
- "Conditionnement / Emballage" : "Packaging", "Conditionnement", "Type d’emballage"
- "Température" : "Température de conservation", "Storage temperature"
- "Composition du produit" : "Ingrédients", "Ingredients"
- "Espèce" : "Viande de Porc", "Porc"
Prends-les en compte lors de l’analyse.
"""

PROMPT_POINT = f"""
{MAPPING_SYNONYMES}
Voici le texte extrait d'une fiche technique en agroalimentaire :

{{text}}

Analyse le point de contrôle suivant : "{{point}}".
Réponds STRICTEMENT dans ce format :
---
{{point}}
Statut : Présent / Partiel / Douteux / Non trouvé
Preuve : (citation du texte ou “non trouvé”)
Criticité : Critique / Majeur / Mineur + explication
Recommandation : (valider, demander complément, bloquant…)
---
(Si tu ne trouves rien, note "Non trouvé" partout sauf Criticité/Recommandation)
N’invente rien, sois exhaustif.
"""

RESUME_PROMPT = f"""
{MAPPING_SYNONYMES}
Voici un rapport qualité complet, avec un bloc d’analyse par point :

{{rapport}}

Sur ce rapport uniquement, produis à la fin UN SEUL résumé global exactement dans ce format :

Résumé :
- Points critiques (nombre) : [liste des points concernés]
---
- Points majeurs (nombre) : [liste des points concernés]
---
- Points mineurs (nombre) : [liste des points concernés]
---
- Décision recommandée : (valider / demander complément / refuser)
---
- Incohérences détectées : [liste]

Répète strictement ce format. Si rien n’est à signaler dans une catégorie, liste vide.
"""

# ------------------------------------------
# FLASK APP
# ------------------------------------------

app = Flask(__name__)
openai.api_key = os.environ.get("OPENAI_API_KEY")

# ------------------------------------------
# OCR ET EXTRACTION TEXTE
# ------------------------------------------

def extract_text_from_pdf_pypdf2(pdf_data: bytes) -> str:
    text_content = []
    try:
        reader = PdfReader(io.BytesIO(pdf_data))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
    except Exception as e:
        logging.error(f"Erreur d'extraction PDF (PyPDF2) : {e}")
    return "\n".join(text_content)

def clean_ocr_text(ocr_text: str) -> str:
    ocr_text = re.sub(r' +', ' ', ocr_text)
    ocr_text = re.sub(r'\n+', '\n', ocr_text)
    ocr_text = re.sub(r'(\w)-\n(\w)', r'\1\2', ocr_text)  # Fusionne coupures de mots
    return ocr_text.strip()

def extract_text_ocr(pdf_data: bytes) -> str:
    from PIL import Image
    text_parts = []
    try:
        images = convert_from_bytes(pdf_data, dpi=300, poppler_path=r"C:\poppler-24.05.0\Library\bin")
        for idx, img in enumerate(images):
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            img = img.convert('L')
            img = ImageOps.autocontrast(img)
            img = img.point(lambda x: 0 if x < 160 else 255, '1')
            ocr_text = pytesseract.image_to_string(img, lang="fra")
            print(f"\n>>> OCR PAGE {idx+1} <<<\n{ocr_text}\n---")  # Debug visuel
            text_parts.append(ocr_text)
    except Exception as e:
        logging.error(f"Erreur d'extraction OCR : {e}")
    return clean_ocr_text("\n".join(text_parts))

def extract_text_with_fallback(pdf_data: bytes) -> str:
    text = extract_text_from_pdf_pypdf2(pdf_data)
    if text.strip() and len(text) > 200:
        print("\n>>>> TEXTE NATIF DETECTE <<<<\n", text[:600])
        return text + "\n\n[INFO] Texte natif PDF utilisé."
    text = extract_text_ocr(pdf_data)
    if text.strip():
        print("\n>>>> OCR DETECTE <<<<\n", text[:600])
        return text + "\n\n[INFO] OCR utilisé."
    # En vrai, fallback vision sur chaque page possible ici si tu veux, mais rarement utile
    return "[ERREUR] Aucun texte détecté"

# ------------------------------------------
# GPT INTERACTIONS (ANALYSE POINT PAR POINT)
# ------------------------------------------

def analyze_point(pdf_text, point):
    prompt = PROMPT_POINT.format(text=pdf_text, point=point)
    try:
        messages = [
            {"role": "system", "content": "Tu es un assistant expert qualité en agroalimentaire."},
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 4o si ça ne time plus, sinon 3.5 pour la fiabilité
            messages=messages,
            temperature=0.0,
            max_tokens=600,
            request_timeout=40
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erreur analyse point {point}: {e}")
        return f"---\n{point}\nStatut : Erreur\nPreuve : Erreur\nCriticité : Erreur\nRecommandation : Erreur\n---"

def generate_resume(report_text):
    prompt = RESUME_PROMPT.format(rapport=report_text)
    try:
        messages = [
            {"role": "system", "content": "Tu es un expert synthèse qualité."},
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0,
            max_tokens=600,
            request_timeout=40
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erreur synthèse finale : {e}")
        return "Résumé indisponible."

def format_report_text(report_text):
    # Mettre en forme pour PDF (ex : titre en gras)
    report_text = re.sub(r'(Recommandation\s?:[^\n]*)', r'\1\n', report_text)
    report_text = re.sub(r'---\n(\w)', r'---\n\n\1', report_text)
    report_text = report_text.replace('---', '---\n')
    report_text = report_text.replace('Résumé :', '\n\nRésumé :\n')
    return report_text

def generate_full_report(pdf_text):
    blocs = []
    for idx, point in enumerate(POINTS_CONTROLE):
        bloc = analyze_point(pdf_text, point)
        blocs.append(bloc)
    rapport = "\n\n".join(blocs)
    resume = generate_resume(rapport)
    rapport_complet = rapport + "\n\n" + resume
    return rapport_complet

# ------------------------------------------
# PDF EN SORTIE (TITRES EN GRAS)
# ------------------------------------------

def generate_pdf_in_memory(report_text: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x_margin, y_margin = 50, 50
    line_height = 16
    max_chars_per_line = 100

    textobject = c.beginText(x_margin, height - y_margin)
    textobject.setFont("Helvetica", 11)
    y = height - y_margin

    for line in report_text.split('\n'):
        if line.strip() == '':
            y -= line_height // 2
            continue
        # Titres en gras (1. Intitulé du produit)
        if re.match(r'^\d+\. ', line.strip()) or line.strip().endswith(":") or line.strip().startswith("Résumé"):
            textobject.setFont("Helvetica-Bold", 11)
            textobject.textLine(line)
            textobject.setFont("Helvetica", 11)
        else:
            wrapped_lines = textwrap.wrap(line, width=max_chars_per_line, break_long_words=False, break_on_hyphens=False)
            for wrapped_line in wrapped_lines:
                if y < y_margin + line_height:
                    c.drawText(textobject)
                    c.showPage()
                    textobject = c.beginText(x_margin, height - y_margin)
                    textobject.setFont("Helvetica", 11)
                    y = height - y_margin
                textobject.textLine(wrapped_line)
                y -= line_height

    c.drawText(textobject)
    c.showPage()
    c.save()
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

# ------------------------------------------
# FLASK ROUTE
# ------------------------------------------

@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    try:
        logging.info("Requête reçue dans /analyze_pdf")
        data = request.get_json()
        if not data or "pdf_base64" not in data:
            logging.error("JSON invalide ou champ 'pdf_base64' manquant")
            return jsonify({"error": "Invalid JSON body"}), 400

        pdf_base64 = data["pdf_base64"]
        pdf_bytes = base64.b64decode(pdf_base64)
        logging.info(f"PDF décodé avec succès (taille : {len(pdf_bytes)} octets)")

        pdf_text = extract_text_with_fallback(pdf_bytes)
        logging.info(f"Texte extrait (ou fallback OCR) : {pdf_text[:300]}...")

        report_text = generate_full_report(pdf_text)
        report_text = format_report_text(report_text)
        report_pdf_bytes = generate_pdf_in_memory(report_text)
        logging.info(f"Rapport PDF généré (taille : {len(report_pdf_bytes)} octets)")

        report_pdf_base64 = base64.b64encode(report_pdf_bytes).decode('utf-8')
        logging.info("Réponse encodée et prête à être renvoyée")

        return jsonify({
            "report_pdf_base64": report_pdf_base64
        }), 200

    except Exception as e:
        logging.exception("Erreur inattendue dans /analyze_pdf")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
