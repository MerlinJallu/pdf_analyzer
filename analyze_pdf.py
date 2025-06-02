import base64
import io
import logging
import os
import re
import textwrap

from flask import Flask, request, jsonify

import openai
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

app = Flask(__name__)

openai.api_key = os.environ.get("OPENAI_API_KEY")

INSTRUCTIONS = """
Tu es un assistant expert qualité en agroalimentaire. Pour chaque point de contrôle ci-dessous :

1. Analyse le texte extrait de la fiche technique : dis si le point est Présent, Partiel, Douteux ou Non trouvé.
2. Donne un exemple concret trouvé dans le texte (citation), ou “non trouvé”.
3. Évalue la criticité de l’absence : Critique (bloquant la validation), Majeur (important mais non bloquant), Mineur (utile, mais non bloquant). Explique en une phrase pourquoi.
4. Donne une recommandation ou action : Valider, Demander complément, Bloquant, etc.
5. Si tu repères une incohérence entre deux infos, signale-la.

Format pour chaque point :

---
[Nom du point]
Statut : Présent / Partiel / Douteux / Non trouvé
Preuve : (citation du texte ou “non trouvé”)
Criticité : Critique / Majeur / Mineur + explication
Recommandation : (valider, demander complément, bloquant…)

---

Résumé :
- Points critiques (nombre) : [liste des points concernés]

- Points majeurs (nombre) : [liste des points concernés]

- Points mineurs (nombre) : [liste des points concernés]

- Décision recommandée : (valider / demander complément / refuser)

- Incohérences détectées : [liste]

Voici la liste à analyser :
1. Intitulé du produit
2. Coordonnées du fournisseur
3. Estampille
4. Présence d’une certification (VRF, VVF, BIO, VPF)
5. Mode de réception (Frais, Congelé)
6. Conditionnement / Emballage
7. Température
8. Conservation
9. Présence d’une DLC / DLUO
10. Espèce
11. Origine
12. Contaminants (1881/2006 , 2022/2388)
13. Corps Etranger
14. VSM
15. Aiguilles
16. Date du document
17. Composition du produit
18. Process
19. Critères Microbiologiques
20. Critères physico-chimiques
"""

@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    try:
        logging.info("Requête reçue dans /analyze_pdf")
        data = request.get_json()
        logging.info(f"Contenu reçu : {data}")

        if not data or "pdf_base64" not in data:
            logging.error("JSON invalide ou champ 'pdf_base64' manquant")
            return jsonify({"error": "Invalid JSON body"}), 400

        pdf_base64 = data["pdf_base64"]

        pdf_bytes = base64.b64decode(pdf_base64)
        logging.info(f"PDF décodé avec succès (taille : {len(pdf_bytes)} octets)")

        pdf_text = extract_text_with_fallback(pdf_bytes)
        logging.info(f"Texte extrait : {pdf_text[:500]}...")

        report_text = analyze_text_with_chatgpt(pdf_text, INSTRUCTIONS)
        if not report_text:
            logging.error("L'analyse ChatGPT a échoué")
            return jsonify({"error": "ChatGPT analysis failed"}), 500

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

def extract_text_with_fallback(pdf_data: bytes) -> str:
    extracted_text = extract_text_from_pdf_pypdf2(pdf_data)
    logging.info(f"Texte extrait via PyPDF2 : {len(extracted_text)} caractères")
    if not extracted_text.strip():
        logging.info("PyPDF2 vide, passage à l'OCR.")
        extracted_text = extract_text_ocr(pdf_data)
        logging.info(f"Texte extrait via OCR : {len(extracted_text)} caractères")
        if not extracted_text.strip():
            logging.warning("OCR vide : document peut-être inexploitable.")
    return extracted_text

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

def extract_text_ocr(pdf_data: bytes) -> str:
    text_parts = []
    try:
        images = convert_from_bytes(pdf_data, dpi=300)
        for img in images:
            ocr_text = pytesseract.image_to_string(img, lang="fra")
            text_parts.append(ocr_text)
    except Exception as e:
        logging.error(f"Erreur d'extraction OCR : {e}")
    return "\n".join(text_parts)

def analyze_text_with_chatgpt(pdf_text: str, instructions: str) -> str:
    try:
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": pdf_text}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0,
            max_tokens=3500,
            request_timeout=30  # Sécurité pour éviter le freeze
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erreur ChatGPT : {e}")
        return None

def format_report_text(report_text):
    # Saut de ligne après chaque bloc Recommandation (si ce n'est pas déjà fait)
    report_text = re.sub(r'(Recommandation\s?:[^\n]*)', r'\1\n', report_text)
    # Ajoute un saut de ligne avant chaque nouveau point (titre en gras)
    report_text = re.sub(r'(\*\*[A-Z][^*]+\*\*)', r'\n\1', report_text)
    # Double saut de ligne après chaque '---'
    report_text = report_text.replace('---', '---\n')
    # Double saut après 'Résumé :'
    report_text = report_text.replace('**Résumé :**', '\n\n**Résumé :**\n')
    return report_text

def generate_pdf_in_memory(report_text: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x_margin, y_margin = 50, 50
    line_height = 14
    max_width = width - 2 * x_margin
    max_chars_per_line = 100  # adapte si besoin

    textobject = c.beginText(x_margin, height - y_margin)
    textobject.setFont("Helvetica", 11)

    y = height - y_margin
    for line in report_text.split('\n'):
        if line.strip() == '':
            y -= line_height // 2
            continue
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
