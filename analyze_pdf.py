import base64
import io
import logging
import os
import re
import textwrap

from flask import Flask, request, jsonify

import openai
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pdf2image import convert_from_bytes
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from PIL import ImageEnhance, ImageOps

app = Flask(__name__)
openai.api_key = os.environ.get("OPENAI_API_KEY")

MAPPING_SYNONYMES = """
Certaines informations de la fiche technique peuvent apparaître sous des intitulés différents. Voici des équivalences :Add commentMore actions
- "Intitulé du produit" : "Dénomination légale", "Nom du produit", "Produit"
- "Estampille" : "Estampille sanitaire", "N° d’agrément", "Sanitary mark"
- "Coordonnées du fournisseur" : "Adresse fournisseur", "Nom et adresse du fabricant"
- "Origine" : "Origine", "Pays d’origine", "Origine viande"
- "DLC / DLUO" : "Durée de vie", "Date limite de consommation", "Use by", "Durée étiquetée"
- "Conditionnement / Emballage" : "Packaging", "Conditionnement", "Type d’emballage"
- "Température" : "Température de conservation", "Storage temperature"
- "Composition du produit" : "Ingrédients", "Ingredients"
Prends-les en compte lors de l’analyse.
"""

INSTRUCTIONS = f"""
Tu es un assistant expert qualité en agroalimentaire. Pour chaque point de contrôle ci-dessous :

**POUR CHAQUE fiche technique reçue, tu dois IMPÉRATIVEMENT analyser les 20 points de contrôle ci-dessous, dans l’ORDRE, un par un, même si l’information est absente ou douteuse.**

{MAPPING_SYNONYMES}

1. Analyse le texte extrait de la fiche technique : dis si le point est Présent, Partiel, Douteux ou Non trouvé.
2. Donne un exemple concret trouvé dans le texte (citation), ou “non trouvé”.
3. Évalue la criticité de l’absence : Critique (bloquant la validation), Majeur (important mais non bloquant), Mineur (utile, mais non bloquant). Explique en une phrase pourquoi.
4. Donne une recommandation ou action : Valider, Demander complément, Bloquant, etc.
5. Si tu repères une incohérence entre deux infos, signale-la.

**Même si la fiche ne donne AUCUNE info sur 15 points, tu dois quand même écrire un bloc “Nom du point…” pour chaque, dans l’ordre. N’arrête jamais l’analyse avant d’avoir commenté tous les points, même si tout est vide.**

Format pour chaque point :

---
Nom du point
Statut : Présent / Partiel / Douteux / Non trouvé
Preuve : (citation du texte ou “non trouvé”)
Criticité : Critique / Majeur / Mineur + explication
Recommandation : (valider, demander complément, bloquant…)

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

Voici la liste à analyser :
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

**Répète exactement ce format pour chaque point. Ne regroupe jamais plusieurs points dans un même bloc. Si un point n’a pas d’information, écris “non trouvé”.**

**Tu ne dois jamais condenser, regrouper ou ignorer des points.**
"""

def extract_text_ocr(pdf_data: bytes) -> str:
    """OCR sur chaque page du PDF via pytesseract (prétraitement pour booster la qualité)"""
    from PIL import Image
    text_parts = []
    try:
        images = convert_from_bytes(pdf_data, dpi=250)  # DPI faible = plus rapide sur Heroku
        for idx, img in enumerate(images):
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            img = img.convert('L')
            img = ImageOps.autocontrast(img)
            img = img.point(lambda x: 0 if x < 160 else 255, '1')
            ocr_text = pytesseract.image_to_string(img, lang="fra")
            print(f"\n>>> OCR PAGE {idx+1} <<<\n{ocr_text}\n---")
            text_parts.append(ocr_text)
    except Exception as e:
        logging.error(f"Erreur d'extraction OCR : {e}")
    text = "\n".join(text_parts)
    # Nettoyage minimal
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

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
            request_timeout=40
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erreur ChatGPT : {e}")
        return None
def format_report_text(report_text):
    # Supprime "untitled" en début de texte s'il existe
    report_text = re.sub(r'^\s*untitled\s*\n+', '', report_text, flags=re.IGNORECASE)
    # Enlève les mini-résumés après chaque point, ne garde que le dernier
    points_blocs = re.split(r'(?=\d+\. )', report_text)
    if len(points_blocs) > 1:
        # On isole la partie finale (le vrai résumé global)
        resume_match = re.search(r'Résumé :(?:.|\n)+', report_text)
        resume_global = resume_match.group(0) if resume_match else ''
        # On retire tous les résumés intermédiaires dans chaque bloc
        points_blocs = [re.sub(r'Résumé :(?:.|\n)+?(?=\d+\. |\Z)', '', bloc, flags=re.MULTILINE) for bloc in points_blocs[:-1]]
        # On regroupe tous les points analysés + résumé final
        report_text = '\n'.join([bloc.strip() for bloc in points_blocs if bloc.strip()]) + '\n\n' + resume_global.strip()

    # Séparateurs clairs
    report_text = re.sub(r'(?<=Recommandation : .+)\n+', '\n\n' + '-'*54 + '\n', report_text)
    return report_text
    
def generate_pdf_in_memory(report_text: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x_margin, y_margin = 50, 50
    line_height = 16
    max_chars_per_line = 90

    textobject = c.beginText(x_margin, height - y_margin)
    y = height - y_margin

    lines = report_text.split('\n')
    for i, line in enumerate(lines):
        # Titre des points en gras (ex: 1. Intitulé du produit)
        if re.match(r'^\d+\.\s', line.strip()):
            textobject.setFont("Helvetica-Bold", 12)
            textobject.setFillColor(HexColor("#22325c"))  # Un bleu foncé par exemple
            textobject.textLine(line.strip())
            textobject.setFont("Helvetica", 11)
            textobject.setFillColor(HexColor("#000000"))
            y -= line_height
        # Séparateur visuel
        elif re.match(r'^-+$', line.strip()):
            c.drawText(textobject)
            y -= 4
            c.setStrokeColor(HexColor("#dddddd"))
            c.line(x_margin, y, width - x_margin, y)
            y -= 10
            textobject = c.beginText(x_margin, y)
            textobject.setFont("Helvetica", 11)
        else:
            # Wrap long lines
            wrapped_lines = textwrap.wrap(line, width=max_chars_per_line)
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
    c.save()
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    try:
        data = request.get_json()
        if not data or "pdf_base64" not in data:
            return jsonify({"error": "Invalid JSON body"}), 400
        pdf_base64 = data["pdf_base64"]
        pdf_bytes = base64.b64decode(pdf_base64)
        ocr_text = extract_text_ocr(pdf_bytes)
        print("\n>>> TEXTE OCR POUR GPT <<<\n", ocr_text[:1200], "\n---")  # debug
        if not ocr_text.strip():
            return jsonify({"error": "OCR extraction failed"}), 500
        report_text = analyze_text_with_chatgpt(ocr_text, INSTRUCTIONS)
        if not report_text:
            return jsonify({"error": "ChatGPT analysis failed"}), 500
        report_pdf_bytes = generate_pdf_in_memory(report_text)
        report_pdf_base64 = base64.b64encode(report_pdf_bytes).decode('utf-8')
        return jsonify({
            "report_pdf_base64": report_pdf_base64
        }), 200
    except Exception as e:
        logging.exception("Erreur inattendue dans /analyze_pdf")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
