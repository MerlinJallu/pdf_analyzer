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

INSTRUCTIONS = """
Tu es un assistant expert qualité en agroalimentaire. Pour chaque point de contrôle ci-dessous :

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

Pour chacun, indique :
- Statut : Présent / Partiel / Douteux / Non trouvé
- Preuve (citation du texte ou “non trouvé”)
- Criticité (Critique / Majeur / Mineur + explication)
- Recommandation

Structure exemple :
---
1. Intitulé du produit
Statut : Présent / Non trouvé...
Preuve : ...
Criticité : ...
Recommandation : ...
---
...

A la fin, fais un résumé général :
- Points critiques : [liste]
- Points majeurs : [liste]
- Points mineurs : [liste]
- Décision recommandée : (valider / demander complément / refuser)
- Incohérences détectées : [liste]

N’invente jamais de points non trouvés, ne regroupe jamais les points. Si aucune info : écris “non trouvé”. Ne fais qu’UN SEUL bloc résumé final.
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
        # Titres en gras si besoin
        if re.match(r'^\d+\.', line.strip()):
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
