import base64
import io
import logging
import os

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
Tu auras pour objectif d'analyser des fichiers PDF avec un certain nombre de points de contrôle 
pour écrire par la suite un rapport en notant sous format PDF, tous les points apparents et 
ceux n'apparaissant pas. Les points de contrôle sont :

1. Intitulé du produit
2. Coordonnées du fournisseur
3. Estampille (Noté l'estampille)
4. Présence d'une certification (VRF, VVF, BIO, VPF)
5. Mode de réception (Frais, Congelé)
6. Conditionnement / Emballage (trace de plastique, bleue de préférence)
7. Température (généralement en °C)
8. Conservation
9. Présence d'une DLC / DLUO
10. Espèce
11. Origine
12. Contaminants (1881/2006 , 2022/2388)
13. Corps Etranger (absence le plus possible)
14. VSM (absence obligatoire)
15. Aiguilles (absence obligatoire)
16. Date du document (doit être de moins de 3 ans)
17. Composition du produit (liste et pourcentage requis)
18. Process (détaillé et structuré sous forme de liste obligatoire)
19. Critères Microbiologiques (FCD, sous forme de liste obligatoire)
20. Critères physico-chimiques (Valeur nutritionnelle, sous forme de liste obligatoire)

Le document reprendra chaque point de contrôle en affichant s’il est présent ou non, 
sous forme de rapport. Chaque fichier analysé doit mener à un rapport unique, 
avec la même structure et la même logique pour tous, sans détails, juste les points annoncés.

Pour apporter des détails, l'intitulé du produit est généralement en Haut au milieu du document, assez souvent en gras, 
les coordonnées sont généralement sous forme d'adresse.
"""

@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    """
    Reçoit un JSON du type { "pdf_base64": "<base64>" },
    renvoie { "report_pdf_base64": "<base64>" }.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        pdf_base64 = data.get("pdf_base64")
        if not pdf_base64:
            return jsonify({"error": "Missing 'pdf_base64'"}), 400

        # Décoder le PDF
        pdf_bytes = base64.b64decode(pdf_base64)

        # 1) Extraire le texte (PyPDF2 + fallback OCR)
        pdf_text = extract_text_with_fallback(pdf_bytes)

        # 2) Analyse ChatGPT
        report_text = analyze_text_with_chatgpt(pdf_text, INSTRUCTIONS)
        if not report_text:
            return jsonify({"error": "ChatGPT analysis failed"}), 500

        # 3) Générer PDF rapport
        report_pdf_bytes = generate_pdf_in_memory(report_text)

        # 4) Encoder le rapport
        report_pdf_base64 = base64.b64encode(report_pdf_bytes).decode('utf-8')

        return jsonify({
            "report_pdf_base64": report_pdf_base64
        }), 200

    except Exception as e:
        logging.exception("Erreur inattendue")
        return jsonify({"error": str(e)}), 500

def extract_text_with_fallback(pdf_data: bytes) -> str:
    """
    Tente d'extraire le texte via PyPDF2.
    Si c'est vide, fait un OCR avec pdf2image + pytesseract.
    """
    extracted_text = extract_text_from_pdf_pypdf2(pdf_data)

    if not extracted_text.strip():
        logging.info("PyPDF2 returned no text; switching to OCR.")
        extracted_text = extract_text_ocr(pdf_data)
        # S'il est toujours vide, c'est probablement un PDF vraiment illisible
        if not extracted_text.strip():
            logging.warning("OCR also returned no text. Document may be unextractable.")

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
    """
    Convertit chaque page du PDF en image (via pdf2image), puis applique pytesseract.
    """
    text_parts = []
    try:
        # Convertir PDF en liste d'images
        images = convert_from_bytes(pdf_data, dpi=300) 
        # Pour chaque image, faire l'OCR
        for img in images:
            ocr_text = pytesseract.image_to_string(img, lang="fra")  # ou lang="eng" si texte anglais
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
            model="gpt-4",
            messages=messages,
            temperature=0.0,
            max_tokens=3500
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
    textobject = c.beginText(x_margin, height - y_margin)
    textobject.setFont("Helvetica", 11)

    lines = report_text.split('\n')
    for line in lines:
        if len(line) > 110:
            segments = [line[i:i+110] for i in range(0, len(line), 110)]
            for seg in segments:
                textobject.textLine(seg)
        else:
            textobject.textLine(line)

    c.drawText(textobject)
    c.showPage()
    c.save()
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
