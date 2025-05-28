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
from reportlab.lib.utils import simpleSplit

app = Flask(__name__)

openai.api_key = os.environ.get("OPENAI_API_KEY")

INSTRUCTIONS = """
Tu es un expert en contrôle qualité pour l’agroalimentaire.
Ton rôle est d’analyser le texte d’une fiche technique extraite d’un PDF et de générer un rapport synthétique, point par point, en suivant cette trame :

Pour chaque point de contrôle listé, réponds selon ce schéma :
- Statut : Présent / Manquant / Douteux
- Preuve : Extrait du texte si trouvé, ou "Non trouvé"
- Remarque : Une remarque utile et concrète pour l’utilisateur (ex : où chercher, ce qui manque exactement, etc.)

Les points de contrôle sont :
1. Intitulé du produit
2. Coordonnées du fournisseur
3. Estampille (Noter l’estampille)
4. Présence d’une certification (VRF, VVF, BIO, VPF)
5. Mode de réception (Frais, Congelé)
6. Conditionnement / Emballage (trace de plastique, bleue de préférence)
7. Température (généralement en °C)
8. Conservation
9. Présence d’une DLC / DLUO
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

Sois synthétique et clair, formatte chaque point ainsi :
---
**[Nom du point]**
Statut : 
Preuve : 
Remarque : 
---
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

        # Décoder PDF
        pdf_bytes = base64.b64decode(pdf_base64)
        logging.info(f"PDF décodé avec succès (taille : {len(pdf_bytes)} octets)")

        # Extraction texte
        pdf_text = extract_text_with_fallback(pdf_bytes)
        logging.info(f"Texte extrait : {pdf_text[:500]}...")

        # Analyse IA
        report_text = analyze_text_with_chatgpt(pdf_text, INSTRUCTIONS)
        logging.info(f"Rapport généré : {report_text[:500]}...")

        if not report_text:
            logging.error("L'analyse ChatGPT a échoué")
            return jsonify({"error": "ChatGPT analysis failed"}), 500

        # Génération PDF rapport
        report_pdf_bytes = generate_pdf_in_memory(report_text)
        logging.info(f"Rapport PDF généré (taille : {len(report_pdf_bytes)} octets)")

        # Encode et renvoi
        report_pdf_base64 = base64.b64encode(report_pdf_bytes).decode('utf-8')
        logging.info("Réponse encodée et prête à être renvoyée")

        return jsonify({
            "report_pdf_base64": report_pdf_base64
        }), 200

    except Exception as e:
        logging.exception("Erreur inattendue dans /analyze_pdf")
        return jsonify({"error": str(e)}), 500

def validate_pdf(pdf_data: bytes) -> bool:
    try:
        PdfReader(io.BytesIO(pdf_data))
        return True
    except Exception as e:
        logging.error(f"Validation échouée pour le fichier PDF : {e}")
        return False

def extract_text_with_fallback(pdf_data: bytes) -> str:
    """
    Extraction hybride : chaque page est traitée en PyPDF2 puis OCR si besoin.
    """
    text_parts = []
    try:
        reader = PdfReader(io.BytesIO(pdf_data))
        images = convert_from_bytes(pdf_data, dpi=300)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ''
            if not page_text.strip() and images:
                ocr_text = pytesseract.image_to_string(images[i], lang="fra")
                text_parts.append(f"[Page {i+1} - OCR]\n{ocr_text}")
            else:
                text_parts.append(f"[Page {i+1} - PDF]\n{page_text}")
    except Exception as e:
        logging.error(f"Erreur mixte extraction PDF/OCR : {e}")
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

    x_margin, y_margin = 60, 60
    max_width = width - 2 * x_margin
    y = height - y_margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_margin, y, "Rapport d'Analyse Fiche Technique")
    y -= 30
    c.setFont("Helvetica", 11)

    line_height = 14

    for line in report_text.split('\n'):
        if y < y_margin + line_height:
            c.showPage()
            y = height - y_margin
            c.setFont("Helvetica", 11)
        wrapped = simpleSplit(line, "Helvetica", 11, max_width)
        for wline in wrapped:
            c.drawString(x_margin, y, wline)
            y -= line_height
            if y < y_margin + line_height:
                c.showPage()
                y = height - y_margin
                c.setFont("Helvetica", 11)

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(x_margin, 20, "Rapport généré automatiquement - Service Qualité")
    c.save()
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
