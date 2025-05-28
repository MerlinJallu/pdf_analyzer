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
        # Loguer la réception de la requête
        logging.info("Requête reçue dans /analyze_pdf")
        
        # Vérifier et loguer le contenu du JSON reçu
        data = request.get_json()
        logging.info(f"Contenu reçu : {data}")

        if not data or "pdf_base64" not in data:
            logging.error("JSON invalide ou champ 'pdf_base64' manquant")
            return jsonify({"error": "Invalid JSON body"}), 400

        pdf_base64 = data["pdf_base64"]

        # Étape 1 : Décoder le PDF et loguer
        pdf_bytes = base64.b64decode(pdf_base64)
        logging.info(f"PDF décodé avec succès (taille : {len(pdf_bytes)} octets)")

        # Étape 2 : Extraire le texte et loguer
        pdf_text = extract_text_with_fallback(pdf_bytes)
        logging.info(f"Texte extrait : {pdf_text[:500]}...")  # Limiter à 500 caractères pour éviter de tout afficher

        # Étape 3 : Analyse ChatGPT et loguer
        report_text = analyze_text_with_chatgpt(pdf_text, INSTRUCTIONS)
        logging.info(f"Rapport généré : {report_text[:500]}...")  # Limiter l'affichage

        if not report_text:
            logging.error("L'analyse ChatGPT a échoué")
            return jsonify({"error": "ChatGPT analysis failed"}), 500

        # Étape 4 : Générer un PDF rapport et loguer
        report_pdf_bytes = generate_pdf_in_memory(report_text)
        logging.info(f"Rapport PDF généré (taille : {len(report_pdf_bytes)} octets)")

        # Étape 5 : Encoder et retourner le PDF
        report_pdf_base64 = base64.b64encode(report_pdf_bytes).decode('utf-8')
        logging.info("Réponse encodée et prête à être renvoyée")

        return jsonify({
            "report_pdf_base64": report_pdf_base64
        }), 200

    except Exception as e:
        logging.exception("Erreur inattendue dans /analyze_pdf")
        return jsonify({"error": str(e)}), 500

def validate_pdf(pdf_data: bytes) -> bool:
    """
    Valide si un fichier est un PDF correct en tentant de le lire avec PyPDF2.
    """
    try:
        PdfReader(io.BytesIO(pdf_data))
        return True
    except Exception as e:
        logging.error(f"Validation échouée pour le fichier PDF : {e}")
        return False

def extract_text_with_fallback(pdf_data: bytes) -> str:
    """
    Tente d'extraire le texte via PyPDF2.
    Si c'est vide, fait un OCR avec pdf2image + pytesseract.
    """
    extracted_text = extract_text_from_pdf_pypdf2(pdf_data)

    if not extracted_text.strip():
        logging.info("PyPDF2 returned no text; switching to OCR.")
        extracted_text = extract_text_ocr(pdf_data)
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
