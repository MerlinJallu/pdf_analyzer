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

from PIL import Image, ImageEnhance, ImageOps

import easyocr

app = Flask(__name__)
openai.api_key = os.environ.get("OPENAI_API_KEY")

INSTRUCTIONS = """
Tu es un assistant expert qualité en agroalimentaire. Pour chaque point de contrôle ci-dessous :

**POUR CHAQUE fiche technique reçue, tu dois IMPÉRATIVEMENT analyser les 20 points de contrôle ci-dessous, dans l’ORDRE, un par un, même si l’information est absente ou douteuse.**

1. Analyse le texte extrait de la fiche technique : dis si le point est Présent, Partiel, Douteux ou Non trouvé.
2. Donne un exemple concret trouvé dans le texte (citation), ou “non trouvé”.
3. Évalue la criticité de l’absence : Critique (bloquant la validation), Majeur (important mais non bloquant), Mineur (utile, mais non bloquant). Explique en une phrase pourquoi.
4. Donne une recommandation ou action : Valider, Demander complément, Bloquant, etc.
5. Si tu repères une incohérence entre deux infos, signale-la.

**Même si la fiche ne donne AUCUNE info sur 15 points, tu dois quand même écrire un bloc “Nom du point…” pour chaque, dans l’ordre. N’arrête jamais l’analyse avant d’avoir commenté tous les points, même si tout est vide.**

**Structure imposée (exemple à suivre pour CHAQUE point, à répéter pour toute la liste) :**

Format pour chaque point :
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

**Répète exactement ce format pour chaque point. Ne regroupe jamais plusieurs points dans un même bloc. Si un point n’a pas d’information, écris “non trouvé”.**

**Tu ne dois jamais condenser, regrouper ou ignorer des points.**
"""

def ocr_preprocess(img):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    img = img.convert('L')
    img = ImageOps.autocontrast(img)
    img = img.point(lambda x: 0 if x < 160 else 255, '1')
    return img

def extract_text_with_fallback(pdf_data: bytes) -> str:
    # 1. PyPDF2
    text = extract_text_from_pdf_pypdf2(pdf_data)
    if text.strip():
        return text + "\n\n[INFO] Texte natif PDF utilisé."
    # 2. OCR boosté (Tesseract + EasyOCR)
    text = extract_text_ocr_boosted(pdf_data)
    if text.strip():
        return text + "\n\n[INFO] OCR utilisé."
    # 3. GPT-4o Vision fallback
    img_pages = convert_from_bytes(pdf_data, dpi=400)
    vision_texts = []
    for i, img in enumerate(img_pages):
        vision_texts.append(analyze_image_with_gpt4o(img, page=i+1))
    return "\n".join(vision_texts) + "\n\n[INFO] GPT-4o Vision utilisé (fallback)."

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

def extract_text_ocr_boosted(pdf_data: bytes) -> str:
    images = convert_from_bytes(pdf_data, dpi=400)
    tesseract_text = []
    easyocr_text = []
    reader = easyocr.Reader(['fr'], gpu=False)
    for img in images:
        img_prep = ocr_preprocess(img)
        # Tesseract
        tesseract_text.append(pytesseract.image_to_string(img_prep, lang="fra"))
        # EasyOCR
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        result_easy = reader.readtext(img_bytes.getvalue(), detail=0)
        easyocr_text.append('\n'.join(result_easy))
    # Combine results
    combined = []
    for t, e in zip(tesseract_text, easyocr_text):
        # Fusionne Tesseract et EasyOCR par page
        combined.append(t + "\n" + e)
    return "\n".join(combined)

def analyze_image_with_gpt4o(img, page=1):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    image_url = f"data:image/png;base64,{img_base64}"
    vision_prompt = f"""Page {page} de la fiche technique. Lis tous les champs visuels (même mal scannés). Suis exactement ces consignes : 

    Tu es un assistant expert qualité en agroalimentaire. Pour chaque point de contrôle ci-dessous :

    **POUR CHAQUE fiche technique reçue, tu dois IMPÉRATIVEMENT analyser les 20 points de contrôle ci-dessous, dans l’ORDRE, un par un, même si l’information est absente ou douteuse.**
    
    1. Analyse le texte extrait de la fiche technique : dis si le point est Présent, Partiel, Douteux ou Non trouvé.
    2. Donne un exemple concret trouvé dans le texte (citation), ou “non trouvé”.
    3. Évalue la criticité de l’absence : Critique (bloquant la validation), Majeur (important mais non bloquant), Mineur (utile, mais non bloquant). Explique en une phrase pourquoi.
    4. Donne une recommandation ou action : Valider, Demander complément, Bloquant, etc.
    5. Si tu repères une incohérence entre deux infos, signale-la.
    
    **Même si la fiche ne donne AUCUNE info sur 15 points, tu dois quand même écrire un bloc “Nom du point…” pour chaque, dans l’ordre. N’arrête jamais l’analyse avant d’avoir commenté tous les points, même si tout est vide.**
    
    **Structure imposée (exemple à suivre pour CHAQUE point, à répéter pour toute la liste) :**
    
    Format pour chaque point :
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
    
    **Répète exactement ce format pour chaque point. Ne regroupe jamais plusieurs points dans un même bloc. Si un point n’a pas d’information, écris “non trouvé”.**
    
    **Tu ne dois jamais condenser, regrouper ou ignorer des points.**
    
    """
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": vision_prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]}
            ],
            max_tokens=2000,
        )
        return f"[ANALYSE IMAGE PAGE {page}]\n" + response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erreur Vision GPT-4o : {e}")
        return f"[ERREUR GPT-4o Vision sur page {page}]"

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
            request_timeout=30
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erreur ChatGPT : {e}")
        return None

# ... (ton format_report_text et generate_pdf_in_memory restent les mêmes) ...

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
        logging.info(f"Texte extrait (ou fallback Vision) : {pdf_text[:500]}...")

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
