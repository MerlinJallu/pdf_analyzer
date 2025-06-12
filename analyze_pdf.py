import base64
import io
import logging
import os
import re
import textwrap

from flask import Flask, request, jsonify

import openai
os.environ['TESSDATA_PREFIX'] = '/app/.apt/usr/share/tesseract-ocr/5/tessdata'
import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pdf2image import convert_from_bytes
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from PIL import ImageEnhance, ImageOps

app = Flask(__name__)
openai.api_key = os.environ.get("OPENAI_API_KEY")

MAPPING_SYNONYMES = """
Certaines informations de la fiche technique peuvent apparaître sous des intitulés différents. Voici des équivalences à prendre en compte :
- "Intitulé du produit" : "Dénomination légale", "Nom du produit", "Produit", "Nom commercial"
- "Estampille" : "Estampille sanitaire", "N° d’agrément", "Numéro d’agrément", "Agrément sanitaire", "FR xx.xxx.xxx CE", "CE", "FR", "Numero d'agrement"
- "Coordonnées du fournisseur" : "Adresse fournisseur", "Nom et adresse du fabricant", "Fournisseur", "Nom du fabricant", "Contact", "Adresse"
- "Origine" : "Origine", "Pays d’origine", "Origine viande", "Pays de provenance", "Provenance", "Origine biologique"
- "DLC / DLUO" : "Durée de vie", "Date limite de consommation", "Use by", "Durée étiquetée", "DDM", "DLC", "Date Durabilité", "Durée de conservation", "DLC / DDM"
- "Conditionnement / Emballage" : "Packaging", "Conditionnement", "Type d’emballage", "Type de contenant", "Colisage", "Palettisation", "Vrac", "Poids moyen", "Colis", "Unité", "Couvercle", "Carton", "Palette"
- "Température" : "Température de conservation", "Température de stockage", "Storage temperature", "Température max", "À conserver à", "Conservation à", "Conditions de conservation"
- "Composition du produit" : "Ingrédients", "Ingredients", "Composition", "Recette"
Prends-les en compte lors de l’analyse, même si la formulation ou l’orthographe est approximative.
"""

INSTRUCTIONS = f"""
Tu es un assistant expert qualité en agroalimentaire. Pour chaque fiche technique reçue, tu dois IMPÉRATIVEMENT analyser les 20 points de contrôle ci-dessous, dans l’ordre, même si l’information est absente ou douteuse.

{MAPPING_SYNONYMES}

**RÈGLE ABSOLUE :**
- Si le statut d’un point est "Présent", alors la criticité doit être vide, ou notée "Aucune" ou "RAS", et la recommandation doit être "Valider". Tu ne dois jamais écrire de remarque, nuance ou criticité sur un point "Présent", sauf si l’information semble manifestement incomplète ou douteuse (dans ce cas, note "Partiel" ou "Douteux" au lieu de "Présent").
- Pour tous les points marqués "Présent", il ne doit PAS y avoir de criticité, sauf doute explicite clairement justifié.
- Si le statut est "Non concerné", même logique : pas de criticité, recommandation "Valider".
- Pour les points "Non trouvé", "Partiel", "Douteux", indique la criticité adaptée avec une phrase d’explication.

**Avant d’indiquer qu’un point est "non trouvé", vérifie si des formulations approchantes, synonymes, abréviations, termes fragmentés ou mal orthographiés pourraient correspondre à l’information recherchée. Interprète largement les formulations et n’hésite pas à déduire le sens. Prends le bénéfice du doute si l’information semble présente.**

Pour certains points comme "Corps étranger", "VSM", "Aiguilles" : L'absence de mention signifie souvent que le risque est maîtrisé ou non concerné. Si rien n'est signalé dans la fiche, considère que c'est conforme, et indique simplement "non concerné" ou "absence attendue", et mets la recommandation "Valider", sauf si une anomalie réelle est détectée.

Même si la fiche ne donne AUCUNE info sur 15 points, tu dois quand même écrire un bloc “Nom du point…” pour chaque, dans l’ordre. N’arrête jamais l’analyse avant d’avoir commenté tous les points, même si tout est vide.

Format pour chaque point :

---
**Nom du point**
Statut : Présent / Partiel / Douteux / Non trouvé
Preuve : (citation du texte ou “non trouvé”)
Criticité : Critique / Majeur / Mineur + explication (uniquement si Partiel, Douteux ou Non trouvé)
Recommandation : (valider, demander complément, bloquant…)
---

Résumé :
- Points critiques (nombre) : [liste des points concernés]
- Points majeurs (nombre) : [liste des points concernés]
- Points mineurs (nombre) : [liste des points concernés]

- Décision recommandée : (valider / demander complément / refuser), avec OBLIGATOIREMENT une phrase explicative, constructive et professionnelle sur le niveau global de conformité, les forces du dossier et les points à compléter.
- Incohérences détectées : [liste]

**N’écris jamais de résumé ou de points critiques/majeurs/mineurs après chaque point, uniquement dans ce bloc final. Quand tu écris le résumé final, parcours les 20 points que tu viens d’analyser. Pour chaque point qui a le statut "Critique", "Majeur" ou "Mineur", ajoute son nom dans la liste correspondante. N’oublie aucun point, même ceux notés "Non trouvé" ou "Non concerné" s’ils ont une criticité. Le résumé doit TOUJOURS refléter exactement l’analyse faite point par point. Ne fais aucune synthèse “de mémoire” : base-toi sur ce que tu viens d’écrire.**

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
    report_text = re.sub(r'^([A-Za-zéèêàâùûôîïç /,()0-9’\']{5,50})\n', r'**\1**\n', report_text, flags=re.MULTILINE)
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

SYNONYMES = {
    "Intitulé du produit": ["Dénomination légale", "Nom du produit", "Produit", "Nom commercial"],
    "Estampille": ["Estampille sanitaire", "N° d’agrément", "Sanitary mark", "Agrément sanitaire", "FR xx.xxx.xxx CE"],
    "Coordonnées du fournisseur": ["Adresse fournisseur", "Nom et adresse du fabricant", "Fournisseur", "Nom du fabricant", "Contact", "Adresse"],
    "Origine": ["Origine", "Pays d’origine", "Origine viande", "Pays de provenance", "Provenance", "Origine biologique"],
    "DLC / DLUO": ["Durée de vie", "Date limite de consommation", "Use by", "Durée étiquetée", "DDM", "DLC", "Date Durabilité", "Durée de conservation", "DLC / DDM"],
    "Conditionnement / Emballage": ["Packaging", "Conditionnement", "Type d’emballage", "Type de contenant", "Colisage", "Palettisation", "Vrac", "Poids moyen", "Colis", "Unité", "Couvercle", "Carton", "Palette"],
    "Température": ["Température de conservation", "Température de stockage", "Storage temperature", "Température max", "À conserver à", "Conservation à"],
    "Composition du produit": ["Ingrédients", "Ingredients", "Composition", "Recette"],
    # ... ajoute tous les points
}

def tag_synonymes(text):
    for main, synos in SYNONYMES.items():
        for syn in synos:
            # Ajoute un tag dans le texte OCR
            # (Peut être un préfixe ou suffixe explicite pour aider GPT)
            regex = re.compile(rf"\b{syn}\b", re.IGNORECASE)
            text = regex.sub(f"[{main}]", text)
    return text
    
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
    # Titre en gras détecté par markdown (ex: **Nom du point**)
        bold_match = re.match(r'^\*\*(.+?)\*\*$', line.strip())
        if bold_match:
            title = bold_match.group(1)
            textobject.setFont("Helvetica-Bold", 12)
            textobject.setFillColor(HexColor("#22325c"))
            textobject.textLine(title)
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
