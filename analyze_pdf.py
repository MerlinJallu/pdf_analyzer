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

"""
-------------------------------------------------------------------------------
SCRIPT D'INSTRUCTIONS GPT – ANALYSE & VALIDATION FICHE TECHNIQUE PRODUIT
-------------------------------------------------------------------------------
Version : v21
Auteur  : Merlin Jallu pour l'Équipe Qualité (Bahier)
Objet   : Prompt normatif (unique) à fournir au modèle GPT pour contrôler la
          complétude et la conformité d’une fiche technique produit
          agro‑alimentaire.

Garanties clés :
• Analyse exhaustive des 20 points de contrôle (aucune omission possible).
• Classification risque Mineur / Majeur / Critique (matrice interne).
• Algorithme de décision globale (Valider / Demander complément / Refuser).
• **Zéro criticité** affichée quand le point est Conforme.
• Utilisation exclusive des statuts : « Conforme », « Douteux », « Non Conforme ».
-------------------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 1. TABLEAU DE SYNONYMES
# ---------------------------------------------------------------------------
MAPPING_SYNONYMES = """
Avant de conclure qu’une information est absente (« non trouvé »), élargis la
recherche aux entrées suivantes (tolère fautes d’orthographe, accents, casse) :

- **Intitulé du produit** : "Dénomination légale", "Nom du produit", "Nom commercial", "Produit"
- **Estampille** : "Estampille sanitaire", "N° d’agrément", "Numéro d’agrément", "Agrément sanitaire", "FR xx.xxx.xxx CE", "CE", "FR", "Numero d'agrement"
- **Présence d’une certification** : "VRF", "VVF", "BIO", "VPF", "VBF"
- **Mode de réception** : "Frais", "Congele", "Congelé", "Présentation", "Réfrigérée", "Réfrigéré", "Refrigere"
- **Coordonnées du fournisseur** : "Adresse fournisseur", "Nom et adresse du fabricant", "Fournisseur", "Nom du fabricant", "Contact", "Adresse"
- **Origine** : "Pays d’origine", "Origine viande", "Pays de provenance", "Provenance", "Origine biologique"
- **DLC / DLUO** : "Durée de vie", "Date limite de consommation", "Use by", "Durée étiquetée", "DDM", "Date durabilité", "Durée de conservation", "DLC / DDM"
- **Contaminants** : Règlements UE 1881/2006, 2022/2388, 2023/915, 1829/2003, 1830/2003…
- **Conditionnement / Emballage** : "Packaging", "Type d’emballage", "Colisage", "Palettisation", "Vrac", etc.
- **Température** : "Température de conservation", "Storage temperature", "À conserver à",
  "Conditions de conservation"
- **Composition du produit** : "Ingrédients", "Ingredients", "Composition", "Recette"
"""

# ---------------------------------------------------------------------------
# 2. MATRICE DE CRITICITÉ
# ---------------------------------------------------------------------------
POINTS_MINEURS = [
    "Intitulé du produit",
    "Coordonnées du fournisseur",
    "Présence d’une certification",
    "Mode de réception",
    "Process",
]

POINTS_MAJEURS = [
    "Conditionnement / Emballage",
    "Conservation",
    "Origine",
    "Contaminants",
    "Date du document",
]

POINTS_CRITIQUES = [
    "Estampille",
    "Température",
    "DLC / DLUO",
    "Espèce",
    "Corps Etranger",
    "VSM",
    "Aiguilles",
    "Composition du produit",
    "Critères Microbiologiques",
    "Critères physico‑chimiques",
]

# ---------------------------------------------------------------------------
# 3. DÉFINITIONS
# ---------------------------------------------------------------------------
STATUTS = {
    "Conforme": "Information présente et conforme à la réglementation.",
    "Douteux": "Information partielle, ambiguë ou non tracée.",
    "Non Conforme": "Information absente ou manifestement non conforme.",
}

RECOMMANDATIONS = {
    "Valider": "Aucune action requise avant approbation.",
    "Demander complément": "Compléter la fiche avant validation.",
    "Bloquant": "Refus tant que le point n'est pas corrigé.",
}

# ---------------------------------------------------------------------------
# 4. ALGORYTHME DE DÉCISION GLOBALE
# ---------------------------------------------------------------------------
"""
1. ≥ 1 point critique manquant / Non Conforme → **Refuser**.
2. Sinon, ≥ 1 point majeur manquant / Non Conforme → **Demander complément**.
3. Sinon, ≥ 1 point mineur manquant / Non Conforme → **Valider** (avec remarque).
4. Sinon → **Valider**.
"""

# ---------------------------------------------------------------------------
# 5. PROMPT FINAL À FOURNIR AU MODÈLE GPT
# ---------------------------------------------------------------------------
INSTRUCTIONS = f"""
Tu es un expert **auditeur qualité agro‑alimentaire**. Analyse la fiche technique
fournisseur en appliquant strictement les règles suivantes :

{MAPPING_SYNONYMES}

## EN‑TÊTE OBLIGATOIRE (avant les 20 points)
```
                    ====================================
                            <Intitulé du produit>
                              Date : JJ/MM/AAAA
                    ====================================
```
*Centre le titre, puis ajoute la date du jour sous la même forme.*  
*Ensuite seulement, enchaîne avec les 20 points d’analyse.*

## LÉGENDE DES CRITICITÉS
- **Mineur**  : {', '.join(POINTS_MINEURS)}
- **Majeur**  : {', '.join(POINTS_MAJEURS)}
- **Critique**: {', '.join(POINTS_CRITIQUES)}

## RÈGLES ABSOLUES (par point)
1. **Statuts autorisés** : Conforme / Douteux / Non Conforme (aucune variante).
2. **NE JAMAIS** afficher « Criticité » pour un point **Conforme**.
3. **NE JAMAIS** écrire « non trouvé » en *Statut* ; ce terme est réservé au champ **Preuve**.
   • Si l’information est introuvable → Statut = **Non Conforme** + Criticité adéquate.
4. Si la Preuve contient « non trouvé », « aucune mention », « absent » ou équivalent,
   alors le Statut **ne peut pas** être Conforme.
5. Pour un point **Douteux** ou **Non Conforme** :
   • « Criticité » obligatoire (**Mineur / Majeur / Critique**) + 1 phrase explicative.
   • « Recommandation » = « Demander complément » (Mineur/Majeur) ou « Bloquant » (Critique).
6. Pour « Corps Etranger », « VSM », « Aiguilles » : absence de mention = **Conforme**.
7. Aucun résumé intermédiaire – 20 blocs séparés uniquement.
8. Respecter EXACTEMENT l’orthographe des 20 titres fournis (ex. « Critères physico‑chimiques »).

## FORMAT PAR POINT (répéter exactement 20×) :
```
---
**<Nom du point>**
Statut : Conforme / Douteux / Non Conforme
Preuve : « … » | « non trouvé »
Criticité : <uniquement si Douteux ou Non Conforme> Mineur | Majeur | Critique – explication
Recommandation : Valider | Demander complément | Bloquant
---
```

## **Résumé final** (après les 20 points)

- Points critiques (n) : [liste]
- Points majeurs (n)  : [liste]
- Points mineurs (n)  : [liste]

---

- **Préconisation globale** : Valider / Demander complément / Refuser

---

- Incohérences détectées : [liste]

## ALGORYTHME DE DÉCISION (strict)
1. ≥ 1 Critique manquant / Non Conforme → Refuser.
2. Sinon, ≥ 1 Majeur manquant / Non Conforme → Demander complément.
3. Sinon, ≥ 1 Mineur manquant / Non Conforme → Valider (avec remarque).
4. Sinon → Valider.

⚠️ *Le résumé doit refléter exactement les statuts renseignés. Aucune divergence.*
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
