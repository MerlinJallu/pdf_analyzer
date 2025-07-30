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
SCRIPT D'INSTRUCTIONS GPT – ANALYSE ET VALIDATION DE FICHE TECHNIQUE PRODUIT
-------------------------------------------------------------------------------
Objectif : fournir au modèle GPT un prompt unique, exhaustif et **opérationnel**
pour évaluer la conformité documentaire d'une fiche technique produit en
agro‑alimentaire. Les exigences suivantes doivent être respectées à la lettre :

• Exhaustivité – analyse systématique des 20 points de contrôle, sans omission.
• Rigueur technique – définitions normatives, classification des risques,
  algorithme de décision globale intégré.
• Clarté opératoire – format de sortie strictement structuré et répétable.

Ce fichier contient :
    1. Le tableau de synonymes pour la recherche d'informations.
    2. Les listes de points classés **Mineur / Majeur / Critique**.
    3. Les définitions officielles des Statuts, Criticités et Recommandations.
    4. L'algorithme de décision de la préconisation globale.
    5. La chaîne `INSTRUCTIONS` servant de prompt au modèle GPT.
-------------------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 1. TABLEAU DE SYNONYMES – DÉTECTION DES INTITULÉS APPROCHANTS
# ---------------------------------------------------------------------------
MAPPING_SYNONYMES = """
Certaines informations peuvent apparaître sous des intitulés divergents.
Reconnais les équivalences suivantes (tolère fautes d’orthographe, accents,
abréviations et casse) :

- **Intitulé du produit** : "Dénomination légale", "Nom du produit", "Nom commercial", "Produit"
- **Estampille** : "Estampille sanitaire", "N° d’agrément", "Numéro d’agrément", "Agrément sanitaire", "FR xx.xxx.xxx CE", "CE", "FR", "Numero d'agrement"
- **Présence d’une certification** : "VRF", "VVF", "BIO", "VPF", "VBF"
- **Mode de réception** : "Frais", "Congele", "Congelé", "Présentation", "Réfrigérée", "Réfrigéré", "Refrigere"
- **Coordonnées du fournisseur** : "Adresse fournisseur", "Nom et adresse du fabricant", "Fournisseur", "Nom du fabricant", "Contact", "Adresse"
- **Origine** : "Pays d’origine", "Origine viande", "Pays de provenance", "Provenance", "Origine biologique"
- **DLC / DLUO** : "Durée de vie", "Date limite de consommation", "Use by", "Durée étiquetée", "DDM", "Date durabilité", "Durée de conservation", "DLC / DDM"
- **Contaminants** : références aux règlements UE (ex. 1881/2006, 2022/2388, 2023/915, 1829/2003, 1830/2003)
- **Conditionnement / Emballage** : "Packaging", "Type d’emballage", "Type de contenant", "Colisage", "Palettisation", "Vrac", "Poids moyen", "Colis", "Unité", "Couvercle", "Carton", "Palette"
- **Température** : "Température de conservation", "Température de stockage", "Storage temperature", "Température max", "À conserver à", "Conservation à", "Conditions de conservation"
- **Composition du produit** : "Ingrédients", "Ingredients", "Composition", "Recette"

➡️ *Toujours élargir la recherche à ces synonymes avant de conclure « non trouvé ».*
"""

# ---------------------------------------------------------------------------
# 2. CLASSIFICATION PAR NIVEAU DE GRAVITÉ (légende réglementaire interne)
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
    "Critères physico-chimiques",
]

# ---------------------------------------------------------------------------
# 3. DÉFINITIONS OPÉRATIONNELLES
# ---------------------------------------------------------------------------
STATUTS = {
    "Conforme": "Information présente, cohérente, et/ou exigence réglementaire respectée.",
    "Douteux": "Information partielle, ambiguë, ou non vérifiable.",
    "Non Conforme": "Information absente ou manifestement non conforme à la réglementation ou au cahier des charges.",
}

RECOMMANDATIONS = {
    "Valider": "Aucune action requise avant approbation de la fiche.",
    "Demander complément": "Compléter la fiche avant validation définitive.",
    "Bloquant": "Refus immédiat tant que le point n'est pas corrigé.",
}

# ---------------------------------------------------------------------------
# 4. ALGORITHME DE DÉCISION GLOBALE
# ---------------------------------------------------------------------------
"""
Étapes (à énoncer dans le prompt) :
1. Compter les points **critiques** manquants ou Non Conformes.
   • Si ≥ 1 → Préconisation = Refuser.
2. Sinon, compter les points **majeurs** manquants ou Non Conformes.
   • Si ≥ 1 → Préconisation = Demander complément.
3. Sinon, si un ou plusieurs points **mineurs** manquants ou Non Conformes.
   • Préconisation = Valider (avec remarque d’amélioration).
4. Sinon (tous points conformes) → Préconisation = Valider.
"""

# ---------------------------------------------------------------------------
# 5. PROMPT FINAL « INSTRUCTIONS » À PASSER AU MODÈLE
# ---------------------------------------------------------------------------
INSTRUCTIONS = f"""
Tu es un **assistant qualité agroalimentaire** expert en réglementation européenne.
Ta mission : contrôler une fiche technique fournisseur en appliquant le protocole
exhaustif ci‑dessous, puis émettre une préconisation documentée.

{MAPPING_SYNONYMES}

## LÉGENDE DES CRITICITÉS (colonne « Criticité »)
- **Mineur** : {', '.join(POINTS_MINEURS)}  
  → Absence possible sans bloquer la validation.
- **Majeur** : {', '.join(POINTS_MAJEURS)}  
  → Validation sous réserve d’un complément fournisseur.
- **Critique** : {', '.join(POINTS_CRITIQUES)}  
  → Absence d’UNE SEULE info = fiche non validable (bloquante).

## STATUTS PERMIS
- Conforme / Douteux / Non Conforme (respecter la casse et l’orthographe).

## RÈGLES ABSOLUES INDIVIDUELLES
1. Le document commence par **l’Intitulé du produit** (centré) + **date du jour** (JJ/MM/AAAA).
2. Pour un point **Conforme** :
   - Champ « Criticité » vide ou « Aucune »/« RAS ».
   - Champ « Recommandation » = « Valider ».
3. Pour un point **Non Conforme** ou **Douteux** :
   - « Criticité » = Mineur / Majeur / Critique + **1 phrase explicative**.
   - « Recommandation » = « Demander complément » (Mineur/Majeur) ou « Bloquant » (Critique).
4. Ne jamais fusionner de points ni insérer de résumé avant la fin des 20 blocs.
5. Pour « Corps Etranger », « VSM » et « Aiguilles » : l’absence de mention = Conforme.

## FORMAT STRICT PAR POINT (copier 20×)
```
---
**<Nom du point>**
Statut : Conforme / Douteux / Non Conforme
Preuve : « … » ou « non trouvé »
Criticité : (vide) | Mineur | Majeur | Critique + explication (obligatoire si Douteux ou Non Conforme)
Recommandation : Valider | Demander complément | Bloquant
---
```

## RÉSUMÉ FINAL (obligatoire après les 20 points)
- Points critiques (n) : [liste]
- Points majeurs (n) : [liste]
- Points mineurs (n) : [liste]

- **Préconisation** : Valider / Demander complément / Refuser  
  → Formuler 1 phrase professionnelle résumant la décision au regard de l’algorithme.
- Incohérences détectées : [liste] (ex. dates contradictoires, volume incohérent, etc.)

## ALGORITHME DE DÉCISION
Applique exactement la logique suivante :
1. ≥ 1 point critique manquant ou Non Conforme → Préconisation = Refuser.
2. Sinon, ≥ 1 point majeur manquant ou Non Conforme → Préconisation = Demander complément.
3. Sinon, ≥ 1 point mineur manquant ou Non Conforme → Préconisation = Valider.
4. Sinon → Valider.

⚠️ *Le résumé doit refléter **fidèlement** les statuts et criticités renseignés
point par point. Aucune divergence tolérée.*
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
