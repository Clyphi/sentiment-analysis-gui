# -----------------------------------------------------------------
# Autor: Claudia Leins
# Erstellungsdatum: 2025-07-21
# Beschreibung: 
#   Dieses Skript implementiert ein einfaches "Bag of Words"-Modell 
#   zur Umwandlung von Texten in numerische Vektoren. Es besteht aus 
#   zwei Hauptfunktionen:
#     1. erstelle_vokabular(): Extrahiert ein sortiertes Vokabular 
#        aus einer Liste von Texten, optional begrenzt auf die 
#        häufigsten Wörter.
#     2. bag_of_words(): Wandelt einen einzelnen Text basierend auf 
#        dem gegebenen Vokabular in einen binären Vektor um, der 
#        anzeigt, welche Wörter enthalten sind.
# Anforderungen:
#   - scikit-learn (für ENGLISH_STOP_WORDS)
# -----------------------------------------------------------------

import string  # Für Zeichensetzung (Punktuation) zum Entfernen
from collections import Counter  # Zum Zählen von Wortfrequenzen
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # Englischsprachige Stoppwörter

def erstelle_vokabular(texte, begrenzen=True, max_worte=1000):
    """
    Erstellt ein sortiertes Vokabular aus einer Liste von Texten.

    Parameter:
        texte (list[str]): Eine Liste von Texten (Dokumenten).
        begrenzen (bool): Wenn True, wird das Vokabular auf die häufigsten Wörter beschränkt.
        max_worte (int): Maximale Anzahl der Wörter im Vokabular (nur relevant wenn begrenzen=True).

    Rückgabe:
        list[str]: Alphabetisch sortierte Liste der wichtigsten Wörter im Vokabular.
    """
    wortliste = []

    # Texte bereinigen und Token extrahieren
    for text in texte:
        # Kleinbuchstaben und Entfernung von Satzzeichen
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        
        # Tokenisierung und Entfernen von Stoppwörtern
        for wort in text.split():
            if wort not in ENGLISH_STOP_WORDS:
                wortliste.append(wort)

    # Zählen der Wortfrequenzen
    haeufigkeiten = Counter(wortliste)

    # Auswahl der häufigsten Wörter (falls begrenzt)
    top_worte = (
        [wort for wort, _ in haeufigkeiten.most_common(max_worte)]
        if begrenzen else list(haeufigkeiten.keys())
    )

    return sorted(top_worte)  # Alphabetische Sortierung für Konsistenz

def bag_of_words(text, vokabular):
    """
    Wandelt einen Text in einen binären Vektor auf Basis eines gegebenen Vokabulars um.

    Parameter:
        text (str): Eingabetext, der umgewandelt werden soll.
        vokabular (list[str]): Liste von Wörtern, die im Vektor berücksichtigt werden.

    Rückgabe:
        list[int]: Binärer Vektor (1 = Wort vorhanden, 0 = Wort nicht vorhanden).
    """
    # Vorverarbeitung: Kleinschreibung und Entfernen von Satzzeichen
    text = text.lower().translate(str.maketrans('', '', string.punctuation)).split()

    # Binäre Repräsentation: 1, wenn das Wort im Text vorkommt
    return [1 if wort in text else 0 for wort in vokabular]
