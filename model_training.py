# -----------------------------------------------------------------------------
# Autor: Claudia Leins
# Erstellungsdatum: 2025-07-21
# Beschreibung: 
# Dieses Skript implementiert eine Trainingsfunktion für ein Sentiment-Analyse-
# Modell. Es liest eine CSV-Datei mit Texten und zugehörigen Sentiment-Labels 
# ein, verarbeitet die Texte mithilfe eines Bag-of-Words-Ansatzes und trainiert 
# anschließend ein Random Forest Klassifikationsmodell. Nach dem Training wird 
# eine MQF-Kennzahl zur Bewertung der Modellqualität berechnet. Die Funktion 
# gibt das trainierte Modell, das Vokabular und den MQF-Wert zurück.
# Abhängigkeiten: pandas, scikit-learn, benutzerdefinierte Module: bag_of_words,
# evaluation
# -----------------------------------------------------------------------------

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from bag_of_words import erstelle_vokabular, bag_of_words
from evaluation import mqf_berechnen

def sentiment_train(filepath):
    """
    Trainiert ein Sentiment-Analysemodell basierend auf Textdaten.

    Parameter:
    filepath (str): Pfad zur CSV-Datei mit den Spalten 'text' und 'sentiment'.

    Rückgabewerte:
    - model: Das trainierte RandomForestClassifier-Modell.
    - vokabular: Die durch Bag-of-Words erzeugte Wortliste.
    - mqf: Der berechnete MQF-Wert zur Evaluierung der Modellgüte.
    """
    
    # Einlesen der CSV-Datei
    df = pd.read_csv(filepath)

    # Überprüfung, ob erforderliche Spalten vorhanden sind
    if 'text' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("Erforderliche Spalten 'text' oder 'sentiment' fehlen.")

    # Sicherstellen, dass alle Texte und Labels als Strings vorliegen
    df['text'] = df['text'].astype(str)
    df['sentiment'] = df['sentiment'].astype(str).str.strip().str.capitalize()

    # Extraktion der Texte und Sentiment-Labels
    texte = df['text'].tolist()
    labels = df['sentiment'].tolist()

    # Erstellung des Vokabulars auf Basis der Trainingsdaten
    vokabular = erstelle_vokabular(texte)

    # Umwandlung der Texte in numerische Vektoren mithilfe von Bag-of-Words
    features = [bag_of_words(text, vokabular) for text in texte]

    # Zulässige Sentiment-Klassen definieren
    allowed = {'Positive', 'Negative', 'Neutral', 'Irrelevant'}

    # Herausfiltern aller Datenpunkte, deren Label nicht zu den erlaubten gehört
    filtered = [(f, l) for f, l in zip(features, labels) if l in allowed]
    
    # Trennen von Merkmalen und Labels
    features_filtered, labels_filtered = zip(*filtered)

    # Aufteilen der Daten in Trainings- und Testmenge (80/20-Split)
    X_train, X_test, y_train, y_test = train_test_split(features_filtered, labels_filtered, test_size=0.2)

    # Initialisierung und Training des Random Forest Modells
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Berechnung der Modellqualitätskennzahl MQF
    mqf = mqf_berechnen(model, X_train, y_train)

    # Rückgabe von Modell, Vokabular und MQF
    return model, vokabular, mqf
