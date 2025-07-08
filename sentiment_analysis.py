# Autor: Claudia Leins
# Erstellungsdatum: 2025-07-08
# Beschreibung: Sentiment-Analyse-Tool mit grafischer Benutzeroberfläche (GUI)
"""
Dieses Python-Programm ermöglicht es, Textdaten aus CSV-Dateien auf ihre Stimmung (positiv oder negativ) zu analysieren.
Die Analyse basiert auf einem RandomForest-Modell, das auf Trainingsdaten mit bekannten Sentiment-Labels trainiert wird.
Die Texte werden mithilfe eines einfachen Bag-of-Words-Verfahrens in numerische Vektoren umgewandelt.

Hauptfunktionen:
- Laden und Trainieren eines Modells mit einer CSV-Trainingsdatei
- Vorhersage von Sentiments für neue Texte in einer CSV-Datei
- Farbcodierte und strukturierte Darstellung der Analyseergebnisse (positiv/negativ)
- Speichern der Ergebnisse als neue CSV-Datei

Dieses Tool eignet sich gut für einfache Textklassifikationsaufgaben (z.B. Produktbewertungen, Kommentare, Meinungsanalysen).
Es ist bewusst einfach gehalten und eignet sich auch für Einsteiger.
"""
#-----------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from tkinter import Tk, Text, Button, Scrollbar, END, filedialog, messagebox
import datetime
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier
import string

# --- Hilfsfunktionen Bag of Words ---
def erstelle_vokabular(texte, begrenzen=True, max_worte=1000):
    wortliste = []
    for text in texte:
        # Satzzeichen entfernen
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        for wort in text.split():
            if wort not in ENGLISH_STOP_WORDS:
                wortliste.append(wort)

    haeufigkeiten = Counter(wortliste)

    if begrenzen:
        top_worte = [wort for wort, _ in haeufigkeiten.most_common(max_worte)]
        return sorted(top_worte)
    else:
        return sorted(list(haeufigkeiten.keys()))

def bag_of_words(text, vokabular):
    text = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    return [1 if wort in text else 0 for wort in vokabular]

def zeit_und_datum():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Globale Variablen ---
sentiment_model = None
vokabular_sentiment = None
sentiment_analyse_ergebnisse = None

# --- Training ---
def sentiment_train(filepath):
    global sentiment_model, vokabular_sentiment

    if not filepath:
        return "Kein Pfad zur Trainingsdatei ausgewählt."

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return f"Fehler beim Laden der Datei: {e}"

    if 'text' not in df.columns or 'sentiment' not in df.columns:
        return "Die Datei muss die Spalten 'text' und 'sentiment' enthalten."

    df['text'] = df['text'].astype(str)
    # Einheitliche Groß-/Kleinschreibung und Whitespace entfernen
    df['sentiment'] = df['sentiment'].astype(str).str.strip().str.capitalize()

    texte = df['text'].tolist()
    labels = df['sentiment'].tolist()

    vokabular_sentiment = erstelle_vokabular(texte, begrenzen=True)

    if not vokabular_sentiment:
        return "Fehler: Kein Vokabular erzeugt."

    features = [bag_of_words(text, vokabular_sentiment) for text in texte]

    # Akzeptierte Klassen erweitern
    allowed_labels = {'Positive', 'Negative', 'Neutral', 'Irrelevant'}
    filtered = [(f, l) for f, l in zip(features, labels) if l in allowed_labels]
    if not filtered:
        return f"Keine passenden Labels zum Trainieren gefunden. Erlaubt sind: {allowed_labels}"

    features_filtered, labels_filtered = zip(*filtered)

    X_train, X_test, y_train, y_test = train_test_split(features_filtered, labels_filtered, test_size=0.2, random_state=42)

    sentiment_model = RandomForestClassifier(n_estimators=100, random_state=42)
    sentiment_model.fit(X_train, y_train)

    accuracy = sentiment_model.score(X_test, y_test)
    return f"Training erfolgreich! Genauigkeit: {accuracy:.2f}"

# --- Analyse ---
def sentiment_analyse(filepath):
    global sentiment_model, vokabular_sentiment, sentiment_analyse_ergebnisse

    if sentiment_model is None or vokabular_sentiment is None:
        return "Bitte zuerst Trainingsdatei laden und trainieren!"

    if not filepath:
        return "Keine Datei für Analyse ausgewählt."

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return f"Fehler beim Laden der Datei: {e}"

    if 'text' not in df.columns:
        return "Die Datei muss eine Spalte 'text' enthalten."

    df = df.dropna(subset=['text'])
    df = df[df['text'].str.len() > 5]
    df['text'] = df['text'].astype(str)

    features = [bag_of_words(text, vokabular_sentiment) for text in df['text'].tolist()]
    vorhersagen = sentiment_model.predict(features)
    df['predicted_sentiment'] = vorhersagen
    sentiment_analyse_ergebnisse = df

    # Ergebnis DataFrames für alle Klassen
    df_pos = df[df['predicted_sentiment'] == 'Positive']
    df_neg = df[df['predicted_sentiment'] == 'Negative']
    df_neutral = df[df['predicted_sentiment'] == 'Neutral']
    df_irrelevant = df[df['predicted_sentiment'] == 'Irrelevant']

    def format_block(df_subset, label, emoji):
        output = f"\n{emoji} {label} Kommentare:\n" + "-" * 65 + "\n"
        for _, row in df_subset.iterrows():
            text = row['text']
            gekuerzt = text[:57] + "..." if len(text) > 60 else text
            output += f"{gekuerzt:<60} | {row['predicted_sentiment']}\n"
        return output

    ausgabe = ""
    ausgabe += format_block(df_pos, "Positive", "🟢")
    ausgabe += format_block(df_neg, "Negative", "🔴")
    ausgabe += format_block(df_neutral, "Neutral", "🟡")
    ausgabe += format_block(df_irrelevant, "Irrelevant", "⚪")

    return ausgabe

# --- Ergebnisse speichern ---
def sentiment_ergebnisse_speichern():
    global sentiment_analyse_ergebnisse
    if sentiment_analyse_ergebnisse is None:
        messagebox.showwarning("Warnung", "Keine Analyseergebnisse zum Speichern!")
        return

    speicherpfad = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV Dateien", "*.csv")],
        title="Speicherort für Analyseergebnisse wählen"
    )
    if speicherpfad:
        sentiment_analyse_ergebnisse.to_csv(speicherpfad, index=False)
        messagebox.showinfo("Erfolg", f"Ergebnisse gespeichert: {speicherpfad}")

# --- Log-Ausgabe in Text-Widget ---
def log_ausgabe(text):
    logbuch.config(state="normal")
    logbuch.insert(END, f"{zeit_und_datum()} - {text}\n\n")
    logbuch.see(END)
    logbuch.config(state="disabled")

# --- GUI ---
fenster = Tk()
fenster.title("Sentiment-Analyse")

scrollbar = Scrollbar(fenster)
scrollbar.grid(column=2, row=0, rowspan=10, sticky='ns')

logbuch = Text(fenster, height=20, width=80, wrap='word', yscrollcommand=scrollbar.set, state="disabled")
logbuch.grid(column=0, row=0, columnspan=2, padx=10, pady=10)
scrollbar.config(command=logbuch.yview)

btn_train_load = Button(
    fenster, text="Trainingsdatei laden & trainieren",
    command=lambda: log_ausgabe(sentiment_train(filedialog.askopenfilename(filetypes=[("CSV Dateien", "*.csv")])))
)
btn_train_load.grid(column=0, row=7, padx=10, pady=5, sticky="w")

btn_analyse_load = Button(
    fenster, text="Analysedatei laden & analysieren",
    command=lambda: log_ausgabe(sentiment_analyse(filedialog.askopenfilename(filetypes=[("CSV Dateien", "*.csv")])))
)
btn_analyse_load.grid(column=1, row=7, padx=10, pady=5, sticky="w")

btn_ergebnisse_speichern = Button(fenster, text="Ergebnisse speichern", command=sentiment_ergebnisse_speichern)
btn_ergebnisse_speichern.grid(column=0, row=8, padx=10, pady=5, sticky="w")

fenster.mainloop()