# Autor: Claudia Leins
# Erstellungsdatum: 2025-07-21
# Beschreibung: Sentiment-Analyse-Tool mit grafischer Benutzeroberfläche (GUI),
# Version 2 mit Modulen
# 
# Dieses Python-Programm ermöglicht es, Textdaten aus CSV-Dateien auf ihre Stimmung (positiv oder negativ) zu analysieren.
# Die Analyse basiert auf einem RandomForest-Modell, das auf Trainingsdaten mit bekannten Sentiment-Labels trainiert wird.
# Die Texte werden mithilfe eines einfachen Bag-of-Words-Verfahrens in numerische Vektoren umgewandelt.
# 
# Hauptfunktionen:
# - Laden und Trainieren eines Modells mit einer CSV-Trainingsdatei
# - Vorhersage von Sentiments für neue Texte in einer CSV-Datei
# - Farbcodierte und strukturierte Darstellung der Analyseergebnisse (positiv/negativ)
# - Speichern der Ergebnisse als neue CSV-Datei
# 
# Dieses Tool eignet sich gut für einfache Textklassifikationsaufgaben (z.B. Produktbewertungen, Kommentare, Meinungsanalysen).
# Es ist bewusst einfach gehalten und eignet sich auch für Einsteiger.
#-----------------------------------------------------------------

import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, Text, END
import pandas as pd
from datetime import datetime

from model_training import sentiment_train
from model_analysis import sentiment_analyse

# Globale Zustände
sentiment_model = None
vokabular_sentiment = None
analyse_ergebnisse = None

# --- Hilfsfunktionen ---
def zeit_und_datum():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_ausgabe(text):
    logbuch.config(state="normal")
    logbuch.insert(END, f"{zeit_und_datum()} - {text}\n\n")
    logbuch.see(END)
    logbuch.config(state="disabled")

# --- Aktionen ---
def lade_und_trainiere():
    global sentiment_model, vokabular_sentiment
    pfad = filedialog.askopenfilename(filetypes=[("CSV Dateien", "*.csv")])
    if not pfad:
        return
    try:
        model, vokabular, mqf = sentiment_train(pfad)
        sentiment_model = model
        vokabular_sentiment = vokabular

        log = f"Training erfolgreich!\nGenauigkeit: {mqf['Accuracy']:.2f}, MQF: {mqf['MQF']:.2f}"
        log += f"\nF1-Score: {mqf['F1-Score']:.2f}, Robustheit: {mqf['Robustheit (CV)']:.2f}"
        log_ausgabe(log)
    except Exception as e:
        log_ausgabe(f"Fehler beim Training: {e}")

def analysiere_datei():
    global analyse_ergebnisse
    if not sentiment_model or not vokabular_sentiment:
        log_ausgabe("Bitte zuerst ein Modell trainieren!")
        return

    pfad = filedialog.askopenfilename(filetypes=[("CSV Dateien", "*.csv")])
    if not pfad:
        return
    try:
        df = sentiment_analyse(pfad, sentiment_model, vokabular_sentiment)
        analyse_ergebnisse = df

        zähler = df['predicted_sentiment'].value_counts().to_dict()
        zusammenfassung = ", ".join(f"{k}: {v}" for k, v in zähler.items())
        log_ausgabe(f"Analyse abgeschlossen – Verteilung: {zusammenfassung}")
    except Exception as e:
        log_ausgabe(f"Fehler bei der Analyse: {e}")

def speichere_ergebnisse():
    if analyse_ergebnisse is None:
        messagebox.showwarning("Warnung", "Keine Analyseergebnisse vorhanden.")
        return

    pfad = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV Dateien", "*.csv")],
        title="Analyseergebnisse speichern"
    )
    if pfad:
        analyse_ergebnisse.to_csv(pfad, index=False)
        messagebox.showinfo("Gespeichert", f"Ergebnisse gespeichert:\n{pfad}")

# --- GUI Aufbau ---
fenster = tk.Tk()
fenster.title("Sentiment-Analyse Tool")

# Scrollbarer Logbereich
scrollbar = Scrollbar(fenster)
scrollbar.grid(column=2, row=0, rowspan=10, sticky='ns')

logbuch = Text(fenster, height=20, width=80, wrap='word', yscrollcommand=scrollbar.set, state="disabled")
logbuch.grid(column=0, row=0, columnspan=2, padx=10, pady=10)
scrollbar.config(command=logbuch.yview)

# Buttons
btn_train = tk.Button(fenster, text="Trainingsdatei laden & trainieren", command=lade_und_trainiere)
btn_train.grid(column=0, row=1, padx=10, pady=5, sticky="w")

btn_analyse = tk.Button(fenster, text="Analysedatei laden & analysieren", command=analysiere_datei)
btn_analyse.grid(column=1, row=1, padx=10, pady=5, sticky="w")

btn_speichern = tk.Button(fenster, text="Analyseergebnisse speichern", command=speichere_ergebnisse)
btn_speichern.grid(column=0, row=2, padx=10, pady=5, sticky="w")


# Hauptloop starten
fenster.mainloop()
