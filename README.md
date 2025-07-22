
# Sentiment-Analyse-Tool mit GUI

**Autor:** Claudia Leins  
**Erstellt am:** 08.07.2025

## 📝 Projektbeschreibung

Dieses Python-Tool bietet eine einfache grafische Oberfläche zur **Sentiment-Analyse von Textdaten aus CSV-Dateien**. Es ermöglicht das **Trainieren eines Modells**, das anschließend neue Texte als _positiv_, _negativ_, _neutral_ oder _irrelevant_ klassifizieren kann. Die Analyse basiert auf einem einfachen **Bag-of-Words-Verfahren** in Kombination mit einem **RandomForestClassifier**.

## 🎯 Hauptfunktionen

- 📂 Laden einer CSV-Trainingsdatei mit Texten und Sentiment-Labels  
- 🤖 Training eines RandomForest-Modells zur Textklassifikation  
- 🔍 Analyse neuer CSV-Dateien mit Texten  
- 🎨 Farbliche und strukturierte GUI-Ausgabe der Ergebnisse  
- 💾 Speichern der Ergebnisse als neue CSV-Datei  

---

## 🛠️ Voraussetzungen

- Python **3.7+**
- Folgende Python-Bibliotheken müssen installiert sein:

```bash
pip install pandas scikit-learn
```

_Hinweis: `tkinter` ist bei den meisten Python-Distributionen bereits enthalten._

## 📁 CSV-Dateiformate

### 🏋️‍♀️ Trainingsdatei

Beispieldateien: `twitter_training.csv`, `sentiment_analysis.csv` (abgespeckt)

**Erforderliche Spalten:**
- `text`: Die zu analysierenden Texte
- `sentiment`: Das zugehörige Label (`Positive`, `Negative`, `Neutral`, `Irrelevant`)

## 📊 Beispiel-Datensätze

Die Dateien `sentiment_analysis.csv` und `twitter_training.csv` wurden im Rahmen einer Lehrveranstaltung bereitgestellt und dienen ausschließlich der Veranschaulichung und Übung zur Sentiment-Analyse. Die enthaltenen Texte stammen aus öffentlich zugänglichen Quellen (z. B. Twitter) und enthalten keine sensiblen personenbezogenen Daten.

**Beispiel:**
```
text,sentiment
"I love this!",Positive
"This is terrible.",Negative
"I don't care.",Neutral
"N/A",Irrelevant
```

### 📊 Analysedatei

- Muss mindestens die Spalte `text` enthalten.

## 🚀 Starten des Tools

```bash
python sentiment_analyse_tool.py
```

Es öffnet sich ein GUI-Fenster mit folgenden Funktionen:

- Trainingsdatei laden & trainieren  
- Modelltraining mit ausgewählter CSV-Datei  
- Analysedatei laden & analysieren  
- Vorhersage der Stimmung für neue Texte  
- Ergebnisse speichern als neue CSV-Datei  

## 🧠 Funktionsweise

- Bag-of-Words-Vektor: Konvertiert Texte in Vektoren (1 = Wort vorhanden, 0 = nicht vorhanden)  
- Stopwörter: Englische Stopwörter werden entfernt  
- Modell: RandomForestClassifier mit 100 Bäumen  
- Label-Filter: Nur die Klassen `Positive`, `Negative`, `Neutral`, `Irrelevant` werden berücksichtigt  

## 📦 Ergebnisanzeige

Nach der Analyse werden die Texte gruppiert und farblich markiert:

| Symbol | Kategorie  |
|--------|------------|
| 🟢     | Positive   |
| 🔴     | Negative   |
| 🟡     | Neutral    |
| ⚪     | Irrelevant |

**Beispielausgabe im Log-Feld:**

🟢 Positive Kommentare:
-------------------------------------------------------------  
This product is amazing!                          | Positive  
Excellent service                                 | Positive  

🔴 Negative Kommentare:
-------------------------------------------------------------  
Worst experience I've had.                        | Negative  

## 💾 Ergebnisse speichern

- Speichern als neue CSV-Datei mit Spalte `predicted_sentiment`  
- Dateiname und Speicherort frei wählbar  

## 🧩 Erweiterungsideen

- Verwendung von TF-IDF oder Wort-Embeddings  
- Visualisierung als Balkendiagramm (z. B. Anzahl pro Sentiment)  
- Modell als `.pkl` exportieren und wiederverwenden  
- Mehrsprachigkeit hinzufügen  

## 📬 Kontakt

Bei Rückfragen oder Feedback: Claudia Leins

## ⚖️ Lizenz

Dieses Projekt ist bewusst einfach gehalten und kann frei für Lernzwecke verwendet werden.  
Für produktive Anwendungen bitte weiterentwickeln oder auf robuste Bibliotheken wie spaCy oder HuggingFace zurückgreifen.

Viel Spaß bei der Sentiment-Analyse!
