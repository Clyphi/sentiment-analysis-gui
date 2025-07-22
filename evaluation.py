# Autor: Claudia Leins
# Erstellungsdatum: 2025-07-21
# Beschreibung:
# Dieses Skript definiert eine Funktion zur Berechnung des sogenannten "MQF" (Modellqualitätsfaktor)
# für ein gegebenes Machine-Learning-Modell. Der MQF ist ein zusammengesetzter Metrikwert, der vier
# Aspekte der Modellqualität berücksichtigt: Genauigkeit (Accuracy), F1-Score, Robustheit mittels 
# Kreuzvalidierung und Interpretierbarkeit basierend auf den Feature-Importances.
# Der MQF wird als gewichtete Summe dieser vier Metriken berechnet.

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

def mqf_berechnen(model, X, y):
    """
    Berechnet den Modellqualitätsfaktor (MQF) für ein gegebenes Modell.

    Parameter:
    model: Ein trainiertes Machine-Learning-Modell mit einer `score()`- und `predict()`-Methode
           sowie dem Attribut `feature_importances_`.
    X: Feature-Matrix (Input-Daten)
    y: Zielvariable (Labels)

    Rückgabe:
    Dictionary mit MQF und den Einzelmetriken: Accuracy, F1-Score, Robustheit, Interpretierbarkeit
    """

    # Berechnung der Genauigkeit (Anteil korrekt vorhergesagter Labels)
    accuracy = model.score(X, y)

    # Vorhersagen des Modells auf den Eingabedaten
    y_pred = model.predict(X)

    # Berechnung des gewichteten F1-Scores (berücksichtigt sowohl Präzision als auch Recall)
    f1 = f1_score(y, y_pred, average='weighted')

    # Robustheit: Durchschnittliche Genauigkeit über 5-fache Kreuzvalidierung
    robustness = cross_val_score(model, X, y, cv=5).mean()

    # Interpretierbarkeit: Anteil wichtiger Features (mit Importanz > 0.01)
    interpretability = sum(1 for i in model.feature_importances_ if i > 0.01) / len(model.feature_importances_)

    # MQF-Berechnung als gewichtete Summe der vier Qualitätsmetriken
    mqf = 0.4 * accuracy + 0.3 * f1 + 0.2 * robustness + 0.1 * interpretability

    # Rückgabe der Ergebnisse als Dictionary
    return {
        "MQF": mqf,
        "Accuracy": accuracy,
        "F1-Score": f1,
        "Robustheit (CV)": robustness,
        "Interpretierbarkeit": interpretability
    }
