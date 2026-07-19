# dropout_student-
# 🎓 Student Burnout & Early Dropout Risk Prediction
> **Un approccio basato sul Machine Learning per l'identificazione precoce degli studenti a rischio di abbandono accademico e l'analisi dell'impatto del burnout sistemico.**

---

## 📌 Indice
1. [Contesto e Obiettivi del Progetto](#-contesto-e-obiettivi-del-progetto)
2. [Struttura e Dizionario dei Dati](#-struttura-e-dizionario-des-dati)
3. [Pipeline del Progetto](#-pipeline-del-progetto)
4. [Analisi Esplorativa dei Dati (EDA) e Risultati Key](#-analisi-esplorativa-dei-dati-eda-e-risultati-key)
5. [Strategia di Preprocessing e Feature Engineering](#-strategia-di-preprocessing-e-feature-engineering)
6. [Sviluppi Futuri (Modellazione e Valutazione)](#-sviluppi-futuri-modellazione-e-valutazione)
7. [Stack Tecnologico](#-stack-tecnologico)

---

## 🎯 Contesto e Obiettivi del Progetto

L'abbandono scolastico e universitario costituisce un problema socio-economico critico. Spesso le cause non sono puramente accademiche, ma risiedono in una complessa interconnessione tra stress psicologico, fattori economici, dinamiche relazionali e burnout.

Questo progetto si propone di sviluppare un **modello predittivo di classificazione binaria** in grado di stimare preventivamente il rischio di abbandono di uno studente (`Dropout_Risk`). L'obiettivo finale è fornire uno strumento di supporto per istituti scolastici e università, permettendo l'attivazione di interventi di tutoraggio e supporto mirati prima che l'abbandono si concretizzi.

---

## 📊 Struttura e Dizionario dei Dati

Il dataset analizzato raccoglie informazioni eterogenee su un campione rappresentativo di studenti, integrando metriche di performance, indicatori psicometrici e dati demografici.

### Dizionario delle Feature

| Nome Variabile | Tipo | Descrizione |
| :--- | :--- | :--- |
| **`Age`** | Numerico (Int) | Età anagrafica dello studente. |
| **`Gender`** | Categoriale (Nominale) | Genere di appartenenza dello studente. |
| **`Department`** | Categoriale (Nominale) | Facoltà o area di studio di iscrizione. |
| **`Attendance_Percent`** | Numerico (Float) | Percentuale di presenza alle lezioni e alle attività didattiche. |
| **`Study_Hours_Per_Day`** | Numerico (Float) | Ore medie giornaliere dedicate allo studio individuale. |
| **`Previous_GPA`** | Numerico (Float) | Media dei voti precedente (Performance accademica storica). |
| **`Sleep_Hours`** | Numerico (Float) | Ore medie di sonno per notte. |
| **`Residence_Type`** | Categoriale (Nominale) | Tipo di alloggio (es. Fuori sede, In sede, Campus). |
| **`Family_Income_Bracket`** | Categoriale (Ordinale) | Fascia di reddito familiare di appartenenza. |
| **`Financial_Stress_Score`**| Numerico (Int/Scale) | Livello percepito di stress finanziario (Scala numerica). |
| **`Family_Support_Score`** | Numerico (Int/Scale) | Livello di supporto percepito da parte della famiglia. |
| **`Stress_Level`** | Numerico (Int/Scale) | Indice generale di stress percepito dallo studente. |
| **`Anxiety_Score`** | Numerico (Int/Scale) | Punteggio derivato da test psicometrici per la valutazione dell'ansia. |
| **`Motivation_Score`** | Numerico (Int/Scale) | Livello intrinseco di motivazione accademica dello studente. |
| **`Peer_Pressure_Score`** | Numerico (Int/Scale) | Grado di pressione sociale o accademica percepito dai pari. |
| **`Counseling_Access`** | Categoriale (Binario) | Accesso o disponibilità a servizi di supporto psicologico (Yes/No). |
| **`Burnout_Level`** | Categoriale (Ordinale) | Grado di burnout complessivo stimato (Low, Medium, High). |
| **`Dropout_Risk` (TARGET)** | Categoriale (Binario) | Stato finale o rischio stimato di abbandono degli studi (**Yes** / **No**). |

---

## 🔄 Pipeline del Progetto

Il workflow segue la metodologia standard **CRISP-DM** per i progetti di Data Science:
1. **Data Ingestion & Cleaning:** Rimozione di feature non informative e controllo consistenza dei tipi di dato.
2. **Exploratory Data Analysis (EDA):** Analisi delle distribuzioni e studio delle relazioni multivariate.
3. **Data Preprocessing:** Codifica, gestione missing values e feature scaling.
4. **Model Training & Selection:** Addestramento e ottimizzazione degli iperparametri di più classificatori.
5. **Evaluation:** Validazione tramite metriche robuste (F1-Score, ROC-AUC) ponendo particolare attenzione ai Falsi Negativi.

---

## 📈 Analisi Esplorativa dei Dati (EDA) e Risultati Key

### 1. Caricamento e Data Sanitation
Nelle prime righe del codice, il dataset viene importato e privato della colonna `Student_ID`. Essendo un identificativo univoco sequenziale, non possiede potere predittivo ed introdurrebbe rumore o overfitting all'interno dei modelli.

### 2. Analisi di Bilanciamento del Target
L'ispezione della variabile dipendente `Dropout_Risk` ha evidenziato le seguenti frequenze:
* **Classe Positiva ("Yes"):** 437 studenti
* **Classe Negativa ("No"):** 363 studenti

> 💡 **Data Science Insight:** Il dataset si presenta **sostanzialmente bilanciato** (rapporto ~55:45). Questo scenario è ottimale in quanto mitiga il rischio di bias statistico verso la classe maggioritaria. Non si rende strettamente necessario l'utilizzo di algoritmi di ricampionamento artificiale (es. SMOTE), permettendo di focalizzarsi su metriche di performance standardizzate come Accuracy e Macro F1-Score.

### 3. Analisi di Correlazione Multivariata
L'estrazione della Matrice di Correlazione di Pearson sulle feature quantitative ha permesso di mappare la struttura spaziale del dataset. I focus principali di questa sotto-fase includono:
* L'individuazione di collinearità tra variabili psicometriche (es. `Stress_Level` vs `Anxiety_Score`).
* La quantificazione dell'impatto delle metriche comportamentali (`Sleep_Hours`, `Attendance_Percent`) sulle performance (`Previous_GPA`).

---

## 🛠️ Strategia di Preprocessing e Feature Engineering

Per rendere il set di dati digeribile dagli algoritmi di Machine Learning, la strategia pianificata si divide in tre direttrici basate sul tipo di dato:

1. **Feature Numeriche Continuous/Discrete:**
   * *Missing Values:* Monitoraggio e potenziale imputazione tramite Mediana o algoritmo `KNNImputer` in caso di distribuzioni asimmetriche.
   * *Scaling:* Applicazione di `StandardScaler` (Z-score normalization) per uniformare la varianza ed evitare che variabili su scale grandi (es. percentuali) dominino variabili su scale piccole (es. ore).
2. **Feature Categoriali Nominali/Ordinali:**
   * *One-Hot Encoding:* Applicato alle variabili nominali senza ordine gerarchico come `Gender` e `Department`.
   * *Ordinal Encoding:* Applicato a feature dotate di una progressione logica intrinseca, come `Burnout_Level` (Low $\rightarrow$ Medium $\rightarrow$ High) e `Family_Income_Bracket`.
3. **Target Variable:**
   * Trasformazione binarizzata via `LabelEncoder` mappando la classe di rischio `No` a `0` e la classe `Yes` a `1`.

---

## 🚀 Sviluppi Futuri (Modellazione e Valutazione)
*Lo sviluppo del codice è in costante aggiornamento. Le fasi successive prevedono:*
* [ ] Implementazione della pipeline di Preprocessing con `scikit-learn`.
* [ ] Benchmark di modelli lineari (Logistic Regression) ed ensemble (Random Forest, Gradient Boosting, XGBoost).
* [ ] Ottimizzazione degli iperparametri tramite Stratified K-Fold Cross Validation.
* [ ] Analisi delle Feature Importance (SHAP values) per interpretare i fattori decisionali del modello finalizzato.

---

## 💻 Stack Tecnologico
* **Language:** Python
* **Data Core:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Ambiente:** Google Colab / Jupyter Notebook
