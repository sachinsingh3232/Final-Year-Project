# Disease Prediction based on Symptoms
# Importing Libraries
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# List of the symptoms is listed here in list l1.

l1 = [
    "back_pain",
    "constipation",
    "abdominal_pain",
    "diarrhoea",
    "mild_fever",
    "yellow_urine",
    "yellowing_of_eyes",
    "acute_liver_failure",
    "fluid_overload",
    "swelling_of_stomach",
    "swelled_lymph_nodes",
    "malaise",
    "blurred_and_distorted_vision",
    "phlegm",
    "throat_irritation",
    "redness_of_eyes",
    "sinus_pressure",
    "runny_nose",
    "congestion",
    "chest_pain",
    "weakness_in_limbs",
    "fast_heart_rate",
    "pain_during_bowel_movements",
    "pain_in_anal_region",
    "bloody_stool",
    "irritation_in_anus",
    "neck_pain",
    "dizziness",
    "cramps",
    "bruising",
    "obesity",
    "swollen_legs",
    "swollen_blood_vessels",
    "puffy_face_and_eyes",
    "enlarged_thyroid",
    "brittle_nails",
    "swollen_extremeties",
    "excessive_hunger",
    "extra_marital_contacts",
    "drying_and_tingling_lips",
    "slurred_speech",
    "knee_pain",
    "hip_joint_pain",
    "muscle_weakness",
    "stiff_neck",
    "swelling_joints",
    "movement_stiffness",
    "spinning_movements",
    "loss_of_balance",
    "unsteadiness",
    "weakness_of_one_body_side",
    "loss_of_smell",
    "bladder_discomfort",
    "foul_smell_of urine",
    "continuous_feel_of_urine",
    "passage_of_gases",
    "internal_itching",
    "toxic_look_(typhos)",
    "depression",
    "irritability",
    "muscle_pain",
    "altered_sensorium",
    "red_spots_over_body",
    "belly_pain",
    "abnormal_menstruation",
    "dischromic _patches",
    "watering_from_eyes",
    "increased_appetite",
    "polyuria",
    "family_history",
    "mucoid_sputum",
    "rusty_sputum",
    "lack_of_concentration",
    "visual_disturbances",
    "receiving_blood_transfusion",
    "receiving_unsterile_injections",
    "coma",
    "stomach_bleeding",
    "distention_of_abdomen",
    "history_of_alcohol_consumption",
    "fluid_overload",
    "blood_in_sputum",
    "prominent_veins_on_calf",
    "palpitations",
    "painful_walking",
    "pus_filled_pimples",
    "blackheads",
    "scurring",
    "skin_peeling",
    "silver_like_dusting",
    "small_dents_in_nails",
    "inflammatory_nails",
    "blister",
    "red_sore_around_nose",
    "yellow_crust_ooze",
]
# List of Diseases is listed in list disease.

disease = [
    "Fungal infection",
    "Allergy",
    "GERD",
    "Chronic cholestasis",
    "Drug Reaction",
    "Peptic ulcer diseae",
    "AIDS",
    "Diabetes ",
    "Gastroenteritis",
    "Bronchial Asthma",
    "Hypertension ",
    "Migraine",
    "Cervical spondylosis",
    "Paralysis (brain hemorrhage)",
    "Jaundice",
    "Malaria",
    "Chicken pox",
    "Dengue",
    "Typhoid",
    "hepatitis A",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Hepatitis E",
    "Alcoholic hepatitis",
    "Tuberculosis",
    "Common Cold",
    "Pneumonia",
    "Dimorphic hemmorhoids(piles)",
    "Heart attack",
    "Varicose veins",
    "Hypothyroidism",
    "Hyperthyroidism",
    "Hypoglycemia",
    "Osteoarthristis",
    "Arthritis",
    "(vertigo) Paroymsal  Positional Vertigo",
    "Acne",
    "Urinary tract infection",
    "Psoriasis",
    "Impetigo",
]

l2 = []
for i in range(0, len(l1)):
    l2.append(0)
# Reading the training .csv file
df = pd.read_csv("training.csv")
DF = pd.read_csv("training.csv", index_col="prognosis")
# Replace the values in the imported file by pandas by the inbuilt function replace in pandas.

df.replace(
    {
        "prognosis": {
            "Fungal infection": 0,
            "Allergy": 1,
            "GERD": 2,
            "Chronic cholestasis": 3,
            "Drug Reaction": 4,
            "Peptic ulcer diseae": 5,
            "AIDS": 6,
            "Diabetes ": 7,
            "Gastroenteritis": 8,
            "Bronchial Asthma": 9,
            "Hypertension ": 10,
            "Migraine": 11,
            "Cervical spondylosis": 12,
            "Paralysis (brain hemorrhage)": 13,
            "Jaundice": 14,
            "Malaria": 15,
            "Chicken pox": 16,
            "Dengue": 17,
            "Typhoid": 18,
            "hepatitis A": 19,
            "Hepatitis B": 20,
            "Hepatitis C": 21,
            "Hepatitis D": 22,
            "Hepatitis E": 23,
            "Alcoholic hepatitis": 24,
            "Tuberculosis": 25,
            "Common Cold": 26,
            "Pneumonia": 27,
            "Dimorphic hemmorhoids(piles)": 28,
            "Heart attack": 29,
            "Varicose veins": 30,
            "Hypothyroidism": 31,
            "Hyperthyroidism": 32,
            "Hypoglycemia": 33,
            "Osteoarthristis": 34,
            "Arthritis": 35,
            "(vertigo) Paroymsal  Positional Vertigo": 36,
            "Acne": 37,
            "Urinary tract infection": 38,
            "Psoriasis": 39,
            "Impetigo": 40,
        }
    },
    inplace=True,
)
DF.head()

X = df[l1]
y = df[["prognosis"]]
np.ravel(y)
# Reading the  testing.csv file
tr = pd.read_csv("testing.csv")

# Using inbuilt function replace in pandas for replacing the values

tr.replace(
    {
        "prognosis": {
            "Fungal infection": 0,
            "Allergy": 1,
            "GERD": 2,
            "Chronic cholestasis": 3,
            "Drug Reaction": 4,
            "Peptic ulcer diseae": 5,
            "AIDS": 6,
            "Diabetes ": 7,
            "Gastroenteritis": 8,
            "Bronchial Asthma": 9,
            "Hypertension ": 10,
            "Migraine": 11,
            "Cervical spondylosis": 12,
            "Paralysis (brain hemorrhage)": 13,
            "Jaundice": 14,
            "Malaria": 15,
            "Chicken pox": 16,
            "Dengue": 17,
            "Typhoid": 18,
            "hepatitis A": 19,
            "Hepatitis B": 20,
            "Hepatitis C": 21,
            "Hepatitis D": 22,
            "Hepatitis E": 23,
            "Alcoholic hepatitis": 24,
            "Tuberculosis": 25,
            "Common Cold": 26,
            "Pneumonia": 27,
            "Dimorphic hemmorhoids(piles)": 28,
            "Heart attack": 29,
            "Varicose veins": 30,
            "Hypothyroidism": 31,
            "Hyperthyroidism": 32,
            "Hypoglycemia": 33,
            "Osteoarthristis": 34,
            "Arthritis": 35,
            "(vertigo) Paroymsal  Positional Vertigo": 36,
            "Acne": 37,
            "Urinary tract infection": 38,
            "Psoriasis": 39,
            "Impetigo": 40,
        }
    },
    inplace=True,
)
tr.head()
X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)


# Decision Tree Algorithm


def DecisionTree(nameEn, S1, S2, S3, S4, S5):
    print(nameEn, S1, S2, S3, S4, S5)
    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()
    clf3 = clf3.fit(X, y)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    y_pred = clf3.predict(X_test)
    print("Decision Tree")
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))
    print("Confusion matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    psymptoms = [S1, S2, S3, S4, S5]

    for k in range(0, len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted = predict[0]

    h = "no"
    illness = "Not found"
    for a in range(0, len(disease)):
        if predicted == a:
            h = "yes"
            break

    if h == "yes":
        
        illness = disease[a]
    return illness


# Random Forest Algorithm


def randomforest(nameEn, S1, S2, S3, S4, S5):
    from sklearn.ensemble import RandomForestClassifier

    clf4 = RandomForestClassifier(n_estimators=100)
    clf4 = clf4.fit(X, np.ravel(y))

    # calculating accuracy
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    y_pred = clf4.predict(X_test)
    print("Random Forest")
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))
    print("Confusion matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    psymptoms = [S1, S2, S3, S4, S5]

    for k in range(0, len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted = predict[0]

    h = "no"
    illness = "Not found"
    for a in range(0, len(disease)):
        if predicted == a:
            h = "yes"
            break

    if h == "yes":
        
        illness = disease[a]
    return illness


# KNearestNeighbour Algorithm


def KNN(nameEn, S1, S2, S3, S4, S5):
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    knn = knn.fit(X, np.ravel(y))

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    y_pred = knn.predict(X_test)
    print("kNearest Neighbour")
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))
    print("Confusion matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    psymptoms = [S1, S2, S3, S4, S5]

    for k in range(0, len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = knn.predict(inputtest)
    predicted = predict[0]

    h = "no"
    illness = "Not found"
    for a in range(0, len(disease)):
        if predicted == a:
            h = "yes"
            break

    if h == "yes":
        
        illness = disease[a]
    return illness


# Naive Bayes Algorithm


def NaiveBayes(nameEn, S1, S2, S3, S4, S5):
    from sklearn.naive_bayes import GaussianNB

    gnb = GaussianNB()
    gnb = gnb.fit(X, np.ravel(y))

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    y_pred = gnb.predict(X_test)
    print("Naive Bayes")
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))
    print("Confusion matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    psymptoms = [S1, S2, S3, S4, S5]
    for k in range(0, len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted = predict[0]

    h = "no"
    illness = "Not found"
    for a in range(0, len(disease)):
        if predicted == a:
            h = "yes"
            break

    if h == "yes":
        
        illness = disease[a]
    return illness


# calling this function because the application is ready to run
@app.route("/receive_data", methods=["POST"])
def receive_data():
    data_received = request.get_json()
    nameEn = data_received.get("name")
    S1 = data_received.get("sym1")
    S2 = data_received.get("sym2")
    S3 = data_received.get("sym3")
    S4 = data_received.get("sym4")
    S5 = data_received.get("sym5")
    illness = []
    illness.append(DecisionTree(nameEn, S1, S2, S3, S4, S5))
    illness.append(randomforest(nameEn, S1, S2, S3, S4, S5))
    illness.append(KNN(nameEn, S1, S2, S3, S4, S5))
    illness.append(NaiveBayes(nameEn, S1, S2, S3, S4, S5))
    response_data = {"diseases": illness}
    return jsonify(response_data)


@app.route("/")
def test():
    return jsonify({"msg": "running"})


if __name__ == "__main__":
    app.run(port=5000)