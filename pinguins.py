import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


df = pd.read_csv("penguins.csv")
df = df.replace(".", float("nan"))
df = df.dropna()


df["island"] = df["island"].replace({"Biscoe": 0, "Dream": 1, "Torgersen": 2})
df["sex"]    = df["sex"].replace({"FEMALE": 0, "MALE": 1})
df["species"]= df["species"].replace({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})


df = df.reindex(columns=["island", "sex", "culmen_length_mm",
                          "culmen_depth_mm", "flipper_length_mm",
                          "body_mass_g", "species"])


X = df.drop(columns=["species"])
y = df["species"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)


y_pred = modelo.predict(X_test)

nomes = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
print("\n=== Métricas de Avaliação ===")
print(classification_report(
    y_test, y_pred,
    target_names=["Adelie", "Chinstrap", "Gentoo"]
))


print("\n=== Classificar um pinguim novo ===")
print("Ilha   → 0=Biscoe  1=Dream  2=Torgersen")
print("Sexo   → 0=Fêmea   1=Macho")

island          = int(input("Ilha (0/1/2): "))
sex             = int(input("Sexo (0/1): "))
culmen_length   = float(input("Comprimento do cúlmen (mm): "))
culmen_depth    = float(input("Profundidade do cúlmen (mm): "))
flipper_length  = float(input("Comprimento da nadadeira (mm): "))
body_mass       = float(input("Massa corporal (g): "))

entrada = pd.DataFrame([[island, sex, culmen_length, culmen_depth, flipper_length, body_mass]],
                        columns=X.columns)
resultado = modelo.predict(entrada)[0]
print(f"\nEspécie prevista: {nomes[resultado]}")
