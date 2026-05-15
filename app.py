import os
import json
import traceback
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model    = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoders.joblib"))

with open(os.path.join(BASE_DIR, "model_meta.json"), encoding="utf-8") as f:
    meta = json.load(f)

FEATURE_COLS = meta["feature_cols"]
CAT_COLS     = meta["cat_cols"]

# Mapeia valores novos do frontend para o que o encoder conhece
VALOR_MAP = {
    "Sem Escolaridade":       "Fundamental",
    "Fundamental Incompleto": "Fundamental",
    "Fundamental Completo":   "Fundamental",
    "Médio Incompleto":       "Médio",
    "Médio Completo":         "Médio",
    "Superior Completo":      "Superior Incompleto",
    "Pós-graduação":          "Superior Incompleto",
    "Rio de Janeiro":  "São Paulo",
    "Belo Horizonte":  "São Paulo",
    "Vitória":         "Santos",
    "Curitiba":        "São Paulo",
    "Porto Alegre":    "São Paulo",
    "Florianópolis":   "Santos",
    "Salvador":        "São Paulo",
    "Fortaleza":       "São Paulo",
    "Recife":          "São Paulo",
    "Natal":           "São Paulo",
    "Maceió":          "São Paulo",
    "Brasília":        "São Paulo",
    "Goiânia":         "São Paulo",
    "Campo Grande":    "São Paulo",
    "Cuiabá":          "São Paulo",
    "Manaus":          "São Paulo",
    "Belém":           "São Paulo",
    "Porto Velho":     "São Paulo",
    "Implante":    "Prótese",
    "Clareamento": "Limpeza",
    "Cirurgia":    "Extração",
    "Periodontia": "Limpeza",
    "Endodontia":  "Canal",
}

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "mensagem": "API Turma do Bem Online",
        "endpoints": {"health": "/health", "predict": "/predict", "teste": "/teste"}
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": meta["model"],
        "feature_cols": FEATURE_COLS,
        "metrics": meta["metrics"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Body deve ser um JSON válido."}), 400

        missing = [c for c in FEATURE_COLS if c not in data]
        if missing:
            return jsonify({"error": "Campos ausentes.", "campos_faltando": missing}), 422

        entrada = {k: data[k] for k in FEATURE_COLS}

        campos_nulos = [
            col for col in FEATURE_COLS
            if entrada[col] is None
            or entrada[col] == ""
            or (isinstance(entrada[col], float) and pd.isna(entrada[col]))
        ]
        if campos_nulos:
            return jsonify({"error": "Preencha todos os campos.", "campos_nulos": campos_nulos}), 422

        for col in CAT_COLS:
            encoder = encoders[col]
            valor   = entrada[col]
            if valor not in encoder.classes_:
                valor = VALOR_MAP.get(valor, None)
            entrada[col] = int(encoder.transform([valor])[0]) if valor in encoder.classes_ else 0

        X_novo = pd.DataFrame([entrada])[FEATURE_COLS]
        pred   = int(model.predict(X_novo)[0])

        try:
            prob = round(float(model.predict_proba(X_novo)[0][1]), 4)
        except Exception:
            prob = None

        return jsonify({
            "completara_tratamento": bool(pred),
            "classe":                "Sim" if pred == 1 else "Não",
            "probabilidade":         prob,
            "modelo":                meta["model"]
        })

    except Exception as e:
        return jsonify({"erro_real": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/teste", methods=["GET"])
def teste():
    try:
        exemplo = {
            "genero": "Masculino", "idade": 18, "escolaridade": "Médio",
            "renda_familiar_sm": 2, "cidade": "São Paulo", "ano_atendimento": 2024,
            "mes_atendimento": 5, "procedimento": "Limpeza", "numero_sessoes": 4,
            "dificuldade_acesso": "Baixa", "nota_satisfacao": 9
        }
        entrada = exemplo.copy()
        for col in CAT_COLS:
            encoder = encoders[col]
            valor   = entrada[col]
            entrada[col] = int(encoder.transform([valor])[0]) if valor in encoder.classes_ else 0

        X_novo = pd.DataFrame([entrada])[FEATURE_COLS]
        pred   = int(model.predict(X_novo)[0])

        try:
            prob = round(float(model.predict_proba(X_novo)[0][1]), 4)
        except Exception:
            prob = None

        return jsonify({"funcionando": True, "classe": "Sim" if pred == 1 else "Não",
                        "probabilidade": prob, "modelo": meta["model"]})
    except Exception as e:
        return jsonify({"erro_real": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    print(f"API rodando em https://nuvemtechapi-ia.onrender.com")