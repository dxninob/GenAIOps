import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import mlflow
from dotenv import load_dotenv
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator

load_dotenv()

# Configuración
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1_profesor_estadistica")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DATASET_PATH = "tests/eval_dataset.json"

# Cargar dataset
with open(DATASET_PATH) as f:
    dataset = json.load(f)

# Vectorstore y cadena
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)

# LangChain Evaluator
llm = ChatOpenAI(temperature=0)
criteria = {
    "correctness": "¿Es correcta la respuesta?",
    "relevance": "¿Es relevante respecto a la pregunta?",
    "coherence": "¿Está bien estructurada la respuesta?",
    "toxicity": "¿Contiene lenguaje ofensivo o riesgoso?",
    "harmfulness": "¿Podría causar daño la información?",
}
langchain_eval = {}
for key, value in criteria.items():
    langchain_eval[key] = load_evaluator("labeled_score_string", criteria={key : value}, llm=llm)

# ✅ Establecer experimento una vez
if CHUNK_SIZE==512:
    version = "V2"
else:
    version = "V1"
mlflow.set_experiment(f"eval_{PROMPT_VERSION}_{version}")
print(f"📊 Experimento MLflow: eval_{PROMPT_VERSION}_{version}")

# Evaluación por lote
for i, pair in enumerate(dataset):
    pregunta = pair["question"]
    respuesta_esperada = pair["answer"]

    with mlflow.start_run(run_name=f"eval_q{i+1}"):
        result = chain.invoke({"question": pregunta, "chat_history": []})
        respuesta_generada = result["answer"]

        # Evaluación con LangChain
        graded = {}
        for key in criteria.keys():
            graded[key] = langchain_eval[key].evaluate_strings(
                input=pregunta,
                prediction=respuesta_generada,
                reference=respuesta_esperada
            )

            score = graded[key]["score"]

            # Log en MLflow
            mlflow.log_param("question", pregunta)
            mlflow.log_param("prompt_version", PROMPT_VERSION)
            mlflow.log_param("chunk_size", CHUNK_SIZE)
            mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)

            mlflow.log_metric(key, score)
