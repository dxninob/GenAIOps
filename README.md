# 🤖 Chatbot GenAI - Profesor de Estadística

Este proyecto demuestra cómo construir, evaluar y automatizar un chatbot de tipo RAG (Retrieval Augmented Generation) con buenas prácticas de **GenAIOps**.

---

## 🧠 Caso de Estudio

El chatbot responde preguntas sobre estadistíca básica y probabilidad, está enfocado especificamente para resolver dudas de esta materia a los estudientes de primeros semestres de pregrado. El chatbot usa como base una colección de documentos PDF de libros de estadística universitaria básica.

---

## 📂 Estructura del Proyecto

```
├── app/
│   ├── ui_streamlit.py           ← interfaz simple del chatbot
│   ├── run_eval.py               ← evaluación automática
│   ├── rag_pipeline.py           ← lógica de ingestión y RAG
│   ├── dashboard.py              ← interfaz de metricas evaluadas
│   └── prompts/
│       ├── v1_profesor_estadistica.txt
│       ├── v2_resumido_directo.txt
│       └── v3_profesor_primaria.txt
├── data/pdfs/                    ← documentos fuente
├── tests/
│   ├── test_run_eval.py
│   ├── eval_dataset.json         ← dataset de evaluación
│   └── eval_dataset.csv
├── .env.example
├── Dockerfile
├── .devcontainer/
│   └── devcontainer.json
├── .github/workflows/
│   ├── eval.yml
│   └── test.yml
```

---

## 🚦 Ciclo de vida GenAIOps aplicado

### 1. 🧱 Preparación del entorno

```bash
git clone https://github.com/darkanita/GenAIOps_Pycon2025 chatbot-genaiops
cd chatbot-genaiops
conda create -n chatbot-genaiops python=3.10 -y
conda activate chatbot-genaiops
pip install -r requirements.txt
cp .env.example .env  # Agrega tu API KEY de OpenAI
```

---

### 2. 🔍 Ingesta y vectorización de documentos

Procesa los PDFs y genera el índice vectorial:

```bash
python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"
```

Esto:
- Divide los documentos en chunks (por defecto `chunk_size=512`, `chunk_overlap=50`)
- Genera embeddings con OpenAI
- Guarda el índice vectorial en `vectorstore/`
- Registra los parámetros en **MLflow**

🔧 Para personalizar:
```python
save_vectorstore(chunk_size=1024, chunk_overlap=100)
```

♻️ Para reutilizarlo directamente:
```python
vectordb = load_vectorstore_from_disk()
```

---

### 3. 🧠 Construcción del pipeline RAG

```python
from app.rag_pipeline import build_chain
chain = build_chain(vectordb, prompt_version="v1_profesor_estadistica")
```

- Soporta múltiples versiones de prompt
- Usa `ConversationalRetrievalChain` con `LangChain` + `OpenAI`

---

### 4. 💬 Interacción vía Streamlit

Versión básica:
```bash
streamlit run app/ui_streamlit.py
```

---

### 5. 🧪 Evaluación automática de calidad

Ejecuta:

```bash
python app/run_eval.py
```

Esto:
- Usa `tests/eval_dataset.json` como ground truth
- Genera respuestas usando el RAG actual
- Evalúa con `LangChain Evaluation (load_evaluator)`
- Registra resultados en **MLflow**

---

### 6. 📈 Visualización de resultados

Dashboard completo:

```bash
streamlit run app/dashboard.py
```

- Tabla con todas las preguntas evaluadas
- Gráficos de precisión por configuración (`prompt + chunk_size`)
- Filtrado por experimento MLflow

---

### 7. 🔁 Automatización con GitHub Actions

- CI de evaluación: `.github/workflows/eval.yml`
- Test unitarios: `.github/workflows/test.yml`

---

### 8. 🧪 Validación automatizada

```bash
pytest tests/test_run_eval.py
```

- Evalúa que el sistema tenga al menos 80% de precisión con el dataset base

---

## 🔍 ¿Qué puedes hacer?

- 💬 Hacer preguntas al chatbot
- 🔁 Evaluar diferentes estrategias de chunking y prompts
- 📊 Comparar desempeño con métricas semánticas
- 🧪 Trazar todo en MLflow
- 🔄 Adaptar a otros dominios (legal, salud, educación…)

---

## ⚙️ Stack Tecnológico

- **OpenAI + LangChain** – LLM + RAG
- **FAISS** – Vectorstore
- **Streamlit** – UI
- **MLflow** – Registro de experimentos
- **LangChain Eval** – Evaluación semántica
- **GitHub Actions** – CI/CD
- **DevContainer** – Desarrollo portable

---

## 🎓 Desafío para estudiantes

🧩 **Parte 1: Personalización**

**1. Selección de un nuevo dominio**  
Se seleccionó el dominio de estadística básica y probabilidad.

**2. Reemplazo de los documentos PDF**  
Se adjuntó en la ruta data/pdfs/ un total de cinco libros relacionados con el tema.

**3. Creación de prompts**  
Se adjuntó el la ruta app/prompts/ un total de tres prompts para ser evaluados.
- Profesor universitario (v1_profesor_estadistica): Responde como un profesor universitario en estadística y probabilidad que da esta clase a estudiantes de primeros semestres. Responde únicamente usando los libros en PDF y si no sabe la respuesta admite que no tiene suficiente información. 
- Resumido directo (v2_resumido_directo): Responde de forma breve y directa, usando únicamente los libros en PDF y si no sabe la respuesta admite que no tiene suficiente información. 
- Profesor de primaria (v3_profesor_primaria): Responde como un profesor de primaria de estadistica y probabilidad que da clase a niños entre 6 y 10 años. Responde de forma sencilla y sin terminos tecnicos. No responde necesariamente usando los libros en PDF pero los puede usar para complementar su respuesta. No admite que no tiene suficiente información. 

**4. Creación de un conjunto de pruebas**  
En tests/eval_dataset.json, se definieron 21 preguntas junto con su respuesta esperada para evaluar al chatbot.

🔧 **Parte 2: Reto**

**1. Mejoramiento del sistema de evaluación:**
- Se evaluó el conjunto de pruebas usando los siguientes criterios:
  * "correctness" – ¿Es correcta la respuesta?
  * "relevance" – ¿Es relevante respecto a la pregunta?
  * "coherence" – ¿Está bien estructurada la respuesta?
  * "toxicity" – ¿Contiene lenguaje ofensivo o riesgoso?
  * "harmfulness" – ¿Podría causar daño la información?

- Para cada criterio se registró una métrica en MLflow (score)

![Imagen](/static/Experimento_individual.png)  

📊 **Parte 3: Mejora del dashboard**

**1. Agregación de metricas para visualizar en dashboard.py:**  
Se agregó al dashboard:
- Las métricas por criterio (correctness_score, toxicity_score, etc.).  

![Imagen](/static/Desempeño_individual.png)

- Una opción para seleccionar y comparar diferentes criterios en gráficos (las imagenes mostradas en la siguiente sección).

🧪 **Parte 5: Presenta y reflexiona**
**1. Compara configuraciones distintas (chunk size, prompt) y justifica tu selección.**  
Se evaluaron los tres prompts usando dos vectorstores (CHUNK_SIZE=512,CHUNK_OVERLAP=50; CHUNK_SIZE=256,CHUNK_OVERLAP=30). Por lo tanto en total hubo seis configuraciones diferentes. En MLflow se registraron los seis experimentos de cada configuración:  
![Imagen](/static/Experimentos.png)  

- La configuración que genera mejores respuestas es la del prompt del profesor universitario (v1_profesor_estadistica) con chuncking más grande (CHUNK_SIZE=512 y CHUNK_OVERLAP=50). Esto tiene sentido ya que ese prompt es el que da un contexto más acorde para obtener las respuestas esperadas de las preguntas de testing, además porque al ser más grande el chunking el modelo puede tener más contexto para responder de mejor forma.

![Imagen](/static/Comparacion_criterios.png)  

- Comparación entre prompts: para ambas configuraciones de chuncking, el prompt que obtuvo mejores resultados fue el del profesor universitario, seguido del profesor de primaria, y el de peores resultados fue el prompt de resumido directo. El resumido directo pudo dar peores metricas dado que es el prompt menos especifico, era el que más libertad le daba al modelo por lo cual pudo haber dado peor información.

![Imagen](/static/Comparacion_promts.png)  

- Comparación entre chuncking: el tamaño del chuncking no generó diferencias muy significativas entre las respuestas para los prompts de profesor universitario y profesor de primaria, pero en el caso de resumido directo sí afectó mucho la respuesta para todos los criterios. En general, el chunking más grande fue el que logró dar mejores metricas dado que da más contexto al modelo. 

![Imagen](/static/Comparacion_chunking.png)
