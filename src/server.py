from flask import Flask, render_template, request
import os
from pdf_to_markdown import pdf_to_markdown
from query_with_ollama import ask_ollama, query_neo4j
import json

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/", methods=["GET"])
def main():
    return render_template("index.html", output="")


@app.route("/upload", methods=["POST"])
def upload_file():
    uploaded_file = request.files.get("file")
    if uploaded_file:
        if not uploaded_file.filename.lower().endswith(".pdf"):
            output = "Error: Only PDF files are allowed."
            return render_template("index.html", output=output)
        content = uploaded_file.read()
        output = f"Uploaded file: {uploaded_file.filename}, size: {len(content)} bytes"
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
        pdf_to_markdown(file_path)
        output += f"<br>Converted {uploaded_file.filename} to markdown."

    else:
        output = "No file uploaded."
    return render_template("index.html", output=output)


def get_documents():
    with open("knowledge_graph.json", "r") as f:
        data = json.load(f)
    return [
        node
        for node in data["nodes"]
        if node.get("path", "").startswith("documents_processed/")
    ]


def getStatsForDocument(doc_id):
    with open("llm_integration_export.json", "r") as f:
        data = json.load(f)
    doc = data["documents"].get(doc_id)

    if not doc:
        # Try without "doc_" prefix
        doc_id_without_prefix = doc_id.replace("doc_", "")
        doc = data["documents"].get(doc_id_without_prefix)
        if not doc:
            return None

    # Gather evidence
    evidence = []
    for ev in doc.get("evidence", []):
        evidence.append(
            {
                "content": ev.get("content", ""),
                "categories": ev.get("categories", []),
                "confidence": ev.get("confidence", 0),
                "chunk_type": ev.get("chunk_type", ""),
            }
        )

    # Calculate strengths, gaps, recommendations (simple heuristics for now)
    strengths = []
    gaps = []
    recommendations = []

    # Example: strengths if evidence exists for a category
    categories = set()
    for ev in evidence:
        categories.update(ev["categories"])
    if "risk_management" in categories:
        strengths.append("Strong evidence in risk management procedures")
    if "governance" in categories:
        strengths.append("Governance structure evidence present")
    if "financial_management" in categories:
        strengths.append("Financial management evidence present")
    if "logistics_support" in categories:
        strengths.append("Logistics support evidence present")

    # Example: gaps if missing expected categories
    expected = {
        "risk_management",
        "governance",
        "financial_management",
        "logistics_support",
    }
    missing = expected - categories
    for cat in missing:
        gaps.append(f"Missing {cat.replace('_', ' ')} evidence")

    # Example: recommendations based on gaps
    for gap in gaps:
        recommendations.append(f"Add comprehensive {gap[8:]} section")

    # Graph position (dummy for now)
    graph_position = {
        "centrality": 0.5,
        "connections": 1,
        "role": "Document",
    }

    # Calculate average confidence
    avg_conf = (
        sum(ev["confidence"] for ev in evidence) / len(evidence) if evidence else 0
    )

    return {
        "label": doc.get("label", doc_id),
        "evidence": evidence,
        "strengths": strengths,
        "gaps": gaps,
        "recommendations": recommendations,
        "graph_position": graph_position,
        "avg_confidence": avg_conf,
        "categories_covered": len(categories),
        "evidence_count": len(evidence),
    }


@app.route("/ask", methods=["GET"])
def serveAsk():
    return render_template("ask.html", documents=get_documents())


def ask_question(doc_id, question):
    doc_type = "Document"
    cypher = (
        f"MATCH (n:Node {{elementId: '{doc_id}', type: '{doc_type}'}})-[r*1..3]-(m) "
        "RETURN n, r, m LIMIT 7500"
    )
    neo4j_uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "hack25gg"

    records = query_neo4j(cypher, neo4j_uri, user, password)
    context = "\n".join(str(r) for r in records)

    prompt = f"{question}\n\nSearch the Neo4j database for relevant information to support your assessment."
    print(f"prompt: {prompt}    ")
    output = ask_ollama(prompt, context)
    return output


@app.route("/ask", methods=["POST"])
def ask():
    doc_id = request.form.get("doc_id", "")
    question = request.form.get("question", "")
    print(f"Processing document ID: {doc_id}, question: {question}")
    if not doc_id:
        return render_template("index.html", output="Error: No document ID provided")

    output = ask_question(doc_id, question)

    return render_template(
        "ask.html",
        output=output,
        doc_id=doc_id,
        documents=get_documents(),
        question=question,
    )


@app.route("/dashboards", methods=["GET"])
def dashboardGet():
    documents = get_documents()
    stats_map = {doc["id"]: getStatsForDocument(doc["id"]) for doc in documents}
    return render_template(
        "dashboard.html", documents=get_documents(), stats_map=stats_map
    )


@app.route("/dashboard/<id>", methods=["GET"])
def dashboardGetId(id):
    print(f"Dashboard request for document ID: {id}")
    documentForId = [doc for doc in get_documents() if doc["id"] == id][0]
    print(f"Found document for ID {id}: {documentForId}")
    return render_template(
        "dashboard_doc_id.html",
        doc_id=id,
        document=documentForId,
        stats=getStatsForDocument(id),
    )


@app.route("/dashboard/<id>", methods=["POST"])
def dashboardPostId(id):
    print(f"Dashboard request for document ID: {id}")
    documentForId = [doc for doc in get_documents() if doc["id"] == id][0]
    print(f"Found document for ID {id}: {documentForId}")

    question = request.form.get("question", "")

    output = ask_question(id, question)

    return render_template(
        "dashboard_doc_id.html",
        doc_id=id,
        document=documentForId,
        stats=getStatsForDocument(id),
        output=output,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
