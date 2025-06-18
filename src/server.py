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
    return [node for node in data["nodes"] if node["type"] == "Document"]


@app.route("/ask", methods=["GET"])
def serveAsk():
    return render_template("ask.html", documents=get_documents())


@app.route("/ask", methods=["POST"])
def ask():
    doc_id = request.form.get("doc_id", "")
    print(f"Processing document ID: {doc_id}")
    if not doc_id:
        return render_template("index.html", output="Error: No document ID provided")
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

    prompt = """You are a schedule assurance specialist reviewing this Project Management Plan (PMP) for compliance with DCMA-14 scheduling principles and GovS 002 project governance. Based on this, please assess the following:  
    Does the PMP attached describe how the project schedule is built, including logic links between tasks and milestone relationships?  
    Are baseline schedules or performance tracking methods defined (e.g., earned value, critical path analysis)? 
    Is there evidence of governance-level monitoring of schedule risk, including escalation or recovery plans? 
    Provide evidence from the PMP and offer suggestions to improve compliance with DCMA-14 and GovS 002 expectations. 
    Search the Neo4j database for relevant information to support your assessment."""
    output = ask_ollama(prompt, context)

    return render_template(
        "ask.html", output=output, doc_id=doc_id, documents=get_documents()
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
