import requests
from neo4j import GraphDatabase


def query_neo4j(cypher_query, neo4j_uri, user, password):
    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))
    with driver.session() as session:
        result = session.run(cypher_query)
        records = [record.data() for record in result]
    driver.close()
    return records


def ask_ollama(prompt, context, model="llama3"):
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": f"{prompt}\nContext:\n{context}", "stream": False}
    response = requests.post(url, json=data)
    return response.json()["response"]


if __name__ == "__main__":
    # Example Cypher query
    doc_id = "doc_20250605-Anonymised_PMP_2_FINAL"
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
    answer = ask_ollama(prompt, context)
    print(answer)
