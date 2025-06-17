import pymupdf4llm
import re
import os
from pathlib import Path
import pandas as pd
import pyneoinstance
import json
from neo4j import GraphDatabase

# Python file - connection to Neo4j

# Retrieving the Neo4j connection credentials from the config.yaml file
configs = pyneoinstance.load_yaml_file("config.yaml")
creds = configs["credentials"]

# Establishing the Neo4j connection
neo4j = pyneoinstance.Neo4jInstance(creds["uri"], creds["user"], creds["password"])


def import_knowledge_graph(json_path, neo4j_uri, user, password):
    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)
    nodes = data["nodes"]
    edges = data["edges"]

    # Connect to Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))
    with driver.session() as session:
        # Create nodes
        node_ids = {}
        for node in enumerate(nodes):
            node_name = node.get("name", node["id"])
            result = session.run(
                "CREATE (n:Node {name: $name, elementId: $elementId, type: $type}) RETURN id(n)",
                name=node_name,
                elementId=node["id"],
                type=node.get("type"),
            )
            node_id = result.single()[0]
            node_ids[node["id"]] = node_id

        # Create edges
        for edge in enumerate(edges):
            # Find source and target node indices
            # Assume edges are between consecutive nodes (i to i+1)
            src_id = node_ids.get(edge["source"])
            tgt_id = node_ids.get(edge["target"])
            if src_id is not None and tgt_id is not None:
                session.run(
                    """
                    MATCH (a),(b)
                    WHERE id(a) = $src_id AND id(b) = $tgt_id
                    CREATE (a)-[r:%s]->(b)
                    """
                    % edge["relationship"].upper().replace(" ", "_"),
                    src_id=src_id,
                    tgt_id=tgt_id,
                )
    driver.close()


# Example usage:
import_knowledge_graph(
    "/home/dirk/code/hack25/knowledge_graph.json",
    "bolt://localhost:7687",
    "neo4j",
    "hack25gg",
)
