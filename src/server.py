from flask import Flask, render_template, request
import os
from pdf_to_markdown import pdf_to_markdown

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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
