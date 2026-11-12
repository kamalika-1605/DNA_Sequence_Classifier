import os
import gradio as gr
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# DNA Dataset
data = {
    "DNA": [
        "ATGCGATC",
        "ATGCGTAC",
        "GCGCGTAT",
        "GCGCGCGC"
    ],
    "Group": ["A", "A", "B", "B"]
}

df = pd.DataFrame(data)

# Feature extraction
def extract_features(seq):
    seq = seq.upper()

    A = seq.count("A")
    T = seq.count("T")
    G = seq.count("G")
    C = seq.count("C")
    length = len(seq)
    gc = ((G + C) / length) * 100

    return [A, T, G, C, length, gc]

# Prepare training data
X = pd.DataFrame(
    [extract_features(s) for s in df["DNA"]],
    columns=["A", "T", "G", "C", "Length", "GC"]
)

y = df["Group"]

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Training accuracy
pred = model.predict(X)
accuracy = accuracy_score(y, pred) * 100

# Prediction function
def predict_dna(sequence):
    sequence = sequence.strip().upper()

    if not sequence:
        return "Please enter a DNA sequence."

    if any(base not in "ATGC" for base in sequence):
        return "Invalid sequence! Use only A, T, G and C."

    features = extract_features(sequence)

    result = model.predict([features])[0]

    return f"""
DNA Sequence: {sequence}

A Count: {features[0]}
T Count: {features[1]}
G Count: {features[2]}
C Count: {features[3]}
Length: {features[4]}
GC Percentage: {features[5]:.2f}%

Predicted Group: {result}

Training Accuracy: {accuracy:.2f}%
"""

# Website interface
app = gr.Interface(
    fn=predict_dna,
    inputs=gr.Textbox(
        label="Enter DNA Sequence",
        placeholder="Example: ATGCGCGATC"
    ),
    outputs=gr.Textbox(label="Prediction Result"),
    title="DNA Sequence Classification System",
    description="AI/ML Based DNA Group Prediction using Decision Tree",
    examples=[
        "ATGCGCGATC",
        "ATGCGATC",
        "GCGCGCGC"
    ]
)

# Start website on Render
port = int(os.environ.get("PORT", 10000))
app.launch(server_name="0.0.0.0", server_port=port)
