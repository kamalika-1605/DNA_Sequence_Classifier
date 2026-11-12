import os
import gradio as gr
import pandas as pd
from datetime import datetime
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

# Train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

pred = model.predict(X)
accuracy = accuracy_score(y, pred) * 100

# Prediction function
def predict_dna(sequence):
    sequence = sequence.strip().upper()

    if not sequence:
        return "Please enter a DNA sequence.", None

    if any(base not in "ATGC" for base in sequence):
        return "Invalid sequence! Use only A, T, G and C.", None

    features = extract_features(sequence)
    result = model.predict([features])[0]

    output = f"""
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

    prediction_data = {
        "DNA Sequence": sequence,
        "Predicted Group": result
    }

    return output, prediction_data

# Flag function
def flag_prediction(prediction_data):
    if prediction_data is None:
        return "Please submit a valid DNA sequence before flagging."

    record = {
        "DNA Sequence": prediction_data["DNA Sequence"],
        "Predicted Group": prediction_data["Predicted Group"],
        "Flagged At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    file_name = "flagged_sequences.csv"

    pd.DataFrame([record]).to_csv(
        file_name,
        mode="a",
        header=not os.path.exists(file_name),
        index=False
    )

    return "Thank you! This prediction has been flagged for review."

# Website interface
with gr.Blocks() as app:

    gr.Markdown("# DNA Sequence Classification System")
    gr.Markdown(
        "AI/ML Based DNA Group Prediction using Decision Tree"
    )

    sequence_input = gr.Textbox(
        label="Enter DNA Sequence",
        placeholder="Example: ATGCGCGATC"
    )

    predict_button = gr.Button("Submit", variant="primary")

    prediction_output = gr.Textbox(
        label="Prediction Result",
        lines=12
    )

    prediction_state = gr.State(None)

    flag_button = gr.Button("Flag")

    flag_message = gr.Textbox(
        label="Feedback Status",
        interactive=False
    )

    predict_button.click(
        fn=predict_dna,
        inputs=sequence_input,
        outputs=[prediction_output, prediction_state]
    )

    flag_button.click(
        fn=flag_prediction,
        inputs=prediction_state,
        outputs=flag_message
    )

    gr.Examples(
        examples=[
            ["ATGCGCGATC"],
            ["ATGCGATC"],
            ["GCGCGCGC"]
        ],
        inputs=sequence_input
    )

# Start website on Render
port = int(os.environ.get("PORT", 10000))

app.launch(
    server_name="0.0.0.0",
    server_port=port
)
