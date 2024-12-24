import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import gradio as gr

action_map = {
    1: "Hand at rest",
    2: "Hand clenched in a fist",
    3: "Wrist flexion",
    4: "Wrist extension",
    5: "Radial deviations",
    6: "Ulnar deviations",
}

def action(e1, e2, e3, e4, e5, e6, e7, e8):
  model = load_model('model6839.keras')
  input_data = np.array([[e1, e2, e3, e4, e5, e6, e7, e8]])
  prediction = model.predict(input_data)
  predicted_class = np.argmax(prediction, axis=-1)
  return action_map.get(predicted_class[0]+1, "Unknown action")

inputs = [
    gr.Number(label="e1"),
    gr.Number(label="e2"),
    gr.Number(label="e3"),
    gr.Number(label="e4"),
    gr.Number(label="e5"),
    gr.Number(label="e6"),
    gr.Number(label="e7"),
    gr.Number(label="e8"),
]

output = gr.Textbox(label="Prediction")

examples = [
    [-2.00e-05, 1.00e-05, 2.20e-04, 1.80e-04, -1.50e-04, -5.00e-05, 1.00e-05, 0],
    [1.60e-04, -1.00e-04, -2.40e-04, 2.00e-04, 1.00e-04, -9.00e-05, -5.00e-05, -5.00e-05],
    [-1.00e-05, 1.00e-05, 1.00e-05, 0, -2.00e-05, 0, -3.00e-05, -3.00e-05],
]

def func(e1, e2, e3, e4, e5, e6, e7, e8):
  return action(e1, e2, e3, e4, e5, e6, e7, e8)

iface = gr.Interface(
    fn=func,
    inputs=inputs,
    outputs=output,
    title="ML Model Predictor",
    examples=examples,
    flagging_options=["Working", "Not Wotking"],
    description="Enter the 8 feature values to get a prediction."
)

#iface.launch(share=True, auth=('emg','emg123'), auth_message="Type in your <strong>login credentials</strong>")
iface.launch(share=True)