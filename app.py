import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import BanglaFakeNewsPredictor
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Initialize predictor
try:
    predictor = BanglaFakeNewsPredictor()
    print("Predictor loaded successfully")
except:
    predictor = None
    print("Predictor not loaded")

# History storage
history_data = []

def analyze_news(text):
    if not text:
        return "Please write some news", None
    
    if predictor is None:
        return "Model not loaded", None
    
    try:
        result = predictor.get_prediction_with_confidence(text)
        
        if result['label'] == 1:
            result_text = "FAKE NEWS (Confidence: " + str(round(result['confidence']*100, 2)) + "%)"
        else:
            result_text = "REAL NEWS (Confidence: " + str(round(result['confidence']*100, 2)) + "%)"
        
        # Create chart
        fig = go.Figure(data=[go.Pie(
            labels=['Real', 'Fake'],
            values=[result['probabilities']['real'], result['probabilities']['fake']],
            marker_colors=['green', 'red']
        )])
        
        fig.update_layout(title="Probability", height=400)
        
        # Add to history
        history_data.append({
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'News': text[:30] + '...' if len(text) > 30 else text,
            'Result': 'Fake' if result['label'] == 1 else 'Real',
            'Confidence': str(round(result['confidence']*100, 2)) + '%'
        })
        
        return result_text, fig
        
    except Exception as e:
        return "Error: " + str(e), None

def batch_analyze(file):
    if file is None:
        return "Upload a file", None
    
    try:
        df = pd.read_csv(file.name)
        text_col = df.columns[0]
        
        results = []
        for text in df[text_col]:
            try:
                pred = predictor.predict(text)
                results.append('Fake' if pred == 1 else 'Real')
            except:
                results.append('Error')
        
        df['Result'] = results
        
        fake = results.count('Fake')
        real = results.count('Real')
        total = len(results)
        
        summary = "Total: " + str(total) + ", Real: " + str(real) + ", Fake: " + str(fake)
        
        return summary, df
    except Exception as e:
        return "Error: " + str(e), None

def get_history():
    if history_data:
        return pd.DataFrame(history_data)
    return pd.DataFrame(columns=["Time", "News", "Result", "Confidence"])

def clear_history():
    history_data.clear()
    return pd.DataFrame(columns=["Time", "News", "Result", "Confidence"])

# Create interface
with gr.Blocks(title="Bangla Fake News Detector") as demo:
    gr.Markdown("#  Bangla Fake News Detector")
    gr.Markdown("---")
    
    with gr.Tab("Single News"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(lines=3, label="Enter Bangla News")
                analyze_btn = gr.Button("Analyze", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(label="Result")
                output_plot = gr.Plot(label="Chart")
        
        analyze_btn.click(
            fn=analyze_news,
            inputs=text_input,
            outputs=[output_text, output_plot]
        )
    
    with gr.Tab("Batch Analysis"):
        file_input = gr.File(label="Upload CSV")
        batch_btn = gr.Button("Analyze", variant="primary")
        batch_output = gr.Textbox(label="Summary")
        batch_df = gr.Dataframe(label="Results")
        
        batch_btn.click(
            fn=batch_analyze,
            inputs=file_input,
            outputs=[batch_output, batch_df]
        )
    
    with gr.Tab("History"):
        refresh_btn = gr.Button("Refresh")
        clear_btn = gr.Button("Clear")
        history_df = gr.Dataframe(
            headers=["Time", "News", "Result", "Confidence"],
            label="History"
        )
        
        refresh_btn.click(fn=get_history, inputs=[], outputs=history_df)
        clear_btn.click(fn=clear_history, inputs=[], outputs=history_df)
    
    with gr.Tab("Help"):
        gr.Markdown("""
        ## How to Use
        - **Single News**: Type news and click Analyze
        - **Batch**: Upload CSV file with news in first column
        - **Model**: BanglaBERT
        """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="127.0.0.1", server_port=7861)