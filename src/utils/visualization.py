import plotly.graph_objects as go
import pandas as pd
import wandb

def plot_al_curve(results, title="Active Learning Curve"):
    """
    Plot Accuracy vs Number of Queries using Plotly.
    results: dict {method_name: [acc_round_0, acc_round_1, ...]}
    """
    fig = go.Figure()
    
    for method, accuracies in results.items():
        x_axis = [i * 10 + 20 for i in range(len(accuracies))] # Assuming 10 queries/round, start 20
        fig.add_trace(go.Scatter(
            x=x_axis, 
            y=accuracies,
            mode='lines+markers',
            name=method
        ))
        
    fig.update_layout(
        title=title,
        xaxis_title="Number of Labeled Images",
        yaxis_title="Test Accuracy",
        hovermode="x unified"
    )
    
    # Save as HTML
    fig.write_html("results/al_curve.html")
    print("Saved interactive plot to results/al_curve.html")
    
    # Log to WANDB if active
    if wandb.run is not None:
        wandb.log({"al_curve": fig})
