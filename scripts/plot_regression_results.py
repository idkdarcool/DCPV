import numpy as np
import plotly.graph_objects as go
import os

def get_premium_layout(title, x_title, y_title):
    """Returns a premium Plotly layout dictionary."""
    return dict(
        title=dict(
            text=title,
            font=dict(family="Arial, sans-serif", size=24, color="#333333"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=18, family="Arial, sans-serif")),
            showgrid=True,
            gridcolor='#E5E5E5',
            zeroline=False,
            showline=True,
            linecolor='#333333',
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=18, family="Arial, sans-serif")),
            showgrid=True,
            gridcolor='#E5E5E5',
            zeroline=False,
            showline=True,
            linecolor='#333333',
            tickfont=dict(size=14)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14)
        ),
        margin=dict(l=80, r=40, t=100, b=80)
    )

def plot_regression_comparison():
    if not os.path.exists('results'):
        print("No results directory found.")
        return

    # Define the 4 experiments
    experiments = [
        ('BLR (Analytic) - PV', 'results/rmse_analytic_predictive_variance.npy', '#2E86C1', 'solid'), # Blue
        ('MFVI - PV',           'results/rmse_mfvi_predictive_variance.npy',     '#28B463', 'solid'), # Green
        ('BLR (Analytic) - Random', 'results/rmse_analytic_random.npy',          '#E74C3C', 'dash'),  # Red
        ('MFVI - Random',           'results/rmse_mfvi_random.npy',              '#884EA0', 'dash')   # Purple
    ]
    
    fig = go.Figure()
    
    for name, filepath, color, dash in experiments:
        if not os.path.exists(filepath):
            print(f"Missing file: {filepath}")
            continue
            
        rmse = np.load(filepath)
        # 100 rounds, 10 queries per round, starting from 20
        x_axis = [i * 10 + 20 for i in range(len(rmse))]
        
        fig.add_trace(go.Scatter(
            x=x_axis, y=rmse,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=3, dash=dash),
            marker=dict(size=6)
        ))
        
    layout = get_premium_layout(
        title="Regression RMSE Comparison (Phase 1)",
        x_title="Number of Labeled Images",
        y_title="Validation RMSE"
    )
    fig.update_layout(layout)
    
    # Save
    fig.write_html("results/regression_comparison.html")
    fig.write_image("results/regression_comparison.png", width=1200, height=800, scale=2)
    print("Saved regression plots to results/regression_comparison.png")

if __name__ == "__main__":
    plot_regression_comparison()
