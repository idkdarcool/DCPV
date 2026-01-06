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

def plot_comparison():
    if not os.path.exists('results'):
        print("No results directory found.")
        return

    try:
        bayes_results = np.load('results/classification_results_bayes.npy', allow_pickle=True).item()
        det_results = np.load('results/classification_results_det.npy', allow_pickle=True).item()
    except FileNotFoundError:
        print("Could not find both result files. Make sure you ran both modes.")
        return

    strategies = ['bald', 'entropy', 'var_ratios']
    
    # Premium Color Palette
    colors = {
        'bayes': '#2E86C1', # Strong Blue
        'det': '#E74C3C'    # Strong Red
    }
    
    for strategy in strategies:
        if strategy in bayes_results and strategy in det_results:
            fig = go.Figure()
            
            # Bayesian Curve
            acc_bayes = bayes_results[strategy]
            x_axis = [i * 10 + 20 for i in range(len(acc_bayes))]
            fig.add_trace(go.Scatter(
                x=x_axis, y=acc_bayes,
                mode='lines+markers', 
                name=f'{strategy.upper()} (Bayesian)',
                line=dict(color=colors['bayes'], width=3),
                marker=dict(size=8, symbol='circle')
            ))
            
            # Deterministic Curve
            acc_det = det_results[strategy]
            x_axis_det = [i * 10 + 20 for i in range(len(acc_det))]
            fig.add_trace(go.Scatter(
                x=x_axis_det, y=acc_det,
                mode='lines+markers', 
                name=f'{strategy.upper()} (Deterministic)',
                line=dict(color=colors['det'], width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
            layout = get_premium_layout(
                title=f"Bayesian vs Deterministic CNN ({strategy.upper()})",
                x_title="Number of Labeled Images",
                y_title="Test Accuracy"
            )
            fig.update_layout(layout)
            
            filename_html = f'results/comparison_{strategy}.html'
            fig.write_html(filename_html)
            
            filename_png = f'results/comparison_{strategy}.png'
            fig.write_image(filename_png, width=1200, height=800, scale=2)
            
            print(f"Saved premium plot to {filename_html} and {filename_png}")

def plot_exp_1_1():
    """Generates the main Exp 1.1 plot (All Bayesian Strategies)."""
    if not os.path.exists('results/classification_results_bayes.npy'):
        print("Bayesian results not found.")
        return

    bayes_results = np.load('results/classification_results_bayes.npy', allow_pickle=True).item()
    
    fig = go.Figure()
    
    # Define colors for strategies
    colors = {
        'bald': '#2E86C1',      # Blue
        'var_ratios': '#28B463', # Green
        'entropy': '#E74C3C',    # Red
        'mean_std': '#884EA0',   # Purple
        'random': '#7F8C8D'      # Grey
    }
    
    for strategy, acc in bayes_results.items():
        x_axis = [i * 10 + 20 for i in range(len(acc))]
        
        # Clean up name
        name = strategy.replace('_', ' ').title()
        if strategy == 'bald': name = 'BALD'
        if strategy == 'mean_std': name = 'Mean STD'
        if strategy == 'var_ratios': name = 'Var Ratios'
        
        fig.add_trace(go.Scatter(
            x=x_axis, y=acc,
            mode='lines+markers',
            name=name,
            line=dict(color=colors.get(strategy, 'black'), width=3),
            marker=dict(size=8)
        ))
        
    layout = get_premium_layout(
        title="Acquisition Function Comparison (Exp 1.1)",
        x_title="Number of Labeled Images",
        y_title="Test Accuracy"
    )
    fig.update_layout(layout)
    
    fig.write_html("results/experiment_1_1.html")
    fig.write_image("results/experiment_1_1.png", width=1200, height=800, scale=2)
    print("Saved Exp 1.1 plot to results/experiment_1_1.png")

if __name__ == "__main__":
    plot_comparison()
    plot_exp_1_1()
