"""
Visualization utilities for the Finite Monkey web interface.
Provides functions to generate charts and visualizations using matplotlib and seaborn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np
import json
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any, Tuple, Optional, Union

# Configure matplotlib for non-interactive environments
plt.switch_backend('Agg')

# Set seaborn style
sns.set_theme(style="darkgrid")

def generate_base64_image(figure: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG image."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', bbox_inches='tight', dpi=100, transparent=True)
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(figure)
    return image_data

def create_chart_html(figure: plt.Figure, width: str = "100%", height: str = "400px") -> str:
    """Create an HTML img tag with a base64-encoded matplotlib figure."""
    image_data = generate_base64_image(figure)
    return f'<img src="data:image/png;base64,{image_data}" style="width:{width};height:{height};object-fit:contain;" />'

def histogram_chart(data: List[float], title: str = "Distribution", 
                    xlabel: str = "Value", ylabel: str = "Frequency", 
                    bins: int = 20) -> plt.Figure:
    """Create a histogram chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data, bins=bins, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

def bar_chart(categories: List[str], values: List[float], 
             title: str = "Bar Chart", xlabel: str = "Category", 
             ylabel: str = "Value") -> plt.Figure:
    """Create a bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=categories, y=values, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def line_chart(x_data: List, y_data: List, title: str = "Line Chart",
              xlabel: str = "X", ylabel: str = "Y") -> plt.Figure:
    """Create a line chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=x_data, y=y_data, marker='o', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

def scatter_chart(x_data: List, y_data: List, title: str = "Scatter Plot",
                 xlabel: str = "X", ylabel: str = "Y", 
                 hue: Optional[List] = None) -> plt.Figure:
    """Create a scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=x_data, y=y_data, hue=hue, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

def pie_chart(labels: List[str], sizes: List[float], title: str = "Pie Chart") -> plt.Figure:
    """Create a pie chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_title(title)
    return fig

def heatmap_chart(data: List[List[float]], title: str = "Heatmap",
                 xlabel: str = "X", ylabel: str = "Y", 
                 x_labels: Optional[List] = None,
                 y_labels: Optional[List] = None) -> plt.Figure:
    """Create a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data, annot=True, cmap="viridis", ax=ax,
               xticklabels=x_labels, yticklabels=y_labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

def radar_chart(categories: List[str], values: List[float], 
               title: str = "Radar Chart") -> plt.Figure:
    """Create a radar chart (also known as a spider or star chart)."""
    # Compute angle for each category
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
    # Make the plot circular by repeating the first value
    values = values + [values[0]]
    angles = angles + [angles[0]]
    categories = categories + [categories[0]]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
    ax.set_title(title)
    ax.grid(True)
    return fig

def multi_line_chart(x_data: List, y_data_dict: Dict[str, List[float]], 
                     title: str = "Multiple Lines", 
                     xlabel: str = "X", ylabel: str = "Y") -> plt.Figure:
    """Create a chart with multiple lines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, y_values in y_data_dict.items():
        ax.plot(x_data, y_values, marker='o', linestyle='-', label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig

def time_series_chart(timestamps: List[datetime], values: List[float], 
                     title: str = "Time Series", ylabel: str = "Value") -> plt.Figure:
    """Create a time series chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=timestamps, y=values, marker='o', ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def grouped_bar_chart(data: Dict[str, Dict[str, float]], 
                      title: str = "Grouped Bar Chart",
                      xlabel: str = "Category", ylabel: str = "Value") -> plt.Figure:
    """Create a grouped bar chart."""
    categories = list(data.keys())
    groups = list(data[categories[0]].keys())
    
    x = np.arange(len(categories))
    width = 0.8 / len(groups)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, group in enumerate(groups):
        values = [data[cat][group] for cat in categories]
        ax.bar(x + i*width - 0.4 + width/2, values, width, label=group)
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def paired_bar_chart(categories: List[str], values1: List[float], values2: List[float],
                    label1: str, label2: str, title: str = "Paired Bar Chart",
                    xlabel: str = "Category", ylabel: str = "Value") -> plt.Figure:
    """Create a paired bar chart with two sets of bars side by side."""
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, values1, width, label=label1)
    ax.bar(x + width/2, values2, width, label=label2)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

# Example function to generate sample data for testing
def generate_sample_data():
    """Generate sample data for testing visualizations."""
    return {
        "histogram": {
            "data": np.random.normal(0, 1, 1000).tolist(),
            "title": "Normal Distribution",
            "xlabel": "Value",
            "ylabel": "Frequency",
            "bins": 30
        },
        "bar_chart": {
            "categories": ["Category A", "Category B", "Category C", "Category D", "Category E"],
            "values": [25, 40, 30, 55, 15],
            "title": "Sample Bar Chart",
            "xlabel": "Category",
            "ylabel": "Value"
        },
        "line_chart": {
            "x_data": list(range(10)),
            "y_data": [x**2 for x in range(10)],
            "title": "Quadratic Function",
            "xlabel": "X",
            "ylabel": "Y = X²"
        },
        "time_series": {
            "timestamps": [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(30, 0, -1)],
            "values": [random.uniform(10, 100) for _ in range(30)],
            "title": "Last 30 Days",
            "ylabel": "Random Value"
        }
    }