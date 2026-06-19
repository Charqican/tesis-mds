import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np

    file = "./data/train_1.parquet"
    df = pd.read_parquet(file)
    def get_obj(df, i): 
        obj_ = df.loc[i]['inputs']
        # Esto convierte tu array de objetos en un array de float64 puro (N, 3)
        obj = np.stack(obj_)
        return obj  # Debería dar (N, 3)

    return df, get_obj


@app.cell
def _(df, get_obj):
    import plotly.graph_objects as go
    data = get_obj(df, 1)
    fig = go.Figure(data=[go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode='markers',
        marker=dict(
            size=2,          # <--- Ajusta este valor (por defecto suele ser 12)
            opacity=0.8      # Opcional: mejora la visibilidad si tienes muchos puntos
        )
    )])

    fig.show()
    return


if __name__ == "__main__":
    app.run()
