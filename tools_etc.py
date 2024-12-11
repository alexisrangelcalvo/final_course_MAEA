import pandas as pd
from typing import List
from llama_index.tools.function_tool import FunctionTool

def create_average_metric_tool(df_tools: pd.DataFrame):
    """
    Esta función recibe un DataFrame y devuelve una función que calcula
    el promedio de la métrica para una lista de registros y una columna específica.
    No hace uso de variables globales, ya que 'df' se captura en el closure.
    """

    def average_metric(countries: List[str], column_name: str) -> float:
        values = []
        for country in countries:
            val = df_tools.loc[df_tools['Country'] == country, column_name].values[0]
            values.append(val)
        return sum(values) / len(values)

    # Ahora creamos la FunctionTool a partir de esta función interna
    return FunctionTool.from_defaults(
        fn=average_metric,
        name="average_metric_calculator",
        description="Dada una lista de registros y una columna numérica, calcula el promedio de dicha métrica para esos países."
    )