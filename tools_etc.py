import pandas as pd
from typing import List
from llama_index.tools.function_tool import FunctionTool
import os


#------- Función de consulta y promedio

def create_average_metric_tool(df_tools: pd.DataFrame):
    """
    Devuelve una FunctionTool que calcula el promedio de una columna numérica especificada en un DataFrame.
    Puede calcular el promedio para todo el DataFrame o solo para filas que coinciden con valores específicos en otra columna.
    """

    def average_metric(column_name: str, filter_column: str = None, filter_values: List[str] = None) -> float:
        """
        Calcula el promedio de 'column_name'. Si se proporcionan 'filter_column' y 'filter_values',
        se calcula el promedio sólo para las filas que tienen esos valores en 'filter_column'.
        """
        try:
            # Verificar que la columna para el promedio existe y es numérica
            if column_name not in df_tools.columns or not pd.api.types.is_numeric_dtype(df_tools[column_name]):
                raise ValueError(f"La columna '{column_name}' no existe o no contiene datos numéricos.")

            if filter_column and filter_values:
                if filter_column not in df_tools.columns:
                    raise ValueError(f"La columna de filtro '{filter_column}' no existe en el DataFrame.")
                
                filtered_df = df_tools[df_tools[filter_column].isin(set(filter_values))]
                if filtered_df.empty:
                    return float('nan')
                return filtered_df[column_name].mean()
            else:
                return df_tools[column_name].mean()
        
        except KeyError as e:
            print(f"Error de clave: {e}")
            return float('nan')
        except ValueError as e:
            print(e)
            return float('nan')
        except Exception as e:
            print(f"Error inesperado: {e}")
            return float('nan')

    return FunctionTool.from_defaults(
        fn=average_metric,
        name="average_metric_calculator",
        description="Calcula el promedio de una columna numérica. Opcionalmente, calcula sólo para filas que coincidan con valores específicos en otra columna."
    )




#------ función de lectura tabla csv

#def describe_csv(file_path: str):
def describe_csv():    
    #file_path = os.path.join("data", file_path)
    file_path = os.path.join("data", "BankChurners.csv")
    df = pd.read_csv(file_path)
    return df.describe()

#------ función de estadisticas basicas a tabla csv

pandas_describe_tool = FunctionTool.from_defaults(
    fn=describe_csv,
    name="pandas_describe",
    description="Devuelve estadísticas descriptivas de un archivo CSV."
)

#--------------------
