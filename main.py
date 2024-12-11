from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent#, Tool
from llama_index.tools.function_tool import FunctionTool
#from llama_index.core.tools import FunctionTool
#from llama_index.tools import Tool
from llama_index.llms import OpenAI
from pdf import canada_engine
from tools_etc import create_average_metric_tool


# TO SET THE API KEY BY TERMINAL
#export OPENAI_API_KEY="***REMOVED***"
# ESTO SE TIENE QUE ELIMINAR ANTES DE HACER UN COMMIT, DE LO CONTRARIO LO RECHAZARA GITHUB


load_dotenv() # ¿Esto para qué se utiliza?

csv_path = os.path.join("data", "population.csv")
csv_df = pd.read_csv(csv_path)

df_query_engine = PandasQueryEngine(
    df=csv_df, verbose=True, instruction_str=instruction_str
)
df_query_engine.update_prompts({"pandas_prompt": new_prompt})

#----- Seccion de creación de herramientas

average_metric_tool = create_average_metric_tool(csv_df)
#---------------------------------------------------------------__#

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=df_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information at the world population and demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=canada_engine,
        metadata=ToolMetadata(
            name="canada_data",
            description="this gives detailed information about canada the country",
        ),
    ),
    average_metric_tool
]


llm = OpenAI(
    model="gpt-3.5-turbo")


agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("HI! Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)