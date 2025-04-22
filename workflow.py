"""
Script to run langgraph workflow that creates pptx file from segments info
using GCP. LLMs in Vertex AI model garden is being used here.
Set the env variable, GOOGLE_CLOUD_PROJECT, with the project ID, before running the script.

Examples:

(1) with local files

    python workflow.py --input_path=cluster_info.csv --final_pptx_path=segments.pptx
    
(2) with files in cloud storage for example in GCP
    (assuming input file, cluster_info.csv, is in a data/ folder in a bucket, and
    a pptx template for 4 segments is in templates/ folder.
    The final slide will be written to output/ folder.)

    python workflow.py --input_path=gs://<bucket>/data/cluster_info.csv \
     --final_pptx_path=gs://<bucket>/output/segments.pptx \
     --template_pptx_prefix=gs://<bucket>/templates

"""
import os
import logging
from typing import Annotated, TypedDict, List, Tuple
import argparse

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from utils import *
from prompts import *

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", None)
if PROJECT_ID is None:
    PROJECT_ID = input(
        "An environment variable, GOOGLE_CLOUD_PROJECT, is not set. PROJECT_ID is:"
    )

LOCATION = "us-east5"
CLIENT = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)

# Initialise the Langchain Model
MODEL_LANGCHAIN = ChatAnthropicVertex(
    model_name=MODEL_NAME,
    location=LOCATION,
    project_id=PROJECT_ID,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    max_output_tokens=MAX_TOKENS,
    max_retries=MAX_RETRIES,
)

# Define the agent's state
class AgentState(TypedDict):
    input_path: str
    inputs: dict
    template_pptx_prefix: str
    final_pptx_filename: str
    num_segments: int
    segments_df_str: str
    search_queries: list
    messages: Annotated[list, add_messages]


def get_input(
    state: AgentState
):
    """
    Function for getting the input for the workflow
    """
    print("* starting get_input")
    filepath = state["input_path"]
    filetype = filepath.split(".")[-1]
    # (1) Get the input data
    if filetype == "csv":
        input_df = pd.read_csv(filepath)
    elif filetype == "parquet":
        input_df = pd.read_parquet(filepath)
    else:
        raise Exception(f"File type, {filetype}, not supported.")
    input_df["segment_id"] = input_df["segment_id"].astype(int)
    if not set(input_df.columns) >= set(INPUT_COLUMNS):
        raise Exception("Check input fields of the input data.")
    segments = sorted(list(input_df["segment_id"].unique()))
    num_segments = len(segments)
    inputs = {}
    for segment_id in segments:
        segment_input = input_df.loc[input_df["segment_id"] == segment_id, INPUT_COLUMNS].to_json(orient="records")
        inputs[int(segment_id)] = segment_input
    return {
        "num_segments": num_segments,
        "inputs": inputs,
    }


def get_all_names_and_descriptions(
    state: AgentState
):
    print("* starting get_all_names_and_descriptions")
    inputs = state["inputs"]
    all_segments = []
    for segment_id, segment_input in inputs.items():
        num_tries = 0
        while num_tries < MAX_RETRIES:
            segment_info = get_name_and_description(
                segment_id,
                segment_input,
                prompt_template1,
                client=CLIENT,
            )
            eval_result = evaluate_name_and_description(
                segment_input,
                segment_info,
                prompt_template2,
                client=CLIENT,
            )
            if eval_result.startswith("y"):
                all_segments.append(segment_info)
                break
            else:
                print(f"Retrying {num_tries} for segment {segment_id}: {eval_result}")
            num_tries += 1
        if num_tries == MAX_RETRIES:
            all_segments.append({'segment_id': segment_id, 'segment_name': 'NA',
                                 'segment_description': 'NA', 'segment_share_in_percent': 'NA'})
    segments_df = pd.DataFrame(all_segments)
    segments_df["segment_id"] = segments_df["segment_id"].astype(int)
    return {"segments_df_str": segments_df.to_json()}


def get_search_queries_for_images(
    state: AgentState
):
    print("* starting get_search_queries_for_images")
    segments_df = pd.DataFrame(json.loads(state["segments_df_str"]))
    segments_df["segment_id"] = segments_df["segment_id"].astype(int)
    search_queries = []
    for segment_id in sorted(segments_df["segment_id"].unique()):
        search_query = segments_df[segments_df["segment_id"] == segment_id].iloc[0]["segment_name"] + " with people"
        search_queries.append({"id": int(segment_id), "search_query": search_query})
    return {
        "search_queries": search_queries,
        "messages": [HumanMessage(content=prompt_template4_user.format(search_queries=json.dumps(search_queries)))]
    }


@tool
def find_and_save_image_for_segment(
        segment_id: str,
        search_query: str,
) -> str:
    """
    Search images in Google Images using the given search query, and retrieve those images.

    Args:
        segment_id (str): id of the segment
        search_query (str): The search term or description

    Returns:
        filename of the saved image
    """
    print(f"* starting find_and_save_image_for_segment for segment_id {segment_id}")
    images = search_google_images_selenium(search_query, num_images=15)
    image_filetype, image_base64 = find_the_best_image(images, search_query, prompt_template=prompt_template3,
                                                       client=CLIENT)
    filename = f"best_image{segment_id}"
    filename = save_base64_image_to_file(image_base64, filename=filename, new_size=140)
    return filename


def get_images(state: AgentState):
    print("* starting get_images")
    messages = state["messages"]
    last_message = messages[-1]
    model_with_tools = MODEL_LANGCHAIN.bind_tools([find_and_save_image_for_segment])
    response = model_with_tools.invoke(messages)
    return {
        "messages": [response]
    }


# Determine if additional tool calls are needed
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "find_and_save_image_tool"
    return "create_slide"


def create_slide(state: AgentState):
    print("* starting create_slide")
    segments_df = pd.DataFrame(json.loads(state["segments_df_str"]))
    segments_df["segment_id"] = segments_df["segment_id"].astype(int)
    num_segments = state["num_segments"]
    template_pptx_prefix = state["template_pptx_prefix"]
    final_pptx = state["final_pptx_filename"]
    try:
        _ = generate_slide(segments_df,
            num_segments=num_segments,
            template_pptx_prefix=template_pptx_prefix,
            final_pptx=final_pptx
        )
        return {"messages": f"{final_pptx} has been created."}
    except Exception as e:
        return {"messages": str(e)}


def define_and_run_workflow(
    input_path: str,
    final_pptx_path: str,
    template_pptx_prefix: str,
):
    # Define the workflow
    # Initialize the LangGraph workflow, specifying the agent's state schema
    workflow = StateGraph(AgentState)

    # Add nodes to the workflow, associating each node with its corresponding function
    workflow.add_node("get_input", get_input)
    workflow.add_node("get_all_names_and_descriptions", get_all_names_and_descriptions)
    workflow.add_node("get_search_queries_for_images", get_search_queries_for_images)
    workflow.add_node("get_images", get_images)
    workflow.add_node("find_and_save_image_tool", ToolNode([find_and_save_image_for_segment]))
    workflow.add_node("create_slide", create_slide)

    # Define the flow of execution between nodes, creating the workflow's logic
    workflow.add_edge(START, "get_input")
    workflow.add_edge("get_input", "get_all_names_and_descriptions")
    workflow.add_edge("get_all_names_and_descriptions", "get_search_queries_for_images")
    workflow.add_edge("get_search_queries_for_images", "get_images")
    workflow.add_conditional_edges("get_images", should_continue, ["find_and_save_image_tool", "create_slide"])
    workflow.add_edge("find_and_save_image_tool", "get_images")
    workflow.add_edge("create_slide", END)


    # Compile the LangGraph workflow, enabling memory-based state management
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    # Initialize a LangGraph thread with a unique ID for state management.
    thread_config = {"configurable": {"thread_id": "1"}}
    #stream_mode = "updates"  # "updates" (only updates) or "values" (all values)
    input_state = {
        "input_path": input_path,
        "template_pptx_prefix": template_pptx_prefix,
        "final_pptx_filename": final_pptx_path,
        "messages": [
            SystemMessage(content=prompt_template4_system),
        ],
    }
    # Execute the LangGraph workflow, streaming the results of each node.
    for state in graph.stream(
        input=input_state,
        config=thread_config,
    ):
        # Print the name of the current node and its output for each step.
        for node_name, node_output in state.items():
            print(f"Agent Node: {node_name}\n")
            print("Agent Result:")
            print(str(node_output))  # Truncate output for display
        print("\n====================\n")
    return state
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input path for csv containing segments info",
    )
    parser.add_argument(
        "--final_pptx_path",
        type=str,
        required=True,
        help="Path for output pptx file",
    )
    parser.add_argument(
        "--template_pptx_prefix",
        type=str,
        default=".",
        help="Path for template pptx's. In this directory, template4.pptx will exist for 4 segments for example.",
    )
    args = parser.parse_args()
    input_path = args.input_path
    final_pptx_path = args.final_pptx_path
    template_pptx_prefix = args.template_pptx_prefix
    define_and_run_workflow(input_path, final_pptx_path, template_pptx_prefix)
