import os
import time

from flask import Flask, request

from workflow import define_and_run_workflow

INPUT_PATH = "cluster_info.csv"
TEMPLATE_PPTX_PREFIX = "."
FINAL_PPTX_PATH = "segments.pptx"

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def run_workflow_app():
    if request.method == "POST":
        data = request.get_json()
    elif request.method == "GET":
        data = request.args
    print(data)
    if data:
        input_path = data.get("input_path", INPUT_PATH)
        final_pptx_path = data.get("final_pptx_path", FINAL_PPTX_PATH)
        template_pptx_prefix = data.get("template_pptx_prefix", TEMPLATE_PPTX_PREFIX)
    else:
        input_path = INPUT_PATH
        final_pptx_path = FINAL_PPTX_PATH
        template_pptx_prefix = TEMPLATE_PPTX_PREFIX
    print(input_path, final_pptx_path, template_pptx_prefix)
    
    start_time = time.time()
    final_state = define_and_run_workflow(input_path, final_pptx_path, template_pptx_prefix)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return f'{final_state["create_slide"]["messages"]} It took {elapsed_time:.2f} seconds.', 200

@app.route("/test", methods=["GET", "POST"])
def test_app():
    if request.method == "POST":
        data = request.get_json()
    elif request.method == "GET":
        data = request.args
    print(data)
    if data:
        input_path = data.get("input_path", INPUT_PATH)
        final_pptx_path = data.get("final_pptx_path", FINAL_PPTX_PATH)
        template_pptx_prefix = data.get("template_pptx_prefix", TEMPLATE_PPTX_PREFIX)
    else:
        input_path = INPUT_PATH
        final_pptx_path = FINAL_PPTX_PATH
        template_pptx_prefix = TEMPLATE_PPTX_PREFIX
    return f"{input_path}, {final_pptx_path}, {template_pptx_prefix}", 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))