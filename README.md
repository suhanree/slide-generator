# Slide Generator for segments information
This is a project created for GlobaL Google Cloud Agent AI Hackathon (4/2025) for Deloitte. 
## Problem Statement
If a company has customer data, for example, it may want to create several segments using customer data to have more personalized marketing communication and more.
Let's assume we already created segments from a group of customers, and the aggregated data for segments look like below. 

<img width="666" alt="Screenshot 2025-04-22 at 1 00 48 AM" src="https://github.com/user-attachments/assets/26814704-09aa-46ec-8a33-551dac7808e9" />

Given this dataset, this python script summarizes this tabular data into segment names and descriptions using LLMs, and create a pptx slide page with some representative images on these segments using AI agents.
Here we created a docker container to serve in GCP, and also use Google cloud storage for input data and output slide.

For agent framework, Langgraph has been used to create a workflow. Also this workflow does not have a chat UI to run it, it is designed to be executed in batch modes or with triggers.

## Architecture
<img width="621" alt="workflow_visualization" src="https://github.com/user-attachments/assets/5506b85f-efc8-46d7-8137-bccbfdac0c61" />

The workflow has many steps.

1. get_input: It reads the input cluster information shown above.
2. get_all_names_and_descriptions: Using the segment information, it creates names and descriptions for segments using LLMs. Here it also uses **judge LLMs** to check outputs for correctness.
3. get_search_queries_for_image: Using the generated segment names, it creates search queries for segment images.
4. get_image: This is a ReAct agent bound with a tool. Given a list of search queries, it invokes a tool for retrieving images and picking the best image out of candidates using LLMs. Here we have 4 segments, so this tool will be invoked 4 times.
5. find_and_save_image_tool: This is a tool that searches the google image to retrieve 15 images for each segment, and picks the best image using LLMs from these candidates for each segment.
6. create_slide: Using images and generated names and descriptions, it creates a powerpoint slide in a nice looking table as below. A python package called "python-pptx" has been used here. It can create a slide from scratch, but here we added contents to a template pptx slide.

<img width="1223" alt="Screenshot 2025-04-22 at 1 16 20 AM" src="https://github.com/user-attachments/assets/e3102d33-5515-4e9f-b855-ef0a6d343d5f" />

### Alternative architecture

- There are many ways to accomplish this, and there may be an easier solution or more complicated workflows.
- Simpler solution: maybe one call to a powerful LLM can do all the job at once, but splitting the task into multiple steps will do a job better.
- Also more complicated workflows can be created. The step, "get_all_names_and_descriptions", can be represented in a sub-workflow with parallel tasks and evaluations tasks, but here for simplicity, it was represented as one step.
- Evaluation steps for names and descriptions were added, but there is no evaluation step for final images for the slide. Those can be added.

## Role of an agent
There is one ReAct agent in this workflow. "get_image", and this agent has one tool "find_and_save_image_tool". Given the list of search queries for segments, this agent will invoke this tool for each segment (4 times here).
The final outcome of this agent will be 4 image files for 4 segments.

## How to run

With an environment variable, GOOGLE_CLOUD_PROJECT, with the project ID for GCP, we can run a script locally like this:
```python workflow.py --input_path=cluster_info.csv --final_pptx_path=segments.pptx```

With Dockerfile, we can created an API using Cloud Run in GCP also.
