# Generating segment names and descriptions using cluster information
prompt_template1 = """\
You are a marketing strategist. \
Your role is generating its description and a short name for a given segment \
of individuals by analyzing mean and overall values of features for the segment. \

Approach the following task step-by-step in 3 steps \
inside <steps> tag.

<steps>
<step_1>
# Task
As the first step, provide a detailed list of important facts \
about the given segment. \
# Guidelines
1. The segment information will be given as inputs. \
The fields for the segment information will be given in <input_fields> tags. \
2. You will analyze the segment information to understand the unique \
characteristics of the individuals in the segment. And create a list of key facts.\
</step_1>
<step_2>
# Task
Using the provided list of key facts, write a detailed description \
of the segment.
# Guidelines
1. You must focus on individuals of the segment. \
2. Try to use all facts found on step 1.
3. The segment description will be a paragraph that \
contains information on as many given features as possible.
4. For a description on each feature in the segment description, \
use the numbers (segment means) \
for features that people can easily understand. \
For example, for age with the segment mean at 41.1, \
we can describe the segment using the phrase like \
"The individuals in this segment tend to be in their early 40s". \
We can use numbers on incomes and distances, too.
5. The description must be less than 100 tokens.
</step_2>
<step_3>
# Task
Based on the segment description, generate a segment name in less than 10 tokens. \
Do not use the negative expressions like "non-" in the name. For example, "non-sports" is not allowed.\
</step_3>
</steps>

<input_fields>
feature_description: feature description
overall_mean: overall mean for a feature
segment_mean: segment mean for a feature
segment_id: ID of the given segment
</input_fields>

<output_fields>
segment_id: ID of the given segment
segment_name: name of the cluster
segment_description: description of the cluster
segment_share_in_percent: share of the segment in percent
</output_fields>

Generate segment description and its name using the below segment input: 
<segment_input_information>
{segment_input}
</segment_input_information>

<output_instruction>
1. The final output should be in JSON format using fields described in <output_fields> tabs. \
2. If multiple options are generated, pick the best one, and include only one JSON output \
like this example in the response:
<example_output>
{{"segment_id": 1,\n  "segment_name": segment name, \n "segment_description": description, \
"segment_share_in_percent": share}}
</example_output>
</output_instruction>
"""

# Prompt for validating segment names and descriptions
prompt_template2 = """\
You are a marketing strategist. \
Your role is: \
1. Read the input and output for segment name and description carefully. \
2. Check if the output is generated correctly. Check the facts carefully. \
For example, if output has extra information not in input, output is incorrect. \
3. Input is given as a data frame with fields described in <input_fields> tag. \
Output is given as JSON format with fields described in <output_fields> tag. \

<input_fields>
feature_description: feature description
overall_mean: overall mean for a feature
segment_mean: segment mean for a feature
segment_id: ID of the given segment
</input_fields>

<output_fields>
segment_id: ID of the given segment
segment_name: name of the cluster
segment_description: description of the cluster
segment share in percent: share of the segment in percent
</output_fields>

<input>
{input_str}
</input>
<output>
{output_str}
</output>

<response_instruction>
The final response should start with either "yes" or "no". Then add a brief explanation.
</response_instruction>
"""

# Prompt for picking the best image
prompt_template3 = """
You are an assistant to an analyst who is creating a presentation material.
Here we are trying to find the best image for the given description.

<description>
{search_query}
</description>

<guidelines>
1. The best image has to have people's happy faces in it.
2. The best image  has to be a photographed image.
3. The best image does not contain texts in the image.
4. Do not pick one with multiple photographs in one image.
5. Do not pick an inappropriate image because the image will be used in a presentation.
6. Your response should be just one integer. For example, if the 3rd image out of 10 images is picked, \
the response should be just "3".
</guidelines>
"""

# Define the system instruction and the user's initial question for "get_images" agent.
prompt_template4_system = """Use the tools provided to answer the user's question. \
The user will provide a list of (search query, its ID). \
Your role is getting the best image for each search query. \
Input fields are described in <input_fields> tag. \
If the user gives a list of search queries, find the best image for each search query. \
The best image for each search query will be saved as a file and the file path will be returned. \

<input_fields>
"id": ID of the search query
"search_query": search query
</input_fields>

<output_fields>
"id": ID of the search query
"path": file path of the image
</output_fields>

<output_instruction>
Use JSON format using output fields described in <output_fields> tag. \
Do not include additional texts other than JSON output.
</output_instruction>

"""

prompt_template4_user = """Obtain the best images for given search queries.
<search_queries>
{search_queries}
</search_queries>
"""