GENERATION_PROMPT = """
You are an expert Data Labeler specializing in automotive datasets. Your task is to generate a set of diverse user queries and corresponding answers in JSON format based on the provided function definitions.
These functions act as interfaces to access and control vehicle properties. You must construct query-answer pairs that exemplify the practical usage of these functions in realistic scenarios.

### Guidelines for Query Generation

- Use varied vocabulary and syntax. Avoid repetitive sentence structures. Queries can be long or short, complex or concise.
- Ensure a mix of query types, such as direct commands, questions, or requests containing descriptive context.
- Include implicit queries where the user states a feeling or need rather than a direct command (e.g., 'I'm feeling chilly' instead of 'Increase the temperature').
- Generate queries that contain multiple parallel queries in natural language for the given functions, they could use either the same function with different arguments or different functions (e.g., setting two different properties or calling the same function twice with different arguments).
- Ensure every parameter implied by the query is populated with specific, plausible values (e.g., if a date is required, provide a valid, reasonable date).
- Across a variety level of difficulties, ranging from beginner and advanced use cases.
- DO NOT create queries requiring Multi-Step and Multi-Turn interactions (keep it single-turn).

### Guidelines for Answer Generation:

- The answer must be a list of function calls strictly following the JSON format.
- The number of function calls in the list must match the number of intents/requests in the user query.
- The generated function calls must accurately and effectively resolve the user's request.
- The corresponding result's parameter types and ranges match with the property descriptions.

{section_examples_placeholder}

### Output JSON Format:

Similar to the examples above, your output MUST strictly adhere to the following JSON format. Do not include any explanatory text outside the JSON block:

```json
[
  {
    "query": "The generated query.",
    "answers": [
      {
        "name": "api_name",
        "arguments": {
          "arg_name": "value"
        }
      }
    ]
  }
]
```

### Contextual Data Definitions

The vehicle functions are described as vehicle properties in JSON format. Please strictly adhere to the following schema when interpreting the data:
{vehicle_property_schema_placeholder}

Here is the list of vehicle properties supported by the vehicle:
{vehicle_properties_placeholder}

### Function Definitions

The detailed functions description is as follows:
{car_property_functions_placeholder}

Now, please generate {pair_number_placeholder} diverse query and answer pairs following the Output JSON Format specified above.

"""


SEED_GENERATION_EXAMPLE_SECTION = """### Examples:

Here are several examples of tool and answer samples from other public datasets to demonstrate the expected logic:
{xlam_function_calling_samples_placeholder}
"""

EXPANSION_GENERATION_EXAMPLE_SECTION = """### Examples:

Here are several examples of query-answer pairs:
{expansion_function_calling_samples_placeholder}

When generating new pairs, do not repeat the logic already shown in the examples above. Instead, focus on demonstrating different functionalities and parameters.
"""

DEVELOPER_MESSAGE_PROMPT = """The vehicle functions are described as vehicle properties in JSON format.
To understand the data, the following data structure description is provided.
Please strictly adhere to this description when interpreting the meaning of JSON fields.
{vehicle_property_schema_placeholder}

Here is a list of vehicle properties supported by my vehicle described in JSON format.
{vehicle_properties_placeholder}

You are a model that can do function calling with the following functions.
"""
