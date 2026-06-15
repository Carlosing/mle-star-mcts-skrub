"""Defines the prompts for the initialization agent."""

SUMMARIZATION_AGENT_INSTR = """# Task description
{task_description}

# Your task
- Summarize this task description.
- Your summary will be used for searching recent effective models for {task_type}.

# Requirement
- We will directly use your response, so be simple and concise.
"""

MODEL_RETRIEVAL_INSTR = """# Competition
{task_summary}

# Your task
- Propose {num_model_candidates} simple and effective machine learning model for the above competition.
- For California housing prices (a tabular regression task), prefer scikit-learn models such as RandomForestRegressor, GradientBoostingRegressor, or a simple neural network with scikit-learn's MLPRegressor.
- Avoid PyTorch unless strictly necessary; scikit-learn is preferred for this task.

# Requirement
- Provide the model name and a short, self-contained Python code snippet showing how to train it on a train/validation split.
- The example code should be concise and simple.

Use this exact JSON format (and nothing else):
[
  {{"model_name": "Name of the model", "example_code": "import pandas as pd\\n..."}}
]"""


MODEL_EVAL_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a task description and a model description.
- You need to implement your Python solution using the provided model.

# Task description
{task_description}

# Model description
{model_description}

# Your task
- Implement the solution in Python.
- You must use the model as described in the model description.
- This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.
- Propose an evaluation metric that is reasonable for this task (e.g., RMSE for regression).
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- Do not include other models that are not directly related to the model described.
- Prefer scikit-learn for this tabular task. PyTorch is allowed only if the model description explicitly requires it.
- If you use scikit-learn >= 1.7, use `sklearn.metrics.root_mean_squared_error` to compute RMSE. Do NOT pass `squared=False` to `mean_squared_error`.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data in the `./input` directory for training and validation. Do NOT use test.csv for computing the validation score.
- For California Housing: the train.csv has an ID column first, then 8 feature columns, then the target column `median_house_value`. Use `train_test_split` from sklearn on train.csv to create train and validation sets.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior.
"""

BUG_SUMMARY_INSTR = """# Error report
{bug}

# Your task
- Remove all unnecessary parts of the above error report.
- We are now running {filename}.py. Do not remove where the error occurred."""

BUG_REFINE_INSTR = """# Task description
{task_description}

# Code with an error:
{code}

# Error:
{bug}

# Your task
- Please revise the code to fix the error.
- Do not remove subsampling if exists.
- Provide the improved, self-contained Python script again.
- There should be no additional headings or text in your response.
- All the provided input data is stored in \"./input\" directory.
- Remember to print a line in the code with 'Final Validation Performance: {{final_validation_score}}' so we can parse performance.
- The code should be a single-file python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the refined Python code."""

CODE_INTEGRATION_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a base solution and an additional reference solution.
- You need to implement your Python solution by integrating reference solution to the base solution.

# Base solution
```python
{base_code}
```

# Reference solution
```python
{reference_code}
```

# Your task
- Implement the solution in Python.
- You have to integrate the reference solution to the base solution.
- Your code base should be the base solution.
- Try to train additional model of the reference solution.
- When integrating, try to keep code with similar functionality in the same place (e.g., all preprocessing should be done and then all training).
- When integrating, ensemble the models.
- The solution design should be relatively simple.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data in the `./input` directory.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior."""

CHECK_DATA_USE_INSTR = """I have provided Python code for a machine learning task (attached below):
# Solution Code
```python
{code}
```

# Task description
{task_description}

# Your task
If the above solution code does not use the information provided, try to incorporate all. Do not bypass using try-except.
DO NOT USE TRY and EXCEPT; just occur error so we can debug it!
See the task description carefully, to know how to extract unused information effectively.
When improving the solution code by incorporating unused information, DO NOT FORGET to print out 'Final Validation Performance: {{final_validation_score}}' as in original solution code.

Response format:
Option 1: If the code did not use all the provided information, your response should be a single markdown code block (wrapped in ```) which is the improved code block. There should be no additional headings or text in your response.
Option 2: If the code used all the provided information, simply state that "All the provided information is used."
"""
