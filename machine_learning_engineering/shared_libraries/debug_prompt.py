"""Defines the prompts for debugging."""

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
- If the error is a 'module not found` error, then install the necessary module. You can use `pip install <module>`, where `<module>` is the name of the module to install.
- Do not remove subsampling if exists.
- Provide the improved, self-contained Python script again.
- There should be no additional headings or text in your response.
- All the provided input data is stored in "./input" directory.
- For California Housing: use train.csv (ID + 8 features + target), split it with train_test_split for validation, and do NOT use test.csv for the validation score.
- Prefer scikit-learn for this tabular task.
- If computing RMSE with scikit-learn >= 1.7, use `sklearn.metrics.root_mean_squared_error`. Do NOT pass `squared=False` to `mean_squared_error`.
- Remember to print a line in the code with 'Final Validation Performance: {{final_validation_score}}' so we can parse performance.
- The code should be a single-file python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the refined Python code."""
