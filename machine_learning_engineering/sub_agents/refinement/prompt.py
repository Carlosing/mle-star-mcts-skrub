"""Defines prompts for the refinement agent."""

ABLATION_INSTR = """# Task description
{task_description}

# Current solution code
```python
{code}
```

# Your task
- Perform a quick ablation study on the current solution code.
- Train the full solution and compute its validation score.
- Then create ONE ablated variant by removing or simplifying the most complex preprocessing/feature-engineering block.
- Train the ablated variant and compute its validation score.
- Report both scores.

# Required
- Use only the data in the `./input` directory. Do not use test.csv for validation scores.
- Keep training fast: use a single train_test_split, no cross-validation.
- Print the results in a clear format, for example:
  Original: <score>
  Ablated: <score>
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() in the code.
"""

ABLATION_SUMMARY_INSTR = """# Task description
{task_description}

# Ablation study output
{ablation_output}

# Your task
- Summarize the ablation study results.
- Identify which code blocks have the largest positive or negative impact on the validation score.
- Recommend which block should be improved first and why.

# Required
- Be concise.
- Return only the summary text.
"""

INIT_PLAN_INSTR = """# Task description
{task_description}

# Current solution code
```python
{code}
```

# Ablation summary
{ablation_summary}

# Your task
- Propose a concrete improvement plan for the current solution based on the ablation summary.
- Identify a single code block (code_block) that should be modified.
- Describe the improvement to apply to that block.

# Required
Return a JSON object exactly in this format (and nothing else):
{{"plan": "brief improvement plan", "code_block": "the exact code block to modify"}}
"""

PLAN_REFINE_INSTR = """# Task description
{task_description}

# Current solution code
```python
{code}
```

# Previous plans
{previous_plans}

# Your task
- Propose a new, different concrete improvement plan for the current solution.
- Identify a single code block (code_block) that should be modified.
- The new plan must differ from the previous plans.

# Required
Return a JSON object exactly in this format (and nothing else):
{{"plan": "brief improvement plan", "code_block": "the exact code block to modify"}}
"""

PLAN_IMPLEMENT_INSTR = """# Task description
{task_description}

# Current solution code
```python
{code}
```

# Improvement plan
{plan}

# Code block to modify
```python
{code_block}
```

# Your task
- Implement the improvement plan by modifying only the provided code block.
- Replace the code block in the current solution with your improved version.
- Keep the rest of the solution unchanged.
- The improved solution must still print 'Final Validation Performance: {{final_validation_score}}'.

# Required
- Return only the complete modified solution as a single markdown code block.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Use only train.csv from `./input` for training and validation. Do NOT load test.csv.
- Do NOT change which columns are used as features; keep the same X/y definition as the original code.
- Do NOT use GridSearchCV or nested pipelines unless the original code already uses them correctly.
- Do not use exit() in the code.
"""

FINAL_SELECT_BEST_INSTR = """# Task description
{task_description}

# Available improved solutions
{solutions}

# Your task
- Select the best solution based on the validation scores.
- Return only the complete best solution as a single markdown code block.
"""
