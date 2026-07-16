"""Defines prompts for the ensemble agent."""

INIT_ENSEMBLE_PLAN_INSTR = """# Task description
{task_description}

# Candidate solutions
{solutions}

# Your task
- Propose a concrete plan to combine the candidate solutions into a single stronger ensemble.
- The ensemble should train each candidate on the full training data and blend their predictions.
- Prefer simple and robust strategies such as averaging or weighted averaging based on validation scores.
- Use ONLY the features and columns that are already used in the candidate solutions. Do NOT add columns that are not present in train.csv.

# Required
Return a JSON object exactly in this format (and nothing else):
{{"plan": "brief ensemble plan"}}
"""

ENSEMBLE_PLAN_REFINE_INSTR = """# Task description
{task_description}

# Candidate solutions
{solutions}

# Previous plans
{previous_plans}

# Your task
- Propose a new, different ensemble plan that may outperform the previous ones.
- Prefer simple and robust strategies.

# Required
Return a JSON object exactly in this format (and nothing else):
{{"plan": "brief ensemble plan"}}
"""

ENSEMBLE_PLAN_IMPLEMENT_INSTR = """# Task description
{task_description}

# Ensemble plan
{plan}

# Candidate solutions
{solutions}

# Your task
- Implement the ensemble plan as a single Python program.
- Load the data from `./input` directory.
- Train each candidate on the full training data, generate predictions for a hold-out validation set, and compute the ensemble prediction.
- Print 'Final Validation Performance: {{final_validation_score}}' with the ensemble validation score.

# Required
- Return only the complete ensemble code as a single markdown code block.
- The code should be self-contained and executable as-is.
- Use ONLY the columns that exist in train.csv. Do NOT add or assume columns such as 'ocean_proximity'.
- Do not use exit() in the code.
"""
