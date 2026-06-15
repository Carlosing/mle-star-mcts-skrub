"""Minimal OpenAI-compatible runner that replaces the ADK runtime."""

from machine_learning_engineering import client, MODEL_NAME


class AgentState(dict):
    """Dictionary-like state object that replaces ADK's callback_context.state."""

    def get(self, key, default=None):
        return super().get(key, default)

    def to_dict(self):
        return dict(self)


def llm_call(messages, temperature=0.0, model=None):
    """Call the OpenAI-compatible chat completions endpoint.

    Args:
        messages: List of dicts with 'role' and 'content' keys.
        temperature: Sampling temperature.
        model: Optional model override. Defaults to MODEL_NAME.

    Returns:
        OpenAI chat completion response object.
    """
    return client.chat.completions.create(
        model=model or MODEL_NAME,
        messages=messages,
        temperature=temperature,
    )


def _resolve_instruction(instruction, state, agent_name):
    """Resolve instruction to a string if it is callable."""
    if callable(instruction):
        try:
            return instruction(state, agent_name)
        except TypeError:
            return instruction(state)
    return instruction


def run_agent(
    state,
    agent_name,
    instruction,
    before_model=None,
    after_model=None,
    temperature=0.0,
):
    """Execute a single LLM agent step.

    Args:
        state: AgentState instance.
        agent_name: Unique string identifier for this agent.
        instruction: String prompt or callable(state, agent_name) -> str.
        before_model: Optional callable(state, agent_name) -> response | None.
            If it returns a non-None value, the LLM call is skipped and that
            value is used as the response.
        after_model: Optional callable(state, agent_name, response) -> None.
        temperature: Sampling temperature.

    Returns:
        The raw LLM response object.
    """
    response = None

    if before_model:
        response = before_model(state, agent_name)

    if response is None:
        instruction_str = _resolve_instruction(instruction, state, agent_name)
        messages = [{"role": "user", "content": instruction_str}]
        response = llm_call(messages, temperature=temperature)

    # Track token spend if usage is available.
    usage = getattr(response, "usage", None)
    if usage:
        total = getattr(usage, "total_tokens", 0)
        state["total_tokens_spent"] = state.get("total_tokens_spent", 0) + total

    if after_model:
        after_model(state, agent_name, response)

    return response


def run_sequential(state, agents, description=""):
    """Run a list of agent definitions sequentially.

    Args:
        state: AgentState instance.
        agents: List of dicts accepted by run_agent (must include 'agent_name'
            and 'instruction'; other keys are passed as kwargs).
        description: Optional description for logging.

    Returns:
        None.
    """
    for agent in agents:
        agent_name = agent["agent_name"]
        instruction = agent["instruction"]
        kwargs = {k: v for k, v in agent.items() if k not in ("agent_name", "instruction")}
        run_agent(state, agent_name, instruction, **kwargs)


def run_loop(state, agent, max_iterations=1, continue_fn=None):
    """Run an agent repeatedly up to max_iterations or until continue_fn returns False.

    Args:
        state: AgentState instance.
        agent: Dict accepted by run_agent.
        max_iterations: Maximum number of iterations.
        continue_fn: Optional callable(state, iteration_index) -> bool.

    Returns:
        None.
    """
    for i in range(max_iterations):
        run_agent(state, **agent)
        if continue_fn and not continue_fn(state, i):
            break


def run_parallel(state, agents, description=""):
    """For the MVP, run agents sequentially to save tokens and simplify debugging.

    In the future this can be replaced with true parallel execution.

    Args:
        state: AgentState instance.
        agents: List of dicts accepted by run_agent.
        description: Optional description for logging.

    Returns:
        None.
    """
    run_sequential(state, agents, description=description)
