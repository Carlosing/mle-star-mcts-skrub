from machine_learning_engineering import client, MODEL_NAME
from machine_learning_engineering import prompt


class ManagerAgent:
    def __init__(self):
        """This function is executed automatically when the agent is born."""
        self.nombre = "MLE_Frontdoor_Manager"
        self.total_tokens_spent = 0
        self.token_limit = 5000

        complete_instructions = (
            prompt.SYSTEM_INSTRUCTION + "\n\n" + prompt.FRONTDOOR_INSTRUCTION
        )

        self.history = [{"role": "system", "content": complete_instructions}]

    def solve_task(self, user_task):

        if self.total_tokens_spent >= self.token_limit:
            return "Limit tokens reached"

        self.history.append({"role": "user", "content": user_task})

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=self.history, temperature=0.02
            )

            agent_response = response.choices[0].message.content

            self.history.append({"role": "assistant", "content": agent_response})

            self.total_tokens_spent += response.usage.total_tokens

            return agent_response

        except Exception as e:
            return f"Connection error {e}"

    def delegate_task(self, employee):

        if self.total_tokens_spent >= self.token_limit:
            return "Limit tokens reached"

        agent_response, cost = employee.work(self.history)

        self.total_tokens_spent += cost

        self.history.append({"role": "assistant", "content": agent_response})

        return agent_response


class SubAgent:
    def __init__(self, Agent_name, instructions, use_web_search=False):

        self.Agent_name = Agent_name

        self.instructions = instructions

        # Opt-in provider-native web search. This is the OpenAI stack's
        # counterpart to the Gemini stack's google_search tool (adk_agent.py).
        # It uses the OpenAI Responses API's built-in web_search tool, which
        # requires a REAL OpenAI endpoint (api.openai.com) + key + model — it is
        # NOT available through AI Studio's OpenAI-compatible endpoint. See
        # docs/agent-architecture.md. Default False keeps the original flow.
        self.use_web_search = use_web_search

    def work(self, Manager_history):

        if self.use_web_search:
            return self._work_with_search(Manager_history)

        temp_history = [{"role": "system", "content": self.instructions}]

        history = temp_history + Manager_history

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=history, temperature=0.02
            )

            agent_response = response.choices[0].message.content

            token_cost = response.usage.total_tokens

            return agent_response, token_cost

        except Exception as e:
            return f"There was a problem {e} with the execution of the subagent", 0

    def _work_with_search(self, Manager_history):
        """Answer with native OpenAI web search (Responses API).

        Mirrors the Gemini stack's google_search analyst. Returns the same
        (text, token_cost) contract as work(). Fails with a clear message if the
        configured endpoint/model does not expose the built-in web_search tool
        (e.g. an OpenAI-compatible proxy rather than real OpenAI).
        """
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                instructions=self.instructions,
                tools=[{"type": "web_search"}],
                input=Manager_history,
            )

            agent_response = response.output_text

            token_cost = response.usage.total_tokens

            return agent_response, token_cost

        except Exception as e:
            return (
                f"There was a problem {e} with the web-search subagent "
                "(native web_search needs a real OpenAI endpoint/key/model)",
                0,
            )


if __name__ == "__main__":
    print("🚀 Starting the SAIA Engine - Multi-Agent Factory...")

    # 1. Nace el Gerente
    Manager = ManagerAgent()

    # 2. Definimos el problema base y se lo pasamos al gerente
    trial_task = "We have the Titanic dataset. We want to build a machine learning pipeline to predict who survived."
    Manager.history.append({"role": "user", "content": trial_task})

    # 3. Redactamos las reglas secretas en inglés para cada fase
    init_prompt = (
        "You are the Initialization Agent. Your task is to read the user's problem "
        "and outline a clear, short 3-step plan to analyze the dataset."
    )

    refine_prompt = (
        "You are the Refinement Agent. Based strictly on the Initialization Agent's plan, "
        "suggest 3 concrete data cleaning steps (e.g., handling missing values, encoding features)."
    )

    ensemble_prompt = (
        "You are the Ensemble Agent. Based on the cleaned data strategy from the previous steps, "
        "suggest 2 specific Machine Learning models that are well-suited for this classification task and briefly explain why."
    )

    submit_prompt = (
        "You are the Submission Agent. Your job is to output the final result. "
        "Write a 1-paragraph executive summary of the entire pipeline designed by your peers, "
        "and provide a professional sign-off."
    )

    # 4. Instanciamos a los 4 empleados
    agent_init = SubAgent("Initialization Expert", init_prompt)
    agent_refine = SubAgent("Refinement Expert", refine_prompt)
    agent_ensemble = SubAgent("Ensemble Expert", ensemble_prompt)
    agent_submit = SubAgent("Submission Expert", submit_prompt)

    # 5. El Gerente delega el trabajo en cadena (Pipeline Secuencial)
    print("\n--- PHASE 1: Initialization ---")
    out_1 = Manager.delegate_task(agent_init)
    print(out_1)

    print("\n--- PHASE 2: Refinement ---")
    out_2 = Manager.delegate_task(agent_refine)
    print(out_2)

    print("\n--- PHASE 3: Ensemble ---")
    out_3 = Manager.delegate_task(agent_ensemble)
    print(out_3)

    print("\n--- PHASE 4: Submission ---")
    out_4 = Manager.delegate_task(agent_submit)
    print(out_4)

    # 6. Auditoría Final
    print("=" * 60)
    print(f"💰 Final session balance: {Manager.total_tokens_spent} tokens spent.")
