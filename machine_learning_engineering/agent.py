from machine_learning_engineering import client, MODEL_NAME
from machine_learning_engineering import prompt

class ManagerAgent:
    def __init__(self):
        """This function is executed automatically when the agent is born."""
        self.nombre = "MLE_Frontdoor_Manager"
        self.total_tokens_spent = 0
        self.token_limit = 5000
        
       
        complete_instructions = prompt.SYSTEM_INSTRUCTION + "\n\n" + prompt.FRONTDOOR_INSTRUCTION
        
        self.history = [{"role": "system", "content": complete_instructions}]
        
    def solve_task(self, user_task):
        
        if self.total_tokens_spent >= self.token_limit:
            
            return "Limit tokens reached"
        
        self.history.append({"role" : "user", "content" : user_task})
        
        try:
            
            response = client.chat.completions.create(model = MODEL_NAME, messages = self.history, temperature = .02)
            
            agent_response = response.choices[0].message.content
            
            self.history.append({"role" : "assistant", "content" : agent_response})
            
            self.total_tokens_spent += response.usage.total_tokens
            
            return agent_response
        
        except Exception as e:
            
            return f"Connection error {e}"
        
    def delegate_task(self, employee):
        
        if self.total_tokens_spent >= self.token_limit:
            
            return "Limit tokens reached"
        
        agent_response, cost = employee.work(self.history)
        
        self.total_tokens_spent += cost
        
        self.history.append({"role" : "assistant", "content" : agent_response})
        
        return agent_response
        
        
        
class SubAgent:
    
    def __init__(self, Agent_name, instructions):
        
        self.Agent_name = Agent_name
        
        self.instructions = instructions
    
    def work(self, Manager_history):
        
        temp_history = [{"role" : "system", "content" : self.instructions}]
        
        history = temp_history + Manager_history 
        
        try:
            
            response = client.chat.completions.create(model = MODEL_NAME, messages = history, temperature = .02)
            
            agent_response = response.choices[0].message.content
            
            token_cost = response.usage.total_tokens
            
            return agent_response, token_cost
            
        except Exception as e:
            
            
            
            return f"There was a problem {e} with the execution of the subagent", 0
    
        
        
        
    
     
        
if __name__ == "__main__":
    
    Manager = ManagerAgent()
    
    trial_task = (
        "Make an exploratory analysis of the titanic data set"
    )
        
    Manager.history.append({"role" : "user", "content": trial_task})
    
    kaggle_agent = SubAgent("Kaggle Grandmaster", prompt.TASK_AGENT_INSTR)
    
    output = Manager.delegate_task(kaggle_agent)
    
    print("Expert output: ", output)
    
    print("Tokens consumed: ", Manager.total_tokens_spent)