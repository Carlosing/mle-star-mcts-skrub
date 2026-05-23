from machine_learning_engineering import client, MODEL_NAME
from machine_learning_engineering import prompt

class GerenteAgent:
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
        
if __name__ == "__main__":
    
    Manager = GerenteAgent()
    
    trial_task = "Write a simple code in python that prints the message 'Hello World!'"
    
    output = Manager.solve_task(trial_task)
    
    print("Agents output: ", output)
    
    print(f"{Manager.total_tokens_spent} spent tokens")