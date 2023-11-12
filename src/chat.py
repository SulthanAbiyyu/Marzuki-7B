import os
import logging
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

logging.basicConfig(level=logging.INFO)

class ChatUB():
    def __init__(self, model: str, args: dict):
        self.args = args
        self.model_path = self.get_model_path(model)

        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        logging.info("Loading model...")
        self.llm = LlamaCpp(
            model_path = self.model_path,
            callback_manager = self.callback_manager,
            verbose=False,
            **self.args
        )
        
        self.prompt = self.prompt_template()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chain = LLMChain(
            llm=self.llm, 
            prompt=self.prompt, 
            memory=self.memory,
            verbose=False
        )
        
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))
        
    def prompt_template(self):
        prompt = ChatPromptTemplate(
            messages = [
                    SystemMessagePromptTemplate.from_template(
                        """Anda adalah chatbot di Universitas Brawijaya. Tugas anda adalah menjawab pertanyaan \
                            dari tamu. Gunakan bahasa yang sopan dan formal."""
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("Pertanyaan:\n{question} \nJawaban:\n")
                ]
        )
        return prompt
    
    def get_model_path(self, model: str):
        base_path = os.getcwd()
        if "work" not in base_path:
            base_path = os.path.join(base_path, "work", "models")
        
        if "models" not in base_path:
            base_path = os.path.join(base_path, "models")
        
        if "merak" in model:
            model_dir = os.path.join(base_path, "merak-7B-v3")
            if "q4" in model:
                model_dir = os.path.join(model_dir, "Merak-7B-v3-model-q4_0.gguf")
            elif "q5" in model:
                model_dir = os.path.join(model_dir, "Merak-7B-v3-model-q5_0.gguf")
            elif "q6" in model:
                model_dir = os.path.join(model_dir, "Merak-7B-v3-model-q6_k.gguf")
            elif "q8" in model:
                model_dir = os.path.join(model_dir, "Merak-7B-v3-model-q8_0.gguf")
            else:
                raise NotImplementedError("Model not found")
        elif "llama2" in model:
            raise NotImplementedError("Model not yet suppported")
        else:
            raise NotImplementedError("Model not found")
        
        del base_path
        return model_dir
    
    def chat(self):
        while True:
            question = input("\nPertanyaan: ")
            if question == "exit":
                break
            else:
                answer = self.chain.run(question=question)
                print(answer)
        

if __name__ == "__main__":
    model = "merak-q8"
    args = {
        "n_gpu_layers":256,
        "n_batch":512,
        "repeat_penalty": 1.2,
        "temperature": 0.9,
        "top_p": 0.9,
    }
    bot = ChatUB(model=model, args=args)
    bot.chat()
    


