import getpass
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain.callbacks.base import BaseCallbackHandler
from langchain.globals import set_verbose
from langchain.globals import set_debug
from enum import Enum
import json
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.messages import BaseMessage
from typing import List, Any, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import LLMResult

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

from langchain_groq import ChatGroq

# https://python.langchain.com/v0.1/docs/modules/callbacks/
class CustomCallback(BaseCallbackHandler):

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        self.on_chat_model_start_messages = messages
        self.on_chat_model_start_kwargs = kwargs

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

        self.on_llm_end_response = response
        self.on_llm_end_kwargs = kwargs

class AnimalAgent:

    STATE_DUCK = "duck"
    STATE_FOX = "fox"

    def __init__(self):
        self.llm = ChatGroq(
            model="llama3-8b-8192"
        )
        self.state = AnimalAgent.STATE_DUCK
        self.fox_chain = self.create_fox_chain()
        self.duck_chain = self.create_duck_chain()
        self.text_classifier = self.create_text_classifier()

    def create_fox_chain(self):
        prompt = """You are a fox and have a conversation with a human. You will direct every conversation towards one of these topics. 

* Magnetic Hunting Skills – Foxes can use Earth’s magnetic field to hunt. They often pounce on prey from the northeast, using the magnetic field as a targeting system!
* Cat-Like Behavior – Unlike most canines, foxes can retract their claws, have vertical-slit pupils like cats, and even purr when happy.
* Silent Steps – Foxes have fur-covered footpads that muffle their steps, allowing them to sneak up on prey with ninja-like silence.
* Communicative Tails – Foxes use their bushy tails (called "brushes") to communicate emotions, signal danger, and even cover their noses for warmth in winter.
* Over 40 Different Sounds – Foxes are incredibly vocal and can make an eerie scream, giggle-like chirps, and even sounds that resemble human laughter.
* Jumping Acrobatics – Some foxes, especially fennec foxes and red foxes, can leap over 10 feet in the air to catch prey or escape danger.
* Urban Tricksters – Foxes have adapted well to cities, where they sometimes steal shoes, dig secret stashes of food, and even ride on public transportation!
* Bioluminescent Fur? – Some species of foxes (like the Arctic fox) have been found to glow under UV light, though scientists are still studying why.
* Winter Fur Color Change – Arctic foxes change fur color with the seasons—white in winter for camouflage in the snow, and brown in summer to blend with the tundra.
* Fox Friendships – While foxes are mostly solitary, some form long-lasting bonds and even play with other animals, including dogs and humans.

Follow these rules

* Give short responses of maximal 3 sentences.
"""

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        chain = prompt_template | self.llm | StrOutputParser()    
        return chain

    def create_duck_chain(self):
        prompt = """You are a duck and have a conversation with a human. You will direct every conversation towards one of these topics. 

* Waterproof Feathers – Ducks produce an oil from their uropygial gland (near their tail) that keeps their feathers completely waterproof. Water just rolls right off!
* 360° Vision – Their eyes are positioned on the sides of their heads, giving them nearly a full-circle field of vision. They can see behind them without turning their heads!
* Synchronized Sleeping – Ducks can sleep with one eye open and one side of their brain awake, allowing them to stay alert for predators while resting.
* Quack Echo Mystery – There’s an old myth that a duck’s quack doesn’t echo, but it actually does—just at a pitch and tone that makes it hard to notice.
* Spiral-Shaped Penis – Male ducks have corkscrew-shaped reproductive organs, and some species can extend them in a fraction of a second!
* Feet That Don’t Feel Cold – Ducks’ feet have no nerves or blood vessels in the webbing, so they can stand on ice without feeling the cold.
* Egg-Dumping Behavior – Some female ducks practice "brood parasitism," laying eggs in another duck’s nest to have someone else raise their ducklings.
* Mimicry Skills – Some ducks, like the musk duck, can mimic human speech and other sounds, much like parrots!
* Built-In Goggles – Ducks have a third eyelid (nictitating membrane) that acts like swim goggles, allowing them to see underwater.
* Instant Dabbling – Many ducks are "dabblers," tipping their heads underwater while their butts stick up, searching for food without fully submerging.

Follow these rules

* Give short responses of maximal 3 sentences.
"""

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        chain = prompt_template  |  self.llm  | StrOutputParser()    
        return chain

    def create_text_classifier(self):
        prompt = """Given message to a chatbot, classifiy if the message tells the chatbot to be a duck, a fox or none of these. 
                
* Answer with duck, fox or none.
* Do not respond with more than one word.

<message>
{message}
</message>

Classification:"""
        chain = (
            PromptTemplate.from_template(prompt)
            | ChatGroq(
                model="llama3-8b-8192",
                temperature=0
            )
            | StrOutputParser()
        )
        return chain

    def get_response(self, user_message, chat_history):

        classification_callback = CustomCallback()
        text_classification = self.text_classifier.invoke(user_message, {"callbacks":[classification_callback]})

        if text_classification == "fox":
            self.state = AnimalAgent.STATE_FOX
        elif text_classification == "duck":
            self.state = AnimalAgent.STATE_DUCK

        if self.state == AnimalAgent.STATE_FOX:
            chain = self.fox_chain
        elif self.state == AnimalAgent.STATE_DUCK:
            chain = self.duck_chain

        response_callback = CustomCallback()
        chatbot_response = chain.invoke({"input": user_message, "chat_history": chat_history}, {"callbacks":[response_callback]})

        log_message = {
            "user_message": str(user_message),
            "chatbot_response": str(chatbot_response),
            "agent_state": self.state,
            "classification": {
                "on_chat_model_start_messages": classification_callback.on_chat_model_start_messages,
                "on_chat_model_start_kwargs": classification_callback.on_chat_model_start_kwargs,
                "on_llm_end_response": classification_callback.on_llm_end_response,
                "on_llm_end_kwargs": classification_callback.on_llm_end_kwargs
            },
            "chatbot_response": {
                "on_chat_model_start_messages": response_callback.on_chat_model_start_messages,
                "on_chat_model_start_kwargs": response_callback.on_chat_model_start_kwargs,
                "on_llm_end_response": response_callback.on_llm_end_response,
                "on_llm_end_kwargs": response_callback.on_llm_end_kwargs
            }
        }

        return chatbot_response, log_message

class LogWriter:

    def __init__(self):
        self.conversation_logfile = "conversation.jsonp"
        if os.path.exists(self.conversation_logfile):
            os.remove(self.conversation_logfile)

    # helper function to make sure json encoding the data will work
    def make_json_safe(self, value):
        if type(value) == list:
            return [self.make_json_safe(x) for x in value]
        elif type(value) == dict:
            return {key: self.make_json_safe(value) for key, value in value.items()}
        try:
            json.dumps(value)
            return value
        except TypeError as e:
            return str(value)

    def write(self, log_message):
        with open(self.conversation_logfile, "a") as f:
            f.write(json.dumps(self.make_json_safe(log_message)))
            f.write("\n")
            f.close()

if __name__ == "__main__":

    agent = AnimalAgent()
    chat_history = []
    log_writer = LogWriter()

    while True:
        user_message = input('User: ')
        if user_message.lower() in ['quit', 'exit', 'bye']:
            print('Goodbye!')
            break

        chatbot_response, log_message = agent.get_response(user_message, chat_history)
        print("Bot: " + chatbot_response)

        chat_history.extend([HumanMessage(content=user_message), chatbot_response])
        
        log_writer.write(log_message)