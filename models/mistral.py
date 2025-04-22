# External imports
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Internal imports
from models.model import LanguageModel

class Mistral(LanguageModel):
    #
    # Load the model and set the desired configuration parameters.
    #
    def __init__(self, config : str):
        #
        # Load the model configuration
        #
        with open(config, 'r') as file:
            self.config = json.load(file)
        #
        # Configure 4-bit quantization to fit in 8GB VRAM
        #
        self.config['bits_and_bytes']['bnb_4bit_compute_dtype'] = eval(self.config['bits_and_bytes']['bnb_4bit_compute_dtype'])
        self.bnb_config = BitsAndBytesConfig(
            **self.config['bits_and_bytes']
        )
        #
        # Load the model and tokenizer
        #
        self.name = self.config['name']
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            quantization_config=self.bnb_config,
            device_map='auto',
            trust_remote_code=True
        )

    #
    # Given strings with the user and system prompts, query the LLM
    # and return the response.
    #
    def generate_response(self, system_prompt : str, user_prompt : str) -> str:
        #
        # Convert prompt strings into the huggingface message format
        #
        messages = self.prompts_to_messages(system_prompt, user_prompt)
        #
        # Format the messages into Mistral's format
        #
        mistral_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        #
        # Tokenize the prompt
        #
        inputs = self.tokenizer(mistral_prompt, return_tensors='pt').to('cuda')
        #
        # Perform the query
        #
        outputs = self.model.generate(
            **inputs,
            **self.config['generate']
        )
        #
        # Return the response
        #
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

