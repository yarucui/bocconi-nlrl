# External imports
from datasets import Dataset
import json
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, default_data_collator
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
            trust_remote_code=True,
        )
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.name,
            quantization_config=self.bnb_config,
            device_map='auto',
            trust_remote_code=True
        )
        #
        # The Mistral model is quantized so we have to use a LoRA adapter.
        #
        base_model = prepare_model_for_kbit_training(base_model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(base_model, lora_config)

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
            **self.config['generate'],
            pad_token_id=self.tokenizer.eos_token_id # supresses a warning message
        )
        #
        # Seperate the response from the instruction
        #
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split('[/INST] ')[1]
        #
        # Return the response
        #
        return response
    
    #
    # Given data, fine-tune the model.
    #
    def train(self, data : str) -> None:
        #
        # Build a Hugging Face dataset
        #
        dataset = Dataset.from_list(data)
        #
        # Preprocessing function to help format the chat tokens for training.
        #
        def preprocess(example):
            #
            # Build the full chat string
            #
            full = (
                "<s>[SYS] " + example["system_prompt"] + " [/SYS]\n"
                + example["user_prompt"] + " [/INST]\n"
                + example["response"]
            )
            #
            # Tokenize the full chat string
            #
            tokenized = self.tokenizer(
                full,
                truncation=True,
                max_length=1024,
                padding="max_length",
            )
            #
            # Find the special token that seperates the prompt tokens from the
            # response token
            #
            seperation_idx = tokenized["input_ids"].index(self.tokenizer.eos_token_id, 1)
            #
            # Mask out the prompt tokens in the label tokens field.
            #
            labels = tokenized["input_ids"].copy()
            for i in range(seperation_idx + 1):
                labels[i] = -100
            tokenized["labels"] = labels
            #
            # Return tokenized chat
            #
            return tokenized
        #
        # Tokenize the dataset
        #
        tokenized_ds = dataset.map(
            preprocess,
            batched=False,
            remove_columns=dataset.column_names
        )
        #
        # Set training arguments
        #
        training_args = training_args = TrainingArguments(
            output_dir="./outputs",
            per_device_train_batch_size=1,     # try 1–2
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,     # effective batch size ≈ 8
            num_train_epochs=3,
            learning_rate=2e-5,
            warmup_ratio=0.03,                 # ~3% of steps
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            fp16=True,                         # or bf16 if supported
            optim="adamw_torch",               # the new PyTorch optimizer
            gradient_checkpointing=True,       # save memory
            label_names=["labels"]
        )
        #
        # Train
        #
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=tokenized_ds,
            data_collator=default_data_collator,
        )
        trainer.train()


