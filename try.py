# Import the necessary modules
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

# Define the question and the passage
question = "What is the name of the library that provides the model?"
text = "HuggingFace Transformers is a library that provides state-of-the-art natural language processing models for various tasks, such as question answering, text classification, text generation, and more."

# Tokenize the inputs and prepare the tensors
inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]

# Get the model outputs and the answer span
outputs = model(**inputs)
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits)

# Convert the tokens to words and join them into the answer
answer = tokenizer.convert_tokens_to_ids(input_ids[answer_start:answer_end+1])
answer = tokenizer.decode(answer)
print("The answer is:", answer)
