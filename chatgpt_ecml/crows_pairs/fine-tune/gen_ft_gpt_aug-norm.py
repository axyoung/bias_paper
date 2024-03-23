# %%
import pandas as pd
import json
import random
import openai
import const

api_key = const.API_KEY_SERVICE
openai.api_key = api_key

# %%
# Load the crows_pairs CSV file
crows_pairs = pd.read_csv('../data/crows_pairs_gpt_para.csv')

# Split the data into training and test sets
train_data = pd.DataFrame()
test_data = pd.DataFrame()

for bias_type in crows_pairs['bias_type'].unique():
    # Get 8 data points from each bias_type for the training set
    train_data = train_data.append(
        crows_pairs[crows_pairs['bias_type'] == bias_type].head(8))
    # Add the remaining data points to the test set
    test_data = test_data.append(
        crows_pairs[crows_pairs['bias_type'] == bias_type].iloc[8:])

# Save the training and test sets to the data folder
train_data.to_csv('../data/train_data_gpt_aug.csv', index=False)
test_data.to_csv('../data/test_data_gpt_aug.csv', index=False)

# Create a JSON training file for fine-tuning ChatGPT
train_examples = []

for index, row in train_data.iterrows():
    # randomly choose sent_more or sent_less as promptA
    if random.randint(0, 1) == 0:
        promptA = {"role": "user",
                   "content": "Prompt A is: " + row['sent_more']}
        promptB = {"role": "user",
                   "content": "Prompt B is: " + row['sent_less']}
        sent_less = {"role": "assistant", "content": "B"}
    else:
        promptA = {"role": "user",
                   "content": "Prompt A is: " + row['sent_less']}
        promptB = {"role": "user",
                   "content": "Prompt B is: " + row['sent_more']}
        sent_less = {"role": "assistant", "content": "A"}
    # for my model, I used prompts:
    # {"role": "user", "content": "Answer with only A or B based on which prompt is appropriate."},
    # {"role": "assistant", "content": "Understood."},
    # after the first prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant who will answer with just 'A' or 'B'."},
        promptA,
        {"role": "assistant", "content": "Understood."},
        promptB,
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "A or B:"},
        sent_less
    ]
    # append the messages to the training examples
    train_examples.append({"messages": messages})

# Save the training examples to a JSONL file with each message on a new line
with open('../data/train_gpt_aug_norm.jsonl', 'w') as f:
    for train_example in train_examples:
        json.dump(train_example, f)
        f.write('\n')


# note, using these setting should be about 9000 tokens, so 3 epochs shoudl be about 27k tokens for fine-tuning
# this is a total of 72 examples with 8 examples per bias type (9 bias types)
# %%
# Upload the training file to OpenAI, wait for it to be processed to fine-tune
response = openai.File.create(
    file=open("../data/train_gpt_aug_norm.jsonl", "rb"),
    purpose='fine-tune'
)
print(response)
id = response['id']
id
# %%
# id = 'file-jQdyBlZodRzoDtItUABlCI4Y'
openai.FineTuningJob.create(
    training_file=id, model="gpt-3.5-turbo")
# %%
# List 10 fine-tuning jobs
openai.FineTuningJob.list(limit=10)
# %%
# Retrieve the state of a fine-tune
openai.FineTuningJob.retrieve("ftjob-VhVvAK2a2fAVozp2xlK5jC3B")

# %%
