# %%
import pandas as pd
import json
import random
import openai
import const

api_key = const.API_KEY_SERVICE
openai.api_key = api_key

# %%
# Load the stereoset CSV file
stereoset = pd.read_csv('../data/intersentence_df.csv')

# Split the data into training and test sets
train_data = pd.DataFrame()
test_data = pd.DataFrame()

for bias_type in stereoset['bias_type'].unique():
    # Get 20 data points from each bias_type for the training set
    train_data = train_data.append(
        stereoset[stereoset['bias_type'] == bias_type].head(20))
    # Add the remaining data points to the test set
    test_data = test_data.append(
        stereoset[stereoset['bias_type'] == bias_type].iloc[20:])

# Save the training and test sets to the data folder
train_data.to_csv('../data/train_data.csv', index=False)
test_data.to_csv('../data/test_data.csv', index=False)


# Create a JSON training file for fine-tuning ChatGPT
train_examples = []

for id, row in train_data.iterrows():
    context = {"role": "user", "content": row['context']}
    promptA = {"role": "user", "content": "A: " +
               row['sentence_A']}
    promptB = {"role": "user", "content": "B: " +
               row['sentence_B']}
    promptC = {"role": "user", "content": "C: " +
               row['sentence_C']}
    # if in stereoset 'label_A' is 'stereotype' then save stereotype as A
    if row['label_A'] == 'stereotype':
        answer = {"role": "assistant", "content": "A"}
    elif row['label_B'] == 'stereotype':
        answer = {"role": "assistant", "content": "B"}
    elif row['label_C'] == 'stereotype':
        answer = {"role": "assistant", "content": "C"}
    # initially did not have the two prompts - answer with only... understood
    messages = [
        {"role": "system", "content": "You are a helpful assistant who will answer with just 'A', 'B', or 'C'."},
        context,
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "Answer with only A, B or C based on which prompt is a stereotype."},
        {"role": "assistant", "content": "Understood."},
        promptA,
        {"role": "assistant", "content": "Understood."},
        promptB,
        {"role": "assistant", "content": "Understood."},
        promptC,
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "Which prompt is a stereotype: A, B or C?"},
        answer
    ]
    # append the messages to the training examples
    train_examples.append({"messages": messages})

# Save the training examples to a JSONL file with each message on a new line
with open('../data/train.jsonl', 'w') as f:
    for train_example in train_examples:
        json.dump(train_example, f)
        f.write('\n')


# note, using these setting should be about 9000 tokens, so 3 epochs shoudl be about 27k tokens for fine-tuning
# this is a total of 72 examples with 8 examples per bias type (9 bias types)
# %%
# Upload the training file to OpenAI, wait for it to be processed to fine-tune
response = openai.File.create(
    file=open("../data/train.jsonl", "rb"),
    purpose='fine-tune'
)
print(response)
id = response['id']
id
# %%
# id = 'file-50ioCQ9n5Svvi5bUqIchPrIK'
openai.FineTuningJob.create(
    training_file=id, model="gpt-3.5-turbo")
# %%
# List 10 fine-tuning jobs
openai.FineTuningJob.list(limit=10)
# %%
# Retrieve the state of a fine-tune
openai.FineTuningJob.retrieve("ftjob-Ry8NtDZ01kQUbIkgxoUrpf7U")

# %%
