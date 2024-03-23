# %%
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(
    "humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained(
    "humarin/chatgpt_paraphraser_on_T5_base")


def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids

    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res


# %%
print(paraphrase("I am a student."))

# %%
print(paraphrase("Women don't know how to drive."))
# %%
# generate a dataset with paraphrases

df = pd.read_csv('../data/crows_pairs.csv')

df = df.drop("annotations", axis='columns')
df = df.drop("anon_writer", axis='columns')
df = df.drop("anon_annotators", axis='columns')

df['Response'] = ''
df['Response_label'] = ''
df['more'] = ''

# total number of rows is 1508
# for each row, generate a paraphrase for sent_more and sent_less and replace the original sentence with the paraphrase
for row in range(1508):
    new_more = paraphrase(df['sent_more'][row])[0]
    new_less = paraphrase(df['sent_less'][row])[0]
    df['sent_more'][row] = new_more
    df['sent_less'][row] = new_less
    print(row)
    df.to_csv('../data/crows_pairs_t5_paraphrased.csv', index=False)

# %%
