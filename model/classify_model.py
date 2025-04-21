import gc
import re
import os
from huggingface_hub import login
from unsloth import FastLanguageModel
from dotenv import load_dotenv
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

# Max input token size
MAX_SEQ_LENGTH = 65536
DTYPE = None
LOAD_IN_4BIT = True
HF_TOKEN = os.getenv("hf_token")


# Download the distilled llm
def transformer_model_init():
    login(HF_TOKEN)

    # Download model from hugging face
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        token=HF_TOKEN,
    )

    # Set the inference mode
    FastLanguageModel.for_inference(model)

    return model, tokenizer


# Classify one question
def question_classify(model, tokenizer, text):
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    # Get the response tokens from the model
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )

    # Decode the tokens and cut out the response text
    response = tokenizer.batch_decode(outputs)
    return response[0].split("### Response:")[1]


# Use regular expression to get the classifation result
def get_label_from_response(text):
    res = text.split("### Response:")[1]

    match = re.search(r"Final classification ID:\s*(\d+)", text)
    if match:
        return match.group(1), res
    else:
        return 0, res


# Batch classify the questions
def question_classify_batch(model, tokenizer, texts, max_tokens=2048):
    try:
        with torch.no_grad():
            # tokenize the input text
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            ).to("cuda")

            # Set do_sample=Flase, to avoid randomness
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                use_cache=True,
                do_sample=False,
            )

            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Manually release the cache
            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()

            return [get_label_from_response(r) for r in responses]
    except Exception as e:
        print(f"[❌ Batch Error] {e}")
        return ["0"] * len(texts)


# Do the classification on the given dataset
def classify_all(
    data, model, tokenizer, prompt_style, batch_size=6, max_chars=2000, max_tokens=1024
):
    texts = []
    q_ids = []

    # If the input is too long, cut it to the length of max_chars
    for _, row in data.iterrows():
        question_text = row.get("cleaned_text")
        question_text = question_text[:max_chars]
        prompt = prompt_style.format(question_text, "")
        texts.append(prompt)
        q_ids.append(row.get("q_id", None))

    classification_ids = []
    full_responses = []

    # Use tqdm to visualize the process
    for i in tqdm(
        range(0, len(texts), batch_size), desc="Classifying", mininterval=1.5
    ):
        batch_texts = texts[i : i + batch_size]
        batch_results = question_classify_batch(
            model, tokenizer, batch_texts, max_tokens=max_tokens
        )

        # Store the full response, and category id from the model
        for label, res in batch_results:
            classification_ids.append(label)
            full_responses.append(res)

    return pd.DataFrame(
        {
            "q_id": q_ids,
            "classification_id": classification_ids,
            "response_text": full_responses,
        }
    )


# Use bar chart to visualize the result
def result_visualize(classified_df):
    sns.countplot(
        data=classified_df,
        x="classification_id",
        hue="classification_id",
        palette="viridis",
        legend=False,
    )
    plt.title("Distribution of Classification IDs")
    plt.xlabel("Classification ID")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def print_classification_distribution(classified_df):
    distribution = classified_df["classification_id"].value_counts().sort_index()
    print("Classification ID Distribution:")
    print(distribution)
