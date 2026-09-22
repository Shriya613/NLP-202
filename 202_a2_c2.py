#PROBLEM 2
#PART 1
# Import required libraries
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from datasets import load_dataset
import json
import subprocess

# Load Covid-QA Dataset
def load_covid_qa_data():
    train_dataset = load_dataset('json', data_files={'train': 'covid-qa/covid-qa-train.json'},cache_dir = None)['train']
    dev_dataset = load_dataset('json', data_files={'dev': 'covid-qa/covid-qa-dev.json'},cache_dir = None)['dev']
    test_dataset = load_dataset('json', data_files={'test': 'covid-qa/covid-qa-test.json'},cache_dir = None)['test']
    return train_dataset, dev_dataset, test_dataset

# Load model and tokenizer
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Inference pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Function to make predictions
def make_predictions(dataset, output_file):
    predictions = {}
    for example in dataset:
        result = qa_pipeline(question=example["question"], context=example["context"])
        predictions[example["id"]] = result["answer"]
    with open(output_file, 'w') as f:
        json.dump(predictions, f)

# Load data
train_dataset, dev_dataset, test_dataset = load_covid_qa_data()

# Generate predictions for dev and test splits
print("Generating predictions for dev split...")
make_predictions(dev_dataset, "predictions_dev.json")
print("Generating predictions for test split...")
make_predictions(test_dataset, "predictions_test.json")

# Evaluate predictions using the provided evaluate.py script
def evaluate_predictions(gold_file, pred_file, eval_file):
    command = f"python3 evaluate.py {gold_file} {pred_file} --out-file {eval_file}"
    subprocess.run(command, shell=True)

print("Evaluating on dev split...")
evaluate_predictions("covid-qa-dev.json", "predictions_dev.json", "evaluation_dev.json")
print("Evaluating on test split...")
evaluate_predictions("covid-qa-test.json", "predictions_test.json", "evaluation_test.json")
# Import required libraries
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline, Trainer, TrainingArguments
from datasets import load_dataset
import json
import subprocess

# Load Covid-QA Dataset
def load_covid_qa_data():
    train_dataset = load_dataset('json', data_files={'train': 'covid-qa/covid-qa-train.json'},cache_dir = None)['train']
    dev_dataset = load_dataset('json', data_files={'dev': 'covid-qa/covid-qa-dev.json'},cache_dir = None)['dev']
    test_dataset = load_dataset('json', data_files={'test': 'covid-qa/covid-qa-test.json'},cache_dir = None)['test']
    return train_dataset, dev_dataset, test_dataset

# Function to make predictions
def make_predictions(qa_pipeline, dataset, output_file):
    predictions = {}
    for example in dataset:
        result = qa_pipeline(question=example["question"], context=example["context"])
        predictions[example["id"]] = result["answer"]
    with open(output_file, 'w') as f:
        json.dump(predictions, f)

# Function to evaluate predictions using the provided evaluate.py script
def evaluate_predictions(gold_file, pred_file, eval_file):
    command = f"python3 evaluate.py {gold_file} {pred_file} --out-file {eval_file}"
    subprocess.run(command, shell=True)

#PART 2
# Preprocess dataset for fine-tuning
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=384,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )
    inputs["start_positions"] = [
        ans_start[0] if len(ans_start) > 0 else 0
        for ans_start in examples["answers"]["answer_start"]
    ]
    inputs["end_positions"] = [
        ans_start[0] + len(ans[0]) if len(ans_start) > 0 else 0
        for ans_start, ans in zip(examples["answers"]["answer_start"], examples["answers"]["text"])
    ]
    return inputs

# Load model and tokenizer
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Inference pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Load Covid-QA data
train_dataset, dev_dataset, test_dataset = load_covid_qa_data()

# Part 1: Baseline Evaluation with Pre-Trained RoBERTa
print("Generating baseline predictions for dev split...")
make_predictions(qa_pipeline, dev_dataset, "baseline_predictions_dev.json")
print("Generating baseline predictions for test split...")
make_predictions(qa_pipeline, test_dataset, "baseline_predictions_test.json")

print("Evaluating baseline model on dev split...")
evaluate_predictions("covid-qa-dev.json", "baseline_predictions_dev.json", "baseline_evaluation_dev.json")
print("Evaluating baseline model on test split...")
evaluate_predictions("covid-qa-test.json", "baseline_predictions_test.json", "baseline_evaluation_test.json")

# Part 2: Fine-Tuning RoBERTa on Covid-QA
print("Preparing dataset for fine-tuning...")
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_dev = dev_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=tokenizer,
)

# Fine-tune the model
print("Fine-tuning RoBERTa on Covid-QA...")
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_roberta")
tokenizer.save_pretrained("./fine_tuned_roberta")

# Fine-tuned Inference Pipeline
fine_tuned_qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Generate predictions with fine-tuned model
print("Generating fine-tuned predictions for dev split...")
make_predictions(fine_tuned_qa_pipeline, dev_dataset, "fine_tuned_predictions_dev.json")
print("Generating fine-tuned predictions for test split...")
make_predictions(fine_tuned_qa_pipeline, test_dataset, "fine_tuned_predictions_test.json")

# Evaluate fine-tuned model
print("Evaluating fine-tuned model on dev split...")
evaluate_predictions("covid-qa-dev.json", "fine_tuned_predictions_dev.json", "fine_tuned_evaluation_dev.json")
print("Evaluating fine-tuned model on test split...")
evaluate_predictions("covid-qa-test.json", "fine_tuned_predictions_test.json", "fine_tuned_evaluation_test.json")
