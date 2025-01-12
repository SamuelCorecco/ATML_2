import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

def main():
    # -------------------------------------------------------------------------
    # 1. Load the test data
    # -------------------------------------------------------------------------
    splits = {'test': 'openassistant_best_replies_eval.jsonl'}
    df_test = pd.read_json(
        f"hf://datasets/timdettmers/openassistant-guanaco/{splits['test']}",
        lines=True
    )

    inputs, expecteds = [], []
    for _, row in df_test.iterrows():
        text = row.iloc[0]
        interactions = text.split("###")
        current_prompt = ""
        
        for interaction in interactions:
            interaction = interaction.strip()
            if interaction.startswith("Human:"):
                current_prompt += interaction.replace("Human:", "").strip() + "\n"
            elif interaction.startswith("Assistant:"):
                response = interaction.replace("Assistant:", "").strip()
                inputs.append(current_prompt.strip())
                expecteds.append(response.strip())
                current_prompt = ""

    test_data = pd.DataFrame({"Input": inputs, "Expected": expecteds})
    test_dataset = Dataset.from_pandas(test_data)

    # -------------------------------------------------------------------------
    # 2. Evaluate function
    # -------------------------------------------------------------------------
    def evaluate_model(model_path, model_name):
        """
        Loads a BERT MaskedLM model/tokenizer from `model_path`,
        tokenizes the test dataset, evaluates MLM loss/perplexity,
        and returns the results.
        """
        
        local_tokenizer = AutoTokenizer.from_pretrained(model_path)
        local_model = AutoModelForMaskedLM.from_pretrained(model_path)

        if local_tokenizer.mask_token is None:
            local_tokenizer.add_special_tokens({"mask_token": "[MASK]"})
            local_model.resize_token_embeddings(len(local_tokenizer))

        def local_tokenize_function(examples):
            return local_tokenizer(
                examples["Expected"],
                truncation=True,
                padding="max_length",
                max_length=128
            )

        tokenized_test = test_dataset.map(
            local_tokenize_function, 
            batched=True, 
            remove_columns=["Input", "Expected"]
        )

        # Create data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=local_tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        # Create a Trainer
        eval_args = TrainingArguments(
            output_dir="./tmp-mlm-eval",
            per_device_eval_batch_size=8,
            report_to="none"
        )
        trainer = Trainer(
            model=local_model,
            args=eval_args,
            data_collator=data_collator
        )

        # Evaluate
        eval_metrics = trainer.evaluate(eval_dataset=tokenized_test)

        # Extract results
        if "eval_loss" in eval_metrics:
            test_loss = eval_metrics["eval_loss"]
            perplexity = torch.exp(torch.tensor(test_loss))
            return test_loss, perplexity.item()
        else:
            print(f"No eval_loss found for {model_name}.")
            return None, None

    # -------------------------------------------------------------------------
    # 3. Evaluate Pretrained BERT (bert-base-uncased)
    # -------------------------------------------------------------------------
    pretrained_path = "bert-base-uncased"
    pretrained_loss, pretrained_ppl = evaluate_model(
        pretrained_path,
        "Pretrained BERT (bert-base-uncased)"
    )

    # -------------------------------------------------------------------------
    # 4. Evaluate my Fine-Tuned BERT
    # -------------------------------------------------------------------------
    finetuned_path = "./Models/bert-finetuned-mlm"
    finetuned_loss, finetuned_ppl = evaluate_model(
        finetuned_path,
        "Fine-Tuned BERT"
    )

    # -------------------------------------------------------------------------
    # 5. Print results in a nicely formatted table
    # -------------------------------------------------------------------------
    if (pretrained_loss is not None) and (finetuned_loss is not None):
        table_header = (
            "==========================================================\n"
            "|       Model        |    MLM Loss    |   Perplexity     |\n"
            "----------------------------------------------------------"
        )
        table_pretrained = f"| Pretrained BERT     | {pretrained_loss:.4f}       | {pretrained_ppl:.4f}           |"
        table_finetuned =  f"| Fine-Tuned BERT     | {finetuned_loss:.4f}       | {finetuned_ppl:.4f}             |"
        table_footer = "=========================================================="

        print("\n" + table_header)
        print(table_pretrained)
        print(table_finetuned)
        print(table_footer)

if __name__ == "__main__":
    main()