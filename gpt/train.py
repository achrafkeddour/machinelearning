from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Charger le tokenizer et le modèle pré-entraîné
model_name = "gpt2-medium"  # Vous pouvez utiliser "gpt2-medium" si vous avez plus de ressources
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Charger les données d'entraînement
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )

# Préparer les données
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Pas de "Masked Language Model" pour GPT-2
)
training_file_path = "traintexte.txt"  # Chemin de votre fichier d'entraînement
dataset = load_dataset(training_file_path, tokenizer)

# Configurer l'entraînement
output_dir = "/home/achraf/Desktop/mymodel"  # Remplacez par le chemin où sauvegarder le modèle
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=100,  # Augmentez si besoin
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Lancer l'entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

# Sauvegarder le modèle
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Modèle et tokenizer sauvegardés dans : {output_dir}")
