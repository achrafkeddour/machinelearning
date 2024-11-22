from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os

# 1. Charger le tokenizer et le modèle GPT-2 pré-entraîné
model_name = "gpt2"  # ou 'gpt2-medium', 'gpt2-large'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 2. Préparer le jeu de données
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )

# 3. Créer le data collator (gestion de padding)
def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Pour GPT-2, pas de "Masked Language Model"
    )

# 4. Chemin du fichier texte à utiliser pour l'entraînement
training_file_path = "traintexte.txt"  # Remplacez par le chemin de votre fichier texte
dataset = load_dataset(training_file_path, tokenizer)

# 5. Configurer l'entraînement
data_collator = create_data_collator(tokenizer)

output_dir = r"C:\Users\pc cam\OneDrive\Desktop\gpt2-finetuned"
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,  # Ajustez le nombre d'époques
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# 6. Lancer l'entraînement
trainer.train()

# 7. Sauvegarder le modèle et le tokenizer localement
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Modèle et tokenizer sauvegardés dans : {output_dir}")
