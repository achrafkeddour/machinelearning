from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Charger le modèle sauvegardé
output_dir = r"C:\Users\pc cam\OneDrive\Desktop\gpt2-finetuned"  # Chemin où vous avez sauvegardé le modèle
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model = GPT2LMHeadModel.from_pretrained(output_dir)

# Générer du texte avec des données nouvelles
input_text = "Comment changer les permissions d'un fichier sous Linux ?"
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=50, num_return_sequences=1, do_sample=True)

# Afficher le texte généré
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Texte généré :", generated_text)
