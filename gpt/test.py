from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Chemin vers votre modèle personnalisé
model_path = "/home/achraf/Desktop/mymodel"

# Charger le modèle et le tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Mode interactif pour tester le modèle
print("=== Tester votre modèle fine-tuné GPT-2 ===")
print("Posez une question ou donnez un texte lié à Linux (tapez 'exit' pour quitter)")

while True:
    # Entrée utilisateur
    user_input = input("\nVotre question : ")
    
    if user_input.lower() == "exit":
        print("Sortie du test. Merci d'avoir utilisé le modèle.")
        break

    # Encoder l'entrée utilisateur
    inputs = tokenizer.encode(user_input, return_tensors="pt")

    # Générer une réponse
    outputs = model.generate(
        inputs,
        max_length=150,  # Longueur maximale de la réponse
        num_return_sequences=1,  # Nombre de réponses générées
        no_repeat_ngram_size=2,  # Évite de répéter des n-grammes
	do_sample=True,
        top_k=50,  # Filtrage des top-k tokens
        top_p=0.95,  # Probabilité cumulative pour nucleus sampling
        temperature=0.7,  # Contrôle la créativité de la génération
        pad_token_id=tokenizer.eos_token_id
    )

    # Décoder la réponse générée
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Réponse du modèle : {response}")
