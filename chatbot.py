from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

# Creating ChatBot Instance
chatbot = ChatBot(
    'ChatBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.BestMatch',
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'Lo siento pero no lo entiendo, seguire aprendiendo',
            'maximum_similarity_threshold': 0.60
        }
    ],
) 

# Entrenamiento con preguntas propias
trainer = ListTrainer(chatbot)

training_data_personal = open('training_data/personal.txt').read().splitlines()

trainer.train(training_data_personal)

# Entrenamiento con corpus propios 
trainer_corpus = ChatterBotCorpusTrainer(chatbot)
trainer_corpus.train('chatterbot.corpus.spanish') 
trainer_corpus.train('training_data/dinero.yml') 
trainer_corpus.train('training_data/emociones.yml') 
trainer_corpus.train('training_data/IA.yml') 
trainer_corpus.train('training_data/perfil.yml') 
trainer_corpus.train('training_data/psicologia.yml')
trainer_corpus.train('training_data/historia.yml')
trainer_corpus.train('training_data/ciencia_cultura.yml')


