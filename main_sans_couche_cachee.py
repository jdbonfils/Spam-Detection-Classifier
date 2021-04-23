import numpy as np # linear algebra
import keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
import math
from keras import backend as K
import matplotlib.pyplot as plt

#Fonction permettant d'extraire les donnes de puis un fichier CSV
def getXYData(fic,max_col=-1,max_lines=-1,indLabel=0,delimiter=";"):
	Y = []
	X = []
	#Ouverture du fichier
	try:
		f = open(fic,"r")
		lines = f.readlines()
	except IOError:
		print("Veuillez verifier le chemin spécifié")
		return None,None
	if(indLabel == 0 and max_col != -1):
		max_col += 1
		
	#Verification des parametres
	if(max_lines>len(lines) or max_lines == -1 ):
		max_lines = len(lines)
	if(max_col > len(lines[0].split(";")) or max_col == -1 ):
		max_col = len(lines[0].split(";"))

	#Recuperation des donnees
	for line in lines[1:max_lines]:
		cases = line.split(";")
		#Recupere le Label
		if(cases[indLabel].strip() == 'Yes' or cases[indLabel].strip() == "1"):
			Y.append(1)
		else:
			Y.append(0)
		#Recupere les attributs
		tmp = []
		for i in range(0, max_col):
			if(i != indLabel):
				tmp.append(float(cases[i].rstrip()))
		X.append(tmp)
	return X,Y
	

#Binary Cross-Entropy
def binary_crossentropy(y_true, y_pred): 
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    false = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  
    true = y_true * K.log(y_pred + K.epsilon())
    return -K.mean(true + false, axis=1)

if __name__ == "__main__":
	
	#Nombre d'époque à modifier en fonction du contexte
	epochs = 250

	#indLabel permet de specifier la colonne corespondant au label
	#max lines le nombre de lignes à prendre en compte 
	#On definit le nombre de ligne et colonnes à prendre en compte
	nbColonnes = 57 #max 57 min au moins 5
	nbLignes = -1 # -1 = toutes les lignes
	#Données de train
	X_train,Y_train = getXYData("donnees/dataTrain.csv",max_lines=nbLignes,max_col=nbColonnes,indLabel=0,delimiter=";")
	#Données de tests
	X_test,Y_test = getXYData("donnees/dataTest.csv",indLabel=57,max_col=nbColonnes,delimiter=";")
	#Données à prédire
	X_pred,Y_pred = getXYData("donnees/dataPred.csv",indLabel=57,max_col=nbColonnes,delimiter=";")

	#On convertie les labels au format One Hot -> Yes: [0,1], No,[1,0] outputshape doit donc etre de la forme (N,2)
	#Et les liste en numyp array
	Y_train = keras.utils.to_categorical(Y_train, 2)
	X_train = np.array(X_train)

	Y_test = keras.utils.to_categorical(Y_test, 2)
	X_test = np.array(X_test)

	model = Sequential()
	#Modele sans couche cachée avec seulement la fonction d'activation sigmoid
	#Input=dim = 57 ccorrespond à l'input shape qui correspond aux nombres d'attributs dans le fichier csv
	model.add(Dense(2, input_dim=nbColonnes, activation='sigmoid'))
	#Output shape doit etre de taille 2 pour correspondre à une classification binaire avec le format One Hot

	#Compilation du modele en utilisant BinaryCrossentropy (recommandé pour une classification binaire)
	model.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=['accuracy'])

	#Entrainement du modèle
	historique = model.fit(X_train, Y_train,epochs=epochs,verbose=1,validation_data=(X_test,Y_test))

	#Affiche le résume du reseau de neuronnes
	model.summary()
	score = model.evaluate(X_train, Y_train, verbose=0)

	print("Evaluation terminée : \n")
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	#Affichage du taux de loss grace à un graphique
	statEpoque = historique.history
	choix = input("\n Voulez vous afficher le graphique résumant le loss en fonction des époques ? Y/N : ")
	if(choix == "Y" or choix == "y"):
		loss = np.array(statEpoque['loss'])
		valLoss = np.array(statEpoque['val_loss'])
		ep = range(0,epochs)
		print(loss.shape)
		print(valLoss.shape)
		plt.plot(ep,loss,'b',color='red',linewidth=2,label='Training loss')
		plt.plot(ep,valLoss,'b',color='green',linewidth=2,label='Validation loss')
		plt.title('Evolution du loss en fonction des époques')
		plt.xlabel('Epoques')
		plt.ylabel('Loss')
		plt.legend()
		plt.show()
	#Affichage du taux de l'accuracy grace à un graphique
	choix = input("\n Voulez vous afficher le graphique résumant l'accuracy en fonction des époques ? Y/N : ")
	if(choix == "Y" or choix == "y"):
		loss = np.array(statEpoque['accuracy'])
		valLoss = np.array(statEpoque['val_accuracy'])
		ep = range(0,epochs)
		print(loss.shape)
		print(valLoss.shape)
		plt.plot(ep,loss,'b',color='red',linewidth=2,label='Training accuracy')
		plt.plot(ep,valLoss,'b',color='green',linewidth=2,label='Validation accuracy')
		plt.title("Evolution de l'accuracy en fonction des époques")
		plt.xlabel('Epoques')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.show()

	input("\n Apuyer sur entré pour passer à la phase de prédiction...")

	#Une fois que l'entrainement est terminé, on essaie de prédire les valeurs
	guesses = model.predict_classes(np.array(X_pred))

	ok = 0 
	#Pour chaque tuple on regarde si la valeur trouvée par le modèle est conforme à la valeur réelle
	for i in range(len(guesses)):
		print("Prediction : "+str(guesses[i])+" --- Realite : "+str(Y_pred[i])+"\n")
		if(guesses[i] == Y_pred[i]):
			ok += 1

	print("Nombre de predictions correctes :"+ str(ok))
	print("Accuracy :"+ str(ok/len(guesses)))
