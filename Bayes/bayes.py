import csv, math, sys

#Fomule de la densite de probabilite
def densiteProbaLoiNormale(x,variance,esperance):
	if(variance == 0):
		if( esperance != x):
			return 0
		else:
			return 0.000001
	return (1/(math.sqrt(2*math.pi*variance))) * math.exp((-1*(((x-esperance)**2)/(2*variance))))

#Creation d'une liste de dictionnaire à partir des donnees de train provenant du csv 	
with open('dataTrain.csv',newline='') as csvfile:
	trainingData = [{k: v for k, v in row.items()}
        for row in csv.DictReader(csvfile,delimiter=';', quotechar='|', skipinitialspace=True)]

#Par defaut on lit toutes les lignes du fichier de train
nbLigneTrain = len(trainingData)
#Si le nombre de ligne en entré du programme est spécifié
if(len(sys.argv) > 1 and sys.argv[1].isdigit() ):
	#On lira autant de ligne que spécifié
	nbLigneTrain = int(sys.argv[1])

#Initialisation des dictionaires de champs pour la variance et l'esperance de chaque group
esperanceSpam = {}
esperanceNonSpam = {}
varianceSpam = {}
varianceNonSpam = {}

#Calcul des espérances et de la variance pour chaque champs pour les deux groupes différents (spam, non spam)
for X in trainingData[0]:
	#Sauf pour le champs qui correspond aux groupes (le premier champs)
	if X != "GOAL-Spam":
		#Calcul de l'espérance
		sumTmpNonSpam,sumTmpSpam,spam,nonSpam = 0,0,0,0
		for row in trainingData[:nbLigneTrain]:
			if(row["GOAL-Spam"] == "Yes"):
				sumTmpSpam += float(row[X])	
				spam += 1
			else:
				sumTmpNonSpam += float(row[X])
				nonSpam += 1
		esperanceSpam[X] = sumTmpSpam/spam
		esperanceNonSpam[X] = sumTmpNonSpam/nonSpam
		#Calcul de la variance
		sumTmpNonSpam,sumTmpSpam = 0,0
		for row in trainingData[:nbLigneTrain]:
			if(row["GOAL-Spam"] == "Yes"):
				sumTmpSpam += (float(row[X])-esperanceSpam[X])**2
			else:
				sumTmpNonSpam += (float(row[X])-esperanceNonSpam[X])**2
		varianceSpam[X] = sumTmpSpam/(spam-1)
		varianceNonSpam[X] = sumTmpNonSpam/(nonSpam-1)
		#print(X + " Variance spam : "+ str(varianceSpam[X]))
		#print(X + " Variance non spam : "+ str(varianceNonSpam[X]))


#Creation d'une liste de dictionnaire à partir des donnees de test provenant du csv 	
with open('prediction.csv',newline='') as csvfile:
	testData = [{k: v for k, v in row.items()}
        for row in csv.DictReader(csvfile,delimiter=',', quotechar='|', skipinitialspace=True)]

#Calcule de la probabilité posterieur pour le groupe Spam et le groupe non spam
bonneprevision = 0 
for row in testData:
	#On initialise la proba posterieur poour chaque groupe à P(Spam) et P(non Spam)
	probaPostSpam = spam/len(trainingData)
	probaPostNonSpam = nonSpam/len(trainingData)
	#Pour chaque champs
	for X in row:
		if(X != "Spam"):
			#Spam
			probaPostSpam *= densiteProbaLoiNormale(float(row[X]),varianceSpam[X],esperanceSpam[X])
			#Non spam
			probaPostNonSpam *= densiteProbaLoiNormale(float(row[X]),varianceNonSpam[X],esperanceNonSpam[X])
				
	#On affiche la prevision et si il s'agit vraiment d'un spam ou non
	if((probaPostSpam > probaPostNonSpam)) :
		print("Prevision : Spam 1 ---- Realite : Spam " + row["Spam"] )
	else:
		print("Prevision : Spam 0 ---- Realite : Spam " + row["Spam"] )
	#Si la prévision est correcte
	if((probaPostSpam > probaPostNonSpam) == int(row["Spam"])) :
		bonneprevision += 1
print("Taux de bonne prévisions : "+ str(bonneprevision/len(testData)))
