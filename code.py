import pickle

happy = pickle.load(open('happy.pkl', 'rb'))
unhappy = pickle.load(open('unhappy.pkl', 'rb'))

happy[-1]
unhappy[-1]

opiniones = happy + unhappy
positividad = [1 for _ in range(len(happy))] + [0 for _ in range(len(unhappy))]

###

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

cv = CountVectorizer(binary=True, strip_accents='unicode', lowercase=True, 
					 stop_words=stopwords.words('spanish'))
cv.fit(opiniones)
x = cv.transform(opiniones)

###

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(
    x, positividad, train_size=0.75
)

###

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c, solver='lbfgs')
    lr.fit(train_x, train_y)
    precision = accuracy_score(test_y, lr.predict(test_x))
    print(f"Acierto para C={c}: {precision}")

###

pruebas = [
    'Todo perfecto como siempre gracias',
    'El pedido ha llegado tarde',
    'El pedido lleva mucho plástico'
]

lr = LogisticRegression(C=0.5, solver='lbfgs')
lr.fit(train_x, train_y)

print(sorted(zip(lr.coef_[0], cv.get_feature_names())))

print(lr.predict(cv.transform(pruebas)))
print(sum(lr.predict(cv.transform(unhappy))) / len(unhappy))