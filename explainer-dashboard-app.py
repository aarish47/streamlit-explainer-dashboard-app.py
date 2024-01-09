from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, features_descriptions

# train test split
x_train, y_train, x_test, y_test = titanic_survive()

# creating a model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(x_train, y_train)

explainer = ClassifierExplainer(model, x_test, y_test,
                                cats=['Sex', 'Deck', 'Embarked'],
                                descriptions=features_descriptions,
                                labels=['Not survived', 'Survived'])

ExplainerDashboard(explainer).run()
