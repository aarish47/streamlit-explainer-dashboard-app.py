from sklearn.model_selection import train_test_split
from explainerdashboard.datasets import titanic_survive

# Load the Titanic dataset using the titanic_survive function
x_train, y_train, x_test, y_test = titanic_survive()

# Replace RandomForestClassifier with the actual model instantiation and training
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(x_train, y_train)

# Replace 'your_model' with the actual model you're using
explainer = ClassifierExplainer(model, x_test, y_test)

# Run the Explainer Dashboard
ExplainerDashboard(explainer).run()
