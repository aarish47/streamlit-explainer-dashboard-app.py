from sklearn.model_selection import train_test_split
from explainerdashboard.data import titanic_data, titanic_labels
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

# Create a train-test split
x_train, x_test, y_train, y_test = train_test_split(
    titanic_data.values, titanic_labels, test_size=0.2, random_state=42
)

# Replace RandomForestClassifier with the actual model instantiation and training
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(x_train, y_train)

# Replace 'your_model' with the actual model you're using
explainer = ClassifierExplainer(model, x_test, y_test)

# Run the Explainer Dashboard
ExplainerDashboard(explainer).run()
