import mlflow

from model_train import train
from model_evaluate import evaluate

def run_pipeline():
    """Runs the full MLflow pipeline"""

    mlflow.set_experiment("Movie Recommender Pipeline")

    with mlflow.start_run():
        user_path = "./Data/content_user_train.csv"
        movies_path = "./Data/content_item_train.csv"
        rating_path = "./Data/content_y_train.csv"

        # Train
        model = train(user_path, movies_path, rating_path)

        # Evaluate
        evaluate(user_path, movies_path, rating_path)

if __name__ == "__main__":
    run_pipeline()
