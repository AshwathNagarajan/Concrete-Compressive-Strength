import os
from trainer.trainer import Trainer
from predict.predict import Predictor


def main():
    while True:
        print("\nConcrete Strength Predictor")
        print("1. Train models")
        print("2. Predict")
        print("3. Exit")

        choice = input("Choose option: ")
        if choice == "1":
            Trainer().trainer()
        elif choice == "2":

            print("\nSelect model:")
            print("1. Random Forest")
            print("2. XGBoost")
            print("3. Both")

            model_choice = input("Choose model (1-3): ")

            if model_choice == "1":
                model = "rf"
            elif model_choice == "2":
                model = "xgb"
            else:
                model = "both"

            predictor = Predictor()
            predictor.predict(model_choice=model)
        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()