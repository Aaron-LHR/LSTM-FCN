with open("model_to_use.csv", mode="r", encoding="utf-8") as file:
    f.write(json.dumps(hyperparameters_of_model))