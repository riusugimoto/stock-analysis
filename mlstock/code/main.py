import pandas as pd
from data_load import load_sp500_data, prepare_data
from model_train import train_model, predict, evaluate_model, backtest, backtest2, rolling
from plot import plot_sp500, plot_combined

def main():
    # Load and prepare data
    sp500 = load_sp500_data()
    plot_sp500(sp500)
    sp500 = prepare_data(sp500)

    # Define predictors
    predictors = ["Close", "Volume", "Open", "High", "Low"]

    # Train model
    train = sp500.iloc[:-100]
    test = sp500.iloc[-100:]
    model = train_model(train, predictors)

    # Predict and evaluate
    preds = predict( test, predictors, model)
    precision = evaluate_model(test["Target"], preds)
    print(f"Precision: {precision}")

    # Plot combined predictions
    combined = pd.concat([test["Target"], preds], axis=1)
    plot_combined(combined)

    # Backtest
    # predictions = backtest(sp500, model, predictors)
    # print(predictions)
    # backtest_precision = evaluate_model(predictions["Target"], predictions["Predictions"])
    # print(f"Backtest Precision: {backtest_precision}")

      # Generate rolling features and backtest
    sp500, new_predictors = rolling(sp500)    ## when a method has more than 1 return DO THIS
    predictions = backtest2(sp500, model, new_predictors)
    print (predictions)
    
    precision = evaluate_model(test["Target"], predictions["Predictions"])
    print(f"Precision: {precision}")

    print(predictions["Target"].value_counts() / predictions.shape[0])









if __name__ == "__main__":
    main()
