import pandas as pd

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data

def main():
    en = load_data("data/En-Dataset.csv")
    fr = load_data("data/Fr-Dataset.csv")
    # TODO: implement stats & print plots

if __name__ == "__main__":
    main()
