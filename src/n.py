import pyiast as pt
import pandas as pd

my_data = pd.read_excel("path_to_excel_file")  # load dataframe from excel
my_data.to_csv("path_to_save_file.csv")  # save it as .csv
isotherm = pt.ModelIsotherm(my_data,
                            loading_key="Loading(mmol/g)",
                            pressure_key="Pressure(bar)",
                            model="Langmuir")  # "Langmuir", "M", "K", "Quadratic", "BET", "DSLangmuir", "TemkinApprox", "Henry"

isotherm.print_params()


small = None
big = None

while True:
    inp = input("please input a number:")
    if inp == "done":
        break

    try:
        num = int(inp)
    except ValueError:
        print("Error: not a number")
        continue

    if small is None:
        small = big = num

    if num < small:
        small = num
        print(f"Small replaced by {small}")
    elif num > big:
        big = num
        print(f"Big replaced by {big}")
    else:
        print(f"No change. {num} between {small} and {big}")

    out = f"Big = {big}, small = {small}"
    print(out)

print(f"finished!\n {out}")
