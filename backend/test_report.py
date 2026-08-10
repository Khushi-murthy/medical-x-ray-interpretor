from report_generator import generate_report

predictions = {

    "Infiltration":0.84,

    "Mass":0.33,

    "Cardiomegaly":0.22,

    "Nodule":0.15,

    "Pneumonia":0.04

}

report = generate_report(predictions)

for key, value in report.items():
    print(f"{key}:\n{value}\n")