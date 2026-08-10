from gemini_report import generate_ai_report

predictions = {

    "Infiltration":0.2167,

    "Cardiomegaly":0.1521,

    "Nodule":0.1557,

    "Mass":0.1294,

    "Pneumonia":0.0235

}

report = generate_ai_report(predictions)

print(report)