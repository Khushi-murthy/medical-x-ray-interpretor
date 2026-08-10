from report_generator import generate_report
from pdf_generator import generate_pdf

predictions = {

    "Infiltration":0.84,

    "Mass":0.32,

    "Cardiomegaly":0.25,

    "Nodule":0.15,

    "Pneumonia":0.04

}

report = generate_report(predictions)

import os

output = os.path.join(os.getcwd(), "medical_report.pdf")

generate_pdf(report, output)

print("PDF saved at:", output)

print("PDF Generated Successfully")