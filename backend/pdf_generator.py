from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


styles = getSampleStyleSheet()


def generate_pdf(report, output_path):

    doc = SimpleDocTemplate(output_path)

    elements = []

    title = Paragraph(
        "<b><font size=20>CHEST X-RAY AI REPORT</font></b>",
        styles["Title"]
    )

    elements.append(title)
    elements.append(
        Paragraph("<b>Disease Probability Table</b>", styles["Heading2"])
    )

    table_data = [["Disease", "Probability (%)"]]

    for disease, probability in report["disease_probabilities"]:
        table_data.append([disease, str(probability)])

    prob_table = Table(table_data, colWidths=[3 * inch, 2 * inch])

    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
    ]))

    elements.append(prob_table)
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(
        Paragraph("<b>Secondary Findings</b>", styles["Heading2"])
    )

    if report.get("secondary_findings"):
        for disease in report["secondary_findings"]:
            elements.append(Paragraph("• " + disease, styles["BodyText"]))
    else:
        elements.append(Paragraph("No significant secondary findings.", styles["BodyText"]))

    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("<b>Recommendations</b>", styles["Heading2"]))

    for rec in report.get("recommendations", []):
        elements.append(Paragraph("• " + rec, styles["BodyText"]))

    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("<b>Disclaimer</b>", styles["Heading2"]))
    elements.append(Paragraph(report.get("disclaimer", ""), styles["BodyText"]))

    doc.build(elements)
    