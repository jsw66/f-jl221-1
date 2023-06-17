#import relevant modules
from transformers import pipeline
from openpyxl import load_workbook
import pandas as pd

#define the pipeline classifier and model used 
#here it is BART found at https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")

#load in the patient responses
df = pd.read_csv('adults_templates.csv')

#create the labels from the list of template names
templates = list(df.iloc[:,0])
df = pd.read_csv('all_patient_data_cleaned.csv')
patient_responses = list(df.iloc[:,1])


#define a function to run the classifier and include some output formatting to obtain the top n templates
def zsc(patient_response, n):

    input = patient_response
    labels = templates

    output = classifier(input, labels, multi_label=True)

    output_labels = list(output.items())[1][1]
    output_scores = list(output.items())[2][1]

    output_percent = []
    for score in output_scores:
        output_percent.append(round(score*100, 1))

    zipped = zip(output_labels, output_percent)

    lst = list(zipped)

    row = [patient_response]

    for i in range(n):
        for item in lst[i]:
            row.append(item)

    return row

bulk_output = []

x = 1
T = len(patient_responses)

for response in patient_responses:
    output = zsc(response, 10)
    bulk_output.append(output)
    print(x, "/", T, "responses")
    x += 1

#exports the data to a excel spreadsheet

wb_append = load_workbook("zsc_responses.xlsx")
 
sheet = wb_append.active
rows = bulk_output
 
#Storing date in tuple of tuples
for row in rows:
    sheet.append(row)
 
#Saving the data in our sample workbook/sheet
wb_append.save('DeBERTa_top_10.xlsx')
