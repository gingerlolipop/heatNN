import urllib.request
import urllib.parse
import csv
import xml.etree.ElementTree as ET
from datetime import datetime

base_url = "http://export.arxiv.org/api/query?"

query = '(all:"machine learning" OR all:"ml" OR all:"artificial intelligence" OR all:"AI" OR all:"neural network" OR all:"NN" OR all:"deep learning" OR all:"DL" OR all:"shallow learning" OR all:"shallow neural network" OR all:"SNN" OR all:"random forest" OR all:"RF" OR all:"tree-based models" OR all:"ensemble methods" OR all:"LSTM" OR all:"learning" OR all:"Convolutional neural network" OR all:"CNN" OR all:"Graph neural network" OR all:"GNN" OR all:"Transformer" OR all:"Transformer models" OR all:"Bayesian network" OR all:"BN" OR all:"support vector machine" OR all:"SVM" OR all:"reinforcement learning" OR all:"RL") AND (all:"forecast" OR all:"forecasts" OR all:"forecasted" OR all:"predict" OR all:"prediction" OR all:"predictive modeling" OR all:"projection" OR all:"proj" OR all:"model" OR all:"modeling") AND ((all:"heatwave" OR all:"heatwaves" OR all:"blocking high" OR all:"heat emergency" OR all:"extreme" OR all:"extremes" OR all:"anomaly" OR all:"anomalies" OR all:"heat" OR all:"drought" OR all:"storm" OR all:"storms" OR all:"cold wave" OR all:"cold waves" OR all:"heat stress") AND (all:"medium range" OR all:"MR" OR all:"extended range" OR all:"ER" OR all:"subseasonal" OR all:"seasonal" OR all:"S2S")) AND (all:"weather" OR all:"climate" OR all:"atmospheric conditions" OR all:"environmental conditions" OR all:"meteorological" OR all:"meteorology" OR all:"satellite" OR all:"remote sensing" OR all:"RS" OR all:"observational data" OR all:"observations" OR all:"NWP" OR all:"numerical weather prediction" OR all:"reanalysis" OR all:"historical weather" OR all:"HW")'
encoded_query = urllib.parse.quote(query)

url = base_url + "search_query=" + encoded_query + "&start=0&max_results=100"

# Make the API request
response = urllib.request.urlopen(url)
data = response.read().decode()

# Parse the XML response
root = ET.fromstring(data)

# Get the current date and time for the filename
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"arxiv_search_results_{current_time}.csv"

# Define the CSV output structure
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header with the requested columns
    writer.writerow(['Title', 'Author', 'Author Affiliation', 'Source', 'Publication Year', 'Volume and Issue', 'Pages', 'Issue Date', 'Monograph Title', 'Database', 'Summary', 'Published Date', 'Link'])

    # Loop through each entry and extract data
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        authors = ', '.join([author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')])
        
        # Placeholder fields for data not present in the response
        author_affiliation = "N/A"  # Not always provided in ArXiv
        source = "ArXiv"  # Assuming the source is ArXiv
        published = entry.find('{http://www.w3.org/2005/Atom}published').text.split('-')[0]  # Extract only the year
        volume_issue = "N/A"  # ArXiv doesn't generally provide this
        pages = "N/A"  # No page numbers in ArXiv
        issue_date = entry.find('{http://www.w3.org/2005/Atom}published').text
        monograph_title = "N/A"  # Typically unavailable for preprints
        database = "ArXiv"
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        published_date = entry.find('{http://www.w3.org/2005/Atom}published').text
        link = entry.find('{http://www.w3.org/2005/Atom}id').text

        # Write each entry to the CSV file
        writer.writerow([title, authors, author_affiliation, source, published, volume_issue, pages, issue_date, monograph_title, database, summary, published_date, link])

print(f"Data saved to {csv_file}")
