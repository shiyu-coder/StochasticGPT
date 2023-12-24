import re

import json
import traceback

from llm_api import GPT, parse_json, llm_request
from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import LatexTextSplitter
from util import multiprocess, get_cpu_count

overall_structure_extraction_prompt = """
I am working with a LaTeX-formatted academic paper and require assistance in organizing its content logically. My goal is to construct a directed acyclic graph (DAG) that depicts the contextual relationships throughout the paper.

To do this, here's what I need:

The nodes of the graph will represent distinct elements of the paper: the title, abstract, sections, and subsections.
The edges of the graph will denote contextual and logical connections between these elements, illustrating how one section relates to and follows from another to contribute towards the paper's overall narrative.
Your task is to analyze the logical flow of the content and create a structured representation in JSON format. This JSON object should include all the titles, the abstract, sections, and subsections as nodes, with directed edges showing the hierarchy and flow of information.

Here is an example of JSON format for your reference:
{json_example}
Please base your construction of the graph on the contextual coherence of the paper's content. The paper text provided as {paper_text} will enable you to identify the logical associations.

Here's the paper for your review and analysis: {paper_text}
"""

overall_structure_extraction_json_example = {
    "nodes": [
        {"name": "Title", "type": "title", "parents": []},
        {"name": "Abstract", "type": "abstract", "parents": ["Title"]},
        {"name": "Introduction", "type": "section", "parents": ["Abstract"]},
        {"name": "Section 1", "type": "section", "parents": ["Introduction"]},
        {"name": "Subsection 1.1", "type": "subsection", "parents": ["Section 1"]},
        {"name": "Subsection 1.2", "type": "subsection", "parents": ["Section 1"]},
        {"name": "Section 2", "type": "section", "parents": ["Section 1"]},
        {"name": "Subsection 2.1", "type": "subsection", "parents": ["Section 2"]},
        {"name": "Conclusions", "type": "section", "parents": ["Section 1", "Section 2"]}
    ],
    "edges": [
        {"from": "Title", "to": "Abstract"},
        {"from": "Abstract", "to": "Introduction"},
        {"from": "Introduction", "to": "Section 1"},
        {"from": "Section 1", "to": "Subsection 1.1"},
        {"from": "Section 1", "to": "Subsection 1.2"},
        {"from": "Section 1", "to": "Section 2"},
        {"from": "Section 2", "to": "Subsection 2.1"},
        {"from": "Section 1", "to": "Conclusions"},
        {"from": "Section 2", "to": "Conclusions"}
    ]
}

section_structure_json_example = {
    "nodes": [
        {
            "name": "Context and Importance",
            "content": "The WHO reports on the prevalence of substandard air quality and its global health impact, along with potential policies and investments to reduce outdoor air pollution.",
            "parents": []
        },
        {
            "name": "Research Focus",
            "content": "The paper focuses on modeling concentrations of particulate matter, specifically PM2.5 and PM10.",
            "parents": ["Context and Importance"]
        },
        {
            "name": "MediaEval 2022 Challenge",
            "content": "Description of the MediaEval 2022 challenge that requires participants to predict AQI values using various data sources.",
            "parents": ["Research Focus"]
        },
        {
            "name": "Problem Statement",
            "content": "Acknowledgment of common data gaps in air quality datasets, especially in poorer or developing countries, and an introduction to the paper's approach to addressing these gaps.",
            "parents": ["MediaEval 2022 Challenge"]
        }
    ],
    "edges": [
        {
            "from": "Context and Importance",
            "to": "Research Focus"
        },
        {
            "from": "Research Focus",
            "to": "MediaEval 2022 Challenge"
        },
        {
            "from": "MediaEval 2022 Challenge",
            "to": "Problem Statement"
        }
    ]
}

section_structure_prompt = """
I am currently working on a research paper and require assistance in analyzing a specific section. 
The paper's structure is represented by the following JSON-encoded directed acyclic graph (DAG), which has nodes designated as the title, abstract, sections, and subsections: {paper_structure}.
I am interested in dissecting the internal logical structure of one particular section named "{section_label}". The text of this section is as follows:

"{section_content}"

To conduct the analysis, please follow these steps:

Examine the text provided for the "{section_label}" section to identify its internal logical structure.
Break down this structure into its elemental components, such as claims, arguments, evidence, and conclusions.
Create a graphical representation of this section's logical structure, ensuring each node includes 'name', 'content', and 'parents' to define its relationship with other components of the section.
Provide the breakdown in a clear and structured JSON format.
The final JSON should reflect the specific section's internal logic and how each part contributes to the overall argument. Please ensure no node names are duplicated from those of the overall paper's DAG.

Here is an example for reference:
{json_example}
Kindly generate logical flow of the section content in JSON format, detailing the internal logical structure of the "{section_label}" section.
"""


def extract_sections(latex_text):
    """
    Extracts sections and subsections from a LaTeX document into a dictionary.
    If a section contains subsections, only the subsections are extracted.
    """
    # Initialize the pattern for sections and subsections
    pattern = r'\\(sub)*section\{([^\}]+)\}'

    # Find all section/subsection commands and their titles
    section_titles = re.findall(pattern, latex_text)

    # Initialize the dictionary to store results
    sections = {}
    current_section = ""
    current_content = ""
    collecting = False

    # Process the LaTeX line by line
    for line in latex_text.splitlines():
        # Check if we are at the start of a new section/subsection
        match = re.match(pattern, line)
        if match:
            # Save the previous section if we were collecting its content
            if collecting:
                # Only keep non-empty sections and avoid nested section overrides
                if current_content.strip() and current_section not in sections:
                    sections[current_section] = current_content.strip()
                current_content = ""

            # Set the new section title
            current_section = match.group(2).strip()
            collecting = True
        elif collecting:
            # Accumulate the lines of content for the current section
            current_content += line + "\n"

    # Save the last section
    if collecting and current_content.strip():
        sections[current_section] = current_content.strip()

    return sections


def extract_and_convert_json(input_text):
    # Find the start of the JSON by looking for the ```json opening
    json_start = input_text.find("```json") + len("```json")

    # Find the end of the JSON by looking for the closing ```
    json_end = input_text.find("```", json_start)

    # Extract the JSON string from the text
    json_str = input_text[json_start:json_end].strip()

    # Convert the JSON string into a Python dictionary
    python_dict = json.loads(json_str)

    return python_dict


def extract_overall_structure(paper_text):
    max_try = 3
    try_count = 0
    while try_count < max_try:
        try:
            request = overall_structure_extraction_prompt.format(json_example=overall_structure_extraction_json_example, paper_text=paper_text)
            reply = llm_request(request, response_type="json_object")
            # reply = model.chat(request, response_type="json_object")
            overall_structure_json = parse_json(reply)
            return overall_structure_json
        except KeyboardInterrupt:
            return
        except:
            traceback.print_exc()
            try_count += 1


def extract_section_structure(section, paper_structure):
    section_label, section_content = section
    if len(section_content) < 100:
        return {"nodes": [], "edges": []}
    max_try = 3
    try_count = 0
    while try_count < max_try:
        try:
            request = section_structure_prompt.format(paper_structure=paper_structure, section_label=section_label, section_content=section_content,
                                                      json_example=section_structure_json_example)
            # reply = model.chat(request, response_type="json_object")
            reply = llm_request(request, response_type="json_object")
            section_structure_json = parse_json(reply)
            print(section_structure_json)
            return section_structure_json
        except KeyboardInterrupt:
            return
        except:
            traceback.print_exc()
            try_count += 1


def extract_paper_structure(paper_text):
    print("Extracting overall structure")
    overall_structure_json = extract_overall_structure(paper_text)
    print(overall_structure_json)
    dt_section = extract_sections(paper_text)
    ls_section_pairs = []
    for section_label in dt_section:
        ls_section_pairs.append((section_label, dt_section[section_label]))
    print("Extracting section structures")
    ls_section_structure_json = multiprocess(
        func=extract_section_structure,
        paras=ls_section_pairs,
        paper_structure=overall_structure_json,
        n_processes=min(len(ls_section_pairs), get_cpu_count())
    )
    dt_section_structure = dict(zip(list(dt_section.keys()), ls_section_structure_json))
    dt_paper = {
        'overall_stu': overall_structure_json,
        'section_stu': dt_section_structure,
        'section_con': dt_section
    }
    return dt_paper


if __name__ == '__main__':
    root_path = f"../StochasticGPT_data/paper/2212.10273/DCU-AQ.tex"
    paper_text = TextLoader(root_path).load()[0].page_content
    print(paper_text)

    # dt_paper_info = extract_paper_structure(paper_text)

    # print(dt_paper_info)
    result = {'overall_stu': {'nodes': [{'name': 'Title', 'type': 'title', 'parents': []}, {'name': 'Abstract', 'type': 'abstract', 'parents': ['Title']},
                                        {'name': 'Introduction', 'type': 'section', 'parents': ['Abstract']},
                                        {'name': 'Methodology', 'type': 'section', 'parents': ['Introduction']},
                                        {'name': 'Data Analysis', 'type': 'subsection', 'parents': ['Methodology']},
                                        {'name': 'Image Processing', 'type': 'subsection', 'parents': ['Methodology']},
                                        {'name': 'Gap Filling', 'type': 'subsection', 'parents': ['Methodology']},
                                        {'name': 'Experiments', 'type': 'section', 'parents': ['Methodology']},
                                        {'name': 'Experimental Approach', 'type': 'subsection', 'parents': ['Experiments']},
                                        {'name': 'Results', 'type': 'subsection', 'parents': ['Experiments']},
                                        {'name': 'Conclusions and Lessons Learned', 'type': 'section', 'parents': ['Results']},
                                        {'name': 'Acknowledgements', 'type': 'section', 'parents': ['Conclusions and Lessons Learned']}],
                              'edges': [{'from': 'Title', 'to': 'Abstract'}, {'from': 'Abstract', 'to': 'Introduction'},
                                        {'from': 'Introduction', 'to': 'Methodology'},
                                        {'from': 'Methodology', 'to': 'Data Analysis'}, {'from': 'Methodology', 'to': 'Image Processing'},
                                        {'from': 'Methodology', 'to': 'Gap Filling'}, {'from': 'Methodology', 'to': 'Experiments'},
                                        {'from': 'Experiments', 'to': 'Experimental Approach'}, {'from': 'Experiments', 'to': 'Results'},
                                        {'from': 'Results', 'to': 'Conclusions and Lessons Learned'},
                                        {'from': 'Conclusions and Lessons Learned', 'to': 'Acknowledgements'}]},
              'section_stu': {
                  'Introduction': {'nodes': [
                      {'name': 'Challenge Overview',
                       'content': 'The Urban Life and Air Pollution task at MediaEval 2022 is introduced, which required participants to predict the air quality index (AQI) value at future intervals using a variety of data sources.',
                       'parents': []}, {'name': 'Data Gaps Issue',
                                        'content': 'The paper acknowledges the common issue of gaps in air quality datasets, which is particularly problematic in poorer or developing countries.',
                                        'parents': ['Challenge Overview']}, {'name': 'Research Contribution',
                                                                             'content': 'The paper outlines its contribution by describing the approach taken to address the large gaps in the air quality data encountered.',
                                                                             'parents': ['Data Gaps Issue']}], 'edges': [
                      {'from': 'Challenge Overview', 'to': 'Data Gaps Issue'}, {'from': 'Data Gaps Issue', 'to': 'Research Contribution'}]},
                  'Methodology': {'nodes': [
                      {'name': 'Data Gaps Challenge',
                       'content': 'The research identifies significant gaps in the training data as a major challenge, which is a common issue with climate datasets.',
                       'parents': []}, {'name': 'Data Download Variability',
                                        'content': 'Due to independent data downloads by participants and server downtimes, there is variability in the training data available to different participants.',
                                        'parents': ['Data Gaps Challenge']},
                      {'name': 'Research Focus on Data Gaps',
                       'content': 'Despite the variability in data, the research focuses on addressing the challenge of data gaps.',
                       'parents': ['Data Download Variability']},
                      {'name': 'Methodology Overview',
                       'content': 'The methodology comprises four steps to address the sensitivity of data gaps and adopt counter-measures.',
                       'parents': ['Research Focus on Data Gaps']}, {'name': 'Step 1: Data Analysis',
                                                                     'content': 'Statistical summary of datasets and computation of spatial data related to the locations of air quality stations and cameras.',
                                                                     'parents': ['Methodology Overview']},
                      {'name': 'Step 2: Gap Filling', 'content': 'Elimination or significant reduction of gaps in air quality data.',
                       'parents': ['Methodology Overview']},
                      {'name': 'Step 3: Image Processing',
                       'content': 'Transformation of CCTV camera images into a set of features to be combined with the air quality feature set.',
                       'parents': ['Methodology Overview']}, {'name': 'Step 4: Model Building',
                                                              'content': 'Building an experimental platform using various machine learning model configurations and feature sets to identify the best performing combination.',
                                                              'parents': ['Methodology Overview']},
                      {'name': 'Detailed Steps Description', 'content': 'The first three steps of the methodology are described in detail within this section.',
                       'parents': ['Step 1: Data Analysis', 'Step 2: Gap Filling', 'Step 3: Image Processing']}, {'name': 'Model Building Description',
                                                                                                                  'content': 'The approach to model building for the air quality prediction task is described in a subsequent section.',
                                                                                                                  'parents': ['Step 4: Model Building']}],
                      'edges': [
                          {'from': 'Data Gaps Challenge', 'to': 'Data Download Variability'},
                          {'from': 'Data Download Variability', 'to': 'Research Focus on Data Gaps'},
                          {'from': 'Research Focus on Data Gaps', 'to': 'Methodology Overview'},
                          {'from': 'Methodology Overview', 'to': 'Step 1: Data Analysis'},
                          {'from': 'Methodology Overview', 'to': 'Step 2: Gap Filling'}, {'from': 'Methodology Overview', 'to': 'Step 3: Image Processing'},
                          {'from': 'Methodology Overview', 'to': 'Step 4: Model Building'},
                          {'from': 'Step 1: Data Analysis', 'to': 'Detailed Steps Description'},
                          {'from': 'Step 2: Gap Filling', 'to': 'Detailed Steps Description'},
                          {'from': 'Step 3: Image Processing', 'to': 'Detailed Steps Description'},
                          {'from': 'Step 4: Model Building', 'to': 'Model Building Description'}]},
                  'Data Analysis': {'nodes': [{'name': 'Data Collection',
                                               'content': 'The downloaded air quality data are collected at 10 monitoring stations in Dalat City, Vietnam from March 2020 to 7th Nov 2022.',
                                               'parents': []},
                                              {'name': 'Data Components',
                                               'content': 'The data includes air pollutant concentration for NO$_2$(ppm), CO, SO$_2$, O$_3$, PM1.0, PM2.5, PM10 as well as environmental measures namely temperature, humidity, UV, rainfall.',
                                               'parents': [
                                                   'Data Collection']},
                                              {'name': 'Additional Data',
                                               'content': 'In addition, traffic data, in form of images, was recorded every minute from 16 CCTV cameras across Dalat City.',
                                               'parents': [
                                                   'Data Collection']},
                                              {'name': 'Data Availability',
                                               'content': 'Figure~\\ref{fig:availability} shows the availability of the dataset as downloaded by our group. This shows huge gaps in data availability.',
                                               'parents': [
                                                   'Data Collection']},
                                              {'name': 'Model Training',
                                               'content': 'In our model building we use the first 80% of available data for training machine learning models',
                                               'parents': [
                                                   'Data Availability']},
                                              {'name': 'Model Validation',
                                               'content': 'and the remaining 20% for validation.',
                                               'parents': ['Model Training']},
                                              {
                                                  'name': 'Data Gaps Illustration',
                                                  'content': 'Missing data (shown in yellow) from across all 10 air quality measurement stations and 16 CCTV cameras for an 8-month period, early March to early November 2022.',
                                                  'parents': [
                                                      'Data Availability']}],
                                    'edges': [
                                        {'from': 'Data Collection',
                                         'to': 'Data Components'},
                                        {'from': 'Data Collection',
                                         'to': 'Additional Data'},
                                        {'from': 'Data Collection',
                                         'to': 'Data Availability'},
                                        {'from': 'Data Availability',
                                         'to': 'Model Training'},
                                        {'from': 'Model Training',
                                         'to': 'Model Validation'},
                                        {'from': 'Data Availability',
                                         'to': 'Data Gaps Illustration'}]},
                  'Image Processing': {'nodes': [
                      {'name': 'Data Volume and Storage',
                       'content': 'For the CCTV data, we downloaded a total of 398,412 images from across all 16 cameras, which took approximately 215GB of storage.',
                       'parents': []}, {'name': 'Theoretical Maximum Calculation',
                                        'content': 'If all data had been available and downloaded, there would be approximately 16 cameras x 8 months x 30 days x 24 hours x 60 minutes = 5,529,600 images so our download represents 7.2% of the theoretical maximum.',
                                        'parents': ['Data Volume and Storage']}, {'name': 'Image Resizing and Processing',
                                                                                  'content': 'We re-sized each image to 640x640 and processed each using a medium-sized YOLOv6 object-detector pre-trained on the COCO dataset.',
                                                                                  'parents': ['Data Volume and Storage']},
                      {'name': 'Object Detection Performance', 'content': 'This performs well with a balanced trade-off between speed and accuracy.',
                       'parents': ['Image Resizing and Processing']}, {'name': 'Output Categorization',
                                                                       'content': 'From an output with more than 80 object categories, we used the average of 4 vehicle types as a proxy for traffic volume.',
                                                                       'parents': ['Object Detection Performance']}, {'name': 'Vehicle Detection Statistics',
                                                                                                                      'content': 'With an average detection per image of 2.58 (cars), 3.90 (motorcycles), 0.16 (bus) and 0.25 (trucks).',
                                                                                                                      'parents': ['Output Categorization']},
                      {'name': 'Feature Utilization for Predictive Model',
                       'content': 'These values are used directly as features for our predictive model without further post-processing.',
                       'parents': ['Vehicle Detection Statistics']}], 'edges': [{'from': 'Data Volume and Storage', 'to': 'Theoretical Maximum Calculation'},
                                                                                {'from': 'Data Volume and Storage', 'to': 'Image Resizing and Processing'},
                                                                                {'from': 'Image Resizing and Processing', 'to': 'Object Detection Performance'},
                                                                                {'from': 'Object Detection Performance', 'to': 'Output Categorization'},
                                                                                {'from': 'Output Categorization', 'to': 'Vehicle Detection Statistics'},
                                                                                {'from': 'Vehicle Detection Statistics',
                                                                                 'to': 'Feature Utilization for Predictive Model'}]},
                  'Gap Filling': {'nodes': [{
                      'name': 'Introduction to Gap Filling',
                      'content': 'Our approach to gap filling used 3 different feature sets.',
                      'parents': []},
                      {
                          'name': 'Feature Set 1 Description',
                          'content': 'Rolling Window (FS1) involves generating training data using rolling windows sliding hourly over the data, with a historical data length of 2 days determined after experimentation.',
                          'parents': [
                              'Introduction to Gap Filling']},
                      {
                          'name': 'FS1 Data Handling',
                          'content': 'To address missing data in FS1, windows with big gaps are removed, and windows with small gaps of less than 30% or data from more than 4 stations are filled.',
                          'parents': [
                              'Feature Set 1 Description']},
                      {
                          'name': 'Feature Set 2 Description',
                          'content': 'Rolling Window with Gap Filling (FS2) uses spatial interpolation with a tree-based gradient boosting model (LightGBM) to infer missing values for each air pollutant at each station.',
                          'parents': [
                              'Introduction to Gap Filling']},
                      {
                          'name': 'FS2 Model Training',
                          'content': 'A separate LightGBM model is trained for each air pollutant at each station using all available data from other stations at that timestamp.',
                          'parents': [
                              'Feature Set 2 Description']},
                      {
                          'name': 'FS2 Model Advantages',
                          'content': 'The LightGBM model is efficient and sparsity-aware, allowing it to learn from incomplete data.',
                          'parents': [
                              'Feature Set 2 Description']},
                      {
                          'name': 'Feature Set 3 Description',
                          'content': 'Rolling Window with Gap Filling and Image/Traffic Features (FS3) includes the output from the count of vehicles detected in CCTV images from 16 cameras.',
                          'parents': [
                              'Introduction to Gap Filling']},
                      {
                          'name': 'Data Conversion',
                          'content': 'All pollutant concentration values are converted into AQI values according to a specific formula.',
                          'parents': [
                              'Introduction to Gap Filling']},
                      {
                          'name': 'Data Aggregation and Normalization',
                          'content': 'Air quality, environmental, and CCTV features are aggregated into hourly mean values and normalized before being input into various machine learning models.',
                          'parents': [
                              'Data Conversion']},
                      {
                          'name': 'Machine Learning Model Input',
                          'content': 'The normalized data is used as input for a selection of different machine learning models.',
                          'parents': [
                              'Data Aggregation and Normalization']}],
                      'edges': [{
                          'from': 'Introduction to Gap Filling',
                          'to': 'Feature Set 1 Description'},
                          {
                              'from': 'Feature Set 1 Description',
                              'to': 'FS1 Data Handling'},
                          {
                              'from': 'Introduction to Gap Filling',
                              'to': 'Feature Set 2 Description'},
                          {
                              'from': 'Feature Set 2 Description',
                              'to': 'FS2 Model Training'},
                          {
                              'from': 'Feature Set 2 Description',
                              'to': 'FS2 Model Advantages'},
                          {
                              'from': 'Introduction to Gap Filling',
                              'to': 'Feature Set 3 Description'},
                          {
                              'from': 'Introduction to Gap Filling',
                              'to': 'Data Conversion'},
                          {
                              'from': 'Data Conversion',
                              'to': 'Data Aggregation and Normalization'},
                          {
                              'from': 'Data Aggregation and Normalization',
                              'to': 'Machine Learning Model Input'}]},
                  'Experiments': {},
                  'Experimental Approach': {'nodes': [
                      {'name': 'Gap Management Strategy',
                       'content': 'To manage the issue of gaps, a rolling window approach was used to remove very large gaps, followed by filling the remaining gaps using the LightGBM method described earlier.',
                       'parents': []}, {
                          'name': 'Feature Extraction and Conversion',
                          'content': 'Air pollutant concentration values are converted to corresponding air quality index (AQI) values. Traffic features are extracted from CCTV images in the form of numbers of vehicle types.',
                          'parents': [
                              'Gap Management Strategy']},
                      {'name': 'Prediction Timeframes',
                       'content': '48 hourly values of features (AQI, weather measures, and traffic) are input into machine learning models to make predictions of AQI values for the next 24, 120, and 168 hours.',
                       'parents': [
                           'Feature Extraction and Conversion']},
                      {'name': 'Baseline Models',
                       'content': 'Baseline machine learning models used include multi-layer perceptrons, long short-term memory (LSTM), and LSTM-GNN, where the prediction is treated as timeseries forecasting.',
                       'parents': [
                           'Prediction Timeframes']},
                      {'name': 'Advanced Models',
                       'content': 'A spatio-temporal graph neural network and custom neural network architectures, TemAtt and SpaTemAtt, are designed to capture more complex temporal and spatio-temporal patterns.',
                       'parents': ['Baseline Models']},
                      {'name': 'Citation Issues',
                       'content': 'There are concerns about citing unpublished work, specifically the SpaTemAtt model, which may not be included if it is not published in time for the final submission of this paper.',
                       'parents': ['Advanced Models']}],
                      'edges': [{
                          'from': 'Gap Management Strategy',
                          'to': 'Feature Extraction and Conversion'},
                          {
                              'from': 'Feature Extraction and Conversion',
                              'to': 'Prediction Timeframes'},
                          {
                              'from': 'Prediction Timeframes',
                              'to': 'Baseline Models'},
                          {
                              'from': 'Baseline Models',
                              'to': 'Advanced Models'},
                          {
                              'from': 'Advanced Models',
                              'to': 'Citation Issues'}]},
                  'Results': {'nodes': [
                      {'name': 'Results Overview',
                       'content': 'The Results section presents the outcomes of the validation process, highlighting the best performing model configurations with bold and underlined text for the top two results respectively.',
                       'parents': []}, {
                          'name': 'Feature and Model Description',
                          'content': 'The first two columns of the results table describe the feature sets and models used, with runs marked 1-5 representing five different prediction attempts.',
                          'parents': [
                              'Results Overview']},
                      {'name': 'Station Data Split',
                       'content': "The results are divided into two categories based on station data availability: 'Station (recent data)' for stations that continued to provide data, and 'Station (no recent data)' for stations that stopped supplying data due to malfunctions.",
                       'parents': ['Results Overview']},
                      {'name': 'Prediction Accuracy',
                       'content': 'The paper provides accuracy metrics for 1-day, 5-days, and 7-days ahead predictions, using root mean squared error (RMSE) as the validation measure.',
                       'parents': ['Station Data Split']},
                      {'name': 'Performance Analysis',
                       'content': 'The analysis indicates that stations with recent data performed better than those without, and predictive accuracy for all models decreases as the forecast window increases.',
                       'parents': [
                           'Prediction Accuracy']}, {
                          'name': 'Feature Set 2 Superiority',
                          'content': 'Feature set 2 (FS2) was the best performing across all time intervals, demonstrating the effectiveness of the gap filling method used in the study.',
                          'parents': [
                              'Performance Analysis']}, {
                          'name': 'Camera Data Ineffectiveness',
                          'content': 'The results indicated that the inclusion of camera data did not provide any added performance benefit.',
                          'parents': [
                              'Feature Set 2 Superiority']},
                      {
                          'name': 'Model Performance Comparison',
                          'content': 'No single model consistently outperformed others, but the Temporal Graph model (TemAtt) was the best for 1-day predictions, and the Spatio-Temporal Graph model (SpaTempAtt) was best for 5-day predictions.',
                          'parents': [
                              'Performance Analysis']},
                      {'name': 'Performance Degradation',
                       'content': 'All test models, including the graph-based models, showed significant degradation in performance as the prediction window increased.',
                       'parents': [
                           'Model Performance Comparison']}],
                      'edges': [{
                          'from': 'Results Overview',
                          'to': 'Feature and Model Description'},
                          {
                              'from': 'Results Overview',
                              'to': 'Station Data Split'},
                          {
                              'from': 'Station Data Split',
                              'to': 'Prediction Accuracy'},
                          {
                              'from': 'Prediction Accuracy',
                              'to': 'Performance Analysis'},
                          {
                              'from': 'Performance Analysis',
                              'to': 'Feature Set 2 Superiority'},
                          {
                              'from': 'Feature Set 2 Superiority',
                              'to': 'Camera Data Ineffectiveness'},
                          {
                              'from': 'Performance Analysis',
                              'to': 'Model Performance Comparison'},
                          {
                              'from': 'Model Performance Comparison',
                              'to': 'Performance Degradation'}]},
                  'Conclusions and Lessons Learned': {
                      'nodes': [{
                          'name': 'Positive Impact of Gap Filled Data',
                          'content': 'High-level conclusions indicate that gap filled data have a positive impact on predictions, approximating missing values better and adding spatial information from other locations.',
                          'parents': []}, {
                          'name': 'Superior Short-Term Prediction Models',
                          'content': 'Time series specialised models such as LSTM and TemAtt architectures marginally outperform MLP and simpler models in short-term predictions.',
                          'parents': [
                              'Positive Impact of Gap Filled Data']},
                          {
                              'name': 'Unanswered Questions',
                              'content': 'Despite some answers provided by the results, there are many other unanswered questions.',
                              'parents': [
                                  'Positive Impact of Gap Filled Data']},
                          {
                              'name': 'TemAtt vs. SpaTemAtt Performance',
                              'content': 'It is unclear why the TemAtt model outperformed the SpaTemAtt model, possibly due to the inclusion of spatial information making SpaTemAtt weaker.',
                              'parents': [
                                  'Unanswered Questions']},
                          {
                              'name': 'Poor Performance of LSTM-GNN',
                              'content': "The LSTM-GNN model's poor performance was surprising, and the CCTV images added no value, with FS3 featuring outside the top-2 across all prediction intervals.",
                              'parents': [
                                  'Unanswered Questions']},
                          {
                              'name': 'Lack of Overlap Between Data Types',
                              'content': 'The lack of overlap between CCTV and air quality data may have left little for the models to train on, contributing to poor performance.',
                              'parents': [
                                  'Poor Performance of LSTM-GNN']},
                          {
                              'name': 'Traffic Features and New Information',
                              'content': 'Additional traffic features may not provide new information, and it is questioned whether the vehicle counting was too naive or if there is another explanation.',
                              'parents': [
                                  'Poor Performance of LSTM-GNN']},
                          {
                              'name': 'Speculation on Traffic Impact',
                              'content': 'Speculation on whether air quality is affected by traffic and if the vehicle counting method was too naive or if there is a different explanation for the observed results.',
                              'parents': [
                                  'Traffic Features and New Information']}],
                      'edges': [{
                          'from': 'Positive Impact of Gap Filled Data',
                          'to': 'Superior Short-Term Prediction Models'},
                          {
                              'from': 'Positive Impact of Gap Filled Data',
                              'to': 'Unanswered Questions'},
                          {
                              'from': 'Unanswered Questions',
                              'to': 'TemAtt vs. SpaTemAtt Performance'},
                          {
                              'from': 'Unanswered Questions',
                              'to': 'Poor Performance of LSTM-GNN'},
                          {
                              'from': 'Poor Performance of LSTM-GNN',
                              'to': 'Lack of Overlap Between Data Types'},
                          {
                              'from': 'Poor Performance of LSTM-GNN',
                              'to': 'Traffic Features and New Information'},
                          {
                              'from': 'Traffic Features and New Information',
                              'to': 'Speculation on Traffic Impact'}]}},
              'section_con': {
                  'Introduction': '\\label{sec:intro}\n\n\\begin{comment}\n% Just in case we want some form of Intro\nAccording to the World Health Organisation (WHO), 91\\% of the world\'s population reside in conditions where WHO\'s air quality guidelines levels were not met \\cite{organizacion2021global}. This report on 2016 also showed that ambient (outdoor) air pollution in both cities and rural areas was estimated to cause 4.2 million premature deaths worldwide. The research concluded that policies and investments supporting cleaner transport, energy-efficient homes, power generation, industry and better municipal waste management would would be crucial to the reduction of outdoor air pollution. In a separate report, it is estimated that air pollution globally accounts for roughly seven million premature deaths a year \\cite{Gar21}, where it was again stated that the majority of those deaths are caused by outdoor air pollution with the rest generally attributed to poor air quality from indoor cooking. While the majority of these deaths occur in developing countries, with China and India accounting for roughly 50\\%, developed countries also have a problem with deaths resulting from air pollution.\nIn this research, the focus will mostly be on the modelling of concentrations in particulate matter - tiny particles in the air generated both by natural processes and human activity. These particles are generally \n12  categorised (in the public health domain) by their diameter; fine particles with diameter less than 2.5 $\\mu$m are referred to as "PM2.5" and coarse particles with diameter between 2.5 and 10 $\\mu$m are referred to as "PM10".\\\\\n\\end{comment}\n\nThe Urban Life and Air Pollution task at MediaEval 2022 required participants to predict the air quality index (AQI) value at +1, +5 and +7 days using an archive of air quality, weather and images from 16 CCTV cameras, one image taken every 60 seconds  \\cite{UA22}. Participating groups were required to download the data from online sources for local processing.\nGaps in air quality datasets are common with the problem exacerbated for data gathered in poorer or developing countries \\cite{PINDER2019116794, Falge2001, Hui2004, Moffat2007, Kim2020}. In this paper we describe how we addressed the very large gaps in data that we encountered in the data we downloaded.\n\n\n%\\section{Related Work}\n%\\label{sec:rr}\n%.. a snapshot of how our graph? technique has been used for solving other problems.\n\n% Mark: I restructured this to stop duplication of text',
                  'Methodology': "\\label{sec:approach}\nThe biggest issue in this research challenge was the significant  gaps in the training data, highlighted in Figure~\\ref{fig:availability} for the data we downloaded, and regarded as a common issue with climate datasets.\nBecause participants downloaded data independently, and because data servers had different periods of downtime, it is likely that participants have different, but overlapping training data and perhaps  other participants managed to download more data than we did. Even so,  the amount of data we downloaded  allows us to focus on the challenge of data gaps.\nThis is directly addressed in our research methodology by first identifying the sensitivity of the data gaps and adopting a counter-measure to eliminate gap data. Our method comprises 4 steps:\n\n\\begin{itemize}[leftmargin=*]\n    \\item Step 1. Data Analysis. This performs a statistical summary of datasets including the computation of spatial data related to the locations of air quality stations and cameras.\n    \\item Step 2. Gap Filling. Elimination or maximising the reduction in the gaps in air quality data.\n    \\item Step 3. Processing CCTV camera images. This step transforms each image into a set of features that can be combined with the air quality feature set.\n    \\item Step 4. Model Building. This step builds an experimental platform using different machine learning model configurations together with different feature sets to identify the best performing model/feature set combination.  \n\\end{itemize}\nIn the remainder of this section, we described the first 3 steps in detail and in Section~\\ref{sec:exper}, we describe our approach to model building for the air quality prediction task.\n\n\n% We didn't use Weather data, did we?\n%In our submission we used all three data sources -- the archive of past AQ data readings, the archive of past weather information and the archive of past images from 16 CCTV traffic cameras.",
                  'Data Analysis': 'The downloaded air quality data are collected at 10 monitoring stations in Dalat City, Vietnam from March 2020  to 7th Nov 2022. The data includes air pollutant concentration for NO$_2$(ppm), CO, SO$_2$, O$_3$, PM1.0, PM2.5, PM10 as well as environmental measures namely temperature, humidity, UV, rainfall. In addition, traffic data, in form of images, was recorded every  minute from 16 CCTV cameras across Dalat City. Figure~\\ref{fig:availability} shows the availability of the  dataset as downloaded by our group. This shows huge gaps in data availability.  In our model building we use the first 80\\% of available data for training machine learning models and the remaining 20\\% for validation.\n\n\\begin{figure}[ht!]\n    \\centering\n    \\includegraphics[width=1.0\\textwidth]{Data-gaps.jpg}\n    \\caption{Missing data (shown in yellow) from across all 10 air quality measurement stations and 16 CCTV cameras for an 8-month period, early March to early November 2022.}\n    \\label{fig:availability}\n\\end{figure}',
                  'Image Processing': 'For the CCTV data, we downloaded a total of  398,412 images from across all 16 cameras, which took approximately 215GB of storage. \n%\nIf all data had been available and  downloaded, there would be approximately 16 cameras x 8 months x 30 days x 24 hours x 60 minutes = 5,529,600 images  so our download represents 7.2\\% of the theoretical maximum. \n\nWe  re-sized each image to 640x640 and processed each using  a medium-sized YOLOv6 object-detector \\cite{yolov6} pre-trained on the COCO dataset \\cite{coco}. This performs well with a balanced trade-off between speed and accuracy.\nFrom an output with more than 80 object categories, we used the average of 4 vehicle types as a proxy for traffic volume with an average detection per image of 2.58 (cars), 3.90 (motorcycles), 0.16 (bus) and 0.25 (trucks). \nThese values are used directly as features for our predictive model without  further post-processing.',
                  'Gap Filling': 'Our approach to gap filling  used 3 different feature sets. \n\n\\begin{itemize}[leftmargin=*]\n    \\item Rolling Window (FS1). We generate training data using rolling windows sliding hourly over the data. \n    %One window is treated as one sample during the training. \n    Following some experimentation, we determine the historical data length to input to the model (size of window) to be 2 days. To address missing data, we remove windows with big gaps and fill windows with small gaps of less than 30\\%  or have data from more than 4 stations. \n    \\item Rolling Window with Gap Filling (FS2).  Spatial interpolation using a tree-based gradient boosting model \\cite{lightgbm} was used to infer missing values. A separate model to predict each of the air pollutants at each station was trained using all available data from  other stations at that timestamp.  The LightGBM model carries the same sparsity-aware learning methods as XGBoost \\cite{xgboost} while improving on efficiency. Such methods for handling sparse arrays allow a model to learn from incomplete data. \n    \\item Rolling Window with Gap Filling and Image/Traffic Features (FS3).  This feature set uses the output from the count of the numbers of cars, trucks, motorcycles and buses detected in the CCTV images from the 16 cameras.\n\n\\end{itemize}\n% Rolling Windows\n\n\\noindent\nWe convert all pollutant concentration values into AQI values according to the formula described in \\cite{AQIformula}. \nAir quality, environmental  and CCTV features are aggregated into hourly mean values which are normalised to a mean of 0.0 and standard deviation of 1.0 before  input into  a selection of different machine learning models.',
                  'Experiments': '\\label{sec:exper}',
                  'Experimental Approach': "%To manage the issue of gaps, we used a rolling window approach to remove very large gaps and then filled the remaining gaps using the LightGBM described earlier. Then, air pollutant concentration values are  converted to corresponding air quality index (AQI). Traffic features are extracted from CCTV images in form of numbers of vehicle types. 48 hourly values of these features (AQI, weather measures and traffic) are input into machine learning models to make prediction of AQI values of the next 24, 120 and 168 hours. \n\nWe used baseline machine learning models including multi-layer perceptrons, long short-term memory (LSTM) \\cite{LSTM} and LSTM-GNN \\cite{LSTMGNN} where the prediction is treated as timeseries forecasting. We also used a spatio-temporal graph neural network \\cite{LSTMGNN} and our own  neural network architecture labelled TemAtt and SpaTemAtt, designed to capture more complex temporal and spatio-temporal patterns.\n\n\n%in \\cite{AttentionGNN},  \n%\\alan{Can't include a citation to a paper that is submitted for publication .. its it on ArXiV that you could cite it from there~? .... it's not actually finalised/submitted yet but it's the only place that SpaTemAtt exists}\n%\\alan{when will you know about that publication .. lets leave it out for now and for the final submission of this MediaEval paper we can include it if its published by then}",
                  'Results': "% \\begin{table}[h!]\n% \\begin{tabular}{lllllll|llll}\n% \\toprule\n% \\multirow{2}{*}{Input Data} & \\multirow{2}{*}{Gapfilled} & \\multirow{2}{*}{Model} & \\multicolumn{4}{l|}{In-Station}                                                       & \\multicolumn{4}{l}{Out-Station}                                                      \\\\ \\cline{4-11} \n%                             &                            &                        & \\multicolumn{1}{l|}{O3} & \\multicolumn{1}{l|}{PM2.5} & \\multicolumn{1}{l|}{PM10} & CO & \\multicolumn{1}{l|}{O3} & \\multicolumn{1}{l|}{PM2.5} & \\multicolumn{1}{l|}{PM10} & CO \\\\ \\midrule\n% Air Pollutants              & No                         & MLP                    & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    \\\\ \n% Air Pollutants              & No                         & LSTM                   & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    \\\\ \n% Air Pollutants              & Yes                        & LSTM                   & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    \\\\ \n% Air Pollutants              & Yes                        & TemAtt                 & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    \\\\ \n% Air Pollutants              & Yes                        & SpaTemAtt              & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    \\\\ \n% Air Pollutants              & Yes                        & LSTM-GNN               & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    \\\\ \n% Air Pollutants+Images       & Yes                        & SpaTemAtt              & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    & \\multicolumn{1}{l|}{}   & \\multicolumn{1}{l|}{}      & \\multicolumn{1}{l|}{}     &    \\\\ \n% \\bottomrule\n% \\end{tabular}\n% \\end{table}\n\n\\begin{table}[h!]\n\\begin{tabular}{lllll|lll}\n\\hline\n\\multirow{2}{*}{Feature Set} & \\multirow{2}{*}{Model} & \\multicolumn{3}{l|}{Station (recent data)}                                                 & \\multicolumn{3}{l}{Station (no recent data)}                                                                    \\\\ \\cline{3-8} \n                            &                        & \\multicolumn{1}{l|}{+1day}          & \\multicolumn{1}{l|}{+5days}         & +7days         & \\multicolumn{1}{l|}{+1day}          & \\multicolumn{1}{l|}{+5days}         & \\multicolumn{1}{l}{+7days}         \\\\ \\hline\nFS1                         & MLP                    & \\multicolumn{1}{l|}{20.51}          & \\multicolumn{1}{l|}{27.77}          & 26.75          & \\multicolumn{1}{l|}{45.46}          & \\multicolumn{1}{l|}{48.75}          & \\multicolumn{1}{l}{46.06}          \\\\\nFS1 (run 1)                         & LSTM                   & \\multicolumn{1}{l|}{19.94}          & \\multicolumn{1}{l|}{25.84}          & {\\ul 25.42}    & \\multicolumn{1}{l|}{44.15}          & \\multicolumn{1}{l|}{46.87}          & \\multicolumn{1}{l}{46.38}          \\\\\nFS2  (run 3)                      & LSTM                   & \\multicolumn{1}{l|}{{\\ul 18.94}}    & \\multicolumn{1}{l|}{25.52}          & \\textbf{24.95} & \\multicolumn{1}{l|}{43.98}          & \\multicolumn{1}{l|}{46.66}          & \\multicolumn{1}{l}{45.36}          \\\\\nFS2  (run 4)                       & TemAtt                 & \\multicolumn{1}{l|}{\\textbf{18.57}} & \\multicolumn{1}{l|}{{\\ul 24.96}}    & 27.73          & \\multicolumn{1}{l|}{44.36}          & \\multicolumn{1}{l|}{46.60}          & \\multicolumn{1}{l}{48.67}          \\\\\nFS2                         & LSTM-GNN               & \\multicolumn{1}{l|}{23.84}          & \\multicolumn{1}{l|}{25.99}          & 30.48          & \\multicolumn{1}{l|}{39.84}          & \\multicolumn{1}{l|}{40.21}          & \\multicolumn{1}{l}{44.48}          \\\\\nFS2  (run 2)                       & SpaTemAtt              & \\multicolumn{1}{l|}{19.93}          & \\multicolumn{1}{l|}{\\textbf{24.91}} & 29.92          & \\multicolumn{1}{l|}{\\textbf{30.93}} & \\multicolumn{1}{l|}{\\textbf{34.42}} & \\multicolumn{1}{l}{\\textbf{34.85}} \\\\\nFS3   (run 5)                      & SpaTemAtt              & \\multicolumn{1}{l|}{20.43}          & \\multicolumn{1}{l|}{25.18}          & 31.75          & \\multicolumn{1}{l|}{{\\ul 35.21}}    & \\multicolumn{1}{l|}{{\\ul 38.46}}    & \\multicolumn{1}{l}{{\\ul 40.32}}    \\\\ \\hline\n\\end{tabular}\n\\caption{Best Performing Model Configurations}\n\\label{tab:results}\n\\end{table}\n\n\n\nTable \\ref{tab:results} presents results after our own validation using bold font to highlight the best performing model with the second best results underlined.  The first 2 columns show the features and models used where runs marked 1-5 represent our 5 submitted prediction attempts. The remaining 6 columns, which present RMSE scores, are split according to those stations that continued to provide data, ``Station (recent data)'', and those that stopped supplying data due to malfunctions, ``Station (no recent data)'' at the station. For the latter, it was necessary to make predictions for these stations based on recent data at other stations. For both station categories, we provide our  accuracy for 1-day ahead, 5-days ahead and 7-days ahead predictions. Validation uses root mean squared error, so the lower the error cost, the better the model. \n\n\\textbf{Analysis.} Stations with recent data far out-performed those without so we will focus our discussion on the former. As expected, the predictive accuracy for all models decreases as the forecast window increased, significantly for 5 and 7 day predictions.\nFeature set 2 (FS2) was the best performing  across all time intervals. This highlighted the benefit of our gap filling method but also indicated that the camera data did not deliver any added performance.  \nIn terms of models, no single model performed best  although our Temporal Graph model (TemAtt) was the best-performing model for 1-day predictions and the Spatio-Temporal Graph model (SpaTempAtt) was best performing for 5-day predictions. Finally, all test models, including both of our Graph models, showed significant degradation in performance as sensitivity decreases. \n\n%\\begin{table}[]\n%\\begin{tabular}{|l|l|l|}\n%%\\toprule\n%Input Data & Model     & Submission Run\\# \\\\ \\midrule\n%FS1        & LSTM      & 1                \\\\ \n%FS2        & LSTM      & 3                \\\\ \n%FS2        & TemAtt    & 4                \\\\ \n%FS2        & SpaTemAtt & 2                \\\\ \n%FS3        & SpaTemAtt & 5                \\\\ \n%\\bottomrule\n%\\end{table}\n\n%\\subsection{Analysis and Discussion}\n\n\n%(Cuong Input) \n\n%The results show that gapfilled data have a positive impact on the predictions. In naive gap-filling, missing values are replaced by long-term means, this causes noises in the input and makes models hard to react to short-term changes. Our gapfilled method, which takes into account information from other locations at that moment, not only approximates missing values better, but also add spatial information from other locations to the models. In fact, our gapfilled data improve LSTM's  as well as other models' performance\n\n%Timeseries-specialized models are better at predictions in short-term, considering spatial correlations is not necessary a good thing. LSTM and TemAtt architectures beat MLP and simpler models marginally. This is understandable as MLP pay equal attention globally having no concepts of time, while timeseries models process a timestep information and inter-timestep correlations separately. However, adding spatial information in short-term prediction can be an overdo, pointed by lower performances of LSTM-GNN and SpaTemAtt models. In LSTM-GNN model, information extracted from LSTM-module is spatially combined by weighted summing all LSTM outputs at other locations. Therefore the model can look at other locations when predicting at a location. But the weights are only represented by a matrix, solely depended on spatial correlation and do not vary accordingly to the inputs, this lead to a drop in the performance. SpaTemAtt model slightly improves by fixing this problem and but still under-perform compared to LSTM and TemmAtt in short-term predictions. This can imply that short-term fluctuations mostly depend on historical data at the location, SpaTemAtt is distracted by information from other stations. In medium term, the results show slightly improvement but not significant. In the long term, the spatial correlation may even disappear.\n\n%In the other hand, reported results on FS3 with information from images negatively affect the results. This can be caused by two reasons. The first one is that the difference in the amount of training data. Figure \\ref{fig:availability} shows that the overlap between air quality and traffic data is little, which reduces the amount of data when training. The second explaination is that the additional traffic features does not really provide new information. Even if we assume air quality is affected by traffic, this information may already embodied in the air quality inputs, and so the traffic inputs just cause noises and overfitting to the models.",
                  'Conclusions and Lessons Learned': 'Some high-level conclusions from our results show that our gap filled data have a positive impact on the predictions, it approximates missing values better and adds spatial information from other locations.\nTime series specialised models are better at predictions in short-term as LSTM and TemAtt architectures out-perform MLP and simpler models marginally.\n\nWhile the results provide answers to some questions, there are may other unanswered ones.  For example it is not clear why  the TemAtt model outperformed the SpaTemAtt model. It may be because the inclusion of spatial info made SpaTemAtt weaker somehow. We were surprised by the comparatively poor performance of the LSTM-GNN.  For us, the CCTV images added no value as FS3 featured outside the top-2 across all prediction intervals possibly because there is little overlap between CCTV and air quality data thus little for the models to train on. One other reason for this could be that additional traffic features do not really provide new information. Even if we assume air quality is affected by traffic, was our vehicle counting too naive or is there another explanation.  \n\n\n\n\n%Why was LSTM-GNN so bad?\n%Why was TemAtt able to beat SpaTemAtt? It must be because the inclusion of spatial info made us weaker somehow. \n\n%Some thoughts are Discussion section:\n%\\begin{itemize}\n%    \\item FS3: could we have gotten more from the camera stills? Was car counting too naive?\n%\\end{itemize}\n\n\n\n\\subsection*{Acknowledgements}\n\nThis work was supported in part by  Science Foundation Ireland through the the Insight Centre for Data Analytics (SFI/12/RC/2289\\_P2) and the Centre for Research Training in Machine Learning (18/CRT/6183). We thank the organisers for running the task.\n\n\n\\def\\bibfont{\\small} % comment this line for a smaller fontsize\n\\bibliography{references} \n\n\\end{document}'}}
