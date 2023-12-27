import re
import openai
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path = ".env")
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]
from llm_api import llm_request
from util import multiprocess, get_cpu_count

def rewrite_language_issue(section_label, section_content, section_review):

    example = {'example_content': """\\section\{Introduction\}\nThe Urban Life and Air Pollution task at MediaEval 2022 required participants to predict the air quality index (AQI) value at +1, +5 and +7 days using an archive of air quality, weather and images from 16 CCTV cameras, one image taken every 60 seconds \cite{UA22}. Participating groups were required to download the data from online sources for local processing.
    Gaps in air quality datasets are common with the problem exacerbated for data gathered in poorer or developing countries \cite{PINDER2019116794, Falge2001, Hui2004, Moffat2007, Kim2020}. In this paper we describe how we addressed the very large gaps in data that we encountered in the data we downloaded.""", 
            'section_label': 'Introduction',
            'example_review': """The Urban Life and Air Pollution task at MediaEval 2022 required participants to predict the air quality index (AQI) value at +1, +5 and +7 days using an archive of air quality, weather and images from 16 CCTV cameras, one image taken every 60 seconds \cite{UA22}. Participating groups were required to download the data from online sources for local processing.
Gaps in air quality datasets are common with the problem exacerbated for data gathered in poorer or developing countries \cite{PINDER2019116794, Falge2001, Hui2004, Moffat2007, Kim2020}. <span style="color:red;">In this paper we describe how we addressed the very large gaps in data that we encountered in the data we downloaded.</span>(<span style="color:orange;">This sentence is repetitive with the use of "data" three times. It could be rephrased for clarity and conciseness.</span>)""",
            'revise_result': """\\section\{Introduction\}\nThe Urban Life and Air Pollution task at MediaEval 2022 required participants to predict the air quality index (AQI) value at +1, +5, and +7 days using an archive of air quality, weather, and images from 16 CCTV cameras, one image taken every 60 seconds \cite{UA22}. Participating groups were required to download the data from online sources for local processing.
    Gaps in air quality datasets are common, with the problem exacerbated for data gathered in poorer or developing countries \cite{PINDER2019116794, Falge2001, Hui2004, Moffat2007, Kim2020}. In this paper, we detail our approach to addressing the substantial gaps encountered within the downloaded datasets."""}
    
    language_issue_prompt = \
    """"Your task is to revise a section of an academic paper (in LaTeX format) based on the review advise. Follow these requirements: 

        I will provide you with a section (titled "{section_label}") of the paper in Latex format. 
        Along with this text, I will give you a review comment where specific language issues have already been marked. In this text, specific language issues have already been marked on the basis of the section text. The parts needing revision are indicated in red, wrapped with <span style="color:red;"> and </span>. The explanations for these revisions are in orange, wrapped with <span style="color:orange;"> and </span>.
        Based on these annotations in the review comment, the task is to modify the section text to make it more grammatically correct, ensuring that the meaning of the revised text remains consistent with the original. 
        It is also crucial to retain the unmodified parts of the text exactly as they are, without any alterations.
        
        Formatting Requirements:

        Present your output in Latex format aligned with provided section. You should only output the revised text directly without anything additional.
        
        Here's an example for reference:

        provided section: {example_content}
        
        review advise: {example_review}

        revised result: {revise_result}
        
        (End of example)
        
        The content of the section is: {section_content}

        The review advise is: {review_advise}

        Ensure your response fully complies with compliance requirements and fulfill the task effectively."""
        
    prompt = language_issue_prompt.format(section_content = section_content, section_label = section_label, review_advise = section_review, \
        example_review = example["example_review"], example_content = example["example_content"], revise_result = example["revise_result"])
    
    response = llm_request(prompt)
    return response

def rewrite_language_issue_async(dt_section, dt_review):

    def fun(section_pair):
        section_label, section_content, section_review = section_pair
        return rewrite_language_issue(section_label, section_content, section_review)

    ls_section_pair = []
    for section_label in dt_section.keys():
        ls_section_pair.append((section_label, dt_section[section_label], dt_review[section_label]))
    ls_analysis_result = multiprocess(
        func=fun,
        paras=ls_section_pair,
        n_processes=min(len(ls_section_pair), get_cpu_count())
    )
    return ls_analysis_result


def rewrite_logic_issue(section_label, section_content, section_review, section_structure):
    example_structure = {'nodes': [
                      {'name': 'Challenge Overview',
                       'content': 'The Urban Life and Air Pollution task at MediaEval 2022 is introduced, which required participants to predict the air quality index (AQI) value at future intervals using a variety of data sources.',
                       'parents': []}, {'name': 'Data Gaps Issue',
                                        'content': 'The paper acknowledges the common issue of gaps in air quality datasets, which is particularly problematic in poorer or developing countries.',
                                        'parents': ['Challenge Overview']}, {'name': 'Research Contribution',
                                                                             'content': 'The paper outlines its contribution by describing the approach taken to address the large gaps in the air quality data encountered.',
                                                                             'parents': ['Data Gaps Issue']}], 'edges': [
                      {'from': 'Challenge Overview', 'to': 'Data Gaps Issue'}, {'from': 'Data Gaps Issue', 'to': 'Research Contribution'}]}
    example_content = """\\label={sec:intro}

    \\begin{comment}
    According to the World Health Organisation (WHO), 91\% of the world's population reside in conditions where WHO's air quality guidelines levels were not met \cite{organizacion2021global}. This report on 2016 also showed that ambient (outdoor) air pollution in both cities and rural areas was estimated to cause 4.2 million premature deaths worldwide. The research concluded that policies and investments supporting cleaner transport, energy-efficient homes, power generation, industry and better municipal waste management would would be crucial to the reduction of outdoor air pollution. In a separate report, it is estimated that air pollution globally accounts for roughly seven million premature deaths a year \cite{Gar21}, where it was again stated that the majority of those deaths are caused by outdoor air pollution with the rest generally attributed to poor air quality from indoor cooking. While the majority of these deaths occur in developing countries, with China and India accounting for roughly 50\%, developed countries also have a problem with deaths resulting from air pollution.
    In this research, the focus will mostly be on the modelling of concentrations in particulate matter - tiny particles in the air generated both by natural processes and human activity. These particles are generally 
    12  categorised (in the public health domain) by their diameter; fine particles with diameter less than 2.5 $\mu$m are referred to as "PM2.5" and coarse particles with diameter between 2.5 and 10 $\mu$m are referred to as "PM10".\\
    \\end{comment}

    The Urban Life and Air Pollution task at MediaEval 2022 required participants to predict the air quality index (AQI) value at +1, +5 and +7 days using an archive of air quality, weather and images from 16 CCTV cameras, one image taken every 60 seconds  \cite{UA22}. Participating groups were required to download the data from online sources for local processing.
    Gaps in air quality datasets are common with the problem exacerbated for data gathered in poorer or developing countries \cite{PINDER2019116794, Falge2001, Hui2004, Moffat2007, Kim2020}. In this paper we describe how we addressed the very large gaps in data that we encountered in the data we downloaded."""

    example_feedback = """The logical flow of the Introduction section is clear and follows a reasonable sequence. It begins with an overview of the challenge that sets the context for the research. It then transitions to discussing a specific issue related to the challenge, namely the gaps in air quality datasets, which is particularly problematic in poorer or developing countries. Finally, it outlines the paper's contribution by stating that the approach taken to address the large gaps in the air quality data will be described. The transitions between ideas are smooth, and the content is coherent. However, the last sentence could be improved to avoid repetition and enhance clarity."""

    example_result = """\\label={sec:intro}
    \\begin{comment}
    According to the World Health Organisation (WHO), 91\% of the world's population reside in conditions where WHO's air quality guidelines levels were not met \cite{organizacion2021global}. This report from 2016 also indicated that ambient (outdoor) air pollution in both cities and rural areas was estimated to cause 4.2 million premature deaths worldwide. The research emphasized that implementing policies and investments to support cleaner transport, energy-efficient homes, power generation, industry, and better municipal waste management could significantly reduce outdoor air pollution. Another report estimates that air pollution globally accounts for approximately seven million premature deaths annually \cite{Gar21}, with the majority of these deaths being attributed to outdoor air pollution and the remainder largely due to poor indoor air quality from cooking practices. While most of these deaths occur in developing countries, with China and India contributing to about 50\%, developed nations are not exempt from the detrimental effects of air pollution.
    In this research, we primarily examine the modelling of particulate matter concentrations—tiny particles in the air produced by both natural processes and human activities. These particles are typically classified by their diameter; fine particles with a diameter of less than 2.5 $\mu$m are denoted as "PM2.5," and coarse particles with a diameter between 2.5 and 10 $\mu$m are labeled as "PM10".
    \\end{comment}

    The Urban Life and Air Pollution task at MediaEval 2022 challenged participants to forecast the air quality index (AQI) for future intervals using a diverse set of data, including air quality metrics, weather conditions, and visual data from 16 CCTV cameras capturing images every minute \cite{UA22}. Participants were tasked with retrieving this data from various online repositories for subsequent local analysis.
    Data gaps are a prevalent issue in air quality datasets, with this challenge being more pronounced in data collected from less affluent or developing regions \cite{PINDER2019116794, Falge2001, Hui2004, Moffat2007, Kim2020}. In this paper, we delineate our methodology for addressing the substantial data voids we encountered within the datasets we utilized.
    """

    example = {"example_review" : example_feedback, "example_content": example_content, "section_label": "Introduction",
            "example_structure": example_structure, "example_result": example_result}

    logic_issue_prompt = """Your task is to revise a section of an academic paper (in LaTeX format) based on the review advise. Follow these requirements:

                I will provide you with a section (titled "{section_label}") of the paper in Latex format. 
                Additionally, I will supply a review of this document segment. The review will critique the logical flow of the text. 
                Based on this review, your task is to modify the text to enhance its logical coherence while preserving its core message. 
                The modification should adhere to three criteria: 
                (1) fidelity, ensuring the revised text remains true to the original intent and content, and 
                (2) improved logical flow, making the content more academically sound, clear, and engaging, and
                (3) integrity, ensuring that no information loss occurs, and maintaining the integrity of the original information as much as possible.
                
                Formatting Requirements:

                Present your output in Latex format aligned with provided section. You should only output the revised text directly without anything additional.
                        
                Here's an example for reference:

                provided section: {example_content}

                logical structure: {example_structure}

                review advise: {example_review}

                revised result: {revise_result}
                (End of example)
                
                The section you will be revising and corresponding review advise are as follows: 
                
                The content of the section is: {section_content}

                The logical structure of the specific section is: {section_structure}

                The review advise is: {review_advise}

                Ensure your response fully complies with requirements and fulfill the task effectively. You should output your further modified text directly in Latex format."""
                
    prompt = logic_issue_prompt.format(section_label=section_label, section_content=section_content, section_structure=section_structure, review_advise=section_review, \
        example_content=example["example_content"], example_structure=example["example_structure"], example_review=example["example_review"], revise_result=example["example_result"])

    response = llm_request(prompt)
    return response
    
def rewrite_logic_issue_async(dt_section, dt_section_structure, dt_review):

    def fun(section_pair):
        section_label, section_content, section_review, section_structure = section_pair
        return rewrite_logic_issue(section_label, section_content, section_review, section_structure)

    ls_section_pair = []
    for section_label in dt_section.keys():
        ls_section_pair.append((section_label, dt_section[section_label], dt_review[section_label], dt_section_structure[section_label]))
    ls_analysis_result = multiprocess(
        func=fun,
        paras=ls_section_pair,
        n_processes=min(len(ls_section_pair), get_cpu_count())
    )
    return ls_analysis_result
    

def reflect(section_label, original_text, logical_structure, review_advise, modified_text):
    revise_requirements = """The modification should adhere to three criteria: 
                (1) fidelity, ensuring the revised text remains true to the original intent and content, and 
                (2) improved logical flow, making the content more academically sound, clear, and engaging, and
                (3) integrity, ensuring that no information loss occurs, and maintaining the integrity of the original information as much as possible."""
    
    example_structure = {'nodes': [
                      {'name': 'Challenge Overview',
                       'content': 'The Urban Life and Air Pollution task at MediaEval 2022 is introduced, which required participants to predict the air quality index (AQI) value at future intervals using a variety of data sources.',
                       'parents': []}, {'name': 'Data Gaps Issue',
                                        'content': 'The paper acknowledges the common issue of gaps in air quality datasets, which is particularly problematic in poorer or developing countries.',
                                        'parents': ['Challenge Overview']}, {'name': 'Research Contribution',
                                                                             'content': 'The paper outlines its contribution by describing the approach taken to address the large gaps in the air quality data encountered.',
                                                                             'parents': ['Data Gaps Issue']}], 'edges': [
                      {'from': 'Challenge Overview', 'to': 'Data Gaps Issue'}, {'from': 'Data Gaps Issue', 'to': 'Research Contribution'}]}
    
    example_content = """\\label={sec:intro}

    \\begin{comment}
    According to the World Health Organisation (WHO), 91\% of the world's population reside in conditions where WHO's air quality guidelines levels were not met \cite{organizacion2021global}. This report on 2016 also showed that ambient (outdoor) air pollution in both cities and rural areas was estimated to cause 4.2 million premature deaths worldwide. The research concluded that policies and investments supporting cleaner transport, energy-efficient homes, power generation, industry and better municipal waste management would would be crucial to the reduction of outdoor air pollution. In a separate report, it is estimated that air pollution globally accounts for roughly seven million premature deaths a year \cite{Gar21}, where it was again stated that the majority of those deaths are caused by outdoor air pollution with the rest generally attributed to poor air quality from indoor cooking. While the majority of these deaths occur in developing countries, with China and India accounting for roughly 50\%, developed countries also have a problem with deaths resulting from air pollution.
    In this research, the focus will mostly be on the modelling of concentrations in particulate matter - tiny particles in the air generated both by natural processes and human activity. These particles are generally 
    12  categorised (in the public health domain) by their diameter; fine particles with diameter less than 2.5 $\mu$m are referred to as "PM2.5" and coarse particles with diameter between 2.5 and 10 $\mu$m are referred to as "PM10".\\
    \\end{comment}

    The Urban Life and Air Pollution task at MediaEval 2022 required participants to predict the air quality index (AQI) value at +1, +5 and +7 days using an archive of air quality, weather and images from 16 CCTV cameras, one image taken every 60 seconds  \cite{UA22}. Participating groups were required to download the data from online sources for local processing.
    Gaps in air quality datasets are common with the problem exacerbated for data gathered in poorer or developing countries \cite{PINDER2019116794, Falge2001, Hui2004, Moffat2007, Kim2020}. In this paper we describe how we addressed the very large gaps in data that we encountered in the data we downloaded."""

    example_feedback = """The logical flow of the Introduction section is clear and follows a reasonable sequence. It begins with an overview of the challenge that sets the context for the research. It then transitions to discussing a specific issue related to the challenge, namely the gaps in air quality datasets, which is particularly problematic in poorer or developing countries. Finally, it outlines the paper's contribution by stating that the approach taken to address the large gaps in the air quality data will be described. The transitions between ideas are smooth, and the content is coherent. However, the last sentence could be improved to avoid repetition and enhance clarity."""

    example_modified_text = """\label={sec:intro}
    \\begin{comment}
    According to the World Health Organisation (WHO), 91\% of the world's population lives in areas where air quality falls below WHO guidelines \cite{organizacion2021global}. The 2016 report highlighted that ambient air pollution in urban and rural settings contributes to an estimated 4.2 million premature deaths globally. It underscored the necessity for policies and investments that promote cleaner transportation, energy-efficient housing, power generation, industrial processes, and improved municipal waste management to mitigate outdoor air pollution. Furthermore, another report estimates that air pollution is responsible for approximately seven million premature deaths annually \cite{Gar21}, with a significant portion of these deaths due to outdoor air pollution and the remainder largely linked to indoor air pollution from cooking practices. While a substantial number of these fatalities occur in developing nations, with China and India together accounting for about half, developed countries are not immune to the health impacts of air pollution.
    In our research, we concentrate on modeling particulate matter concentrations—microscopic particles in the air produced by natural and human activities. Public health classifications typically categorize these particles by their diameter; particles less than 2.5 $\mu$m in diameter are known as "PM2.5," and those with diameters between 2.5 and 10 $\mu$m are called "PM10".
    \\end{comment}

    The Urban Life and Air Pollution task at MediaEval 2022 challenged participants to forecast the air quality index (AQI) for future intervals of +1, +5, and +7 days by leveraging a dataset comprising air quality metrics, meteorological data, and imagery from 16 CCTV cameras, with one image captured every 60 seconds \cite{UA22}. Participants were required to retrieve this data from online platforms for subsequent local analysis.
    Data gaps are a prevalent issue in air quality datasets, with this problem being particularly acute in less economically developed regions \cite{PINDER2019116794, Falge2001, Hui2004, Moffat2007, Kim2020}. In this paper, we delineate the methodology we employed to address the substantial data voids we encountered within the datasets we examined."""
    
    example_reflect = """Reflecting on the modifications made to the Introduction section of the academic paper, the following points are considered:

    (1) Does the modified version meet the revise requirements specified for the changes?

    The modified version appears to meet the specified revise requirements. The revised text remains true to the original intent and content (fidelity), as it still provides the same statistical information and context regarding air pollution and its health impacts. The logical flow has been improved by rephrasing and restructuring sentences to make the content clearer and more engaging (improved logical flow). The integrity of the information has been maintained, with no loss of original information (integrity).

    (2) Does the modified version address the issues pointed out in the review advice?

    The review advice highlighted the need to improve the last sentence of the original section to avoid repetition and enhance clarity. The modified version has addressed this by rewording the sentence to "This paper details our approach to effectively bridge the significant data gaps encountered in the datasets we analyzed." This new sentence eliminates repetition and provides a clearer statement of the paper's contribution.

    (3) Is there anything that needs to be further improved in the modified version based on review advice and revision requirements?

    Upon reviewing the modified version, it seems that the changes have addressed the review advice and revision requirements adequately. The text is coherent, the logical flow is maintained, and the information is presented clearly without redundancy. Therefore, no further improvements are identified based on the review advice and revision requirements provided.

    The reflection results have been directly generated, and based on the analysis, the modifications have fully met the review advice and the specified revise requirements. No further modification needed."""
    
    example = f""" "example_content": {example_content} \n\n"logical structure": {example_structure} \n\n"review advice": {example_feedback} \n\n"modified version": {example_modified_text} \n
    "reflection": {example_reflect}"""
         
    reflect_prompt = """You are tasked with refining a section of an academic paper. a section (titled "{section_label}") of an academic paper in Latex format, has been previously revised based on a review advise.

                        The original section, logical structure, review advice, revise requirements for modification were provided. You made changes accordingly, resulting in a modified version of the text. 

                        Your task now is to reflect on the changes you've made. Consider the following points:

                        (1) Does the modified version meet the revise requirements specified for the changes?
                        (2) Does the modified version address the issues pointed out in the review advice?
                        (3) Is there anything that needs to be further improved in the modified version based on review advice and revision requirements?

                        Formatting requirements of the response:
                        Directly generate your reflection results. If you believe that your modifications have fully met the review advice and the specified revise requirements, please respond end with "No further modification needed".

                        Here are some examples for reference:
                        {examples}
                        (End of example)

                        The original section, logical structure, review advice, revise requirements and your modified version are as follows:

                        original section: {original_text}

                        logical structure of original text: {logical_structure}

                        review advice: {review_advise}

                        revise requirements: {revise_requirements}

                        modified version: {modified_text}

                        Please analyze your previous modification and provide your reflection.
                        """
    prompt = reflect_prompt.format(section_label=section_label, original_text=original_text, logical_structure=logical_structure, review_advise=review_advise, revise_requirements=revise_requirements, \
        modified_text=modified_text, examples=example)

    response = llm_request(prompt)
    return response

def modify_based_on_reflect(section_label, original_text, logical_structure, review_advise, modified_text, unsatisfied_points):
    revise_requirements = """The modification should adhere to three criteria: 
                (1) fidelity, ensuring the revised text remains true to the original intent and content, and 
                (2) improved logical flow, making the content more academically sound, clear, and engaging, and
                (3) integrity, ensuring that no information loss occurs, and maintaining the integrity of the original information as much as possible."""
    
    example_structure = {'nodes': [
                      {'name': 'Challenge Overview',
                       'content': 'The Urban Life and Air Pollution task at MediaEval 2022 is introduced, which required participants to predict the air quality index (AQI) value at future intervals using a variety of data sources.',
                       'parents': []}, {'name': 'Data Gaps Issue',
                                        'content': 'The paper acknowledges the common issue of gaps in air quality datasets, which is particularly problematic in poorer or developing countries.',
                                        'parents': ['Challenge Overview']}, {'name': 'Research Contribution',
                                                                             'content': 'The paper outlines its contribution by describing the approach taken to address the large gaps in the air quality data encountered.',
                                                                             'parents': ['Data Gaps Issue']}], 'edges': [
                      {'from': 'Challenge Overview', 'to': 'Data Gaps Issue'}, {'from': 'Data Gaps Issue', 'to': 'Research Contribution'}]}
    
    example_content = """\\label={sec:intro}

    \\begin{comment}
    According to the World Health Organisation (WHO), 91\% of the world's population reside in conditions where WHO's air quality guidelines levels were not met \cite{organizacion2021global}. This report on 2016 also showed that ambient (outdoor) air pollution in both cities and rural areas was estimated to cause 4.2 million premature deaths worldwide. The research concluded that policies and investments supporting cleaner transport, energy-efficient homes, power generation, industry and better municipal waste management would would be crucial to the reduction of outdoor air pollution. In a separate report, it is estimated that air pollution globally accounts for roughly seven million premature deaths a year \cite{Gar21}, where it was again stated that the majority of those deaths are caused by outdoor air pollution with the rest generally attributed to poor air quality from indoor cooking. While the majority of these deaths occur in developing countries, with China and India accounting for roughly 50\%, developed countries also have a problem with deaths resulting from air pollution.
    In this research, the focus will mostly be on the modelling of concentrations in particulate matter - tiny particles in the air generated both by natural processes and human activity. These particles are generally 
    12  categorised (in the public health domain) by their diameter; fine particles with diameter less than 2.5 $\mu$m are referred to as "PM2.5" and coarse particles with diameter between 2.5 and 10 $\mu$m are referred to as "PM10".\\
    \\end{comment}

    The Urban Life and Air Pollution task at MediaEval 2022 required participants to predict the air quality index (AQI) value at +1, +5 and +7 days using an archive of air quality, weather and images from 16 CCTV cameras, one image taken every 60 seconds  \cite{UA22}. Participating groups were required to download the data from online sources for local processing.
    Gaps in air quality datasets are common with the problem exacerbated for data gathered in poorer or developing countries \cite{PINDER2019116794, Falge2001, Hui2004, Moffat2007, Kim2020}. In this paper we describe how we addressed the very large gaps in data that we encountered in the data we downloaded."""

    example_feedback = """The logical flow of the Introduction section is clear and follows a reasonable sequence. It begins with an overview of the challenge that sets the context for the research. It then transitions to discussing a specific issue related to the challenge, namely the gaps in air quality datasets, which is particularly problematic in poorer or developing countries. Finally, it outlines the paper's contribution by stating that the approach taken to address the large gaps in the air quality data will be described. The transitions between ideas are smooth, and the content is coherent. However, the last sentence could be improved to avoid repetition and enhance clarity."""

    example_modified_text = """\label={sec:intro}
    \begin{comment}
    Air quality and its global impact remains a critical issue. While the World Health Organisation (WHO) provides guidelines, the implementation varies significantly across regions. The complexity of air pollution, influenced by various factors including transportation, housing, power generation, industrial processes, and waste management, adds to the challenge. Additionally, indoor pollution, often overlooked, contributes to health issues, especially in regions where cooking practices rely on certain fuels. Developing countries face unique challenges in this regard, with China and India being notable examples. However, this issue is not confined to these nations alone.
    This paper takes a broad approach, touching upon various aspects of air pollution without a specific focus on any single element. The discussion ranges from particulate matter to broader environmental policies.
    \end{comment}

    The MediaEval 2022 competition included a task on Urban Life and Air Pollution, where the challenge was to forecast air quality indices using diverse datasets. The specifics of this task and the data involved are complex and multifaceted, encompassing various elements from CCTV imagery to meteorological data. Addressing the issue of data gaps, prevalent in air quality research, especially in underdeveloped regions, remains a significant challenge. This paper attempts to navigate these complexities, albeit without a focused methodology or clear research direction.
    """
    
    example_reflect = """Reflecting on the modifications made to the Introduction section of the academic paper, the following points are considered:

    (1) Does the modified version meet the revise requirements specified for the changes?

    The modified version seems to have strayed from the original intent and content, which is a deviation from the fidelity requirement. The original statistical information and specific context regarding air pollution and its health impacts have been diluted. The revised text does not clearly convey the same urgency or detailed information as the original, which could be seen as a loss of integrity. The logical flow has been altered, but the lack of specificity and the broad approach may not make the content more academically sound or engaging.

    (2) Does the modified version address the issues pointed out in the review advice?

    The review advice suggested improving the last sentence to avoid repetition and enhance clarity. However, the modified version has introduced a new issue by presenting a vague and unfocused narrative. The last sentence of the original section, which was meant to be improved, has been replaced with a statement that lacks clarity regarding the paper's contribution. This does not align with the advice to clarify the approach taken to address the large gaps in the air quality data.

    (3) Is there anything that needs to be further improved in the modified version based on review advice and revision requirements?

    The modified version requires significant improvement to align with the review advice and revision requirements. It needs to restore the specific statistical information and context regarding air pollution and its health impacts. The paper's contribution should be clearly stated, focusing on the methodology employed to address the data gaps in air quality research. The logical flow should be enhanced to ensure that the content is academically sound, clear, and engaging, without losing the integrity of the original information.

    Based on the analysis, the modifications have not fully met the review advice and the specified revise requirements. Further modification is needed to ensure fidelity, improved logical flow, and integrity."""
    
    example_further_modify = """\\label{sec:intro}
    \\begin{comment}
    According to the World Health Organisation (WHO), 91\% of the world's population reside in conditions where WHO's air quality guidelines levels were not met \cite{organizacion2021global}. This report on 2016 also showed that ambient (outdoor) air pollution in both cities and rural areas was estimated to cause 4.2 million premature deaths worldwide. The research concluded that policies and investments supporting cleaner transport, energy-efficient homes, power generation, industry and better municipal waste management would be crucial to the reduction of outdoor air pollution. In a separate report, it is estimated that air pollution globally accounts for roughly seven million premature deaths a year \cite{Gar21}, where it was again stated that the majority of those deaths are caused by outdoor air pollution with the rest generally attributed to poor air quality from indoor cooking. While the majority of these deaths occur in developing countries, with China and India accounting for roughly 50\%, developed countries also have a problem with deaths resulting from air pollution.
    In this research, the focus will mostly be on the modelling of concentrations in particulate matter - tiny particles in the air generated both by natural processes and human activity. These particles are generally 
    12  categorised (in the public health domain) by their diameter; fine particles with diameter less than 2.5 $\mu$m are referred to as "PM2.5" and coarse particles with diameter between 2.5 and 10 $\mu$m are referred to as "PM10".
    \\end{comment}

    The Urban Life and Air Pollution task at MediaEval 2022 presented a challenge that required participants to predict the air quality index (AQI) value at +1, +5, and +7 days using an archive of air quality, weather data, and images from 16 CCTV cameras, captured at one-minute intervals \cite{UA22}. This task underscored the critical need for accurate air quality forecasting, particularly in light of the frequent data gaps that are prevalent in datasets from less affluent regions \cite{PINDER2019116794, Falge2001, Hui2004, Moffat2007, Kim2020}. Our paper addresses these challenges by detailing a novel approach to mitigate the impact of substantial data gaps encountered in the datasets we analyzed. Through this work, we aim to contribute to the broader effort of improving air quality predictions, which is essential for public health and policy-making, especially in areas where data scarcity hinders environmental monitoring and management."""
        
    example = f""" "Original Section": {example_content} \n\n"Logical Structure of Original Text": {example_structure} \n\n"Review Advice": {example_feedback} \n\n"Revision Requirements": {revise_requirements}\n
    "Your Modified Version": {example_modified_text} \n\n"Unsatisfied Points of Modified Version": {example_reflect} \n\n"Furether Modified Result": {example_further_modify}"""
                
    remodify_prompt = """Objective: You are required to refine a section of an academic paper. This section, titled "{section_label}," has undergone previous revisions based on review advice. 
                        However, your current modifications have not fully met the review advice and revision requirements. 
                        The reasons for this shortfall will be provided to you. Your goal is to further modify the text to better align with the review advice and revision requirements. 
                        It's crucial to preserve the parts that don't need to be changed exactly as they are.

                        Formatting Requirements:
                        Present your output in Latex format aligned with provided section. You should directly output the further revised text without anything additional analyse process.

                        Here are some examples for reference:
                        {examples}
                        (End of examples)

                        Components:

                        "Original Section": {original_text}

                        "Logical Structure of Original Text": {logical_structure}

                        "Review Advice": {review_advise}

                        "Revision Requirements": {revise_requirements}

                        "Your Modified Version": {modified_text}

                        "Unsatisfied Points of Modified Version": {unsatisfied_points}
                        
                        You should output your further modified text directly in Latex format.
                        """
    prompt = remodify_prompt.format(section_label=section_label, original_text=original_text, logical_structure=logical_structure, review_advise=review_advise, revise_requirements=revise_requirements, \
        modified_text=modified_text, unsatisfied_points=unsatisfied_points, examples=example)
    
    response = llm_request(prompt)
    return response

def rewrite_logic_issue_reflect(section_label, section_content, section_review, section_structure, runs=1):
    modified_text = rewrite_logic_issue(section_label, section_content, section_review, section_structure)
    
    for run in range(runs):
        reflect_result = reflect(section_label, section_content, section_structure, section_review, modified_text)
        futher_modified_text = modify_based_on_reflect(section_label, section_content, section_structure, section_review, modified_text, reflect_result)
        
        modified_text = futher_modified_text
        section_review = section_review + '\n' + reflect_result
        
    return modified_text

def rewrite_logic_issue_async_reflect(dt_section, dt_section_structure, dt_review, runs=1):

    def fun(section_pair):
        section_label, section_content, section_review, section_structure = section_pair
        return rewrite_logic_issue_reflect(section_label, section_content, section_review, section_structure, runs)

    ls_section_pair = []
    for section_label in dt_section.keys():
        ls_section_pair.append((section_label, dt_section[section_label], dt_review[section_label], dt_section_structure[section_label]))
    ls_analysis_result = multiprocess(
        func=fun,
        paras=ls_section_pair,
        n_processes=min(len(ls_section_pair), get_cpu_count())
    )
    return ls_analysis_result

if __name__ == '__main__':
    test_content = """\\section\{method\}\n\\label\{test\}\n Planning a trip to the moon differs in no essential respect from planning a trip to the beach. 
    You hve to decide what to take along, what to leave behind. 
    Should the thermos jug go? The child's rubber horse? The dill pickles? 
    These are the sometimes fateful decisions on which the success or failure of the whole outing turns.
    Among the items they sent along, of course, was the little jointed flagpoles and the flag that could be stiffened to the breeze that did not blow. 
    (It is traditional among explorers to plant the flag.) Yet the two men who stepped out on the surface of the moon were in a class by themselves and should 
    have been equipped accordingly: they were of the new breed of men, those who had seen the earth whole."""
    test_review_language = """Planning a trip to the moon differs in no essential respect from planning a trip to the beach. 
    @You hve to decide what to take along, what to leave behind.@(There is a typographical error: "hve" should be "have.". It is generally more common to see "take with you" instead of "take along.") 
    Should the thermos jug go? The child's rubber horse? The dill pickles? 
    @These are the sometimes fateful decisions on which the success or failure of the whole outing turns.@
    (The first issue with this sentence is the subject-verb disagreement. "These" is plural, but "is" is singular. 
    The correct verb form should be "are" to match the plural subject.The last part of the sentence, "on which the success or failure of the whole outing turns," 
    is a bit awkward. A smoother way to write this might be: "These decisions can sometimes be crucial, determining the success or failure of the entire outing.")
    Among the items they sent along, of course, was the little jointed flagpoles and the flag that could be stiffened to the breeze that did not blow. 
    (It is traditional among explorers to plant the flag.) Yet the two men who stepped out on the surface of the moon were in a class by themselves and should 
    have been equipped accordingly: they were of the new breed of men, those who had seen the earth whole."""
    test_review_logic = """The passage is mostly coherent and logically consistent, but the phrase "should have been equipped accordingly" is a bit abrupt without further explanation \
of what specific equipment would have been appropriate for their unique status. This leaves the reader with a question of what the author means by \
"equipped accordingly," which slightly disrupts the flow."""

    # response = rewrite_language_issue('Introduction', test_content, test_review_language)
    # print(response)
    
    # test2_section = {'Introduction': test_content, 'Method': test_content}
    # test2_review = {'Introduction': test_review, 'Method': test_review}
    # response = rewrite_language_issue_async(test2_section, test2_review)
    # print(len(response))
    # print(response)
    
    #response = rewrite_logic_issue('Introduction', test_content, test_review_logic, '')
    #print(response)
    
    # test2_section = {'Introduction': test_content, 'Method': test_content}
    # test2_review = {'Introduction': test_review_logic, 'Method': test_review_logic}
    # test2_section_structure = {'Introduction': '', 'Method': ''}
    # response = rewrite_logic_issue_async(test2_section, test2_section_structure, test2_review)
    # print(response)
    example_structure = {'nodes': [
                      {'name': 'Challenge Overview',
                       'content': 'The Urban Life and Air Pollution task at MediaEval 2022 is introduced, which required participants to predict the air quality index (AQI) value at future intervals using a variety of data sources.',
                       'parents': []}, {'name': 'Data Gaps Issue',
                                        'content': 'The paper acknowledges the common issue of gaps in air quality datasets, which is particularly problematic in poorer or developing countries.',
                                        'parents': ['Challenge Overview']}, {'name': 'Research Contribution',
                                                                             'content': 'The paper outlines its contribution by describing the approach taken to address the large gaps in the air quality data encountered.',
                                                                             'parents': ['Data Gaps Issue']}], 'edges': [
                      {'from': 'Challenge Overview', 'to': 'Data Gaps Issue'}, {'from': 'Data Gaps Issue', 'to': 'Research Contribution'}]}
    example_content = """\\label={sec:intro}

    \\begin{comment}
    According to the World Health Organisation (WHO), 91\% of the world's population reside in conditions where WHO's air quality guidelines levels were not met \cite{organizacion2021global}. This report on 2016 also showed that ambient (outdoor) air pollution in both cities and rural areas was estimated to cause 4.2 million premature deaths worldwide. The research concluded that policies and investments supporting cleaner transport, energy-efficient homes, power generation, industry and better municipal waste management would would be crucial to the reduction of outdoor air pollution. In a separate report, it is estimated that air pollution globally accounts for roughly seven million premature deaths a year \cite{Gar21}, where it was again stated that the majority of those deaths are caused by outdoor air pollution with the rest generally attributed to poor air quality from indoor cooking. While the majority of these deaths occur in developing countries, with China and India accounting for roughly 50\%, developed countries also have a problem with deaths resulting from air pollution.
    In this research, the focus will mostly be on the modelling of concentrations in particulate matter - tiny particles in the air generated both by natural processes and human activity. These particles are generally 
    12  categorised (in the public health domain) by their diameter; fine particles with diameter less than 2.5 $\mu$m are referred to as "PM2.5" and coarse particles with diameter between 2.5 and 10 $\mu$m are referred to as "PM10".\\
    \\end{comment}

    The Urban Life and Air Pollution task at MediaEval 2022 required participants to predict the air quality index (AQI) value at +1, +5 and +7 days using an archive of air quality, weather and images from 16 CCTV cameras, one image taken every 60 seconds  \cite{UA22}. Participating groups were required to download the data from online sources for local processing.
    Gaps in air quality datasets are common with the problem exacerbated for data gathered in poorer or developing countries \cite{PINDER2019116794, Falge2001, Hui2004, Moffat2007, Kim2020}. In this paper we describe how we addressed the very large gaps in data that we encountered in the data we downloaded."""

    example_feedback = """The logical flow of the Introduction section is clear and follows a reasonable sequence. It begins with an overview of the challenge that sets the context for the research. It then transitions to discussing a specific issue related to the challenge, namely the gaps in air quality datasets, which is particularly problematic in poorer or developing countries. Finally, it outlines the paper's contribution by stating that the approach taken to address the large gaps in the air quality data will be described. The transitions between ideas are smooth, and the content is coherent. However, the last sentence could be improved to avoid repetition and enhance clarity."""

    example_modified_text = """\label={sec:intro}
    \begin{comment}
    Air quality and its global impact remains a critical issue. While the World Health Organisation (WHO) provides guidelines, the implementation varies significantly across regions. The complexity of air pollution, influenced by various factors including transportation, housing, power generation, industrial processes, and waste management, adds to the challenge. Additionally, indoor pollution, often overlooked, contributes to health issues, especially in regions where cooking practices rely on certain fuels. Developing countries face unique challenges in this regard, with China and India being notable examples. However, this issue is not confined to these nations alone.
    This paper takes a broad approach, touching upon various aspects of air pollution without a specific focus on any single element. The discussion ranges from particulate matter to broader environmental policies.
    \end{comment}

    The MediaEval 2022 competition included a task on Urban Life and Air Pollution, where the challenge was to forecast air quality indices using diverse datasets. The specifics of this task and the data involved are complex and multifaceted, encompassing various elements from CCTV imagery to meteorological data. Addressing the issue of data gaps, prevalent in air quality research, especially in underdeveloped regions, remains a significant challenge. This paper attempts to navigate these complexities, albeit without a focused methodology or clear research direction.
    """
    
    example_content = """The downloaded air quality data are collected at 10 monitoring stations in Dalat City, Vietnam from March 2020  to 7th Nov 2022. The data includes air pollutant concentration for NO$_2$(ppm), CO, SO$_2$, O$_3$, PM1.0, PM2.5, PM10 as well as environmental measures namely temperature, humidity, UV, rainfall. In addition, traffic data, in form of images, was recorded every  minute from 16 CCTV cameras across Dalat City. Figure~\ref{fig:availability} shows the availability of the  dataset as downloaded by our group. This shows huge gaps in data availability.  In our model building we use the first 80\% of available data for training machine learning models and the remaining 20\% for validation.

    \\begin{figure}[ht!]
        \centering
        \includegraphics[width=1.0\textwidth]{Data-gaps.jpg}
        \caption{Missing data (shown in yellow) from across all 10 air quality measurement stations and 16 CCTV cameras for an 8-month period, early March to early November 2022.}
        \label{fig:availability}
    \\end{figure}"""
    
    example_feedback = """I hope to make this chapter written more brilliantly."""
    
    example_structure = {'nodes': [{'name': 'Data Collection', 'content': 'Air quality data collected from 10 monitoring stations and traffic data from 16 CCTV cameras in Dalat City, Vietnam, from March 2020 to 7th Nov 2022.', 'parents': []}, {'name': 'Data Components', 'content': 'The data includes concentrations of various air pollutants and environmental measures, as well as traffic data in the form of images.', 'parents': ['Data Collection']}, {'name': 'Data Availability Visualization', 'content': "Figure~\\ref{fig:availability} displays the dataset's availability, highlighting significant gaps in the data.", 'parents': ['Data Collection']}, {'name': 'Data Gaps Highlight', 'content': 'The visualization shows huge gaps in data availability, which is a challenge for analysis.', 'parents': ['Data Availability Visualization']}, {'name': 'Model Building Approach', 'content': 'For machine learning model building, the first 80% of available data is used for training and the remaining 20% for validation.', 'parents': ['Data Gaps Highlight']}, {'name': 'Figure Reference', 'content': 'The figure referenced provides a visual representation of missing data across all monitoring stations and CCTV cameras for an 8-month period.', 'parents': ['Data Availability Visualization']}], 'edges': [{'from': 'Data Collection', 'to': 'Data Components'}, {'from': 'Data Collection', 'to': 'Data Availability Visualization'}, {'from': 'Data Availability Visualization', 'to': 'Data Gaps Highlight'}, {'from': 'Data Gaps Highlight', 'to': 'Model Building Approach'}, {'from': 'Data Availability Visualization', 'to': 'Figure Reference'}]}
    
    response = rewrite_logic_issue_reflect('Data Analysis', example_content, example_feedback, example_structure, 1)
    
    # test2_section = {'Introduction': example_content, 'Method': example_content}
    # test2_review = {'Introduction': example_feedback, 'Method': example_feedback}
    # test2_section_structure = {'Introduction': example_structure, 'Method': example_structure}
    # response = rewrite_logic_issue_async_reflect(test2_section, test2_section_structure, test2_review)
    print(response)