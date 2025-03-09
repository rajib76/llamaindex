import asyncio
import json
import os

import openai
from dotenv import load_dotenv
from llama_index.core.workflow import Workflow, step, Event, StartEvent, StopEvent, draw_all_possible_flows
from pydantic import BaseModel

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

client = openai.OpenAI(api_key=OPENAI_API_KEY)


class LoopEvent(Event):
    loop_output: str


class SummaryEvent(Event):
    summary: str
    original_text: str


class ReviewEvent(Event):
    review: str
    score: int


class ReviewOutput(BaseModel):
    review: str
    score: int


content = """Introduction: Climate change has emerged as one of the most pressing challenges of the 21st century, 
affecting ecosystems, economies, and human livelihoods across the globe. While rising temperatures, extreme weather 
events, and biodiversity loss are often discussed, the indirect consequences of climate change on geopolitical 
stability, public health, and global food security are equally significant yet less acknowledged. 

Rising Temperatures and Extreme Weather Events: Over the past century, global temperatures have risen by an average 
of 1.2°C above pre-industrial levels, with some regions experiencing higher-than-average increases due to 
geographical and environmental factors. This rise has led to the melting of polar ice caps, contributing to a 
sea-level rise of approximately 3.3 mm per year. Coastal cities like Jakarta, Miami, and Dhaka face existential 
threats, with millions at risk of displacement by 2050. 

More alarming is the increase in the frequency and intensity of extreme weather events. The past decade has seen an 
unprecedented rise in Category 4 and 5 hurricanes, heatwaves causing wildfires in California, Australia, 
and the Amazon rainforest, and unpredictable monsoons disrupting economies dependent on agriculture. The 2022 
Pakistan floods submerged one-third of the country, displacing over 33 million people and causing an estimated $30 
billion in damages. 

Economic Consequences: The economic impact of climate change is immense and multifaceted. A 2021 study by the 
International Monetary Fund (IMF) estimated that climate-related disasters cost the global economy $2.5 trillion 
annually. Agricultural losses due to droughts and floods have led to food shortages, particularly in vulnerable 
regions like sub-Saharan Africa and South Asia. 

Moreover, industries reliant on stable climatic conditions—such as tourism, insurance, and agriculture—are facing 
severe disruptions. The ski industry in the Alps and the Rocky Mountains is shrinking as snowfall becomes unreliable. 
Insurance firms in hurricane-prone regions have either increased premiums by 200% or exited markets altogether, 
leaving homeowners unprotected. 

Public Health Crisis: Climate change has direct and indirect consequences on public health. Rising temperatures have 
increased vector-borne diseases such as malaria and dengue, particularly in previously temperate regions like 
southern Europe and North America. Additionally, air pollution linked to fossil fuel combustion is responsible for 
over 7 million premature deaths annually, according to the World Health Organization (WHO). 

Heat-related illnesses are becoming more common. A 2023 report from the Lancet Countdown on Health and Climate Change 
found that heatwaves led to an 11% increase in global cardiovascular-related deaths over the past decade. Vulnerable 
populations—elderly individuals, outdoor workers, and children—face the greatest risks. 

Geopolitical Instability and Climate Refugees: One of the most overlooked consequences of climate change is its role 
in exacerbating geopolitical tensions. Climate-induced water shortages have heightened disputes over transboundary 
rivers, such as the Indus Water Treaty between India and Pakistan and the Nile River conflict involving Egypt, Sudan, 
and Ethiopia. As agricultural productivity declines, nations are increasingly competing for limited natural 
resources, leading to diplomatic strains. 

The world is also witnessing a rising number of climate refugees. According to the United Nations High Commissioner 
for Refugees (UNHCR), an estimated 21.5 million people per year are forcibly displaced due to climate-related 
disasters. By 2050, this number is projected to surpass 200 million, putting enormous pressure on already strained 
immigration policies in the Global North. 

Technological and Policy Responses: In response to these challenges, governments, corporations, and research 
institutions have intensified efforts toward climate mitigation and adaptation. Key technological innovations include: 

Direct Air Capture (DAC) technologies, which extract CO₂ from the atmosphere. Companies like Climeworks and Carbon 
Engineering have pioneered large-scale carbon removal projects. Renewable Energy Expansion: In 2023, 
global investments in solar and wind energy surpassed $500 billion, outpacing fossil fuel investments for the first 
time. Sustainable Agriculture Initiatives: Innovations like drought-resistant crop varieties, vertical farming, 
and regenerative agriculture are helping reduce climate vulnerability in the food supply chain. On the policy front, 
nations have pledged emission reductions through the Paris Agreement. However, recent assessments show that most 
countries are not on track to meet their commitments. The United States Inflation Reduction Act (IRA) of 2022 
allocated $369 billion for clean energy initiatives, while the European Green Deal aims to cut emissions by 55% by 
2030. Yet, policy enforcement remains inconsistent, with some nations reversing climate pledges due to economic 
pressures. 

Corporate Responsibility and Greenwashing Concerns: While several multinational corporations have announced 
"net-zero" targets, many have been criticized for greenwashing—exaggerating or fabricating sustainability efforts. A 
2024 investigation by Greenpeace revealed that 60% of major oil and gas companies claiming carbon neutrality were 
still actively expanding fossil fuel operations. 

Consumer-driven activism is shaping corporate policies, as evidenced by increased demand for carbon-labeling on products and shareholder pressure for transparent ESG (Environmental, Social, Governance) reporting. However, critics argue that voluntary corporate pledges lack accountability, emphasizing the need for stricter regulatory oversight.

Conclusion: The climate crisis is not a singular problem but a complex web of interconnected challenges affecting 
economies, public health, global security, and technological innovation. Addressing climate change requires a 
multilateral approach—one that combines policy reform, corporate accountability, technological advancements, 
and global cooperation. As the world races against time to curb emissions and limit global temperature rise to below 
1.5°C, the next decade will be pivotal in determining the planet’s long-term habitability. """

review_prompt = f"""You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while 
reviewing, and refer to it as needed. 

Evaluation Criteria:

Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of 
structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not 
just be a heap of related information, but should build from sentence to a coherent body of information about a topic." 

Evaluation Steps:

1. Read the news article carefully and identify the main topic and key points. 2. Read the summary and compare it to 
the news article. Check if the summary covers the main topic and key points of the news article, and if it presents 
them in a clear and logical order. 3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 
is the highest based on the Evaluation Criteria. 

**REMEMBER** TO OUTPUT YOUR REVIEW COMMENT AND SCORE.

Example:


Source Text:

{{original_text}}

Summary:

{{summary}}


"""

summary_prompt = f"""
Summarize the following content.You will incorporate feedback, if it is available

content:
{{content}}

feedback:
{{feedback}}
"""


class SummaryWorkflow(Workflow):
    @step
    async def create_summary(self, start_ev: StartEvent | ReviewEvent) -> SummaryEvent:
        model = "gpt-4o-mini"
        print("----")
        print(start_ev)

        feedback = start_ev.review if isinstance(start_ev, ReviewEvent) else ""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI that summarizes text concisely."},
                {"role": "user", "content": summary_prompt.format(content=content, feedback=feedback)}
            ],
            temperature=0.5,
            max_tokens=150
        )
        print(response.choices[0].message.content)
        return SummaryEvent(summary=response.choices[0].message.content, original_text=content)

    @step
    async def review_summary(self, summary_ev: SummaryEvent) -> ReviewEvent | StopEvent:
        model = "gpt-4o"
        summary = summary_ev.summary
        content = summary_ev.original_text
        # review_prompt.format(summary=summary, original_text=content)
        print(review_prompt)
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are an AI that reviews summaries for clarity, conciseness, and completeness."},
                {"role": "user", "content": review_prompt.format(summary=summary, original_text=content)}
            ],
            temperature=0.3,
            max_tokens=150,
            response_format=ReviewOutput
        )

        feedback = json.loads(response.choices[0].message.content)
        print("feedback ", feedback)
        review_score = feedback["score"]
        review = feedback["review"]
        print("score ", review_score)
        if review_score >= 4:
            print("stopping now")
            return StopEvent(result=summary)
        else:
            print("recreating now")
            return ReviewEvent(score=review_score, review=review)


critique_workflow = SummaryWorkflow(timeout=10, verbose=False)
draw_all_possible_flows(
    critique_workflow,
    filename="critique_workflow.html"
)
# result = asyncio.run(critique_workflow.run(content=content))
# print(result)
