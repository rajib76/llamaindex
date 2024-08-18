# Implemented the reflection pattern in llamaindex workflow
import os
from typing import Union

from dotenv import load_dotenv
from llama_index.core.workflow import Context, step, Event, StartEvent, Workflow, StopEvent, draw_all_possible_flows, \
    draw_most_recent_execution
from llama_index.llms.openai import OpenAI

# Loading OpenAI API Key and MEM0 api key
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class MathEvent(Event):
    query: str


class PhysicsEvent(Event):
    query: str


class ReviewEvent(Event):
    query: str


class TeacherCrew(Workflow):
    llm = OpenAI()
    attempt = 0

    @step(pass_context=True)
    async def router(self, ctx: Context, ev: StartEvent) -> Union[MathEvent, PhysicsEvent]:
        query = ev.query
        category = ev.category
        ctx.data["attempt"] = 0
        ctx.data["category"] = category

        if category.lower() == "math":
            return MathEvent(query=query)
        else:
            return PhysicsEvent(query=query)

    @step(pass_context=True)
    async def physics_agent(self, ctx: Context, ev: PhysicsEvent) -> Union[ReviewEvent, StopEvent]:
        attempt = ctx.data.get("attempt")
        response = ctx.data.get("response")
        feedback = ctx.data.get("feedback")
        query = ev.query
        # print("In Physics Agent ")
        # print("attempt: ", attempt)
        # print("response: ",  response)
        # print("response: ", feedback)
        # print("query: ", query)
        # print("--------------------------------")
        prompt = f"""You are an experienced physics teacher.
        Please provide an answer to the user query \n query:{query}. Please ensure to incorporate any 
        feedback provided for your response \n
        feedback:{feedback}"""

        if attempt > 1:
            return StopEvent(result="final response : \n" + str(response) + "\n feedback:" + str(feedback))
        else:
            attempt = attempt + 1
            ctx.data["attempt"] = attempt
            response = await self.llm.acomplete(prompt)
            ctx.data["response"] = response
            print("attempt #: ", attempt, "\nfeeback :", feedback, "\nresponse :", response)
            return ReviewEvent(query=query)

    @step(pass_context=True)
    async def math_agent(self, ctx: Context, ev: MathEvent) -> Union[ReviewEvent, StopEvent]:
        attempt = ctx.data.get("attempt")
        response = ctx.data.get("response")
        feedback = ctx.data.get("feedback")
        query = ev.query
        # print("In Physics Agent ")
        # print("attempt: ", attempt)
        # print("response: ",  response)
        # print("response: ", feedback)
        # print("query: ", query)
        # print("--------------------------------")
        prompt = f"""You are an experienced maths teacher.
        Please provide an answer to the user query \n query:{query}. Please ensure to incorporate any 
        feedback provided for your response \n
        feedback:{feedback}"""

        if attempt > 1:
            return StopEvent(result="final response :\n" + str(response) + "\n feedback:" + str(feedback))
        else:
            attempt = attempt + 1
            ctx.data["attempt"] = attempt
            response = await self.llm.acomplete(prompt)
            print("attempt #: ", attempt, "\nfeedback :", feedback, "\nresponse :", response)
            ctx.data["response"] = response
            return ReviewEvent(query=query)

    @step(pass_context=True)
    async def review_agent(self, ctx: Context, ev: ReviewEvent) -> Union[PhysicsEvent, MathEvent]:
        category = ctx.data.get("category")
        query = ev.query
        response = ctx.data.get("response")
        # print("In review Agent ")
        # print("category: ", category)
        # print("response: ",  response)
        # print("query: ", query)

        prompt = f"""You are reviewer of a response provided against a query.
        Please provide your feedback to improve the response.
        Below is the query and response for your review \n
        query:{query} \n
        response:{response} \n
        feedback:
        """
        feedback = await self.llm.acomplete(prompt)
        # print("feedback: ", feedback)
        # print("--------------------------------")
        ctx.data["feedback"] = feedback

        if category == "math":
            return MathEvent(query=query, response=response)
        else:
            return PhysicsEvent(query=query, response=response)


async def main():
    w = TeacherCrew(timeout=60, verbose=False)
    draw_all_possible_flows(TeacherCrew, filename="teachercrew.html")
    result = await w.run(query="what is reflection?", category="physics")
    print(str(result))
    draw_most_recent_execution(w, filename="teachercrewrun.html")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
