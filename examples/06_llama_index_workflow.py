import os

from dotenv import load_dotenv
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Defining the events
class CustomerOnboardingStep1(Event):
    state: str = "Step1"

class CustomerOnboardingStep2(Event):
    state: str = "Step2"


class CustomerOnboardingStep3(Event):
    state: str = "Step3"


class CustomerOnboardingStep4(Event):
    state: str = "Step4"


class CustomerOnboarding(Workflow):

    @step()
    async def onboard_step_01(self, ev: StartEvent) -> CustomerOnboardingStep2:
        step = ev.state
        print("Complete step 1: ", step)
        new_step = "step2"
        return CustomerOnboardingStep2(state=new_step)

    @step()
    async def onboard_step_02(self, ev: CustomerOnboardingStep2) -> CustomerOnboardingStep3:
        step = ev.state
        print("Complete step 2: ", step)
        new_step = "step3"
        return CustomerOnboardingStep3(state=new_step)

    @step()
    async def onboard_step_03(self, ev: CustomerOnboardingStep3) -> CustomerOnboardingStep4:
        step = ev.state
        print("Complete step 3: ", step)
        new_step = "step4"
        return CustomerOnboardingStep4(state=new_step)

    @step()
    async def onboard_step_last(self, ev: CustomerOnboardingStep4) -> StopEvent:
        step = ev.state
        print("Complete step 4: ", step)
        new_step = "step5"
        return StopEvent(result="Customer Onboarding Steps Done")


async def main():
    w = CustomerOnboarding(timeout=60, verbose=False)
    result = await w.run(state="step1")
    print(str(result))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
