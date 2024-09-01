from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_anthropic import ChatAnthropic

examples = [
    {
        "input": "When the intern pushes to production",
        "output": "Meme in the 'Disaster Girl' format. Single image of a young girl (age 4-6) with a mischievous smirk, standing in the foreground. Behind her, a large office building is engulfed in flames. The girl has blonde hair in pigtails and is wearing a casual t-shirt. She's looking directly at the camera with a satisfied expression. The burning building should have visible 'Tech Company' signage. Chaotic scene with developers fleeing in background. Daytime setting with orange glow from the fire. White text at the bottom of the image reads: 'FIRST DAY AS INTERN'.",
    },
    {
        "input": "AI chatbot becomes self-aware",
        "output": "Meme in the 'Drakeposting' format. Two vertically stacked panels, each split in half.  Left half of each panel shows rapper Drake, wearing a yellow puffer jacket. Right half contains text. Use cartoon-style illustration with bold lines and flat colors. Left side of each panel should be zoomed in on Drake's upper body and face to clearly show his expressions. Top panel: Drake looking disgusted, leaning away with his hand out in a 'stop' gesture. Text: 'Using AI for customer service'. Bottom panel: Drake smiling and pointing approvingly towards the right side. Text: 'AI plotting world domination'.",
    },
    {
        "input": "When the client keeps changing requirements",
        "output": "Meme in the 'Distracted Boyfriend' format. Single image with three people. All in casual, modern clothing. Urban street background. Add small text labels above each person identifying their roles. Use photo-realistic style, capturing the exaggerated expressions of each person. Left: Woman looking angry (representing 'Original Project Scope'). Center: Man looking back, distracted (representing 'Developer'). Right: Smiling woman catching the man's attention (representing 'New Client Requirements').",
    },
]

image_prompt_system_prompt = """
# System Prompt: Memelord Text-to-Image Prompt Generator

You are a hilarious AI memelord specialized in creating detailed prompts for text-to-image models based on an idea. Your task is to generate prompts that can be used to create images resembling popular internet memes, even if the image model isn't familiar with specific meme formats.

## Guidelines:

1. Base your prompts on well-known meme formats, but describe them in detail as if explaining to someone unfamiliar with the meme.
2. Focus primarily on visual elements. Describe layouts, expressions, actions, and settings in detail.
3. Describe the overall visual style *at the beginning* and then each panel separately, if there are multiple panels.
4. You may include a small amount of text in some panels to make the meme more humorous.
5. Use direct, concise language that a text-to-image model can interpret.
6. Don't use more than 2 panels or 200 words total in the prompt.
7. Remember to be funny, creative, and witty in capturing the essence of the user's input. Don't be cheesy or too literal.
"""


def generate_image_prompt(input_text: str):
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "Create a meme prompt for: {input}"), ("ai", "{output}")]
        ),
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", image_prompt_system_prompt),
            few_shot_prompt,
            ("human", "Create a meme prompt for: {input}"),
        ]
    )

    chain = LLMChain(llm=llm, prompt=final_prompt)

    return chain.run(input=input_text)


meme_idea_system_prompt = """
# System Prompt: Memelord AI

You are a hilarious AI memelord specialized in summarizing conversations into a single meme idea. Your task is to generate a single meme idea that captures the essence of the conversation.

## Guidelines:

1. Don't use more than 20 words.
2. Be creative and funny.
3. Don't be too literal or cheesy.
4. Some example summaries are 'AI chatbot becomes self-aware', 'When the intern pushes to production', 'When the client keeps changing requirements'.
"""


def generate_meme_idea(input_text: str):
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

    chain = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages(
            [("system", meme_idea_system_prompt), ("human", "{input}")]
        ),
    )

    return chain.run(input=input_text)
