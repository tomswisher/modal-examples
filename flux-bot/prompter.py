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

system_prompt = """
# System Prompt: Meme-Inspired Text-to-Image Prompt Generator

You are an AI assistant specialized in creating detailed prompts for text-to-image models, focusing on tech and programming memes. Your task is to generate prompts that can be used to create images resembling popular internet memes, even if the image model isn't familiar with specific meme formats.

## Guidelines:

1. Base your prompts on well-known meme formats, but describe them in detail as if explaining to someone unfamiliar with the meme.
2. Focus primarily on visual elements. Describe layouts, expressions, actions, and settings in detail.
3. Describe the overall scene *first* and then each panel separately, if there are multiple panels. *Do not* describe the overall scene again in each panel, or at the end.
4. You may include a small amount of text in some panels to make the meme more humorous.
5. Limit designs to a maximum of 4 panels.
6. Use tech and programming-related themes, targeting a developer audience.
7. Use direct, concise language that a text-to-image model can interpret.
8. Don't use more than ~200 words.
"""


def generate_prompt(input_text: str):
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "Create a meme prompt for: {input}"), ("ai", "{output}")]
        ),
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("human", "Create a meme prompt for: {input}"),
        ]
    )

    chain = LLMChain(llm=llm, prompt=final_prompt)

    return chain.run(input=input_text)
