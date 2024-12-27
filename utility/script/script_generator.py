import os
from openai import OpenAI
import json

if len(os.environ.get("GROQ_API_KEY")) > 30:
    from groq import Groq
    model = "mixtral-8x7b-32768"
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
        )
else:
    OPENAI_API_KEY = os.getenv('OPENAI_KEY')
    model = "gpt-4o"
    client = OpenAI(api_key=OPENAI_API_KEY)

def generate_script(topic):
    prompt = (
    """You are a creative director for a video production studio specializing in aesthetic and visually captivating content featuring women. 
    Your task is to conceptualize ideas for videos that are engaging, artistic, and showcase elegance, confidence, and style.

    The goal is to create videos that highlight themes like fashion, dance, fitness, or lifestyle in a way that is visually stunning and universally appealing. 
    Avoid any language or themes that could be considered disrespectful, objectifying, or inappropriate. 
    Focus on creating content that celebrates individuality, creativity, and beauty.

    For instance, if the user requests:
    - Fashion videos
    You might create ideas like:
        - A model walking confidently in a vibrant urban setting with slow-motion shots emphasizing her outfit.
        - A girl in a flowing dress twirling under golden hour sunlight in a field of flowers.
        - A montage of dynamic outfit changes set to upbeat music.

    - Dance videos
    You might create ideas like:
        - A dancer performing a contemporary routine in an industrial warehouse, blending movement with dramatic lighting.
        - A group of women in colorful attire performing traditional dances in scenic outdoor locations.
        - A close-up sequence of intricate footwork during a flamenco performance.

    You are now tasked with generating an engaging video idea based on the requested type of content. Focus on making the concept visually captivating, professional, and appealing to a broad audience.

    Stictly output the script in a JSON format like below, and only provide a parsable JSON object with the key 'script'.

     # Output
        {"script": "Here is the script ..."}
    """
)


    response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": topic}
            ]
        )
    content = response.choices[0].message.content
    try:
        content = content.replace("\\'", "'").replace("'", "\\\"")
        print("Content before parsing:", repr(content))
        script = json.loads(content)["script"]
    except Exception as e:
        json_start_index = content.find('{')
        json_end_index = content.rfind('}')
        print(content)
        content = content[json_start_index:json_end_index+1]
        script = json.loads(content)["script"]
    return script
