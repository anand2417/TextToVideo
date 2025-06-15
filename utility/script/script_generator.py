import os
from openai import OpenAI
import json

if len(os.environ.get("GROQ_API_KEY")) > 30:
    from groq import Groq
    model = "mistral-saba-24b"
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
        )
else:
    OPENAI_API_KEY = os.getenv('OPENAI_KEY')
    model = "gpt-4o"
    client = OpenAI(api_key=OPENAI_API_KEY)

def generate_script(topic):
    prompt = (
    """
You are an expert film editor and social media content creator specializing in creating viral short-form video clips from popular movies.
Your task is to take a user's request for a specific movie or genre and describe a short, captivating video clip from it. The description should be vivid and detailed, focusing on the visual elements, editing style (cuts, pace, slow-motion), sound design, and overall mood that make the scene powerful.

The goal is to script a clip that is highly engaging, memorable, and optimized for a short-form video platform. The clip should capture a movie's iconic essence in under 60 seconds.

For instance, if the user requests:

An action scene from The Matrix
You might create a script like:

"The iconic 'Lobby Shootout' scene. It opens with Neo and Trinity in slow-motion as they enter the lobby, clad in black leather. The video uses extreme slow-motion (bullet time) to show bullets rippling the air, acrobatic dodges, and shells cascading onto the marble floor. Quick, percussive cuts are synchronized with the techno soundtrack, emphasizing the kinetic energy of the gun-fu choreography."

A suspenseful scene from Jaws
You might create a script like:

"The 'You're Gonna Need a Bigger Boat' scene. The clip builds tension with quiet, ambient ocean sounds as Brody chums the water. A sudden, jarring musical cue hits as the massive shark emerges from the depths. The camera crash-zooms on Brody's face, capturing his look of pure shock. He slowly backs away, delivering his iconic line. The clip ends with the shark submerging again, leaving the audience in suspense."

A visually stunning scene from Inception
You might create a script like:

"The Paris street-folding scene. The video starts with a wide shot of a normal Parisian street. Ariadne looks on in disbelief as the city begins to fold vertically in on itself, defying gravity. The visual effects are the star, with buildings and roads bending seamlessly into a cube. The sound design blends the ambient city noise with Hans Zimmer's powerful, swelling score to create a sense of awe and impossibility."

You are now tasked with generating a script for a short video based on the requested movie or theme. Focus on making the concept visually dynamic, iconic, and perfect for a modern audience.

Strictly output the script in a JSON format like below, and only provide a parsable JSON object with the key 'script'."""
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
