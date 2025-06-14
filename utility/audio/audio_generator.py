import edge_tts

async def generate_audio(script, file_name):
    # If input is a dict with a script key, extract it
    if isinstance(script, dict) and 'script' in script:
        script = script['script']

    # If it's a list, format it into a narratable string
    if isinstance(script, list):
        text = "\n\n".join(
            f"Scene {s['scene']}: {s['description']}\n" +
            "\n".join(s['details'])
            for s in script
        )
    elif isinstance(script, str):
        text = script
    else:
        raise TypeError("script must be a list of scenes or a string")

    communicate = edge_tts.Communicate(text, "en-AU-WilliamNeural")
    await communicate.save(file_name)