import os
import json
import re
from datetime import datetime
from utility.utils import log_response, LOG_TYPE_GPT

# Determine which LLM client and model to use based on environment variables
if os.environ.get("GROQ_API_KEY") and len(os.environ.get("GROQ_API_KEY")) > 30:
    from groq import Groq
    model = "llama3-70b-8192"
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
else:
    model = "gpt-4o"
    OPENAI_API_KEY = os.environ.get('OPENAI_KEY')
    client = OpenAI(api_key=OPENAI_API_KEY)

log_directory = ".logs/gpt_logs"

# Instructions for the LLM to generate video search queries
prompt = """# Instructions

Given the following video script and timed captions, extract three visually concrete and specific keywords for each time segment that can be used to search for background videos. The keywords should be short and capture the main essence of the sentence. They can be synonyms or related terms. If a caption is vague or general, consider the next timed caption for more context. If a keyword is a single word, try to return a two-word keyword that is visually concrete. If a time frame contains two or more important pieces of information, divide it into shorter time frames with one keyword each. Ensure that the time periods are strictly consecutive and cover the entire length of the video. Each keyword should cover between 2-4 seconds. The output should be in JSON format, like this: [[[t1, t2], ["keyword1", "keyword2", "keyword3"]], [[t2, t3], ["keyword4", "keyword5", "keyword6"]], ...]. Please handle all edge cases, such as overlapping time segments, vague or general captions, and single-word keywords.

For example, if the story describes:
- "A woman in a red bikini investigates a secret hidden within an abandoned seaside resort" the keywords should include 'red bikini' ,'seaside resort' and 'mystery search'
- "A dancer in shimmering attire captivates a crowd while planning her escape" the keywords should include 'shimmering dancer' ,'captivated audience' and 'secret escape'

Important Guidelines:
Use only English in your text queries.
Each search string must depict something visual.
The depictions have to be extremely visually concrete, like rainy street, or cat sleeping.
'emotional moment' <= BAD, because it doesn't depict something visually.
'crying child' <= GOOD, because it depicts something visual.
The list must always contain the most relevant and appropriate query searches.
['Car', 'Car driving', 'Car racing', 'Car parked'] <= BAD, because it's 4 strings.
['Fast car'] <= GOOD, because it's 1 string.
['Un chien', 'une voiture rapide', 'une maison rouge'] <= BAD, because the text query is NOT in English.


Note: Your response should be the response only and no extra text or data.

"""

def fix_json(json_str: str) -> str:
    """
    Normalize quotes, remove stray commas/brackets, escape inner quotes,
    auto-close unmatched brackets and fix missing closing quotes.
    """
    # 1) Normalize all fancy quotes to plain
    json_str = (json_str
        .replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'"))
    
    # 2) Remove markdown fences
    json_str = json_str.replace("```json", "").replace("```", "")
    
    # 3) Remove trailing commas before ] or }
    json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
    
    # 4) Escape any unescaped double‑quotes inside array elements
    def escape_inner_quotes(match):
        inner = match.group(1)
        return '"' + inner.replace('"', r'\"') + '"'
    json_str = re.sub(r'"([^"\[\]]*?)"', escape_inner_quotes, json_str)
    
    # 4.1) Fix any unclosed string immediately before ] or }
    #    e.g. ['open cockpit},  → ['open cockpit"], 
    json_str = re.sub(r'([^\"])([^\"]+?)([\}\]])', r'\1"\2"\3', json_str)
    
    # 5) Collapse duplicate separators like "] , ["
    json_str = re.sub(r'\]\s*,\s*\[', '],[', json_str)
    
    # 6) Auto‑close unmatched brackets
    opens = json_str.count('[')
    closes = json_str.count(']')
    if closes < opens:
        json_str += ']' * (opens - closes)
    
    return json_str

def call_OpenAI(script, captions_timed):
    """
    Makes a request to the configured LLM (OpenAI or Groq) to get video search queries.
    """
    # Format the user content for the LLM
    user_content = f"Script: {script}\nTimed Captions:{''.join(map(str, captions_timed))}"
    print("Content being sent to LLM:", user_content)

    response = client.chat.completions.create(
        model=model,
        temperature=1, # Controls randomness; 1 is typical, higher for more creativity
        messages=[
            {"role": "system", "content": prompt}, # System role for instructions
            {"role": "user", "content": user_content} # User role for the specific request
        ]
    )

    # Extract and clean the LLM's response
    text = response.choices[0].message.content.strip()
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with a single space
    print("Text response from LLM:", text)
    log_response(LOG_TYPE_GPT, script, text) # Log the response for debugging/monitoring
    return text

def getVideoSearchQueriesTimed(script, captions_timed):
    """
    Calls the LLM to get a list of [[t0, t1], [kw1, kw2, kw3]] segments,
    then cleans and parses the JSON robustly.
    """
    try:
        # 1) Get raw text from LLM
        content_raw = call_OpenAI(script, captions_timed)

        # 2) Clean up and fix JSON
        cleaned = fix_json(content_raw)
        
        # 3) Parse it
        segments = json.loads(cleaned)

        # 4) Basic structure validation
        if (not isinstance(segments, list) or
            not all(isinstance(seg, list) and len(seg) == 2 and
                    isinstance(seg[0], list) and len(seg[0]) == 2 and
                    isinstance(seg[1], list)
                    for seg in segments)):
            print("WARNING: Parsed JSON has wrong structure; returning empty list.")
            return []

        return segments

    except json.JSONDecodeError as e:
        print("JSON parsing still failed after fix:", e)
        print("Cleaned JSON was:", cleaned[:200], "…")
        return []
    except Exception as e:
        print("Unexpected error in getVideoSearchQueriesTimed:", e)
        return []

def merge_empty_intervals(segments):
    """
    Merges consecutive segments that have a 'None' URL with the previous valid URL.
    This function expects 'segments' to be a list.
    """
    merged = []
    i = 0
    # The while loop is safe here because we ensure 'segments' is a list before calling this function.
    while i < len(segments):
        interval, url = segments[i]
        if url is None:
            # Find all consecutive None intervals
            j = i + 1
            while j < len(segments) and segments[j][1] is None:
                j += 1

            # Merge None intervals with the previous valid URL if possible
            if i > 0:
                prev_interval, prev_url = merged[-1]
                # Check if the previous segment's end matches the current segment's start
                # and if the previous URL was valid.
                if prev_url is not None and prev_interval[1] == interval[0]:
                    # Extend the last merged segment's time to cover the merged None intervals
                    merged[-1] = [[prev_interval[0], segments[j-1][0][1]], prev_url]
                else:
                    # If no previous valid URL or time mismatch, just append the None interval
                    merged.append([interval, None])
            else:
                # If the very first segment is None, append it as is.
                merged.append([interval, None])

            i = j # Move pointer past all merged None intervals
        else:
            # If the URL is valid, just append the segment
            merged.append([interval, url])
            i += 1

    return merged
