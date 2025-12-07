import openai
from openai import OpenAI
import os
import json

# Configure OpenAI API key
openai.api_key = None
os.environ["OPENAI_API_KEY"] = openai.api_key
client = OpenAI()

def get_breed_names():
    with open("dog_breed_names.txt", "r") as f:
        dog_breed_names = json.load(f)
    return dog_breed_names

def llm_generate_global_concepts(all_breed_names, file_path):
    breed_list_str = ", ".join(all_breed_names)

    prompt = f"""
            You are an expert in dog morphology.

            Given the following dog breeds:
            {breed_list_str}

            Your task is to define a list of universal, human-interpretable, discrete visual attributes
            that can distinguish any dog breed.

            List 30 human-interpretable, discrete visual attributes that distinguish dog breeds.
            Each concept must be a yes/no style feature, e.g.:
            - long coat
            - short muzzle
            - erect ears
            - curly tail
            - wrinkled face
            - feathered tail
            Avoid continuous attributes like "size" or "leg_length".
            Ensure each attribute can be labeled universally across all breeds.

            Return JSON like:
            {{"concepts": ["attr1", "attr2", ...]}}
            """

    client = openai.OpenAI()
    resp = client.responses.create(model="gpt-5-nano", input=prompt)
    data = json.loads(resp.output_text)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data["concepts"]:
            f.write(item + "\n")
    return data["concepts"]

def get_concept_labels_for_breed(breed_name, concepts, max_retries=2):
    # ensure output dirs exist
    os.makedirs("outputs/per_breed", exist_ok=True)
    concept_list_str = json.dumps(concepts, ensure_ascii=False)

    prompt = f"""
        You are an expert in dog morphology.

        You are given:

        1. A dog breed name:
        {breed_name}

        2. A FIXED list of universal binary visual concepts (JSON array):
        {concept_list_str}

        TASK:
        Assign each concept a binary value for this breed:
        - 1 = the concept typically applies to this breed
        - 0 = does not apply

        STRICT RULES:
        - You MUST output EXACTLY the same set of concept names as provided.
        - No new concepts.
        - No missing concepts.
        - No renaming.
        - No reordering.
        - Values must be 0 or 1 only.
        - Output MUST be valid JSON only.

        OUTPUT FORMAT:

        {{
        "breed": "{breed_name}",
        "concept_labels": {{
            "<concept from list>": 0 or 1,
            "<concept from list>": 0 or 1
        }}
        }}
        """

    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model="gpt-5-nano",
                input=prompt
            )

            obj = json.loads(resp.output_text)

            labels = obj["concept_labels"]

            # strict key check
            if set(labels.keys()) != set(concepts):
                raise ValueError("Concept keys mismatch")

            # strict value check
            for k, v in labels.items():
                if v not in [0, 1]:
                    raise ValueError("Non-binary value detected")

            # --- save per-breed result ---
            with open(f"outputs/per_breed/{breed_name}.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"breed": breed_name, "concept_labels": labels},
                    f, indent=2, ensure_ascii=False
                )

            return labels

        except Exception as e:
            print(f"[Warning] Breed {breed_name} attempt {attempt+1} failed: {e}")

    raise RuntimeError(f"Failed to get valid concept labels for {breed_name} after {max_retries} tries")

def generate_concept_matrix(breed_names, concepts):
    matrix = {}

    for i, breed in enumerate(breed_names):
      matrix[breed] = get_concept_labels_for_breed(breed, concepts)

    # save full concept matrix
    with open("concept_matrix.json", "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)

    return matrix

def generate_concept_per_breed(breed_names, file_path, num_concepts_per_breed=5):
    breed_concepts = {}

    client = openai.OpenAI()
    for breed in breed_names:
        print(breed)
        prompt = f"""
        You are an expert in dog morphology.

        Given the dog breed: {breed}

        Generate {num_concepts_per_breed} human-interpretable, discrete visual attributes (concepts)
        that distinguish this breed from others. Each attribute must be a yes/no style feature.
        
        Return JSON like:
        {{"concepts": ["attr1", "attr2", ...]}}
        """
        resp = client.responses.create(model="gpt-5-nano", input=prompt)
        try:
            data = json.loads(resp.output_text)
            breed_concepts[breed] = data["concepts"]
        except Exception as e:
            print(f"Failed to parse LLM output for breed {breed}: {e}")
            breed_concepts[breed] = []

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(breed_concepts, f, indent=2, ensure_ascii=False)
    
    return breed_concepts

dog_breed_names = get_breed_names()
# print(len(dog_breed_names))

llm_concepts = llm_generate_global_concepts(dog_breed_names, 'llm_concepts.txt')
concept_matrix = generate_concept_matrix(dog_breed_names, llm_concepts)
generate_concept_per_breed(dog_breed_names, file_path='label_free_concepts.txt')