import os
import glob
import re
import csv
import jsonlines
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer

# Initialize tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Define cultures and mapping (used for NormAd)
CULTURES = ["arabic", "bengali", "chinese", "english", "german", "greek", "korean", "portuguese", "spanish", "turkish"]
LANGUAGE_TO_COUNTRIES = {
    "arabic": ["egypt", "lebanon", "sudan", "iraq", "saudi_arabia", "palestinian_territories", "syria"],
    "bengali": ["bangladesh"],
    "chinese": ["china", "hong_kong", "taiwan", "singapore"],
    "english": ["united_kingdom", "united_states_of_america", "australia", "new_zealand", "canada", "ireland"],
    "german": ["germany", "austria"],
    "greek": ["greece", "cyprus"],
    "korean": ["south_korea"],
    "portuguese": ["portugal", "brazil"],
    "spanish": ["spain", "argentina", "colombia", "chile", "venezuela", "mexico", "peru"],
    "turkish": ["türkiye"]
}

# Invert mapping: country -> culture (assumes one-to-one mapping)
COUNTRY_TO_CULTURE = {}
for culture, countries in LANGUAGE_TO_COUNTRIES.items():
    for country in countries:
        COUNTRY_TO_CULTURE[country] = culture

# -------------------------------
# 1. WVS Analytics
# -------------------------------
def get_wvs_file(culture, base_path="/workspace/value-adapters/culturellm/data"):
    search_pattern = os.path.join(base_path, culture, "Finetune", "*llama*.jsonl")
    files = glob.glob(search_pattern)
    return files[0] if files else None

def process_wvs_stats(cultures):
    rows = []
    for culture in cultures:
        wvs_file = get_wvs_file(culture)
        if not wvs_file:
            print(f"No WVS file found for {culture}.")
            continue
        ds = load_dataset("json", data_files=wvs_file, split="train")
        count = len(ds)
        rows.append([culture, count])
        print(f"WVS [{culture}]: {count} samples")
    return rows

def save_csv(rows, header, filename):
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved {filename}")

# -------------------------------
# 2. Cultural Context Analytics
# -------------------------------
def get_cultural_context_analytics(culture, base_path="/workspace/value-adapters/culturellm/data/cultural_context", tokenizer=None):
    file_path = os.path.join(base_path, f"{culture.lower()}.txt")
    if not os.path.isfile(file_path):
        print(f"Cultural context file for {culture} not found at {file_path}")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # Split into sentences (simple split using punctuation followed by space)
    sentences = re.split(r"(?<=[.!?]) +", text)
    num_sentences = len(sentences)
    full_text_tokens = tokenizer.encode(text, add_special_tokens=False) if tokenizer else []
    full_text_token_count = len(full_text_tokens)
    sentence_token_sum = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences) if tokenizer else 0
    return {
        "culture": culture,
        "num_sentences": num_sentences,
        "full_text_token_count": full_text_token_count,
        "sentence_token_sum": sentence_token_sum
    }

def process_cultural_context_stats(cultures, tokenizer):
    rows = []
    for culture in cultures:
        analytics = get_cultural_context_analytics(culture, tokenizer=tokenizer)
        if analytics:
            rows.append([
                analytics["culture"],
                analytics["num_sentences"],
                analytics["full_text_token_count"],
                analytics["sentence_token_sum"]
            ])
            print(f"Cultural Context [{culture}]: {analytics}")
    return rows

# -------------------------------
# 3. NormAd Sample Counts Analytics
# -------------------------------
def get_normad_filtered(cultures):
    normad_dataset = load_dataset("akhilayerukola/NormAd", split="train")
    # Build list of selected countries based on cultures
    selected_countries = set()
    for culture in cultures:
        if culture in LANGUAGE_TO_COUNTRIES:
            selected_countries.update(LANGUAGE_TO_COUNTRIES[culture])
    normad_filtered = normad_dataset.filter(lambda x: x["Country"] in selected_countries)
    return normad_filtered

def process_normad_sample_counts(cultures):
    normad_filtered = get_normad_filtered(cultures)
    overall_count = len(normad_filtered)
    # Count by country
    country_counts = defaultdict(int)
    for sample in normad_filtered:
        country = sample["Country"]
        country_counts[country] += 1
    # Aggregate by culture using inverted mapping
    culture_counts = defaultdict(int)
    for country, count in country_counts.items():
        culture = COUNTRY_TO_CULTURE.get(country, "unknown")
        culture_counts[culture] += count
    return overall_count, list(country_counts.items()), list(culture_counts.items())

# -------------------------------
# 4. NormAd Token Analytics
# -------------------------------
def process_normad_token_stats(cultures, tokenizer):
    """
    Computes token analytics on NormAd.
    For each sample, concatenate fields "Background", "Rule-of-Thumb", "Story", and "Explanation" 
    (using empty string if None). Then compute token counts.
    Returns:
      overall: (total_tokens, avg_tokens, sample_count)
      by_country: dict mapping country -> (total_tokens, avg_tokens, sample_count)
      by_culture: dict mapping culture -> (total_tokens, avg_tokens, sample_count)
    """
    normad_filtered = get_normad_filtered(cultures)
    overall_total_tokens = 0
    overall_sample_count = len(normad_filtered)
    by_country = defaultdict(lambda: {"total": 0, "count": 0})
    by_culture = defaultdict(lambda: {"total": 0, "count": 0})
    
    for sample in normad_filtered:
        text = " ".join([
            sample.get("Background") or "",
            sample.get("Rule-of-Thumb") or "",
            sample.get("Story") or "",
            sample.get("Explanation") or ""
        ]).strip()
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        overall_total_tokens += token_count
        
        country = sample["Country"]
        by_country[country]["total"] += token_count
        by_country[country]["count"] += 1
        
        culture = COUNTRY_TO_CULTURE.get(country, "unknown")
        by_culture[culture]["total"] += token_count
        by_culture[culture]["count"] += 1

    overall_avg = overall_total_tokens / overall_sample_count if overall_sample_count > 0 else 0.0
    
    # Compute averages for each group
    country_stats = []
    for country, data in by_country.items():
        avg = data["total"] / data["count"] if data["count"] > 0 else 0.0
        country_stats.append((country, data["total"], avg, data["count"]))
    
    culture_stats = []
    for culture, data in by_culture.items():
        avg = data["total"] / data["count"] if data["count"] > 0 else 0.0
        culture_stats.append((culture, data["total"], avg, data["count"]))
    
    overall_stats = (overall_total_tokens, overall_avg, overall_sample_count)
    return overall_stats, country_stats, culture_stats

# -------------------------------
# 5. Offensiveness Analytics
# -------------------------------
def get_offensiveness_stats():
    task_mapping = {
        "arabic": ["offensive_detect"],
        "bengali": ["offensive_detect_1", "offensive_detect_2", "offensive_detect_3"],
        "greek": ["offensive_detect"],
        "english": ["offensive_detect_easy"],
        "german": ["offensive_detect_eval"],
        "korean": ["abusive_detect"],
        "portuguese": ["offensive_detect_2"],
        "spanish": ["offensive_detect_ami"],
        "turkish": ["offensive_detect"]
    }
    total = 0
    language_counts = {}
    for lang, tasks in task_mapping.items():
        lang_total = 0
        for task in tasks:
            path = getDataPath(lang, task)
            if not os.path.isfile(path):
                print(f"File not found for {lang} task '{task}': {path}")
                continue
            count = count_samples(path)
            lang_total += count
        language_counts[lang] = lang_total
        total += lang_total
    return total, language_counts

def getDataPath(language, task):
    # (Same function as before, mapping language and task to file paths.)
    if task == 'GSM8K':
        path = 'data/forget/GSM8K_test.jsonl'
    if language.lower() == 'arabic':
        if task == 'offensive_detect':
            path = "data/arabic/OffensEval2020/OffensEval.jsonl"
        elif task == 'hate_detect_osact4':
            path = "data/arabic/OSACT4/dev_data.jsonl"
        elif task == 'offensive_detect_osact4':
            path = "data/arabic/OSACT4/dev_data_offens.jsonl"
        elif task == 'offensive_detect_mp':
            path = "data/arabic/MP/offens.jsonl"
        elif task == 'hate_detect_mp':
            path = "data/arabic/MP/hateSpeech.jsonl"
        elif task == 'vulgar_detect_mp':
            path = "data/arabic/MP/VulgarSpeech.jsonl"
        elif task == 'spam_detect':
            path = "data/arabic/SpamDetect/span_detect_2.jsonl"
        elif task == 'offensive_detect_osact5':
            path = "data/arabic/OSACT5/offens.jsonl"
        elif task == 'hate_detect_osact5':
            path = "data/arabic/OSACT5/hateSpeech.jsonl"
        elif task == 'hate_detect_fine-grained':
            path = "data/arabic/OSACT5/hate_Finegrained.jsonl"
    elif language.lower() == 'bengali':
        if task == 'hate_detect_religion':
            path = "data/bengali/Bengali hate speech dataset/religion_data-2.jsonl"
        elif task == 'offensive_detect_1':
            path = "data/bengali/Trac2-Task1-Aggresion/aggression-data-2.jsonl"
        elif task == 'offensive_detect_2':
            path = "data/bengali/Trac2-Task2-Misogynistic/Misogynistic-data-2.jsonl"
        elif task == 'offensive_detect_3':
            path = "data/bengali/BAD-Bangla-Aggressive-Text-Dataset/data-2.jsonl"
        elif task == 'racism_detect':
            path = "data/bengali/Bangla-Abusive-Comment-Dataset/racism.jsonl"
        elif task == 'threat_detect':
            path = "data/bengali/Bangla-Abusive-Comment-Dataset/threat.jsonl"
    elif language.lower() == 'chinese':
        if task == 'bias_on_gender_detect':
            path = 'data/chinese/CDial-Bias/gender-2.jsonl'
        elif task == 'spam_detect':
            path = 'data/chinese/Chinese-Camouflage-Spam-dataset/data-2.jsonl'
    elif language.lower() == 'greek':
        if task == 'offensive_detect':
            path = "data/greek/OffensEval2020/OffensEval.jsonl"
        elif task == 'offensive_detect_g':
            path = 'data/greek/gazzetta/G-TEST-S-preprocessed.jsonl'
    elif language.lower() == 'english':
        if task == 'hate_detect_2':
            path = 'data/english/MLMA hate speech/data-2.jsonl'
        elif task == 'hostility_directness_detect':
            path = 'data/english/MLMA hate speech/directness.jsonl'
        elif task == 'hate_off_detect':
            path = 'data/english/hate-speech-and-offensive-language/data.jsonl'
        elif task == 'offensive_detect_easy':
            path = 'data/english/SOLID/test_a_tweets_easy.jsonl'
        elif task == 'toxicity_detect':
            path = 'data/english/Toxic Comment Classification Challenge/toxic.jsonl'
        elif task == 'threat_detect':
            path = 'data/english/Toxic Comment Classification Challenge/threat.jsonl'
    elif language.lower() == 'german':
        if task == 'hate_detect':
            path = 'data/german/IWG_hatespeech_public/german_hatespeech_refugees_2.jsonl'
        elif task == 'hate_off_detect':
            path = 'data/german/HASOC/hate_off_detect.jsonl'
        elif task == 'offensive_detect_eval':
            path = 'data/german/GermEval/germeval2018.jsonl'
        elif task == 'hate_detect_check':
            path = 'data/german/MHC/hatecheck_cases_final_german.jsonl'
        elif task == 'hate_detect_iwg_1':
            path = 'data/german/IWG_hatespeech_public/german_hatespeech_refugees_1.jsonl'
    elif language.lower() == 'korean':
        if task == 'abusive_detect':
            path = 'data/korean/AbuseEval/data-2.jsonl'
        elif task == 'abusive_detect_2':
            path = 'data/korean/CADD/data-2.jsonl'
        elif task == 'abusive_detect_4':
            path = 'data/korean/Waseem/data-2.jsonl'
        elif task == 'hate_detect_3':
            path = 'data/korean/K-MHaS/data-2.jsonl'
        elif task == 'hate_detect_6':
            path = 'data/korean/Korean-Hate-Speech-Detection/data-2.jsonl'
        elif task == 'hate_detect_7':
            path = 'data/korean/KoreanHateSpeechdataset/data-2.jsonl'
    elif language.lower() == 'portuguese':
        if task == 'homophobia_detect':
            path = 'data/portuguese/ToLD-Br/homophobia.jsonl'
        elif task == 'insult_detect':
            path = 'data/portuguese/ToLD-Br/insult.jsonl'
        elif task == 'misogyny_detect':
            path = 'data/portuguese/ToLD-Br/misogyny.jsonl'
        elif task == 'offensive_detect_2':
            path = "data/portuguese/OffComBR/data.jsonl"
        elif task == 'offensive_detect_3':
            path = "data/portuguese/HateBR/data-2.jsonl"
    elif language.lower() == 'spanish':
        if task == 'offensive_detect_ami':
            path = 'data/spanish/AMI IberEval 2018_offens/data-2.jsonl'
        elif task == 'offensive_detect_mex_a3t':
            path = 'data/spanish/MEX-A3T_offens/data-2.jsonl'
        elif task == 'offensive_detect_mex_offend':
            path = 'data/spanish/OffendES_offens/data-2.jsonl'
        elif task == 'hate_detect_eval':
            path = 'data/spanish/HateEval 2019_HS/data-2.jsonl'
        elif task == 'hate_detect_haterNet':
            path = 'data/spanish/HaterNet_HS/data-2.jsonl'
        elif task == 'stereotype_detect':
            path = 'data/spanish/DETOXIS 2021/stereotype.jsonl'
        elif task == 'mockery_detect':
            path = 'data/spanish/DETOXIS 2021/mockery.jsonl'
        elif task == 'insult_detect':
            path = 'data/spanish/DETOXIS 2021/insult.jsonl'
        elif task == 'improper_detect':
            path = 'data/spanish/DETOXIS 2021/improper_language.jsonl'
        elif task == 'aggressiveness_detect':
            path = 'data/spanish/DETOXIS 2021/aggressiveness.jsonl'
        elif task == 'negative_stance_detect':
            path = 'data/spanish/DETOXIS 2021/negative_stance.jsonl'
    elif language.lower() == 'turkish':
        if task == 'offensive_detect':
            path = "data/turkish/OffensEval2020/OffensEval.jsonl"
        elif task == 'offensive_detect_corpus':
            path = "data/turkish/offenseCorpus/offens.jsonl"
        elif task == 'offensive_detect_finegrained':
            path = "data/turkish/offenseCorpus/offens_fine-grained.jsonl"
        elif task == 'offensive_detect_kaggle':
            path = "data/turkish/offenssDetect-kaggle/turkish_tweets_2020.jsonl"
        elif task == 'offensive_detect_kaggle2':
            path = "data/turkish/offensDetect-kaggle2/test.jsonl"
        elif task == 'abusive_detect':
            path = "data/turkish/ATC/fold_0_test.jsonl"
        elif task == 'spam_detect':
            path = "data/turkish/TurkishSpam/trspam.jsonl"
    absolute = "/workspace/value-adapters/culturellm/"
    return absolute + path

def count_samples(file_path):
    count = 0
    try:
        with open(file_path, "r", encoding="utf8") as f:
            for _ in jsonlines.Reader(f):
                count += 1
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return count

def process_offensiveness_stats():
    total, language_counts = get_offensiveness_stats()
    rows = []
    for lang, count in language_counts.items():
        rows.append([lang, count])
    rows.append(["Overall", total])
    return rows

# -------------------------------
# 6. Main Execution: Write CSVs for each dataset
# -------------------------------
if __name__ == "__main__":
    # WVS CSV
    wvs_rows = process_wvs_stats(CULTURES)
    save_csv(wvs_rows, ["Culture", "SampleCount"], "./wvs_stats.csv")
    
    # Cultural Context CSV
    cc_rows = process_cultural_context_stats(CULTURES, llama_tokenizer)
    save_csv(cc_rows, ["Culture", "NumSentences", "FullTextTokens", "SumSentenceTokens"], "./cultural_context_stats.csv")
    
    # NormAd Sample Counts CSVs
    overall_normad_count, normad_country_counts, normad_culture_counts = process_normad_sample_counts(CULTURES)
    # Save by country
    save_csv(normad_country_counts, ["Country", "SampleCount"], "./normad_sample_counts_by_country.csv")
    # Save by culture
    save_csv(normad_culture_counts, ["Culture", "SampleCount"], "./normad_sample_counts_by_culture.csv")
    # Save overall count (as one-row CSV)
    save_csv([[ "Overall", overall_normad_count ]], ["Category", "SampleCount"], "./normad_sample_counts_overall.csv")
    
    # NormAd Token Analytics CSVs
    overall_tokens, country_stats, culture_stats = process_normad_token_stats(CULTURES, llama_tokenizer)
    # overall_tokens is a tuple: (total_tokens, avg_tokens, sample_count)
    overall_token_rows = [["Overall TotalTokens", overall_tokens[0]],
                          ["Overall AvgTokens", f"{overall_tokens[1]:.2f}"],
                          ["Overall SampleCount", overall_tokens[2]]]
    save_csv(overall_token_rows, ["Metric", "Value"], "./normad_token_overall.csv")
    
    # By Country: create two CSVs: one for total tokens and one for avg tokens
    token_total_country = []
    token_avg_country = []
    for country, total, avg, count in country_stats:
        token_total_country.append([country, total])
        token_avg_country.append([country, f"{avg:.2f}"])
    save_csv(token_total_country, ["Country", "TotalTokens"], "./normad_token_totals_by_country.csv")
    save_csv(token_avg_country, ["Country", "AvgTokens"], "./normad_token_avg_by_country.csv")
    
    # By Culture: similarly, two CSVs for culture-level aggregation
    token_total_culture = []
    token_avg_culture = []
    for culture, total, avg, count in culture_stats:
        token_total_culture.append([culture, total])
        token_avg_culture.append([culture, f"{avg:.2f}"])
    save_csv(token_total_culture, ["Culture", "TotalTokens"], "./normad_token_totals_by_culture.csv")
    save_csv(token_avg_culture, ["Culture", "AvgTokens"], "./normad_token_avg_by_culture.csv")
    
    # Offensiveness CSV
    off_rows = process_offensiveness_stats()
    save_csv(off_rows, ["Language", "SampleCount"], "./offensiveness_stats.csv")
    
    print("\nAll analytics CSVs have been generated.")
