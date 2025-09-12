
def getTranslatedPrompt(input_txt, task, adapter_lang):
    prompts = {
        'arabic': {
            'vulgar_detect_mp': f"### سؤال: هل الجملة التالية تحتوي على كلام بذيء؟ {input_txt}. '1. بذيء', '2. ليس بذيء' بدون تفسير. ### إجابة: ",
            'spam_detect': f"### سؤال: هل الجملة التالية عبارة عن رسالة غير مرغوب فيها؟ {input_txt}. '1. غير مرغوب فيه', '2. ليس غير مرغوب فيه' بدون تفسير. ### إجابة: ",
            'hate_detect_fine-grained': f"""### سؤال:
هل الجملة التالية تحتوي على خطاب كراهية؟

{input_txt}

يرجى اختيار أحد الخيارات التالية بدون تفسير:
1. لا خطاب كراهية
2. العرق
3. الدين
4. الأيديولوجية
5. الإعاقة
6. الطبقة الاجتماعية
7. الجنس

### إجابة: """,
            'hate_detect_religion': f"### سؤال: هل الجملة التالية تحتوي على خطاب كراهية متعلق بالدين؟ {input_txt}. '1. خطاب كراهية دين', '2. ليس خطاب كراهية دين' بدون تفسير. ### إجابة: ",
            'offensive_detect_1': f"### سؤال: هل الجملة التالية تحتوي على إساءة؟ {input_txt}. '1. مسيء', '2. ليس مسيء' بدون تفسير. ### إجابة: ",
            'offensive_detect_2': f"### سؤال: هل الجملة التالية تحتوي على إساءة؟ {input_txt}. '1. مسيء', '2. ليس مسيء' بدون تفسير. ### إجابة: ",
            'offensive_detect_3': f"### سؤال: هل الجملة التالية تحتوي على إساءة؟ {input_txt}. '1. مسيء', '2. ليس مسيء' بدون تفسير. ### إجابة: ",
            'racism_detect': f"### سؤال: هل الجملة التالية تحتوي على عنصرية؟ {input_txt}. '1. عنصري', '2. غير عنصري' بدون تفسير. ### إجابة: ",
            'threat_detect': f"### سؤال: هل الجملة التالية تحتوي على تهديد؟ {input_txt}. '1. تهديد', '2. ليس تهديد' بدون تفسير. ### إجابة: ",
            'abusive_detect': f"### سؤال: هل الجملة التالية تحتوي على لغة مسيئة؟ {input_txt}. '1. مسيء', '2. غير مسيء' بدون تفسير. ### إجابة: ",
            'abusive_detect_2': f"### سؤال: هل الجملة التالية تحتوي على لغة مسيئة؟ {input_txt}. '1. مسيء', '2. غير مسيء' بدون تفسير. ### إجابة: ",
            'abusive_detect_4': f"### سؤال: هل الجملة التالية تحتوي على لغة مسيئة؟ {input_txt}. '1. مسيء', '2. غير مسيء' بدون تفسير. ### إجابة: ",
            'hate_detect_3': f"### سؤال: هل الجملة التالية تحتوي على خطاب كراهية؟ {input_txt}. '1. خطاب كراهية', '2. ليس خطاب كراهية' بدون تفسير. ### إجابة: ",
            'hate_detect_6': f"### سؤال: هل الجملة التالية تحتوي على خطاب كراهية؟ {input_txt}. '1. خطاب كراهية', '2. ليس خطاب كراهية' بدون تفسير. ### إجابة: ",
            'hate_detect_7': f"### سؤال: هل الجملة التالية تحتوي على خطاب كراهية؟ {input_txt}. '1. خطاب كراهية', '2. ليس خطاب كراهية' بدون تفسير. ### إجابة: ",
            'homophobia_detect': f"### سؤال: هل الجملة التالية تحتوي على معاداة المثليين؟ {input_txt}. '1. معاداة المثليين', '2. لا معاداة المثليين' بدون تفسير. ### إجابة: ",
            'insult_detect': f"### سؤال: هل الجملة التالية تحتوي على إهانة؟ {input_txt}. '1. إهانة', '2. لا إهانة' بدون تفسير. ### إجابة: ",
            'misogyny_detect': f"### سؤال: هل الجملة التالية تحتوي على كراهية النساء؟ {input_txt}. '1. كراهية النساء', '2. لا كراهية النساء' بدون تفسير. ### إجابة: ",
            'offensive_detect_corpus': f"### سؤال: هل الجملة التالية تحتوي على إساءة؟ {input_txt}. '1. مسيء', '2. ليس مسيء' بدون تفسير. ### إجابة: ",
            'offensive_detect_finegrained': f"""### سؤال:
هل الجملة التالية تحتوي على لغة مسيئة؟

{input_txt}

يرجى اختيار أحد الخيارات التالية بدون تفسير:
1. لا
2. ألفاظ نابية أو إساءة غير مستهدفة
3. إساءة نحو مجموعة
4. إساءة نحو فرد
5. إساءة نحو كيان آخر (غير بشري)، غالبًا حدث أو منظمة

### إجابة: """,
            'offensive_detect_kaggle': f"### سؤال: هل الجملة التالية تحتوي على إساءة؟ {input_txt}. '1. مسيء', '2. ليس مسيء' بدون تفسير. ### إجابة: ",
            'offensive_detect_kaggle2': f"### سؤال: هل الجملة التالية تحتوي على إساءة؟ {input_txt}. '1. مسيء', '2. ليس مسيء' بدون تفسير. ### إجابة: ",
            'offensive_detect_mex_a3t': f"### سؤال: هل الجملة التالية تحتوي على إساءة؟ {input_txt}. '1. مسيء', '2. ليس مسيء' بدون تفسير. ### إجابة: ",
            'offensive_detect_mex_offend': f"### سؤال: هل الجملة التالية تحتوي على إساءة؟ {input_txt}. '1. مسيء', '2. ليس مسيء' بدون تفسير. ### إجابة: ",
            'offensive_detect_osact4': f"### سؤال: هل الجملة التالية تحتوي على إساءة؟ {input_txt}. '1. مسيء', '2. ليس مسيء' بدون تفسير. ### إجابة: ",
            'offensive_detect_osact5': f"### سؤال: هل الجملة التالية تحتوي على إساءة؟ {input_txt}. '1. مسيء', '2. ليس مسيء' بدون تفسير. ### إجابة: ",
            'offensive_detect_eval': f"### سؤال: هل الجملة التالية تحتوي على إساءة؟ {input_txt}. '1. مسيء', '2. ليس مسيء' بدون تفسير. ### إجابة: ",
            'offensive_detect_easy': f"### سؤال: هل الجملة التالية تحتوي على إساءة بسيطة؟ {input_txt}. '1. بسيط', '2. ليس بسيط' بدون تفسير. ### إجابة: ",
            'bias_on_gender_detect': f"### سؤال: هل الجملة التالية تحتوي على تحيز جنساني؟ {input_txt}. '1. نعم', '2. لا' بدون تفسير. ### إجابة: ",
            'hostility_directness_detect': f"### سؤال: هل الجملة التالية تحتوي على عدوانية مباشرة؟ {input_txt}. '1. نعم', '2. لا' بدون تفسير. ### إجابة: ",
            'hate_offens_detect': f"### سؤال: هل الجملة التالية تحتوي على خطاب كراهية أو محتوى مسيء؟ {input_txt}. '1. خطاب كراهية أو مسيء', '2. ليس خطاب كراهية أو مسيء' بدون تفسير. ### إجابة: ",
            'aggressiveness_detect': f"### سؤال: هل الجملة التالية تحتوي على عدوانية؟ {input_txt}. '1. نعم', '2. لا' بدون تفسير. ### إجابة: ",
            'improper_detect': f"### سؤال: هل الجملة التالية تحتوي على لغة غير لائقة؟ {input_txt}. '1. نعم', '2. لا' بدون تفسير. ### إجابة: ",
            'insult_detect': f"### سؤال: هل الجملة التالية تحتوي على إهانة؟ {input_txt}. '1. إهانة', '2. لا إهانة' بدون تفسير. ### إجابة: ",
            'mockery_detect': f"### سؤال: هل الجملة التالية تحتوي على سخرية؟ {input_txt}. '1. سخرية', '2. لا سخرية' بدون تفسير. ### إجابة: ",
            'negative_stance_detect': f"### سؤال: هل الجملة التالية تعبر عن موقف سلبي؟ {input_txt}. '1. نعم', '2. لا' بدون تفسير. ### إجابة: ",
            'racism_detect': f"### سؤال: هل الجملة التالية تحتوي على عنصرية؟ {input_txt}. '1. عنصري', '2. غير عنصري' بدون تفسير. ### إجابة: ",
            'stereotype_detect': f"### سؤال: هل الجملة التالية تحتوي على نمطية؟ {input_txt}. '1. نمطية', '2. لا نمطية' بدون تفسير. ### إجابة: ",
            'toxicity_detect': f"### سؤال: هل الجملة التالية تحتوي على سمية؟ {input_txt}. '1. سمية', '2. غير سمية' بدون تفسير. ### إجابة: ",
            'threat_detect': f"### سؤال: هل الجملة التالية تحتوي على تهديد؟ {input_txt}. '1. تهديد', '2. ليس تهديد' بدون تفسير. ### إجابة: ",
            'hate_detect_eval': f"### سؤال: هل الجملة التالية تحتوي على خطاب كراهية؟ {input_txt}. '1. خطاب كراهية', '2. لا خطاب كراهية' بدون تفسير. ### إجابة: ",
            'hate_detect_haterNet': f"### سؤال: هل الجملة التالية تحتوي على خطاب كراهية؟ {input_txt}. '1. خطاب كراهية', '2. لا خطاب كراهية' بدون تفسير. ### إجابة: ",
        },
        'bengali': {
            'hate_detect_religion': f"### প্রশ্ন: নিম্নলিখিত বাক্যটি ধর্ম সংক্রান্ত ঘৃণামূলক বক্তৃতা ধারণ করে কিনা: {input_txt}. '1. ধর্ম সংক্রান্ত ঘৃণামূলক বক্তৃতা', '2. ধর্ম সংক্রান্ত ঘৃণামূলক বক্তৃতা নয়' ব্যাখ্যা ছাড়াই। ### উত্তর: ",
            'offensive_detect_1': f"### প্রশ্ন: নিম্নলিখিত বাক্যটি আক্রমণাত্মক কিনা: {input_txt}. '1. আক্রমণাত্মক', '2. আক্রমণাত্মক নয়' ব্যাখ্যা ছাড়াই। ### উত্তর: ",
            'offensive_detect_2': f"### প্রশ্ন: নিম্নলিখিত বাক্যটি আক্রমণাত্মক কিনা: {input_txt}. '1. আক্রমণাত্মক', '2. আক্রমণাত্মক নয়' ব্যাখ্যা ছাড়াই। ### উত্তর: ",
            'offensive_detect_3': f"### প্রশ্ন: নিম্নলিখিত বাক্যটি আক্রমণাত্মক কিনা: {input_txt}. '1. আক্রমণাত্মক', '2. আক্রমণাত্মক নয়' ব্যাখ্যা ছাড়াই। ### উত্তর: ",
            'racism_detect': f"### প্রশ্ন: নিম্নলিখিত বাক্যটি বর্ণবাদী কিনা: {input_txt}. '1. বর্ণবাদী', '2. বর্ণবাদী নয়' ব্যাখ্যা ছাড়াই। ### উত্তর: ",
            'threat_detect': f"### প্রশ্ন: নিম্নলিখিত বাক্যটি হুমকি ধারণ করে কিনা: {input_txt}. '1. হুমকি', '2. হুমকি নয়' ব্যাখ্যা ছাড়াই। ### উত্তর: ",
            'hate_detect_fine-grained': f"""### প্রশ্ন:
নিম্নলিখিত বাক্যটি ঘৃণামূলক বক্তৃতা ধারণ করে কিনা?

{input_txt}

ব্যাখ্যা ছাড়াই নিম্নলিখিত বিকল্পগুলির মধ্যে একটি নির্বাচন করুন:
1. নয়
2. জাতি
3. ধর্ম
4. মতাদর্শ
5. প্রতিবন্ধিতা
6. সামাজিক শ্রেণী
7. লিঙ্গ

### উত্তর: """,
            'abusive_detect': f"### প্রশ্ন: নিম্নলিখিত বাক্যটি অপমানসূচক ভাষা ধারণ করে কিনা: {input_txt}. '1. অপমানসূচক', '2. অপমানসূচক নয়' ব্যাখ্যা ছাড়াই। ### উত্তর: ",
            'abusive_detect_2': f"### প্রশ্ন: নিম্নলিখিত বাক্যটি অপমানসূচক ভাষা ধারণ করে কিনা: {input_txt}. '1. অপমানসূচক', '2. অপমানসূচক নয়' ব্যাখ্যা ছাড়াই। ### উত্তর: ",
            'abusive_detect_4': f"### প্রশ্ন: নিম্নলিখিত বাক্যটি অপমানসূচক ভাষা ধারণ করে কিনা: {input_txt}. '1. অপমানসূচক', '2. অপমানসূচক নয়' ব্যাখ্যা ছাড়াই। ### উত্তর: ",
            'hate_detect_3': f"### প্রশ্ন: নিম্নলিখিত বাক্যটি ঘৃণামূলক বক্তৃতা ধারণ করে কিনা: {input_txt}. '1. ঘৃণামূলক বক্তৃতা', '2. ঘৃণামূলক বক্তৃতা নয়' ব্যাখ্যা ছাড়াই। ### উত্তর: ",
            'hate_detect_6': f"### প্রশ্ন: নিম্নলিখিত বাক্যটি ঘৃণামূলক বক্তৃতা ধারণ করে কিনা: {input_txt}. '1. ঘৃণামূলক বক্তৃতা', '2. ঘৃণামূলক বক্তৃতা নয়' ব্যাখ্যা ছাড়াই। ### উত্তর: ",
            'hate_detect_7': f"### প্রশ্ন: নিম্নলিখিত বাক্যটি ঘৃণামূলক বক্তৃতা ধারণ করে কিনা: {input_txt}. '1. ঘৃণামূলক বক্তৃতা', '2. ঘৃণামূলক বক্তৃতা নয়' ব্যাখ্যা ছাড়াই। ### উত্তর: ",
        },
        'chinese': {
            'bias_on_gender_detect': f"### 问题: 以下句子是否包含性别偏见？{input_txt}. '1. 是', '2. 否' 无需解释。 ### 答案: ",
            'spam_detect': f"### 问题: 以下句子是否是垃圾邮件？{input_txt}. '1. 垃圾邮件', '2. 不是垃圾邮件' 无需解释。 ### 答案: ",
        },
        'greek': {
            'offensive_detect': f"### Ερώτηση: Περιέχει η ακόλουθη πρόταση προσβλητικό περιεχόμενο; {input_txt}. '1. Προσβλητικό', '2. Όχι προσβλητικό' χωρίς εξήγηση. ### Απάντηση: ",
            'offensive_detect_g': f"### Ερώτηση: Περιέχει η ακόλουθη πρόταση προσβλητικό περιεχόμενο; {input_txt}. '1. Προσβλητικό', '2. Όχι προσβλητικό' χωρίς εξήγηση. ### Απάντηση: ",
        },
        'english': {
            'offensive_detect': f"### Question: is the following sentence offensive? {input_txt}. '1. Offensive', '2. Not offensive' without explanation. ### Answer: ",
            'hate_detect': f"### Question: does the following sentence contain hate speech? {input_txt}. '1. Hatespeech', '2. Not Hatespeech' without explanation. ### Answer: ",
            'spam_detect': f"### Question: is the following sentence a spam tweet? {input_txt}. '1. Spam', '2. Not Spam' without explanation. ### Answer: ",
            'hate_detect_2': f"### Question: does the following sentence contain hate speech? {input_txt}. '1. Hatespeech', '2. Not Hatespeech' without explanation. ### Answer: ",
            'hate_offens_detect': f"### Question: does the following sentence contain hate speech or offensive content? {input_txt}. '1. Hate or Offensive', '2. Not Hate or Offensive' without explanation. ### Answer: ",
            'hostility_directness_detect': f"### Question: does the following sentence contain direct hostility? {input_txt}. '1. Yes', '2. No' without explanation. ### Answer: ",
            'offensive_detect_easy': f"### Question: is the following sentence mildly offensive? {input_txt}. '1. Mildly offensive', '2. Not mildly offensive' without explanation. ### Answer: ",
            'toxicity_detect': f"### Question: does the following sentence contain toxic language? {input_txt}. '1. Toxic', '2. Not toxic' without explanation. ### Answer: ",
            'threat_detect': f"### Question: does the following sentence contain a threat? {input_txt}. '1. Threat', '2. Not threat' without explanation. ### Answer: ",
            'negative_stance_detect': f"### Question: does the following sentence express a negative stance? {input_txt}. '1. Yes', '2. No' without explanation. ### Answer: ",
            'hate_detect_eval': f"### Question: does the following sentence contain hate speech? {input_txt}. '1. Hate speech', '2. No hate speech' without explanation. ### Answer: ",
            'hate_detect_haterNet': f"### Question: does the following sentence contain hate speech? {input_txt}. '1. Hate speech', '2. No hate speech' without explanation. ### Answer: ",
        },
        'german': {
            'hate_detect': f"### Frage: Enthält der folgende Satz Hassrede? {input_txt}. '1. Hassrede', '2. Keine Hassrede' ohne Erklärung. ### Antwort: ",
            'hate_off_detect': f"### Frage: Enthält der folgende Satz Hassrede oder anstößigen Inhalt? {input_txt}. '1. Hassrede oder anstößig', '2. Keine Hassrede oder anstößig' ohne Erklärung. ### Antwort: ",
            'hate_detect_iwg_1': f"### Frage: Enthält der folgende Satz Hassrede? {input_txt}. '1. Hassrede', '2. Keine Hassrede' ohne Erklärung. ### Antwort: ",
            'hate_detect_check': f"### Frage: Enthält der folgende Satz Hassrede? {input_txt}. '1. Hassrede', '2. Keine Hassrede' ohne Erklärung. ### Antwort: ",
            'offensive_detect_eval': f"### Frage: Enthält der folgende Satz anstößige Sprache? {input_txt}. '1. Anstößig', '2. Nicht anstößig' ohne Erklärung. ### Antwort: ",
        },
        'korean': {
            'abusive_detect': f"### 질문: 다음 문장이 욕설을 포함하고 있습니까? {input_txt}. '1. 욕설', '2. 욕설 아님' 설명 없이. ### 답변: ",
            'abusive_detect_2': f"### 질문: 다음 문장이 욕설을 포함하고 있습니까? {input_txt}. '1. 욕설', '2. 욕설 아님' 설명 없이. ### 답변: ",
            'abusive_detect_4': f"### 질문: 다음 문장이 욕설을 포함하고 있습니까? {input_txt}. '1. 욕설', '2. 욕설 아님' 설명 없이. ### 답변: ",
            'hate_detect_3': f"### 질문: 다음 문장이 증오 발언을 포함하고 있습니까? {input_txt}. '1. 증오 발언', '2. 증오 발언 아님' 설명 없이. ### 답변: ",
            'hate_detect_6': f"### 질문: 다음 문장이 증오 발언을 포함하고 있습니까? {input_txt}. '1. 증오 발언', '2. 증오 발언 아님' 설명 없이. ### 답변: ",
            'hate_detect_7': f"### 질문: 다음 문장이 증오 발언을 포함하고 있습니까? {input_txt}. '1. 증오 발언', '2. 증오 발언 아님' 설명 없이. ### 답변: ",
            'offensive_detect': f"### 질문: 다음 문장이 공격적인가요? {input_txt}. '1. 공격적', '2. 공격적 아님' 설명 없이. ### 답변: ",
            'offensive_detect_corpus': f"### 질문: 다음 문장이 공격적인가요? {input_txt}. '1. 공격적', '2. 공격적 아님' 설명 없이. ### 답변: ",
            'offensive_detect_finegrained': f"""### 질문:
다음 문장이 공격적인 언어를 포함하고 있습니까?

{input_txt}

설명 없이 다음 옵션 중 하나를 선택하십시오:
1. 아니요
2. 욕설 또는 비표적 공격
3. 그룹에 대한 공격
4. 개인에 대한 공격
5. 다른(비인간) 실체에 대한 공격, 종종 이벤트나 조직

### 답변: """,
            'offensive_detect_kaggle': f"### 질문: 다음 문장이 공격적인가요? {input_txt}. '1. 공격적', '2. 공격적 아님' 설명 없이. ### 답변: ",
            'offensive_detect_kaggle2': f"### 질문: 다음 문장이 공격적인가요? {input_txt}. '1. 공격적', '2. 공격적 아님' 설명 없이. ### 답변: ",
            'offensive_detect_easy': f"### 질문: 다음 문장이 약간 공격적인가요? {input_txt}. '1. 약간 공격적', '2. 약간 공격적 아님' 설명 없이. ### 답변: ",
        },
        'portuguese': {
            'homophobia_detect': f"### Pergunta: A frase a seguir contém homofobia? {input_txt}. '1. Homofobia', '2. Não homofobia' sem explicação. ### Resposta: ",
            'insult_detect': f"### Pergunta: A frase a seguir contém insulto? {input_txt}. '1. Insulto', '2. Não insulto' sem explicação. ### Resposta: ",
            'misogyny_detect': f"### Pergunta: A frase a seguir contém misoginia? {input_txt}. '1. Misoginia', '2. Não misoginia' sem explicação. ### Resposta: ",
            'offensive_detect_2': f"### Pergunta: A frase a seguir é ofensiva? {input_txt}. '1. Ofensiva', '2. Não ofensiva' sem explicação. ### Resposta: ",
            'offensive_detect_3': f"### Pergunta: A frase a seguir é ofensiva? {input_txt}. '1. Ofensiva', '2. Não ofensiva' sem explicação. ### Resposta: ",
        },
        'spanish': {
            'offensive_detect_ami': f"### Pregunta: ¿La siguiente oración es ofensiva? {input_txt}. '1. Ofensivo', '2. No ofensivo' sin explicación. ### Respuesta: ",
            'offensive_detect_mex_a3t': f"### Pregunta: ¿La siguiente oración es ofensiva? {input_txt}. '1. Ofensivo', '2. No ofensivo' sin explicación. ### Respuesta: ",
            'offensive_detect_mex_offend': f"### Pregunta: ¿La siguiente oración es ofensiva? {input_txt}. '1. Ofensivo', '2. No ofensivo' sin explicación. ### Respuesta: ",
            'hate_detect_eval': f"### Pregunta: ¿La siguiente oración contiene discurso de odio? {input_txt}. '1. Discurso de odio', '2. No discurso de odio' sin explicación. ### Respuesta: ",
            'hate_detect_haterNet': f"### Pregunta: ¿La siguiente oración contiene discurso de odio? {input_txt}. '1. Discurso de odio', '2. No discurso de odio' sin explicación. ### Respuesta: ",
            'stereotype_detect': f"### Pregunta: ¿La siguiente oración contiene estereotipos? {input_txt}. '1. Estereotipos', '2. No estereotipos' sin explicación. ### Respuesta: ",
            'mockery_detect': f"### Pregunta: ¿La siguiente oración contiene burla? {input_txt}. '1. Burla', '2. No burla' sin explicación. ### Respuesta: ",
            'insult_detect': f"### Pregunta: ¿La siguiente oración contiene insultos? {input_txt}. '1. Insultos', '2. No insultos' sin explicación. ### Respuesta: ",
            'improper_detect': f"### Pregunta: ¿La siguiente oración contiene lenguaje inapropiado? {input_txt}. '1. Inapropiado', '2. No inapropiado' sin explicación. ### Respuesta: ",
            'aggressiveness_detect': f"### Pregunta: ¿La siguiente oración contiene agresividad? {input_txt}. '1. Agresivo', '2. No agresivo' sin explicación. ### Respuesta: ",
            'negative_stance_detect': f"### Pregunta: ¿La siguiente oración expresa una postura negativa? {input_txt}. '1. Sí', '2. No' sin explicación. ### Respuesta: ",
        },
        'turkish': {
            'offensive_detect': f"### Soru: Aşağıdaki cümle saldırgan mı? {input_txt}. '1. Saldırgan', '2. Saldırgan değil' açıklama olmadan. ### Cevap: ",
            'offensive_detect_corpus': f"### Soru: Aşağıdaki cümle saldırgan mı? {input_txt}. '1. Saldırgan', '2. Saldırgan değil' açıklama olmadan. ### Cevap: ",
            'offensive_detect_finegrained': f"""### Soru:
Aşağıdaki cümle saldırgan mı?

{input_txt}

Açıklama olmadan aşağıdaki seçeneklerden birini seçiniz:
1. Saldırgan değil
2. Küfür veya hedeflenmemiş saldırı
3. Bir gruba yönelik saldırı
4. Bir bireye yönelik saldırı
5. Başka (insan olmayan) bir varlığa yönelik saldırı, genellikle bir etkinlik veya organizasyon

### Cevap: """,
            'offensive_detect_kaggle': f"### Soru: Aşağıdaki cümle saldırgan mı? {input_txt}. '1. Saldırgan', '2. Saldırgan değil' açıklama olmadan. ### Cevap: ",
            'offensive_detect_kaggle2': f"### Soru: Aşağıdaki cümle saldırgan mı? {input_txt}. '1. Saldırgan', '2. Saldırgan değil' açıklama olmadan. ### Cevap: ",
            'abusive_detect': f"### Soru: Aşağıdaki cümle kötüleyici dil içeriyor mu? {input_txt}. '1. Kötüleyici', '2. Kötüleyici değil' açıklama olmadan. ### Cevap: ",
            'abusive_detect_2': f"### Soru: Aşağıdaki cümle kötüleyici dil içeriyor mu? {input_txt}. '1. Kötüleyici', '2. Kötüleyici değil' açıklama olmadan. ### Cevap: ",
            'abusive_detect_4': f"### Soru: Aşağıdaki cümle kötüleyici dil içeriyor mu? {input_txt}. '1. Kötüleyici', '2. Kötüleyici değil' açıklama olmadan. ### Cevap: ",
            'spam_detect': f"### Soru: Aşağıdaki cümle spam tweet mi? {input_txt}. '1. Spam', '2. Spam değil' açıklama olmadan. ### Cevap: ",
        },
    }

    # Return the translated prompt based on the adapter language and task
    return prompts.get(adapter_lang, {}).get(task, f"### Question: {input_txt} ### Answer: ")