# THIS CODE IS MAINLY FROM: https://github.com/ltgoslo/definition_modeling

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tqdm


def define(in_prompts, lm, cur_tokenizer, targets, maxl=256, bsize=4, filter_target=False, num_beams=1,
        num_beam_groups=1, sampling=False, temperature=1.0, repetition_penalty=1.0):
    
    inputs = cur_tokenizer(
        in_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=maxl,
    )

    target_ids = cur_tokenizer(targets, add_special_tokens=False).input_ids
    target_ids = torch.tensor([el[-1] for el in target_ids])

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
        target_ids = target_ids.to("cuda")

    test_dataset = torch.utils.data.TensorDataset(inputs["input_ids"], inputs["attention_mask"],
                                                  target_ids)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=bsize, shuffle=False)

    gen_args = dict(do_sample=sampling, num_beams=num_beams, num_beam_groups=num_beam_groups,
            temperature=temperature, repetition_penalty=repetition_penalty)
    if num_beam_groups > 1:
        gen_args["diversity_penalty"] = 0.5
    definitions = []
    for inp, att, targetwords in tqdm.tqdm(test_iter):
        if filter_target:
            bad = [[el] for el in targetwords.tolist()]
            outputs = lm.generate(input_ids=inp, attention_mask=att, max_new_tokens=60,
                                  bad_words_ids=bad, **gen_args)
        else:
            outputs = lm.generate(input_ids=inp, attention_mask=att, max_new_tokens=60,
                                  **gen_args)
        predictions = cur_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        definitions += predictions
    
    return definitions



if __name__ == "__main__":

    model_name = 'ltg/flan-t5-definition-en-xl' # or base, large or xl
    data_file = 'HateWiC_IndividualAnnos.csv'
    test_df = pd.read_csv(data_file, sep=';')
    

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    if torch.cuda.is_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)

    task_prefix = "What is the definition of <TRG>?"
    input_sentences = []
    for target, context in zip(test_df['term'], test_df['example']):
        prompt = " ".join([context, task_prefix.replace("<TRG>", target)])
        input_sentences.append(prompt)

    targets = test_df['term'].tolist()
    answers = define(input_sentences, model, tokenizer, targets, filter_target=1) # all default values from Giulianelli et al. (2023)
    
    test_df["t5xl_definition"] = answers
    test_df.to_csv(data_file.replace('.csv', '_GenDefs.csv'), index=False) 