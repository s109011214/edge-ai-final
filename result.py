
from hqq.utils.patching import prepare_for_inference
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np
import sys
import csv
import argparse
from hqq.core.quantize import BaseQuantizeConfig
from hqq_utils import AutoHQQHFModel, get_size_of_model
from hqq.utils.patching import recommended_inductor_config_setter
from quant_cfg import get_quant_config_slm

def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    # with torch.inference_mode():
    with torch.no_grad():
        outputs = model.prefill_forward(
            input_ids,
            past_key_values=past_key_values, 
            position_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=1
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Token-by-token Decoding
        for _ in range(max_new_tokens):
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return input_ids

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)

    nsamples = test_enc.numel() // model.seqlen
    nlls = []
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]

        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    return ppl.item()
def truncate_model_layers(model, keep_layers=16):
    if hasattr(model, "model") and hasattr(model.model, "model"):
        model.model.model.layers = torch.nn.ModuleList(model.model.model.layers[:keep_layers])
        model.config.num_hidden_layers = keep_layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = torch.nn.ModuleList(model.model.layers[:keep_layers])
        model.config.num_hidden_layers = keep_layers
    elif hasattr(model, "layers"):
        model.layers = torch.nn.ModuleList(model.layers[:keep_layers])
        model.config.num_hidden_layers = keep_layers
    else:
        raise AttributeError("Cannot locate layers in the model.")
    return model
def main():

    recommended_inductor_config_setter()
    '''parser = argparse.ArgumentParser(description="Quantized LLaMA3 Evaluation")
    parser.add_argument("--num_layers", type=int, required=True)

    parser.add_argument("--output_csv", type=str, default="no_complie_v1.csv")
    parser.add_argument("--result_csv", type=str, default="result.csv")
    args = parser.parse_args()'''


    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)

    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'

    ### === TODO: Load your model (you may change this part) ===
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        # attn_implementation="xformers"
    )
    model=truncate_model_layers(model, keep_layers=28)
    model.eval()
    model.prefill_forward = model.forward
    model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=False, fullgraph=True)
    #####################################
    
    #model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # === (Optional) Uncomment the following lines if using the custom generate() function. ===
    #model.prefill_forward = model.forward
    #model.forward = torch.compile(model.forward, mode='max-autotune',dynamic=False,fullgraph=True)


    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # === (Optional) Set up StaticCache for manual KV cache management ===
    from transformers import StaticCache
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_new_tokens + 16,
        device=model.device,
        dtype=torch.float16
    )


    ####################################################################

    '''warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    for i in tqdm(range(5), desc="Warm Up..."):
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )'''
    quant_config = get_quant_config_slm(model)
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)

    from hqq.utils.patching import prepare_for_inference
    backend = 'gemlite'
    prepare_for_inference(model, backend=backend)
    
    prompt = "How to learn a new language?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tputs = []
    time_record = []
    # quant_config = get_quant_config_slm(model)

    # AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)
    # backend = 'gemlite'
    # from hqq.utils.patching import prepare_for_inference
    # prepare_for_inference(model, backend=backend)
    # torch.cuda.empty_cache()

    # with torch.inference_mode():
    for _ in tqdm(range(10), desc="Test Inference"):
          print("Running inference...")
          torch.cuda.synchronize()
          start = torch.cuda.Event(enable_timing=True)
          end = torch.cuda.Event(enable_timing=True)
          start.record()

          # === Default: Use model.generate() for end-to-end timing ===
          # generated = model.generate(
          #     input_ids=input_ids,
          #     attention_mask=attention_mask,
          #     max_new_tokens=max_new_tokens,
          #     pad_token_id=tokenizer.eos_token_id,
          # )

          # === Optional: Use custom generate() if uncommented ===
          generated = generate(model, input_ids, past_key_values, max_new_tokens)
          past_key_values.reset()

          end.record()
          torch.cuda.synchronize()
          elapsed_ms = start.elapsed_time(end)
          # tput = max_new_tokens / (elapsed_ms / 1000)
          tput = generated[0][input_ids.shape[1]:].shape[0]/(elapsed_ms / 1000)
          time_record.append(elapsed_ms / 1000)
          tputs.append(tput)

    response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')

    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')

    # new_row = [num, round(org_tput, 2)]

    ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")

    # Save results to CSV
    import csv
    rounded_tput = round(org_tput, 1)
    ppl = round(ppl, 2)

    with open("result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, ppl])
        writer.writerow([1, rounded_tput])
    # Append to summary csv
    # with open(args.output_csv, mode="a", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow([args.num_layers, org_tput, ppl])


    # print(f"[✅] 寫入完成: {args.num_layers} → {org_tput:.2f} toks/s")

if __name__ == '__main__':
    main()