import os  
import json
import re
import torch
from flask import Flask, request, Response
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

app = Flask(__name__)
model_bundle = None

def model_fn(model_dir):
    model_path = model_dir
    
    # Verify critical files exist
    required_files = [
        "config.json",
        "pytorch_model.bin.index.json",
        "tokenizer_config.json"
    ]
    
    for f in required_files:
        if not os.path.exists(os.path.join(model_path, f)): 
            raise ValueError(f"Missing required model file: {f}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    try:
        model = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            load_in_4bit=True,
            device_map="auto"
        )[0]
        
        model.eval()
        FastLanguageModel.for_inference(model)
        return {"model": model, "tokenizer": tokenizer}
        
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

@app.route('/ping', methods=['GET'])
def ping():
    if not torch.cuda.is_available():
        return Response("GPU not available", status=503)
    
    if model_bundle is None:
        return Response("Model not loaded", status=503)
    
    return Response("OK", status=200)

@app.route('/invocations', methods=['POST'])
def invocations():
    try:
        data = request.get_json()
        if model_bundle is None:
            return Response("Model not loaded", status=503)
            
        result = predict_fn(data, model_bundle)
        return Response(json.dumps(result), status=200, mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')


def predict_fn(data, model_bundle):
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]

    # ----------- Input Normalization -----------
    pages = []

    if isinstance(data, dict) and "ocr" in data:
        pages = data["ocr"]
        if not isinstance(pages, list):
            return {"error": "'ocr' must be a list of strings.", "valid": False}
    elif isinstance(data.get("inputs"), list):
        pages = data["inputs"]
    elif isinstance(data.get("inputs"), str):
        pages = [data["inputs"]]  # Single string treated as one page
    else:
        return {"error": "Invalid input format.", "valid": False}

    # Optional generation params
    max_tokens = data.get("max_tokens", 64)
    temperature = data.get("temperature", 0.3)
    top_p = data.get("top_p", 0.95)

    results = {}

    for i, page_text in enumerate(pages):
        if not isinstance(page_text, str) or not page_text.strip():
            continue  # Skip empty or non-string pages

        messages = [
            {
                "role": "system",
                "content": "Classify this document into EXACTLY ONE of these types: invoice, bill_of_lading, packing_list, mill_certificate. Return ONLY the label."
            },
            {"role": "user", "content": page_text}
        ]

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=False,
            )

            decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
            if "<|start_header_id|>assistant<|end_header_id|>" in decoded_output:
                response = decoded_output.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            else:
                response = decoded_output

            cleaned = (
                response.replace("<|eot_id|>", "")
                        .replace("<|end_of_turn|>", "")
                        .strip()
            )

            def extract_label(text):
                text = text.lower().strip()
                text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("_", " ")

                if "invoice" in text:
                    return "invoice"
                elif "bill of lading" in text:
                    return "bill_of_lading"
                elif "packing list" in text:
                    return "packing_list"
                elif "mill" in text:
                    return "mill_certificate"
                return "unknown"

            label = extract_label(cleaned)

            if label not in results:
                results[label] = []
            results[label].append([i])

        except Exception as e:
            if "error" not in results:
                results["error"] = []
            results["error"].append({"page": i, "message": str(e)})

    return results



if __name__ == "__main__":
    try:
        model_bundle = model_fn("/opt/ml/model")
        app.run(host='0.0.0.0', port=8080)
    except Exception as e:
        print(f"Failed to start server: {str(e)}")
        raise

