from model_api import query_model_with_image, query_model_with_text

def run_extraction(models, pages, extraction_prompt, max_tokens, temperature):
    results = {}
    for model in models:
        results[model] = []
        for page in pages:
            response = query_model_with_image(model, page, extraction_prompt, max_tokens, temperature)
            results[model].append(response["content"])
    return results

def run_inference(models, extracted_data, user_prompt, max_tokens, temperature):
    inference_results = {}
    for model in models:
        full_context = "\n\n".join(extracted_data[model])
        response = query_model_with_text(model, full_context, user_prompt, max_tokens, temperature)
        inference_results[model] = response
    return inference_results
