import openai
openai.api_key = "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"

def get_completion_from_messages(message):
    chat_models = "gpt-3.5-turbo" 
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}]
    
    response = openai.ChatCompletion.create(
        model=chat_models,
        messages=messages
    )
    return str(response.choices[0].message['content'])



