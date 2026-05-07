import json
import os
import sys
import urllib.error
import urllib.request

try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential
    has_azure_sdk = True
except ImportError:
    has_azure_sdk = False

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-mini"
token = os.environ.get("GITHUB_TOKEN")

if not token:
    sys.exit("Error: GITHUB_TOKEN is not set in the environment.")

messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "What is the capital of Telangana?"},
]

if has_azure_sdk:
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    response = client.complete(
        messages=[
            SystemMessage(""),
            UserMessage("What is the capital of Telangana?"),
        ],
        temperature=1,
        top_p=1,
        model=model,
    )

    print(response.choices[0].message.content)
else:
    def call_inference(url, headers):
        body = {
            "messages": messages,
            "temperature": 1,
            "top_p": 1,
            "model": model,
        }
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))

    url_candidates = [
        f"{endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-12-01",
        f"{endpoint}/v1/chat/completions",
        f"{endpoint}/chat/completions",
        f"{endpoint}/openai/chat/completions",
        f"{endpoint}/v1/engines/{model}/completions",
    ]
    headers_candidates = [
        {"Content-Type": "application/json", "api-key": token},
        {"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    ]

    result = None
    last_error = None
    for url in url_candidates:
        for headers in headers_candidates:
            try:
                result = call_inference(url, headers)
                print(f"Request succeeded via {url}")
                break
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP Error {exc.code}: {exc.reason}\n{exc.read().decode('utf-8', errors='ignore')}"
                if exc.code in {401, 403, 404}:
                    continue
                break
            except urllib.error.URLError as exc:
                last_error = f"URL Error: {exc.reason}"
                continue
        if result is not None:
            break

    if result is not None:
        content = result["choices"][0]["message"]["content"]
        print(content)
    else:
        print("Inference request failed. Check the endpoint, token, and model configuration.")
        if last_error:
            print(last_error)

