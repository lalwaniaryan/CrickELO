import os, time, json, requests
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ Missing OPENROUTER_API_KEY in .env")

BASE_URL = "https://openrouter.ai/api/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",  # required by OpenRouter
    "X-Title": "SmartCoach Test",         # required by OpenRouter
}

# Allow overriding the model via env; fall back to your choice
MODEL_ID = os.getenv("OR_MODEL_ID", "openai/gpt-oss-120b:free")


# --- Helpers ---
def post_with_retries(url: str, json_payload: dict, max_retries: int = 4, timeout: int = 90):
    """
    POST with simple backoff. Handles 429 using Retry-After if present.
    Raises for non-2xx (except 429 which is retried).
    """
    attempt = 0
    while True:
        r = requests.post(url, headers=HEADERS, json=json_payload, timeout=timeout)
        if r.status_code == 404 and "Zero data retention" in r.text:
            raise SystemExit(
                "❌ ZDR-only is ON and this model has no ZDR route.\n"
                "Pick a ZDR-listed model or disable ZDR-only in OpenRouter privacy settings."
            )
        if r.status_code != 429:
            r.raise_for_status()
            return r

        # 429 handling
        attempt += 1
        if attempt > max_retries:
            raise SystemExit(f"❌ Rate limited (429) after {max_retries} retries.\nResponse: {r.text}")

        retry_after = r.headers.get("Retry-After")
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 2 ** attempt  # fallback
        else:
            delay = 2 ** attempt  # exponential backoff: 2,4,8,...

        print(f"⚠️ 429 Too Many Requests — retrying in {delay:.1f}s (attempt {attempt}/{max_retries})")
        time.sleep(delay)


def extract_text_safely(data: dict) -> str:
    """
    Try multiple places providers may put content:
    - choices[0].message.content
    - choices[0].message.reasoning
    - choices[0].message.tool_calls (as JSON preview)
    """
    try:
        choice = data["choices"][0]
        msg = choice.get("message", {}) or {}

        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        reasoning = msg.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

        tools = msg.get("tool_calls")
        if tools:
            return "[Tool call]\n" + json.dumps(tools, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return ""


# --- Main flow ---
# Step 1: List available models (sanity check)
print("=== Fetching available models ===")
models_resp = requests.get(f"{BASE_URL}/models", headers=HEADERS, timeout=30)
if models_resp.status_code != 200:
    print("❌ Failed to fetch models:", models_resp.status_code, models_resp.text)
    raise SystemExit(1)

models = models_resp.json()
print(f"✅ Found {len(models.get('data', []))} models.")
print("Example models:", [m["id"] for m in models["data"][:5]], "\n")

# Step 2: Pick a model (from env or hardcoded)
print(f"Using model: {MODEL_ID}\n")

# Step 3: Chat completion (with robust params)
payload = {
    "model": MODEL_ID,
    "messages": [
        {"role": "system", "content": "You are a helpful fitness assistant."},
        {"role": "user", "content": "Give me 3 simple workout tips for beginners."},
    ],
    # Nudge providers to return plain text in message.content
    "response_format": {"type": "text"},
    # Request reasoning if supported (ignored otherwise)
    "include_reasoning": True,
    # Some “thinking” models only return content if thinking is disabled
    "thinking": {"type": "disabled"},
    "provider": {"allow_fallbacks": True},
    "max_tokens": 300,
    "temperature": 0.5,
}

print("=== Sending chat request ===")
resp = post_with_retries(f"{BASE_URL}/chat/completions", payload, max_retries=4, timeout=90)

data = resp.json()
reply = extract_text_safely(data)

if reply:
    print("✅ AI Response:\n")
    print(reply)
else:
    print("⚠️ Empty content. Raw JSON for debugging:\n")
    print(json.dumps(data, ensure_ascii=False, indent=2))
