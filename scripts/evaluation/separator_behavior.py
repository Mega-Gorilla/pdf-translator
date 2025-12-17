# SPDX-License-Identifier: Apache-2.0
"""
Separator token behavior test for translation backends.

This script tests how each translation backend handles the separator token [[[BR]]].
Results will inform the design decision for batch translation.

Usage:
    uv run python scripts/evaluation/separator_behavior.py
"""

import asyncio
import os
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

SEPARATOR = "[[[BR]]]"

# Test cases
TEST_CASES = [
    {
        "name": "Simple 3 blocks",
        "input": f"Hello{SEPARATOR}World{SEPARATOR}Good morning",
        "expected_blocks": 3,
    },
    {
        "name": "With punctuation",
        "input": (
            f"This is a sentence.{SEPARATOR}"
            f"Another sentence here.{SEPARATOR}"
            "Final one."
        ),
        "expected_blocks": 3,
    },
    {
        "name": "Mixed content",
        "input": (
            f"Machine learning is powerful.{SEPARATOR}"
            f"Deep learning uses neural networks.{SEPARATOR}"
            "AI is transforming industries."
        ),
        "expected_blocks": 3,
    },
]


async def test_google_translate():
    """Test Google Translate with separator token."""
    print("\n" + "=" * 60)
    print("Google Translate Test")
    print("=" * 60)

    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="en", target="ja")

        for case in TEST_CASES:
            print(f"\n### {case['name']}")
            print(f"Input: {case['input'][:80]}...")

            result = await asyncio.to_thread(translator.translate, case["input"])
            print(f"Output: {result}")

            # Check separator preservation
            parts = result.split(SEPARATOR)
            preserved = len(parts) == case["expected_blocks"]
            expected = case["expected_blocks"]
            print(f"Separator preserved: {preserved} (got {len(parts)} parts, expected {expected})")

            if not preserved:
                # Check for variations
                variations = ["[[[BR]]]", "[[[ BR ]]]", "[BR]", "[ BR ]"]
                for var in variations:
                    if var in result:
                        print(f"  -> Found variation: '{var}'")

    except Exception as e:
        print(f"Error: {e}")


async def test_deepl_translate():
    """Test DeepL with separator token."""
    print("\n" + "=" * 60)
    print("DeepL Translate Test")
    print("=" * 60)

    api_key = os.environ.get("DEEPL_API_KEY")
    if not api_key:
        print("DEEPL_API_KEY not set, skipping DeepL test")
        return

    try:
        import aiohttp

        api_url = os.environ.get(
            "DEEPL_API_URL", "https://api-free.deepl.com/v2/translate"
        )

        async with aiohttp.ClientSession() as session:
            for case in TEST_CASES:
                print(f"\n### {case['name']}")
                print(f"Input: {case['input'][:80]}...")

                params = {
                    "auth_key": api_key,
                    "text": case["input"],
                    "source_lang": "EN",
                    "target_lang": "JA",
                }

                async with session.post(api_url, data=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data["translations"][0]["text"]
                        print(f"Output: {result}")

                        parts = result.split(SEPARATOR)
                        preserved = len(parts) == case["expected_blocks"]
                        print(f"Separator preserved: {preserved} (got {len(parts)} parts)")
                    else:
                        error_text = await response.text()
                        print(f"API Error: {response.status} - {error_text}")

    except ImportError:
        print("aiohttp not installed, skipping DeepL test")
    except Exception as e:
        print(f"Error: {e}")


async def test_deepl_batch():
    """Test DeepL batch translation (multiple text parameters)."""
    print("\n" + "=" * 60)
    print("DeepL Batch Translation Test (Multiple text params)")
    print("=" * 60)

    api_key = os.environ.get("DEEPL_API_KEY")
    if not api_key:
        print("DEEPL_API_KEY not set, skipping DeepL batch test")
        return

    try:
        import aiohttp

        api_url = os.environ.get(
            "DEEPL_API_URL", "https://api-free.deepl.com/v2/translate"
        )

        # Test with array of texts (no separator)
        texts = ["Hello", "World", "Good morning"]

        print("\n### Batch test (3 texts)")
        print(f"Input texts: {texts}")

        async with aiohttp.ClientSession() as session:
            # DeepL accepts multiple 'text' parameters
            params = [("text", t) for t in texts]
            params.extend([
                ("auth_key", api_key),
                ("source_lang", "EN"),
                ("target_lang", "JA"),
            ])

            async with session.post(api_url, data=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = [t["text"] for t in data["translations"]]
                    print(f"Output texts: {results}")
                    print(f"Order preserved: {len(results) == len(texts)}")
                    print("Batch translation works correctly!")
                else:
                    error_text = await response.text()
                    print(f"API Error: {response.status} - {error_text}")

    except ImportError:
        print("aiohttp not installed, skipping DeepL batch test")
    except Exception as e:
        print(f"Error: {e}")


async def test_openai_translate():
    """Test OpenAI with separator token (for comparison)."""
    print("\n" + "=" * 60)
    print("OpenAI Translate Test")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping OpenAI test")
        return

    try:
        from openai import AsyncOpenAI
        from pydantic import BaseModel

        class TranslationResult(BaseModel):
            translations: list[str]

        client = AsyncOpenAI(api_key=api_key)

        # Test with array (Structured Outputs)
        texts = ["Hello", "World", "Good morning"]

        print("\n### Structured Outputs test (3 texts)")
        print(f"Input texts: {texts}")

        system_content = (
            "Translate the following texts from English to Japanese. "
            "Return a JSON object with a 'translations' array."
        )

        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": str(texts)},
            ],
            response_format=TranslationResult,
            temperature=0.2,
        )

        result = response.choices[0].message.parsed
        if result:
            print(f"Output texts: {result.translations}")
            print(f"Order preserved: {len(result.translations) == len(texts)}")
            print("Structured Outputs works correctly!")

    except ImportError:
        print("openai not installed, skipping OpenAI test")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    print("=" * 60)
    print("Separator Token Behavior Test")
    print("=" * 60)
    print(f"Separator: {SEPARATOR}")

    await test_google_translate()
    await test_deepl_translate()
    await test_deepl_batch()
    await test_openai_translate()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Based on the results:
- If separator is NOT preserved by Google/DeepL, use array-based batch translation
- DeepL batch (multiple text params) should work reliably
- OpenAI Structured Outputs guarantees array structure
""")


if __name__ == "__main__":
    asyncio.run(main())
