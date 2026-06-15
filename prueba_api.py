from machine_learning_engineering import client, MODEL_NAME

print(f"Sending a 'Ping' to the model: {MODEL_NAME} on SAIA...")

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "Respond only with the phrase: 'Connection successful'.",
            }
        ],
        max_tokens=10,
        temperature=0.1,
    )

    # Print the server's response
    print("\n✅ AGENT RESPONSE:")
    print(response.choices[0].message.content)

    # --- AUDIT DETAILS ---
    print("\n📊 TOKEN RECEIPT:")
    print(f"Input tokens (Your question): {response.usage.prompt_tokens}")
    print(f"Output tokens (Its response): {response.usage.completion_tokens}")
    print(f"TOTAL SPENT: {response.usage.total_tokens} tokens")
    print("-" * 30)

except Exception as e:
    print("\n❌ CONNECTION ERROR:")
    print(e)
