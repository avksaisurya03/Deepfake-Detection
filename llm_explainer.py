import ollama

def generate_explanation(label, overall_score, total_frames,
                         fake_count, duration,
                         flow_score=None, flow_verdict=None):

    real_count = total_frames - fake_count

    prompt = f"""
You are a forensic analyst.

Explain in 3 short paragraphs:

Final Verdict: {label}
Frames: {total_frames}
Fake frames: {fake_count}
Real frames: {real_count}
Motion Score: {flow_score}
Motion Verdict: {flow_verdict}

Focus strongly on facial motion irregularities and how they impact the decision.
Keep it clear and professional.
"""

    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content'].strip()