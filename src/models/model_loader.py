import os
from rich.console import Console

console = Console()

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


class VantaAI:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self, show_progress: bool = True) -> bool:
        if self.is_loaded:
            return True

        try:
            if show_progress:
                console.print("[cyan]Loading AI model...[/cyan]")
                console.print("[dim]First run downloads ~3GB, next runs use cache[/dim]")

            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            self.is_loaded = True

            if show_progress:
                console.print("[green]Model loaded successfully[/green]")

            return True

        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            return False

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        if not self.is_loaded:
            if not self.load_model():
                return "Model not available"

        try:
            messages = [
                {"role": "system", "content": "You are a cybersecurity expert assistant."},
                {"role": "user", "content": prompt}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            return f"Generation error: {str(e)}"

    def classify_threat(self, finding: dict) -> dict:
        prompt = f"""Analyze this security finding and classify it.

Type: {finding.get('type', 'unknown')}
Details: {finding.get('details', {})}

Respond in this exact format:
SEVERITY: [CRITICAL/HIGH/MEDIUM/LOW]
SUMMARY: [one line summary]
RECOMMENDATION: [one line fix]"""

        response = self.generate(prompt, max_tokens=100)

        severity = "MEDIUM"
        summary = finding.get('type', 'Unknown finding')
        recommendation = "Review manually"

        for line in response.split('\n'):
            if line.startswith('SEVERITY:'):
                severity = line.replace('SEVERITY:', '').strip()
            elif line.startswith('SUMMARY:'):
                summary = line.replace('SUMMARY:', '').strip()
            elif line.startswith('RECOMMENDATION:'):
                recommendation = line.replace('RECOMMENDATION:', '').strip()

        return {
            "severity": severity,
            "summary": summary,
            "recommendation": recommendation
        }


if __name__ == "__main__":
    ai = VantaAI()
    if ai.load_model():
        result = ai.classify_threat({
            "type": "sql_injection",
            "details": {"url": "http://example.com/login", "param": "username"}
        })
        console.print(result)