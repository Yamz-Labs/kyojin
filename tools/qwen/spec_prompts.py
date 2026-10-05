CODE_SNIP = '''def moving_average(values, window):
    if window <= 0:
        raise ValueError("window must be positive")
    result = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        if i >= window - 1:
            result.append(total / window)
    return result

def median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2
'''
QUOTE = ("The committee noted that the proposed regulation would require all operators of automated "
 "vehicles to maintain a continuous audit log of sensor inputs, control decisions and manual overrides, "
 "retained for no fewer than twenty-four months and made available to the supervisory authority within "
 "five working days of a written request. Operators who fail to comply may be subject to a fine of up to "
 "two percent of annual turnover.")
EXTRA = {
 "multi": [
  "Explique-moi en detail comment fonctionne une pompe a chaleur et pourquoi elle est plus efficace qu'un radiateur electrique.",
  "Escribe un cuento corto sobre un pescador que encuentra una lampara magica en la playa.",
  "Erklaere mir ausfuehrlich den Unterschied zwischen einer GmbH und einer AG in Deutschland.",
  "请详细介绍一下如何准备一次长途徒步旅行，包括装备、路线和安全注意事项。",
  "Spiegami come funziona il sistema immunitario umano, con esempi concreti.",
  "Redige une lettre formelle de reclamation a un fournisseur d'electricite pour une facture erronee.",
 ],
 "copy": [
  "Here is a Python module:\n\n```python\n" + CODE_SNIP + "```\n\nRewrite the whole module with type hints and docstrings, keeping the logic identical.",
  "Here is a Python module:\n\n```python\n" + CODE_SNIP + "```\n\nRename `values` to `data` everywhere and print the full updated module.",
  "Quote the following passage word for word, then summarize it in two sentences.\n\n" + QUOTE,
  "Here is a passage:\n\n" + QUOTE + "\n\nExtract every obligation as a bullet list, reusing the exact wording of the passage.",
 ],
}
